# SPDX-License-Identifier: Apache-2.0
"""Client side of a remote-DRAM HiCache L3: services that each hold one pinned DRAM pool.

Each service (``nixl_dram_l3.py``, one per decode node or NUMA node) owns a key -> slot index and answers a few
JSON requests over ZMQ; the bytes never pass through it. This client routes every key to one service by CRC32,
reserves or looks up slots there, and moves the data itself with one-sided NIXL WRITE/READ (UCX: RDMA on InfiniBand)
between the caller's registered host memory and the service's pool, then commits or releases.

The store interface mirrors the subset of Mooncake's ``MooncakeDistributedStore`` that ``MooncakeStore`` uses
(``register_buffer``, ``batch_is_exist``, ``batch_put_from``, ``batch_get_into`` and their multi-buffer forms,
``remove_all``), so the HiCache object layout and batching of ``MooncakeStore`` carry over unchanged.

Imports only stdlib, msgspec, pyzmq, numpy and nixl (the last three lazily), so this file also runs on its own.
"""

import ctypes
import glob
import itertools
import json
import logging
import math
import os
import socket
import threading
import time
import uuid
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import msgspec

logger = logging.getLogger(__name__)

PROTO = 1
# Service codes for a slot request (alloc) or lookup.
EXISTS, BUSY, NO_SPACE, TOO_BIG = -1, -2, -3, -4
# Per-object results returned to the store: 0 (put) or the byte count (get) on success, else one of these.
ERR_MISSING = -2  # get: not stored
ERR_SIZE = -3  # get: stored with another size
ERR_XFER = -4  # the transfer failed or timed out
ERR_RPC = -5  # the service did not answer
ERR_SPACE = -6  # put: no slot
ERR_LEASE = -7  # put: written, but the reservation had expired
ERR_BUSY = -8  # put: another writer had the key reserved and has not stored it

EXTRA_PREFIX = "nixl_dram_"


class NixlDramError(RuntimeError):
    pass


class NixlDramConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Settings, each from ``--hicache-storage-backend-extra-config`` key ``nixl_dram_<name>``."""

    registry_dir: str = ""  # shared directory the services publish themselves in
    endpoints: str = ""  # or explicit host:port[,host:port...]
    expected_services: int = (
        0  # required with registry_dir: every rank must route over the same set
    )
    ucx_devices: str = (
        "mlx5_0,mlx5_1,mlx5_2,mlx5_5"  # IB devices UCX may use; "" lets UCX choose
    )
    transport: str = "nixl"  # nixl (UCX) | shm (same-host test transport)
    num_workers: int = 2  # UCX workers of this process's NIXL agent
    engine_config: str = ""  # extra UCX settings for NIXL's engine_config, K=V,K=V
    # Register each host buffer as NIXL descriptors of at most this many MiB, cut on whole pages, instead of one
    # region: UCX then makes one ibv_reg_mr per piece and NIC, not one thread per 2 GiB of the region. 0: one region.
    reg_chunk_mib: int = 0
    rpc_timeout_s: float = 30.0
    xfer_timeout_s: float = 60.0
    lease_s: float = 300.0  # how long a service keeps a reservation or read pin without a commit or release
    discovery_timeout_s: float = 600.0

    @classmethod
    def load(cls, extra_config: Optional[dict] = None) -> "NixlDramConfig":
        values = {
            k[len(EXTRA_PREFIX) :]: v
            for k, v in (extra_config or {}).items()
            if k.startswith(EXTRA_PREFIX)
        }
        unknown = set(values) - set(cls.__struct_fields__)
        if unknown:
            raise ValueError(
                f"unknown nixl_dram settings: {sorted(EXTRA_PREFIX + k for k in unknown)}"
            )
        if isinstance(values.get("endpoints"), (list, tuple)):
            values["endpoints"] = ",".join(values["endpoints"])
        cfg = msgspec.convert(values, cls, strict=False)
        cfg.validate()
        return cfg

    def validate(self):
        if self.transport not in ("nixl", "shm"):
            raise ValueError(
                f"nixl_dram_transport must be nixl or shm, not {self.transport!r}"
            )
        if not self.endpoints and not self.registry_dir:
            raise ValueError(
                "nixl_dram needs nixl_dram_registry_dir or nixl_dram_endpoints"
            )
        if self.registry_dir and not self.endpoints and self.expected_services <= 0:
            raise ValueError(
                "nixl_dram_expected_services must be set with a registry directory, so that every rank routes "
                "keys over the same services"
            )
        if self.reg_chunk_mib < 0:
            raise ValueError(
                f"nixl_dram_reg_chunk_mib must be 0 (one region) or positive, not {self.reg_chunk_mib}"
            )
        if min(self.rpc_timeout_s, self.xfer_timeout_s, self.lease_s) <= 0:
            raise ValueError("nixl_dram timeouts and lease must be positive")
        if self.lease_s <= self.xfer_timeout_s:
            # A slot whose lease ran out can be handed to another key; a write still in flight would then land in it.
            raise ValueError(
                f"nixl_dram_lease_s ({self.lease_s}) must exceed nixl_dram_xfer_timeout_s ({self.xfer_timeout_s})"
            )

    def devices(self) -> List[str]:
        return [d.strip() for d in self.ucx_devices.split(",") if d.strip()]


def registration_ranges(
    ptr: int, size: int, unit: int, limit: int
) -> List[Tuple[int, int]]:
    """[(address, bytes)] covering [ptr, ptr + size) in pieces of whole units, each at most `limit` bytes (one unit
    if a unit is larger), so a transfer part inside one unit never spans two registrations. One piece if `limit` or
    `unit` is 0 or the buffer fits in `limit`."""
    if limit <= 0 or unit <= 0 or size <= limit:
        return [(ptr, size)]
    # Cuts also on 64 KiB (aarch64's largest base page), so no two pieces share an OS page UCX's rcache could merge.
    unit = math.lcm(unit, 1 << 16)
    step = max(limit // unit, 1) * unit
    return [(ptr + off, min(step, size - off)) for off in range(0, size, step)]


def discover_services(
    cfg: NixlDramConfig, clock=time.monotonic, sleep=time.sleep
) -> List[dict]:
    """Every service record, sorted, so that all ranks route keys identically."""
    if cfg.endpoints:
        return [
            {"endpoint": e.strip(), "service": e.strip()}
            for e in cfg.endpoints.split(",")
            if e.strip()
        ]
    deadline = clock() + cfg.discovery_timeout_s
    while True:
        records = []
        for path in sorted(glob.glob(os.path.join(cfg.registry_dir, "*.json"))):
            try:
                with open(path) as f:
                    records.append(json.load(f))
            except (OSError, ValueError):
                continue  # being replaced; read it on the next pass
        if len(records) > cfg.expected_services:
            raise NixlDramError(
                f"{len(records)} services in {cfg.registry_dir}, more than nixl_dram_expected_services "
                f"{cfg.expected_services}: {[r.get('service') for r in records]}"
            )
        if len(records) == cfg.expected_services:
            return sorted(
                records,
                key=lambda r: (
                    str(r.get("host")),
                    int(r.get("numa", 0)),
                    int(r.get("port", 0)),
                ),
            )
        if clock() >= deadline:
            raise NixlDramError(
                f"only {len(records)} of {cfg.expected_services} services in {cfg.registry_dir} after "
                f"{cfg.discovery_timeout_s:.0f} s: {[r.get('service') for r in records]}"
            )
        sleep(1.0)


class _Rpc:
    """One REQ socket per (thread, service); requests to several services are sent first, then all are received."""

    def __init__(self, endpoints: Sequence[str], timeout_s: float):
        import zmq

        self.zmq = zmq
        self.endpoints = list(endpoints)
        self.timeout_ms = int(timeout_s * 1000)
        self.ctx = zmq.Context()
        self.local = threading.local()
        self.lock = threading.Lock()
        self.all_socks = []

    def _sock(self, i):
        socks = getattr(self.local, "socks", None)
        if socks is None:
            socks = self.local.socks = {}
        s = socks.get(i)
        if s is None:
            s = self.ctx.socket(self.zmq.REQ)
            s.setsockopt(self.zmq.LINGER, 0)
            s.setsockopt(self.zmq.RCVTIMEO, self.timeout_ms)
            s.setsockopt(self.zmq.SNDTIMEO, self.timeout_ms)
            s.connect(f"tcp://{self.endpoints[i]}")
            socks[i] = s
            with self.lock:
                self.all_socks.append(s)
        return s

    def _reset(self, i):
        s = self.local.socks.pop(i, None)
        if s is not None:
            s.close(0)

    def call_many(
        self, requests: Dict[int, Tuple[dict, Optional[bytes]]]
    ) -> Tuple[Dict[int, Tuple[dict, Optional[bytes]]], Dict[int, str]]:
        """({service: (reply, payload)}, {service: error}) for {service: (request, payload)}."""
        replies, errors, sent = {}, {}, []
        for i, (req, payload) in requests.items():
            try:
                frames = [json.dumps(req).encode()] + (
                    [payload] if payload is not None else []
                )
                self._sock(i).send_multipart(frames)
                sent.append(i)
            except self.zmq.ZMQError as e:
                errors[i] = f"send to {self.endpoints[i]}: {e}"
                self._reset(i)
        for i in sent:
            try:
                frames = self._sock(i).recv_multipart()
            except (
                self.zmq.ZMQError
            ) as e:  # timeout: a REQ socket cannot send again until it gets its reply
                errors[i] = (
                    f"no reply from {self.endpoints[i]} in {self.timeout_ms} ms: {e}"
                )
                self._reset(i)
                continue
            reply = json.loads(frames[0])
            if not reply.get("ok"):
                errors[i] = f"{self.endpoints[i]}: {reply.get('error')}"
                continue
            replies[i] = (reply, frames[1] if len(frames) > 1 else None)
        return replies, errors

    def close(self):
        with self.lock:
            for s in self.all_socks:
                s.close(0)
            self.all_socks.clear()
        self.ctx.term()


class NixlTransport:
    """One-sided READ/WRITE between this process's registered host memory and the services' pools via NIXL's UCX
    backend."""

    def __init__(self, name: str, cfg: NixlDramConfig):
        from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t

        # Prefetch and backup threads post transfers concurrently.
        conf = nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=False,
            backends=[],
            sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_RW,
        )
        self.agent = nixl_agent(name, conf)
        params = {"num_workers": str(cfg.num_workers)}
        if cfg.devices():
            # NIXL 1.5 splits device_list on ", " and appends ":1" to each (UCX NET_DEVICES).
            params["device_list"] = ", ".join(cfg.devices())
        if cfg.engine_config:
            params["engine_config"] = cfg.engine_config
        self.agent.create_backend("UCX", params)
        self.params = params
        self.regs = []
        self.remotes = []

    def register_local(self, ranges: Sequence[Tuple[int, int]]):
        self.regs.append(
            self.agent.register_memory([(p, n, 0, "") for p, n in ranges], "DRAM")
        )

    def add_remote(self, hello: dict, payload: Optional[bytes]):
        if not payload:
            raise NixlDramError(
                f"service {hello.get('service')} sent no NIXL metadata (transport {hello.get('transport')})"
            )
        name = self.agent.add_remote_agent(payload)
        self.remotes.append(name)
        return name

    def post(self, op: str, remote, local, remote_descs):
        import numpy as np

        agent = self.agent
        ld = agent.get_xfer_descs(np.asarray(local, dtype=np.uint64), "DRAM")
        rd = agent.get_xfer_descs(np.asarray(remote_descs, dtype=np.uint64), "DRAM")
        handle = agent.initialize_xfer(op, ld, rd, remote)
        state = agent.transfer(handle)
        if state == "ERR":
            agent.release_xfer_handle(handle)
            raise NixlDramError(
                f"NIXL {op} of {len(local)} descriptor(s) to {remote} failed to post"
            )
        return [handle, state]

    def wait(self, posted, deadline: float) -> str:
        """DONE, ERR (finished with an error), or TIMEOUT (may still be moving data)."""
        handle, state = posted
        agent = self.agent
        try:
            while state == "PROC":
                if time.monotonic() >= deadline:
                    return "TIMEOUT"
                time.sleep(0.00005)
                state = agent.check_xfer_state(handle)
            return "DONE" if state == "DONE" else "ERR"
        finally:
            try:
                agent.release_xfer_handle(handle)
            except Exception as e:
                # NIXL could not cancel it, so the handle stays leaked rather than reused.
                logger.error(
                    "HiCache NIXL DRAM L3: release of a transfer handle failed: %s", e
                )

    def close(self):
        for name in self.remotes:
            try:
                self.agent.remove_remote_agent(name)
            except Exception:
                pass
        for reg in self.regs:
            self.agent.deregister_memory(reg)
        self.regs, self.remotes = [], []


class ShmTransport:
    """Same-host transport for tests and smoke checks: maps a service's /dev/shm pool and copies with memmove. It
    enforces what NIXL does: local ranges must be registered, remote ranges must lie in the pool."""

    def __init__(self, name: str, cfg: NixlDramConfig):
        self.local = []
        self.maps = []
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.mmap.restype = ctypes.c_void_p
        self.libc.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
        ]
        self.libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

    def register_local(self, ranges: Sequence[Tuple[int, int]]):
        self.local.extend((p, p + n) for p, n in ranges)

    def add_remote(self, hello: dict, payload: Optional[bytes]):
        fd = os.open(hello["shm_path"], os.O_RDWR)
        try:
            addr = self.libc.mmap(
                None, hello["pool_bytes"], 3, 0x01, fd, 0
            )  # PROT_READ|WRITE, MAP_SHARED
        finally:
            os.close(fd)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise NixlDramError(
                f"mmap of {hello['shm_path']} failed: errno {ctypes.get_errno()}"
            )
        remote = (hello["pool_base"], hello["pool_bytes"], addr)
        self.maps.append(remote)
        return remote

    def post(self, op: str, remote, local, remote_descs):
        base, size, mapped = remote
        for (la, n, _), (ra, rn, _) in zip(local, remote_descs):
            if n != rn or not any(s <= la and la + n <= e for s, e in self.local):
                raise NixlDramError(f"local range 0x{la:x}+{n} is not registered")
            if not (base <= ra and ra + n <= base + size):
                raise NixlDramError(f"remote range 0x{ra:x}+{n} is outside the pool")
            dst, src = (
                (mapped + ra - base, la) if op == "WRITE" else (la, mapped + ra - base)
            )
            ctypes.memmove(dst, src, n)
        return None

    def wait(self, posted, deadline: float) -> str:
        return "DONE"

    def close(self):
        for _, size, mapped in self.maps:
            self.libc.munmap(mapped, size)
        self.maps, self.local = [], []


class NixlDramClient:
    """A store over every service: keys are routed by CRC32, so all ranks must see the same sorted service list."""

    def __init__(self, cfg: NixlDramConfig, name: Optional[str] = None, transport=None):
        self.cfg = cfg
        self.name = (
            name
            or f"sglang-hicache-l3-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        self.services = discover_services(cfg)
        if not self.services:
            raise NixlDramError("no NIXL DRAM L3 service configured")
        self.rpc = _Rpc([s["endpoint"] for s in self.services], cfg.rpc_timeout_s)
        if transport is None:
            transport = (NixlTransport if cfg.transport == "nixl" else ShmTransport)(
                self.name, cfg
            )
        self.transport = transport
        self.remotes = []
        self.page_units: Dict[int, int] = {}  # buffer address -> bytes no transfer part crosses (set_page_units)
        replies, errors = self.rpc.call_many(
            {
                i: ({"op": "hello", "proto": PROTO, "client": self.name}, None)
                for i in range(len(self.services))
            }
        )
        if errors:
            self.rpc.close()
            raise NixlDramError(f"NIXL DRAM L3 services unreachable: {errors}")
        self.service_names = []
        for i in range(len(self.services)):
            hello, payload = replies[i]
            if hello.get("transport") != cfg.transport:
                raise NixlDramError(
                    f"service {hello.get('service')} serves transport {hello.get('transport')}, this client "
                    f"{cfg.transport}"
                )
            self.remotes.append(self.transport.add_remote(hello, payload))
            self.service_names.append(hello.get("service"))
        self.pool_bytes = sum(
            replies[i][0]["pool_bytes"] for i in range(len(self.services))
        )
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self.counters = dict(
            puts=0,
            put_bytes=0,
            put_skipped=0,
            put_failed=0,
            gets=0,
            get_bytes=0,
            get_missing=0,
            get_failed=0,
            exists_queries=0,
            rpc_errors=0,
            xfer_errors=0,
        )
        self._first = {"put": False, "get": False}
        logger.info(
            "HiCache NIXL DRAM L3: %d service(s) %s, %.1f GiB in all, transport %s, UCX devices %s",
            len(self.services),
            self.service_names,
            self.pool_bytes / (1 << 30),
            cfg.transport,
            cfg.devices() or "any",
        )

    # -- helpers
    def _token(self) -> str:
        return f"{self.name}:{next(self._seq)}"

    def _route(self, keys: Sequence[str]) -> Dict[int, List[int]]:
        n, groups = len(self.services), {}
        for i, k in enumerate(keys):
            groups.setdefault(zlib.crc32(k.encode()) % n, []).append(i)
        return groups

    def _count(self, **kw):
        with self._lock:
            for k, v in kw.items():
                self.counters[k] += v

    def _rpc_failed(self, what: str, errors: Dict[int, str]):
        self._count(rpc_errors=len(errors))
        for i, e in errors.items():
            logger.error(
                "HiCache NIXL DRAM L3: %s on %s failed: %s",
                what,
                self.service_names[i],
                e,
            )

    @staticmethod
    def _parts(ptrs, sizes) -> Tuple[List[List[int]], List[List[int]]]:
        return (
            [list(p) if isinstance(p, (list, tuple)) else [p] for p in ptrs],
            [list(s) if isinstance(s, (list, tuple)) else [s] for s in sizes],
        )

    @staticmethod
    def _descs(items, ptr_parts, size_parts):
        """Local and remote descriptor rows for [(object index, remote address)]: an object's parts are contiguous in
        its slot."""
        local, remote = [], []
        for i, addr in items:
            off = 0
            for p, s in zip(ptr_parts[i], size_parts[i]):
                local.append((int(p), int(s), 0))
                remote.append((addr + off, int(s), 0))
                off += int(s)
        return local, remote

    def _transfer(
        self, op: str, plans: Dict[int, list], ptr_parts, size_parts
    ) -> Dict[int, str]:
        """Post one transfer per service, then wait for all; {service: DONE | ERR | TIMEOUT}."""
        posted, state = {}, {}
        for s, items in plans.items():
            if not items:
                continue
            local, remote = self._descs(items, ptr_parts, size_parts)
            try:
                posted[s] = self.transport.post(op, self.remotes[s], local, remote)
            except Exception as e:
                logger.error(
                    "HiCache NIXL DRAM L3: %s of %d object(s) on %s failed to post: %s",
                    op,
                    len(items),
                    self.service_names[s],
                    e,
                )
                state[s] = "ERR"
        deadline = time.monotonic() + self.cfg.xfer_timeout_s
        for s, p in posted.items():
            state[s] = self.transport.wait(p, deadline)
            if state[s] != "DONE":
                logger.error(
                    "HiCache NIXL DRAM L3: %s of %d object(s) on %s: %s%s",
                    op,
                    len(plans[s]),
                    self.service_names[s],
                    state[s],
                    (
                        f" after {self.cfg.xfer_timeout_s:.0f} s; its slots stay reserved or pinned until the "
                        "lease ends, as data may still be moving"
                        if state[s] == "TIMEOUT"
                        else ""
                    ),
                )
        failed = sum(1 for v in state.values() if v != "DONE")
        if failed:
            self._count(xfer_errors=failed)
        return state

    # -- the store interface MooncakeStore uses
    def set_page_units(self, units: Dict[int, int]):
        """{buffer address: a whole number of pages in bytes} for buffers registered later (reg_chunk_mib)."""
        self.page_units.update(units)

    def register_buffer(self, ptr: int, size: int) -> int:
        ptr, size = int(ptr), int(size)
        ranges = registration_ranges(
            ptr, size, self.page_units.get(ptr, 0), self.cfg.reg_chunk_mib << 20
        )
        t0 = time.monotonic()
        self.transport.register_local(ranges)
        logger.info(
            "HiCache NIXL DRAM L3: registered host buffer 0x%x, %.2f GiB%s",
            ptr,
            size / (1 << 30),
            (
                f", {len(ranges)} descriptors of <= {max(n for _, n in ranges) / (1 << 20):.0f} MiB "
                f"in {time.monotonic() - t0:.1f} s"
                if len(ranges) > 1
                else ""
            ),
        )
        return 0

    def batch_is_exist(self, keys: List[str]) -> List[int]:
        out = [0] * len(keys)
        groups = self._route(keys)
        replies, errors = self.rpc.call_many(
            {
                s: ({"op": "exists", "keys": [keys[i] for i in idx]}, None)
                for s, idx in groups.items()
            }
        )
        if errors:
            self._rpc_failed("exists", errors)
        for s, (reply, _) in replies.items():
            for i, present in zip(groups[s], reply["present"]):
                out[i] = 1 if present else 0
        self._count(exists_queries=len(keys))
        return out

    def _request_all(self, op: str, groups, keys, tokens, per_key=None, **fields):
        """One ``op`` request per service for its keys (and per_key lists, split the same way); failed services are
        logged and left out of the replies."""
        per_key = per_key or {}
        replies, errors = self.rpc.call_many(
            {
                s: (
                    {
                        "op": op,
                        "keys": [keys[i] for i in idx],
                        "token": tokens[s],
                        **{k: [v[i] for i in idx] for k, v in per_key.items()},
                        **fields,
                    },
                    None,
                )
                for s, idx in groups.items()
            }
        )
        if errors:
            self._rpc_failed(op, errors)
        return replies

    def batch_put_from(
        self, keys: List[str], ptrs: List[Any], sizes: List[Any]
    ) -> List[int]:
        ptr_parts, size_parts = self._parts(ptrs, sizes)
        totals = [sum(s) for s in size_parts]
        results = [ERR_RPC] * len(keys)
        groups = self._route(keys)
        tokens = {s: self._token() for s in groups}
        replies = self._request_all(
            "alloc",
            groups,
            keys,
            tokens,
            per_key={"sizes": totals},
            lease_s=self.cfg.lease_s,
        )
        plans, skipped, busy = self._plan_writes(groups, replies, results)
        state = self._transfer("WRITE", plans, ptr_parts, size_parts)
        put, put_bytes = self._commit(plans, state, keys, tokens, totals, results)
        if (
            busy
        ):  # written by another writer of the same key: stored only once it committed
            present = self.batch_is_exist([keys[i] for i in busy])
            for i, p in zip(busy, present):
                results[i] = 0 if p else ERR_BUSY
            skipped += sum(present)
        self._count(
            puts=put,
            put_bytes=put_bytes,
            put_skipped=skipped,
            put_failed=len(keys) - put - skipped,
        )
        if put and not self._first["put"]:
            self._first["put"] = True
            logger.info(
                "HiCache NIXL DRAM L3: first write ok, %d object(s), %d bytes",
                put,
                put_bytes,
            )
        return results

    @staticmethod
    def _plan_writes(groups, replies, results):
        """({service: [(object, slot address)]} to write, objects already stored, objects another writer of the same
        key, hence the same bytes, is writing); a stored object succeeds without a write."""
        plans, skipped, busy, no_space = {}, 0, [], 0
        for s, (reply, _) in replies.items():
            plans[s] = []
            for i, addr in zip(groups[s], reply["addrs"]):
                if addr >= 0:
                    plans[s].append((i, addr))
                elif addr == EXISTS:
                    results[i] = 0
                    skipped += 1
                elif addr == BUSY:
                    busy.append(i)
                else:
                    results[i] = ERR_SPACE
                    no_space += 1
        if no_space:
            logger.error(
                "HiCache NIXL DRAM L3: no slot for %d object(s) (the pool is full of objects being written or "
                "read, or an object is larger than a slab)",
                no_space,
            )
        return plans, skipped, busy

    def _commit(self, plans, state, keys, tokens, totals, results):
        """Publish what was written, free what failed; a timed-out write keeps its slots until the lease ends.
        Returns (objects, bytes) stored."""
        commits = {
            s: (
                {
                    "op": "commit",
                    "token": tokens[s],
                    "keys": [keys[i] for i, _ in items],
                    "ok": [state.get(s) == "DONE"] * len(items),
                },
                None,
            )
            for s, items in plans.items()
            if items and state.get(s) != "TIMEOUT"
        }
        replies, errors = self.rpc.call_many(commits) if commits else ({}, {})
        if errors:  # the reservations expire with their lease
            self._rpc_failed("commit", errors)
        put = put_bytes = lost = 0
        for s, items in plans.items():
            done = replies[s][0]["done"] if s in replies else [False] * len(items)
            for (i, _), d in zip(items, done):
                if d:
                    results[i] = 0
                    put += 1
                    put_bytes += totals[i]
                else:
                    results[i] = ERR_LEASE if state.get(s) == "DONE" else ERR_XFER
                    lost += state.get(s) == "DONE"
        if lost:
            logger.error(
                "HiCache NIXL DRAM L3: %d written object(s) were not committed (lease expired?)",
                lost,
            )
        return put, put_bytes

    def batch_get_into(
        self, keys: List[str], ptrs: List[Any], sizes: List[Any]
    ) -> List[int]:
        ptr_parts, size_parts = self._parts(ptrs, sizes)
        totals = [sum(s) for s in size_parts]
        results = [ERR_RPC] * len(keys)
        groups = self._route(keys)
        tokens = {s: self._token() for s in groups}
        replies = self._request_all(
            "lookup", groups, keys, tokens, lease_s=self.cfg.lease_s
        )
        plans, missing = self._plan_reads(groups, replies, totals, results)
        state = self._transfer("READ", plans, ptr_parts, size_parts)
        pinned = {
            s
            for s, (reply, _) in replies.items()
            if max(reply["addrs"]) >= 0 and state.get(s) != "TIMEOUT"
        }
        _, errors = self.rpc.call_many(
            {s: ({"op": "release", "token": tokens[s]}, None) for s in pinned}
        )
        if errors:  # the pins expire with their lease
            self._rpc_failed("release", errors)
        got = got_bytes = 0
        for s, items in plans.items():
            for i, _ in items:
                if state.get(s) == "DONE":
                    results[i] = totals[i]
                    got += 1
                    got_bytes += totals[i]
                else:
                    results[i] = ERR_XFER
        self._count(
            gets=got,
            get_bytes=got_bytes,
            get_missing=missing,
            get_failed=len(keys) - got - missing,
        )
        if got and not self._first["get"]:
            self._first["get"] = True
            logger.info(
                "HiCache NIXL DRAM L3: first read ok, %d object(s), %d bytes",
                got,
                got_bytes,
            )
        return results

    @staticmethod
    def _plan_reads(groups, replies, totals, results):
        """{service: [(object, slot address)]} to read; a stored object of another size is never read."""
        plans, missing, mismatched = {}, 0, 0
        for s, (reply, _) in replies.items():
            plans[s] = []
            for i, addr, size in zip(groups[s], reply["addrs"], reply["sizes"]):
                if addr < 0:
                    results[i] = ERR_MISSING
                    missing += 1
                elif size != totals[i]:
                    results[i] = ERR_SIZE
                    mismatched += 1
                else:
                    plans[s].append((i, addr))
        if mismatched:
            logger.error(
                "HiCache NIXL DRAM L3: %d stored object(s) differ in size from the page they would fill; "
                "not read",
                mismatched,
            )
        return plans, missing

    def batch_put_from_multi_buffers(self, keys, ptrs, sizes) -> List[int]:
        return self.batch_put_from(keys, ptrs, sizes)

    def batch_get_into_multi_buffers(self, keys, ptrs, sizes) -> List[int]:
        return self.batch_get_into(keys, ptrs, sizes)

    def remove_all(self) -> int:
        replies, errors = self.rpc.call_many(
            {s: ({"op": "clear"}, None) for s in range(len(self.services))}
        )
        if errors:
            self._rpc_failed("clear", errors)
        return sum(r["removed"] for r, _ in replies.values())

    def service_stats(self) -> Dict[str, dict]:
        replies, errors = self.rpc.call_many(
            {s: ({"op": "stats"}, None) for s in range(len(self.services))}
        )
        if errors:
            self._rpc_failed("stats", errors)
        return {self.service_names[s]: r for s, (r, _) in replies.items()}

    def close(self):
        try:
            self.transport.close()
        finally:
            self.rpc.close()
