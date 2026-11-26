# SPDX-License-Identifier: Apache-2.0
"""Client side of a remote-DRAM HiCache L3: services that each hold one pinned DRAM pool.

Each service (``nixl_dram_l3.py``, one per node or NUMA node that lends its DRAM) owns a key -> slot index and
answers a few JSON requests over ZMQ; the bytes never pass through it. This client routes every key to one service
by CRC32, reserves or looks up slots there, and moves the data itself with one-sided NIXL WRITE/READ (UCX: RDMA on
InfiniBand or RoCE) between the caller's host memory and the service's pool, then commits or releases.

The store interface mirrors the subset of Mooncake's ``MooncakeDistributedStore`` that ``MooncakeStore`` uses
(``register_buffer``, ``batch_is_exist``, ``batch_put_from``, ``batch_get_into`` and their multi-buffer forms,
``remove_all``), so the HiCache object layout and batching of ``MooncakeStore`` carry over unchanged.
"""

import ctypes
import glob
import itertools
import json
import logging
import math
import mmap
import os
import socket
import threading
import time
import uuid
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import msgspec
import numpy as np
import torch
import zmq

from sglang.srt.environ import envs

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

RECONNECT_MIN_S, RECONNECT_MAX_S = 1.0, 30.0
# Longest a reconnect waits for other threads' transfers on the old agent.
RECONNECT_DRAIN_S = 5.0


class NixlDramError(RuntimeError):
    pass


class NixlDramConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Settings, each from ``--hicache-storage-backend-extra-config`` key ``nixl_dram_<name>``."""

    # Shared directory the services publish themselves in, or explicit host:port[,host:port...].
    registry_dir: str = ""
    endpoints: str = ""
    # Required with registry_dir: every rank must route keys over the same services.
    expected_services: int = 0
    # RDMA devices this process's UCX may use, comma-separated; "" lets UCX choose.
    ucx_devices: str = ""
    num_workers: int = 2  # UCX workers of this process's NIXL agent
    engine_config: str = ""  # extra UCX settings for NIXL's engine_config, K=V,K=V
    # Register each host buffer as NIXL descriptors of at most this many MiB, cut on whole pages, instead of one
    # region: UCX then makes one ibv_reg_mr per piece and NIC, not one thread per 2 GiB of the region. 0: one region.
    reg_chunk_mib: int = 0
    rpc_timeout_s: float = 30.0
    xfer_timeout_s: float = 60.0
    # How long a service keeps a reservation or read pin without a commit or release.
    lease_s: float = 300.0
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
            s = self.ctx.socket(zmq.REQ)
            s.setsockopt(zmq.LINGER, 0)
            s.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            s.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
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
            except zmq.ZMQError as e:
                errors[i] = f"send to {self.endpoints[i]}: {e}"
                self._reset(i)
        for i in sent:
            try:
                frames = self._sock(i).recv_multipart()
            except zmq.ZMQError as e:
                # A timeout: a REQ socket cannot send again until it gets its reply.
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

    def __init__(self, name: str, cfg: NixlDramConfig, reconnect: bool = False):
        from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t

        # Prefetch and backup threads post transfers concurrently.
        conf = nixl_agent_config(
            enable_prog_thread=envs.SGLANG_HICACHE_NIXL_DRAM_PROG_THREAD.get(),
            enable_listen_thread=False,
            backends=[],
            sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_RW,
        )
        self.agent = nixl_agent(name, conf)
        params = {"num_workers": str(cfg.num_workers)}
        if cfg.devices():
            # NIXL splits device_list on ", " and appends ":1" to each (UCX NET_DEVICES).
            params["device_list"] = ", ".join(cfg.devices())
        if cfg.engine_config:
            params["engine_config"] = cfg.engine_config
        self.agent.create_backend("UCX", params)
        self.reconnect = reconnect
        self.regs = []
        self.remotes = []

    def register_local(self, ranges: Sequence[Tuple[int, int]]):
        self.regs.append(
            self.agent.register_memory([(p, n, 0, "") for p, n in ranges], "DRAM")
        )

    def add_remote(self, hello: dict, payload: Optional[bytes]):
        if not payload:
            raise NixlDramError(f"service {hello.get('service')} sent no NIXL metadata")
        name = self.agent.add_remote_agent(payload)
        self.remotes.append(name)
        return name

    def remove_remote(self, name):
        if name in self.remotes:
            self.remotes.remove(name)
        self.agent.remove_remote_agent(name)

    def post(self, op: str, remote, local, remote_descs):
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
        return [handle, state, remote]

    def wait(self, posted, deadline: float) -> str:
        """DONE, ERR (finished with an error), or TIMEOUT (may still be moving data)."""
        handle, state, _ = posted
        agent = self.agent
        try:
            while state == "PROC":
                if time.monotonic() >= deadline:
                    return "TIMEOUT"
                time.sleep(0.00005)
                state = agent.check_xfer_state(handle)
            return "DONE" if state == "DONE" else "ERR"
        except Exception as e:
            if not self.reconnect:
                raise
            # e.g. nixlRemoteDisconnectError: an error of this service's transfer only, so the other services'
            # posted transfers are still waited for and released.
            logger.error(
                "HiCache NIXL DRAM L3: transfer status to %s: %r", posted[2], e
            )
            return "ERR"
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


class _Conn:
    """One service's connection state: transfers in flight on its current agent, and reconnect backoff."""

    def __init__(self, pid):
        self.cv = threading.Condition()
        self.inflight = 0
        self.reconnecting = False
        self.next_try = 0.0  # the first reconnect is immediate
        self.backoff = RECONNECT_MIN_S
        self.pid = pid
        # Its control plane stopped answering: it may come back as a new process, which transfers to the old agent
        # do not report (they hang until the transfer timeout), so the next transfer re-adds the agent first.
        self.suspect = False


class NixlDramClient:
    """A store over every service: keys are routed by CRC32, so all ranks must see the same sorted service list."""

    def __init__(self, cfg: NixlDramConfig, name: Optional[str] = None, transport=None):
        self.cfg = cfg
        self.name = (
            name
            or f"sglang-hicache-l3-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        self.reconnect = envs.SGLANG_HICACHE_NIXL_DRAM_RECONNECT.get()
        self.services = discover_services(cfg)
        if not self.services:
            raise NixlDramError("no NIXL DRAM L3 service configured")
        self.rpc = _Rpc([s["endpoint"] for s in self.services], cfg.rpc_timeout_s)
        if transport is None:
            transport = NixlTransport(self.name, cfg, self.reconnect)
        self.transport = transport
        self.remotes = []
        # Buffer address -> bytes no transfer part crosses (set_page_units).
        self.page_units: Dict[int, int] = {}
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
            self.remotes.append(self.transport.add_remote(hello, payload))
            self.service_names.append(hello.get("service"))
        self.pool_bytes = sum(
            replies[i][0]["pool_bytes"] for i in range(len(self.services))
        )
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self._conn = [
            _Conn(replies[i][0].get("pid")) for i in range(len(self.services))
        ]
        self._reconnect_logged = time.monotonic()
        self.counters = dict(
            xfer_errors=0,
            reconnects=0,
            reconnect_failed=0,
            reconnect_skipped=0,
            retried_reads=0,
            retried_writes=0,
        )
        logger.info(
            "HiCache NIXL DRAM L3: %d service(s) %s, %.1f GiB in all, UCX devices %s, reconnect %s",
            len(self.services),
            self.service_names,
            self.pool_bytes / (1 << 30),
            self.cfg.devices() or "any",
            "on" if self.reconnect else "off",
        )
        self._staging = None
        self._init_staging(envs.SGLANG_HICACHE_NIXL_DRAM_STAGING_MIB.get())

    def _init_staging(self, staging_mib: int):
        """Register only a staging region of staging_mib MiB and bounce every transfer through it, instead of
        registering the whole host pool."""
        if staging_mib <= 0:
            return
        buf = mmap.mmap(-1, staging_mib << 20)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        self.transport.register_local([(addr, staging_mib << 20)])
        self._staging = (buf, addr, staging_mib << 20)
        self._stage_t = torch.frombuffer(buf, dtype=torch.uint8)
        self._staging_lock = threading.Lock()
        self._host_views = []  # (base, size, uint8 tensor over the host buffer), from register_buffer
        self._row_views = {}  # (buffer index, row bytes) -> [rows, row bytes] view of that buffer
        logger.info(
            "HiCache NIXL DRAM L3: staging region %d MiB registered; host pool is not registered",
            staging_mib,
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
        for i, e in errors.items():
            if self.reconnect and e.startswith(("send to ", "no reply from ")):
                self._conn[i].suspect = True
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
        if self._staging is None:
            return self._transfer_direct(op, plans, ptr_parts, size_parts)
        # One staged transfer at a time per client bounds the bytes in flight.
        with self._staging_lock:
            return self._transfer_staged(op, plans, ptr_parts, size_parts)

    def _transfer_direct(
        self, op: str, plans: Dict[int, list], ptr_parts, size_parts
    ) -> Dict[int, str]:
        """Post one transfer per service, then wait for all; {service: DONE | ERR | TIMEOUT}."""
        descs = {}
        for s, items in plans.items():
            if items:
                local, remote = self._descs(items, ptr_parts, size_parts)
                descs[s] = (local, remote, len(items))
        return self._post_wait(op, descs)

    def _transfer_staged(self, op, plans, ptr_parts, size_parts) -> Dict[int, str]:
        """Bounce through the registered staging region in chunks. Host parts are grouped by (host buffer, size) and
        moved with one torch index_select / index_copy_ per group, so the copies run without the GIL; parts that do
        not sit on a row of their buffer fall back to memmove. A service's state is its worst chunk's."""
        _, base, cap = self._staging
        rank = {"DONE": 0, "TIMEOUT": 1, "ERR": 2}
        state: Dict[int, str] = {}
        items = [(s, i, addr) for s, its in plans.items() for i, addr in its]
        totals = [sum(int(x) for x in size_parts[i]) for _, i, _ in items]
        k = 0
        while k < len(items):
            chunk, used = [], 0
            while k < len(items):
                n = totals[k]
                if n > cap:
                    logger.error(
                        "HiCache NIXL DRAM L3: object of %d bytes exceeds the %d byte staging region",
                        n,
                        cap,
                    )
                    state[items[k][0]] = "ERR"
                    k += 1
                    continue
                if used + n > cap:
                    break
                chunk.append(k)
                used += n
                k += 1
            if not chunk:
                continue
            # Flatten parts: service, host pointer, size, remote address.
            svc, ptr, size, raddr = [], [], [], []
            for c in chunk:
                s, i, addr = items[c]
                off = 0
                for p, n in zip(ptr_parts[i], size_parts[i]):
                    svc.append(s)
                    ptr.append(int(p))
                    size.append(int(n))
                    raddr.append(addr + off)
                    off += int(n)
            svc, ptr, size, raddr = (
                np.asarray(x, dtype=np.int64) for x in (svc, ptr, size, raddr)
            )
            dst = self._stage_layout(ptr, size)  # staging offset per part
            if op == "WRITE":
                self._stage_copy(True, ptr, size, dst)
            descs = {}
            for s in np.unique(svc):
                m = svc == s
                zeros = np.zeros(int(m.sum()), np.int64)
                local = np.stack([base + dst[m], size[m], zeros], axis=1)
                remote = np.stack([raddr[m], size[m], zeros], axis=1)
                descs[int(s)] = (
                    local.astype(np.uint64),
                    remote.astype(np.uint64),
                    sum(1 for c in chunk if items[c][0] == s),
                )
            st = self._post_wait(op, descs)
            if op == "READ":
                done = np.isin(svc, [s for s, v in st.items() if v == "DONE"])
                if done.any():
                    self._stage_copy(False, ptr[done], size[done], dst[done])
            for s, v in st.items():
                if rank.get(v, 2) > rank.get(state.get(s, "DONE"), 2):
                    state[s] = v
                else:
                    state.setdefault(s, v)
        return state

    def _stage_layout(self, ptr, size):
        """Staging offset of each part: parts are laid out group by group ((host buffer, size) groups, in order of
        first appearance), so each group is one contiguous [rows, size] block of the staging region."""
        bases = np.asarray([v[0] for v in self._host_views] or [0], dtype=np.int64)
        buf = np.searchsorted(bases, ptr, side="right") - 1
        key = buf * (1 << 40) + size
        _, first, inv = np.unique(key, return_index=True, return_inverse=True)
        order = np.argsort(
            first, kind="stable"
        )  # group ids in order of first appearance
        rank_of = np.empty_like(order)
        rank_of[order] = np.arange(len(order))
        g = rank_of[inv]
        dst = np.empty(len(ptr), dtype=np.int64)
        off = 0
        for gi in range(len(order)):
            m = np.nonzero(g == gi)[0]
            dst[m] = off + np.arange(len(m), dtype=np.int64) * size[m[0]]
            off += int(size[m].sum())
        return dst

    def _stage_copy(self, to_stage: bool, ptr, size, dst):
        """Host parts <-> staging. One torch op per (host buffer, size) group whose parts sit on rows of the buffer;
        the rest (unregistered addresses, unaligned parts) are copied one by one with memmove."""
        _, base, _ = self._staging
        bases = np.asarray([v[0] for v in self._host_views] or [0], dtype=np.int64)
        buf = np.searchsorted(bases, ptr, side="right") - 1
        slow = np.ones(len(ptr), dtype=bool)
        if self._host_views:
            for b in np.unique(buf[buf >= 0]):
                hb, hs, ht = self._host_views[int(b)]
                for n in np.unique(size[buf == b]):
                    m = np.nonzero((buf == b) & (size == n))[0]
                    rel = ptr[m] - hb
                    ok = (rel % n == 0) & (rel + n <= hs)
                    m = m[ok]
                    if len(m) == 0:
                        continue
                    # Staging rows of a group are contiguous by layout.
                    if not np.all(np.diff(dst[m]) == n):
                        continue
                    rows = self._row_views.get((int(b), int(n)))
                    if rows is None:
                        rows = ht[: (hs // int(n)) * int(n)].view(-1, int(n))
                        self._row_views[(int(b), int(n))] = rows
                    idx = torch.from_numpy((ptr[m] - hb) // n)
                    start = int(dst[m[0]])
                    stage = self._stage_t[start : start + len(m) * int(n)].view(
                        len(m), int(n)
                    )
                    if to_stage:
                        torch.index_select(rows, 0, idx, out=stage)
                    else:
                        rows.index_copy_(0, idx, stage)
                    slow[m] = False
        for j in np.nonzero(slow)[0]:
            a, b = (
                (base + int(dst[j]), int(ptr[j]))
                if to_stage
                else (int(ptr[j]), base + int(dst[j]))
            )
            ctypes.memmove(a, b, int(size[j]))

    def _post_wait(self, op: str, descs: Dict[int, tuple]) -> Dict[int, str]:
        """descs: {service: (local, remote, object count)}; post one transfer per service, then wait for all.
        With reconnect on, a service whose transfer ends in ERR gets its NIXL agent re-added (_reconnect) and the
        transfer is retried once (a READ so a broken connection does not cut the caller's hit prefix; a WRITE
        rewrites the same bytes into the same reserved slots). A TIMEOUT re-adds only if the registry shows the
        service's process was replaced."""
        state = self._post_wait_once(op, descs)
        if not self.reconnect:
            return state
        broken = [s for s, v in state.items() if v == "ERR"]
        replaced = [s for s, v in state.items() if v == "TIMEOUT" and self._replaced(s)]
        if broken or replaced:
            again = {s: descs[s] for s in broken if self._reconnect(s)}
            for s in (
                replaced
            ):  # its data may still be moving: no retry; the slots wait for their lease
                self._reconnect(s, "a transfer timeout to a replaced process")
            if again:
                retried = "retried_reads" if op == "READ" else "retried_writes"
                self._count(**{retried: len(again)})
                state.update(self._post_wait_once(op, again))
            self._log_reconnects()
        return state

    def _replaced(self, s: int) -> bool:
        """Whether the registry's record of service s names another process than the one this client talks to."""
        if not self.cfg.registry_dir:
            return False
        try:
            path = os.path.join(self.cfg.registry_dir, f"{self.service_names[s]}.json")
            with open(path) as f:
                pid = json.load(f).get("pid")
        except (OSError, ValueError):
            return False
        return pid is not None and pid != self._conn[s].pid

    def _post_wait_once(self, op: str, descs: Dict[int, tuple]) -> Dict[int, str]:
        posted, state = {}, {}
        for s, (local, remote, count) in descs.items():
            if self.reconnect:
                conn = self._conn[s]
                if conn.suspect and not self._reconnect(s, "its control plane failed"):
                    state[s] = "ERR"
                    continue
                with conn.cv:
                    if (
                        conn.reconnecting
                    ):  # its agent is being re-added: a miss now, never a wait
                        state[s] = "ERR"
                        continue
                    conn.inflight += 1
            try:
                posted[s] = self.transport.post(op, self.remotes[s], local, remote)
            except Exception as e:
                if self.reconnect:
                    self._leave(s)
                logger.error(
                    "HiCache NIXL DRAM L3: %s of %d object(s) on %s failed to post: %s",
                    op,
                    count,
                    self.service_names[s],
                    e,
                )
                state[s] = "ERR"
        deadline = time.monotonic() + self.cfg.xfer_timeout_s
        for s, p in posted.items():
            state[s] = self.transport.wait(p, deadline)
            if self.reconnect:
                self._leave(s, state[s] == "DONE")
            if state[s] != "DONE":
                logger.error(
                    "HiCache NIXL DRAM L3: %s of %d object(s) on %s: %s%s",
                    op,
                    descs[s][2],
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

    def _leave(self, s: int, done: bool = False):
        conn = self._conn[s]
        with conn.cv:
            conn.inflight -= 1
            if done:
                conn.backoff = RECONNECT_MIN_S  # the connection works again
            if conn.reconnecting and conn.inflight == 0:
                conn.cv.notify_all()

    def _reconnect(self, s: int, why: str = "a transfer error") -> bool:
        """Re-add service s's NIXL agent from a fresh hello (the registry's record must still name the same
        endpoint). At most one reconnect per service at a time and per backoff interval; callers never wait on it."""
        conn = self._conn[s]
        with conn.cv:
            now = time.monotonic()
            if conn.reconnecting or now < conn.next_try:
                skip = True
            else:
                skip = False
                conn.reconnecting = True
                conn.next_try = now + conn.backoff
                conn.backoff = min(2 * conn.backoff, RECONNECT_MAX_S)
                # Other threads' transfers on the old agent end first (each ends by its own deadline).
                drained = conn.cv.wait_for(
                    lambda: conn.inflight == 0, timeout=RECONNECT_DRAIN_S
                )
        if skip:
            self._count(reconnect_skipped=1)
            return False
        err, pid = None, None
        try:
            if not drained:
                raise NixlDramError(
                    f"{conn.inflight} transfer(s) still in flight on the old agent"
                )
            pid = self._readd(s)
        except Exception as e:
            err = e
        with conn.cv:
            conn.reconnecting = False
            old_pid, conn.pid = conn.pid, pid if err is None else conn.pid
            if err is None:
                conn.suspect = False
        self._count(**({"reconnects": 1} if err is None else {"reconnect_failed": 1}))
        if err is None:
            logger.warning(
                "HiCache NIXL DRAM L3: reconnect to %s after %s ok (service pid %s -> %s%s)",
                self.service_names[s],
                why,
                old_pid,
                pid,
                (
                    ", same process: its data is kept"
                    if pid == old_pid
                    else ", a new process: its data is gone"
                ),
            )
        else:
            logger.error(
                "HiCache NIXL DRAM L3: reconnect to %s after %s failed: %s; its objects are misses until the next "
                "try in %.0f s",
                self.service_names[s],
                why,
                err,
                conn.next_try - time.monotonic(),
            )
        return err is None

    def _readd(self, s: int):
        if self.cfg.registry_dir:
            path = os.path.join(self.cfg.registry_dir, f"{self.service_names[s]}.json")
            with open(path) as f:
                rec = json.load(f)
            if rec.get("endpoint") != self.rpc.endpoints[s]:
                raise NixlDramError(
                    f"registry {path} names endpoint {rec.get('endpoint')}, this client routes to "
                    f"{self.rpc.endpoints[s]}"
                )
        replies, errors = self.rpc.call_many(
            {s: ({"op": "hello", "proto": PROTO, "client": self.name}, None)}
        )
        if errors:
            raise NixlDramError(errors[s])
        hello, payload = replies[s]
        if hello.get("service") != self.service_names[s]:
            raise NixlDramError(f"hello answered by {hello.get('service')}")
        try:
            self.transport.remove_remote(self.remotes[s])
        except Exception as e:  # already gone (a failed earlier re-add): add it anyway
            logger.warning(
                "HiCache NIXL DRAM L3: remove of %s's agent: %r",
                self.service_names[s],
                e,
            )
        self.remotes[s] = self.transport.add_remote(hello, payload)
        return hello.get("pid")

    def _log_reconnects(self):
        now = time.monotonic()
        with self._lock:
            due = now - self._reconnect_logged >= 60.0
            if due:
                self._reconnect_logged = now
                c = dict(self.counters)
        if due:
            logger.info(
                "HiCache NIXL DRAM L3: reconnects %d ok, %d failed, %d skipped (backoff or in progress); %d reads "
                "and %d writes retried; %d transfer errors",
                c["reconnects"],
                c["reconnect_failed"],
                c["reconnect_skipped"],
                c["retried_reads"],
                c["retried_writes"],
                c["xfer_errors"],
            )

    # -- the store interface MooncakeStore uses
    def set_page_units(self, units: Dict[int, int]):
        """{buffer address: a whole number of pages in bytes} for buffers registered later (reg_chunk_mib)."""
        self.page_units.update(units)

    def register_buffer(self, ptr: int, size: int) -> int:
        ptr, size = int(ptr), int(size)
        if self._staging is not None:
            # Not registered; a uint8 view of it lets staged copies run as torch ops.
            view = (ctypes.c_uint8 * size).from_address(ptr)
            self._host_views.append(
                (ptr, size, torch.frombuffer(view, dtype=torch.uint8))
            )
            self._host_views.sort(key=lambda v: v[0])
            self._row_views.clear()
            return 0
        ranges = registration_ranges(
            ptr, size, self.page_units.get(ptr, 0), self.cfg.reg_chunk_mib << 20
        )
        self.transport.register_local(ranges)
        logger.info(
            "HiCache NIXL DRAM L3: registered host buffer 0x%x, %.2f GiB in %d descriptor(s)",
            ptr,
            size / (1 << 30),
            len(ranges),
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
        plans, busy = self._plan_writes(groups, replies, results)
        state = self._transfer("WRITE", plans, ptr_parts, size_parts)
        self._commit(plans, state, keys, tokens, totals, results)
        if busy:
            # Another writer of the same key (hence the same bytes): stored only once it committed.
            present = self.batch_is_exist([keys[i] for i in busy])
            for i, p in zip(busy, present):
                results[i] = 0 if p else ERR_BUSY
        return results

    @staticmethod
    def _plan_writes(groups, replies, results):
        """({service: [(object, slot address)]} to write, objects another writer of the same key is writing); an
        object already stored succeeds without a write."""
        plans, busy, no_space = {}, [], 0
        for s, (reply, _) in replies.items():
            plans[s] = []
            for i, addr in zip(groups[s], reply["addrs"]):
                if addr >= 0:
                    plans[s].append((i, addr))
                elif addr == EXISTS:
                    results[i] = 0
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
        return plans, busy

    def _commit(self, plans, state, keys, tokens, totals, results):
        """Publish what was written, free what failed; a timed-out write keeps its slots until the lease ends."""
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
        lost = 0
        for s, items in plans.items():
            done = replies[s][0]["done"] if s in replies else [False] * len(items)
            for (i, _), d in zip(items, done):
                if d:
                    results[i] = 0
                else:
                    results[i] = ERR_LEASE if state.get(s) == "DONE" else ERR_XFER
                    lost += state.get(s) == "DONE"
        if lost:
            logger.error(
                "HiCache NIXL DRAM L3: %d written object(s) were not committed (lease expired?)",
                lost,
            )

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
        plans = self._plan_reads(groups, replies, totals, results)
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
        for s, items in plans.items():
            for i, _ in items:
                results[i] = totals[i] if state.get(s) == "DONE" else ERR_XFER
        return results

    @staticmethod
    def _plan_reads(groups, replies, totals, results):
        """{service: [(object, slot address)]} to read; a stored object of another size is never read."""
        plans, mismatched = {}, 0
        for s, (reply, _) in replies.items():
            plans[s] = []
            for i, addr, size in zip(groups[s], reply["addrs"], reply["sizes"]):
                if addr < 0:
                    results[i] = ERR_MISSING
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
        return plans

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

    def close(self):
        try:
            self.transport.close()
        finally:
            self.rpc.close()
