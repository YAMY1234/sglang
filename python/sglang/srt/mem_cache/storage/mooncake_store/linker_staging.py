"""SGLANG_LINKER_EARLY_READS: speculative reads of a looked-up prefix into pinned
host staging on every PP stage as soon as PP0's lookup lands (pp_linker_relay.py
has the protocol). A staged request is (epoch, cycle) -> per-pool object slots;
when the relay's load reaches the stage, the request's private device slots are
filled from those slots by one scatter copy instead of a store read.

Never blocks: a request that does not fit the slab is not staged (the stage reads
at load time, as without early reads). Staging is keyed by (epoch, cycle); a cold
reset drops older epochs, and a staged request is freed when consumed, stopped,
finished or dropped. An entry whose early vote was posted stays until consumed.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from collections import deque

import msgspec
import numpy as np

logger = logging.getLogger(__name__)


class StagingSlab:
    """Per-pool fixed-size object slots in one host buffer (pinned on GPUs).
    ``pools``: {name: (object_bytes, slots)}; ``buffer``: a uint8 array or tensor of
    at least the total size, whose address is ``base``."""

    def __init__(self, pools: dict, buffer, base: int):
        self.buffer, self.base = buffer, base
        self.layout, offset = {}, 0
        for name, (obj_bytes, slots) in pools.items():
            self.layout[name] = (offset, obj_bytes, slots)
            offset += obj_bytes * slots
        self.size = offset
        self.free = {name: deque(range(slots)) for name, (_, _, slots) in self.layout.items()}
        self.lock = threading.Lock()

    @staticmethod
    def size_of(pools: dict) -> int:
        return sum(obj * n for obj, n in pools.values())

    def alloc(self, need: dict):
        """{name: count} -> {name: [slot]}, all or nothing (None when short)."""
        with self.lock:
            if any(len(self.free[name]) < n for name, n in need.items()):
                return None
            return {name: [self.free[name].popleft() for _ in range(n)] for name, n in need.items()}

    def release(self, slots: dict) -> None:
        with self.lock:
            for name, ids in slots.items():
                self.free[name].extend(ids)

    def ptrs(self, name: str, ids) -> np.ndarray:
        offset, obj_bytes, _ = self.layout[name]
        return self.base + offset + np.asarray(ids, dtype=np.int64) * obj_bytes

    def object_bytes(self, name: str) -> int:
        return self.layout[name][1]


def slab_pools(object_bytes: dict, state_pool, pages: int, states: int, cap_bytes: int) -> dict:
    """{pool: (object bytes, slots)}: ``states`` slots for ``state_pool``, ``pages`` for
    every other pool (the same page count on every stage, whatever its layer count);
    when that exceeds ``cap_bytes`` the page slots shrink to fit."""
    state_bytes = object_bytes.get(state_pool, 0) * states
    page_bytes = sum(b for name, b in object_bytes.items() if name != state_pool)
    if page_bytes and state_bytes + pages * page_bytes > cap_bytes:
        pages = max(0, (cap_bytes - state_bytes) // page_bytes)
    return {name: (b, states if name == state_pool else pages) for name, b in object_bytes.items()}


def scatter_plan(table: np.ndarray, rows: np.ndarray, slot_ptrs: np.ndarray):
    """(dst, src, sizes) int64 arrays copying staged objects into device rows.
    ``table``: (src, base, stride, size, offset) per item of one key (the read's
    key_range_table, coalesced or not); ``rows``: [keys, columns] device rows;
    ``slot_ptrs``: [keys] staged object addresses."""
    src_col, base, stride, size, offset = table.T
    dst = base[None, :] + rows[:, src_col] * stride[None, :]
    src = slot_ptrs[:, None] + offset[None, :]
    sizes = np.broadcast_to(size, dst.shape)
    return dst.reshape(-1), src.reshape(-1), np.ascontiguousarray(sizes).reshape(-1)


def cpu_scatter(dst, src, sizes) -> None:
    for d, s, n in zip(dst.tolist(), src.tolist(), sizes.tolist()):
        ctypes.memmove(d, s, n)


def coalesce_ranges(dst, src, sizes):
    """Merge consecutive ranges that are contiguous on both sides (fewer entries
    for the batched copy). Arrays of equal length, in copy order."""
    dst, src, sizes = (np.asarray(a, dtype=np.int64) for a in (dst, src, sizes))
    if len(dst) < 2:
        return dst, src, sizes
    joined = (dst[:-1] + sizes[:-1] == dst[1:]) & (src[:-1] + sizes[:-1] == src[1:])
    starts = np.flatnonzero(np.concatenate(([True], ~joined)))
    return dst[starts], src[starts], np.add.reduceat(sizes, starts)


def copy_engine_scatter(plans, stream):
    """Run (dst, src, sizes) plans as one cudaMemcpyBatchAsync each on ``stream``
    (copy engines, no SM time); returns the event recorded after them. Called from
    the staging thread: the driver spends about 0.3 us per entry."""
    import torch

    from sglang.srt.mem_cache.storage.mooncake_store.linker_staging_kernel import loaded_cudart, memcpy_batch

    with torch.cuda.stream(stream):
        for dst, src, sizes in plans:
            if len(dst):
                try:
                    memcpy_batch(*coalesce_ranges(dst, src, sizes), stream.cuda_stream, stream.device.index)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"{e}; stream {stream.device}, torch current {torch.cuda.current_device()}, cudart {loaded_cudart()}"
                    ) from None
        event = torch.cuda.Event()
        event.record()
    return event


def gpu_scatter(plans, stream):
    """Run (dst, src, sizes) plans with one Triton launch each on ``stream`` (the GPU
    reads the pinned staging directly); returns the event recorded after them."""
    import torch

    from sglang.srt.mem_cache.storage.mooncake_store.linker_staging_kernel import scatter_copy

    with torch.cuda.stream(stream):
        for dst, src, sizes in plans:
            if len(dst):
                scatter_copy(dst, src, sizes)
        event = torch.cuda.Event()
        event.record()
    return event


class EarlyMessage(msgspec.Struct, array_like=True, frozen=True):
    epoch: int
    cycle: int
    rid: str
    hashes: list[str]


class _Staged:
    __slots__ = ("pages", "slots", "state", "voted", "rid")

    def __init__(self, pages, slots, rid=None):
        self.pages, self.slots, self.rid = pages, slots, rid
        self.state = "reading"  # reading -> landed | failed; dropped while reading -> released on completion
        self.voted = False


class EarlyReads:
    """One stage's early reads. ``plan(hashes)`` -> {pool: (object keys, object_bytes)}
    for this stage; ``read(keys_by_pool, ptrs_by_pool)`` -> Future[bool]; ``send``
    (PP0 only) publishes an EarlyMessage to every other stage."""

    def __init__(self, slab: StagingSlab, plan, read, send=None):
        self.slab, self.plan, self.read, self.send = slab, plan, read, send
        self.entries: dict = {}
        self.landed: deque = deque()
        self.lock = threading.Lock()
        self.epoch = 0
        self.counts = {"staged": 0, "full": 0, "failed": 0, "used": 0, "dropped": 0, "voted_unused": 0, "voted_direct": 0}

    # ---- PP0 ---------------------------------------------------------------------
    def publish(self, epoch, cycle, handle, lookup, pages) -> None:
        msg = EarlyMessage(epoch, cycle, str(getattr(handle, "rid", handle)), list(lookup.hashes[:pages]))
        if self.send is not None:
            self.send(msg)
        self.start(msg)

    # ---- every stage -----------------------------------------------------------
    def start(self, msg: EarlyMessage) -> None:
        if msg.epoch < self.epoch or not msg.hashes:
            return
        objects = self.plan(msg.hashes)
        slots = self.slab.alloc({name: len(keys) for name, (keys, _) in objects.items()})
        if slots is None:
            self.counts["full"] += 1
            return
        entry = _Staged(len(msg.hashes), slots, msg.rid)
        with self.lock:
            self.entries[(msg.epoch, msg.cycle)] = entry
        ptrs = {name: self.slab.ptrs(name, ids) for name, ids in slots.items()}
        future = self.read({name: keys for name, (keys, _) in objects.items()}, ptrs)
        future.add_done_callback(lambda f, key=(msg.epoch, msg.cycle), e=entry: self._done(key, e, f))

    def _done(self, key, entry, future) -> None:
        try:
            ok = future.result() is True
        except Exception:
            ok = False
        with self.lock:
            if entry.state == "dropped" or self.entries.get(key) is not entry:
                entry.state = "dropped"
                self.slab.release(entry.slots)
                return
            entry.state = "landed" if ok else "failed"
            if ok:
                self.landed.append(key)
                self.counts["staged"] += 1
            else:
                self.counts["failed"] += 1
                self.entries.pop(key, None)
                self.slab.release(entry.slots)

    def take_landed(self) -> list:
        """Landed reads to vote on; their entries stay until consumed or dropped."""
        out = []
        with self.lock:
            while self.landed:
                key = self.landed.popleft()
                entry = self.entries.get(key)
                if entry is not None and entry.state == "landed" and key[0] >= self.epoch:
                    entry.voted = True
                    out.append(key)
        return out

    def take(self, epoch, cycle, pages, rid=None):
        """The landed staging of this load, removed from the table (the caller
        consumes and then ``release``s it), or None (not staged, other pages).
        ``rid``: counts a load that reads directly although an early read of the
        same request was voted under any key (the divergence-risk case)."""
        with self.lock:
            entry = self.entries.get((epoch, cycle))
            if entry is None:
                if rid is not None and any(e.voted and e.rid == rid for e in self.entries.values()):
                    self.counts["voted_direct"] += 1
                    logger.warning("Early read of %s voted under another key; load %s reads directly",
                                   rid, (epoch, cycle))
                return None
            if entry.voted and (entry.state != "landed" or entry.pages != pages):
                # Near miss: PP0 may finish on this vote while the stage reads at load
                # time (the protocol rules it out; counted to size the risk).
                self.counts["voted_unused"] += 1
                logger.warning("Early read %s voted but not usable at load (state %s, pages %d vs %d)",
                               (epoch, cycle), entry.state, entry.pages, pages)
            if entry.state != "landed" or entry.pages != pages:
                # The stage reads at load time instead; a read still in flight must
                # never be voted after this (its vote would stand for a read the
                # stage did not use).
                if entry.state == "reading":
                    del self.entries[(epoch, cycle)]
                    entry.state = "dropped"
                    self.counts["dropped"] += 1
                return None
            del self.entries[(epoch, cycle)]
            self.counts["used"] += 1
            return entry

    def release(self, entry) -> None:
        self.slab.release(entry.slots)

    def drop(self, epoch, cycle) -> None:
        with self.lock:
            entry = self.entries.pop((epoch, cycle), None)
            if entry is None:
                return
            self.counts["dropped"] += 1
            if entry.state == "reading":
                entry.state = "dropped"  # released when the read completes
                return
        self.slab.release(entry.slots)

    def reset(self, epoch) -> None:
        with self.lock:
            self.epoch = epoch
            stale = [k for k in self.entries if k[0] < epoch]
        for key in stale:
            self.drop(*key)


class EarlyChannel:
    """PP0 -> every stage: PUB on PP0's host, SUB elsewhere (address broadcast over
    the PP group, as PPConsensusStore). msgpack only (the socket is unauthenticated);
    each stage receives on a daemon thread and calls ``on_message``."""

    def __init__(self, pp_rank, pp_group, on_message=None):
        import zmq

        from sglang.srt.utils import network

        self.rank = pp_rank
        self.ctx = zmq.Context()
        self.encoder = msgspec.msgpack.Encoder()
        # The network helpers configure PUSH/PULL-style sockets only: PUB/SUB here.
        if pp_rank == 0:
            host = network.get_local_ip_auto()
            self.socket = self.ctx.socket(zmq.PUB)
            self.socket.setsockopt(zmq.SNDHWM, 0)
            self.socket.setsockopt(zmq.LINGER, 0)
            port = self.socket.bind_to_random_port(f"tcp://{host}")
            info = (host, port)
        else:
            info = None
        host, port = pp_group.broadcast_object(info, src=0)
        if pp_rank > 0:
            self.socket = self.ctx.socket(zmq.SUB)
            self.socket.setsockopt(zmq.RCVHWM, 0)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.socket.connect(network.NetworkAddress(host, port).to_tcp())
            decoder = msgspec.msgpack.Decoder(EarlyMessage)

            def loop():
                while True:
                    try:
                        raw = self.socket.recv()
                    except zmq.error.ContextTerminated:
                        self.socket.close(linger=0)  # term() waits for every socket
                        return
                    try:
                        msg = decoder.decode(raw)
                    except msgspec.MsgspecError:
                        logger.warning("Early-read channel dropped a malformed message")
                        continue
                    on_message(msg)

            threading.Thread(target=loop, name="linker-early-rx", daemon=True).start()
        self.lock = threading.Lock()

    def send(self, msg: EarlyMessage) -> None:
        with self.lock:
            self.socket.send(self.encoder.encode(msg))

    def close(self) -> None:
        if self.rank == 0:
            self.socket.close(linger=0)
        self.ctx.term()
