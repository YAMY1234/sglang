"""SGLANG_DSV41_TAIL_STAGE on an encoder-only prefill (stage 0): the tail server's client.

Stage 0 asks the tail server for a late-fed request's slot when the request's final
prefill chunk is scheduled. Once its forward has run all of the prefix [0, e), it writes
the prefix's c1 KV and c1-index pages and the window's late-layer input into the granted
slot, reports the slot ready, and holds its pages until both this write and its decode
send have ended. Messages are in `dsv41_tail_stage.protocol`, buffers in `dsv41_tail_stage.layout`.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import msgspec
import numpy as np

from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.dsv41_tail_stage.protocol import (
    Error,
    Heartbeat,
    Hello,
    ProtocolError,
    SlotDefer,
    SlotGrant,
    SlotRequest,
    TailAbort,
    TailDone,
    TailReady,
    Welcome,
    decode,
    encode,
)

logger = logging.getLogger(__name__)

# Arbitrary: how often an unwelcomed stage 0 repeats its Hello (a tail server that
# starts later, or restarted, answers the next one).
HELLO_RETRY_S = 5.0
# Period of the per-request timing summary.
STATS_LOG_S = 5.0
# Arbitrary: slot writes in flight at once, and parallel Mooncake batches per write
# (the KV sender also spreads one request over its executors under the NVLink pool).
WRITE_ROOMS = 4
WRITE_PARTS = 8
# Arbitrary: no write is cut into batches smaller than this.
MIN_WRITE_PART_BYTES = 1 << 20
UNREACHABLE = "DSV4.1 tail stage unreachable"


class Stage0Pools(msgspec.Struct, frozen=True):
    """What stage 0 writes a slot from: its c1 KV and c1-index entries (one item = one
    full page) and its late-tail capture pool (one item = one window)."""

    page_size: int
    c1_kv_ptrs: List[int]
    c1_kv_item_lens: List[int]
    c1_index_ptrs: List[int]
    c1_index_item_lens: List[int]
    residual_ptr: int
    residual_item_len: int


def layout_error(pools: Stage0Pools, welcome: Welcome) -> Optional[str]:
    """Why stage 0 cannot write into this tail server's slots, if it cannot."""
    mine = (
        pools.page_size,
        list(pools.c1_kv_item_lens),
        list(pools.c1_index_item_lens),
        pools.residual_item_len,
    )
    theirs = (
        welcome.page_size,
        list(welcome.c1_kv_item_lens),
        list(welcome.c1_index_item_lens),
        welcome.residual_item_len,
    )
    if mine != theirs:
        return (
            "tail stage layout (page size, c1 KV items, c1 index items, residual "
            f"item) {theirs} differs from this prefill's {mine}"
        )
    if len(welcome.c1_kv_ptrs) != len(pools.c1_kv_ptrs) or len(
        welcome.c1_index_ptrs
    ) != len(pools.c1_index_ptrs):
        return "tail stage registers a different number of c1 entries"
    return None


def push_blocks(
    *,
    pools: Stage0Pools,
    welcome: Welcome,
    grant: SlotGrant,
    src_pages: np.ndarray,
    capture_slot: int,
) -> List[Tuple[int, int, int]]:
    """(src, dst, length) of one slot's writes: every c1 KV and c1-index page of the
    prefix, page for page in position order, then the window's residual."""
    dst_pages = np.asarray(grant.c1_pages, dtype=np.int64)
    src_pages = np.asarray(src_pages, dtype=np.int64)
    assert src_pages.shape == dst_pages.shape, (src_pages.shape, dst_pages.shape)
    src_runs, dst_runs = group_concurrent_contiguous(src_pages, dst_pages)
    blocks = []
    for src_ptrs, dst_ptrs, items in (
        (pools.c1_kv_ptrs, welcome.c1_kv_ptrs, pools.c1_kv_item_lens),
        (pools.c1_index_ptrs, welcome.c1_index_ptrs, pools.c1_index_item_lens),
    ):
        for src_ptr, dst_ptr, item in zip(src_ptrs, dst_ptrs, items):
            for src_run, dst_run in zip(src_runs, dst_runs):
                blocks.append(
                    (
                        src_ptr + int(src_run[0]) * item,
                        dst_ptr + int(dst_run[0]) * item,
                        item * len(src_run),
                    )
                )
    item = pools.residual_item_len
    blocks.append(
        (
            pools.residual_ptr + capture_slot * item,
            welcome.residual_ptr + grant.residual_slot * item,
            item,
        )
    )
    return blocks


def split_blocks(
    blocks: List[Tuple[int, int, int]], *, parts: int, min_bytes: int
) -> List[List[Tuple[int, int, int]]]:
    """The writes cut into at most `parts` runs of about equal bytes (at least
    `min_bytes`), one Mooncake batch each; a larger block is cut into pieces."""
    total = sum(length for _, _, length in blocks)
    target = max(min_bytes, -(-total // parts))
    pieces = []
    for src, dst, length in blocks:
        for offset in range(0, length, target):
            pieces.append((src + offset, dst + offset, min(target, length - offset)))
    out, part, size = [], [], 0
    for piece in pieces:
        part.append(piece)
        size += piece[2]
        if size >= target and len(out) < parts - 1:
            out.append(part)
            part, size = [], 0
    if part:
        out.append(part)
    return out


class _Room(msgspec.Struct):
    request: SlotRequest
    t_request: float
    t_grant: Optional[float] = None
    t_push: Optional[float] = None
    retry_at: Optional[float] = None
    grant: Optional[SlotGrant] = None
    error: Optional[str] = None
    push: Optional[concurrent.futures.Future] = None
    descriptors: int = 0
    nbytes: int = 0
    # The request left; the message goes once the slot's write has ended.
    closing: Optional[object] = None


class _Timings:
    """Per-request milliseconds over one log period, by stage."""

    NAMES = (
        "wait_grant",
        "grant_to_ready",
        "write",
        "descriptors",
        "write_mb",
        "s1_queue",
        "s1_forward",
        "s1_send",
    )

    def __init__(self) -> None:
        self.samples: Dict[str, List[float]] = {name: [] for name in self.NAMES}

    def add(self, name: str, ms: float) -> None:
        self.samples[name].append(ms)

    def summary(self) -> str:
        parts = []
        for name, values in self.samples.items():
            if values:
                p50, p90 = np.percentile(values, [50, 90])
                parts.append(f"{name} {p50:.1f}/{p90:.1f} (n={len(values)})")
        return ", ".join(parts)


class TailStageClient:
    """One per stage-0 scheduler. Every method runs on the scheduler thread except
    the Mooncake writes, which run on workers and are read back by `push_result`, and
    the heartbeat thread, which keeps a stalled scheduler (a cold forward) alive."""

    def __init__(
        self,
        *,
        socket,
        hello: Hello,
        pools: Stage0Pools,
        transfer,
        heartbeat_s: float,
        timeout_s: float,
        clock=time.monotonic,
        heartbeat_thread: bool = False,
    ) -> None:
        self._socket = socket
        self._hello = hello
        self._pools = pools
        # transfer(session, src_addrs, dst_addrs, lengths) -> 0 on success.
        self._transfer = transfer
        self._heartbeat_s = heartbeat_s
        self._timeout_s = timeout_s
        self._clock = clock
        self._rooms: Dict[int, _Room] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=WRITE_ROOMS, thread_name_prefix="dsv41-tail-push"
        )
        self._part_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=WRITE_PARTS, thread_name_prefix="dsv41-tail-push-part"
        )
        # The socket is not thread-safe: the heartbeat thread shares it under this lock.
        self._socket_lock = threading.Lock()
        self.welcome: Optional[Welcome] = None
        self.refused: Optional[str] = None
        self._heard_at: Optional[float] = None
        self._hello_at = -float("inf")
        self._beat_at = -float("inf")
        self.tails_pushed = 0
        self._timings = _Timings()
        self._timings_at = clock()
        self._stop = threading.Event()
        if heartbeat_thread:
            threading.Thread(
                target=self._heartbeat_loop, name="dsv41-tail-beat", daemon=True
            ).start()

    # -- the scheduler loop --

    def poll(self) -> None:
        """Drain the tail server's messages, resend due requests, keep alive."""
        now = self._clock()
        while (data := self._recv()) is not None:
            try:
                self._on_message(decode(data), now)
            except ProtocolError as exc:
                logger.error("DSV4.1 tail stage: %s", exc)
        # A silent tail server may have restarted: it registers again on a Hello.
        if (self.welcome is None or self.unreachable()) and (
            now - self._hello_at >= HELLO_RETRY_S
        ):
            self._hello_at = now
            self._send(self._hello)
        for room_id, room in list(self._rooms.items()):
            if room.closing is not None:
                if room.push.done():
                    del self._rooms[room_id]
                    self._send(room.closing)
                continue
            if room.retry_at is not None and now >= room.retry_at:
                room.retry_at = None
                self._send(room.request)
        self._heartbeat(now)
        if now - self._timings_at >= STATS_LOG_S:
            summary = self._timings.summary()
            if summary:
                logger.info("DSV4.1 tail stage ms p50/p90: %s", summary)
            self._timings = _Timings()
            self._timings_at = now

    def unreachable(self) -> bool:
        """A welcomed tail server that has gone silent."""
        return (
            self._heard_at is not None
            and self._clock() - self._heard_at > self._timeout_s
        )

    # -- one room --

    def request(self, *, room: int, rid: str, s: int, e: int, late_fed: bool) -> None:
        """Ask for the room's slot once; a pass-through room asks too, so the tail
        server answers decode for it."""
        if room in self._rooms:
            return
        msg = SlotRequest(
            room=room, rid=rid, s=s, e=e if late_fed else 0, late_fed=late_fed
        )
        self._rooms[room] = _Room(request=msg, t_request=self._clock())
        self._send(msg)
        if not late_fed:
            # Nothing comes back that stage 0 waits for.
            del self._rooms[room]

    def requested(self, room: int) -> bool:
        return room in self._rooms

    def grant(self, room: int) -> Optional[SlotGrant]:
        entry = self._rooms.get(room)
        if entry is None or entry.closing is not None or entry.error is not None:
            return None
        return entry.grant

    def error(self, room: int) -> Optional[str]:
        """Why the room's slot will never come, if it will not."""
        if self.refused is not None:
            return self.refused
        if self.unreachable():
            return UNREACHABLE
        entry = self._rooms.get(room)
        return None if entry is None else entry.error

    def start_push(
        self, *, room: int, src_pages: np.ndarray, capture_slot: int
    ) -> bool:
        """Start the slot's writes; refused (False) once the slot has failed, as the
        tail server may hand a freed slot to another room."""
        if self.grant(room) is None:
            return False
        entry = self._rooms[room]
        assert entry.push is None
        blocks = push_blocks(
            pools=self._pools,
            welcome=self.welcome,
            grant=entry.grant,
            src_pages=src_pages,
            capture_slot=capture_slot,
        )
        entry.t_push = self._clock()
        entry.descriptors = len(blocks)
        entry.nbytes = sum(length for _, _, length in blocks)
        entry.push = self._executor.submit(self._write, blocks)
        return True

    def push_result(self, room: int) -> Optional[bool]:
        """None while the room's writes run, else whether they succeeded; a write
        into a slot that failed meanwhile does not count."""
        entry = self._rooms[room]
        if entry.push is None or not entry.push.done():
            return None
        return entry.push.result() and entry.error is None

    def pushing(self, room: int) -> bool:
        entry = self._rooms.get(room)
        return entry is not None and entry.push is not None

    def ready(self, room: int, *, ok: bool) -> None:
        """The slot's writes have returned (or stage 0 gave up on it)."""
        entry = self._rooms.get(room)
        if entry is None or entry.grant is None or entry.closing is not None:
            return
        self._close(room, TailReady(room=room, slot=entry.grant.slot, ok=ok))
        self.tails_pushed += ok
        now = self._clock()
        self._timings.add("grant_to_ready", (now - entry.t_grant) * 1e3)
        if entry.t_push is not None:
            self._timings.add("write", (now - entry.t_push) * 1e3)
            self._timings.add("descriptors", entry.descriptors)
            self._timings.add("write_mb", entry.nbytes / 1e6)

    def abort(self, room: int) -> None:
        """The request left before its slot was ready. A granted slot is given back
        with TailReady(ok=False): the tail server holds a granted slot until that."""
        entry = self._rooms.get(room)
        if entry is None or entry.closing is not None:
            return
        if entry.grant is None:
            self._close(room, TailAbort(room=room))
        else:
            self._close(room, TailReady(room=room, slot=entry.grant.slot, ok=False))

    def _close(self, room: int, msg) -> None:
        entry = self._rooms[room]
        if entry.push is not None and not entry.push.done():
            # The slot is reused once the tail server hears of it: a write still
            # in flight must land first.
            entry.closing = msg
            return
        del self._rooms[room]
        self._send(msg)

    def close(self) -> None:
        self._stop.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._part_executor.shutdown(wait=False, cancel_futures=True)

    # -- internals --

    def _write(self, blocks: List[Tuple[int, int, int]]) -> bool:
        parts = split_blocks(blocks, parts=WRITE_PARTS, min_bytes=MIN_WRITE_PART_BYTES)
        if len(parts) == 1:
            return self._write_part(parts[0])
        futures = [self._part_executor.submit(self._write_part, p) for p in parts]
        return all([f.result() for f in futures])

    def _write_part(self, blocks: List[Tuple[int, int, int]]) -> bool:
        src, dst, lengths = zip(*blocks)
        rc = self._transfer(
            self.welcome.tail_session, list(src), list(dst), list(lengths)
        )
        if rc != 0:
            logger.error("DSV4.1 tail stage: slot write failed (rc=%s)", rc)
        return rc == 0

    def _heartbeat(self, now: float) -> None:
        with self._socket_lock:
            if self.welcome is None or now - self._beat_at < self._heartbeat_s:
                return
            self._beat_at = now
            self._socket.send(encode(Heartbeat(inflight=len(self._rooms))))

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self._heartbeat_s / 2):
            self._heartbeat(self._clock())

    def _send(self, msg) -> None:
        with self._socket_lock:
            self._socket.send(encode(msg))

    def _recv(self) -> Optional[bytes]:
        with self._socket_lock:
            return self._socket.recv()

    def _on_message(self, msg, now: float) -> None:
        self._heard_at = now
        if isinstance(msg, Welcome):
            self.refused = layout_error(self._pools, msg)
            if self.refused is not None:
                logger.error(
                    "DSV4.1 tail stage: %s; late-fed requests fail", self.refused
                )
                return
            if self.welcome is None:
                logger.info(
                    "DSV4.1 tail stage: welcomed by %s as PP rank %d, %d slots",
                    msg.tail_session,
                    msg.registered_pp_rank,
                    msg.num_slots,
                )
            self.welcome = msg
        elif isinstance(msg, SlotGrant):
            entry = self._rooms.get(msg.room)
            if msg.slot < 0:
                # A pass-through room's answer: nothing to wait for or give back.
                return
            if entry is None:
                # Aborted meanwhile: give the slot back.
                self._send(TailReady(room=msg.room, slot=msg.slot, ok=False))
            else:
                entry.grant = msg
                entry.t_grant = now
                self._timings.add("wait_grant", (now - entry.t_request) * 1e3)
        elif isinstance(msg, SlotDefer):
            entry = self._rooms.get(msg.room)
            if entry is not None:
                entry.retry_at = now + msg.retry_ms / 1000
        elif isinstance(msg, Error):
            logger.error("DSV4.1 tail stage error (room %s): %s", msg.room, msg.reason)
            entry = None if msg.room is None else self._rooms.get(msg.room)
            if entry is not None:
                entry.error = msg.reason
        elif isinstance(msg, TailDone):
            self._timings.add("s1_queue", msg.queue_ms)
            self._timings.add("s1_forward", msg.forward_ms)
            self._timings.add("s1_send", msg.send_ms)
            if msg.status != "ok":
                logger.warning(
                    "DSV4.1 tail stage: room %d %s (%s)",
                    msg.room,
                    msg.status,
                    msg.reason,
                )
                entry = self._rooms.get(msg.room)
                if entry is not None and entry.closing is None:
                    # No write may start into the slot; the request fails, and its
                    # answer (TailReady(ok=False) or TailAbort) frees the slot.
                    entry.error = f"tail stage {msg.status}: {msg.reason}"
        elif not isinstance(msg, Heartbeat):
            logger.warning("DSV4.1 tail stage: unexpected %s", type(msg).__name__)


class ZmqDealer:
    """The client's socket: one DEALER to the tail server's ROUTER, identity
    stage0_id; `recv` returns None when nothing is waiting."""

    def __init__(self, *, addr: str, identity: str) -> None:
        import zmq

        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.setsockopt(zmq.IDENTITY, identity.encode())
        self._sock.setsockopt(zmq.LINGER, 0)
        # Messages queue until the tail server is up; none may be dropped.
        self._sock.setsockopt(zmq.SNDHWM, 0)
        self._sock.connect(f"tcp://{addr}")

    def send(self, data: bytes) -> None:
        self._sock.send(data)

    def recv(self) -> Optional[bytes]:
        try:
            return self._sock.recv(flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            return None
