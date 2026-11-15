"""SGLANG_DSV41_TAIL_STAGE on an encoder-only prefill (stage 0): the tail server's client.

Stage 0 asks the tail server for a slot at bootstrap (a late-fed request waits there for
its grant), and once its own decode transfer has succeeded writes the c1 KV and c1-index
pages of the prefix [0, e) and the window's late-layer input into the granted slot, then
reports the slot ready. Messages are in `dsv41_tail_stage.protocol`.
"""

from __future__ import annotations

import concurrent.futures
import logging
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


def c1_buf_infos(pool) -> Tuple[List[int], List[int], List[int], List[int]]:
    """The ratio-1 (c1) KV and index-K entries of a DeepSeek-V4.1 pool, addressed by
    full page id as `get_contiguous_buf_infos` registers them for PD."""
    kv_ptrs, kv_items, index_ptrs, index_items = [], [], [], []
    for buf in pool.kv_pools[1].kv_buffer:
        assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
        kv_ptrs.append(buf.data_ptr())
        kv_items.append(buf[0].nbytes)
    index_pool = pool.index_pools[1]
    index_pages_per_full_page = pool.page_size // index_pool.page_size
    for buf in index_pool.contiguous_page_row_buffers():
        assert buf.ndim == 2, f"expected 2D buffer, got {buf.ndim}D"
        index_ptrs.append(buf.data_ptr())
        index_items.append(buf[0].nbytes * index_pages_per_full_page)
    return kv_ptrs, kv_items, index_ptrs, index_items


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


class _Room(msgspec.Struct):
    request: SlotRequest
    retry_at: Optional[float] = None
    grant: Optional[SlotGrant] = None
    error: Optional[str] = None
    push: Optional[concurrent.futures.Future] = None


class TailStageClient:
    """One per stage-0 scheduler. Every method runs on the scheduler thread except
    the Mooncake writes, which run on a worker and are read back by `push_result`."""

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
            max_workers=1, thread_name_prefix="dsv41-tail-push"
        )
        self.welcome: Optional[Welcome] = None
        self.refused: Optional[str] = None
        self._heard_at: Optional[float] = None
        self._hello_at = -float("inf")
        self._beat_at = -float("inf")
        self.tails_pushed = 0

    # -- the scheduler loop --

    def poll(self) -> None:
        """Drain the tail server's messages, resend due requests, keep alive."""
        now = self._clock()
        while (data := self._socket.recv()) is not None:
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
        for room in self._rooms.values():
            if room.retry_at is not None and now >= room.retry_at:
                room.retry_at = None
                self._send(room.request)
        if self.welcome is not None and now - self._beat_at >= self._heartbeat_s:
            self._beat_at = now
            self._send(Heartbeat(inflight=len(self._rooms)))

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
        msg = SlotRequest(room=room, rid=rid, s=s, e=e if late_fed else 0, late_fed=late_fed)
        self._rooms[room] = _Room(request=msg)
        self._send(msg)
        if not late_fed:
            # Nothing comes back that stage 0 waits for.
            del self._rooms[room]

    def grant(self, room: int) -> Optional[SlotGrant]:
        entry = self._rooms.get(room)
        return None if entry is None else entry.grant

    def error(self, room: int) -> Optional[str]:
        """Why the room's slot will never come, if it will not."""
        if self.refused is not None:
            return self.refused
        if self.unreachable():
            return UNREACHABLE
        entry = self._rooms.get(room)
        return None if entry is None else entry.error

    def start_push(self, *, room: int, src_pages: np.ndarray, capture_slot: int) -> None:
        entry = self._rooms[room]
        assert entry.grant is not None and entry.push is None
        blocks = push_blocks(
            pools=self._pools,
            welcome=self.welcome,
            grant=entry.grant,
            src_pages=src_pages,
            capture_slot=capture_slot,
        )
        entry.push = self._executor.submit(self._write, blocks)

    def push_result(self, room: int) -> Optional[bool]:
        """None while the room's writes run, else whether they succeeded."""
        push = self._rooms[room].push
        if push is None or not push.done():
            return None
        return push.result()

    def pushing(self, room: int) -> bool:
        entry = self._rooms.get(room)
        return entry is not None and entry.push is not None

    def ready(self, room: int, *, ok: bool) -> None:
        """The slot's writes have returned (or stage 0 gave up on it)."""
        entry = self._rooms.pop(room, None)
        if entry is None or entry.grant is None:
            return
        if entry.push is not None and not entry.push.done():
            # A write still in flight may yet land; its result is dropped.
            ok = False
        self._send(TailReady(room=room, slot=entry.grant.slot, ok=ok))
        self.tails_pushed += ok

    def abort(self, room: int) -> None:
        """The request left before its slot was ready."""
        entry = self._rooms.pop(room, None)
        if entry is not None:
            self._send(TailAbort(room=room))

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- internals --

    def _write(self, blocks: List[Tuple[int, int, int]]) -> bool:
        src, dst, lengths = zip(*blocks)
        rc = self._transfer(
            self.welcome.tail_session, list(src), list(dst), list(lengths)
        )
        if rc != 0:
            logger.error("DSV4.1 tail stage: slot write failed (rc=%s)", rc)
        return rc == 0

    def _send(self, msg) -> None:
        self._socket.send(encode(msg))

    def _on_message(self, msg, now: float) -> None:
        self._heard_at = now
        if isinstance(msg, Welcome):
            self.refused = layout_error(self._pools, msg)
            if self.refused is not None:
                logger.error("DSV4.1 tail stage: %s; late-fed requests fail", self.refused)
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
            if entry is None:
                # Aborted meanwhile: give the slot back.
                self._send(TailReady(room=msg.room, slot=msg.slot, ok=False))
            else:
                entry.grant = msg
        elif isinstance(msg, SlotDefer):
            entry = self._rooms.get(msg.room)
            if entry is not None:
                entry.retry_at = now + msg.retry_ms / 1000
        elif isinstance(msg, Error):
            logger.error(
                "DSV4.1 tail stage error (room %s): %s", msg.room, msg.reason
            )
            entry = None if msg.room is None else self._rooms.get(msg.room)
            if entry is not None:
                entry.error = msg.reason
        elif isinstance(msg, TailDone):
            if msg.status != "ok":
                logger.warning(
                    "DSV4.1 tail stage: room %d %s (%s)", msg.room, msg.status, msg.reason
                )
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
