"""Stage 1's slot accounting: requests, grants with backpressure, and the ready queue.

A slot holds one tail from grant to the end of its send to decode: a residual slot,
the c1 pages of its prefix [0, e - W), its SWA rows and its aux row. Grants go strictly
first-come over every stage 0, so a long prompt is never passed over; a stage 0 waits
in bootstrap until its grant arrives.
"""

from __future__ import annotations

import enum
import heapq
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Protocol, Set, Tuple

import msgspec
from sglang.srt.disaggregation.dsv41_tail_stage.protocol import (
    Error,
    SlotDefer,
    SlotGrant,
    SlotRequest,
    TailDone,
    TailMessage,
    TailReady,
    slot_request_error,
)

# Arbitrary: how long a stage 0 waits before resending a deferred request.
DEFER_RETRY_MS = 20


class SlotState(enum.IntEnum):
    REQUESTED = 0
    GRANTED = 1
    READY = 2
    RUNNING = 3
    SENDING = 4
    # Failed by stage 1 after its grant: stage 0 may still be writing the slot, so the
    # slot and pages stay held until stage 0 answers (TailReady or TailAbort).
    FAILED = 5


class TailSlot(msgspec.Struct):
    room: int
    stage0_id: str
    rid: str
    s: int
    e: int
    state: SlotState
    t_request: float
    slot: int = -1
    c1_pages: List[int] = []
    t_grant: float = 0.0
    t_ready: float = 0.0
    t_run: float = 0.0
    t_send: float = 0.0
    t_fail: float = 0.0
    fail_reason: str = ""
    # An abort that arrived while stage 0 may still write the slot or its send
    # reads it; the slot frees at TailReady or when the send ends.
    abort_pending: bool = False


class Reply(msgspec.Struct, frozen=True):
    stage0_id: str
    msg: TailMessage


class PageSource(Protocol):
    """Where a granted tail's prefix pages come from: stage 0 writes the c1 KV of
    [0, e - window) into them."""

    capacity: int

    def take(self, tail: TailSlot) -> Optional[List[int]]:
        """The prefix's page ids in position order, or None while they do not fit.
        `tail.slot` is already the slot it gets."""

    def end_round(self) -> None:
        """Once per loop step, after its grants: their device writes, at once."""

    def give_back(self, tail: TailSlot) -> None: ...


class CountedPages:
    """Page ids 1..num_pages handed out first come first served (page 0 holds the
    padding token slot, as in every SGLang KV pool)."""

    def __init__(self, *, num_pages: int, page_size: int, window: int) -> None:
        self.capacity = num_pages
        self.page_size = page_size
        self.window = window
        self.free: Deque[int] = deque(range(1, num_pages + 1))

    def take(self, tail: TailSlot) -> Optional[List[int]]:
        need = prefix_pages(e=tail.e, window=self.window, page_size=self.page_size)
        if need > len(self.free):
            return None
        return [self.free.popleft() for _ in range(need)]

    def end_round(self) -> None:
        pass

    def give_back(self, tail: TailSlot) -> None:
        self.free.extend(tail.c1_pages)


def prefix_pages(*, e: int, window: int, page_size: int) -> int:
    return -(-(e - window) // page_size)


class SlotTable:
    def __init__(
        self,
        *,
        num_slots: int,
        pages: PageSource,
        page_size: int,
        window: int,
        queue_cap: int,
        ready_timeout_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.num_slots = num_slots
        self.pages = pages
        self.page_size = page_size
        self.window = window
        self.queue_cap = queue_cap
        self.ready_timeout_s = ready_timeout_s
        self.clock = clock
        self.free_slots: Deque[int] = deque(range(num_slots))
        self.queue: Deque[int] = deque()
        self.ready_order: Deque[int] = deque()
        self.tails: Dict[int, TailSlot] = {}
        # Pass-through rooms: no slot, an empty send to decode.
        self.passthrough: Dict[int, str] = {}
        # (deadline, room, stamp): a granted or failed slot's timeout. Stale entries
        # (the room moved on) are dropped when they come due.
        self._deadlines: List[Tuple[float, int, float]] = []
        # Silent stage 0s whose queued and granted rooms were already failed.
        self._silenced: Set[str] = set()

    def counts(self) -> Dict[str, int]:
        by_state = {state: 0 for state in SlotState}
        for tail in self.tails.values():
            by_state[tail.state] += 1
        return {
            "free_slots": len(self.free_slots),
            "queued": by_state[SlotState.REQUESTED],
            "granted": by_state[SlotState.GRANTED],
            "ready": by_state[SlotState.READY],
            "running": by_state[SlotState.RUNNING],
            "sending": by_state[SlotState.SENDING],
            "failed_held": by_state[SlotState.FAILED],
            "passthrough": len(self.passthrough),
        }

    def request(self, stage0_id: str, req: SlotRequest) -> List[Reply]:
        err = slot_request_error(req, self.window)
        if err is not None:
            return [Reply(stage0_id, Error(reason=err, room=req.room))]
        owner = self._owner(req.room)
        if owner is not None:
            if owner != stage0_id:
                return [Reply(stage0_id, Error(reason="room_in_use", room=req.room))]
            return self._repeat(req.room)
        if not req.late_fed:
            self.passthrough[req.room] = stage0_id
            grant = SlotGrant(room=req.room, slot=-1, residual_slot=-1, c1_pages=[])
            return [Reply(stage0_id, grant)]
        need = prefix_pages(e=req.e, window=self.window, page_size=self.page_size)
        if need > self.pages.capacity:
            reason = f"prefix of {req.e} tokens exceeds the tail stage's KV budget"
            return [Reply(stage0_id, Error(reason=reason, room=req.room))]
        if len(self.queue) >= self.queue_cap:
            defer = SlotDefer(room=req.room, retry_ms=DEFER_RETRY_MS)
            return [Reply(stage0_id, defer)]
        self.tails[req.room] = TailSlot(
            room=req.room,
            stage0_id=stage0_id,
            rid=req.rid,
            s=req.s,
            e=req.e,
            state=SlotState.REQUESTED,
            t_request=self.clock(),
        )
        self.queue.append(req.room)
        return self._grant()

    def ready(self, stage0_id: str, msg: TailReady) -> List[Reply]:
        tail = self.tails.get(msg.room)
        if (
            tail is not None
            and tail.stage0_id == stage0_id
            and tail.slot == msg.slot
            and tail.state == SlotState.FAILED
        ):
            return self.finish(msg.room, status="failed", notify=False)
        if (
            tail is None
            or tail.stage0_id != stage0_id
            or tail.slot != msg.slot
            or tail.state != SlotState.GRANTED
        ):
            reason = f"TailReady for room {msg.room} slot {msg.slot} not granted"
            return [Reply(stage0_id, Error(reason=reason, room=msg.room))]
        if tail.abort_pending:
            return self.finish(msg.room, status="aborted", reason="aborted")
        if not msg.ok:
            return self.finish(msg.room, status="failed", reason="stage 0 gave up")
        tail.state = SlotState.READY
        tail.t_ready = self.clock()
        self.ready_order.append(msg.room)
        return []

    def num_ready(self) -> int:
        return len(self.ready_order)

    def oldest_ready_wait_ms(self) -> float:
        if not self.ready_order:
            return 0.0
        return (self.clock() - self.tails[self.ready_order[0]].t_ready) * 1e3

    def take(self, n: int) -> List[TailSlot]:
        """Up to n ready tails, oldest first, into RUNNING."""
        taken = []
        now = self.clock()
        while self.ready_order and len(taken) < n:
            tail = self.tails[self.ready_order.popleft()]
            tail.state = SlotState.RUNNING
            tail.t_run = now
            taken.append(tail)
        return taken

    def sending(self, room: int) -> None:
        tail = self.tails[room]
        assert tail.state == SlotState.RUNNING, tail.state
        tail.state = SlotState.SENDING
        tail.t_send = self.clock()

    def abort(self, room: int, *, stage0_id: Optional[str] = None) -> List[Reply]:
        """From stage 0 (its own rooms only) or from decode (stage0_id None)."""
        owner = self._owner(room)
        if owner is None or (stage0_id is not None and owner != stage0_id):
            return []
        tail = self.tails.get(room)
        if tail is not None and tail.state == SlotState.FAILED:
            if stage0_id is None:
                return []
            return self.finish(room, status="failed", notify=False)
        if tail is not None and tail.state in (
            SlotState.GRANTED,
            SlotState.RUNNING,
            SlotState.SENDING,
        ):
            tail.abort_pending = True
            return []
        return self.finish(room, status="aborted", reason="aborted")

    def finish(
        self, room: int, *, status: str, reason: str = "", notify: bool = True
    ) -> List[Reply]:
        """Free the room's slot and pages; the freed room can be granted again."""
        if self.passthrough.pop(room, None) is not None:
            return []
        tail = self.tails.pop(room, None)
        if tail is None:
            return []
        if tail.state == SlotState.REQUESTED:
            self.queue.remove(room)
        elif tail.state == SlotState.READY:
            self.ready_order.remove(room)
        if tail.slot >= 0:
            self.free_slots.append(tail.slot)
            self.pages.give_back(tail)
        if tail.abort_pending and status == "ok":
            status, reason = "aborted", "aborted while running"
        now = self.clock()
        done = TailDone(
            room=room,
            slot=tail.slot,
            status=status,
            reason=reason,
            queue_ms=_ms(tail.t_request, tail.t_grant or now),
            forward_ms=_ms(tail.t_run, tail.t_send) if tail.t_run else 0.0,
            send_ms=_ms(tail.t_send, now) if tail.t_send else 0.0,
        )
        return ([Reply(tail.stage0_id, done)] if notify else []) + self._grant()

    def expire(self, *, silent_stage0s: List[str]) -> List[Reply]:
        """Fail granted slots whose TailReady is overdue, and every queued or granted
        slot of a stage 0 that went silent (its TailReady cannot come). A granted slot
        is held as FAILED until stage 0 answers, or for another ready timeout. Due
        deadlines come off a heap; a stage 0's rooms are scanned once, when it falls
        silent."""
        now = self.clock()
        replies = []
        newly_silent = set(silent_stage0s) - self._silenced
        self._silenced &= set(silent_stage0s)
        if newly_silent:
            self._silenced |= newly_silent
            for tail in list(self.tails.values()):
                if tail.stage0_id in newly_silent:
                    replies += self._expire_one(tail, reason="stage 0 silent", now=now)
            for room, owner in list(self.passthrough.items()):
                if owner in newly_silent:
                    del self.passthrough[room]
        while self._deadlines and self._deadlines[0][0] <= now:
            _, room, stamp = heapq.heappop(self._deadlines)
            tail = self.tails.get(room)
            if tail is None:
                continue
            if tail.state == SlotState.FAILED and tail.t_fail == stamp:
                replies += self.finish(room, status="failed", notify=False)
            elif tail.state == SlotState.GRANTED and tail.t_grant == stamp:
                replies += self._expire_one(tail, reason="ready timeout", now=now)
        return replies

    def _expire_one(self, tail: TailSlot, *, reason: str, now: float) -> List[Reply]:
        if tail.state == SlotState.REQUESTED:
            return self.finish(tail.room, status="failed", reason=reason)
        if tail.state == SlotState.GRANTED:
            return self._fail_granted(tail, reason=reason, now=now)
        return []

    def _fail_granted(self, tail: TailSlot, *, reason: str, now: float) -> List[Reply]:
        tail.state = SlotState.FAILED
        tail.t_fail = now
        tail.fail_reason = reason
        heapq.heappush(self._deadlines, (now + self.ready_timeout_s, tail.room, now))
        done = TailDone(
            room=tail.room,
            slot=tail.slot,
            status="failed",
            reason=reason,
            queue_ms=_ms(tail.t_request, tail.t_grant),
        )
        return [Reply(tail.stage0_id, done)]

    def _owner(self, room: int) -> Optional[str]:
        tail = self.tails.get(room)
        return tail.stage0_id if tail is not None else self.passthrough.get(room)

    def _repeat(self, room: int) -> List[Reply]:
        """A resent request for a live room: repeat its grant, if it has one."""
        tail = self.tails.get(room)
        if tail is None:
            grant = SlotGrant(room=room, slot=-1, residual_slot=-1, c1_pages=[])
            return [Reply(self.passthrough[room], grant)]
        if tail.state == SlotState.GRANTED:
            return [Reply(tail.stage0_id, self._grant_msg(tail))]
        return []

    def _grant(self) -> List[Reply]:
        replies = []
        now = self.clock()
        while self.queue and self.free_slots:
            tail = self.tails[self.queue[0]]
            tail.slot = self.free_slots[0]
            pages = self.pages.take(tail)
            if pages is None:
                tail.slot = -1
                break
            self.queue.popleft()
            self.free_slots.popleft()
            tail.c1_pages = pages
            tail.state = SlotState.GRANTED
            tail.t_grant = now
            heapq.heappush(
                self._deadlines, (now + self.ready_timeout_s, tail.room, now)
            )
            replies.append(Reply(tail.stage0_id, self._grant_msg(tail)))
        return replies

    @staticmethod
    def _grant_msg(tail: TailSlot) -> SlotGrant:
        return SlotGrant(
            room=tail.room,
            slot=tail.slot,
            residual_slot=tail.slot,
            c1_pages=list(tail.c1_pages),
        )


def _ms(start: float, end: float) -> float:
    return max(0.0, (end - start) * 1e3)
