"""SGLANG_DSV41_LATE_TAIL_ROWS: decode's per-rank feed of late-only tail rows.

A late-fed request arrives with its KV through the early layers up to e and the late
layers' input for its window rows [e - window, e) in a staging slot. Each verify step
appends a fixed budget of those rows, as pseudo-slots of the verify width, and runs
them through the late layers only. Attention-DP ranks run each layer in lockstep, so
every rank runs the same budget (or none); a fixed budget keeps their late rows equal.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Deque, List, Optional, Tuple

import msgspec

from sglang.srt.disaggregation.utils import dsv41_late_tail_rows

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class LateTailSlot(msgspec.Struct, frozen=True):
    """`width` late-only rows of one tail, at positions [position, position + width)."""

    req: "Req"
    position: int
    floor: int
    staging_slot: int
    # The first row's index within the tail's staging slot.
    staging_row: int


class LateTailStep(msgspec.Struct, frozen=True):
    slots: Tuple[LateTailSlot, ...]
    # Tails whose last late rows run in this step; each joins the running batch after it.
    finished: Tuple["Req", ...]


class _Tail(msgspec.Struct):
    req: "Req"
    start: int
    next_position: int
    end: int
    floor: int
    staging_slot: int


class LateTailFeed:
    def __init__(self, *, rows_per_step: int, width: int, window: int) -> None:
        assert rows_per_step > 0 and rows_per_step % width == 0, (
            f"SGLANG_DSV41_LATE_TAIL_ROWS={rows_per_step} must be a positive "
            f"multiple of the verify width {width}"
        )
        assert window % width == 0, f"window {window} is not whole verify slots"
        self.slots_per_step = rows_per_step // width
        self.width = width
        self.window = window
        self._tails: Deque[_Tail] = deque()

    def push(self, *, req: "Req", prompt_len: int, staging_slot: int) -> None:
        start, end, floor = dsv41_late_tail_rows(prompt_len, self.window)
        self._tails.append(
            _Tail(
                req=req,
                start=start,
                next_position=start,
                end=end,
                floor=floor,
                staging_slot=staging_slot,
            )
        )

    def drop(self, predicate) -> List[Tuple["Req", int]]:
        """Remove the tails whose request matches (aborted ones); returns each with
        its staging slot."""
        dropped = [(t.req, t.staging_slot) for t in self._tails if predicate(t.req)]
        if dropped:
            self._tails = deque(t for t in self._tails if not predicate(t.req))
        return dropped

    def pending_rows(self) -> int:
        return sum(t.end - t.next_position for t in self._tails)

    def take(self) -> Optional[LateTailStep]:
        """This step's pseudo-slots, oldest tail first; None when nothing is pending."""
        if not self._tails:
            return None
        slots: List[LateTailSlot] = []
        finished: List["Req"] = []
        while self._tails and len(slots) < self.slots_per_step:
            tail = self._tails[0]
            slots.append(
                LateTailSlot(
                    req=tail.req,
                    position=tail.next_position,
                    floor=tail.floor,
                    staging_slot=tail.staging_slot,
                    staging_row=tail.next_position - tail.start,
                )
            )
            tail.next_position += self.width
            if tail.next_position == tail.end:
                finished.append(tail.req)
                self._tails.popleft()
        return LateTailStep(slots=tuple(slots), finished=tuple(finished))
