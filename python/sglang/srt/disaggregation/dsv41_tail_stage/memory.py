"""The tail server's KV memory, taken from the scheduler's pools once at startup.

Per slot: one request row and two window (SWA) pages, both fixed. Full pages (the c1
prefix and the window's rows) come from a host free list: a grant round takes each
granted tail's pages there, then writes all the round's request rows and window
mappings with one host-to-device copy and two scatters; freed pages go back to the list
in bulk. Nothing per room touches the allocator, the device or a device read.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import msgspec
import numpy as np

# The window spans at most two SWA pages when it is no longer than one page.
SWA_PAGES_PER_SLOT = 2


class PageStack:
    """Free page ids, last freed first out."""

    def __init__(self, pages: np.ndarray) -> None:
        self._pages = np.array(pages, dtype=np.int64)
        self._top = len(self._pages)
        self._returned: List[np.ndarray] = []

    @property
    def capacity(self) -> int:
        return len(self._pages)

    def available(self) -> int:
        return self._top + sum(len(p) for p in self._returned)

    def take(self, n: int) -> Optional[np.ndarray]:
        if n > self._top:
            self._merge()
            if n > self._top:
                return None
        self._top -= n
        return self._pages[self._top : self._top + n].copy()

    def give(self, pages: np.ndarray) -> None:
        """Deferred: merged on the next take that runs short (bulk return)."""
        self._returned.append(pages)

    def _merge(self) -> None:
        if not self._returned:
            return
        back = np.concatenate(self._returned)
        self._returned = []
        self._pages[self._top : self._top + len(back)] = back
        self._top += len(back)


def tail_pages(*, e: int, page_size: int) -> int:
    """Full pages a tail holds: its prefix and window, [0, e)."""
    return -(-e // page_size)


def window_full_slots(
    pages: np.ndarray, *, e: int, page_size: int, window: int
) -> np.ndarray:
    positions = np.arange(e - window, e, dtype=np.int64)
    return pages[positions // page_size] * page_size + positions % page_size


def window_swa(
    slot_swa_pages: np.ndarray, *, e: int, swa_page_size: int, window: int
) -> Tuple[np.ndarray, List[int]]:
    """The window's SWA slots, and the SWA pages they fill in position order (the
    DSV41_TAIL_SWA payload)."""
    positions = np.arange(e - window, e, dtype=np.int64)
    page_index = positions // swa_page_size - (e - window) // swa_page_size
    slots = slot_swa_pages[page_index] * swa_page_size + positions % swa_page_size
    return slots, slot_swa_pages[: int(page_index[-1]) + 1].tolist()


class RoundPlan(msgspec.Struct, frozen=True):
    """One grant round's device writes, as host arrays."""

    rows: np.ndarray  # [T]: each page's request row
    cols: np.ndarray  # [T]: the page's index in its row
    pages: np.ndarray  # [T]
    window_full: np.ndarray  # [tails * W]: full slots of each window
    window_swa: np.ndarray  # [tails * W]: their SWA slots


class TailHold(msgspec.Struct, frozen=True):
    """What a granted tail holds until its slot is freed."""

    row: int
    pages: np.ndarray  # full pages of [0, e)
    window_full: np.ndarray  # [W]: full slots of the window
    window_swa: np.ndarray  # [W]: their SWA slots
    swa_payload: List[int]  # SWA pages of the window, the DSV41_TAIL_SWA payload


class TailMemory:
    """The slot table's PageSource over owned memory. `write_round` applies a plan:
    req_to_token[rows, cols * page_size + j] = pages * page_size + j, and
    full_to_swa[window_full] = window_swa."""

    def __init__(
        self,
        *,
        rows: Sequence[int],
        swa_pages: np.ndarray,
        full_pages: np.ndarray,
        page_size: int,
        swa_page_size: int,
        window: int,
        write_round: Callable[[RoundPlan], None],
    ) -> None:
        assert swa_pages.shape == (len(rows), SWA_PAGES_PER_SLOT), swa_pages.shape
        assert window <= swa_page_size, (window, swa_page_size)
        self.rows = list(rows)
        self.swa_pages = swa_pages
        self.page_size = page_size
        self.swa_page_size = swa_page_size
        self.window = window
        self.stack = PageStack(full_pages)
        self.capacity = self.stack.capacity
        self._write_round = write_round
        # room -> TailHold, from grant to the slot's release.
        self.held: Dict[int, TailHold] = {}
        # Rooms granted since the last end_round, whose device writes are due.
        self._pending: Dict[int, TailHold] = {}

    def take(self, tail) -> Optional[List[int]]:
        """A grant's pages; its device writes wait for end_round (once per loop step;
        the tail's forward comes after its TailReady, in a later step)."""
        pages = self.stack.take(tail_pages(e=tail.e, page_size=self.page_size))
        if pages is None:
            return None
        window_slots, swa_payload = window_swa(
            self.swa_pages[tail.slot],
            e=tail.e,
            swa_page_size=self.swa_page_size,
            window=self.window,
        )
        hold = TailHold(
            row=self.rows[tail.slot],
            pages=pages,
            window_full=window_full_slots(
                pages, e=tail.e, page_size=self.page_size, window=self.window
            ),
            window_swa=window_slots,
            swa_payload=swa_payload,
        )
        self.held[tail.room] = hold
        self._pending[tail.room] = hold
        prefix = -(-(tail.e - self.window) // self.page_size)
        return pages[:prefix].tolist()

    def end_round(self) -> None:
        if not self._pending:
            return
        holds = list(self._pending.values())
        self._pending = {}
        self._write_round(
            RoundPlan(
                rows=np.concatenate(
                    [np.full(len(h.pages), h.row, dtype=np.int64) for h in holds]
                ),
                cols=np.concatenate(
                    [np.arange(len(h.pages), dtype=np.int64) for h in holds]
                ),
                pages=np.concatenate([h.pages for h in holds]),
                window_full=np.concatenate([h.window_full for h in holds]),
                window_swa=np.concatenate([h.window_swa for h in holds]),
            )
        )

    def give_back(self, tail) -> None:
        hold = self.held.pop(tail.room, None)
        if hold is not None:
            # A slot freed before its writes went out: they would race the next
            # grant of the same row in one round.
            self._pending.pop(tail.room, None)
            self.stack.give(hold.pages)
