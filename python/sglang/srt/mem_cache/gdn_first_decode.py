"""First decode does not wait for the publication (docs/223 §13.A; lead 02:15 / 02:18). Host-side state machine.

Today (K4.2 AGG deferred publication) every runner.forward joins the side-stream k31 publication first, so the first
decode after an extend waits for it (served: +2.2 .. 2.7 ms per request, docs/223 §12).  With
SGLANG_GDN_FIRST_DECODE_NO_WAIT=1 the first decode step of a freshly published row instead reads the exact dense
prompt-end state S0 (the publication's own input) with a dense step, leaves the row's factor slot alone (the publication
is still writing it), and keeps its step inputs x1; the next reader of the pool (any later forward or pool operation, via
pool.pside_join) joins the publication and then replays the factored step on x1 (the catch-up), after which the row is
on the factored path exactly as if it had taken step 1 there (G3).

This module only decides and tracks; the dense step and the catch-up are supplied by the decode backend:
    decide(...)        -> Decision for the coming forward (join as today, or dense-first rows)
    scope(decision)    -> context around that forward; catch-ups registered at exit
    on_join()          -> called by pool.pside_join after the publication join: runs pending catch-ups
Default off: no object is installed and every forward joins exactly as before.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SWITCH = "SGLANG_GDN_FIRST_DECODE_NO_WAIT"
MAX_BATCH = "SGLANG_GDN_FIRST_DECODE_NO_WAIT_MAX_BATCH"


def enabled() -> bool:
    return os.environ.get(SWITCH, "0") == "1"


@dataclass(frozen=True)
class Decision:
    join: bool
    reason: str
    rows: Tuple[int, ...] = ()  # batch rows that take the dense first step
    sources: Tuple[int, ...] = ()  # their rows in the publication's input states (S0)
    slots: Tuple[int, ...] = ()  # their factor slots


JOIN = Decision(True, "default")


@dataclass
class FirstDecodeNoWait:
    max_batch: int = 1
    catchup_fn: Optional[Callable[[Tuple[int, ...], object], None]] = None
    candidates: Dict[int, int] = field(default_factory=dict)  # slot -> row in the publication input states
    pending: List[Tuple[Tuple[int, ...], object]] = field(default_factory=list)  # (slots, saved step inputs)
    active: Optional[Decision] = None
    stats: Dict[str, int] = field(default_factory=lambda: dict(
        submitted=0, dense_first=0, dense_rows=0, catchups=0, join_other=0, join_batch=0, join_pool_op=0,
        join_not_decode=0, dropped=0))
    _logged: bool = False

    @classmethod
    def from_env(cls, catchup_fn=None) -> Optional["FirstDecodeNoWait"]:
        if not enabled():
            return None
        return cls(max_batch=int(os.environ.get(MAX_BATCH, "1")), catchup_fn=catchup_fn)

    # ---------------------------------------------------------------- publication side
    def on_submit(self, slots: Sequence[int], final: Sequence[bool]) -> None:
        """A deferred publication was launched for these extend rows (row i -> slots[i]); only prompt-final rows
        decode next.  A new extend also replaces S0 in the shared input buffers: earlier candidates are void."""
        self.stats["dropped"] += len(self.candidates)
        self.candidates = {int(s): i for i, (s, f) in enumerate(zip(slots, final)) if int(s) >= 0 and f}
        self.stats["submitted"] += 1

    def on_forward_stream_publication(self) -> None:
        """The extend published on the forward stream (fallback / A-label): nothing is pending for its rows."""
        self.stats["dropped"] += len(self.candidates)
        self.candidates = {}

    # ---------------------------------------------------------------- forward side
    def decide(self, *, is_decode: bool, batch_slots: Sequence[int], pool_op_slots: Sequence[int] = (),
               publication_pending: bool) -> Decision:
        if not publication_pending or not self.candidates or self.catchup_fn is None:
            return JOIN  # catchup_fn is None: the decode backend has not wired the dense first step -> inert
        if not is_decode:
            self.stats["join_not_decode"] += 1
            self._drop()
            return Decision(True, "not decode")
        rows = [r for r, s in enumerate(batch_slots) if int(s) in self.candidates]
        if not rows:
            self.stats["join_other"] += 1  # unchanged: a forward without a fresh row still joins first
            return Decision(True, "no fresh row")
        if len(batch_slots) > self.max_batch:
            self.stats["join_batch"] += 1
            self._drop()
            return Decision(True, "batch above dense-first bucket")
        touched = {int(s) for s in pool_op_slots} & set(self.candidates)
        if touched:
            self.stats["join_pool_op"] += 1
            self._drop()
            return Decision(True, "pool operation on a fresh slot")
        slots = tuple(int(batch_slots[r]) for r in rows)
        decision = Decision(False, "dense first", tuple(rows), tuple(self.candidates[s] for s in slots), slots)
        self._drop()  # one dense step per row; rows not in this batch decode later on the joined path
        return decision

    def _drop(self) -> None:
        self.stats["dropped"] += len(self.candidates)
        self.candidates = {}

    @contextmanager
    def scope(self, decision: Decision, saved=None):
        """Around the dense-first forward.  Catch-ups become due only when it has been enqueued: a join inside the
        forward (an unforeseen pool reader) must not replay a step whose inputs do not exist yet."""
        if decision.join:
            yield
            return
        if self.active is not None:
            raise RuntimeError("nested first-decode scope")
        if not self._logged:
            self._logged = True
            logger.info("FIRST_DECODE_NO_WAIT_ACTIVE rows=%d max_batch=%d: first decode reads the exact prompt-end "
                        "state, publication joined by the next pool reader, factored catch-up then", len(decision.rows),
                        self.max_batch)
        self.active = decision
        try:
            yield
        finally:
            self.active = None
        self.pending.append((decision.slots, saved))
        self.stats["dense_first"] += 1
        self.stats["dense_rows"] += len(decision.rows)

    def on_join(self) -> None:
        """pool.pside_join has joined the publication: replay every pending first step on the factored path."""
        if self.active is not None or not self.pending:
            return
        due, self.pending = self.pending, []
        for slots, saved in due:
            if self.catchup_fn is None:
                raise RuntimeError("first-decode catch-up pending without a catch-up function")
            self.catchup_fn(slots, saved)
            self.stats["catchups"] += 1

    def needs_join(self) -> bool:
        return bool(self.pending)
