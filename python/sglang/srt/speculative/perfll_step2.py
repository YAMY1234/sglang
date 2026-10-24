"""PERF-LL step 2 (SGLANG_PERFLL_STEP2): the next round's draft inputs staged at the draft
extend's tail.

Mode 1 counts only. At the end of a decode pass it notes what the staging would need -- the
batch's request slots in order, whether the draft graph runs that size unpadded, and whether
every request already holds the next round's whole window (kv_allocated_len covers the host's
kv_committed_len plus three verify windows: the two rounds whose acceptance the host has not
read yet, then the window itself). The next decode pass says whether its draft inputs could
have come from there (``staged``) or why not, in one log line every LOG_EVERY passes.
"""

from __future__ import annotations

import collections
import logging

logger = logging.getLogger(__name__)

LOG_EVERY = 2000


class Step2Counter:
    def __init__(self, width: int, graph_sizes):
        self.width = width
        self.graph_sizes = frozenset(graph_sizes or ())
        self.noted = None
        self.counts = collections.Counter()
        self.decode_pass_ct = 0

    def note_extend(self, batch) -> None:
        """At a decode pass's end: what staging the next pass's draft inputs would rest on."""
        reqs = batch.reqs
        need = 3 * self.width
        self.noted = (
            tuple(r.req_pool_idx for r in reqs),
            len(reqs) in self.graph_sizes,
            all(r.kv.kv_allocated_len >= r.kv.kv_committed_len + need for r in reqs),
        )

    def forget(self) -> None:
        """A pass that is not a decode round (prefill, mixed): nothing is staged from it."""
        self.noted = None

    def count_draft(self, batch) -> None:
        """At a decode pass's start: whether the previous pass could have staged its draft
        inputs, else the first reason it could not."""
        noted, self.noted = self.noted, None
        if noted is None:
            reason = "no_previous_decode"
        elif noted[0] != tuple(r.req_pool_idx for r in batch.reqs):
            reason = "batch_changed"
        elif not noted[1]:
            reason = "padded"
        elif not noted[2]:
            reason = "window_not_allocated"
        else:
            reason = "staged"
        self.counts[reason] += 1
        self.decode_pass_ct += 1
        if self.decode_pass_ct % LOG_EVERY == 0:
            logger.info(
                "PERF-LL step2: %d decode passes: %s",
                self.decode_pass_ct,
                ", ".join(f"{k} {v}" for k, v in sorted(self.counts.items())),
            )
