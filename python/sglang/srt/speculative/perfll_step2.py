"""PERF-LL step 2 (SGLANG_PERFLL_STEP2): the next round's draft inputs staged one round early.

Mode 1 counts only. At the end of a decode pass it notes what the staging would need -- the
batch's request slots in order, whether the draft graph runs that size unpadded, and whether
every request already holds the next round's whole window (kv_allocated_len covers the host's
kv_committed_len plus three verify windows: the two rounds whose acceptance the host has not
read yet, then the window itself). The next decode pass says whether its draft inputs could
have come from there (``staged``) or why not, in one log line every LOG_EVERY passes.

Mode 2 also stages them: the post-verify graph ends in PERF-ST's E2 (prepare_draft_window),
which writes the next round's draft inputs and write window from this round's new lengths
into its slot's NextDraft buffers; a staged pass compares them with what its eager path
computes (device-side mismatch counts, read only when the line is logged).

Mode 3 uses them: E2 writes the draft fields straight into the draft graph runner's static
inputs (the window stays in the slot, whose buffers alternate per round), and a staged pass
skips prepare_for_draft's slot / position launch and the runner's copies of those fields.
"""

from __future__ import annotations

import collections
import logging
from typing import Optional

import msgspec
import torch

logger = logging.getLogger(__name__)

LOG_EVERY = 2000


class NextDraft(msgspec.Struct):
    """The next round's draft inputs for ``num_requests`` rows: int64 lengths, positions,
    M-RoPE rows ([3, num_requests] or None), the draft steps' write slots
    [num_requests * num_steps] and the round's write window [num_requests * window]."""

    num_requests: int
    seq_lens: torch.Tensor
    positions: torch.Tensor
    mrope: Optional[torch.Tensor]
    draft_out_cache_loc: torch.Tensor
    window: torch.Tensor


def make_next_draft(
    *,
    num_requests: int,
    num_steps: int,
    window: int,
    mrope: bool,
    device,
    runner_buffers=None,
) -> NextDraft:
    """Own buffers (mode 2), or with ``runner_buffers`` (mode 3) views of the draft graph
    runner's static inputs for the draft fields; the window is always the slot's own."""

    def zeros(*shape):
        return torch.zeros(shape, dtype=torch.int64, device=device)

    if runner_buffers is None:
        return NextDraft(
            num_requests=num_requests,
            seq_lens=zeros(num_requests),
            positions=zeros(num_requests),
            mrope=zeros(3, num_requests) if mrope else None,
            draft_out_cache_loc=zeros(num_requests * num_steps),
            window=zeros(num_requests * window),
        )
    rb = runner_buffers
    return NextDraft(
        num_requests=num_requests,
        seq_lens=rb.seq_lens[:num_requests],
        positions=rb.positions[:num_requests],
        mrope=rb.mrope_positions[:, :num_requests] if mrope else None,
        draft_out_cache_loc=rb.out_cache_loc[: num_requests * num_steps],
        window=zeros(num_requests * window),
    )


def stage_next_draft(
    next_draft: NextDraft,
    *,
    req_pool_indices: torch.Tensor,
    new_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    num_steps: int,
    window: int,
    seq_len_fill_value: int,
) -> None:
    from sglang.kernels.ops.speculative.cache_locs import prepare_draft_window

    prepare_draft_window(
        req_pool_indices,
        new_seq_lens,
        req_to_token,
        num_steps=num_steps,
        window=window,
        seq_len_fill_value=seq_len_fill_value,
        seq_lens_out=next_draft.seq_lens,
        positions=next_draft.positions,
        mrope=next_draft.mrope,
        draft_out_cache_loc=next_draft.draft_out_cache_loc,
        write_virtual=next_draft.window,
    )


class Step2Check:
    """Device-side counts of staged passes whose staged inputs differ from the eager ones."""

    FIELDS = ("seq_lens", "positions", "mrope", "draft_out_cache_loc", "window")

    def __init__(self, device):
        self.mismatch = torch.zeros(len(self.FIELDS), dtype=torch.int64, device=device)
        self.check_ct = 0
        self.shape_mismatch_ct = 0

    def compare(
        self,
        staged: NextDraft,
        *,
        seq_lens: torch.Tensor,
        positions: torch.Tensor,
        mrope: Optional[torch.Tensor],
        window: Optional[torch.Tensor],
        num_steps: int,
    ) -> None:
        bs = staged.num_requests
        draft_locs = None
        if window is not None and window.numel() % bs == 0:
            draft_locs = window.view(bs, -1)[:, :num_steps].reshape(-1)
        eager = (seq_lens, positions, mrope, draft_locs, window)
        mine = (staged.seq_lens, staged.positions, staged.mrope, staged.draft_out_cache_loc, staged.window)
        for i, (a, b) in enumerate(zip(mine, eager)):
            if a is None or b is None:
                continue
            if a.shape != b.shape:
                self.shape_mismatch_ct += 1
                continue
            self.mismatch[i] += (a != b.to(a.dtype)).any()
        self.check_ct += 1

    def summary(self) -> str:
        counts = self.mismatch.tolist()
        return f"compared {self.check_ct}, mismatching " + ", ".join(
            f"{name} {n}" for name, n in zip(self.FIELDS, counts)
        ) + f", shape {self.shape_mismatch_ct}"


class Step2Counter:
    def __init__(self, width: int, graph_sizes, needs_next_draft: bool = False):
        self.width = width
        self.graph_sizes = frozenset(graph_sizes or ())
        self.needs_next_draft = needs_next_draft
        self.check: Optional[Step2Check] = None
        self.noted = None
        self.counts = collections.Counter()
        self.decode_pass_ct = 0

    def note_extend(self, batch, next_draft: Optional[NextDraft] = None) -> None:
        """At a decode pass's end: what staging the next pass's draft inputs would rest on."""
        reqs = batch.reqs
        need = 3 * self.width
        self.noted = (
            tuple(r.kv.req_pool_idx for r in reqs),
            len(reqs) in self.graph_sizes,
            all(r.kv.kv_allocated_len >= r.kv.kv_committed_len + need for r in reqs),
            next_draft,
        )

    def forget(self) -> None:
        """A pass that is not a decode round (prefill, mixed): nothing is staged from it."""
        self.noted = None

    def count_draft(self, batch) -> Optional[NextDraft]:
        """At a decode pass's start: the inputs the previous pass staged for it when they hold
        (counted ``staged``), else None, counting the first reason they do not."""
        noted, self.noted = self.noted, None
        staged = None
        if noted is None:
            reason = "no_previous_decode"
        elif noted[0] != tuple(r.kv.req_pool_idx for r in batch.reqs):
            reason = "batch_changed"
        elif not noted[1]:
            reason = "padded"
        elif not noted[2]:
            reason = "window_not_allocated"
        elif self.needs_next_draft and noted[3] is None:
            reason = "not_staged_by_post_verify_graph"
        else:
            reason = "staged"
            staged = noted[3]
        self.counts[reason] += 1
        self.decode_pass_ct += 1
        if self.decode_pass_ct % LOG_EVERY == 0:
            logger.info(
                "PERF-LL step2: %d decode passes: %s%s",
                self.decode_pass_ct,
                ", ".join(f"{k} {v}" for k, v in sorted(self.counts.items())),
                "" if self.check is None else "; " + self.check.summary(),
            )
        return staged
