"""docs/219 §11.3 D1: per decode step, replay one of two decode graphs that are both correct (TRUNC_INLINE_SELECT).

trunc_S = the served graph: each GDN layer's post-order expiry truncation forks onto the side stream after the layer's
step, one join at the last GDN layer.  trunc_I = the same truncation (same kernel, same arguments, same post order, same
layer order) inline on the forward stream right after each layer's step: no fork / join, no side-stream concurrency.
Layer l's truncation reads and writes only layer l's U / W / count and nothing reads them before the join, so both graphs
give bitwise the same trajectory; the choice changes time only.  The host replays trunc_I when it predicts that no slot of
the batch reaches rfull this step (the inline truncation launches are then no-ops, ~1.5 us each), trunc_S otherwise (due
truncations overlap the later layers on the side stream).  A wrong prediction costs time, never numerics.

Predictor: after every decode replay of a bucket <= max_bs the layer-0 / head-0 count of the batch's slots is copied
(async, pinned) with the batch's host seq_lens; at selection the newest completed snapshot k is advanced to the current
step t when the batch holds the same requests in the same order (seq_lens_t == seq_lens_k + j elementwise, one j in
[1, J_MAX]).
Under the post order a slot with count c after step k is due at step k + j iff c + j >= rfull and (c + j - rfull) % m == 0
(m = rfull - r).  Unknown (no completed snapshot, changed / reordered batch, spec decoding, non-decode) -> trunc_S.
TP ranks predict independently from identical counts; a rank that has no completed snapshot yet picks trunc_S while
another picks trunc_I -- both graphs issue the same collectives in the same order, so this too only costs time.

Switches: SGLANG_GDN_FACTORED_TRUNC_INLINE_SELECT=1 (default off: no variants, the decode runner is unchanged);
SGLANG_GDN_FACTORED_TRUNC_INLINE_MAX_BS (default 4) = largest bucket that gets a trunc_I graph.
Not combinable with ab_measure (both own the decode runner's attention-graph variants); the in-process A/B uses
`ShadowSelect` (prediction only, no graphs) from an ab_measure hook.
"""
from __future__ import annotations

import collections
import logging
import os
from typing import Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

S_LABEL = "trunc_S"
I_LABEL = "trunc_I"
ENV = "SGLANG_GDN_FACTORED_TRUNC_INLINE_SELECT"
# the newest completed snapshot is 1-3 decode steps old; a larger gap means a different request (a one-request batch's
# seq_lens continuity check alone is vacuous: any new request 'continues' the departed one)
J_MAX = 4


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def due_at(c: int, j: int, r: int, rfull: int) -> bool:
    """Post order: a slot with count c after step k is truncated inside step k + j (j >= 1)."""
    x = c + j - rfull
    return x >= 0 and x % (rfull - r) == 0


class DuePredictor:
    """Pure-python core (unit-tested without torch).  add() takes completed snapshots in step order."""

    def __init__(self, r: int, rfull: int, keep: int = 8):
        assert 1 <= r < rfull, (r, rfull)
        self.r, self.rfull = r, rfull
        self.snaps = collections.deque(maxlen=keep)  # (seq_lens, counts), oldest first
        self.picks = {}  # seq_lens -> label picked for that step (for verification)
        self.stats = collections.Counter()

    def predict(self, seq_lens: Tuple[int, ...]) -> Tuple[str, str]:
        if not seq_lens:
            return S_LABEL, "empty"
        if not self.snaps:
            return S_LABEL, "no_snapshot"
        sl, cnt = self.snaps[-1]
        if len(sl) != len(seq_lens):
            return S_LABEL, "changed"
        j = seq_lens[0] - sl[0]
        if j < 1 or j > J_MAX or any(a - b != j for a, b in zip(seq_lens, sl)):
            return S_LABEL, "changed"
        if any(due_at(c, j, self.r, self.rfull) for c in cnt):
            return S_LABEL, "due"
        return I_LABEL, "nondue"

    def record_pick(self, seq_lens: Tuple[int, ...], label: str, reason: str):
        self.stats[f"pick_{label}"] += 1
        self.stats[f"why_{reason}"] += 1
        self.picks[seq_lens] = label
        if len(self.picks) > 256:
            for k in list(self.picks)[:128]:
                del self.picks[k]

    def add(self, seq_lens: Tuple[int, ...], counts: Sequence[int]):
        counts = tuple(int(c) for c in counts)
        if self.snaps:
            # verification: the previous completed snapshot is the step before this one (same requests) -> the truth of
            # "due at this step" is known; a non-due step appends one column to every slot (count + 1)
            sl, cnt = self.snaps[-1]
            if len(sl) == len(seq_lens) and all(a - b == 1 for a, b in zip(seq_lens, sl)):
                due = any(c != p + 1 for c, p in zip(counts, cnt))
                label = self.picks.pop(seq_lens, None)
                if label is not None:
                    self.stats[f"{'due' if due else 'nondue'}_{'I' if label == I_LABEL else 'S'}"] += 1
        self.snaps.append((tuple(seq_lens), counts))


class _SelectBase:
    """Torch glue shared by the variants object (served) and ShadowSelect (prediction only)."""

    def __init__(self, pool, gdn_backend, max_bs: int, ring: int = 64, log_every: int = 10000):
        import torch

        self.torch = torch
        self.pool, self.gdn = pool, gdn_backend
        self.max_bs = max_bs
        self.pred = DuePredictor(pool.cfg.r, pool.cfg.rfull)
        self.ring = ring
        self.pinned = torch.empty((ring, max_bs), dtype=torch.int32, pin_memory=True)
        self.dev = torch.empty(max_bs, dtype=torch.int32, device=pool.count.device)
        self.events = [torch.cuda.Event() for _ in range(ring)]
        self.pending = collections.deque()  # (ring slot, n, seq_lens)
        self.k = 0
        self.steps = 0
        self.log_every = log_every
        self.last = None  # (seq_lens, label) of the current step (select may be called twice per step)

    @staticmethod
    def _seq_lens(forward_batch) -> Optional[Tuple[int, ...]]:
        s = getattr(forward_batch, "seq_lens_cpu", None)
        return None if s is None else tuple(int(x) for x in s.tolist())

    def _eligible(self, forward_batch, graph_bs) -> bool:
        return (
            graph_bs is not None
            and graph_bs <= self.max_bs
            and forward_batch.forward_mode.is_decode()
            and getattr(forward_batch, "spec_info", None) is None
        )

    def _poll(self):
        while self.pending and self.events[self.pending[0][0]].query():
            slot, n, seq = self.pending.popleft()
            self.pred.add(seq, self.pinned[slot, :n].tolist())

    def predict_for(self, forward_batch, graph_bs) -> Tuple[str, str]:
        if not self._eligible(forward_batch, graph_bs):
            return S_LABEL, "ineligible"
        seq = self._seq_lens(forward_batch)
        if seq is None:
            return S_LABEL, "no_seq_lens"
        if self.last is not None and self.last[0] == seq:
            return self.last[1], "same_step"
        self._poll()
        label, why = self.pred.predict(seq)
        self.pred.record_pick(seq, label, why)
        self.last = (seq, label)
        return label, why

    def after_replay(self, runner, forward_batch):
        """Snapshot the layer-0 / head-0 count of this replay's slots (the static index buffer the graph read)."""
        raw, bs = getattr(runner, "raw_bs", None), getattr(runner, "bs", None)
        if raw is None or bs is None or not self._eligible(forward_batch, bs) or raw > self.max_bs:
            return
        seq = self._seq_lens(forward_batch)
        if seq is None or len(seq) != raw or len(self.pending) >= self.ring - 1:
            return
        torch = self.torch
        idx = self.gdn.state_indices_list[bs - 1][:raw]
        torch.index_select(self.pool.count[0, :, 0], 0, idx, out=self.dev[:raw])
        slot = self.k % self.ring
        self.k += 1
        self.pinned[slot, :raw].copy_(self.dev[:raw], non_blocking=True)
        self.events[slot].record()
        self.pending.append((slot, raw, seq))
        self.steps += 1
        if self.steps in (1000, 5000) or self.steps % self.log_every == 0:
            logger.info("FACTORED_TRUNC_INLINE_SELECT_STATS steps=%d %s", self.steps, dict(sorted(self.pred.stats.items())))


class FactoredTruncSelectVariants(_SelectBase):
    """AttentionGraphVariants for the decode runner: trunc_S / trunc_I graphs per bucket <= max_bs, trunc_S above."""

    capture_labels = (S_LABEL, I_LABEL)

    def labels_for_bs(self, bs: int):
        return self.capture_labels if bs <= self.max_bs else (S_LABEL,)

    def select(self, forward_batch) -> str:
        return S_LABEL  # without the graph bucket never pick trunc_I

    def select_for_bs(self, forward_batch, graph_bs) -> str:
        return self.predict_for(forward_batch, graph_bs)[0]


class ShadowSelect(_SelectBase):
    """Prediction only (no graphs): ab_measure pre / post hooks log the label D1 would have picked for every step."""


def find_gdn_backend(model_runner):
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend

    ab = getattr(model_runner, "attn_backend", None)
    for cand in (ab, getattr(ab, "linear_attn_backend", None)):
        if isinstance(cand, GDNAttnBackend):
            return cand
    return None


def _check(model_runner):
    gdn = find_gdn_backend(model_runner)
    pool = getattr(gdn, "factored", None) if gdn is not None else None
    if pool is None or gdn._factored_side_stream is None:
        raise ValueError(f"{ENV}=1 needs the factored GDN pool with the side-stream post-order expiry")
    if gdn._factored_group_plan() or gdn._factored_batch_trunc:
        raise ValueError(f"{ENV}=1 is not combinable with grouped / batched-layer expiry")
    return gdn, pool


def maybe_create(model_runner, runner, existing):
    """Called by the decode runner's __init__ after the model's own variants: the D1 variants or None (switch off)."""
    if not enabled():
        return existing
    if existing is not None:
        raise ValueError(f"{ENV}=1 is not combinable with other attention graph variants (ab_measure / DSA)")
    gdn, pool = _check(model_runner)
    max_bs = int(os.environ.get("SGLANG_GDN_FACTORED_TRUNC_INLINE_MAX_BS", "4"))
    v = FactoredTruncSelectVariants(pool, gdn, max_bs)
    logger.info(
        "FACTORED_TRUNC_INLINE_SELECT_ACTIVE max_bs=%d r=%d rfull=%d labels=%s",
        max_bs, pool.cfg.r, pool.cfg.rfull, v.capture_labels,
    )
    return v


def make_shadow(model_runner, max_bs: Optional[int] = None):
    gdn, pool = _check(model_runner)
    max_bs = max_bs or int(os.environ.get("SGLANG_GDN_FACTORED_TRUNC_INLINE_MAX_BS", "4"))
    return ShadowSelect(pool, gdn, max_bs)


def capture_label() -> Optional[str]:
    from sglang.srt.model_executor.runner_utils.capture_mode import get_capture_attention_variant

    return get_capture_attention_variant()
