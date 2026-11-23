"""NaN watch (SGLANG_ENABLE_NAN_WATCH): where a non-finite value first appears in an extend forward, with no host sync.

Per request row, on this pipeline stage: non-finite tokens of the incoming hidden state, Mamba layers whose state slot
(SSM or conv) is non-finite before the layer reads it, and non-finite tokens of the outgoing hidden state (last stage:
logits rows). Per forward: Mamba slots and new KV slots shared within the batch, or also on the allocators' free lists.
The flags are copied to pinned memory asynchronously; a later forward logs the records whose copy has landed.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# Row columns; the last row of a record holds the per-forward counters. IN / OUT read the "hidden_states" proxy
# (last stage OUT: logits rows); the K3 attention-residual bank rides beside it as "attn_res_bank".
_IN, _STATE, _OUT, _SLOT, _KV_MIN, _KV_MAX, _IN_BANK, _OUT_BANK = range(8)
# Counter row, past the counters: up to 4 values the Mamba free list holds twice, up to 2 batch slots on it.
_FREE_DUP_VALS, _SLOTS_ON_FREE = slice(6, 10), slice(10, 12)
_NUM_COLS = 12
_NONE = torch.iinfo(torch.int64).max
_COUNTERS = (
    "dup_mamba_slot",
    "mamba_slot_free",
    "mamba_free_dup",
    "dup_new_kv",
    "new_kv_page_free",
    "new_kv_in_prefix",
)
_SUMMARY_EVERY_S = 60.0


class _Forward:
    def __init__(self, *, stage, rids, prefix_lens, extend_lens, dev):
        self.stage, self.rids = stage, rids
        self.prefix_lens, self.extend_lens = prefix_lens, extend_lens
        self.dev = dev
        self.t = time.time()


_active: Optional[_Forward] = None
_pending: deque = deque()
_totals = {
    "forwards": 0,
    "rows": 0,
    "hit_rows": 0,
    "in": 0,
    "state": 0,
    "out": 0,
    "in_bank": 0,
    "out_bank": 0,
}
_totals.update({k: 0 for k in _COUNTERS})
_last_summary = [time.monotonic()]


def enabled() -> bool:
    return envs.SGLANG_ENABLE_NAN_WATCH.get()


def _bad_tokens(t: torch.Tensor, num_tokens: int) -> torch.Tensor:
    return (~torch.isfinite(t[:num_tokens]).reshape(num_tokens, -1).all(-1)).long()


def _add_proxy_flags(dev, cols, proxy, token_row, num_tokens):
    for key, col in zip(("hidden_states", "attn_res_bank"), cols):
        t = proxy.tensors.get(key)
        if t is not None:
            dev[:-1, col].index_add_(0, token_row, _bad_tokens(t, num_tokens))


def _dups(sorted_ids: torch.Tensor) -> torch.Tensor:
    return (sorted_ids[1:] == sorted_ids[:-1]).sum()


def _member(elements: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
    # Not torch.isin: for large test sets it takes a unique-based path whose output size syncs the host.
    if test.numel() == 0:
        return torch.zeros_like(elements, dtype=torch.bool)
    sorted_test = torch.sort(test).values
    idx = torch.searchsorted(sorted_test, elements).clamp_max(test.numel() - 1)
    return sorted_test[idx] == elements


def _count_in(elements: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
    return _member(elements, test).sum()


def _first_values(mask: torch.Tensor, values: torch.Tensor, k: int) -> torch.Tensor:
    """The k smallest values where mask holds, padded with _NONE; fixed size, so no sync."""
    picked = torch.where(mask, values, _NONE)
    if picked.numel() < k:
        picked = torch.cat([picked, picked.new_full((k - picked.numel(),), _NONE)])
    return torch.sort(picked).values[:k]


def _keep_for_this_stream(t: torch.Tensor) -> None:
    # A scheduler-owned tensor read on the forward stream: its block must not be reused before the read runs.
    if t.is_cuda:
        t.record_stream(torch.cuda.current_stream(t.device))


def begin(
    forward_batch: ForwardBatch, pp_proxy_tensors, *, model_runner, rids, stage
) -> None:
    """Start watching one target extend forward; logs the earlier records that are ready."""
    global _active
    poll()
    mode = forward_batch.forward_mode
    if (
        not mode.is_extend()
        or mode.is_target_verify()
        or forward_batch.extend_seq_lens_cpu is None
    ):
        return
    extend_lens = list(forward_batch.extend_seq_lens_cpu)
    prefix_lens = list(forward_batch.extend_prefix_lens_cpu or [0] * len(extend_lens))
    bs, num_tokens = len(extend_lens), sum(extend_lens)
    device = forward_batch.out_cache_loc.device
    dev = torch.zeros((bs + 1, _NUM_COLS), dtype=torch.int64, device=device)
    token_row = torch.repeat_interleave(
        torch.arange(bs, device=device),
        forward_batch.extend_seq_lens[:bs].long(),
        output_size=num_tokens,
    )
    if pp_proxy_tensors is not None:
        _add_proxy_flags(dev, (_IN, _IN_BANK), pp_proxy_tensors, token_row, num_tokens)

    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    pool = model_runner.req_to_token_pool
    req_idx = forward_batch.req_pool_indices[:bs].long()
    if isinstance(pool, HybridReqToTokenPool):
        slots = pool.get_mamba_indices(req_idx).long()
        dev[:bs, _SLOT] = slots
        dev[bs, 0] = _dups(torch.sort(slots).values)
        _keep_for_this_stream(pool.mamba_allocator.free_slots)
        free = pool.mamba_allocator.free_slots.long()
        on_free = _member(slots, free)
        dev[bs, 1] = on_free.sum()
        dev[bs, _SLOTS_ON_FREE] = _first_values(on_free, slots, 2)
        free_sorted = torch.sort(free).values
        twice = free_sorted[1:] == free_sorted[:-1]
        dev[bs, 2] = twice.sum()
        dev[bs, _FREE_DUP_VALS] = _first_values(twice, free_sorted[1:], 4)

    loc = forward_batch.out_cache_loc[:num_tokens].long()
    big = torch.iinfo(torch.int64).max
    dev[:bs, _KV_MIN] = torch.full((bs,), big, device=device).scatter_reduce(
        0, token_row, loc, "amin"
    )
    dev[:bs, _KV_MAX] = torch.full((bs,), -1, device=device).scatter_reduce(
        0, token_row, loc, "amax"
    )
    dev[bs, 3] = _dups(torch.sort(loc).values)
    allocator = model_runner.token_to_kv_pool_allocator
    free_pages = allocator.get_all_free_pages()
    if free_pages is not None:
        _keep_for_this_stream(free_pages)
        dev[bs, 4] = _count_in(loc // allocator.page_size, free_pages.long())
    rows_cpu = forward_batch.req_pool_indices_cpu
    if rows_cpu is not None and not model_runner.kv_index_translator.is_translating:
        rtt = pool.req_to_token
        prefix = [
            rtt[int(r), :p]
            for r, p in zip(rows_cpu[:bs].tolist(), prefix_lens)
            if p > 0
        ]
        if prefix:
            dev[bs, 5] = _count_in(loc, torch.cat(prefix).long())
    else:
        dev[bs, 5] = -1  # ids not comparable with req_to_token, or no host row ids
    _active = _Forward(
        stage=stage,
        rids=rids,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
        dev=dev,
    )


def check_state(
    cache_indices: torch.Tensor, ssm: torch.Tensor, conv: torch.Tensor
) -> None:
    """Count, per row, a Mamba layer whose SSM or conv state slot is non-finite before the layer reads it."""
    if _active is None:
        return
    dev = _active.dev
    bs = dev.shape[0] - 1
    idx = cache_indices[:bs].long()
    safe = idx.clamp_min(0)
    bad = ~torch.isfinite(ssm[safe]).reshape(bs, -1).all(-1)
    bad |= ~torch.isfinite(conv[safe]).reshape(bs, -1).all(-1)
    dev[:bs, _STATE] += (bad & (idx >= 0)).long()


def end(
    *,
    logits: Optional[torch.Tensor] = None,
    pp_proxy_tensors=None,
    forward_batch: ForwardBatch,
) -> None:
    """Flag the outgoing hidden state (or the last stage's logits rows) and queue the async copy."""
    global _active
    watch, _active = _active, None
    if watch is None:
        return
    dev = watch.dev
    bs = dev.shape[0] - 1
    if logits is not None:
        dev[:bs, _OUT] = (~torch.isfinite(logits[:bs]).all(-1)).long()
    elif pp_proxy_tensors is not None:
        num_tokens = sum(watch.extend_lens)
        token_row = torch.repeat_interleave(
            torch.arange(bs, device=dev.device),
            forward_batch.extend_seq_lens[:bs].long(),
            output_size=num_tokens,
        )
        _add_proxy_flags(
            dev, (_OUT, _OUT_BANK), pp_proxy_tensors, token_row, num_tokens
        )
    # Power-of-two pinned buffers, so the caching host allocator reuses blocks instead of a new cudaHostAlloc.
    rows = dev.shape[0]
    host = torch.empty(
        (1 << (rows - 1).bit_length(), dev.shape[1]),
        dtype=dev.dtype,
        pin_memory=dev.is_cuda,
    )[:rows]
    host.copy_(dev, non_blocking=True)
    event = None
    if dev.is_cuda:
        event = torch.cuda.Event()
        event.record()
    _pending.append((watch, host, event))


def poll() -> None:
    """Log the queued records whose copy has landed (event.query(): never waits)."""
    while _pending and (_pending[0][2] is None or _pending[0][2].query()):
        watch, host, _ = _pending.popleft()
        _report(watch, host.tolist())
    if time.monotonic() - _last_summary[0] >= _SUMMARY_EVERY_S:
        _last_summary[0] = time.monotonic()
        logger.info("NaN watch: %s", _totals)


def _report(watch: _Forward, rows) -> None:
    counters = dict(zip(_COUNTERS, rows[-1][: len(_COUNTERS)]))
    free_dup_vals = [v for v in rows[-1][_FREE_DUP_VALS] if v != _NONE]
    slots_on_free = [v for v in rows[-1][_SLOTS_ON_FREE] if v != _NONE]
    rows = rows[:-1]
    _totals["forwards"] += 1
    _totals["rows"] += len(rows)
    for k, v in counters.items():
        _totals[k] += int(v > 0)
    # Bank-only rows are counted, not logged: rows of blocks a stage has not written may not be initialized.
    hits = [i for i, r in enumerate(rows) if r[_IN] or r[_STATE] or r[_OUT]]
    for col, key in (
        (_IN, "in"),
        (_STATE, "state"),
        (_OUT, "out"),
        (_IN_BANK, "in_bank"),
        (_OUT_BANK, "out_bank"),
    ):
        _totals[key] += sum(1 for r in rows if r[col])
    _totals["hit_rows"] += len(hits)
    anomaly = {k: v for k, v in counters.items() if v > 0}
    if not hits and not anomaly:
        return
    batch = [
        (
            watch.rids[i] if i < len(watch.rids) else None,
            watch.prefix_lens[i],
            watch.extend_lens[i],
            r[_SLOT],
        )
        for i, r in enumerate(rows)
    ]
    for i in hits:
        r = rows[i]
        logger.warning(
            "NaN watch hit PP%d rid=%s in_bad_tokens=%d state_bad_layers=%d out_bad=%d in_bank_bad=%d out_bank_bad=%d "
            "prefix=%d extend=%d mamba_slot=%d kv_slots=[%d,%d] counters=%s batch=%s t=%.3f",
            watch.stage,
            batch[i][0],
            r[_IN],
            r[_STATE],
            r[_OUT],
            r[_IN_BANK],
            r[_OUT_BANK],
            watch.prefix_lens[i],
            watch.extend_lens[i],
            r[_SLOT],
            r[_KV_MIN],
            r[_KV_MAX],
            counters,
            batch,
            watch.t,
        )
    if anomaly and not hits:
        logger.warning(
            "NaN watch slot anomaly PP%d counters=%s free_dup_vals=%s slots_on_free=%s batch=%s t=%.3f",
            watch.stage,
            anomaly,
            free_dup_vals,
            slots_on_free,
            batch,
            watch.t,
        )
