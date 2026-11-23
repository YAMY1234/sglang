"""NaN watch (SGLANG_ENABLE_NAN_WATCH): where a non-finite value first appears in an extend forward, with no host sync.

Per request row, on this pipeline stage: non-finite tokens of the incoming hidden state, Mamba layers whose state slot
(SSM or conv) is non-finite before the layer reads it, and non-finite tokens of the outgoing hidden state (last stage:
logits rows). Per forward: Mamba slots and new KV slots shared within the batch, or also on the allocators' free lists.
The flags are copied to pinned memory asynchronously; a later forward logs the records whose copy has landed.
"""

from __future__ import annotations

import logging
import time
import zlib
from collections import deque
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.debug_utils import nan_capture
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# Row columns; the last row of a record holds the per-forward counters. IN / OUT read the "hidden_states" proxy
# (last stage OUT: logits rows); the K3 attention-residual bank rides beside it as "attn_res_bank".
_IN, _STATE, _OUT, _SLOT, _KV_MIN, _KV_MAX, _IN_BANK, _OUT_BANK = range(8)
# Row checksums of the proxy received (after the forward stream's wait on it) and the proxy sent; the readout
# matches a stage's sent rows to the next stage's received rows of the same micro-batch.
_XFER_IN, _XFER_OUT = 8, 9
# The received proxy re-read at the stage's first layer entry, and its bank region at the forward's end (the model
# copies the bank out at entry and never writes it): 1 when the bytes differ from what arrived.
_RECV_CHANGED_ENTRY, _RECV_BANK_CHANGED_END = 10, 11
# Counter row: the CE-push receive slot's full flag (the last message seq + 1 written into it) read right after this
# forward's wait and again at the first layer's entry; a later message landing in the slot shows as a larger value.
_FULL_AT_WAIT, _FULL_AT_ENTRY = 12, 13
_XFER_KEYS = ("hidden_states", "attn_res_bank", "dspark_hidden_states")
# Counter row, past the counters: up to 4 values the Mamba free list holds twice, up to 2 batch slots on it.
_FREE_DUP_VALS, _SLOTS_ON_FREE = slice(6, 10), slice(10, 12)
_NONE = torch.iinfo(torch.int64).max
# Per watched point (see _POINT_NAMES; layer and point id): non-finite tokens, offset of the first one.
_POINT_BASE, _MAX_POINTS = 12, 96
_NUM_COLS = _POINT_BASE + 2 * _MAX_POINTS
_POINT_NAMES = (
    "attn",
    "out",
    "in",
    "kda_qkv",
    "kda_gate",
    "kda_core",
    "kda_onorm",
    "kda_beta",
    "kda_q",
    "kda_k",
    "kda_v",
    "entry_head",
    "entry_bank",
)
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
    def __init__(
        self, *, stage, rids, prefix_lens, extend_lens, dev, token_row, token_offset
    ):
        self.stage, self.rids = stage, rids
        self.prefix_lens, self.extend_lens = prefix_lens, extend_lens
        self.dev = dev
        self.token_row, self.token_offset = token_row, token_offset
        self.points = {}  # (layer, point) -> column, in first-seen order
        self.received = self.sent = False
        self.proxy = None  # the received proxy, re-read at entry and at the end
        self.recv_slot = None  # ((slot, seq), full flags view)
        self.recv_bank_sum = None
        self.capture: Optional[nan_capture.Capture] = None
        self.t = time.time()


_active: Optional[_Forward] = None
_recv_slot = None  # ((slot, seq), full flags, flag after the wait) noted by the scheduler for the next begin()
_on = None
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
    "recv_changed_entry": 0,
    "recv_bank_changed_end": 0,
}
_totals.update({k: 0 for k in _COUNTERS})
_last_summary = [time.monotonic()]


def enabled() -> bool:
    return envs.SGLANG_ENABLE_NAN_WATCH.get()


def note_recv_slot(slot_seq, full_flags) -> None:
    """The CE-push receive slot the next forward reads, after the forward stream's wait on it."""
    global _recv_slot, _on
    if _on is None:
        _on = enabled()
    if not _on or full_flags is None:
        return
    slot = slot_seq[0]
    _recv_slot = (slot_seq, full_flags, full_flags[slot : slot + 1].long())


def _bad_tokens(t: torch.Tensor, num_tokens: int) -> torch.Tensor:
    return (~torch.isfinite(t[:num_tokens]).reshape(num_tokens, -1).all(-1)).long()


def _add_proxy_flags(dev, cols, proxy, token_row, num_tokens):
    for key, col in zip(("hidden_states", "attn_res_bank"), cols):
        t = proxy.tensors.get(key)
        if t is not None:
            dev[:-1, col].index_add_(0, token_row, _bad_tokens(t, num_tokens))


def _row_checksum(
    dev_col, proxy, token_row, token_offset, num_tokens, keys=_XFER_KEYS
) -> None:
    """Per row: sum over its tokens of (offset + 1) * (sum of the token's 16-bit words); exact, in int64."""
    for key in keys:
        t = proxy.tensors.get(key)
        if t is None or t.shape[0] < num_tokens or t.element_size() not in (2, 4):
            continue
        words = t[:num_tokens].reshape(num_tokens, -1).view(torch.int16)
        per_token = words.sum(-1, dtype=torch.int64) * (token_offset + 1)
        dev_col.index_add_(0, token_row, per_token)


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
    lens = forward_batch.extend_seq_lens[:bs].long()
    token_offset = (
        torch.arange(num_tokens, device=device)
        - (torch.cumsum(lens, 0) - lens)[token_row]
    )
    dev[:bs, _POINT_BASE + 1 :: 2] = _NONE
    if pp_proxy_tensors is not None:
        _add_proxy_flags(dev, (_IN, _IN_BANK), pp_proxy_tensors, token_row, num_tokens)
        _row_checksum(
            dev[:bs, _XFER_IN], pp_proxy_tensors, token_row, token_offset, num_tokens
        )
    received = pp_proxy_tensors is not None

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
        token_row=token_row,
        token_offset=token_offset,
    )
    _active.received = received
    global _recv_slot
    if received and _recv_slot is not None:
        slot_seq, flags, at_wait = _recv_slot
        _active.recv_slot = (slot_seq, flags)
        dev[bs, _FULL_AT_WAIT] = at_wait[0]
    _recv_slot = None
    if received:
        _active.proxy = pp_proxy_tensors
        _active.recv_bank_sum = torch.zeros(bs, dtype=torch.int64, device=device)
        _row_checksum(
            _active.recv_bank_sum,
            pp_proxy_tensors,
            token_row,
            token_offset,
            num_tokens,
            keys=("attn_res_bank",),
        )
    if nan_capture.enabled():
        _active.capture = nan_capture.select(prefix_lens, extend_lens)
    if _active.capture is not None and pp_proxy_tensors is not None:
        for key in _XFER_KEYS:
            t = pp_proxy_tensors.tensors.get(key)
            if t is not None and t.shape[0] >= num_tokens:
                nan_capture.rows_of(_active.capture, f"recv.{key}", t)


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
    if _active.capture is not None:
        nan_capture.state(_active.capture, cache_indices, ssm, conv)


def check_point(layer_idx: int, point: int, hidden) -> None:
    """Per row: non-finite tokens at one point of a layer (an index into _POINT_NAMES) and the first one's offset."""
    watch = _active
    if watch is None or not isinstance(hidden, torch.Tensor):
        return
    num_tokens = watch.token_row.numel()
    if hidden.shape[0] < num_tokens:
        return  # rows sharded across ranks: not mapped to requests here
    col = watch.points.get((layer_idx, point))
    if col is None:
        if len(watch.points) == _MAX_POINTS:
            return
        col = _POINT_BASE + 2 * len(watch.points)
        watch.points[(layer_idx, point)] = col
    dev = watch.dev
    bs = dev.shape[0] - 1
    bad = _bad_tokens(hidden, num_tokens)
    dev[:bs, col].index_add_(0, watch.token_row, bad)
    first = torch.where(bad.bool(), watch.token_offset, _NONE)
    dev[:bs, col + 1].scatter_reduce_(0, watch.token_row, first, "amin")
    if watch.capture is not None:
        nan_capture.point(
            watch.capture, layer_idx, _POINT_NAMES[point], hidden, num_tokens
        )


def check_entry(layer_idx: int, head: torch.Tensor, bank) -> None:
    """At a later stage's entry, before its first layer: the stream head and the bank rows copied out of the received
    proxy (non-finite tokens), and whether the received proxy still holds the bytes that arrived."""
    watch = _active
    if watch is None or watch.proxy is None:
        return
    check_point(layer_idx, 11, head)
    if bank is not None and bank.num_valid_blocks > 0:
        check_point(layer_idx, 12, bank.block_residual[:, : bank.num_valid_blocks])
    dev = watch.dev
    bs = dev.shape[0] - 1
    now = torch.zeros(bs, dtype=torch.int64, device=dev.device)
    _row_checksum(
        now, watch.proxy, watch.token_row, watch.token_offset, watch.token_row.numel()
    )
    dev[:bs, _RECV_CHANGED_ENTRY] = (now != dev[:bs, _XFER_IN]).long()
    if watch.recv_slot is not None:
        (slot, _), flags = watch.recv_slot
        dev[bs, _FULL_AT_ENTRY] = flags[slot].long()


def capture_params(layer_idx: int, **tensors) -> None:
    """A layer's static parameters, for a replay bundle (no-op unless the capture has rows)."""
    if _active is not None and _active.capture is not None:
        nan_capture.params(_active.capture, layer_idx, tensors)


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
        _row_checksum(
            dev[:bs, _XFER_OUT],
            pp_proxy_tensors,
            watch.token_row,
            watch.token_offset,
            num_tokens,
        )
        watch.sent = True
    if watch.recv_bank_sum is not None:
        now = torch.zeros_like(watch.recv_bank_sum)
        _row_checksum(
            now,
            watch.proxy,
            watch.token_row,
            watch.token_offset,
            watch.token_row.numel(),
            keys=("attn_res_bank",),
        )
        dev[:bs, _RECV_BANK_CHANGED_END] = (now != watch.recv_bank_sum).long()
    if watch.capture is not None:
        nan_capture.finish(watch.capture)
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
    counter_row = rows[-1]
    counters = dict(zip(_COUNTERS, counter_row[: len(_COUNTERS)]))
    free_dup_vals = [v for v in rows[-1][_FREE_DUP_VALS] if v != _NONE]
    slots_on_free = [v for v in rows[-1][_SLOTS_ON_FREE] if v != _NONE]
    rows = rows[:-1]
    if watch.received or watch.sent:
        # One line per forward: the readout pairs stage s's sent rows with stage s+1's received rows by key.
        key = zlib.crc32(
            repr((watch.rids, watch.prefix_lens, watch.extend_lens)).encode()
        )
        logger.info(
            "NaN watch xfer PP%d key=%08x in=%s out=%s",
            watch.stage,
            key,
            [r[_XFER_IN] for r in rows] if watch.received else None,
            [r[_XFER_OUT] for r in rows] if watch.sent else None,
        )
    _totals["forwards"] += 1
    _totals["rows"] += len(rows)
    for k, v in counters.items():
        _totals[k] += int(v > 0)
    # Bank-only rows are counted, not logged: rows of blocks a stage has not written may not be initialized.
    point_cols = list(watch.points.items())
    hits = [
        i
        for i, r in enumerate(rows)
        if r[_IN] or r[_STATE] or r[_OUT] or any(r[c] for _, c in point_cols)
    ]
    for col, key in (
        (_IN, "in"),
        (_STATE, "state"),
        (_OUT, "out"),
        (_IN_BANK, "in_bank"),
        (_OUT_BANK, "out_bank"),
        (_RECV_CHANGED_ENTRY, "recv_changed_entry"),
        (_RECV_BANK_CHANGED_END, "recv_bank_changed_end"),
    ):
        _totals[key] += sum(1 for r in rows if r[col])
    _totals["hit_rows"] += len(hits)
    anomaly = {k: v for k, v in counters.items() if v > 0}
    changed = [
        (i, r[_RECV_CHANGED_ENTRY], r[_RECV_BANK_CHANGED_END])
        for i, r in enumerate(rows)
        if r[_RECV_CHANGED_ENTRY] or r[_RECV_BANK_CHANGED_END]
    ]
    if not hits and not anomaly and not changed:
        return
    # (slot, seq, full flag after the wait, at the first layer's entry); the flag holds the last seq written + 1.
    ring = (
        (*watch.recv_slot[0], counter_row[_FULL_AT_WAIT], counter_row[_FULL_AT_ENTRY])
        if watch.recv_slot is not None
        else None
    )
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
        # (layer, point, non-finite tokens, offset of the first) for each watched point with any.
        points = [
            (layer, _POINT_NAMES[point], r[c], r[c + 1])
            for (layer, point), c in point_cols
            if r[c]
        ]
        logger.warning(
            "NaN watch hit PP%d rid=%s in_bad_tokens=%d state_bad_layers=%d out_bad=%d in_bank_bad=%d out_bank_bad=%d "
            "prefix=%d extend=%d mamba_slot=%d kv_slots=[%d,%d] counters=%s batch=%s t=%.3f points=%s "
            "recv_changed=%d/%d ring=%s",
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
            points,
            r[_RECV_CHANGED_ENTRY],
            r[_RECV_BANK_CHANGED_END],
            ring,
        )
        if watch.capture is not None and not r[_IN] and not r[_STATE]:
            meta = {
                "stage": watch.stage,
                "rid": batch[i][0] or "",
                "prefix": watch.prefix_lens[i],
                "extend": watch.extend_lens[i],
                "mamba_slot": r[_SLOT],
                "kv_slots": (r[_KV_MIN], r[_KV_MAX]),
                "points": points,
                "counters": counters,
                "capture_layer": watch.capture.layer,
                "batch": batch,
                "t": watch.t,
            }
            path = nan_capture.dump(watch.capture, i, meta)
            if path is not None:
                logger.warning("NaN watch capture saved %s", path)
    if changed:
        # (row, changed by first-layer entry, bank changed by the forward's end): the received bytes moved under it.
        logger.warning(
            "NaN watch recv changed PP%d rows=%s ring=%s batch=%s t=%.3f",
            watch.stage,
            changed,
            ring,
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
