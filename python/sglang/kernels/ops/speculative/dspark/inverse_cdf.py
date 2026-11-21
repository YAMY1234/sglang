"""Inverse-CDF draws for DSpark sampling (SGLANG_OPT_DSPARK_INVERSE_CDF_SAMPLING).

The block-wise rejection's resample and the Markov draft step draw one token per
row with a Gumbel-max over the whole vocabulary: one random number and two logs per
entry. Here each row draws from one 64-bit uniform instead:

1. a block pass writes each 4,096-column block's mass as (max m_b, FP64 sum s_b of
   exp(x - m_b)), or, for the resample's residual max(p - q, 0), (0, FP64 sum);
2. a pick pass weighs the blocks by s_b * exp(m_b - max), finds the block holding
   u * total, recomputes that block's terms exactly as pass 1 did, and scans them
   in FP64 for the token.

Same distributions as the Gumbel-max kernels, different random draws. A token's
probability carries only the FP32 rounding of its own exp() term (a few 1e-7
relative); the FP64 sums and scan resolve tokens far below 1e-7 of the mass.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.dspark.blockwise_rejection import (
    _STATS_BLOCK,
    RejectionNoise,
    _rejection_kernel,
    _stats_kernel,
    _step_seed,
)

logger = logging.getLogger(__name__)

_BLOCK = 4096
# Seconds between the engagement lines of InverseCdfCounters.
_LOG_PERIOD_S = 120.0
_NO_INDEX = tl.constexpr(2147483647)
_PHILOX_ROUNDS = 10
_INV32 = tl.constexpr(2.3283064365386963e-10)


@triton.jit
def _uniform64(w0, w1):
    # [0, 1) from two 32-bit words; FP64 holds all 64 bits' worth that matter.
    return (w0.to(tl.float64) + w1.to(tl.float64) * _INV32) * _INV32


@triton.jit
def _pick_block(
    m_ptr, s_ptr, base, u, NUM_BLOCKS: tl.constexpr, PADDED: tl.constexpr
):
    # (block holding u * total mass, the offset into it in its own m_b units,
    # whether any mass exists). Empty rows return block 0.
    blocks = tl.arange(0, PADDED)
    valid = blocks < NUM_BLOCKS
    m = tl.load(m_ptr + base + blocks, mask=valid, other=float("-inf"))
    s = tl.load(s_ptr + base + blocks, mask=valid, other=0.0)
    top = tl.max(m, axis=0)
    live = valid & (s > 0.0)
    scale = tl.exp(tl.where(live, m - top, 0.0).to(tl.float64))
    w = tl.where(live, s * scale, 0.0)
    total = tl.sum(w, axis=0)
    target = u * total
    cum = tl.cumsum(w, axis=0)
    last = tl.max(tl.where(live, blocks, -1), axis=0)
    first = tl.min(tl.where(cum > target, blocks, PADDED), axis=0)
    pick = tl.maximum(tl.minimum(first, last), 0)
    before = tl.sum(tl.where(blocks < pick, w, 0.0), axis=0)
    pick_scale = tl.sum(tl.where(blocks == pick, scale, 0.0), axis=0)
    # Back to the picked block's own units, where pass 2's terms live.
    return pick, (target - before) / pick_scale, total > 0.0


@triton.jit
def _scan_pick(terms, offs, target):
    # The first column whose FP64 running sum passes target, else the last with mass.
    cum = tl.cumsum(terms.to(tl.float64), axis=0)
    first = tl.min(tl.where(cum > target, offs, _NO_INDEX), axis=0)
    last = tl.max(tl.where(terms > 0.0, offs, -1), axis=0)
    return tl.where(first == _NO_INDEX, last, first)


class InverseCdfCounters:
    """Sampled accepts that drew by inverse CDF, and those that took another accept
    (top-k / top-p, the vocab-shard accept, the classic kernel), with a periodic line."""

    inverse_batches = 0
    inverse_requests = 0
    fallback_batches = 0
    fallback_requests = 0
    _next_log = 0.0

    @classmethod
    def note(cls, *, inverse: bool, requests: int) -> None:
        if inverse:
            cls.inverse_batches += 1
            cls.inverse_requests += requests
        else:
            cls.fallback_batches += 1
            cls.fallback_requests += requests
        now = time.monotonic()
        if now >= cls._next_log:
            cls._next_log = now + _LOG_PERIOD_S
            logger.info(
                "Inverse-CDF sampling: %d sampled accept batches (%d requests) drew by "
                "inverse CDF; %d (%d requests) fell back to another accept",
                cls.inverse_batches,
                cls.inverse_requests,
                cls.fallback_batches,
                cls.fallback_requests,
            )


# ---------------------------------------------------------------- resample
@triton.jit
def _residual_terms(
    target_logits_ptr,
    target_stride,
    draft_logits_ptr,
    draft_stride_0,
    draft_stride_1,
    candidates_ptr,
    candidates_stride,
    rej_lse_ptr,
    bs,
    req,
    r,
    t_temp,
    d_temp,
    offs,
    target_vocab,
    draft_vocab,
    STRIDE: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    # max(p - q, 0) at the rejected row r < STRIDE - 1, FP32, zero outside the vocab.
    mask = offs < target_vocab
    t = (
        tl.load(
            target_logits_ptr + (req * STRIDE + r) * target_stride + offs,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        / t_temp
    )
    p = tl.exp(t - tl.load(rej_lse_ptr + req))
    if HAS_DRAFT_LOGITS:
        d = (
            tl.load(
                draft_logits_ptr + req * draft_stride_0 + r * draft_stride_1 + offs,
                mask=offs < draft_vocab,
                other=float("-inf"),
            ).to(tl.float32)
            / d_temp
        )
        terms = tl.maximum(p - tl.exp(d - tl.load(rej_lse_ptr + bs + req)), 0.0)
    else:
        rejected = tl.load(candidates_ptr + req * candidates_stride + r + 1)
        terms = tl.where(offs != rejected, p, 0.0)
    return tl.where(mask, terms, 0.0)


@triton.jit
def _bonus_row(target_logits_ptr, target_stride, req, t_temp, offs, target_vocab,
               STRIDE: tl.constexpr):
    return (
        tl.load(
            target_logits_ptr + (req * STRIDE + STRIDE - 1) * target_stride + offs,
            mask=offs < target_vocab,
            other=float("-inf"),
        ).to(tl.float32)
        / t_temp
    )


@triton.jit
def _resample_mass_kernel(
    m_ptr,
    s_ptr,
    counter_snap_ptr,
    counter_ptr,
    raw_len_ptr,
    rej_lse_ptr,
    candidates_ptr,
    candidates_stride,
    target_logits_ptr,
    target_stride,
    draft_logits_ptr,
    draft_stride_0,
    draft_stride_1,
    target_temp_ptr,
    draft_temp_ptr,
    target_vocab,
    draft_vocab,
    bs,
    STRIDE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    if (req == 0) & (block_idx == 0):
        # The pick kernel draws from this counter while its program 0 advances it.
        tl.store(counter_snap_ptr, tl.load(counter_ptr))
    offs = block_idx * BLOCK + tl.arange(0, BLOCK)
    out = req * NUM_BLOCKS + block_idx
    r = tl.load(raw_len_ptr + req).to(tl.int64)
    t_temp = tl.load(target_temp_ptr + req).to(tl.float32)
    if r == STRIDE - 1:
        # The bonus row: p itself, as block max and sum-exp.
        t = _bonus_row(target_logits_ptr, target_stride, req, t_temp, offs,
                       target_vocab, STRIDE)
        m = tl.max(t, axis=0)
        s = tl.sum(tl.where(t > float("-inf"), tl.exp(t - m), 0.0).to(tl.float64))
    else:
        d_temp = 1.0
        if HAS_DRAFT_LOGITS:
            d_temp = tl.load(draft_temp_ptr + req).to(tl.float32)
        terms = _residual_terms(
            target_logits_ptr, target_stride, draft_logits_ptr, draft_stride_0,
            draft_stride_1, candidates_ptr, candidates_stride, rej_lse_ptr, bs, req,
            r, t_temp, d_temp, offs, target_vocab, draft_vocab, STRIDE,
            HAS_DRAFT_LOGITS,
        )  # fmt: skip
        m = tl.zeros([], tl.float32)
        s = tl.sum(terms.to(tl.float64))
    tl.store(m_ptr + out, m)
    tl.store(s_ptr + out, s)


@triton.jit
def _resample_pick_kernel(
    bonus_ptr,
    correct_len_ptr,
    raw_len_ptr,
    m_ptr,
    s_ptr,
    rej_lse_ptr,
    candidates_ptr,
    candidates_stride,
    target_logits_ptr,
    target_stride,
    draft_logits_ptr,
    draft_stride_0,
    draft_stride_1,
    target_temp_ptr,
    draft_temp_ptr,
    target_vocab,
    draft_vocab,
    bs,
    key,
    counter_snap_ptr,
    counter_ptr,
    STRIDE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    PADDED_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
):
    # blockwise_rejection._insert_kernel with the inverse-CDF resample.
    req = tl.program_id(0).to(tl.int64)
    capped = tl.load(correct_len_ptr + req)
    raw = tl.load(raw_len_ptr + req)
    if capped < raw:
        bonus = tl.load(candidates_ptr + req * candidates_stride + capped + 1).to(
            tl.int64
        )
    else:
        # The coins draw Philox counters below bs * STRIDE; the resample takes the next bs.
        lane = tl.zeros([1], tl.uint32)
        seed = _step_seed(key, counter_snap_ptr)
        w0, w1, _, _ = tl.randint4x(seed, (lane + bs * STRIDE + req).to(tl.int32),
                                    PHILOX_ROUNDS)
        u = tl.sum(_uniform64(w0, w1), axis=0)
        pick, target, has_mass = _pick_block(
            m_ptr, s_ptr, req * NUM_BLOCKS, u, NUM_BLOCKS, PADDED_BLOCKS
        )
        offs = pick * BLOCK + tl.arange(0, BLOCK)
        r = raw.to(tl.int64)
        t_temp = tl.load(target_temp_ptr + req).to(tl.float32)
        if r == STRIDE - 1:
            t = _bonus_row(target_logits_ptr, target_stride, req, t_temp, offs,
                           target_vocab, STRIDE)
            m = tl.load(m_ptr + req * NUM_BLOCKS + pick)
            terms = tl.where(t > float("-inf"), tl.exp(t - m), 0.0)
        else:
            d_temp = 1.0
            if HAS_DRAFT_LOGITS:
                d_temp = tl.load(draft_temp_ptr + req).to(tl.float32)
            terms = _residual_terms(
                target_logits_ptr, target_stride, draft_logits_ptr, draft_stride_0,
                draft_stride_1, candidates_ptr, candidates_stride, rej_lse_ptr, bs,
                req, r, t_temp, d_temp, offs, target_vocab, draft_vocab, STRIDE,
                HAS_DRAFT_LOGITS,
            )  # fmt: skip
        # A residual with no mass in FP32 takes token 0, as the Gumbel-max over an
        # all -inf row does.
        bonus = tl.where(has_mass, _scan_pick(terms, offs, target), 0).to(tl.int64)
    tl.store(bonus_ptr + req, bonus)
    if req == 0:
        # The other kernels of this call have read the counter (stream order).
        tl.store(counter_ptr, tl.load(counter_ptr) + 1)


def blockwise_rejection_inverse_cdf(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_logits: Optional[torch.Tensor],
    target_temperatures: torch.Tensor,
    draft_temperatures: torch.Tensor,
    noise: RejectionNoise,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``blockwise_rejection_triton`` with the resample drawn by inverse CDF: the
    same stats, coins, lengths and resample distribution."""
    bs, stride = candidates.shape
    target_vocab = target_logits.shape[-1]
    assert target_logits.stride(-1) == 1 and target_logits.shape[0] == bs * stride
    has_draft = draft_logits is not None
    draft_vocab = draft_logits.shape[-1] if has_draft else target_vocab
    if has_draft:
        assert draft_logits.stride(-1) == 1 and draft_logits.shape[1] >= stride - 1
    device = candidates.device
    target_temperatures = target_temperatures.reshape(-1).contiguous()
    draft_temperatures = draft_temperatures.reshape(-1).contiguous()
    num_blocks = triton.cdiv(max(target_vocab, draft_vocab), _STATS_BLOCK)
    stats = torch.empty(
        (4, bs * stride, num_blocks), dtype=torch.float32, device=device
    )
    lens = torch.empty((3, bs), dtype=torch.int32, device=device)
    rej_lse = torch.empty((2, bs), dtype=torch.float32, device=device)
    pick_blocks = triton.cdiv(target_vocab, _BLOCK)
    mass_m = torch.empty((bs, pick_blocks), dtype=torch.float32, device=device)
    mass_s = torch.empty((bs, pick_blocks), dtype=torch.float64, device=device)
    bonus = torch.empty((bs,), dtype=torch.int64, device=device)
    correct_len, raw_len, trim = lens[0], lens[1], lens[2]
    t_max, t_sumexp, d_max, d_sumexp = stats[0], stats[1], stats[2], stats[3]
    d_ptr = draft_logits if has_draft else target_logits
    d_s0 = draft_logits.stride(0) if has_draft else 0
    d_s1 = draft_logits.stride(1) if has_draft else 0
    _stats_kernel[(bs * stride, num_blocks)](
        t_max, t_sumexp, d_max, d_sumexp,
        target_logits, target_logits.stride(0),
        d_ptr, d_s0, d_s1,
        target_temperatures, draft_temperatures,
        target_vocab, draft_vocab,
        STRIDE=stride, NUM_BLOCKS=num_blocks, BLOCK=_STATS_BLOCK,
        HAS_DRAFT_LOGITS=has_draft,
    )  # fmt: skip
    has_cutoff = cutoff_verify_lens is not None
    verify_lens = (
        cutoff_verify_lens.to(device=device).contiguous() if has_cutoff else correct_len
    )
    _rejection_kernel[(bs,)](
        correct_len, raw_len, trim, rej_lse,
        candidates, candidates.stride(0), verify_lens,
        t_max, t_sumexp, d_max, d_sumexp,
        target_logits, target_logits.stride(0),
        d_ptr, d_s0, d_s1,
        target_temperatures, draft_temperatures,
        target_vocab, draft_vocab, bs,
        noise.key, noise.counter,
        STRIDE=stride, NUM_BLOCKS=num_blocks,
        PADDED_BLOCKS=triton.next_power_of_2(num_blocks),
        HAS_DRAFT_LOGITS=has_draft, HAS_CUTOFF=has_cutoff,
        num_warps=1,
    )  # fmt: skip
    resample_inverse_cdf(
        bonus=bonus, correct_len=correct_len, raw_len=raw_len, rej_lse=rej_lse,
        candidates=candidates, target_logits=target_logits, draft_logits=draft_logits,
        target_temperatures=target_temperatures, draft_temperatures=draft_temperatures,
        noise=noise, mass_m=mass_m, mass_s=mass_s,
    )  # fmt: skip
    return correct_len, bonus, trim


def resample_inverse_cdf(
    *,
    bonus,
    correct_len,
    raw_len,
    rej_lse,
    candidates,
    target_logits,
    draft_logits,
    target_temperatures,
    draft_temperatures,
    noise,
    mass_m,
    mass_s,
):
    """The resample and insert of blockwise_rejection_triton, after its rejection
    kernel; advances the noise counter as the insert does."""
    bs, stride = candidates.shape
    target_vocab = target_logits.shape[-1]
    has_draft = draft_logits is not None
    draft_vocab = draft_logits.shape[-1] if has_draft else target_vocab
    d_ptr = draft_logits if has_draft else target_logits
    d_s0 = draft_logits.stride(0) if has_draft else 0
    d_s1 = draft_logits.stride(1) if has_draft else 0
    pick_blocks = mass_m.shape[1]
    counter_snap = torch.empty_like(noise.counter)
    _resample_mass_kernel[(bs, pick_blocks)](
        mass_m, mass_s, counter_snap, noise.counter, raw_len, rej_lse,
        candidates, candidates.stride(0),
        target_logits, target_logits.stride(0),
        d_ptr, d_s0, d_s1,
        target_temperatures, draft_temperatures,
        target_vocab, draft_vocab, bs,
        STRIDE=stride, NUM_BLOCKS=pick_blocks, BLOCK=_BLOCK,
        HAS_DRAFT_LOGITS=has_draft,
    )  # fmt: skip
    _resample_pick_kernel[(bs,)](
        bonus, correct_len, raw_len, mass_m, mass_s, rej_lse,
        candidates, candidates.stride(0),
        target_logits, target_logits.stride(0),
        d_ptr, d_s0, d_s1,
        target_temperatures, draft_temperatures,
        target_vocab, draft_vocab, bs,
        noise.key, counter_snap, noise.counter,
        STRIDE=stride, NUM_BLOCKS=pick_blocks,
        PADDED_BLOCKS=triton.next_power_of_2(pick_blocks), BLOCK=_BLOCK,
        HAS_DRAFT_LOGITS=has_draft, PHILOX_ROUNDS=_PHILOX_ROUNDS,
    )  # fmt: skip


# ---------------------------------------------------------------- Markov step
@triton.jit
def _markov_mass_kernel(
    B,
    X,
    T,
    L,
    M,
    S,
    I,
    N,
    BS,
    XS,
    LS,
    WIDTH,
    PADDED,
    PARTS,
    BLOCK: tl.constexpr,
    ROUND: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # markov_sample._markov_sample_partial_kernel's step logits and store, then per
    # block: the max of value / t and FP64 sum of exp(value / t - max) (sampled
    # rows), the lowest index of the max (t = 0 rows) and the lowest NaN index.
    row, part = tl.program_id(0), tl.program_id(1)
    i = part * BLOCK + tl.arange(0, BLOCK)
    valid = i < WIDTH
    in_shard = i < PADDED
    value = tl.load(X + row * XS + i, in_shard, 0).to(tl.float32)
    if HAS_BIAS:
        value += tl.load(B + row * BS + i, valid, 0).to(tl.float32)
    if ROUND == 1:
        value = value.to(tl.bfloat16).to(tl.float32)
    if ROUND == 2:
        value = value.to(tl.float16).to(tl.float32)
    tl.store(L + row * LS + i, value.to(L.dtype.element_ty), in_shard)
    out = row * PARTS + part
    nan = valid & (value != value)
    tl.store(N + out, tl.min(tl.where(nan, i, _NO_INDEX), axis=0))
    temperature = tl.load(T + row)
    if temperature != 0.0:
        x = tl.where(valid & ~nan, value / temperature, float("-inf"))
        m = tl.max(x, axis=0)
        s = tl.sum(tl.where(x > float("-inf"), tl.exp(x - m), 0.0).to(tl.float64))
        tl.store(I + out, tl.full([], _NO_INDEX, tl.int32))
    else:
        x = tl.where(valid & ~nan, value, float("-inf"))
        m = tl.max(x, axis=0)
        s = tl.zeros([], tl.float64)
        tl.store(I + out, tl.min(tl.where(valid & ~nan & (x == m), i, _NO_INDEX), axis=0))
    tl.store(M + out, m)
    tl.store(S + out, s)


@triton.jit
def _markov_pick_kernel(
    L,
    T,
    OFFSET,
    M,
    S,
    I,
    N,
    OUT,
    LS,
    WIDTH,
    KEY,
    STEP,
    PARTS: tl.constexpr,
    PADDED_PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
):
    # markov_sample's finish: NaN rows take the lowest NaN index, t = 0 rows the
    # lowest index of the max, sampled rows an inverse-CDF draw.
    row = tl.program_id(0)
    blocks = tl.arange(0, PADDED_PARTS)
    valid = blocks < PARTS
    base = row * PARTS
    nan_idx = tl.min(tl.load(N + base + blocks, mask=valid, other=_NO_INDEX), axis=0)
    temperature = tl.load(T + row)
    if nan_idx != _NO_INDEX:
        token = nan_idx
    elif temperature == 0.0:
        m = tl.load(M + base + blocks, mask=valid, other=float("-inf"))
        idx = tl.load(I + base + blocks, mask=valid, other=_NO_INDEX)
        top = tl.max(m, axis=0)
        token = tl.min(tl.where(m == top, idx, _NO_INDEX), axis=0)
    else:
        # Counters off the Gumbel draw's: c0 = 2**32 - 1 is no column quad.
        lane = tl.zeros([1], tl.int64)
        w0, w1, _, _ = tl.philox(
            KEY,
            (lane + 0xFFFFFFFF).to(tl.uint32),
            (lane + row).to(tl.uint32),
            (lane + STEP).to(tl.uint32),
            ((lane + tl.load(OFFSET)) & 0xFFFFFFFF).to(tl.uint32),
            PHILOX_ROUNDS,
        )
        u = tl.sum(_uniform64(w0, w1), axis=0)
        pick, target, has_mass = _pick_block(M, S, base, u, PARTS, PADDED_PARTS)
        i = pick * BLOCK + tl.arange(0, BLOCK)
        valid_i = i < WIDTH
        value = tl.load(L + row * LS + i, valid_i, 0).to(tl.float32)
        x = tl.where(valid_i, value / temperature, float("-inf"))
        block_max = tl.load(M + base + pick)
        terms = tl.where(x > float("-inf"), tl.exp(x - block_max), 0.0)
        token = tl.where(has_mass, _scan_pick(terms, i, target), 0)
    tl.store(OUT + row, token.to(tl.int64))


def markov_step_inverse_cdf(
    *,
    base_logits: torch.Tensor,
    bias: torch.Tensor,
    temperatures: torch.Tensor,
    key: int,
    offset: torch.Tensor,
    step: int,
    round_code: int,
    logits_out: torch.Tensor,
) -> torch.Tensor:
    """MarkovSampleStep.triton for a local (unsharded) step that keeps its logits:
    the same step logits written to ``logits_out`` and the same token law per row."""
    rows, padded = base_logits.shape
    width = bias.shape[1]
    parts = triton.cdiv(padded, _BLOCK)
    device = base_logits.device
    mass_m = torch.empty((rows, parts), dtype=torch.float32, device=device)
    mass_s = torch.empty((rows, parts), dtype=torch.float64, device=device)
    arg = torch.empty((rows, parts), dtype=torch.int32, device=device)
    nan = torch.empty((rows, parts), dtype=torch.int32, device=device)
    out = torch.empty((rows,), dtype=torch.int64, device=device)
    _markov_mass_kernel[(rows, parts)](
        bias, base_logits, temperatures, logits_out, mass_m, mass_s, arg, nan,
        bias.stride(0), base_logits.stride(0), logits_out.stride(0),
        width, padded, parts,
        BLOCK=_BLOCK, ROUND=round_code, HAS_BIAS=True,
    )  # fmt: skip
    _markov_pick_kernel[(rows,)](
        logits_out, temperatures, offset, mass_m, mass_s, arg, nan, out,
        logits_out.stride(0), width, key, step,
        PARTS=parts, PADDED_PARTS=triton.next_power_of_2(parts), BLOCK=_BLOCK,
        PHILOX_ROUNDS=_PHILOX_ROUNDS,
    )  # fmt: skip
    return out
