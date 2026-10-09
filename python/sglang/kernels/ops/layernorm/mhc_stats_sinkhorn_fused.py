"""mHC triplet in one launch: the bf16x3 split-K mix GEMM, its slice reduce and the
Sinkhorn normalization of ``hc_mix_stats_sinkhorn_bf16x3``.

The split-K CTAs of a row block count themselves in on a semaphore; the last one
reads every slice's partials back and runs the reduce + Sinkhorn of
``_hc_mix_reduce_sinkhorn_kernel`` for the block's rows, with the same slice order
and the same float operations, so the triplet matches the two-launch path.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.layernorm.mhc import (
    _HC_MIX_BF16X3_BLOCK_M,
    _HC_MIX_BLOCK_K,
    _HC_MIX_COMPENSATED_SLICES,
)

# 65536 rows (the bf16x3 path's upper bound) / 128-row blocks.
_MAX_ROW_BLOCKS = 512
_locks: dict = {}


def _row_block_locks(device: torch.device) -> torch.Tensor:
    # Zero between launches: the last CTA of each row block resets its counter.
    # One buffer per (device, stream); launches on one stream never overlap.
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    locks = _locks.get(key)
    if locks is None:
        locks = _locks[key] = torch.zeros(
            _MAX_ROW_BLOCKS, dtype=torch.int32, device=device
        )
    return locks


@triton.jit
def _hc_mix_stats_sinkhorn_bf16x3_kernel(
    X,
    W_HI,
    W_MID,
    W_LO,
    MIX,
    SQ,
    LOCKS,
    SCALE,
    BASE,
    PRE,
    POST,
    COMB,
    M,
    INV_K,
    RMS_EPS,
    K: tl.constexpr,
    K_PER_SLICE: tl.constexpr,
    MIX_COLS: tl.constexpr,
    MIX_PAD: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HC: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
):
    # Stage 1: _hc_mix_stats_bf16x3_kernel, unchanged.
    block = tl.program_id(0)
    rows = block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, MIX_PAD)
    start = tl.program_id(1) * K_PER_SLICE
    ks = start + tl.arange(0, BLOCK_K)
    hi = tl.zeros((BLOCK_M, MIX_PAD), tl.float32)
    mid = tl.zeros((BLOCK_M, MIX_PAD), tl.float32)
    lo = tl.zeros((BLOCK_M, MIX_PAD), tl.float32)
    sq = tl.zeros((BLOCK_M,), tl.float32)
    for kb in range(K_PER_SLICE // BLOCK_K):
        k = ks + kb * BLOCK_K
        x = tl.load(
            X + rows[:, None].to(tl.int64) * K + k[None, :],
            rows[:, None] < M,
            0,
        )
        offsets = cols[None, :] * K + k[:, None]
        w_hi = tl.load(W_HI + offsets, cols[None, :] < MIX_COLS, 0)
        w_mid = tl.load(W_MID + offsets, cols[None, :] < MIX_COLS, 0)
        w_lo = tl.load(W_LO + offsets, cols[None, :] < MIX_COLS, 0)
        hi = tl.dot(x, w_hi, hi)
        mid = tl.dot(x, w_mid, mid)
        lo = tl.dot(x, w_lo, lo)
        xf = x.to(tl.float32)
        sq += tl.sum(xf * xf, 1)
    offsets = (tl.program_id(1) * M + rows[:, None]) * MIX_COLS + cols[None, :]
    tl.store(
        MIX + offsets, (hi + mid) + lo, (rows[:, None] < M) & (cols[None, :] < MIX_COLS)
    )
    tl.store(SQ + tl.program_id(1) * M + rows, sq, rows < M)

    # Stage 2 in the row block's last slice CTA: _hc_mix_reduce_sinkhorn_kernel.
    tl.debug_barrier()
    arrived = tl.atomic_add(LOCKS + block, 1, sem="acq_rel")
    if arrived == NUM_SLICES - 1:
        tl.atomic_xchg(LOCKS + block, 0)
        live = rows < M
        j = tl.arange(0, HC)
        r2 = rows[:, None]
        r3 = rows[:, None, None]
        jj = j[None, :, None]
        kk = j[None, None, :]
        a_pre = tl.zeros([BLOCK_M, HC], dtype=tl.float32)
        a_post = tl.zeros([BLOCK_M, HC], dtype=tl.float32)
        a_comb = tl.zeros([BLOCK_M, HC, HC], dtype=tl.float32)
        a_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
        for s in tl.static_range(NUM_SLICES):
            off2 = (s * M + r2) * MIX_COLS
            off3 = (s * M + r3) * MIX_COLS
            m2 = r2 < M
            m3 = r3 < M
            a_pre += tl.load(MIX + off2 + j[None, :], m2, 0, cache_modifier=".cg")
            a_post += tl.load(
                MIX + off2 + HC + j[None, :], m2, 0, cache_modifier=".cg"
            )
            a_comb += tl.load(
                MIX + off3 + 2 * HC + jj * HC + kk, m3, 0, cache_modifier=".cg"
            )
            a_sq += tl.load(SQ + s * M + rows, live, 0, cache_modifier=".cg")
        rsqrt = 1.0 / tl.sqrt(a_sq * INV_K + RMS_EPS)

        s0 = tl.load(SCALE + 0)
        s1 = tl.load(SCALE + 1)
        s2 = tl.load(SCALE + 2)

        pre = tl.sigmoid(a_pre * rsqrt[:, None] * s0 + tl.load(BASE + j)[None, :]) + EPS
        tl.store(PRE + r2 * HC + j[None, :], pre, r2 < M)
        post = 2.0 * tl.sigmoid(
            a_post * rsqrt[:, None] * s1 + tl.load(BASE + HC + j)[None, :]
        )
        tl.store(POST + r2 * HC + j[None, :], post, r2 < M)

        comb = a_comb * rsqrt[:, None, None] * s2 + tl.load(
            BASE + 2 * HC + jj * HC + kk
        )
        comb = tl.exp(comb - tl.max(comb, axis=2)[:, :, None])
        comb = comb / tl.sum(comb, axis=2)[:, :, None] + EPS
        comb = comb / (tl.sum(comb, axis=1)[:, None, :] + EPS)
        for _ in tl.static_range(ITERS - 1):
            comb = comb / (tl.sum(comb, axis=2)[:, :, None] + EPS)
            comb = comb / (tl.sum(comb, axis=1)[:, None, :] + EPS)
        tl.store(COMB + r3 * HC * HC + jj * HC + kk, comb, r3 < M)


def hc_mix_stats_sinkhorn_bf16x3_fused(
    x: torch.Tensor,
    weight_parts,
    scale: torch.Tensor,
    base: torch.Tensor,
    sinkhorn_iters: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int = 4,
):
    """``hc_mix_stats_sinkhorn_bf16x3`` (Blackwell, 4096..65536 rows) in one launch."""
    m, k = x.shape
    mix = (2 + hc_mult) * hc_mult
    slices = _HC_MIX_COMPENSATED_SLICES
    block_m = _HC_MIX_BF16X3_BLOCK_M
    assert x.is_contiguous() and x.dtype == torch.bfloat16
    assert 4096 <= m <= _MAX_ROW_BLOCKS * block_m
    assert k % (slices * _HC_MIX_BLOCK_K) == 0
    assert len(weight_parts) == 3
    assert all(
        w.shape == (mix, k) and w.dtype == torch.bfloat16 and w.is_contiguous()
        for w in weight_parts
    )
    part_mix = torch.empty((slices, m, mix), device=x.device, dtype=torch.float32)
    sq = torch.empty((slices, m), device=x.device, dtype=torch.float32)
    pre = torch.empty((m, hc_mult), device=x.device, dtype=torch.float32)
    post = torch.empty_like(pre)
    comb = torch.empty((m, hc_mult, hc_mult), device=x.device, dtype=torch.float32)
    _hc_mix_stats_sinkhorn_bf16x3_kernel[(triton.cdiv(m, block_m), slices)](
        x,
        *weight_parts,
        part_mix,
        sq,
        _row_block_locks(x.device),
        scale,
        base,
        pre,
        post,
        comb,
        m,
        1.0 / k,
        rms_eps,
        K=k,
        K_PER_SLICE=k // slices,
        MIX_COLS=mix,
        MIX_PAD=triton.next_power_of_2(mix),
        BLOCK_K=_HC_MIX_BLOCK_K,
        BLOCK_M=block_m,
        HC=hc_mult,
        NUM_SLICES=slices,
        ITERS=sinkhorn_iters,
        EPS=hc_eps,
        num_warps=4,
        num_stages=3,
    )
    return pre, post, comb
