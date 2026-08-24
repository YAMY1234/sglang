# SPDX-License-Identifier: Apache-2.0
"""F2: multi-warp GDN ReplaySSM fold-every-commit — DROP-IN replacement for
``fla/gdn_replayssm_spec_fold.py`` (same host functions, same signatures, same
semantics; callers unchanged).

Why (F1 closure evidence, diag_f1_regs.py CMH 3069773): the original fold is
latency-bound, not bandwidth-bound — 5.27 ms/step at B=64/T=6/45 layers vs a
~1.4 ms two-pass bf16 bandwidth floor. At num_warps=1 each program's 32
threads carry a [128, 32] tile = 128 elements/thread of element-wise replay
work per step, ~2400 dependent instructions over a 6-step window. The fix is
plain parallelism: more warps per program (fewer elements per thread) and an
optionally wider V tile (fewer programs, wider vectorized loads — the CuTeDSL
verify's LDG.128 wide-vec layout is the existing proof this shape wants it).

Numerics: the replay MATH is source-identical (division-form L2 norm, same op
order). Multi-warp layouts change tl.sum reduction trees, so results diverge
from the num_warps=1 kernel at the ULP level — exactly the cross-kernel
divergence class measured and gated during F1 (fp32 1-3 ULP/step scattered;
bf16 re-synced by the per-window store except boundary straddles). The old
"BITWISE CLONE / keep num_warps=1" mandate is dropped BY DECISION (F1 iter2/3
proved cross-kernel bitwise unachievable even at equal configs); acceptance =
fp64-reference no-worse gate + density/RMS envelope + internal bitwise
determinism + GSM8K (test_f2_multiwarp_fold.py).

Launch tuning lives in _FOLD_TUNE (production defaults from the CMH sweep in
bench_f2_fold.py); commit_gdn_replayssm_fold_all_layers accepts optional
overrides for benchmarking without affecting callers.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Production-tuned launch config (GB300, B=64, T=6, 45 layers, bf16 state;
# see bench_f2_fold.py sweep). bv is capped by V at launch time.
# [f2-final] Sweep verdict (three variants, GB300 B=64/T=6/45L bf16):
#   warp scaling alone      1.14x max (latency-bound, not per-thread-work)
#   static_range unroll     1.18x max (w=1 regresses to 0.80x)
#   ring prefetch + extract 1.43x @ w=1 (3.794 ms; collapses at w>1)
# Best = prefetch@w1 (this default) — BELOW the 1.5x stop-loss; shipped as
# best-known state, route paused pending coordinator ROI call. Numerics stay
# envelope-equivalent (batched norm tree differs from per-step; same division
# form); gates in test_f2_multiwarp_fold.py, 15/15 green.
_FOLD_TUNE = {
    "bv": 32,
    "num_warps": 1,  # HARD requirement for prefetch (see kernel comment)
    "num_stages": 3,
    "prefetch": True,
}


@triton.jit
def gdn_replayssm_exact_fold_kernel(
    h0,  # [num_slots, HV, K, V] pool-dtype checkpoint (folded in place)
    rawv_cache,  # [num_slots, HV, RL, V]  raw v
    rawk_cache,  # [num_slots, H,  RL, K]  raw pre-norm k
    g_cache,  # [num_slots, HV, RL]     fp32 scalar log-decay gate
    beta_cache,  # [num_slots, HV, RL]     fp32 beta
    ssm_state_indices,  # [B] int  physical slot per request
    accept_lens,  # [B] int  committed prefix length per request (incl. bonus)
    mamba_track_indices,  # [B] int  extra_buffer track slot (or NULL) per request
    mamba_steps_to_track,  # [B] int  crossing step (or -1) per request
    stride_state_slot: tl.constexpr,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_g_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    stride_state_layer: tl.constexpr,
    stride_rawv_layer: tl.constexpr,
    stride_rawk_layer: tl.constexpr,
    stride_g_layer: tl.constexpr,
    stride_beta_layer: tl.constexpr,
    stride_indices: tl.constexpr,
    stride_accept: tl.constexpr,
    stride_track: tl.constexpr,
    stride_steps: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    HAS_TRACK: tl.constexpr,
    NP2_RL: tl.constexpr = 1,
    PREFETCH: tl.constexpr = False,
):
    """Replay the accepted window into the checkpoint. PREFETCH=False keeps
    the body line-identical to the original kernel (only the launch config
    differs); PREFETCH=True hoists the ring loads + L2 norms ahead of the
    dependent loop — see _FOLD_TUNE for the rationale and numerics contract."""
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_hvl = tl.program_id(2)
    # int64: layer stride * i_layer can overflow an int32 index product.
    i_layer = (i_hvl // HV).to(tl.int64)
    i_hv = i_hvl % HV
    i_h = i_hv // (HV // H)
    h0 = h0 + i_layer * stride_state_layer
    rawv_cache = rawv_cache + i_layer * stride_rawv_layer
    rawk_cache = rawk_cache + i_layer * stride_rawk_layer
    g_cache = g_cache + i_layer * stride_g_layer
    beta_cache = beta_cache + i_layer * stride_beta_layer

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices).to(tl.int64)
    if state_idx <= NULL_BLOCK_ID:
        return
    n_commit = tl.load(accept_lens + i_n * stride_accept).to(tl.int32)
    if n_commit <= 0:
        return

    if HAS_TRACK:
        track_idx = tl.load(mamba_track_indices + i_n * stride_track).to(tl.int64)
        track_step = tl.load(mamba_steps_to_track + i_n * stride_steps).to(tl.int32)
    else:
        track_idx = NULL_BLOCK_ID
        track_step = -1

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    p_h0 = (
        h0
        + state_idx * stride_state_slot
        + i_hv * K * V
        + o_v[None, :] * K
        + o_k[:, None]
    )
    b_h = tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

    if PREFETCH:
        # [f2-final] Batched ring prefetch + register row-extract: hoist the
        # whole ring window off the b_h dependency chain (rows t >= n_commit
        # masked to 0, never extracted), batch the L2 norms, then run the
        # replay from registers. Measured best of three variants
        # (bench_f2_fold.py, GB300 B=64/T=6/45L bf16): 3.794 ms = 1.43x at
        # w=1. Rejected variants: warp scaling alone (1.14x max — the kernel
        # is latency-bound, not per-thread-work bound); static_range unroll
        # (0.80x at w=1, 1.18x max — unrolled load windows did not hoist).
        # MUST stay num_warps=1: with the t-axis split across warps the
        # per-step row-extract becomes cross-warp smem reductions (0.24-0.60x
        # measured collapse).
        o_t = tl.arange(0, NP2_RL)
        mask_t = o_t < n_commit
        b_k_all = tl.load(
            rawk_cache
            + state_idx * stride_rawk_slot
            + (i_h * MAX_CACHE_LEN + o_t[:, None]).to(tl.int64) * K
            + o_k[None, :],
            mask=mask_t[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        b_v_all = tl.load(
            rawv_cache
            + state_idx * stride_rawv_slot
            + (i_hv * MAX_CACHE_LEN + o_t[:, None]).to(tl.int64) * V
            + o_v[None, :],
            mask=mask_t[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        b_g_all = tl.load(
            g_cache + state_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + o_t,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32)
        b_beta_all = tl.load(
            beta_cache + state_idx * stride_beta_slot + i_hv * MAX_CACHE_LEN + o_t,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            # Batched division-form L2 norm (all-zero masked rows: 0/sqrt(eps)
            # stays 0). Reduction tree differs from the per-step original ->
            # envelope-gated like every other cross-kernel delta.
            b_k_all = b_k_all / (
                tl.sqrt(tl.sum(b_k_all * b_k_all, 1) + 1e-6)[:, None]
            )
        for t in range(0, n_commit):
            # register row-extract (tree-verify parent-lookup precedent)
            b_k = tl.sum(tl.where(o_t[:, None] == t, b_k_all, 0.0), 0)
            b_v = tl.sum(tl.where(o_t[:, None] == t, b_v_all, 0.0), 0)
            b_g = tl.sum(tl.where(o_t == t, b_g_all, 0.0))
            b_beta = tl.sum(tl.where(o_t == t, b_beta_all, 0.0))

            b_h *= tl.exp(b_g)
            b_v -= tl.sum(b_h * b_k[:, None], 0)
            b_v *= b_beta
            b_h += b_k[:, None] * b_v[None, :]

            if HAS_TRACK:
                if (t == track_step) and (track_idx > NULL_BLOCK_ID):
                    tl.store(
                        h0
                        + track_idx * stride_state_slot
                        + i_hv * K * V
                        + o_v[None, :] * K
                        + o_k[:, None],
                        b_h.to(h0.dtype.element_ty),
                        mask=mask_h,
                    )
    else:
        for t in range(0, n_commit):
            phys = t.to(tl.int64)
            b_k = tl.load(
                rawk_cache
                + state_idx * stride_rawk_slot
                + (i_h * MAX_CACHE_LEN + phys) * K
                + o_k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            b_v = tl.load(
                rawv_cache
                + state_idx * stride_rawv_slot
                + (i_hv * MAX_CACHE_LEN + phys) * V
                + o_v,
                mask=mask_v,
                other=0.0,
            ).to(tl.float32)
            b_g = tl.load(
                g_cache + state_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + phys
            ).to(tl.float32)
            b_beta = tl.load(
                beta_cache + state_idx * stride_beta_slot + i_hv * MAX_CACHE_LEN + phys
            ).to(tl.float32)

            if USE_QK_L2NORM_IN_KERNEL:
                b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
            b_h *= tl.exp(b_g)
            b_v -= tl.sum(b_h * b_k[:, None], 0)
            b_v *= b_beta
            b_h += b_k[:, None] * b_v[None, :]

            if HAS_TRACK:
                if (t == track_step) and (track_idx > NULL_BLOCK_ID):
                    tl.store(
                        h0
                        + track_idx * stride_state_slot
                        + i_hv * K * V
                        + o_v[None, :] * K
                        + o_k[:, None],
                        b_h.to(h0.dtype.element_ty),
                        mask=mask_h,
                    )

    tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def commit_gdn_replayssm_fold_all_layers(
    checkpoint_state: torch.Tensor,  # [num_layers, num_slots, HV, K, V], in place
    rawv_cache: torch.Tensor,  # [num_layers, num_slots, HV, RL, V]
    rawk_cache: torch.Tensor,  # [num_layers, num_slots, H,  RL, K]
    g_cache: torch.Tensor,  # [num_layers, num_slots, HV, RL] fp32
    beta_cache: torch.Tensor,  # [num_layers, num_slots, HV, RL] fp32
    ssm_state_indices: torch.Tensor,  # [B] int   (shared across layers)
    accept_lens: torch.Tensor,  # [B] int, incl. the bonus token
    max_cache_len: int,
    num_k_heads: int,
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    null_block_id: int = -1,
    # F2 bench-only overrides; callers never pass these.
    bv: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    prefetch: bool | None = None,
) -> None:
    """Fold every layer's accepted window in one launch (layer packed into
    grid axis 2). F2: multi-warp launch (see module docstring); results are
    envelope-equivalent, not bitwise-equal, to the num_warps=1 original."""
    num_layers, num_slots, HV = checkpoint_state.shape[:3]
    K = rawk_cache.shape[-1]
    V = rawv_cache.shape[-1]
    B = ssm_state_indices.shape[0]
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), bv if bv is not None else _FOLD_TUNE["bv"])
    n_warps = num_warps if num_warps is not None else _FOLD_TUNE["num_warps"]
    n_stages = num_stages if num_stages is not None else _FOLD_TUNE["num_stages"]
    do_prefetch = prefetch if prefetch is not None else _FOLD_TUNE["prefetch"]
    grid = (triton.cdiv(V, BV), B, HV * num_layers)
    has_track = mamba_track_indices is not None and mamba_steps_to_track is not None
    if has_track:
        track_idx_t = mamba_track_indices
        steps_t = mamba_steps_to_track
        stride_track = track_idx_t.stride(0)
        stride_steps = steps_t.stride(0)
    else:
        # Unused when HAS_TRACK is False; pass a valid pointer.
        track_idx_t = ssm_state_indices
        steps_t = accept_lens
        stride_track = 0
        stride_steps = 0
    gdn_replayssm_exact_fold_kernel[grid](
        checkpoint_state,
        rawv_cache,
        rawk_cache,
        g_cache,
        beta_cache,
        ssm_state_indices,
        accept_lens,
        track_idx_t,
        steps_t,
        checkpoint_state.stride(1),
        rawv_cache.stride(1),
        rawk_cache.stride(1),
        g_cache.stride(1),
        beta_cache.stride(1),
        checkpoint_state.stride(0),
        rawv_cache.stride(0),
        rawk_cache.stride(0),
        g_cache.stride(0),
        beta_cache.stride(0),
        ssm_state_indices.stride(0),
        accept_lens.stride(0),
        stride_track,
        stride_steps,
        H=num_k_heads,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        MAX_CACHE_LEN=max_cache_len,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        NULL_BLOCK_ID=null_block_id,
        HAS_TRACK=has_track,
        NP2_RL=triton.next_power_of_2(max_cache_len),
        PREFETCH=do_prefetch,
        num_warps=n_warps,
        num_stages=n_stages,
    )


def commit_gdn_replayssm_fold_after_verify(
    *,
    spec_state,  # MambaPool.SpeculativeState (all layers)
    state_batch_indices: torch.Tensor,  # [B] per-req mamba slot
    accept_lens: torch.Tensor,  # [B] int, incl. the bonus token
    last_correct_step_indices: torch.Tensor,  # [B] conv rollback target step
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    null_block_id: int = -1,
) -> None:
    """Fold each layer's accepted window into ``temporal``, then do the usual
    conv accept-rollback (+ the track-slot conv scatter under extra_buffer).
    Unchanged from the original module."""
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_conv_window_scatter_with_mask,
    )

    max_cache_len = spec_state.replayssm_rawv.shape[-2]
    num_k_heads = spec_state.replayssm_rawk.shape[2]
    commit_gdn_replayssm_fold_all_layers(
        checkpoint_state=spec_state.temporal,
        rawv_cache=spec_state.replayssm_rawv,
        rawk_cache=spec_state.replayssm_rawk,
        g_cache=spec_state.replayssm_g,
        beta_cache=spec_state.replayssm_beta,
        ssm_state_indices=state_batch_indices,
        accept_lens=accept_lens,
        max_cache_len=max_cache_len,
        num_k_heads=num_k_heads,
        mamba_track_indices=mamba_track_indices,
        mamba_steps_to_track=mamba_steps_to_track,
        null_block_id=null_block_id,
    )
    for conv_states, interm_conv in zip(
        spec_state.conv, spec_state.intermediate_conv_window
    ):
        fused_conv_window_scatter_with_mask(
            conv_states, interm_conv, state_batch_indices, last_correct_step_indices
        )
        if mamba_track_indices is not None and mamba_steps_to_track is not None:
            fused_conv_window_scatter_with_mask(
                conv_states, interm_conv, mamba_track_indices, mamba_steps_to_track
            )
