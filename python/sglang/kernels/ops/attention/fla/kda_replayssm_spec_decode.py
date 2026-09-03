# SPDX-License-Identifier: Apache-2.0
"""ReplaySSM speculative-decode state commit for KDA (Kimi Delta Attention).

KDA keeps its own recurrent verify kernel for the per-step OUTPUT (unchanged),
so -- unlike the GDN spec kernel (gdn_replayssm_spec_decode.py) -- this module
does NOT reconstruct the verify output. It only replaces the per-draft-token
full-SSM snapshot cache (``intermediate_ssm``, ``max_running x (gamma+1) x
[HV,V,K]``, the memory hog that collapses dspark concurrency) with a small
per-request input window + an exact fold on commit.

Scheme (fold-every-commit, no circular ring / periodic flush):
  * during verify, the KDA backend stores the draft window's raw inputs
    (raw v, raw pre-norm k, per-K log-decay gate ``gk`` (fp32), beta (fp32))
    into the request's window row at positions ``0..spec_len``. The window is
    request-scoped scratch: rows are indexed by the same per-request scratch
    row as ``intermediate_conv_window`` (``ring_indices``), not by the
    persistent mamba slot, so it is sized by max running requests instead of
    by the radix-cache slot count;
  * on commit, :func:`commit_kda_replayssm_spec` replays the *accepted* prefix
    ``0..accept_len`` from the persistent checkpoint ``h0`` (indexed by the
    mamba slot, ``ssm_state_indices``) into ``h0`` in place -- ``h0`` is always
    the current committed state, so the next verify reads it directly as its
    initial state (no lag, no chunked reconstruction).

The exact fold is a BITWISE CLONE of ``fused_recurrent_gated_delta_rule_fwd_kernel``'s
``IS_KDA`` branch (fused_recurrent.py): same [BK, BV] fp32 tile (K rows, V cols),
same division-form L2 norm (eps inside sqrt), same per-K gate decay
``h *= exp(gk)``, same decay->delta->rank-1 op order. Given identical inputs the
folded checkpoint is bit-identical to the recurrent baseline's committed state
(the delta-rule recurrence is contractive, so no length-dependent error). Do NOT
reorder into tl.dot / reciprocal-multiply; keep num_warps=1 so the reduction
trees match.

Linear chain only (dspark gamma chain, topk<=1).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def kda_replayssm_exact_fold_kernel(
    h0,  # [num_slots, HV, V, K] fp32 checkpoint (folded in place)
    rawv_cache,  # [num_rows, HV, L, V]  raw v
    rawk_cache,  # [num_rows, H,  L, K]  raw pre-norm k
    gk_cache,  # [num_rows, HV, L, K]  fp32 per-K log-decay gate
    beta_cache,  # [num_rows, HV, L]     fp32 beta
    ssm_state_indices,  # [B] int  physical checkpoint slot per request
    ring_indices,  # [B] int  window row per request
    accept_lens,  # [B] int  committed prefix length per request
    mamba_track_indices,  # [B] int  extra_buffer track slot (or NULL) per request
    mamba_steps_to_track,  # [B] int  crossing step (or -1) per request
    stride_state_slot: tl.constexpr,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_gk_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    stride_state_layer: tl.constexpr,
    stride_rawv_layer: tl.constexpr,
    stride_rawk_layer: tl.constexpr,
    stride_gk_layer: tl.constexpr,
    stride_beta_layer: tl.constexpr,
    stride_indices: tl.constexpr,
    stride_ring_indices: tl.constexpr,
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
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    # program_id(2) packs (layer, v-head): layer-major so a single launch folds
    # all KDA layers. num_layers=1 launches (per-layer entry) keep i_layer == 0.
    i_hvl = tl.program_id(2)
    # int64: layer stride * i_layer overflows int32 at K3 scale (69 layers x
    # ~34M-element per-layer stride > 2^31).
    i_layer = (i_hvl // HV).to(tl.int64)
    i_hv = i_hvl % HV
    i_h = i_hv // (HV // H)
    # Shift the layer-indexed bases once; every pointer below is layer-relative.
    h0 = h0 + i_layer * stride_state_layer
    rawv_cache = rawv_cache + i_layer * stride_rawv_layer
    rawk_cache = rawk_cache + i_layer * stride_rawk_layer
    gk_cache = gk_cache + i_layer * stride_gk_layer
    beta_cache = beta_cache + i_layer * stride_beta_layer

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices).to(tl.int64)
    if state_idx <= NULL_BLOCK_ID:
        return
    n_commit = tl.load(accept_lens + i_n * stride_accept).to(tl.int32)
    if n_commit <= 0:
        return
    ring_idx = tl.load(ring_indices + i_n * stride_ring_indices).to(tl.int64)

    # extra_buffer: snapshot the interval-crossing state into the track ping-pong
    # slot on the step it crosses (mask-gated: -1 track step / NULL slot = skip).
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

    # [BK, BV] tile: K rows / V cols, matching the recurrent baseline's memory
    # offset (v * K + k, K contiguous).
    p_h0 = (
        h0
        + state_idx * stride_state_slot
        + i_hv * V * K
        + o_v[None, :] * K
        + o_k[:, None]
    )
    b_h = tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

    for t in range(0, n_commit):
        phys = t.to(tl.int64)
        b_k = tl.load(
            rawk_cache
            + ring_idx * stride_rawk_slot
            + (i_h * MAX_CACHE_LEN + phys) * K
            + o_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            rawv_cache
            + ring_idx * stride_rawv_slot
            + (i_hv * MAX_CACHE_LEN + phys) * V
            + o_v,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        b_gk = tl.load(
            gk_cache
            + ring_idx * stride_gk_slot
            + (i_hv * MAX_CACHE_LEN + phys) * K
            + o_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        b_beta = tl.load(
            beta_cache + ring_idx * stride_beta_slot + i_hv * MAX_CACHE_LEN + phys
        ).to(tl.float32)

        # --- verbatim recurrent update, IS_KDA branch (see module docstring) ---
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_h *= tl.exp(b_gk[:, None])  # per-K gate decay, broadcast over V
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]

        # Interval-crossing snapshot -> track slot (state AFTER step `track_step`).
        if HAS_TRACK:
            if (t == track_step) and (track_idx > NULL_BLOCK_ID):
                tl.store(
                    h0
                    + track_idx * stride_state_slot
                    + i_hv * V * K
                    + o_v[None, :] * K
                    + o_k[:, None],
                    b_h.to(h0.dtype.element_ty),
                    mask=mask_h,
                )

    tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def _launch_kda_exact_fold(
    *,
    checkpoint_state: torch.Tensor,
    rawv_cache: torch.Tensor,
    rawk_cache: torch.Tensor,
    gk_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    ring_indices: Optional[torch.Tensor],
    accept_lens: torch.Tensor,
    max_cache_len: int,
    num_k_heads: int,
    mamba_track_indices: Optional[torch.Tensor],
    mamba_steps_to_track: Optional[torch.Tensor],
    use_qk_l2norm_in_kernel: bool,
    null_block_id: int,
    layered: bool,
) -> None:
    """Shared launcher for the per-layer and all-layer fold entry points.

    Tiling clones the recurrent kernel (full-K rows, BV = min(np2(V), 32) cols,
    num_warps=1) so the folded checkpoint is bit-identical to the recurrent
    baseline's committed state. ``ring_indices`` defaults to
    ``ssm_state_indices`` for slot-keyed windows.
    """
    if layered:
        num_layers, _, HV, V, K = checkpoint_state.shape
        slot_dim = 1
    else:
        num_layers = 1
        _, HV, V, K = checkpoint_state.shape
        slot_dim = 0
    if ring_indices is None:
        ring_indices = ssm_state_indices
    B = ssm_state_indices.shape[0]
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    grid = (triton.cdiv(V, BV), B, HV * num_layers)
    has_track = mamba_track_indices is not None and mamba_steps_to_track is not None
    if has_track:
        track_idx_t = mamba_track_indices
        steps_t = mamba_steps_to_track
        stride_track = track_idx_t.stride(0)
        stride_steps = steps_t.stride(0)
    else:
        track_idx_t = ssm_state_indices  # unused (HAS_TRACK False); valid ptr
        steps_t = accept_lens
        stride_track = 0
        stride_steps = 0

    def layer_stride(t: torch.Tensor) -> int:
        return t.stride(0) if layered else 0

    kda_replayssm_exact_fold_kernel[grid](
        checkpoint_state,
        rawv_cache,
        rawk_cache,
        gk_cache,
        beta_cache,
        ssm_state_indices,
        ring_indices,
        accept_lens,
        track_idx_t,
        steps_t,
        checkpoint_state.stride(slot_dim),
        rawv_cache.stride(slot_dim),
        rawk_cache.stride(slot_dim),
        gk_cache.stride(slot_dim),
        beta_cache.stride(slot_dim),
        layer_stride(checkpoint_state),
        layer_stride(rawv_cache),
        layer_stride(rawk_cache),
        layer_stride(gk_cache),
        layer_stride(beta_cache),
        ssm_state_indices.stride(0),
        ring_indices.stride(0),
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
        num_warps=1,
        num_stages=3,
    )


def commit_kda_replayssm_spec(
    checkpoint_state: torch.Tensor,  # [num_slots, HV, V, K] fp32, folded in place
    rawv_cache: torch.Tensor,  # [num_rows, HV, L, V]
    rawk_cache: torch.Tensor,  # [num_rows, H,  L, K]
    gk_cache: torch.Tensor,  # [num_rows, HV, L, K] fp32
    beta_cache: torch.Tensor,  # [num_rows, HV, L]    fp32
    ssm_state_indices: torch.Tensor,  # [B] int
    accept_lens: torch.Tensor,  # [B] int  (incl. the bonus token)
    max_cache_len: int,
    num_k_heads: int,
    mamba_track_indices: Optional[torch.Tensor] = None,  # [B] extra_buffer track slot
    mamba_steps_to_track: Optional[torch.Tensor] = None,  # [B] crossing step (or -1)
    use_qk_l2norm_in_kernel: bool = True,
    null_block_id: int = 0,
    ring_indices: Optional[torch.Tensor] = None,  # [B] window row; default = slot
) -> None:
    """Replay each request's accepted window into its fp32 checkpoint in place.

    With extra_buffer (mamba_track_indices given) the same replay snapshots the
    interval-crossing state into the track slot in one pass, so no separate
    track scatter / force-flush is needed.
    """
    _launch_kda_exact_fold(
        checkpoint_state=checkpoint_state,
        rawv_cache=rawv_cache,
        rawk_cache=rawk_cache,
        gk_cache=gk_cache,
        beta_cache=beta_cache,
        ssm_state_indices=ssm_state_indices,
        ring_indices=ring_indices,
        accept_lens=accept_lens,
        max_cache_len=max_cache_len,
        num_k_heads=num_k_heads,
        mamba_track_indices=mamba_track_indices,
        mamba_steps_to_track=mamba_steps_to_track,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        null_block_id=null_block_id,
        layered=False,
    )


def commit_kda_replayssm_spec_all_layers(
    checkpoint_state: torch.Tensor,  # [num_layers, num_slots, HV, V, K] fp32, in place
    rawv_cache: torch.Tensor,  # [num_layers, num_rows, HV, L, V]
    rawk_cache: torch.Tensor,  # [num_layers, num_rows, H,  L, K]
    gk_cache: torch.Tensor,  # [num_layers, num_rows, HV, L, K] fp32
    beta_cache: torch.Tensor,  # [num_layers, num_rows, HV, L]    fp32
    ssm_state_indices: torch.Tensor,  # [B] int   (shared across layers)
    accept_lens: torch.Tensor,  # [B] int
    max_cache_len: int,
    num_k_heads: int,
    mamba_track_indices: Optional[torch.Tensor] = None,
    mamba_steps_to_track: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    null_block_id: int = 0,
    ring_indices: Optional[torch.Tensor] = None,  # [B] window row; default = slot
) -> None:
    """Fold every layer's accepted window in a single launch.

    The layer is packed into the head grid axis (program_id(2) = layer * HV +
    head), so the result is bit-identical to a per-layer loop of
    :func:`commit_kda_replayssm_spec`. ssm_state_indices / ring_indices /
    accept_lens / track are per-request and shared across layers.
    """
    _launch_kda_exact_fold(
        checkpoint_state=checkpoint_state,
        rawv_cache=rawv_cache,
        rawk_cache=rawk_cache,
        gk_cache=gk_cache,
        beta_cache=beta_cache,
        ssm_state_indices=ssm_state_indices,
        ring_indices=ring_indices,
        accept_lens=accept_lens,
        max_cache_len=max_cache_len,
        num_k_heads=num_k_heads,
        mamba_track_indices=mamba_track_indices,
        mamba_steps_to_track=mamba_steps_to_track,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        null_block_id=null_block_id,
        layered=True,
    )


def commit_kda_replayssm_after_verify(
    *,
    spec_state,  # MambaPool.SpeculativeState (all layers)
    state_batch_indices: torch.Tensor,  # [B] per-req mamba slot
    ring_indices: torch.Tensor,  # [B] per-req window row (verify scratch row)
    accept_lens: torch.Tensor,  # [B] int, incl. the bonus token
    last_correct_step_indices: torch.Tensor,  # [B] conv rollback target step
    mamba_track_indices: Optional[torch.Tensor] = None,
    mamba_steps_to_track: Optional[torch.Tensor] = None,
    null_block_id: int = -1,
) -> None:
    """Fold each layer's accepted window into `temporal` and roll back conv.

    Single commit entry point shared by the generic spec_utils commit and the
    dspark/dflash direct `update_mamba_state_after_mtp_verify` path. The
    window rows were written during verify at the request's scratch row
    (``ring_indices``, the same row `intermediate_conv_window` uses); the fold
    replays the accepted prefix into the fp32 checkpoint at the request's
    mamba slot, so `temporal` stays current. Conv still needs its usual
    accept-rollback.
    """
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_conv_window_scatter_with_mask,
    )

    L = spec_state.replayssm_rawv.shape[-2]
    num_k_heads = spec_state.replayssm_rawk.shape[2]
    commit_kda_replayssm_spec_all_layers(
        checkpoint_state=spec_state.temporal,
        rawv_cache=spec_state.replayssm_rawv,
        rawk_cache=spec_state.replayssm_rawk,
        gk_cache=spec_state.replayssm_g,
        beta_cache=spec_state.replayssm_beta,
        ssm_state_indices=state_batch_indices,
        accept_lens=accept_lens,
        max_cache_len=L,
        num_k_heads=num_k_heads,
        mamba_track_indices=mamba_track_indices,
        mamba_steps_to_track=mamba_steps_to_track,
        null_block_id=null_block_id,
        ring_indices=ring_indices,
    )
    # Conv rollback + track-slot conv snapshot, per conv group (fold already did
    # the ssm side via HAS_TRACK). Loop mirrors the recurrent commit's zip; track
    # scatter is mask-gated (step -1 => skip).
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
