"""Emitter projections whose outputs are consumed by recurrent/KV state.

Only the opt-in emitter calls these helpers; target and draft layers retain
their original forward. Weight row views preserve TP shard order and ownership.
GPU admission must check rounding after changing the GEMM output width.
"""

import torch


def project_rows(hidden, linear, start, stop):
    """Use the configured BF16 GEMM backend on a contiguous row interval."""
    weight = linear.weight
    if hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("state-only emitters require BF16 projections")
    if getattr(linear, "bias", None) is not None:
        raise ValueError("state-only emitters require bias-free projections")
    if not 0 <= start < stop <= weight.shape[0]:
        raise ValueError("emitter projection row range is outside the TP shard")
    # No concatenation, parameter mutation, or per-call weight allocation.
    if hidden.is_cuda:
        from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch

        return bf16_gemm_dispatch(hidden, weight[start:stop], None)
    return torch.nn.functional.linear(hidden, weight[start:stop])


def emit_gdn_state(gdn, hidden, forward_batch):
    """Keep the identical Conv/GDN state update; omit Z, norm and out_proj."""
    if not forward_batch.forward_mode.is_extend_without_speculative():
        raise ValueError("state-only GDN emitter requires ordinary prompt extend")
    k = gdn.key_dim // gdn.attn_tp_size
    v = gdn.value_dim // gdn.attn_tp_size
    heads = gdn.num_v_heads // gdn.attn_tp_size
    mixed_qkv = project_rows(hidden, gdn.in_proj_qkvz, 0, 2 * k + v)
    ba, _ = gdn.in_proj_ba(hidden)
    # These are the same B/A views used by qwen3_5_gdn_prefill_projection_views.
    # The backend owns conv, dense-ring continuation and factor publication.
    gdn.attn(
        forward_batch, mixed_qkv=mixed_qkv, a=ba[:, heads : 2 * heads], b=ba[:, :heads]
    )


def project_qsa_kv(source, hidden, positions):
    """Retain the original fused kernel's K branch, with zero query heads."""
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )

    q_width = source.q_size * (2 if source.attn_output_gate else 1)
    kv = project_rows(hidden, source.qkv_proj, q_width, q_width + 2 * source.kv_size)
    k, v = kv.split(source.kv_size, dim=-1)
    _, k, _ = fused_qk_gemma_rmsnorm_rope_gate(
        k[:, :0],
        k,
        source.q_norm.weight.data,
        source.k_norm.weight.data,
        source.rotary_emb.cos_sin_cache,
        positions,
        source.q_norm.variance_epsilon,
        0,
        source.num_kv_heads,
        source.head_dim,
        source.rotary_emb.rotary_dim,
        has_gate=False,
        mrope_axis_map=(source.rotary_emb.axis_map if positions.dim() == 2 else None),
    )
    return k, v


def project_index_key(indexer, hidden):
    """Raw index K; the existing update method stores and compresses it."""
    start = indexer.index_n_heads * indexer.index_head_dim
    stop = start + indexer.index_kv_heads * indexer.index_head_dim
    return project_rows(hidden, indexer.index_qk_proj, start, stop).reshape(
        hidden.shape[0], indexer.index_kv_heads, indexer.index_head_dim
    )
