"""Lightning emitter extend/commit with qualified reference FP32 arithmetic.

Like GDN emitter extend, projections do not own the recurrent pool. The conv
and scan use the backend's packed-sequence metadata; commit owns final states.
Native packed arithmetic remains available for numerical research below; it
has not passed variable-prompt greedy equivalence. The serving entry preserves
request-local projection/conv/scan rounding while batching state publication.
No prefill graph, output projection or resident dense copy.
"""

import torch
import torch.nn.functional as F

from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
from sglang.kernels.ops.mamba.triton_ops import mamba_chunk_scan_combined


def emitter_extend(components, layer_id, residual, metadata, pool, plan):
    """One extend plan owns all requests; retain the qualified FP32 operator shapes.

    Numerical ablations require all three reference emitter stages. Grouping
    their state publication is safe; substituting the native packed operators
    is a separate, currently unqualified change, not a transparent optimization.
    """
    outputs, offset = [], 0
    for row, length in enumerate(metadata.mixed_metadata.extend_seq_lens_cpu):
        emitted = components.emitter(layer_id, residual[offset:offset+length])
        outputs.append(emitted)
        if layer_id in components.mamba_emitters:
            pool.conv[pool.layer_map[layer_id], plan.slots[row]].copy_(emitted["conv"])
        offset += length
    if offset != residual.shape[0] or len(outputs) != plan.slots.numel():
        raise ValueError("emitter input differs from extend row plan")
    if layer_id in components.attention_emitters:
        return {key: torch.cat([out[key] for out in outputs]) for key in ("k", "v")}
    pool.commit_extend(layer_id, plan, torch.stack([out["state"] for out in outputs]))
    return None


def emitter_extend_native(components, layer_id, residual, metadata, pool, plan):
    """Experimental native packed arithmetic; not the qualified serving entry."""
    g = components.geometry
    weight = lambda name: components.weights[f"emitters.{layer_id}.{name}"]
    # Same FP32 RMS and full-width GEMM as the frozen emitter (TF32 stays off).
    normalized = residual.float()
    normalized = weight("norm.weight") * (
        normalized * torch.rsqrt(normalized.square().mean(-1, keepdim=True) + components.eps)
    )
    if layer_id in components.attention_emitters:
        return {
            k: F.linear(normalized, weight(f"{k}_proj.weight")).reshape(-1, g.kv_heads, g.attn_dim)
            for k in ("k", "v")
        }
    projected = F.linear(normalized, weight("mixer.in_proj.weight"))
    _, xbc, raw_dt = projected.split([g.intermediate, g.conv_dim, g.heads], -1)
    mixed = metadata.mixed_metadata
    i = pool.layer_map[layer_id]
    convolved = causal_conv1d_fn(
        xbc.T, weight("mixer.conv1d.weight")[:, 0], weight("mixer.conv1d.bias"),
        conv_states=pool.conv[i], query_start_loc=metadata.query_start_loc,
        seq_lens_cpu=mixed.extend_seq_lens_cpu, cache_indices=plan.slots,
        has_initial_state=mixed.has_initial_states, activation="silu",
    ).T
    x, b, c = convolved.split([g.intermediate, g.bc_dim, g.bc_dim], -1)
    # Preserve Torch softplus/add/exp rounding for the FP32 emitter, as GDN's
    # FP32 gating dispatcher does, before passing to the standard scan.
    dt = F.softplus(raw_dt.float() + weight("mixer.dt_bias"))
    scan_x = x.reshape(1, -1, g.heads, g.head_dim)
    final = mamba_chunk_scan_combined(
        scan_x, dt[None], -weight("mixer.A_log").exp(),
        b.reshape(1, -1, g.groups, g.state_dim), c.reshape(1, -1, g.groups, g.state_dim),
        chunk_size=mixed.chunk_size, seq_idx=mixed.seq_idx,
        chunk_indices=mixed.chunk_indices, chunk_offsets=mixed.chunk_offsets,
        cu_seqlens=metadata.query_start_loc, return_varlen_states=True,
        return_final_states=False, state_dtype=torch.float32, out=torch.empty_like(scan_x),
        state_input_precision="ieee",
    )
    pool.commit_extend(layer_id, plan, final)
    return None
