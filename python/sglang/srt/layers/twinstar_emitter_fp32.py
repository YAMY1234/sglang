"""#873 fp32 emitter (DUET default; --duet-emitter-precision=bf16 selects production arithmetic).

Mingyuan's release loader builds every emitter in fp32 over bf16-rounded weights (twinstar/models/ckpt.py
``_load_qwen4_twinstar``: ``Qwen4Emitter(cfg, l).float()``; A_log / dt_bias stay fp32) and rounds only what reaches a
cache: K, V and the raw indexer key to the model dtype; the GDN conv window when the target layer reads it; the
recurrent state stays fp32.  The helpers compute the same quantities from the served parameters, upcast per call
(no persistent fp32 copies), in the reference's operation order (twinstar/models/qwen4_exp.py ``GatedResidual.mix``,
``QSAAttention.kv``, ``QSAIndexer.keys``; twinstar/models/blocks.py ``GemmaRMSNorm``, ``PartialRotary``,
``apply_rope_partial``, ``gdn_mix``):

  * ``hc_mix``      per-stream Gemma RMS norm, low-rank sigmoid gate, stream mean;
  * ``qsa_kv``      k_proj -> Gemma k_norm -> partial neox RoPE, v_proj; the RoPE tables are rounded to the model
                    dtype, as the reference builds them with ``rotary(positions, h.dtype)``;
  * ``index_key``   the raw (pre-norm, pre-RoPE) indexer key rows;
  * ``gdn_inputs``  in_proj_qkv rows and in_proj_b / in_proj_a (the backend runs the fp32 conv);
  * ``gdn_gating``  decay and beta with torch softplus / sigmoid;
  * ``gdn_extend``  fla 0.5.2 (``/work/pyuser``) chunk gated delta rule on fp32 inputs -- the reference's kernel; the
                    vendored copy refuses fp32 -- called piecewise in ``ChunkGatedDeltaRuleFunction.forward`` order so
                    the per-chunk states that radix tracking reads come out too.

fp32 GEMMs run with TF32 off whatever the process setting is.
"""
import contextlib

import torch
import torch.nn.functional as F

CHUNK = 64


@contextlib.contextmanager
def _no_tf32():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def linear(x, weight):
    """fp32 x @ weightᵀ with the served (bf16-exact) weight upcast for this call."""
    with _no_tf32():
        return F.linear(x.float(), weight.float())


def hc_mix(hc, streams):
    """GatedResidual.mix on flat streams (T, hc * D) -> x (T, D) fp32."""
    count, hidden = hc.hc_count, hc.hidden_size
    s = streams.float().view(-1, count, hidden)
    normed = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + hc.hc_norm.variance_epsilon)
    normed = normed * (1.0 + hc.hc_norm.weight.float().view(count, hidden))
    flat = normed.flatten(-2)
    gate = torch.sigmoid(linear(F.silu(linear(flat, hc.input_mix_weight_down.weight) / count),
                                hc.input_mix_weight_up.weight))
    return (gate.view_as(normed) * normed).mean(dim=-2)


def gemma_rms(x, norm):
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    return out * (1.0 + norm.weight.float())


_INV_FREQ = {}


def rope_tables(rotary_emb, positions, dtype):
    """PartialRotary(rotary_dim, theta)(positions, dtype): cos / sin (T, rotary_dim) rounded to `dtype`."""
    r, base = rotary_emb.rotary_dim, rotary_emb.base
    key = (r, float(base), positions.device)
    inv = _INV_FREQ.get(key)
    if inv is None:  # built on the host in fp32, as the reference's buffer
        inv = _INV_FREQ[key] = (1.0 / (base ** (torch.arange(0, r, 2, dtype=torch.int64).float() / r))).to(positions.device)
    freqs = positions.float()[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rope_partial(x, cos, sin):
    """apply_rope_partial on (T, heads, head_dim) with (T, rotary_dim) tables (neox halves)."""
    r = cos.shape[-1]
    cos, sin = cos[:, None], sin[:, None]
    xr, xp = x[..., :r], x[..., r:]
    x1, x2 = xr.chunk(2, dim=-1)
    return torch.cat([xr * cos + torch.cat([-x2, x1], dim=-1) * sin, xp], dim=-1)


def logical_positions(positions):
    return positions[0] if positions.ndim == 2 else positions


def qsa_kv(src, x, positions, dtype):
    """QSAAttention.kv for the TP-local K / V heads: (T, kv_size) K and V in the cache dtype."""
    q_width = src.q_size * (2 if src.attn_output_gate else 1)
    kv = linear(x, src.qkv_proj.weight[q_width:q_width + 2 * src.kv_size])
    k, v = kv.split(src.kv_size, dim=-1)
    k = gemma_rms(k.reshape(-1, src.num_kv_heads, src.head_dim), src.k_norm)
    cos, sin = rope_tables(src.rotary_emb, logical_positions(positions), dtype)
    k = rope_partial(k, cos, sin)
    return k.reshape(-1, src.kv_size).to(dtype), v.to(dtype)


def index_key(indexer, x, dtype):
    """QSAIndexer.keys: raw indexer key (T, kv_heads, head_dim) in the cache dtype."""
    start = indexer.index_n_heads * indexer.index_head_dim
    stop = start + indexer.index_kv_heads * indexer.index_head_dim
    key = linear(x, indexer.index_qk_proj.weight[start:stop])
    return key.to(dtype).reshape(x.shape[0], indexer.index_kv_heads, indexer.index_head_dim)


def gdn_inputs(gdn, x):
    """gdn_mix up to the conv: fp32 (mixed_qkv, a, b) for the TP-local heads."""
    k = gdn.key_dim // gdn.attn_tp_size
    v = gdn.value_dim // gdn.attn_tp_size
    heads = gdn.num_v_heads // gdn.attn_tp_size
    mixed = linear(x, gdn.in_proj_qkvz.weight[:2 * k + v])
    ba = gdn.in_proj_ba.weight  # rows: b, then a; projected separately like the reference's in_proj_b / in_proj_a
    return mixed, linear(x, ba[heads:2 * heads]), linear(x, ba[:heads])


def gdn_gating(A_log, dt_bias, a, b):
    """g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b): fp32 (1, T, HV)."""
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    return g.unsqueeze(0), b.float().sigmoid().unsqueeze(0)


def gdn_extend(q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc, inplace_update=True, **_):
    """fp32 chunk update of the rows `cache_indices` of `ssm_states` ([N, HV, V, K], sglang V-first layout).

    Returns (output (1, T, HV, V) fp32, None, per-chunk states (1, NT, HV, V, K) in the served tracking dtype (bf16,
    as the vendored kernel's), the same contract as the vendored ``chunk_gated_delta_rule``."""
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.common.chunk_o import chunk_fwd_o
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.index import prepare_chunk_indices

    if not inplace_update:
        raise NotImplementedError("fp32 emitter GDN: multi-item scoring is not supported")
    hk, hv = q.shape[2], v.shape[2]
    if hv != hk:  # the reference repeats q / k to the value heads before the kernel
        q = q.repeat_interleave(hv // hk, dim=2)
        k = k.repeat_interleave(hv // hk, dim=2)
    q, k, v, beta = q.float().contiguous(), k.float().contiguous(), v.float().contiguous(), beta.float().contiguous()
    scale = k.shape[-1] ** -0.5
    rows = cache_indices.long()
    h0 = ssm_states[rows].float().transpose(-1, -2).contiguous()  # fla default layout [N, HV, K, V]
    cu = query_start_loc.long()
    chunk_indices = prepare_chunk_indices(cu, CHUNK)
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)
    g = chunk_local_cumsum(g.float(), chunk_size=CHUNK, scale=RCP_LN2, cu_seqlens=cu, chunk_indices=chunk_indices)
    w, u, _A = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta, cu_seqlens=cu, chunk_indices=chunk_indices,
                                               chunk_size=CHUNK)
    h, v_new, final = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g, initial_state=h0, output_final_state=True,
                                                   cu_seqlens=cu, chunk_indices=chunk_indices, state_v_first=False,
                                                   chunk_size=CHUNK)
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, cu_seqlens=cu, chunk_indices=chunk_indices,
                    state_v_first=False, chunk_size=CHUNK)
    ssm_states[rows] = final.transpose(-1, -2).to(ssm_states.dtype)
    return o, None, h.transpose(-1, -2).to(torch.bfloat16)
