"""#873 fp32 emitter GDN update on GPU: ``twinstar_emitter_fp32.gdn_extend`` against the reference's call.

The reference emitter (twinstar/models/blocks.py ``gdn_mix``) runs fla 0.5.2 ``chunk_gated_delta_rule`` per request
(batch 1, no cu_seqlens) on fp32 q / k / v repeated to the value heads, with fp32 g / beta and an fp32 initial state
([N, HV, K, V]).  ``gdn_extend`` runs the same fla functions piecewise on a varlen batch and writes sglang's V-first
state rows.  Checked: final states and outputs per request (bitwise expected), the per-chunk states against the
per-request final states at chunk boundaries, and the fp32 torch port ``_torch_chunk_gdn`` (tolerance: fla's dots may
round at tf32).

    python test/manual/test_twinstar_emitter_fp32_gpu.py --out result.json
"""
import argparse
import json

import torch
import torch.nn.functional as F

from sglang.srt.layers import twinstar_emitter_fp32 as e32


def torch_chunk_gdn(q, k, v, g, beta, initial_state, chunk_size=64):
    """twinstar/models/blocks.py _torch_chunk_gdn (verbatim, fp32)."""
    dt = v.dtype
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g)]
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    q = q * (Dk ** -0.5)
    pad = (chunk_size - T % chunk_size) % chunk_size
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    n_chunks = (T + pad) // chunk_size
    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    q, k, k_beta, v_beta = [x.reshape(B, H, n_chunks, chunk_size, x.shape[-1]) for x in (q, k, k_beta, v_beta)]
    g = g.reshape(B, H, n_chunks, chunk_size)
    dev = q.device
    strict = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=dev).triu(1)
    cum = g.cumsum(dim=3)
    pair = (cum.unsqueeze(4) - cum.unsqueeze(3)).masked_fill(strict, float("-inf")).exp()
    ut = (k_beta @ k.transpose(-1, -2)) * pair
    intra = (q @ k.transpose(-1, -2)) * pair
    decayed_k_beta = k_beta * cum.exp().unsqueeze(-1)
    new_v = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut, decayed_k_beta, upper=False, unitriangular=True)
    S = torch.zeros(B, H, Dk, Dv, dtype=torch.float32, device=dev) if initial_state is None else initial_state.float().to(dev)
    out = torch.zeros_like(new_v)
    q = q * cum.exp().unsqueeze(-1)
    k = k * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    chunk_decay = cum[..., -1].exp()[..., None, None]
    for i in range(n_chunks):
        v_new = new_v[:, :, i] - k_cumdecay[:, :, i] @ S
        out[:, :, i] = q[:, :, i] @ S + intra[:, :, i] @ v_new
        S = S * chunk_decay[:, :, i] + k[:, :, i].transpose(-1, -2) @ v_new
    out = out.reshape(B, H, -1, Dv)[:, :, :T].transpose(1, 2).contiguous().to(dt)
    return out, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(0)
    dev = torch.device("cuda")
    Hk, Hv, K, V = 8, 24, 128, 128           # one TP2 shard of the 16 / 48-head GDN layer
    lens = [200, 64, 1000]                   # a chunk-aligned request among unaligned ones
    T = sum(lens)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev)
    q = torch.randn(1, T, Hk, K, device=dev)
    k = torch.randn(1, T, Hk, K, device=dev)
    v = torch.randn(1, T, Hv, V, device=dev) * 0.5
    A_log = torch.randn(Hv, device=dev) * 0.5
    dt_bias = torch.linspace(-8.87, 0.56, Hv, device=dev)
    a, b = torch.randn(T, Hv, device=dev), torch.randn(T, Hv, device=dev)
    g, beta = e32.gdn_gating(A_log, dt_bias, a, b)
    pool = torch.randn(8, Hv, V, K, device=dev) * 0.1  # sglang V-first rows
    rows = torch.tensor([5, 1, 3], dtype=torch.int32, device=dev)
    before = pool.clone()
    o, none, h = e32.gdn_extend(q, k, v, g, beta, ssm_states=pool, cache_indices=rows, query_start_loc=cu)
    res = dict(requests=[], untouched_rows_equal=bool(torch.equal(pool[[0, 2, 4, 6, 7]], before[[0, 2, 4, 6, 7]])),
               returns_none=none is None, h_shape=list(h.shape), h_dtype=str(h.dtype))
    ok = res["untouched_rows_equal"] and none is None
    chunk0 = 0
    for i, (s0, n) in enumerate(zip(cu.tolist()[:-1], lens)):
        qi = q[:, s0:s0 + n].repeat_interleave(Hv // Hk, dim=2)
        ki = k[:, s0:s0 + n].repeat_interleave(Hv // Hk, dim=2)
        h0 = before[rows[i]].transpose(-1, -2)[None].contiguous()   # [1, HV, K, V]
        ref_o, ref_s = chunk_gated_delta_rule(qi, ki, v[:, s0:s0 + n], g[:, s0:s0 + n], beta[:, s0:s0 + n],
                                              initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True)
        got_s = pool[rows[i]].transpose(-1, -2)
        tor_o, tor_s = torch_chunk_gdn(qi, ki, v[:, s0:s0 + n], g[:, s0:s0 + n], beta[:, s0:s0 + n], h0)
        # chunk states: h[chunk j] = state before chunk j; chunk 1 of this request = final state of its first 64 tokens
        n_chunks = (n + 63) // 64
        h_first = None
        if n > 64:
            _, s64 = chunk_gated_delta_rule(qi[:, :64], ki[:, :64], v[:, s0:s0 + 64], g[:, s0:s0 + 64],
                                            beta[:, s0:s0 + 64], initial_state=h0, output_final_state=True,
                                            use_qk_l2norm_in_kernel=True)
            h_first = float((h[0, chunk0 + 1].float().transpose(-1, -2) - s64[0]).abs().max())
        row = dict(len=n, state_bitwise=bool(torch.equal(got_s, ref_s[0])),
                   state_max_abs=float((got_s - ref_s[0]).abs().max()),
                   out_bitwise=bool(torch.equal(o[:, s0:s0 + n], ref_o)),
                   out_max_abs=float((o[:, s0:s0 + n] - ref_o).abs().max()),
                   h0_max_abs=float((h[0, chunk0].float().transpose(-1, -2) - h0[0]).abs().max()),
                   h_chunk1_max_abs=h_first,
                   torch_state_rel=float((got_s - tor_s[0]).norm() / tor_s[0].norm()),
                   torch_out_rel=float((o[:, s0:s0 + n] - tor_o).norm() / tor_o.norm()))
        res["requests"].append(row)
        # the same fla kernels in the same order: expect bitwise; h is stored in bf16 (the served tracking dtype)
        ok &= row["state_max_abs"] <= 1e-6 and row["out_max_abs"] <= 1e-5 and row["h0_max_abs"] <= 2e-2 * float(h0.abs().max())
        ok &= row["torch_state_rel"] < 1e-2
        chunk0 += n_chunks
    res["passed"] = bool(ok)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
