"""#1019 GPU check: the k31 prompt-end truncation inside a CUDA graph.

For the whole-layer batch the commit graph factorises (B, 36 layers x 24 TP-local heads, V = K = 128, r = 8):
  1. cholesky_ex vs the reference's torch.linalg.cholesky on the pipeline's Gram matrices: bitwise;
  2. the fp64 Jacobi eigh vs torch.linalg.eigh on the pipeline's 16 x 16 Grams: eigenvalues, top-8 projector (<= 1e-12
     where the split is well conditioned, else within 64 eps |G| / gap);
  3. factorize_prefill_k31 captured in a CUDA graph and replayed == eager (bitwise), and == the torch-eigh path up to
     fp32 factor rounding (reconstructed state);
  4. timing: the old eager path (torch.linalg.cholesky / eigh with host checks) vs one graph replay, B = 1 and 8.

    python test/manual/test_k31_commit_graph_gpu.py --out result.json
"""
import argparse
import json
import statistics
import time

import torch

from sglang.srt.layers.attention.linear.kernels import gdn_k31_eigh, gdn_prefill_reference as ref
from sglang.srt.layers.attention.linear.kernels.gdn_prefill_reference import K31_POWER, K31_SEED, factorize_prefill_k31

L, HV, V, K, R, M = 36, 24, 128, 128, 8, 16


def old_orth(y):  # 0bb7088b64c: torch.linalg.cholesky (host info check)
    yd = y.double()
    for _ in range(2):
        g = yd.transpose(-1, -2) @ yd
        g = g + (1e-7 * g.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30) * torch.eye(
            g.shape[-1], device=g.device, dtype=g.dtype)
        chol = torch.linalg.cholesky(g)
        yd = torch.linalg.solve_triangular(chol, yd.transpose(-1, -2), upper=False).transpose(-1, -2)
    return yd.to(y.dtype)


def old_factorize(s, vbar, omega):  # 0bb7088b64c factorize_prefill_k31 (torch.linalg.cholesky / eigh)
    s = s.float(); vb = vbar.float()
    a = torch.einsum("bhvk,hv->bhk", s, vb) / vb.square().sum(-1).clamp_min(1e-12)[None, :, None]
    x = (s - vb[None, :, :, None] * a[:, :, None, :]).transpose(-1, -2)
    y = x @ omega.float()
    for _ in range(K31_POWER):
        y = x @ old_orth(x.transpose(-1, -2) @ old_orth(y))
    q = old_orth(y)
    bm = q.transpose(-1, -2) @ x
    g = (bm @ bm.transpose(-1, -2)).double()
    g = g + (1e-7 * g.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30) * torch.eye(M, device=g.device, dtype=g.dtype)
    wr = torch.linalg.eigh(g)[1].float()
    u_ref = q @ wr[..., -R:]
    uts = u_ref.transpose(-1, -2) @ x
    return a, u_ref.transpose(-1, -2), uts, g


def states(b, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # content of rank 24 with a decaying spectrum (a prompt-end GDN state) plus a small full-rank floor
    spec = torch.logspace(0, -2, 24, device="cuda")
    s = (torch.randn(b, L * HV, V, 24, device="cuda", generator=g) * spec) @ torch.randn(b, L * HV, 24, K, device="cuda", generator=g)
    s = s + 1e-3 * torch.randn(b, L * HV, V, K, device="cuda", generator=g)
    vbar = torch.randn(L * HV, V, device="cuda", generator=g)
    omega = torch.randn(1, L * HV, V, M, device="cuda", generator=torch.Generator(device="cuda").manual_seed(K31_SEED)).expand(b, -1, -1, -1).contiguous()
    return s, vbar, omega


def recon(vbar, a, u, w):
    return vbar[None, :, :, None] * a[:, :, None, :] + torch.einsum("bhrv,bhrk->bhvk", w[:, :, :R], u[:, :, :R])


def wall(fn, n=20):
    out = []
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    for _ in range(n):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); out.append(1e3 * (time.perf_counter() - t))
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    res, ok = dict(batches={}), True
    ref.K31_EIGH = "jacobi"
    for b in (1, 8):
        s, vbar, omega = states(b, 7 + b)
        row = {}
        # 1-2: the pipeline's own Gram matrices
        a0, u0, w0, g = old_factorize(s, vbar, omega)
        d_t, z_t = torch.linalg.eigh(g)
        d_j, z_j = gdn_k31_eigh.eigh(g)
        P = lambda z, r: z[..., -r:] @ z[..., -r:].transpose(-1, -2)
        gap = d_t[..., M - R] - d_t[..., M - R - 1]
        proj = (P(z_j, R) - P(z_t, R)).abs().amax((-1, -2))
        bound = torch.clamp(64 * torch.finfo(torch.float64).eps * d_t[..., -1] / gap, min=1e-12)
        row.update(eig_rel=float(((d_j - d_t).abs() / d_t[..., -1:].abs()).max()), proj8_max=float(proj.max()),
                   proj8_within_bound=bool((proj <= bound).all()), proj8_le_1e12=float((proj <= 1e-12).double().mean()),
                   min_rel_gap=float((gap / d_t[..., -1]).min()))
        gg = (torch.randn(b * L * HV, M, 64, device="cuda", dtype=torch.float64))
        gg = gg @ gg.transpose(-1, -2)
        row["cholesky_ex_bitwise"] = bool(torch.equal(torch.linalg.cholesky_ex(gg)[0], torch.linalg.cholesky(gg)))
        # 3: eager (new) vs captured replay, and vs the old torch-eigh path through the reconstructed state
        eager = factorize_prefill_k31(s, vbar, R, 16, torch.float32, omega)
        static = [t.clone() for t in (s, vbar, omega)]
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                factorize_prefill_k31(*static[:2], R, 16, torch.float32, static[2])
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = factorize_prefill_k31(*static[:2], R, 16, torch.float32, static[2])
        graph.replay(); torch.cuda.synchronize()
        row["graph_bitwise_eager"] = all(bool(torch.equal(x, y)) for x, y in zip(outs, eager))
        s_old, s_new = recon(vbar, a0, u0, w0), recon(vbar, *eager)
        row["state_rel_vs_old"] = float((s_old - s_new).abs().max() / s_old.abs().max())
        # 4: timing (median of 20, wall clock with a final synchronize)
        row["ms_old_eager"] = wall(lambda: old_factorize(s, vbar, omega))
        row["ms_new_eager"] = wall(lambda: factorize_prefill_k31(s, vbar, R, 16, torch.float32, omega))
        row["ms_graph_replay"] = wall(graph.replay)
        ok &= row["proj8_within_bound"] and row["cholesky_ex_bitwise"] and row["graph_bitwise_eager"]
        ok &= row["eig_rel"] < 1e-12 and row["state_rel_vs_old"] < 1e-5
        res["batches"][b] = row
        print(b, json.dumps(row), flush=True)
    res["passed"] = bool(ok)
    json.dump(res, open(args.out, "w"), indent=1)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
