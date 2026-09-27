"""#1019 follow-up: GPU time per stage of the k31 prompt-end truncation (B, 36 layers x 24 heads, V = K = 128, r = 8).

    python test/manual/bench_k31_commit_stages.py --out result.json
"""
import argparse
import json

import torch

from sglang.srt.layers.attention.linear.kernels import gdn_k31_eigh
from sglang.srt.layers.attention.linear.kernels.gdn_prefill_reference import K31_POWER

L, HV, V, K, R, M = 36, 24, 128, 128, 8, 16


def timed(name, fn, acc):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); out = fn(); b.record(); b.synchronize()
    acc.setdefault(name, []).append(a.elapsed_time(b))
    return out


def orth(y, acc, tag):
    yd = timed(f"{tag}.cast64", lambda: y.double(), acc)
    for i in range(2):
        g = timed(f"{tag}.gram", lambda: yd.transpose(-1, -2) @ yd, acc)
        g = timed(f"{tag}.jitter", lambda: g + (1e-7 * g.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30)
                  * torch.eye(M, device=g.device, dtype=g.dtype), acc)
        c = timed(f"{tag}.cholesky_ex", lambda: torch.linalg.cholesky_ex(g)[0], acc)
        yd = timed(f"{tag}.solve_triangular", lambda: torch.linalg.solve_triangular(c, yd.transpose(-1, -2), upper=False)
                   .transpose(-1, -2), acc)
    return timed(f"{tag}.cast32", lambda: yd.float(), acc)


def run(s, vbar, omega, acc):
    a = timed("sink", lambda: torch.einsum("bhvk,hv->bhk", s, vbar) / vbar.square().sum(-1).clamp_min(1e-12)[None, :, None], acc)
    x = timed("x", lambda: (s - vbar[None, :, :, None] * a[:, :, None, :]).transpose(-1, -2), acc)
    y = timed("y=x@omega", lambda: x @ omega, acc)
    for _ in range(K31_POWER):
        o1 = orth(y, acc, "orth1")
        t = timed("xT@q", lambda: x.transpose(-1, -2) @ o1, acc)
        o2 = orth(t, acc, "orth2")
        y = timed("x@q2", lambda: x @ o2, acc)
    q = orth(y, acc, "orth3")
    bm = timed("bm=qT@x", lambda: q.transpose(-1, -2) @ x, acc)
    g = timed("gram16", lambda: (bm @ bm.transpose(-1, -2)).double(), acc)
    g = g + (1e-7 * g.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30) * torch.eye(M, device=g.device, dtype=g.dtype)
    wr = timed("jacobi_eigh", lambda: gdn_k31_eigh.eigh(g)[1].float(), acc)
    timed("torch_eigh(ref)", lambda: torch.linalg.eigh(g), acc)
    u = timed("u=q@w", lambda: q @ wr[..., -R:], acc)
    timed("uts", lambda: u.transpose(-1, -2) @ x, acc)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
    res = {}
    for b in (1, 8):
        g = torch.Generator(device="cuda").manual_seed(b)
        s = torch.randn(b, L * HV, V, K, device="cuda", generator=g)
        vbar = torch.randn(L * HV, V, device="cuda", generator=g)
        omega = torch.randn(b, L * HV, V, M, device="cuda", generator=g)
        acc = {}
        for _ in range(3):
            run(s, vbar, omega, {})
        for _ in range(10):
            run(s, vbar, omega, acc)
        per_run = {k: sum(v) / 10 for k, v in acc.items()}  # mean per run of the summed calls of each stage
        res[b] = dict(ms_per_run=per_run, total_ms=sum(t for k, t in per_run.items() if k != "torch_eigh(ref)"))
        print(b, json.dumps(res[b], indent=None), flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
