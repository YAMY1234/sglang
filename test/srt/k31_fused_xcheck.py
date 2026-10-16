"""Per-matrix cross-check of the fused k31 kernels against factorize_prefill_k31 (docs/223 §9.1, lead 22:0x ②).

Chained stage checks: every fused stage output is compared with the reference operation applied to the
*fused* input of that stage, so each stage's own error is isolated; then end to end against the reference
(served = mixed CholQR + mixed eigh). Runs under the Triton CPU interpreter (TRITON_INTERPRET=1, zero GPU)
or on CUDA. Interpreter agreement checks algorithm and branch semantics only; fp64 FMA contraction is left to
the GPU module gate.

  TRITON_INTERPRET=1 SGLANG_GDN_K31_EIGH=jacobi python3 test/srt/k31_fused_xcheck.py --fixture
  TRITON_INTERPRET=1 SGLANG_GDN_K31_EIGH=jacobi python3 test/srt/k31_fused_xcheck.py --dump d.pt --sample 48
"""
import argparse
import importlib
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("SGLANG_GDN_K31_EIGH", "jacobi")  # the reference takes the mixed solver on CPU too

import torch  # noqa: E402

PYTHON = Path(__file__).resolve().parents[2] / "python"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "k31_real_states_fixture.pt"
KERNELS = "sglang.srt.layers.attention.linear.kernels"


def load_modules():
    """Namespace stubs so only the leaf modules execute (no sglang package import)."""
    for name in ("sglang", "sglang.srt", "sglang.srt.duet", "sglang.srt.layers", "sglang.srt.layers.attention",
                 "sglang.srt.layers.attention.linear", KERNELS):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(PYTHON / name.replace(".", "/"))]
            sys.modules[name] = package
    ref = importlib.import_module(f"{KERNELS}.gdn_prefill_reference")
    mixed = importlib.import_module(f"{KERNELS}.gdn_k31_eigh_mixed")
    fused = importlib.import_module(f"{KERNELS}.gdn_k31_fused")
    return ref, mixed, fused


def _rel(a, b):
    """Per-matrix ||a - b|| / ||b|| over the last two dims."""
    a, b = a.double(), b.double()
    return (a - b).flatten(-2).norm(dim=-1) / b.flatten(-2).norm(dim=-1).clamp_min(1e-300)


def _span_diff(qf, qr, y):
    """||(Qf Qf^T - Qr Qr^T) y|| / ||y||: span agreement where y lives (robust to kappa(y))."""
    qf, qr, y = qf.double(), qr.double(), y.double()
    d = qf @ (qf.transpose(-1, -2) @ y) - qr @ (qr.transpose(-1, -2) @ y)
    return d.flatten(-2).norm(dim=-1) / y.flatten(-2).norm(dim=-1).clamp_min(1e-300)


def _orth_err(q):
    q = q.double()
    return (q.transpose(-1, -2) @ q - torch.eye(q.shape[-1], dtype=torch.float64)).flatten(-2).norm(dim=-1)


def _ok_flags(ref, y):
    first, info_first = ref._cholqr_fp32(y.float(), shifted=True)
    second, info_second = ref._cholqr_fp32(first, shifted=False)
    return (info_first == 0) & (info_second == 0) & torch.isfinite(second).all(-1).all(-1)


def _stats(t):
    t = t.double().reshape(-1)
    return dict(median=float(t.median()), max=float(t.max()))


def stored(a, u, w, vb):
    """sglang layout S[v, k] = vbar[v] a[k] + sum_r W[r, v] U[r, k]."""
    return vb[None, :, :, None] * a.float()[:, :, None, :] + torch.einsum("bhrv,bhrk->bhvk", w.float(), u.float())


def xcheck(ref, mixed, fused, s, vbar, omega, heads, r=16, rmax=16, dtype=torch.float16):
    """s (B, LH, 128, 128) fp32, vbar (LH, 128), omega (B or 1, heads, 128, 24). Returns a metrics dict."""
    m = fused.M_DIRECTIONS
    b, lh = s.shape[:2]
    dbg = {}
    count_key = s.device
    before = ref.mixed_cholqr_fallbacks(count_key)
    a_f, u_f, w_f = fused.factorize_k31_fused(s, vbar, omega, heads, r, rmax, dtype, debug=dbg)
    fused_count = ref.mixed_cholqr_fallbacks(count_key) - before
    st = dbg["stages"].reshape(b, lh, 5, 128, fused.M_PAD)[..., :m].double().float()
    y0, o1, z, o2, y2 = (st[:, :, i] for i in range(5))
    q = dbg["q"].reshape(b, lh, 128, fused.M_PAD)
    pad_zero = bool((q[..., m:] == 0).all())
    q = q[..., :m]
    om = omega.expand(b, heads, 128, m).repeat(1, lh // heads, 1, 1)
    vb = vbar.float()
    a = torch.einsum("bhvk,hv->bhk", s, vb) / vb.square().sum(-1).clamp_min(1e-12)[None, :, None]
    x = (s - vb[None, :, :, None] * a[:, :, None, :]).transpose(-1, -2)
    out = dict(matrices=b * lh, a_bitwise=bool(torch.equal(a_f, a)), q_pad_zero=pad_zero, stages={})
    # stage-by-stage, each against the reference op on the fused stage input
    out["stages"]["y0=x@omega"] = _stats(_rel(y0, x @ om))
    flags, ref_count = [], 0
    for name, inp, outp in (("orth1", y0, o1), ("orth2", z, o2), ("orth3", y2, q)):
        r_out = ref._orth_cholqr2(inp, mixed=True)
        ok = _ok_flags(ref, inp)
        ref_count += int((~ok).sum())
        flags.append(ok)
        out["stages"][name] = dict(span=_stats(_span_diff(outp, r_out, inp)), rel=_stats(_rel(outp, r_out)),
                                   orth_fused=_stats(_orth_err(outp)), orth_ref=_stats(_orth_err(r_out)),
                                   fallbacks=int((~ok).sum()))
    out["stages"]["z=x^T@o1"] = _stats(_rel(z, x.transpose(-1, -2) @ o1))
    out["stages"]["y2=x@o2"] = _stats(_rel(y2, x @ o2))
    out["fallback_count"] = dict(fused=int(fused_count), reference_chained=ref_count)
    # small Gram, jitter, pad_below_spectrum, fp32_vectors normalization
    from sglang.srt.duet.state_factor import pad_below_spectrum
    bm = q.transpose(-1, -2) @ x
    g = (bm @ bm.transpose(-1, -2)).double()
    g = g + (1e-7 * g.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30) * torch.eye(m, dtype=g.dtype)
    gp = pad_below_spectrum(g).reshape(-1, 32, 32)
    g64 = dbg["g64"]
    out["stages"]["gram64"] = _stats(_rel(g64, gp))
    scale = g64.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-300)
    out["stages"]["gn_bitwise_given_g64"] = bool(torch.equal(dbg["gn"], (g64 / scale).float()))
    z_ref, _ = mixed.fp32_vectors(g64)
    out["stages"]["jacobi_bitwise_given_g64"] = bool(torch.equal(dbg["z"].double(), z_ref))
    d_ref, v_ref = mixed.correct_once(g64, dbg["z"].double())
    eig = dbg["eig"]
    out["stages"]["newton_vectors"] = _stats(_rel(eig[:, :32], v_ref))
    out["stages"]["newton_values"] = float(((eig[:, 32] - d_ref).abs().amax(-1)
                                            / d_ref.abs().amax(-1).clamp_min(1e-300)).max())
    wr = eig[:, :32][:, :m, -r:].float().reshape(b, lh, m, r)
    u_ref = q @ wr
    uts = u_ref.transpose(-1, -2) @ x
    out["stages"]["U_given_q_wr"] = _stats(_rel(u_f[:, :, :r].float(), u_ref.transpose(-1, -2).to(dtype).float()))
    out["stages"]["W_given_q_wr"] = _stats(_rel(w_f[:, :, :r].float(), uts.to(dtype).float()))
    out["rows_ge_r_zero"] = bool((u_f[:, :, r:] == 0).all() and (w_f[:, :, r:] == 0).all())
    # end to end against the served reference
    a_r, u_r, w_r = ref.factorize_prefill_k31(s, vbar, r, rmax, dtype, om, mixed_eigh=True, mixed_cholqr=True)
    sf, sr = stored(a_f, u_f, w_f, vb), stored(a_r, u_r, w_r, vb)
    norm = s.flatten(2).norm(dim=-1).clamp_min(1e-30)
    content = (sf - sr).flatten(2).norm(dim=-1) / norm
    ratio = (s - sf).flatten(2).norm(dim=-1) / (s - sr).flatten(2).norm(dim=-1).clamp_min(1e-30)
    nonfinite = sum(int((~torch.isfinite(t.float())).sum()) for t in (a_f, u_f, w_f))
    out["end_to_end"] = dict(content=_stats(content), content_p99=float(torch.quantile(content.double(), .99)),
                             ratio_max=float(ratio.max()), ratio_frac_le_1_01=float((ratio <= 1.01).double().mean()),
                             a_bitwise=bool(torch.equal(a_f, a_r)), nonfinite=nonfinite)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--dump", action="append", default=[])
    ap.add_argument("--sample", type=int, default=48, help="matrices sampled per dump (seeded)")
    ap.add_argument("--out")
    args = ap.parse_args()
    ref, mixed, fused = load_modules()
    results = []
    if args.fixture:
        fx = torch.load(FIXTURE, weights_only=False)
        results.append(dict(source="fixture", **xcheck(ref, mixed, fused, fx["dense"][None], fx["vbar"],
                                                       fx["omega"][None], 12)))
    for path in args.dump:
        d = torch.load(path, map_location="cpu", weights_only=False)
        s = d["S"][:1].float()                              # (1, L, hv, V, K)
        layers, hv = s.shape[1:3]
        pick = torch.randperm(layers * hv, generator=torch.Generator().manual_seed(7))[:args.sample].sort().values
        dense = s.reshape(1, layers * hv, 128, 128)[:, pick]
        vbar = d["vbar"].float().reshape(layers * hv, 128)[pick]
        omega = d["omega"].float().reshape(1, hv, 128, -1)[:, pick % hv]
        # sampled heads each carry their own omega row: present them as one layer of `sample` heads
        results.append(dict(source=os.path.basename(path), **xcheck(ref, mixed, fused, dense, vbar, omega,
                                                                    len(pick))))
    for res in results:
        print(json.dumps(res))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
