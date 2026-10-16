"""B' GPU check (docs/223 §4.3): gap-routed k31 eigh on real prompt-end dumps, arms A / B / B'.

  A  : pre-B served path for b >= 2 (mixed CholQR, fp64 Jacobi eigh)    SGLANG_GDN_K31_MIXED_EIGH_ANY_BATCH=0
  B  : mixed eigh for any b                                              ANY_BATCH=1
  B' : B with near-degenerate-cut matrices on the fp64 Jacobi            ANY_BATCH=1, SGLANG_GDN_K31_EIGH_GAP_ROUTE=1
Inputs: k31dump-*.pt (S (b, L, hv, 128, 128), vbar (L, hv, 128), omega (1, hv, 128, 24)); b-row batches are built by
stacking b single-prompt dumps of the same head count. Per head: content diff ||stored_X - stored_A|| / ||S||,
residual ratio ||S - stored_X|| / ||S - stored_A||, and whether B' equals A bitwise on routed heads / B elsewhere.
Spectral stats (cut gap, l16 / l1) per head for the routed-fraction pre-registration. CUDA-graph replay timing of the
36-layer factorize_layers per arm and b.

  PYTHONPATH=python python3 test/srt/k31_gap_route_check.py --dumps 'dir/k31dump-*.pt' --batches 2,4 --out out.json
"""
import argparse
import glob
import json
import os
import statistics
import sys
import tempfile

os.environ.setdefault("SGLANG_GDN_K31_MIXED_EIGH", "1")
os.environ.setdefault("SGLANG_GDN_K31_CHOLQR_MIXED", "1")
os.environ.pop("SGLANG_GDN_K31_EIGH", None)

import torch  # noqa: E402

from sglang.srt.layers.attention.linear.kernels import gdn_k31_eigh_mixed as mixed  # noqa: E402
from sglang.srt.mem_cache import gdn_factored_pool as gp  # noqa: E402

ARMS = {"A": {"SGLANG_GDN_K31_MIXED_EIGH_ANY_BATCH": "0", "SGLANG_GDN_K31_EIGH_GAP_ROUTE": "0"},
        "B": {"SGLANG_GDN_K31_MIXED_EIGH_ANY_BATCH": "1", "SGLANG_GDN_K31_EIGH_GAP_ROUTE": "0"},
        "Bp": {"SGLANG_GDN_K31_MIXED_EIGH_ANY_BATCH": "1", "SGLANG_GDN_K31_EIGH_GAP_ROUTE": "1"}}


def cfg_for(tmp, vbar):
    path = os.path.join(tmp, "consts.pt")
    torch.save({"vbar": {i: vbar[i] for i in range(vbar.shape[0])}}, path)
    return gp.FactoredGDNConfig.parse(
        f"r=16,m=16,dtype=fp16,ring=16,init_method=k31,decode_method=iter,strict_chunk=1,factored_prefix=1,vbar={path}")


def stored(a, u, w, vbar_l):
    # sglang layout: S[v, k] = vbar[v] a[k] + sum_r W[r, v] U[r, k]
    return vbar_l[None, :, :, None] * a.float()[:, :, None, :] + torch.einsum(
        "bhrv,bhrk->bhvk", w.float(), u.float())


def run_arm(arm, states, vbar, cfg, omega):
    os.environ.update(ARMS[arm])
    out = gp.factorize_layers(states, vbar, cfg, omega=omega)
    torch.cuda.synchronize()
    return out


def graph_time(arm, states, vbar, cfg, omega, reps):
    os.environ.update(ARMS[arm])
    static = [s.clone() for s in states]
    for _ in range(2):
        gp.factorize_layers(static, vbar, cfg, omega=omega)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gp.factorize_layers(static, vbar, cfg, omega=omega)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph, stream=stream):
        gp.factorize_layers(static, vbar, cfg, omega=omega)
    times = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); graph.replay(); e.record(); e.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def spectral(states, vbar):
    out = []
    for l, s in enumerate(states):
        sd, vb = s.double(), vbar[l].double()
        a = torch.einsum("nhvk,hv->nhk", sd, vb) / vb.square().sum(-1).clamp_min(1e-12)[None, :, None]
        c = sd - vb[None, :, :, None] * a[:, :, None, :]
        lam = torch.linalg.svdvals(c.reshape(-1, 128, 128)).square()
        l1 = lam[:, 0].clamp_min(1e-300)
        out.append(torch.stack([(lam[:, 15] - lam[:, 16]) / l1, lam[:, 15] / l1], -1))
    st = torch.cat(out)
    crit = (st[:, 0] < mixed.GAP_ROUTE_GAP) & (st[:, 1] > mixed.GAP_ROUTE_FLOOR)
    return dict(heads=int(st.shape[0]), criterion_fraction=float(crit.float().mean()),
                gap_below_thr=float((st[:, 0] < mixed.GAP_ROUTE_GAP).float().mean()),
                below_floor=float((st[:, 1] < mixed.GAP_ROUTE_FLOOR).float().mean()))


def q(t, p):
    return float(torch.quantile(t.float().cpu(), p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True, action="append")
    ap.add_argument("--batches", default="2,4")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--no-spectral", action="store_true",
                    help="skip the fp64 SVD statistics (slow batched fp64 on GPU); routed fraction still reported")
    ap.add_argument("--max-groups", type=int, default=4, help="b-row groups per batch size")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(f for pattern in a.dumps for f in glob.glob(pattern))
    groups = {}
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        if d["S"].shape[0] != 1:
            continue
        groups.setdefault(d["S"].shape[2], []).append(d)
    tmp = tempfile.mkdtemp()
    result = dict(files=len(files), cases=[])
    ok = True
    for hv, ds in sorted(groups.items()):
        vbar = ds[0]["vbar"].float().cuda()
        omega1 = ds[0]["omega"].float().cuda()
        cfg = cfg_for(tmp, ds[0]["vbar"].float())
        case_spec = None if a.no_spectral else spectral(
            [torch.cat([d["S"][:, l] for d in ds]).cuda() for l in range(vbar.shape[0])], vbar)
        result["cases"].append(dict(hv=hv, prompts=len(ds), spectral=case_spec))
        print(json.dumps(dict(hv=hv, prompts=len(ds), spectral=case_spec)))
        for b in [int(x) for x in a.batches.split(",")]:
            for start in list(range(0, len(ds) - b + 1, b))[:a.max_groups]:
                grp = ds[start:start + b]
                states = [torch.cat([d["S"][:, l] for d in grp]).float().cuda().contiguous()
                          for l in range(vbar.shape[0])]
                omega = omega1.expand(b, *omega1.shape[1:]).contiguous()
                mixed._GAP_ROUTE_STATS.clear()
                outs = {arm: run_arm(arm, states, vbar, cfg, omega) for arm in ARMS}
                routed, overflow = mixed.gap_route_stats("cuda")
                rows = []
                for l, s in enumerate(states):
                    S = s
                    st = {arm: stored(*outs[arm][l], vbar[l]) for arm in ARMS}
                    norm = S.flatten(2).norm(dim=-1).clamp_min(1e-30)
                    resA = (S - st["A"]).flatten(2).norm(dim=-1).clamp_min(1e-30)
                    rows.append(torch.stack([
                        (st["B"] - st["A"]).flatten(2).norm(dim=-1) / norm,
                        (st["Bp"] - st["A"]).flatten(2).norm(dim=-1) / norm,
                        (S - st["B"]).flatten(2).norm(dim=-1) / resA,
                        (S - st["Bp"]).flatten(2).norm(dim=-1) / resA], -1).reshape(-1, 4))
                m = torch.cat(rows)
                case = dict(hv=hv, b=b, group=start, matrices=int(m.shape[0]), routed=routed, overflow=overflow,
                            routed_fraction=routed / max(1, m.shape[0]),
                            content_B_vs_A=dict(median=q(m[:, 0], .5), p99=q(m[:, 0], .99), max=float(m[:, 0].max())),
                            content_Bp_vs_A=dict(median=q(m[:, 1], .5), p99=q(m[:, 1], .99), max=float(m[:, 1].max())),
                            ratio_B=dict(p99=q(m[:, 2], .99), max=float(m[:, 2].max()),
                                         frac_le_1_01=float((m[:, 2] <= 1.01).float().mean())),
                            ratio_Bp=dict(p99=q(m[:, 3], .99), max=float(m[:, 3].max()),
                                          frac_le_1_01=float((m[:, 3] <= 1.01).float().mean())))
                if start == 0:
                    case["graph_ms"] = {arm: graph_time(arm, states, vbar, cfg, omega, a.reps) for arm in ARMS}
                    case["Bp_minus_B_ms"] = case["graph_ms"]["Bp"] - case["graph_ms"]["B"]
                    case["saving_kept"] = ((case["graph_ms"]["A"] - case["graph_ms"]["Bp"])
                                           / max(1e-9, case["graph_ms"]["A"] - case["graph_ms"]["B"]))
                result["cases"].append(case)
                print(json.dumps(case))
                ok = ok and overflow == 0
    result["pass_no_overflow"] = ok
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print("DONE", a.out)


if __name__ == "__main__":
    main()
