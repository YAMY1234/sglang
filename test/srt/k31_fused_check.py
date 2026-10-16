"""R1 GPU check (docs/223 §9.2): step 1 micro + step 2 module gate for the fused k31 (SGLANG_GDN_K31_FUSED=1).

  --mode micro : graph-replay time of the 36-layer factorize_layers, service (FUSED=0) vs fused (FUSED=1; both with
                 MIXED_EIGH_ANY_BATCH=1 so b >= 2 compares against B), on real
                 prompt-end states, b in --batches, normal (b rows) and tracked (b rows + b tracked chains = 2b),
                 L2 flushed and warm; per-kernel eager breakdown of the fused chain; the admitted fp32 Jacobi alone,
                 idle vs right after an in-situ GEMM burst (lead 21:5x (a)). --warps lists the layout round.
  --mode gate  : every dump row (and tracked chain) as its own b = 1 forward + the 12-head fixture; service vs fused
                 on identical input: a and fallback count bitwise, non-finite 0, content ||stored_f - stored_s|| / ||S||
                 (median <= 3e-5, max <= 2.5e-4), residual ratio vs the fp64 control (CHOLQR_MIXED=0, MIXED_EIGH=0):
                 fraction <= 1.01 and max for both arms, fallback delta.

  PYTHONPATH=python python3 test/srt/k31_fused_check.py --mode gate --dumps 'd/k31dump-r0-*.pt' --fixture --out g.json
"""
import argparse
import glob
import json
import os
import statistics
import sys
import tempfile
import time

os.environ.setdefault("SGLANG_GDN_K31_MIXED_EIGH", "1")
os.environ.setdefault("SGLANG_GDN_K31_CHOLQR_MIXED", "1")
os.environ.pop("SGLANG_GDN_K31_EIGH", None)
os.environ["SGLANG_GDN_K31_MIXED_EIGH_ANY_BATCH"] = "0"
os.environ["SGLANG_GDN_K31_EIGH_GAP_ROUTE"] = "0"
os.environ["SGLANG_GDN_K31_FUSED"] = "0"

import torch  # noqa: E402

from sglang.srt.layers.attention.linear.kernels import gdn_k31_eigh_mixed as mixed  # noqa: E402
from sglang.srt.layers.attention.linear.kernels import gdn_k31_fused as fusedmod  # noqa: E402
from sglang.srt.layers.attention.linear.kernels import gdn_prefill_reference as ref  # noqa: E402
from sglang.srt.mem_cache import gdn_factored_pool as gp  # noqa: E402

DEV = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "k31_real_states_fixture.pt")


def cfg_for(tmp, vbar):
    path = os.path.join(tmp, f"consts-{vbar.shape[1]}.pt")
    torch.save({"vbar": {i: vbar[i] for i in range(vbar.shape[0])}}, path)
    return gp.FactoredGDNConfig.parse(
        f"r=16,m=16,dtype=fp16,ring=16,init_method=k31,decode_method=iter,strict_chunk=1,factored_prefix=1,vbar={path}")


def arm(fused, *, fp64=False):
    # every switch is read from os.environ per call (factorize_layers, envs.EnvBool.get)
    os.environ["SGLANG_GDN_K31_FUSED"] = "1" if fused else "0"
    os.environ["SGLANG_GDN_K31_MIXED_EIGH"] = "0" if fp64 else "1"
    os.environ["SGLANG_GDN_K31_CHOLQR_MIXED"] = "0" if fp64 else "1"


def run(states, vbar, cfg, omega, fused, *, fp64=False):
    arm(fused, fp64=fp64)
    out = gp.factorize_layers(states, vbar, cfg, omega=omega)
    sync()
    arm(False)
    return out


def stored(a, u, w, vbar_l):
    return vbar_l[None, :, :, None] * a.float()[:, :, None, :] + torch.einsum("bhrv,bhrk->bhvk", w.float(), u.float())


def load_events(patterns):
    """-> list of (tag, hv, states (L list of (1, hv, V, K)), vbar (L, hv, V), omega (1, hv, V, 24))."""
    events = []
    for f in sorted(x for p in patterns for x in glob.glob(p)):
        d = torch.load(f, map_location="cpu", weights_only=False)
        chains = [("row%d" % i, d["S"][i:i + 1]) for i in range(d["S"].shape[0])]
        if d.get("S_tracked") is not None:
            chains += [("trk%d" % i, d["S_tracked"][i:i + 1]) for i in range(d["S_tracked"].shape[0])]
        for tag, s in chains:
            events.append((f"{os.path.basename(f)}:{tag}", s.shape[2], s, d["vbar"].float(), d["omega"].float()))
    return events


def gate(a):
    tmp = tempfile.mkdtemp()
    events = load_events(a.dumps)
    if a.max_events:
        events = events[:a.max_events]
    rows, cases = [], []
    counter_dev = DEV
    engaged = False
    for tag, hv, s, vbar, omega in events:
        vbar_c, omega_c = vbar.to(DEV), omega.to(DEV)
        cfg = cfg_for(tmp, vbar)
        states = [s[:, l].to(DEV).contiguous() for l in range(vbar.shape[0])]
        counts, outs = {}, {}
        for name, fused, fp64 in (("service", False, False), ("fused", True, False), ("fp64", False, True)):
            before = ref.mixed_cholqr_fallbacks(counter_dev)
            outs[name] = run(states, vbar_c, cfg, omega_c, fused, fp64=fp64)
            counts[name] = ref.mixed_cholqr_fallbacks(counter_dev) - before
        engaged = engaged or getattr(gp.factorize_layers, "_fused_logged", False)
        per = []
        a_bitwise, nonfinite = True, 0
        for l, S in enumerate(states):
            st = {k: stored(*outs[k][l], vbar_c[l]) for k in outs}
            a_bitwise = a_bitwise and torch.equal(outs["fused"][l][0], outs["service"][l][0])
            nonfinite += sum(int((~torch.isfinite(t.float())).sum()) for t in outs["fused"][l])
            norm = S.flatten(2).norm(dim=-1).clamp_min(1e-30)
            res64 = (S - st["fp64"]).flatten(2).norm(dim=-1).clamp_min(1e-30)
            per.append(torch.stack([(st["fused"] - st["service"]).flatten(2).norm(dim=-1) / norm,
                                    (S - st["service"]).flatten(2).norm(dim=-1) / res64,
                                    (S - st["fused"]).flatten(2).norm(dim=-1) / res64], -1).reshape(-1, 3))
        m = torch.cat(per).cpu()
        rows.append(m)
        case = dict(event=tag, hv=hv, matrices=int(m.shape[0]), a_bitwise=a_bitwise, nonfinite=nonfinite,
                    fallbacks=counts, content_median=float(m[:, 0].median()), content_max=float(m[:, 0].max()),
                    ratio_service_max=float(m[:, 1].max()), ratio_fused_max=float(m[:, 2].max()))
        cases.append(case)
        print(json.dumps(case), flush=True)
    if a.fixture:
        fx = torch.load(FIXTURE, weights_only=False)
        s, vbar, omega = fx["dense"][None].to(DEV), fx["vbar"].to(DEV), fx["omega"][None].to(DEV)
        out = {}
        for name, fused in (("service", False), ("fused", True)):
            if fused and DEV.type == "cuda":
                out[name] = fusedmod.factorize_k31_fused(s, vbar, omega, 12, 16, 16, torch.float16)
            else:  # (CPU smoke: the fused arm repeats the service path)
                out[name] = ref.factorize_prefill_k31(s, vbar, 16, 16, torch.float16, omega, mixed_eigh=True,
                                                      mixed_cholqr=True)
        out["fp64"] = ref.factorize_prefill_k31(s, vbar, 16, 16, torch.float16, omega, mixed_eigh=False,
                                                mixed_cholqr=False)
        st = {k: stored(*v, vbar) for k, v in out.items()}
        norm = s.flatten(2).norm(dim=-1)
        res64 = (s - st["fp64"]).flatten(2).norm(dim=-1)
        m = torch.stack([(st["fused"] - st["service"]).flatten(2).norm(dim=-1) / norm,
                         (s - st["service"]).flatten(2).norm(dim=-1) / res64,
                         (s - st["fused"]).flatten(2).norm(dim=-1) / res64], -1).reshape(-1, 3).cpu()
        rows.append(m)
        case = dict(event="fixture12", matrices=12, a_bitwise=bool(torch.equal(out["fused"][0], out["service"][0])),
                    content_max=float(m[:, 0].max()), ratio_fused_max=float(m[:, 2].max()))
        cases.append(case)
        print(json.dumps(case), flush=True)
    m = torch.cat(rows)
    n = m.shape[0]
    fb_s = sum(c["fallbacks"]["service"] for c in cases if "fallbacks" in c)
    fb_f = sum(c["fallbacks"]["fused"] for c in cases if "fallbacks" in c)
    frac_s = float((m[:, 1] <= 1.01).double().mean())
    frac_f = float((m[:, 2] <= 1.01).double().mean())
    summary = dict(
        matrices=n, events=len(cases), fused_engaged=bool(engaged),
        a_bitwise=all(c["a_bitwise"] for c in cases), nonfinite=sum(c.get("nonfinite", 0) for c in cases),
        fallbacks=dict(service=fb_s, fused=fb_f, count_bitwise=all(
            c["fallbacks"]["service"] == c["fallbacks"]["fused"] for c in cases if "fallbacks" in c),
            delta_fraction=(fb_f - fb_s) / max(1, n)),
        content=dict(median=float(m[:, 0].median()), p99=float(torch.quantile(m[:, 0].double(), .99)),
                     max=float(m[:, 0].max())),
        ratio=dict(service_frac_le_1_01=frac_s, fused_frac_le_1_01=frac_f, service_max=float(m[:, 1].max()),
                   fused_max=float(m[:, 2].max())))
    summary["gate"] = dict(
        a_and_count_bitwise=summary["a_bitwise"] and summary["fallbacks"]["count_bitwise"],
        nonfinite_0=summary["nonfinite"] == 0,
        content=summary["content"]["median"] <= 3e-5 and summary["content"]["max"] <= 2.5e-4,
        ratio_frac=frac_f >= 0.99968, ratio_max=summary["ratio"]["fused_max"] <= 1.069,
        fallback_delta=summary["fallbacks"]["delta_fraction"] <= 1e-3, engaged=bool(engaged))
    summary["pass"] = all(summary["gate"].values())
    return dict(summary=summary, cases=cases)


def graph_time(states, vbar, cfg, omega, fused, reps, flush):
    arm(fused)
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
    out = {}
    for mode in ("warm", "flushed"):
        times = []
        for _ in range(reps):
            if mode == "flushed":
                flush.add_(1)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            graph.replay()
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e))
        out[mode] = statistics.median(times)
    del graph
    return out


def fused_breakdown(states, vbar, h, omega, reps):
    """Eager per-kernel times of the fused chain (CUDA events around each launch)."""
    layers = len(states)
    b = states[0].shape[0]
    dense = torch.stack(states, dim=1).reshape(b, layers * h, 128, 128).contiguous()
    vb = vbar.reshape(layers * h, 128)
    times = {"phase1": [], "jacobi": [], "finalize": []}
    real = {k: getattr(fusedmod, k) for k in ("_k31_phase1", "_fp32_jacobi", "_k31_finalize")}
    names = {"_k31_phase1": "phase1", "_fp32_jacobi": "jacobi", "_k31_finalize": "finalize"}
    events = []

    class Timed:
        def __init__(self, key):
            self.key = key

        def __getitem__(self, grid):
            launch = real[self.key][grid]

            def call(*args, **kw):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                launch(*args, **kw)
                e.record()
                events.append((names[self.key], s, e))
            return call
    for k in real:
        setattr(fusedmod, k, Timed(k))
    try:
        for _ in range(reps + 1):
            events.clear()
            fusedmod.factorize_k31_fused(dense, vb, omega, h, 16, 16, torch.float16)
            torch.cuda.synchronize()
            for name, s, e in events:
                times[name].append(s.elapsed_time(e))
    finally:
        for k, v in real.items():
            setattr(fusedmod, k, v)
    return {k: statistics.median(v[1:]) for k, v in times.items()}


def jacobi_after_burst(n, reps):
    """The admitted fp32 Jacobi alone on n random normalized Grams: idle start vs right after a GEMM burst."""
    g = torch.randn(n, 32, 32, device="cuda", dtype=torch.float64)
    g = g @ g.transpose(-1, -2)
    gn = (g / g.abs().amax(dim=(-2, -1), keepdim=True)).float().contiguous()
    d = torch.empty(n, 32, device="cuda")
    z = torch.empty(n, 32, 32, device="cuda")
    c = torch.empty(n, device="cuda", dtype=torch.int32)
    big = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    out = {}
    for mode in ("idle", "after_gemm"):
        times = []
        for _ in range(reps):
            if mode == "after_gemm":
                for _ in range(10):
                    big @ big
            else:
                torch.cuda.synchronize()
                time.sleep(0.05)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            mixed._fp32_jacobi[(n,)](gn, d, z, c, N=32, SWEEPS=12, EARLY_EXIT=True, num_warps=1)
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e))
        out[mode] = statistics.median(times)
    return out


def micro(a):
    # b >= 2: both arms on the any-batch mixed eigh (service = B, admitted on the BF16 stack), else the fused
    # dispatch (mixed_eigh is True) never engages and both arms would time the same chain
    os.environ["SGLANG_GDN_K31_MIXED_EIGH_ANY_BATCH"] = "1"
    tmp = tempfile.mkdtemp()
    events = load_events(a.dumps)
    normal = [e for e in events if ":row" in e[0]]
    hv = normal[0][1]
    vbar = normal[0][3].to(DEV)
    omega1 = normal[0][4].to(DEV)
    cfg = cfg_for(tmp, normal[0][3])
    flush = torch.empty(512 * 1024 * 1024 // 4, device="cuda")
    result = dict(hv=hv, cases=[], layout=[])
    first_b1 = None
    for warps in a.warps.split(";"):
        os.environ["SGLANG_GDN_K31_FUSED_WARPS"] = warps
        for b in [int(x) for x in a.batches.split(",")]:
            for mode in ("normal", "tracked"):
                rows = b if mode == "normal" else 2 * b
                if rows > len(normal):
                    continue
                grp = normal[:rows]
                states = [torch.cat([e[2][:, l] for e in grp]).float().to(DEV).contiguous()
                          for l in range(vbar.shape[0])]
                omega = omega1.expand(rows, *omega1.shape[1:]).contiguous()
                case = dict(warps=warps, b=b, mode=mode, rows=rows)
                if mode == "normal" or warps == a.warps.split(";")[0]:
                    t0 = time.time()
                    case["fused_ms"] = graph_time(states, vbar, cfg, omega, True, a.reps, flush)
                    case["service_ms"] = graph_time(states, vbar, cfg, omega, False, a.reps, flush)
                    case["breakdown_ms"] = fused_breakdown(states, vbar, hv, omega1, a.reps)
                    case["saving_flushed_ms"] = case["service_ms"]["flushed"] - case["fused_ms"]["flushed"]
                    case["wall_s"] = round(time.time() - t0, 1)
                    result["cases"].append(case)
                    print(json.dumps(case), flush=True)
                    if b == 1 and mode == "normal":
                        first_b1 = case if first_b1 is None or case["fused_ms"]["flushed"] < first_b1[
                            "fused_ms"]["flushed"] else first_b1
    result["jacobi_burst"] = jacobi_after_burst(36 * hv, a.reps)
    print(json.dumps(dict(jacobi_burst=result["jacobi_burst"])), flush=True)
    kill = 2.2 if hv == 24 else 4.0
    result["kill_line_ms"] = kill
    result["best_b1"] = first_b1
    result["killed"] = first_b1 is not None and first_b1["fused_ms"]["flushed"] > kill
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("micro", "gate"), required=True)
    ap.add_argument("--dumps", action="append", default=[])
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--batches", default="1,2,4")
    ap.add_argument("--warps", default="4,4;8,8", help="layout round: phase1,finalize num_warps; ';' separated")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--max-events", type=int, default=0, help="smoke: first N events only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    result = gate(a) if a.mode == "gate" else micro(a)
    result["wall_s"] = round(time.time() - t0, 1)
    result["tf32"] = torch.backends.cuda.matmul.allow_tf32
    result["device"] = str(DEV)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print("DONE", a.out, json.dumps(result.get("summary", {}).get("gate", result.get("killed"))))


if __name__ == "__main__":
    main()
