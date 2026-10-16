"""K4.1 GPU check: numerics and CUDA-graph micro timing of the fp32 k31 chain (SGLANG_GDN_K31_FP32_CHAIN).

python3 test/srt/k31_fp32_check.py --states <dump.pt|fixture.pt> [...] --out result.json

Inputs (either format, any number of files):
  fixture  {dense (n, 128, 128), vbar (n, 128), omega (n, 128, 24)}            e.g. test/srt/fixtures/k31_real_states_fixture.pt
  dump     {S (b, L, hv, 128, 128) fp32, vbar (L, hv, 128), omega (b, hv, 128, 24), meta}   k31dump-*.pt
Numerics, per input (factorize_prefill_k31 on every head, chunked):
  ref     fp64 CholeskyQR2 + torch fp64 eigh (mixed_cholqr=False, mixed_eigh=False, K31_EIGH=torch)
  served  today's admitted path: mixed CholeskyQR (fp32 x2 + fp64) + mixed eigh (fp32 Jacobi + fp64 Newton)
  stage1  fp32 shifted CholeskyQR3 + today's mixed eigh            (fp32_chain=1)
  stage2  stage 1 + fp32 24-tournament Jacobi, fp32 early exit     (fp32_chain=2)
  per-head residual ||S - stored||_F (stored = vbar a^T + W^T U, fp16 factors) and its ratio to ref; non-finite
  counts; fallback counts (mixed counter for served, [any, pass1, pass2, pass3] for the chain); sweep histogram;
  sink `a` bitwise vs served; worst heads with their (file, b, layer, head) index.
Timing: the 36-layer factorize_layers call captured in a CUDA graph (served / stage1 / stage2), states tiled from
  the inputs' real heads, b in --batches x h in --heads; median of --replays replays after warmup.
Pre-registered (docs/220 K4.1): ratio <= 1.01 on >= 99.9 % of heads and max <= 1.02; non-finite 0; fallbacks 0
  (kill if rate > 0.1 %); a bitwise; sweep median <= 8; micro stage2 h24 b1 > 2.4 ms or h48 b1 > 3.8 ms
  -> root-cause before service (expected 1.6-2.0 / 2.5-3.2 ms).
"""

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("ref", "served", "stage1", "stage2")
TIMED_ARMS = ("served", "stage1", "stage2")
KILL_MS = {24: 2.4, 48: 3.8}
EXPECTED_MS = {24: (1.6, 2.0), 48: (2.5, 3.2)}


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", nargs="+", required=True, help="fixture or k31dump-*.pt files")
    p.add_argument("--out", required=True)
    p.add_argument("--source", default=str(ROOT / "python"), help="sglang python tree to import (default: this one)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    p.add_argument("--chunk", type=int, default=1728, help="heads per numerics call")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--no-timing", action="store_true")
    p.add_argument("--batches", default="1,2,4")
    p.add_argument("--heads", default="24,48")
    p.add_argument("--layers", type=int, default=36)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--replays", type=int, default=50)
    p.add_argument("--worst", type=int, default=10)
    p.add_argument("--stub-packages", action="store_true",
                   help="CPU smoke test without the sglang package deps: import only the leaf modules")
    return p.parse_args()


@contextlib.contextmanager
def env(**values):
    old = {k: os.environ.get(k) for k in values}
    os.environ.update({k: str(v) for k, v in values.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def load_states(path, torch):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "S" in blob:
        s = blob["S"].float()
        b, layers, hv, v, k = s.shape
        dense = s.reshape(b, layers * hv, v, k)
        vbar = blob["vbar"].float().reshape(layers * hv, v)
        omega = blob["omega"].float()[:, None].expand(b, layers, hv, v, omega_m(blob)).reshape(b, layers * hv, v, -1)
        index = [(bi, li, hi) for bi in range(b) for li in range(layers) for hi in range(hv)]
        meta = {k: v for k, v in (blob.get("meta") or {}).items() if isinstance(v, (str, int, float, bool, list))}
        return dict(kind="dump", dense=dense, vbar=vbar, omega=omega, layers=layers, hv=hv, index=index, meta=meta)
    if "dense" in blob:
        dense = blob["dense"].float()[None]
        n = dense.shape[1]
        index = [(0, -1, i) for i in range(n)]
        return dict(kind="fixture", dense=dense, vbar=blob["vbar"].float(), omega=blob["omega"].float()[None],
                    layers=1, hv=n, index=index, meta={"source": blob.get("source")})
    raise ValueError(f"{path}: neither a dump (S, vbar, omega) nor a fixture (dense, vbar, omega)")


def omega_m(blob):
    return blob["omega"].shape[-1]


def quantiles(torch, x):
    x = x.double().cpu()
    if x.numel() == 0:
        return {}
    q = torch.quantile(x, torch.tensor([0.001, 0.5, 0.9, 0.99, 0.999], dtype=torch.float64)).tolist()
    return dict(n=x.numel(), min=x.min().item(), p001=q[0], median=q[1], p90=q[2], p99=q[3], p999=q[4],
                max=x.max().item(), mean=x.mean().item())


def numerics(torch, ref, item, args, device):
    dense_all, vbar_all, omega_all = item["dense"], item["vbar"], item["omega"]
    b, n = dense_all.shape[:2]
    dtype = getattr(torch, args.dtype)
    arms = [a for a in args.arms.split(",") if a]
    res = {a: [] for a in arms}
    rel = {a: [] for a in arms}
    diff_served = {a: [] for a in arms}
    nonfinite = {a: 0 for a in arms}
    a_equal = {a: True for a in arms}
    seconds = {a: 0.0 for a in arms}
    mixed_delta = 0
    chain = {a: [0, 0, 0, 0] for a in arms}
    hist = [0] * 13
    for start in range(0, n, args.chunk):
        stop = min(n, start + args.chunk)
        dense = dense_all[:, start:stop].to(device)
        vbar = vbar_all[start:stop].to(device)
        omega = omega_all[:, start:stop].contiguous().to(device)
        norm = dense.norm(dim=(-2, -1))
        outs = {}
        for arm in arms:
            kw = dict(ref=dict(mixed_cholqr=False, mixed_eigh=False), served=dict(mixed_cholqr=True, mixed_eigh=True),
                      stage1=dict(mixed_eigh=True, fp32_chain=1), stage2=dict(mixed_eigh=True, fp32_chain=2))[arm]
            eigh = "torch" if arm == "ref" else ref.K31_EIGH
            m0 = ref.mixed_cholqr_fallbacks(device) if arm == "served" else 0
            c0 = ref.fp32_chain_cholqr_fallbacks(device)
            h0 = ref.fp32_chain_sweep_histogram(device)
            old = ref.K31_EIGH
            ref.K31_EIGH = eigh
            try:
                sync(torch, device)
                t0 = time.perf_counter()
                out = ref.factorize_prefill_k31(dense, vbar, 16, 32, dtype, omega, **kw)
                sync(torch, device)
                seconds[arm] += time.perf_counter() - t0
            finally:
                ref.K31_EIGH = old
            if arm == "served":
                mixed_delta += ref.mixed_cholqr_fallbacks(device) - m0
            chain[arm] = [x + y - z for x, y, z in zip(chain[arm], ref.fp32_chain_cholqr_fallbacks(device), c0)]
            h1 = ref.fp32_chain_sweep_histogram(device)
            for i, x in enumerate(h1):
                hist[i] += x - (h0[i] if i < len(h0) else 0)
            outs[arm] = out
            nonfinite[arm] += sum(int((~torch.isfinite(t.float())).sum()) for t in out)
        stored = {arm: vbar[None, :, :, None] * out[0][:, :, None, :] + out[2].float().transpose(-1, -2) @ out[1].float()
                  for arm, out in outs.items()}
        for arm in arms:
            r = (dense - stored[arm]).norm(dim=(-2, -1))
            res[arm].append(r.cpu())
            rel[arm].append((r / norm.clamp_min(1e-30)).cpu())
            if "served" in outs:
                a_equal[arm] &= bool(torch.equal(outs[arm][0], outs["served"][0]))
                diff_served[arm].append(((stored[arm] - stored["served"]).norm(dim=(-2, -1))
                                         / norm.clamp_min(1e-30)).cpu())
        del outs, stored, dense
    res = {a: torch.cat(v, 1).flatten() for a, v in res.items()}
    rel = {a: torch.cat(v, 1).flatten() for a, v in rel.items()}
    out = dict(matrices=b * n, seconds_eager=seconds, arms={})
    flat_index = [(bi,) + tuple(item["index"][j][1:]) for bi in range(b) for j in range(n)]
    for arm in arms:
        entry = dict(rel_residual=quantiles(torch, rel[arm]), nonfinite=nonfinite[arm],
                     a_bitwise_vs_served=a_equal[arm] if "served" in arms else None)
        if diff_served[arm]:
            entry["content_diff_vs_served"] = quantiles(torch, torch.cat(diff_served[arm], 1).flatten())
        if "ref" in arms and arm != "ref":
            ratio = res[arm].double() / res["ref"].double().clamp_min(1e-300)
            worst = ratio.argsort(descending=True)[: args.worst].tolist()
            entry["ratio_vs_ref"] = quantiles(torch, ratio)
            entry["ratio_frac_le_1_01"] = (ratio <= 1.01).double().mean().item()
            entry["worst"] = [dict(ratio=ratio[i].item(), rel_residual=rel[arm][i].item(),
                                   ref_rel_residual=rel["ref"][i].item(), batch=flat_index[i % len(flat_index)][0],
                                   layer=flat_index[i % len(flat_index)][1], head=flat_index[i % len(flat_index)][2])
                              for i in worst]
            entry["gate_ratio"] = bool(entry["ratio_frac_le_1_01"] >= 0.999 and ratio.max().item() <= 1.02)
        out["arms"][arm] = entry
    if "served" in arms:
        out["arms"]["served"]["mixed_cholqr_fallbacks"] = mixed_delta
    for arm in ("stage1", "stage2"):
        if arm in arms:
            # A matrix here = one orthonormalisation call's matrix; three calls per head.
            out["arms"][arm]["fp32_chain_fallbacks_any_pass1_pass2_pass3"] = chain[arm]
            out["arms"][arm]["fp32_chain_fallback_rate"] = chain[arm][0] / max(1, 3 * b * n)
    if "stage2" in arms:
        sweeps = torch.repeat_interleave(torch.arange(len(hist)), torch.tensor(hist, dtype=torch.int64))
        out["stage2_sweep_histogram"] = hist
        out["stage2_sweep_median"] = sweeps.median().item() if sweeps.numel() else None
    return out


def sync(torch, device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def tiled(torch, pool_dense, pool_vbar, pool_omega, b, h, layers, device):
    """States for the timing call: real heads tiled to (layers, b, h); layer l head j of row i uses pool head
    (i * layers * h + l * h + j) mod n; the probe uses layer 0's heads."""
    n = pool_dense.shape[0]
    idx = (torch.arange(b * layers * h) % n).reshape(b, layers, h)
    states = [pool_dense[idx[:, l]].contiguous().to(device) for l in range(layers)]   # each (b, h, 128, 128)
    vbar = pool_vbar[idx[0]].to(device)                                                 # (layers, h, 128)
    omega = pool_omega[idx[:, 0]].contiguous().to(device)                              # (b, h, 128, 24)
    return states, vbar, omega


def time_graph(torch, fn, warmup, replays):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            eager = fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    same = all(torch.equal(x, y) for row_e, row_c in zip(eager, captured) for x, y in zip(row_e, row_c))
    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(replays):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    t = torch.tensor(times, dtype=torch.float64)
    del graph
    return dict(median_ms=t.median().item(), p10_ms=torch.quantile(t, 0.1).item(),
                p90_ms=torch.quantile(t, 0.9).item(), min_ms=t.min().item(), replays=replays,
                graph_output_equals_eager=bool(same))


def timing(torch, pool, ref, items, args):
    pool_dense = torch.cat([it["dense"].reshape(-1, 128, 128) for it in items])
    pool_vbar = torch.cat([it["vbar"].repeat(it["dense"].shape[0], 1) for it in items])
    pool_omega = torch.cat([it["omega"].reshape(-1, 128, it["omega"].shape[-1]) for it in items])
    cfg = pool.FactoredGDNConfig(r=16, m=16, dtype=getattr(torch, args.dtype), init_method="k31",
                                 init_oversample=8, decode_method="iter")
    rows = []
    for h in [int(x) for x in args.heads.split(",") if x]:
        for b in [int(x) for x in args.batches.split(",") if x]:
            states, vbar, omega = tiled(torch, pool_dense, pool_vbar, pool_omega, b, h, args.layers, args.device)
            for arm in TIMED_ARMS:
                stage = {"served": 0, "stage1": 1, "stage2": 2}[arm]
                with env(SGLANG_GDN_K31_CHOLQR_MIXED=1, SGLANG_GDN_K31_MIXED_EIGH=1,
                         SGLANG_GDN_K31_MIXED_EIGH_ANY_BATCH=1, SGLANG_GDN_K31_FP32_CHAIN=stage), \
                        mock_dense(pool) as seen:
                    row = time_graph(torch, lambda: pool.factorize_layers(states, vbar, cfg, omega=omega),
                                     args.warmup, args.replays)
                row.update(h=h, b=b, arm=arm, selected=seen[-1] if seen else None)
                if arm == "stage2" and b == 1 and h in KILL_MS:
                    row["kill_line_ms"] = KILL_MS[h]
                    row["expected_ms"] = EXPECTED_MS[h]
                    row["root_cause_before_service"] = row["median_ms"] > KILL_MS[h]
                print(json.dumps(row), flush=True)
                rows.append(row)
            del states
            torch.cuda.empty_cache()
    return rows


@contextlib.contextmanager
def mock_dense(pool):
    """Record which solver factorize_layers selected (fp32_chain, mixed_eigh) without changing the call."""
    real = pool.factorize_dense
    seen = []

    def spy(*a, **kw):
        seen.append(dict(fp32_chain=kw.get("fp32_chain"), mixed_eigh=kw.get("mixed_eigh")))
        return real(*a, **kw)

    pool.factorize_dense = spy
    try:
        yield seen
    finally:
        pool.factorize_dense = real


def main():
    args = parse()
    sys.path.insert(0, args.source)
    import torch

    if args.stub_packages:
        import types
        for name in ("sglang", "sglang.srt", "sglang.srt.duet", "sglang.srt.layers", "sglang.srt.layers.attention",
                     "sglang.srt.layers.attention.linear", "sglang.srt.layers.attention.linear.kernels",
                     "sglang.srt.mem_cache"):
            package = types.ModuleType(name)
            package.__path__ = [str(Path(args.source) / name.replace(".", "/"))]
            sys.modules[name] = package

    from sglang.srt.layers.attention.linear.kernels import gdn_prefill_reference as ref
    from sglang.srt.mem_cache import gdn_factored_pool as pool

    assert Path(ref.__file__).resolve().is_relative_to(Path(args.source).resolve()), ref.__file__
    result = dict(argv=sys.argv, torch=torch.__version__, modules=[ref.__file__, pool.__file__],
                  k31_eigh=ref.K31_EIGH, env={k: v for k, v in os.environ.items() if k.startswith("SGLANG_GDN")})
    try:
        result["source_rev"] = subprocess.run(["git", "-C", args.source, "rev-parse", "HEAD"], capture_output=True,
                                              text=True).stdout.strip()
        result["source_dirty"] = bool(subprocess.run(["git", "-C", args.source, "status", "--porcelain", "--",
                                                      "."], capture_output=True, text=True).stdout.strip())
    except OSError:
        pass
    if torch.device(args.device).type == "cuda":
        if torch.device(args.device).index is None:
            args.device = f"cuda:{torch.cuda.current_device()}"  # the fallback counters key on tensor.device
        result["gpu"] = torch.cuda.get_device_name()
        import triton
        result["triton"] = triton.__version__
    if os.environ.get("SGLANG_GDN_K31_EIGH", "auto") == "torch":
        raise SystemExit("SGLANG_GDN_K31_EIGH=torch would bypass the served Jacobi; unset it")
    items = []
    result["inputs"] = {}
    for path in args.states:
        item = load_states(path, torch)
        items.append(item)
        t0 = time.perf_counter()
        entry = dict(kind=item["kind"], shape=list(item["dense"].shape), meta=item["meta"])
        entry.update(numerics(torch, ref, item, args, args.device))
        entry["wall_s"] = time.perf_counter() - t0
        result["inputs"][path] = entry
        print(json.dumps({path: {a: dict(ratio=e.get("ratio_vs_ref"), nonfinite=e["nonfinite"])
                                 for a, e in entry["arms"].items()}}), flush=True)
    if not args.no_timing:
        result["timing"] = timing(torch, pool, ref, items, args)
    gates = {}
    for arm in ("stage1", "stage2"):
        per = [e["arms"][arm] for e in result["inputs"].values() if arm in e["arms"]]
        if per:
            gates[arm] = dict(ratio=all(p.get("gate_ratio", False) for p in per),
                              nonfinite=all(p["nonfinite"] == 0 for p in per),
                              a_bitwise=all(p["a_bitwise_vs_served"] for p in per))
    fallbacks = [a.get("fp32_chain_fallbacks_any_pass1_pass2_pass3", [0])[0]
                 for e in result["inputs"].values() for a in e["arms"].values()]
    gates["fallbacks_zero"] = sum(fallbacks) == 0
    medians = [e["stage2_sweep_median"] for e in result["inputs"].values() if e.get("stage2_sweep_median") is not None]
    if medians:
        gates["stage2_sweep_median_le_8"] = all(m <= 8 for m in medians)
    if "timing" in result:
        gates["micro_root_cause_before_service"] = any(r.get("root_cause_before_service") for r in result["timing"])
    result["gates"] = gates
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(gates), flush=True)


if __name__ == "__main__":
    main()
