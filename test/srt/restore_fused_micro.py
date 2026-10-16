"""Fused prefix restore (SGLANG_GDN_PREFIX_RESTORE_FUSED) micro check: bitwise vs the PrefillRestoreGraph, + timing.

Builds a served-shape FactoredGDNPool (same construction as test/srt/k42_module_bitwise.py: R16 m16 -> RMAX 32, fp16
factors, k31 init, strict_chunk + factored_prefix, 36 GDN layers, k = v = 128, 480 + 1 slots, TP1 h 48 or TP2 rank-0
h 24), fills random factors (U ~ orthonormal-row scale, W ~ state scale, fp32 a, vbar, counts in [8, 32] with ~25 %
of heads at the full 32), prewarms the restore graph through pool.prewarm_restore_graph() (switch on, so the fused
kernel is JIT-warmed there too), then for several slots compares
  graph : pool.prefill_restore_graph.run(pool, plan) with the switch off (served today: bind + replay + clone)
  eager : torch.stack([pool._initial_dense_eager(lid, plan) for lid in layers])
  fused : pool.prefill_restore_graph.run(pool, plan) with the switch on (must go through the fused kernel)
  direct: fused_restore(pool, plan.slots)
Gate (exit 0): torch.equal(fused, graph) and torch.equal(direct, graph) for every slot and head count.
Diagnostics: bit-pattern equality, max abs / max ulp, max abs vs an fp64 reference, TF32 flags, timings (CUDA events,
median over --reps): "gpu" = queue pre-filled with torch.cuda._sleep so only device time is measured, "call" =
events around one call from an idle device (includes launch overhead), host = perf_counter per call.

Run in the sglang container on one GPU from the fork worktree root:
  PYTHONPATH=python python3 test/srt/restore_fused_micro.py --heads 24,48 --reps 50 --out restore_fused.json
"""
import argparse
import gc
import json
import os
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace

os.environ["SGLANG_GDN_PREFIX_RESTORE_FUSED"] = "1"  # on during prewarm: exercises the one-time JIT warm

import torch  # noqa: E402
import triton  # noqa: E402

from sglang.srt.layers.attention.linear.kernels.gdn_prefix_restore import fused_restore, restore_bytes  # noqa: E402
from sglang.srt.mem_cache import gdn_factored_pool as gp  # noqa: E402

DEV = "cuda"
K = V = 128
GDN_LAYERS = [i for i in range(48) if i % 4 != 3]  # 36 GDN layers of the 48-layer hybrid
HEADS_TOTAL = 48
SIZE = 480  # pool slots = SIZE + 1 = 481
ENV = "SGLANG_GDN_PREFIX_RESTORE_FUSED"


def build_pool(h, tmpdir, seed):
    """k42_module_bitwise.build_pool construction, without its k31 batch-graph prewarm (not used here)."""
    gen = torch.Generator(device="cpu").manual_seed(4242)
    vbar = {lid: torch.randn(HEADS_TOTAL, V, generator=gen) * 0.1 for lid in GDN_LAYERS}
    path = os.path.join(tmpdir, "consts.pt")
    torch.save({"vbar": vbar}, path)
    cfg = gp.FactoredGDNConfig.parse(
        f"r=16,m=16,dtype=fp16,ring=16,init_method=k31,decode_method=iter,strict_chunk=1,factored_prefix=1,vbar={path}")
    params = SimpleNamespace(shape=SimpleNamespace(temporal=(h, V, K)))
    pool = gp.FactoredGDNPool(size=SIZE, cache_params=params, mamba_layer_ids=list(GDN_LAYERS), device=DEV,
                              cfg=cfg, tp_rank=0)
    g = torch.Generator(device=DEV).manual_seed(seed)
    pool.a.copy_(torch.randn(pool.a.shape, generator=g, device=DEV) * 0.05)
    for t, scale in ((pool.U, K ** -0.5), (pool.W, 0.3)):  # U rows ~ unit norm; W carries the state magnitude
        for l in range(t.shape[0]):
            t[l].copy_(torch.randn(t[l].shape, generator=g, device=DEV) * scale)
    R = pool.U.shape[3]
    count = torch.randint(8, R + 1, pool.count.shape, generator=g, device=DEV, dtype=torch.int32)
    full = torch.rand(pool.count.shape, generator=g, device=DEV) < 0.25
    pool.count.copy_(torch.where(full, torch.full_like(count, R), count))
    pool.vbar.copy_(torch.randn(pool.vbar.shape, generator=g, device=DEV) * 0.1)
    torch.cuda.synchronize()
    return pool


def make_plan(pool, slot):
    slots = torch.tensor([slot], dtype=torch.long, device=DEV)
    return gp.FactoredExtendPlan(
        slots=slots, use_ring=torch.zeros_like(slots, dtype=torch.bool), ring_src=torch.zeros_like(slots),
        ring_dst=torch.full_like(slots, -1), ring_dst_rows=slots[:0], last_layer=len(pool.layer_ids) - 1)


def bits(t):
    return t.contiguous().view(torch.int32)


def ordered(t):
    b = bits(t).long()
    return torch.where(b < 0, -(b & 0x7FFFFFFF), b)


def compare(x, ref):
    d = (x.double() - ref.double()).abs().nan_to_num(nan=float("inf"))
    ulp = (ordered(x) - ordered(ref)).abs()
    return dict(equal=bool(torch.equal(x, ref)), bits_equal=bool(torch.equal(bits(x), bits(ref))),
                max_abs=float(d.max()), max_ulp=int(ulp.max()), n_diff=int((bits(x) != bits(ref)).sum()),
                numel=x.numel())


def fp64_reference(pool, slot):
    s = max(slot, 0)
    out = []
    for li in range(len(pool.layer_ids)):
        R = pool.U.shape[3]
        m = (torch.arange(R, device=DEV)[None, :] < pool.count[li, s][:, None]).double()  # (hv, R)
        U = pool.U[li, s].double() * m[..., None]
        W = pool.W[li, s].double() * m[..., None]
        out.append(torch.einsum("hrv,hrk->hvk", W, U) + pool.vbar[li].double()[:, :, None]
                   * pool.a[li, s].double()[:, None, :])
    return torch.stack(out)[:, None]


def run_graph(pool, plan):
    os.environ[ENV] = "0"
    return pool.prefill_restore_graph.run(pool, plan)


def run_fused(pool, plan):
    os.environ[ENV] = "1"
    return pool.prefill_restore_graph.run(pool, plan)


def run_eager(pool, plan):
    return torch.stack([pool._initial_dense_eager(lid, plan) for lid in pool.layer_ids])


def time_fn(fn, reps, sleep_cycles):
    gpu, call, host = [], [], []
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(sleep_cycles)  # device busy while the host enqueues: events see device time only
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        gpu.append(s.elapsed_time(e))
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        s.record()
        fn()
        e.record()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        call.append(s.elapsed_time(e))
        host.append((t1 - t0) * 1e3)
    return dict(gpu_ms_median=statistics.median(gpu), gpu_ms_min=min(gpu), call_ms_median=statistics.median(call),
                host_ms_median=statistics.median(host))


def check_heads(h, args, tmpdir):
    t0 = time.time()
    pool = build_pool(h, tmpdir, args.seed + h)
    os.environ[ENV] = "1"
    pool.prewarm_restore_graph()
    graph = pool.prefill_restore_graph
    if graph is None or graph.entry is None:
        raise SystemExit(f"h={h}: prewarm_restore_graph did not capture a restore graph")
    res = dict(heads=h, build_s=time.time() - t0, U_shape=list(pool.U.shape), U_dtype=str(pool.U.dtype),
               slots=[], pass_=True)
    g = torch.Generator(device="cpu").manual_seed(args.seed * 7 + h)
    slots = sorted(set(torch.randint(1, SIZE + 1, (args.slots - 1,), generator=g).tolist()) | {SIZE})
    for slot in slots:
        plan = make_plan(pool, slot)
        fused_before = graph.fused
        ref = run_graph(pool, plan)
        if ref is None:
            raise SystemExit(f"h={h} slot={slot}: restore graph fell back (stats {graph.stats})")
        fused = run_fused(pool, plan)
        went_fused = graph.fused == fused_before + 1
        direct = fused_restore(pool, plan.slots)
        eager = run_eager(pool, plan)
        torch.cuda.synchronize()
        exact = fp64_reference(pool, slot)
        counts = pool.count[:, slot]
        row = dict(slot=slot, went_fused=went_fused, count_min=int(counts.min()), count_max=int(counts.max()),
                   n_full=int((counts == pool.U.shape[3]).sum()),
                   fused_vs_graph=compare(fused, ref), direct_vs_graph=compare(direct, ref),
                   eager_vs_graph=compare(eager, ref),
                   graph_vs_fp64_max_abs=float((ref.double() - exact).abs().max()),
                   fused_vs_fp64_max_abs=float((fused.double() - exact).abs().max()),
                   bytes=restore_bytes(pool, counts))
        row["pass"] = went_fused and row["fused_vs_graph"]["equal"] and row["direct_vs_graph"]["equal"]
        res["pass_"] &= row["pass"]
        res["slots"].append(row)
        print(f"h{h} slot{slot}: {'EQUAL' if row['pass'] else 'DIFF'} fused_vs_graph bits="
              f"{row['fused_vs_graph']['bits_equal']} max_ulp={row['fused_vs_graph']['max_ulp']} "
              f"max_abs={row['fused_vs_graph']['max_abs']:.3e} went_fused={went_fused} "
              f"| eager_vs_graph {row['eager_vs_graph']['equal']} | vs fp64 graph="
              f"{row['graph_vs_fp64_max_abs']:.3e} fused={row['fused_vs_fp64_max_abs']:.3e}", flush=True)
        del ref, fused, direct, eager, exact
    plan = make_plan(pool, slots[0])
    res["timing"] = dict(
        graph_replay_clone=time_fn(lambda: run_graph(pool, plan), args.reps, args.sleep_cycles),
        fused_via_run=time_fn(lambda: run_fused(pool, plan), args.reps, args.sleep_cycles),
        fused_kernel=time_fn(lambda: fused_restore(pool, plan.slots), args.reps, args.sleep_cycles),
        eager_stack=time_fn(lambda: run_eager(pool, plan), max(5, args.reps // 5), args.sleep_cycles),
    )
    b = restore_bytes(pool, pool.count[:, slots[0]])
    res["bytes"] = b
    gpu = res["timing"]["fused_kernel"]["gpu_ms_median"]
    res["fused_kernel_GBps_full_r"] = b["total_full_r"] / (gpu * 1e-3) / 1e9 if gpu > 0 else None
    res["graph_stats"] = dict(graph.stats, fused=graph.fused)
    for k, v in res["timing"].items():
        print(f"h{h} {k}: gpu {v['gpu_ms_median']:.4f} ms (min {v['gpu_ms_min']:.4f}) call {v['call_ms_median']:.4f}"
              f" ms host {v['host_ms_median']:.4f} ms", flush=True)
    print(f"h{h} bytes total_full_r={b['total_full_r'] / 1e6:.1f} MB total_counted={b['total_counted'] / 1e6:.1f} MB",
          flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heads", default="24,48", help="comma list of hv (24 = TP2 rank 0, 48 = TP1)")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--slots", type=int, default=6, help="slots checked per head count (incl. the last slot)")
    ap.add_argument("--seed", type=int, default=219)
    ap.add_argument("--sleep-cycles", type=int, default=4_000_000, help="torch.cuda._sleep pre-fill for gpu timing")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    heads = [int(x) for x in args.heads.split(",") if x]
    torch.manual_seed(args.seed)
    tmpdir = tempfile.mkdtemp(prefix="restore_fused_")
    result = dict(meta=dict(argv=vars(args), device=torch.cuda.get_device_name(), torch=torch.__version__,
                            triton=triton.__version__, cuda=torch.version.cuda,
                            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                            float32_matmul_precision=torch.get_float32_matmul_precision(),
                            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                            pool_file=gp.__file__), heads=[])
    ok = True
    for h in heads:
        r = check_heads(h, args, tmpdir)
        gc.collect()  # the pool <-> restore graph entry cycle holds ~7-14 GB of factors
        torch.cuda.empty_cache()
        ok &= r["pass_"]
        result["heads"].append(r)
        result["pass"] = ok
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    result["pass"] = ok
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"{'PASS' if ok else 'FAIL'}: heads {heads}, result {args.out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
