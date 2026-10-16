"""K4.2 module-level bitwise check (docs/223): AGG prompt-end k31 publication on the side stream.

Builds a served-shape FactoredGDNPool (R16 m16 -> RMAX 32, fp16 factors, k31 init, strict_chunk + factored_prefix,
36 GDN layers, k = v = 128, 480 + 1 slots, TP1 h 48 or TP2 rank-0 h 24) without a server, prewarms the whole-prefix
k31 batch graph through pool.prewarm_k31_batch_graph(), and compares per rep, from identical pool bytes:
  F  today: forward stand-in work, prompt-end states written, graph.run on the current stream, post work, reader;
  S  K4.2:  same, but PDBatchPublication(agg, pool_route).submit + start_after_forward (side stream, after the
            "forward" event), post work overlaps it, pool.pside_join() makes the current stream wait, then the reader.
The reader is factored_packed_decode on all 36 layers for the target slots and the final-copy destinations.
Every pool tensor (a, U, W, count, stale, dense_of, prefix_valid, dense_required, dense_ring) and every reader output
is compared bitwise. Exit 0 only if every S-vs-F comparison is equal. Diagnostics (never gate): F-vs-F determinism
baseline; --negative-control = S without pside_join before the reader (and no post work), i.e. a missing join.

Run in the sglang container on one GPU from the fork worktree root:
  PYTHONPATH=python python3 test/srt/k42_module_bitwise.py --heads 24 --batch 1,2,4 --reps 3 --out k42.json
"""
import argparse
import json
import os
import sys
import tempfile
import time
from types import SimpleNamespace

# Served C+ recipe (docs/223 §3.1); set before sglang reads them. The deferred flag stays off: this script installs
# the publication itself instead of the server's ModelRunner hook.
os.environ.setdefault("SGLANG_GDN_PREFILL_FACTOR_GRAPH_K31", "1")
os.environ.setdefault("SGLANG_GDN_PREFILL_FACTOR_GRAPH_K31_MAX_BATCH", "4")
os.environ.setdefault("SGLANG_GDN_K31_MIXED_EIGH", "1")
os.environ.setdefault("SGLANG_GDN_K31_CHOLQR_MIXED", "1")
os.environ.pop("SGLANG_GDN_AGG_BATCH_PUBLISH_DEFERRED", None)
os.environ.pop("SGLANG_GDN_K42_CONTROL", None)

import torch  # noqa: E402

from sglang.srt.layers.attention.linear.kernels import gdn_factored as gf  # noqa: E402
from sglang.srt.mem_cache import gdn_factored_pool as gp  # noqa: E402
from sglang.srt.mem_cache.gdn_pd_publication import CudaPublicationRuntime, PDBatchPublication  # noqa: E402

DEV = "cuda"
K = V = 128
GDN_LAYERS = [i for i in range(48) if i % 4 != 3]  # 36 GDN layers of the 48-layer hybrid
HEADS_TOTAL = 48
SIZE = 480  # pool slots = SIZE + 1 = 481


def build_pool(h, tmpdir):
    gen = torch.Generator(device="cpu").manual_seed(4242)
    vbar = {lid: torch.randn(HEADS_TOTAL, V, generator=gen) * 0.1 for lid in GDN_LAYERS}
    path = os.path.join(tmpdir, "consts.pt")
    torch.save({"vbar": vbar}, path)
    cfg = gp.FactoredGDNConfig.parse(
        f"r=16,m=16,dtype=fp16,ring=16,init_method=k31,decode_method=iter,strict_chunk=1,factored_prefix=1,vbar={path}")
    params = SimpleNamespace(shape=SimpleNamespace(temporal=(h, V, K)))
    pool = gp.FactoredGDNPool(size=SIZE, cache_params=params, mamba_layer_ids=list(GDN_LAYERS), device=DEV,
                              cfg=cfg, tp_rank=0)
    pool.prewarm_k31_batch_graph()
    graph = pool._k31_batch_graph
    if graph is None or not graph.warmed or graph.include_tail:
        raise SystemExit("prewarm_k31_batch_graph did not build a warmed include_tail=False graph "
                         f"(k31_graph_safe={gp.k31_graph_safe(pool.device)}, K31_EIGH={gp.K31_EIGH})")
    # non-trivial bytes everywhere so untouched slots and partial writes are visible
    g = torch.Generator(device=DEV).manual_seed(7)
    pool.a.copy_(torch.randn(pool.a.shape, generator=g, device=DEV) * 0.05)
    for t in (pool.U, pool.W):
        for l in range(t.shape[0]):
            t[l].copy_(torch.randn(t[l].shape, generator=g, device=DEV) * 0.1)
    pool.count.copy_(torch.randint(pool.cfg.r, pool.cfg.rfull, pool.count.shape, generator=g, device=DEV))
    pool.dense_ring.copy_(torch.randn(pool.dense_ring.shape, generator=g, device=DEV) * 0.01)
    torch.cuda.synchronize()
    return pool, graph


def pool_tensors(pool):
    out = dict(a=pool.a, U=pool.U, W=pool.W, count=pool.count, stale=pool.stale, dense_of=pool.dense_of,
               prefix_valid=pool.prefix_valid, dense_ring=pool.dense_ring)
    if pool.dense_required is not None:
        out["dense_required"] = pool.dense_required
    return out


def snapshot(pool):
    return {k: v.clone() for k, v in pool_tensors(pool).items()}


def restore(pool, snap):
    for k, v in pool_tensors(pool).items():
        v.copy_(snap[k])
    torch.cuda.synchronize()


def make_rep(pool, b, tracked, seed):
    """Prompt-end dense states (low rank + noise), plan, controls and decode-reader inputs for one rep."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    h = pool.hv

    def states():
        rank = 24
        sv = 2.0 * 0.8 ** torch.arange(rank, device=DEV, dtype=torch.float32)
        u = torch.randn(len(GDN_LAYERS), b, h, V, rank, generator=g, device=DEV) / rank ** 0.5
        w = torch.randn(len(GDN_LAYERS), b, h, rank, K, generator=g, device=DEV) / K ** 0.5
        x = (u * sv) @ w + 1e-3 * torch.randn(len(GDN_LAYERS), b, h, V, K, generator=g, device=DEV)
        return x.contiguous()

    perm = torch.randperm(SIZE, generator=g, device=DEV)[:3 * b] + 1  # distinct live slots (0 = padding)
    slots, extra, final_dst = perm[:b].long(), perm[b:2 * b].long(), perm[2 * b:].long()
    src = dict(normal=states(), tracked=states() if tracked else None)
    ring_dst = torch.where(torch.arange(b, device=DEV) % 2 == 0, torch.randperm(pool.cfg.ring, generator=g,
                           device=DEV)[:b], torch.full((b,), -1, device=DEV)).long()
    plan = SimpleNamespace(slots=slots, ring_dst=ring_dst, dense_required_after_commit=torch.randint(
        0, 2, (b,), generator=g, device=DEV, dtype=torch.int32))
    controls = (extra if tracked else None, slots.clone(), final_dst)  # track_slots, final_src, final_dst
    read_slots = torch.cat([slots, final_dst]).to(torch.int32)
    B, Hq = read_slots.numel(), h // 3
    reader = dict(
        slots=read_slots,
        mixed=(torch.randn(B, 2 * Hq * K + h * V, generator=g, device=DEV) * 0.5).to(torch.bfloat16),
        a=torch.randn(B, h, generator=g, device=DEV).to(torch.bfloat16),
        b=torch.randn(B, h, generator=g, device=DEV).to(torch.bfloat16),
        A_log=(torch.rand(len(GDN_LAYERS), h, generator=g, device=DEV) * 2 - 4).float(),
        dt_bias=torch.randn(len(GDN_LAYERS), h, generator=g, device=DEV).float(),
    )
    return src, plan, controls, reader


def run_reader(pool, rd):
    L, B, h = len(GDN_LAYERS), rd["slots"].numel(), pool.hv
    out = torch.empty(L, B, 1, h, V, dtype=torch.bfloat16, device=DEV)
    for l in range(L):
        gf.factored_packed_decode(
            rd["mixed"], rd["a"], rd["b"], A_log=rd["A_log"][l], dt_bias=rd["dt_bias"][l], scale=K ** -0.5,
            vbar=pool.vbar[l], fa=pool.a[l], fu=pool.U[l], fw=pool.W[l], fcount=pool.count[l], stale=pool.stale,
            ssm_state_indices=rd["slots"], num_q_heads=h // 3, num_v_heads=h, head_k_dim=K, head_v_dim=V,
            r=pool.cfg.r, rfull=pool.cfg.rfull, out=out[l], truncate=True, **pool.cfg.kernel_kwargs())
    return out


class StandIn:
    """Logits/sampler stand-in on the current stream (large bf16 matmuls), so the side stream truly overlaps."""

    def __init__(self, n, iters):
        g = torch.Generator(device=DEV).manual_seed(3)
        self.x = torch.randn(n, n, generator=g, device=DEV).to(torch.bfloat16)
        self.y = torch.empty_like(self.x)
        self.iters = iters

    def __call__(self):
        for _ in range(self.iters):
            torch.matmul(self.x, self.x, out=self.y)


def variant(kind, pool, graph, src, plan, controls, reader, live, standin, runtime):
    """kind: F (today), S (K4.2 + join), N (K4.2 without the reader join). Returns reader out, elapsed ms."""
    eager = gp.factorize_layers
    policy = (gp.ORTH_METHOD, gp.ORTH_WARPS_OVERRIDE, gp.factorize_dense)
    for t in live.values():  # garbage until the "forward" writes them: catches a side stream that does not wait
        if t is not None:
            t.zero_()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    standin()  # rest of the forward
    for k, t in live.items():
        if t is not None:
            t.copy_(src[k])  # last GDN layer outputs the prompt-end states on the forward stream
    tracked = live["tracked"]
    states = [(live["normal"][l], None if tracked is None else tracked[l]) for l in range(len(GDN_LAYERS))]
    publication = None
    if kind == "F":
        pool._pd_batch_publication = None
        graph.run(pool, plan, states, *controls, eager=eager, policy=policy)
        standin()
    else:
        publication = PDBatchPublication(pool, runtime, agg=True, pool_route=True)
        pool._pd_batch_publication = publication
        publication.submit(graph, plan, states, controls, eager=eager, policy=policy)
        publication.start_after_forward()  # what the ModelRunner.forward hook does after logits
        if kind == "S":
            standin()
            pool.pside_join()  # must make the current stream wait for the publication
            if publication.pending is not None:
                raise RuntimeError("pside_join left the publication pending")
    out = run_reader(pool, reader)
    t1.record()
    torch.cuda.synchronize()
    if publication is not None:
        publication.join()  # N: clear the pending ticket after the device sync
        pool._pd_batch_publication = None
    return out, t0.elapsed_time(t1)


def bits(t):
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def compare(x, y):
    eq = torch.equal(bits(x), bits(y))
    res = dict(equal=eq, max_abs=0.0, n_diff=0, numel=x.numel())
    if not eq:  # chunk over dim 0: U/W are 7 GB at TP1, a full fp64 diff would not fit comfortably
        for xi, yi in zip(x, y) if x.dim() > 1 else ((x, y),):
            d = (xi.double() - yi.double()).abs().nan_to_num(nan=float("inf"))
            res["max_abs"] = max(res["max_abs"], float(d.max()))
            res["n_diff"] += int((bits(xi) != bits(yi)).sum())
    return res


def compare_all(ref_pool, ref_out, pool, out):
    rows = {k: compare(ref_pool[k], v) for k, v in pool_tensors(pool).items()}
    rows["reader_out"] = compare(ref_out, out)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heads", type=int, choices=(24, 48), default=24)
    ap.add_argument("--batch", default="1,2,4", help="comma list of prompt-end rows per forward (<= 4)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=223)
    ap.add_argument("--tracked", action="store_true", help="also publish a tracked (stale=1) checkpoint branch")
    ap.add_argument("--negative-control", action="store_true")
    ap.add_argument("--standin-n", type=int, default=8192)
    ap.add_argument("--standin-iters", type=int, default=6)
    args = ap.parse_args()
    batches = [int(x) for x in args.batch.split(",") if x]
    if any(not 1 <= b <= 4 for b in batches):
        raise SystemExit("--batch values must be in 1..4 (k31 graph MAX_BATCH 4)")
    torch.manual_seed(args.seed)
    tmpdir = tempfile.mkdtemp(prefix="k42_bitwise_")
    t_start = time.time()
    pool, graph = build_pool(args.heads, tmpdir)
    standin = StandIn(args.standin_n, args.standin_iters)
    runtime = CudaPublicationRuntime(pool.a.device)
    result = dict(meta=dict(argv=vars(args), device=torch.cuda.get_device_name(), torch=torch.__version__,
                            pool_file=gp.__file__, orth=(gp.ORTH_METHOD, gp.ORTH_WARPS_OVERRIDE),
                            cfg=pool.cfg.raw, kernel_kwargs={k: str(v) for k, v in pool.cfg.kernel_kwargs().items()},
                            graph_entries=sorted(str(k[:2]) for k in graph.entries),
                            env={k: v for k, v in os.environ.items() if k.startswith("SGLANG_GDN")}),
                  reps=[])
    # warm-up (Triton JIT / autotune of the reader, side-stream creation); discarded
    init = snapshot(pool)
    src, plan, controls, reader = make_rep(pool, batches[0], args.tracked, args.seed)
    live = {k: None if v is None else torch.empty_like(v) for k, v in src.items()}
    for kind in ("F", "S"):
        variant(kind, pool, graph, src, plan, controls, reader, live, standin, runtime)
        restore(pool, init)
    del init, src, live
    ok = True
    print(f"pool + k31 graph prewarm ready in {time.time() - t_start:.0f}s on {result['meta']['device']}", flush=True)
    for b in batches:
        for rep in range(args.reps):
            seed = args.seed * 100003 + b * 1009 + rep
            src, plan, controls, reader = make_rep(pool, b, args.tracked, seed)
            live = {k: None if v is None else torch.empty_like(v) for k, v in src.items()}
            init = snapshot(pool)
            row = dict(batch=b, rep=rep, seed=seed, timing_ms={})
            out_f, row["timing_ms"]["F"] = variant("F", pool, graph, src, plan, controls, reader, live, standin, runtime)
            ref = snapshot(pool)
            restore(pool, init)
            out_f2, row["timing_ms"]["F2"] = variant("F", pool, graph, src, plan, controls, reader, live, standin,
                                                     runtime)
            row["F_vs_F"] = compare_all(ref, out_f, pool, out_f2)
            restore(pool, init)
            out_s, row["timing_ms"]["S"] = variant("S", pool, graph, src, plan, controls, reader, live, standin, runtime)
            row["S_vs_F"] = compare_all(ref, out_f, pool, out_s)
            # the publication must have written something: the target slots differ from the initial bytes
            row["publication_wrote"] = not torch.equal(bits(init["U"][:, plan.slots]), bits(pool.U[:, plan.slots]))
            row["pass"] = all(v["equal"] for v in row["S_vs_F"].values()) and row["publication_wrote"]
            ok &= row["pass"]
            if args.negative_control:
                restore(pool, init)
                out_n, row["timing_ms"]["N"] = variant("N", pool, graph, src, plan, controls, reader, live, standin,
                                                       runtime)
                row["N_vs_F"] = compare_all(ref, out_f, pool, out_n)
                row["negative_control_detected"] = not all(v["equal"] for v in row["N_vs_F"].values())
            restore(pool, init)  # every rep starts from the same pool bytes
            del init, ref
            torch.cuda.empty_cache()
            bad = [k for k, v in row["S_vs_F"].items() if not v["equal"]]
            print(f"b{b} rep{rep}: S_vs_F {'EQUAL' if row['pass'] else 'DIFF ' + ','.join(bad)}"
                  f" | F_vs_F {'equal' if all(v['equal'] for v in row['F_vs_F'].values()) else 'DIFF'}"
                  + (f" | negative control detected={row['negative_control_detected']}"
                     if args.negative_control else "")
                  + " | ms " + " ".join(f"{k}={v:.3f}" for k, v in row["timing_ms"].items()), flush=True)
            result["reps"].append(row)
            result["pass"] = ok
            with open(args.out, "w") as f:
                json.dump(result, f, indent=1)
    result["pass"] = ok
    result["meta"]["wall_s"] = time.time() - t_start
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"{'PASS' if ok else 'FAIL'}: {len(result['reps'])} reps, result {args.out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
