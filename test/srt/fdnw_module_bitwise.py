"""docs/223 §13.A (b) module-level bitwise gate: first decode does not wait for the K4.2 publication.

Built on k42_module_bitwise.py (served-shape FactoredGDNPool, prewarmed whole-prefix k31 batch graph, PDBatchPublication on
the side stream, factored_packed_decode reader), without a server.  Per rep, from identical pool bytes and with the SAME
step inputs x1 / x2 on both paths (teacher-forced: the gate is "after the switch, identical to the factored path"):
  T  today: forward, publication submit + launch, join, step 1 = factored step on every decode row, step 2.
  N  13.A:  forward, publication submit + launch, NO join; step 1 = for the fresh rows (the published slots) the stock packed
            decode (TritonGDNKernel.packed_decode, the S32 kernel) on a scratch copy of S0 = the publication's own input
            states, and the factored step for the other decode rows with the fresh slots masked to -1; then join and the
            catch-up (factored step of the fresh rows from x1); step 2.
Gates (all bitwise):
  G1  scratch == S0 rows, and N's fresh-row step-1 outputs == the stock packed decode recomputed afterwards on a fresh copy
      of S0 (the dense step is exactly the stock kernel on the exact state);
  G2  N's other-row step-1 outputs == T's;
  G3  every pool tensor after step 1 (a, U, W, count, stale, dense_of, prefix_valid, dense_ring, dense_required) and every
      step-2 output: N == T;
  race  N with the publication delayed on its stream (spin) still passes G2 / G3 (the masked step never touches a
      publishing slot, the join orders the catch-up).
Negative controls (each must be detected): n1 one ulp flipped in the scratch -> G1; n2 catch-up skipped -> G3; n3 catch-up
without the join while the publication is delayed -> G3.
  PYTHONPATH=python python3 test/srt/fdnw_module_bitwise.py --heads 24 --batch 1,2 --others 0,2 --reps 3 --out fdnw.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k42_module_bitwise as km  # noqa: E402  (sets the served k31 env before sglang imports)
import torch  # noqa: E402

from sglang.srt.layers.attention.linear.kernels import gdn_factored as gf  # noqa: E402
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel  # noqa: E402
from sglang.srt.mem_cache import gdn_factored_pool as gp  # noqa: E402
from sglang.srt.mem_cache.gdn_pd_publication import CudaPublicationRuntime, PDBatchPublication  # noqa: E402

DEV, K, V, L = km.DEV, km.K, km.V, len(km.GDN_LAYERS)
STOCK = TritonGDNKernel()


def make_inputs(pool, b, others, seed):
    src, plan, controls, _ = km.make_rep(pool, b, False, seed)
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    used = set(int(x) for x in torch.cat([plan.slots, controls[2]]).tolist())
    free = [s for s in torch.randperm(km.SIZE, generator=g, device=DEV).add(1).tolist() if s not in used][:others]
    rows = torch.cat([plan.slots, torch.tensor(free, dtype=torch.long, device=DEV)]).to(torch.int32)
    B, h = rows.numel(), pool.hv
    Hq = h // 3

    def x():
        return dict(mixed=(torch.randn(B, 2 * Hq * K + h * V, generator=g, device=DEV) * 0.5).to(torch.bfloat16),
                    a=torch.randn(B, h, generator=g, device=DEV).to(torch.bfloat16),
                    b=torch.randn(B, h, generator=g, device=DEV).to(torch.bfloat16))

    params = dict(A_log=(torch.rand(L, h, generator=g, device=DEV) * 2 - 4).float(),
                  dt_bias=torch.randn(L, h, generator=g, device=DEV).float())
    return src, plan, controls, rows, x(), x(), params


def factored_step(pool, x, params, idx, l, out=None, row_sel=None):
    mixed, a, b = x["mixed"], x["a"], x["b"]
    if row_sel is not None:
        mixed, a, b = mixed.index_select(0, row_sel), a.index_select(0, row_sel), b.index_select(0, row_sel)
    h = pool.hv
    return gf.factored_packed_decode(
        mixed, a, b, A_log=params["A_log"][l], dt_bias=params["dt_bias"][l], scale=K ** -0.5, vbar=pool.vbar[l],
        fa=pool.a[l], fu=pool.U[l], fw=pool.W[l], fcount=pool.count[l], stale=pool.stale, ssm_state_indices=idx,
        num_q_heads=h // 3, num_v_heads=h, head_k_dim=K, head_v_dim=V, r=pool.cfg.r, rfull=pool.cfg.rfull, out=out,
        truncate=True, **pool.cfg.kernel_kwargs())


def stock_dense(pool, x, params, state_rows, l, nrows):
    """The stock packed decode on a scratch copy; returns ([1, nrows, h, V] output, the scratch before the kernel)."""
    scratch = state_rows.clone()
    before = scratch.clone()
    out = STOCK.packed_decode(mixed_qkv=x["mixed"][:nrows], a=x["a"][:nrows], b=x["b"][:nrows],
                              A_log=params["A_log"][l], dt_bias=params["dt_bias"][l], scale=K ** -0.5,
                              ssm_states=scratch, cache_indices=torch.arange(nrows, dtype=torch.int32, device=DEV),
                              num_v_heads=pool.hv, head_v_dim=V)
    return out, before


def run(kind, pool, graph, src, plan, controls, rows, x1, x2, params, standin, runtime, *, delay=0, perturb=False,
        skip_catchup=False, no_join=False):
    """kind T or N; returns step-1 outputs [L, B, 1, h, V], step-2 outputs, scratch-before list (N), S0 used (N)."""
    eager, policy = gp.factorize_layers, (gp.ORTH_METHOD, gp.ORTH_WARPS_OVERRIDE, gp.factorize_dense)
    live = {k: None if v is None else torch.empty_like(v) for k, v in src.items()}
    torch.cuda.synchronize()
    standin()
    for k, t in live.items():
        if t is not None:
            t.copy_(src[k])
    states = [(live["normal"][l], None) for l in range(L)]
    pub = PDBatchPublication(pool, runtime, agg=True, pool_route=True)
    pool._pd_batch_publication = pub
    if delay:
        with torch.cuda.stream(runtime.stream):
            torch.cuda._sleep(delay)  # the publication starts late: anything racing it shows
    pub.submit(graph, plan, states, controls, eager=eager, policy=policy)
    pub.start_after_forward()
    B, b = rows.numel(), plan.slots.numel()
    out1 = torch.empty(L, B, 1, pool.hv, V, dtype=torch.bfloat16, device=DEV)
    scratch_before = []
    if kind == "T":
        pool.pside_join()
        for l in range(L):
            factored_step(pool, x1, params, rows, l, out=out1[l])
    else:
        masked = rows.clone()
        masked[:b] = -1  # the fresh rows: their slots are being published
        for l in range(L):
            factored_step(pool, x1, params, masked, l, out=out1[l])  # other rows; fresh rows treated as padding
            s0 = live["normal"][l][:b]
            if perturb and l == 0:
                s0 = s0.clone()
                s0.view(torch.int32)[0, 0, 0, 0] ^= 1  # n1: one ulp
            dense, before = stock_dense(pool, x1, params, s0, l, b)
            scratch_before.append(before)
            out1[l, :b] = dense.transpose(0, 1)
        standin()  # the rest of step 1 / the next schedule overlaps the publication
        if not no_join:
            pool.pside_join()
        if not skip_catchup:
            sel = torch.arange(b, device=DEV)
            for l in range(L):
                factored_step(pool, x1, params, rows[:b], l, row_sel=sel)
    out2 = torch.empty_like(out1)
    if kind == "N" and no_join:
        pass  # n3: step 2 still reads; the publication may land later (that is the defect)
    for l in range(L):
        factored_step(pool, x2, params, rows, l, out=out2[l])
    torch.cuda.synchronize()
    pub.join()
    pool._pd_batch_publication = None
    return out1, out2, scratch_before, [live["normal"][l][:b].clone() for l in range(L)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heads", type=int, choices=(24, 48), default=24)
    ap.add_argument("--batch", default="1,2", help="fresh rows per forward (published slots, <= 4)")
    ap.add_argument("--others", default="0,2", help="other decode rows in the same step")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--delay-cycles", type=int, default=50_000_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2231)
    args = ap.parse_args()
    import tempfile

    t_start = time.time()
    pool, graph = km.build_pool(args.heads, tempfile.mkdtemp(prefix="fdnw_bitwise_"))
    standin = km.StandIn(8192, 6)
    runtime = CudaPublicationRuntime(pool.a.device)
    result = dict(meta=dict(argv=vars(args), device=torch.cuda.get_device_name(), cfg=pool.cfg.raw), reps=[])
    ok = True
    # warm-up (JIT) discarded
    init = km.snapshot(pool)
    w = make_inputs(pool, 1, 1, args.seed)
    for kind in ("T", "N"):
        run(kind, pool, graph, *w, standin, runtime)
        km.restore(pool, init)
    for b in [int(v) for v in args.batch.split(",")]:
        for others in [int(v) for v in args.others.split(",")]:
            for rep in range(args.reps):
                seed = args.seed * 1009 + b * 101 + others * 11 + rep
                src, plan, controls, rows, x1, x2, params = make_inputs(pool, b, others, seed)
                init = km.snapshot(pool)
                row = dict(batch=b, others=others, rep=rep, seed=seed)
                o1t, o2t, _, _ = run("T", pool, graph, src, plan, controls, rows, x1, x2, params, standin, runtime)
                ref = km.snapshot(pool)

                def check_n(**kw):
                    km.restore(pool, init)
                    o1n, o2n, before, s0 = run("N", pool, graph, src, plan, controls, rows, x1, x2, params, standin,
                                               runtime, **kw)
                    g1 = all(torch.equal(km.bits(bf), km.bits(s)) for bf, s in zip(before, s0))
                    for l in range(L):  # recompute the stock dense step independently on a fresh copy of S0
                        d, _ = stock_dense(pool, x1, params, src["normal"][l][:b].clone(), l, b)
                        g1 &= torch.equal(km.bits(d.transpose(0, 1)), km.bits(o1n[l, :b]))
                    g2 = torch.equal(km.bits(o1n[:, b:]), km.bits(o1t[:, b:]))
                    pool_rows = {k: km.compare(ref[k], v) for k, v in km.pool_tensors(pool).items()}
                    g3 = all(v["equal"] for v in pool_rows.values()) and torch.equal(km.bits(o2n), km.bits(o2t))
                    return dict(G1=g1, G2=g2, G3=g3, pool_diff=[k for k, v in pool_rows.items() if not v["equal"]],
                                step2_equal=torch.equal(km.bits(o2n), km.bits(o2t)))

                row["N"] = check_n()
                row["N_race"] = check_n(delay=args.delay_cycles)
                row["n1_perturb"] = check_n(perturb=True)
                row["n2_skip_catchup"] = check_n(skip_catchup=True)
                row["n3_no_join"] = check_n(delay=args.delay_cycles, no_join=True)
                row["pass"] = (all(row["N"][g] for g in ("G1", "G2", "G3"))
                               and all(row["N_race"][g] for g in ("G1", "G2", "G3")))
                row["negatives_detected"] = dict(n1=not row["n1_perturb"]["G1"], n2=not row["n2_skip_catchup"]["G3"],
                                                 n3=not row["n3_no_join"]["G3"])
                ok &= row["pass"] and all(row["negatives_detected"].values())
                km.restore(pool, init)
                del init, ref
                torch.cuda.empty_cache()
                print(f"b{b} others{others} rep{rep}: N {row['N']['G1']}/{row['N']['G2']}/{row['N']['G3']} "
                      f"race {row['N_race']['G1']}/{row['N_race']['G2']}/{row['N_race']['G3']} | negatives "
                      f"{row['negatives_detected']} | N pool diff {row['N']['pool_diff']}", flush=True)
                result["reps"].append(row)
                result["pass"] = ok
                with open(args.out, "w") as f:
                    json.dump(result, f, indent=1, default=str)
    result["pass"] = ok
    result["meta"]["wall_s"] = time.time() - t_start
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(f"{'PASS' if ok else 'FAIL'}: {len(result['reps'])} reps -> {args.out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
