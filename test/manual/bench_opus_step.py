"""#ssmoff-opus: served-shape step-kernel microbenchmark with DRAM-resident state (L2 flushed between layers).

For each variant: bitwise equality with the frozen step over 12 steps x 36 layers (expiry cuts between steps), and
CUDA-graph replay time per layer of [flush 256 MB, step] minus [flush] (state then comes from DRAM, as in the
engine where ~6 GB of weights stream through L2 every decode step). The stock dense packed-decode kernel is timed in
the same harness. Prints one JSON line.
"""
import json
import torch

from sglang.srt.layers.attention.linear.kernels.gdn_factored import (
    factored_expiry_truncate_layers,
    factored_packed_decode,
)

DEV = "cuda"
L, S, H, HV, K, V, RMAX, R, RFULL = 36, 64, 8, 24, 128, 128, 16, 8, 16
VARIANTS = {
    "frozen": dict(prefetch_uw=False),
    "d1": dict(prefetch_uw=True),
    "d1_reorder": dict(prefetch_uw=True, reorder=True),
    "d1_hoist": dict(prefetch_uw=True, hoist_inputs=True),
    "d1_stale1": dict(prefetch_uw=True, stale_once=True),
    "d1_reorder_hoist": dict(prefetch_uw=True, reorder=True, hoist_inputs=True),
    "d1_reorder_stale1": dict(prefetch_uw=True, reorder=True, stale_once=True),
    "d1_reorder_hoist_stale1": dict(prefetch_uw=True, reorder=True, hoist_inputs=True, stale_once=True),
}


def pool(gen):
    return dict(a=torch.randn(L, S, HV, K, generator=gen).to(DEV),
                U=torch.randn(L, S, HV, RMAX, K, generator=gen).half().to(DEV),
                W=torch.randn(L, S, HV, RMAX, V, generator=gen).half().to(DEV),
                count=torch.randint(R, RFULL, (L, S, HV), generator=gen, dtype=torch.int32).to(DEV),
                stale=torch.zeros(S, dtype=torch.int32, device=DEV))


def inputs(gen, B=1):
    return [dict(mixed=torch.randn(B, 2 * H * K + HV * V, generator=gen).to(torch.bfloat16).to(DEV),
                 ga=torch.randn(B, HV, generator=gen).to(torch.bfloat16).to(DEV),
                 gb=torch.randn(B, HV, generator=gen).to(torch.bfloat16).to(DEV),
                 A_log=torch.randn(HV, generator=gen).to(DEV), dt_bias=torch.randn(HV, generator=gen).to(DEV),
                 vbar=torch.randn(HV, V, generator=gen).to(DEV)) for _ in range(L)]


def step(p, l, x, slots, kw):
    return factored_packed_decode(x["mixed"], x["ga"], x["gb"], A_log=x["A_log"], dt_bias=x["dt_bias"],
        scale=K ** -0.5, vbar=x["vbar"], fa=p["a"][l], fu=p["U"][l], fw=p["W"][l], fcount=p["count"][l],
        stale=p["stale"], ssm_state_indices=slots, num_q_heads=H, num_v_heads=HV, head_k_dim=K, head_v_dim=V,
        r=R, rfull=RFULL, truncate=False, post_order=True, kernel="split", **kw)


def bitwise(kw):
    gen = torch.Generator().manual_seed(11)
    base = pool(gen)
    ref, cand = ({k: v.clone() for k, v in base.items()} for _ in range(2))
    bad = []
    for t in range(12):
        xs = inputs(gen)
        slots = torch.tensor([(7 * t + 3) % S], dtype=torch.int32, device=DEV)
        for l in range(L):
            o1 = step(ref, l, xs[l], slots, VARIANTS["frozen"])
            o2 = step(cand, l, xs[l], slots, kw)
            if not torch.equal(o1, o2):
                bad.append(f"t{t} l{l} out")
        for p in (ref, cand):
            factored_expiry_truncate_layers(p["U"], p["W"], p["count"], slots, R, RFULL)
        for k in ref:
            if not torch.equal(ref[k], cand[k]):
                bad.append(f"t{t} {k}")
    return bad


def graph_time(fn, reps=100):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay(); torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(reps):
        g.replay()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) * 1000 / reps


def main():
    flush = torch.empty(64 << 20, dtype=torch.float32, device=DEV)  # 256 MB
    gen = torch.Generator().manual_seed(5)
    p = pool(gen)
    p["count"].fill_(R)
    xs = inputs(gen)
    slots = torch.tensor([3], dtype=torch.int32, device=DEV)
    res = {}
    graphs = {}

    def build(fn):
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        g.replay(); torch.cuda.synchronize()
        return g

    graphs["flush"] = build(lambda: [flush.fill_(1.0) for _ in range(L)])
    for name, kw in VARIANTS.items():
        def run(kw=kw):
            for l in range(L):
                flush.fill_(1.0)
                step(p, l, xs[l], slots, kw)
        graphs[name] = build(run)
        res[name] = dict(bitwise=(not bitwise(kw)) if name != "frozen" else True)
    # stock dense packed decode, same harness (bf16 pool [S, HV, V, K] per layer)
    from sglang.kernels.ops.attention.fla.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
    states = [torch.randn(S, HV, V, K, device=DEV).to(torch.bfloat16) for _ in range(L)]
    outs = [torch.empty(1, 1, HV, V, device=DEV, dtype=torch.bfloat16) for _ in range(L)]

    def stock():
        for l in range(L):
            flush.fill_(1.0)
            fused_recurrent_gated_delta_rule_packed_decode(xs[l]["mixed"], xs[l]["ga"], xs[l]["gb"], xs[l]["A_log"],
                xs[l]["dt_bias"], K ** -0.5, states[l], outs[l], slots, use_qk_l2norm_in_kernel=True)
    graphs["stock"] = build(stock)
    res["stock"] = {}
    # interleaved rounds; per-variant median of (graph - flush-only graph) per layer
    import statistics
    samples = {k: [] for k in graphs}
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for rnd in range(7):
        for k, g in graphs.items():
            st.record()
            for _ in range(40):
                g.replay()
            en.record(); torch.cuda.synchronize()
            samples[k].append(st.elapsed_time(en) * 1000 / 40)
    base = statistics.median(samples["flush"])
    res["flush_only_us_per_layer"] = round(base / L, 3)
    for k in graphs:
        if k == "flush":
            continue
        per = sorted((x - base) / L for x in samples[k])
        res[k].update(us_per_layer_median=round(statistics.median(per), 3), min=round(per[0], 3), max=round(per[-1], 3))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
