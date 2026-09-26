"""#ssmoff-opus / #756: fused gated-RMS output norm in the factored step == step + served layernorm kernel (bitwise).

Reference: the frozen step (no prefetch) followed by the served `_layer_norm_fwd_1pass_kernel` (ROWS 1 or 4,
sigmoid / silu), exactly as the model's RMSNormGated runs it at B1. Candidate: the admitted opus step (prefetched
tiles, fused prefix invalidation) with norm_context. Outputs and all pool state compared bitwise over 10 steps with
expiry cuts, 36 layers. Also times [step + layernorm] vs [fused] in the L2-flushed harness. One JSON line.
"""
import contextlib
import json
import os
import sys

import torch
from sglang.kernels.ops.attention.fla import layernorm_gated
from sglang.srt.layers.attention.linear.kernels.gdn_factored import (
    factored_expiry_truncate_layers,
    factored_packed_decode,
)

DEV = "cuda" if torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1" else "cpu"
STEPS = int(os.environ.get("OPUS_NORM_STEPS", "10"))
if DEV == "cpu":
    # interpreter only: the served launch enters a CUDA device context (no torch.cpu.device); kernel args unchanged
    layernorm_gated.device_context = lambda device: contextlib.nullcontext()
L, S, H, HV, K, V, RMAX, R, RFULL = int(os.environ.get("OPUS_NORM_LAYERS", "36")), 32, 8, 24, 128, 128, 16, 8, 16


def layernorm(x, z, weight, rows, activation):
    """The served RMSNormGated launch (layernorm_fn, 2D rows as in qwen3_5), row block forced to `rows`."""
    served = layernorm_gated.calc_rows_per_block
    layernorm_gated.calc_rows_per_block = lambda M, device: rows
    try:
        out = layernorm_gated.layernorm_fn(x.reshape(-1, x.shape[-1]), weight, None, z=z.reshape(-1, z.shape[-1]),
                                           eps=1e-6, group_size=None, norm_before_gate=True, is_rms_norm=True,
                                           activation=activation)
    finally:
        layernorm_gated.calc_rows_per_block = served
    return out.reshape_as(x)


def pool(gen):
    return dict(a=torch.randn(L, S, HV, K, generator=gen).to(DEV),
                U=torch.randn(L, S, HV, RMAX, K, generator=gen).half().to(DEV),
                W=torch.randn(L, S, HV, RMAX, V, generator=gen).half().to(DEV),
                count=torch.randint(R, RFULL, (L, S, HV), generator=gen, dtype=torch.int32).to(DEV),
                stale=torch.zeros(S, dtype=torch.int32, device=DEV), pv=torch.ones(S, dtype=torch.int32, device=DEV))


def inputs(gen):
    return [dict(mixed=torch.randn(1, 2 * H * K + HV * V, generator=gen).to(torch.bfloat16).to(DEV),
                 ga=torch.randn(1, HV, generator=gen).to(torch.bfloat16).to(DEV),
                 gb=torch.randn(1, HV, generator=gen).to(torch.bfloat16).to(DEV),
                 A_log=torch.randn(HV, generator=gen).to(DEV), dt_bias=torch.randn(HV, generator=gen).to(DEV),
                 vbar=torch.randn(HV, V, generator=gen).to(DEV),
                 z=torch.randn(1, HV, V, generator=gen).to(torch.bfloat16).to(DEV),
                 w=torch.randn(V, generator=gen).to(torch.bfloat16).to(DEV)) for _ in range(L)]


def step(p, l, x, slots, fused, rows, act, extra=None):
    first = l == 0
    if first and not fused:
        p["pv"].index_fill_(0, slots.long().clamp_min(0), 0)
    kw = dict(prefetch_uw=True, prefix_valid=p["pv"] if first else None,
              norm_context=(x["z"], x["w"], 1e-6, rows, act)) if fused else {}
    kw.update(extra or {})
    out = factored_packed_decode(x["mixed"], x["ga"], x["gb"], A_log=x["A_log"], dt_bias=x["dt_bias"],
        scale=K ** -0.5, vbar=x["vbar"], fa=p["a"][l], fu=p["U"][l], fw=p["W"][l], fcount=p["count"][l],
        stale=p["stale"], ssm_state_indices=slots, num_q_heads=H, num_v_heads=HV, head_k_dim=K, head_v_dim=V,
        r=R, rfull=RFULL, truncate=False, post_order=True, kernel="split", **kw)
    return out if fused else layernorm(out, x["z"], x["w"], rows, act)


def check(rows, act, slot, fused_rows=None):
    """fused step with NROWS=fused_rows (default rows) against the served launch with row block `rows`."""
    fused_rows = fused_rows or rows
    gen = torch.Generator().manual_seed(756 + rows)
    base = pool(gen)
    ref, cand = ({k: v.clone() for k, v in base.items()} for _ in range(2))
    bad, n_out, max_diff = [], 0, 0.0
    for t in range(STEPS):
        xs = inputs(gen)
        slots = torch.tensor([slot if t % 4 else -1], dtype=torch.int32, device=DEV)
        for l in range(L):
            o_ref = step(ref, l, xs[l], slots, False, rows, act)
            o_cand = step(cand, l, xs[l], slots, True, fused_rows, act)
            if not torch.equal(o_ref, o_cand):
                bad.append(f"t{t} l{l} out")
                n_out += int((o_ref != o_cand).sum())
                max_diff = max(max_diff, float((o_ref.float() - o_cand.float()).abs().max()))
        for p in (ref, cand):
            factored_expiry_truncate_layers(p["U"], p["W"], p["count"], slots, R, RFULL)
        for k in ref:
            if not torch.equal(ref[k], cand[k]):
                bad.append(f"t{t} {k}")
    return dict(rows=rows, fused_rows=fused_rows, activation=act, slot=slot, bitwise=not bad, first=bad[:3],
                mismatched_layers=len(bad), mismatched_elements=n_out, max_abs_diff=max_diff)


def served_row_invariance(n=400):
    """Is the served launch's per-row result independent of its row block (1 vs 2 vs 4) on the B1 shape [HV, V]?"""
    gen = torch.Generator().manual_seed(4)
    bad = {2: 0, 4: 0}
    for _ in range(n):
        x = torch.randn(1, HV, V, generator=gen).to(torch.bfloat16).to(DEV)
        z = torch.randn(1, HV, V, generator=gen).to(torch.bfloat16).to(DEV)
        w = torch.randn(V, generator=gen).to(torch.bfloat16).to(DEV)
        for act in ("sigmoid", "silu"):
            one = layernorm(x, z, w, 1, act)
            for r in bad:
                bad[r] += int(not torch.equal(one, layernorm(x, z, w, r, act)))
    return dict(samples=2 * n, mismatches_vs_rows1=bad)


def timing():
    flush = torch.empty(64 << 20, dtype=torch.float32, device=DEV)
    gen = torch.Generator().manual_seed(5)
    p = pool(gen); p["count"].fill_(R); xs = inputs(gen)
    slots = torch.tensor([3], dtype=torch.int32, device=DEV)
    graphs = {}

    def build(fn):
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        g.replay(); torch.cuda.synchronize()
        return g
    graphs["flush"] = build(lambda: [flush.fill_(1.0) for _ in range(L)])

    def sep():
        for l in range(L):
            flush.fill_(1.0)
            out = factored_packed_decode(xs[l]["mixed"], xs[l]["ga"], xs[l]["gb"], A_log=xs[l]["A_log"],
                dt_bias=xs[l]["dt_bias"], scale=K ** -0.5, vbar=xs[l]["vbar"], fa=p["a"][l], fu=p["U"][l],
                fw=p["W"][l], fcount=p["count"][l], stale=p["stale"], ssm_state_indices=slots, num_q_heads=H,
                num_v_heads=HV, head_k_dim=K, head_v_dim=V, r=R, rfull=RFULL, truncate=False, post_order=True,
                kernel="split", prefetch_uw=True)
            layernorm(out, xs[l]["z"], xs[l]["w"], 1, "sigmoid")

    def fused(rows=1):
        for l in range(L):
            flush.fill_(1.0)
            factored_packed_decode(xs[l]["mixed"], xs[l]["ga"], xs[l]["gb"], A_log=xs[l]["A_log"],
                dt_bias=xs[l]["dt_bias"], scale=K ** -0.5, vbar=xs[l]["vbar"], fa=p["a"][l], fu=p["U"][l],
                fw=p["W"][l], fcount=p["count"][l], stale=p["stale"], ssm_state_indices=slots, num_q_heads=H,
                num_v_heads=HV, head_k_dim=K, head_v_dim=V, r=R, rfull=RFULL, truncate=False, post_order=True,
                kernel="split", prefetch_uw=True, norm_context=(xs[l]["z"], xs[l]["w"], 1e-6, rows, "sigmoid"))
    graphs["step+layernorm"] = build(sep)
    graphs["fused"] = build(fused)
    graphs["fused_rows4"] = build(lambda: fused(4))
    import statistics
    samples = {k: [] for k in graphs}
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(7):
        for k, g in graphs.items():
            st.record()
            for _ in range(40):
                g.replay()
            en.record(); torch.cuda.synchronize()
            samples[k].append(st.elapsed_time(en) * 1000 / 40)
    base = statistics.median(samples["flush"])
    return {k: round((statistics.median(v) - base) / L, 3) for k, v in samples.items() if k != "flush"}


@torch.inference_mode()
def main():
    grid = [(1, "sigmoid", 2), (4, "silu", 17)] if DEV == "cpu" else \
        [(rows, act, slot) for rows in (1, 4) for act in ("sigmoid", "silu") for slot in (2, 17)]
    cases = [check(*c) for c in grid]
    if DEV == "cuda":
        # served row block 1 (production at B1) against the fused step with a 2- or 4-row broadcast tile
        cases_cross = [check(1, act, slot, fused_rows=fr) for fr in (2, 4) for act in ("sigmoid", "silu") for slot in (2, 17)]
    else:
        cases_cross = []
    passed = all(c["bitwise"] for c in cases)
    res = dict(device=DEV, layers=L, steps=STEPS, passed=passed, cases=cases, cross=cases_cross)
    if DEV == "cuda":
        res["served_row_invariance"] = served_row_invariance()
        res["us_per_layer"] = timing()
    print(json.dumps(res))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
