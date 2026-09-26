"""#ssmoff-opus / #756: fused gated-RMS output norm in the factored step == step + served layernorm kernel (bitwise).

Reference: the frozen step (no prefetch) followed by the served `_layer_norm_fwd_1pass_kernel` (ROWS 1 or 4,
sigmoid / silu), exactly as the model's RMSNormGated runs it at B1. Candidate: the admitted opus step (prefetched
tiles, fused prefix invalidation) with norm_context. Outputs and all pool state compared bitwise over 10 steps with
expiry cuts, 36 layers. Also times [step + layernorm] vs [fused] in the L2-flushed harness. One JSON line.
"""
import json
import sys

import torch
import triton

from sglang.kernels.ops.attention.fla.layernorm_gated import _layer_norm_fwd_1pass_kernel
from sglang.srt.layers.attention.linear.kernels.gdn_factored import (
    factored_expiry_truncate_layers,
    factored_packed_decode,
)

DEV = "cuda"
L, S, H, HV, K, V, RMAX, R, RFULL = 36, 32, 8, 24, 128, 128, 16, 8, 16


def layernorm(x, z, weight, rows, activation):
    value = x.reshape(-1, x.shape[-1]); gate = z.reshape_as(value)
    output = torch.empty_like(value); m, n = value.shape
    rstd = torch.empty(m, dtype=torch.float32, device=DEV)
    _layer_norm_fwd_1pass_kernel[(triton.cdiv(m, rows), 1)](
        value, output, weight, None, gate, None, rstd, n, n, n, 0, 0, m, n, 1e-6,
        BLOCK_N=n, ROWS_PER_BLOCK=rows, HAS_BIAS=False, HAS_Z=True, Z_IS_3D=False, Z_HEADS=1,
        NORM_BEFORE_GATE=True, IS_RMS_NORM=True, ACTIVATION=activation, USE_GDC=True, launch_pdl=True, num_warps=1)
    return output.reshape_as(x)


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


def check(rows, act, slot):
    gen = torch.Generator().manual_seed(756 + rows)
    base = pool(gen)
    ref, cand = ({k: v.clone() for k, v in base.items()} for _ in range(2))
    bad = []
    for t in range(10):
        xs = inputs(gen)
        slots = torch.tensor([slot if t % 4 else -1], dtype=torch.int32, device=DEV)
        for l in range(L):
            if not torch.equal(step(ref, l, xs[l], slots, False, rows, act), step(cand, l, xs[l], slots, True, rows, act)):
                bad.append(f"t{t} l{l} out")
        for p in (ref, cand):
            factored_expiry_truncate_layers(p["U"], p["W"], p["count"], slots, R, RFULL)
        for k in ref:
            if not torch.equal(ref[k], cand[k]):
                bad.append(f"t{t} {k}")
    return dict(rows=rows, activation=act, slot=slot, bitwise=not bad, first=bad[:3])


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

    def fused():
        for l in range(L):
            flush.fill_(1.0)
            factored_packed_decode(xs[l]["mixed"], xs[l]["ga"], xs[l]["gb"], A_log=xs[l]["A_log"],
                dt_bias=xs[l]["dt_bias"], scale=K ** -0.5, vbar=xs[l]["vbar"], fa=p["a"][l], fu=p["U"][l],
                fw=p["W"][l], fcount=p["count"][l], stale=p["stale"], ssm_state_indices=slots, num_q_heads=H,
                num_v_heads=HV, head_k_dim=K, head_v_dim=V, r=R, rfull=RFULL, truncate=False, post_order=True,
                kernel="split", prefetch_uw=True, norm_context=(xs[l]["z"], xs[l]["w"], 1e-6, 1, "sigmoid"))
    graphs["step+layernorm"] = build(sep)
    graphs["fused"] = build(fused)
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


def main():
    cases = [check(rows, act, slot) for rows in (1, 4) for act in ("sigmoid", "silu") for slot in (2, 17)]
    passed = all(c["bitwise"] for c in cases)
    res = dict(passed=passed, cases=cases)
    if passed:
        res["us_per_layer"] = timing()
    print(json.dumps(res))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
