"""#ssmoff-opus decode kernels: bitwise equality with the frozen launches.

1. factored_packed_decode(prefetch_uw=True, prefix_valid=pv) vs the frozen step + pool.invalidate_prefix_dense
   (index_fill_ of slots.long().clamp_min(0)): output, a/U/W/count/stale and prefix_valid bitwise, for several
   steps with the batched expiry cut between steps (counts cross rfull), padded negative slots included.
2. factored_track_copy(prefix_valid=pv, bool mask) vs the frozen copy (mask.to(int32)) + the torch prefix update.
Runs on CUDA, or on CPU with TRITON_INTERPRET=1. Prints one JSON line; exit 1 on any mismatch.
"""
import json
import os
import sys

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_factored import (
    factored_expiry_truncate_layers,
    factored_packed_decode,
    factored_track_copy,
)

DEV = "cuda" if torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1" else "cpu"
# PDL prologue (griddepcontrol) is GPU-only; without a preceding trigger the wait returns once the prior grid is done
GDC = DEV == "cuda" and os.environ.get("OPUS_TEST_GDC", "1") == "1"
L, S, H, HV, K, V, RMAX, R, RFULL = 3, 12, 2, 4, 128, 128, 16, 8, 16


def pool(gen):
    fa = torch.randn(L, S, HV, K, generator=gen).to(DEV)
    fu = torch.randn(L, S, HV, RMAX, K, generator=gen).half().to(DEV)
    fw = torch.randn(L, S, HV, RMAX, V, generator=gen).half().to(DEV)
    cnt = torch.randint(R, RFULL, (L, S, HV), generator=gen, dtype=torch.int32).to(DEV)
    stale = torch.zeros(S, dtype=torch.int32, device=DEV)
    pv = torch.ones(S, dtype=torch.int32, device=DEV)
    return dict(a=fa, U=fu, W=fw, count=cnt, stale=stale, pv=pv)


def clone(p):
    return {k: v.clone() for k, v in p.items()}


def step(p, layer, inputs, slots, opus):
    mixed, ga, gb, A_log, dt_bias, vbar = inputs
    first = layer == 0
    if first and not opus:
        p["pv"].index_fill_(0, slots.long().clamp_min(0), 0)
    return factored_packed_decode(
        mixed, ga, gb, A_log=A_log, dt_bias=dt_bias, scale=K**-0.5, vbar=vbar,
        fa=p["a"][layer], fu=p["U"][layer], fw=p["W"][layer], fcount=p["count"][layer], stale=p["stale"],
        ssm_state_indices=slots, num_q_heads=H, num_v_heads=HV, head_k_dim=K, head_v_dim=V, r=R, rfull=RFULL,
        truncate=False, post_order=True, kernel="split",
        prefix_valid=p["pv"] if (opus and first) else None, prefetch_uw=bool(opus),
        use_gdc=bool(opus) and GDC and MODE > 0, gdc_mode=max(MODE, 1),
        trigger_dependents=bool(opus) and GDC)


MODE = 0


def random_slots(gen, n):
    return torch.randperm(S, generator=gen)[:n].tolist()


def main():
    gen = torch.Generator().manual_seed(121)
    base = pool(gen)
    ref, cand = clone(base), clone(base)
    mism = []
    checks = 0
    for t in range(12):
        B = 3
        # distinct live slots per batch (a slot appears at most once per decode batch), one padded row every 3rd step
        live = random_slots(gen, 3)
        slots = torch.tensor([live[0], live[1], -1 if t % 3 == 0 else live[2]], dtype=torch.int32, device=DEV)
        for layer in range(L):
            mixed = torch.randn(B, 2 * H * K + HV * V, generator=gen).to(torch.bfloat16).to(DEV)
            ga = torch.randn(B, HV, generator=gen).to(torch.bfloat16).to(DEV)
            gb = torch.randn(B, HV, generator=gen).to(torch.bfloat16).to(DEV)
            A_log = torch.randn(HV, generator=gen).to(DEV)
            dt_bias = torch.randn(HV, generator=gen).to(DEV)
            vbar = torch.randn(HV, V, generator=gen).to(DEV)
            inputs = (mixed, ga, gb, A_log, dt_bias, vbar)
            o_ref = step(ref, layer, inputs, slots, False)
            o_cand = step(cand, layer, inputs, slots, True)
            checks += 1
            if not torch.equal(o_ref, o_cand):
                mism.append(f"t{t} l{layer} output")
        for p in (ref, cand):
            factored_expiry_truncate_layers(p["U"], p["W"], p["count"], slots, R, RFULL)
        for k in ref:
            checks += 1
            if not torch.equal(ref[k], cand[k]):
                d = (ref[k].float() - cand[k].float()).abs()
                mism.append(f"t{t} state {k} n={int((d > 0).sum())} max={float(d.max()):.3g}")
        # checkpoint copy of the step's slots into other slots
        src = slots.clone()
        others = [x for x in torch.randperm(S, generator=gen).tolist() if x not in slots.tolist()]
        dst = torch.tensor([others[0], -1, others[1]], dtype=torch.int32, device=DEV)
        mask = torch.tensor([True, t % 2 == 0, t % 4 != 1], device=DEV)
        factored_track_copy(ref["a"], ref["U"], ref["W"], ref["count"], ref["stale"], src, mask, dst)
        d = dst.long().clamp_min(0)
        ref["pv"][d] = torch.where(mask, 0, ref["pv"][d])
        factored_track_copy(cand["a"], cand["U"], cand["W"], cand["count"], cand["stale"], src, mask, dst,
                            prefix_valid=cand["pv"])
        for k in ref:
            checks += 1
            if not torch.equal(ref[k], cand[k]):
                mism.append(f"t{t} copy {k}")
        # re-validate some prefixes so later invalidations are observable
        ref["pv"][t % S] = 1
        cand["pv"][t % S] = 1
    crossed = int((base["count"] != ref["count"]).sum())
    return dict(checks=checks, mismatches=mism, count_changes=crossed, passed=not mism)


def main_all():
    global MODE
    modes = [0] + ([1, 2, 3, 4, 5] if GDC else [])
    res = {}
    for m in modes:
        MODE = m
        r = main()
        res[f"mode{m}"] = dict(passed=r["passed"], n_mismatch=len(r["mismatches"]), first=r["mismatches"][:3],
                               checks=r["checks"], count_changes=r["count_changes"])
    passed = res["mode0"]["passed"]  # the admitted path; GDC modes are reported, used only if they pass too
    out = dict(device=DEV, interpret=os.environ.get("TRITON_INTERPRET") == "1", gdc=GDC, modes=res,
               gdc_passing=[k for k, v in res.items() if k != "mode0" and v["passed"]], passed=passed)
    print(json.dumps(out))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main_all()
