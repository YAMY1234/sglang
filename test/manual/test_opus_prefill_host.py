"""#ssmoff-opus prefill host path: SGLANG_GDN_OPUS_PREFILL on vs off give identical pool state and plans.

Drives FactoredGDNPool.reset_slots / copy_slots / plan_extend on a minimal pool object (the real methods, bound to a
namespace holding the pool tensors) through a sequence of resets, copies and extend plans that exercise ring reuse,
ring eviction (owners gather) and the prefix/required validations. CUDA if available, else CPU. One JSON line.
"""
import copy
import json
import random
import sys
import types

import torch

from sglang.srt.mem_cache import gdn_factored_pool as gp

DEV = "cuda" if torch.cuda.is_available() else "cpu"
S, L, HV, RMAX, K, V, RING = 24, 3, 2, 16, 8, 8, 4


def make():
    cfg = types.SimpleNamespace(r=8, strict_chunk=1, ring=RING, factored_prefix=1)
    ns = types.SimpleNamespace(
        cfg=cfg, device=torch.device(DEV), spec_state=None, layer_ids=list(range(L)),
        a=torch.randn(L, S, HV, K, device=DEV), U=torch.randn(L, S, HV, RMAX, K, device=DEV).half(),
        W=torch.randn(L, S, HV, RMAX, V, device=DEV).half(),
        count=torch.randint(8, 16, (L, S, HV), device=DEV, dtype=torch.int32),
        stale=torch.randint(0, 2, (S,), device=DEV, dtype=torch.int32),
        dense_of=torch.full((S,), -1, device=DEV, dtype=torch.int32),
        dense_required=torch.zeros(S, device=DEV, dtype=torch.int32),
        prefix_valid=torch.ones(S, device=DEV, dtype=torch.int32),
        prefix_dense=None, ring_owner=[-1] * RING, ring_lru=list(range(RING)),
        stats=dict(extends=0, rows=0, ring_src=0, ring_miss=0, densified=0))
    for name in ("reset_slots", "copy_slots", "plan_extend", "prefix_layer_count"):
        setattr(ns, name, types.MethodType(getattr(gp.FactoredGDNPool, name), ns))
    return ns


def state(ns):
    t = {k: getattr(ns, k).cpu() for k in ("a", "U", "W", "count", "stale", "dense_of", "dense_required", "prefix_valid")}
    t["ring_owner"] = list(ns.ring_owner)
    t["ring_lru"] = list(ns.ring_lru)
    return t


def plan_fields(plan):
    out = {}
    for k in ("slots", "use_ring", "ring_src", "ring_dst", "ring_dst_rows", "dense_required_after_commit", "use_prefix"):
        v = getattr(plan, k)
        out[k] = None if v is None else (v.cpu(), v.dtype)
    for k in ("n_ring_src", "n_ring_miss", "all_fresh", "next_layer", "last_layer"):
        out[k] = getattr(plan, k)
    return out


def run(flag, seed):
    gp.OPUS_PREFILL = flag
    torch.manual_seed(seed)
    rnd = random.Random(seed)
    ns = make()
    trace = []
    for step in range(40):
        op = rnd.choice(("reset", "copy", "plan", "plan", "plan"))
        if op == "reset":
            idx = torch.tensor(rnd.sample(range(S), 3), device=DEV, dtype=torch.int32)
            ns.reset_slots(idx)
        elif op == "copy":
            # COW copies: checkpoint slots -> newly allocated slots (disjoint), as MambaPool.copy_from issues them
            pick = rnd.sample(range(S), 4)
            src = torch.tensor(pick[:2], device=DEV, dtype=torch.int64)
            dst = torch.tensor(pick[2:], device=DEV, dtype=torch.int64)
            ns.copy_slots(src, dst)
        else:
            B = rnd.choice((1, 2, 3))
            slots = torch.tensor(rnd.sample(range(S), B), device=DEV, dtype=torch.int32)
            ns.prefix_valid.fill_(1)
            ns.dense_required.zero_()
            prompt_final = [rnd.random() < .7 for _ in range(B)]
            prefix = [rnd.choice((0, 64)) for _ in range(B)]
            try:
                plan = ns.plan_extend(slots, [256] * B, prefix_lens=prefix, prompt_final=prompt_final)
                trace.append(("plan", plan_fields(plan)))
            except RuntimeError as e:
                trace.append(("error", str(e)))
        trace.append((op, state(ns)))
    return trace


def same(x, y):
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and x.dtype == y.dtype and torch.equal(x, y)
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    return x == y


def slab_check(side=False):
    """GPU: PrefillSlab.restore == the frozen per-layer _initial_dense_eager (densify graph, ring gather, fresh);
    side=True restores on the side stream and reads after the layer-0 event wait, as initial_dense does."""
    import os
    from sglang.srt.mem_cache.gdn_prefill_slab import PrefillSlab
    os.environ['SGLANG_GDN_OPUS_SLAB_STREAM'] = '1' if side else '0'
    ns = make()
    ns.hv, ns.v, ns.k = HV, V, K
    ns.vbar = torch.randn(L, HV, V, device=DEV)
    ns.dense_ring = torch.randn(L, RING, HV, V, K, device=DEV)
    ns.layer_map = {lid: i for i, lid in enumerate(ns.layer_ids)}
    ns._initial_dense_eager = types.MethodType(gp.FactoredGDNPool._initial_dense_eager, ns)
    slab = PrefillSlab(ns)
    bad, n = [], 0
    for step in range(8):
        slot = torch.tensor([step * 3 % S], device=DEV, dtype=torch.long)
        kind = ("graph", "graph", "ring", "fresh", "graph", "ring", "graph", "fresh")[step]
        plan = gp.FactoredExtendPlan(
            slots=slot, use_ring=torch.tensor([kind == "ring"], device=DEV),
            ring_src=torch.tensor([step % RING], device=DEV, dtype=torch.long),
            ring_dst=torch.tensor([-1], device=DEV, dtype=torch.long),
            ring_dst_rows=torch.empty(0, device=DEV, dtype=torch.long),
            n_ring_src=int(kind == "ring"), all_fresh=kind == "fresh", next_layer=0, last_layer=L - 1)
        ns.U.normal_()  # the graph must read live factors, not captured copies
        if not slab.restore(plan):
            bad.append(f"declined {step}")
            continue
        if getattr(plan, 'opus_slab_event', None) is not None:
            torch.cuda.current_stream().wait_event(plan.opus_slab_event)
        # the next restore must not overwrite the slab before these reads: they are on the main stream, and the
        # side stream waits for the main stream at the next restore
        for li, lid in enumerate(ns.layer_ids):
            n += 1
            if not torch.equal(plan.opus_slab[li], ns._initial_dense_eager(lid, plan)):
                bad.append(f"step{step} {kind} layer{li}")
    return n, bad, slab.stats


def main():
    mism, checks = [], 0
    for seed in range(6):
        a, b = run(False, seed), run(True, seed)
        checks += len(a)
        if not same(a, b):
            mism.append(seed)
    gp.OPUS_PREFILL = False
    res = dict(device=DEV, seeds=6, checks=checks, mismatched_seeds=mism)
    if DEV == "cuda":
        n, bad, stats = slab_check()
        n2, bad2, stats2 = slab_check(side=True)
        res.update(slab_checks=n + n2, slab_mismatches=bad + [f"side {b}" for b in bad2],
                   slab_stats=stats, slab_side_stats=stats2)
        mism = mism + bad + bad2
    res["passed"] = not mism
    print(json.dumps(res))
    sys.exit(0 if not mism else 1)


if __name__ == "__main__":
    main()
