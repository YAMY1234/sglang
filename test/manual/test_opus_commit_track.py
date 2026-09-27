"""#ssmoff-opus #881: (b) the track-only drain leaves the tracked slots exactly as the queued/stream commit does (the
commit graph into the request slot, then the factored slot copy into the tracked slot), bytewise, for the chunk-end
copy target and the intermediate tracked states, 1 and 2 rows; (a) the fast key lookup replays the same captured
entry with the same result. CUDA (commit graph); on CPU the imports and the drain's signature only. One JSON line.
"""
import json
import sys
import types

import torch

from sglang.srt.mem_cache import gdn_factored_pool as gp
from sglang.srt.mem_cache import gdn_prefill_commit_graph as cg

L, HV, V, K, S = 36, 24, 128, 128, 12
POLICY = (gp.ORTH_METHOD, gp.ORTH_WARPS_OVERRIDE, gp.factorize_dense)


def make(seed=0):
    dev = 'cuda'
    cfg = gp.FactoredGDNConfig(r=8, m=8, dtype=torch.float16, strict_chunk=1, factored_prefix=1)
    g = torch.Generator(device=dev).manual_seed(seed)
    ns = types.SimpleNamespace(
        cfg=cfg, device=torch.device(dev), layer_ids=list(range(L)), spec_state=None,
        a=torch.randn(L, S, HV, K, device=dev, generator=g),
        U=torch.randn(L, S, HV, cfg.rmax, K, device=dev, generator=g).to(cfg.dtype),
        W=torch.randn(L, S, HV, cfg.rmax, V, device=dev, generator=g).to(cfg.dtype),
        count=torch.full((L, S, HV), cfg.r, device=dev, dtype=torch.int32),
        stale=torch.zeros(S, device=dev, dtype=torch.int32), dense_of=torch.arange(S, device=dev, dtype=torch.int32),
        dense_ring=torch.zeros(L, cfg.ring, HV, V, K, device=dev), vbar=torch.randn(L, HV, V, device=dev, generator=g) * .1,
        prefix_valid=torch.zeros(S, device=dev, dtype=torch.int32), dense_required=torch.ones(S, device=dev, dtype=torch.int32),
        prefix_dense=None, batch_prefill_final_copy=True, _commit_side=None, _track_queue=[], _track_slots_cpu=set(),
        _track_hold=False, _queue_holds=[], stats={})
    ns.prefix_layer_count = types.MethodType(gp.FactoredGDNPool.prefix_layer_count, ns)
    ns._drain_track = types.MethodType(gp.FactoredGDNPool._drain_track, ns)
    ns._make_track_job = types.MethodType(gp.FactoredGDNPool._make_track_job, ns)
    return ns


def plan_for(rows, tracked, seed):
    dev = 'cuda'
    g = torch.Generator(device=dev).manual_seed(100 + seed)
    B = len(rows)
    pending = [((torch.randn(B, HV, V, K, device=dev, generator=g) * .02),
                (torch.randn(B, HV, V, K, device=dev, generator=g) * .02) if tracked else None) for _ in range(L)]
    return types.SimpleNamespace(pending=pending, slots=torch.tensor(rows, device=dev, dtype=torch.int32),
                                 ring_dst=torch.tensor([1 + i for i in range(B)], device=dev, dtype=torch.long),
                                 dense_required_after_commit=torch.zeros(B, device=dev, dtype=torch.int32),
                                 opus_slots_cpu=list(rows))


def fields(ns, slot):
    return [x[:, slot].clone() for x in (ns.a, ns.U, ns.W, ns.count)] + \
        [x[slot].clone() for x in (ns.stale, ns.dense_of, ns.dense_required, ns.prefix_valid)]


def same(x, y):
    return all(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8) if a.is_floating_point() else a,
                                                                        b.view(torch.uint8) if b.is_floating_point() else b)
               for a, b in zip(x, y))


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        assert callable(gp.FactoredGDNPool._drain_track) and hasattr(cg, 'OPUS_COMMIT_FAST')
        print(json.dumps(dict(device='cpu', imports=True, passed=True)))
        return
    cases = []
    for seed, (rows, final_src, final_dst, tracked, track_slots) in enumerate((
            ([2], [2], [6], False, None),
            ([2], [2], [6], True, [7]),
            ([2, 3], [3], [8], False, None),
            ([2, 3], [2, 3], [8, 9], True, [10, 11]))):
        dev = 'cuda'
        fs = torch.tensor(final_src, device=dev, dtype=torch.int32)
        fd = torch.tensor(final_dst, device=dev, dtype=torch.int32)
        ts = torch.tensor(track_slots, device=dev, dtype=torch.int32) if track_slots else None
        # reference: the stream commit (captured commit graph on a side stream) then the factored slot copy
        ref = make(seed)
        plan = plan_for(rows, tracked, seed)
        graph = cg.PrefillCommitGraph()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            ok = graph.run(ref, plan, ts, factorize=gp.factorize_layers, policy=POLICY)
            from sglang.srt.layers.attention.linear.kernels.gdn_factored import factored_cow_copy
            factored_cow_copy(ref.a, ref.U, ref.W, ref.count, ref.stale, ref.dense_of, ref.dense_required,
                              ref.prefix_valid, fs.to(torch.long), fd.to(torch.long))
        torch.cuda.current_stream().wait_stream(side)
        # (b): the track-only drain on an identical pool
        new = make(seed)
        plan2 = plan_for(rows, tracked, seed)
        new._track_queue = [new._make_track_job((plan2, L - 1, None, ts, fs, fd))]
        new._drain_track()
        torch.cuda.synchronize()
        targets = list(final_dst) + (list(track_slots) if track_slots else [])
        res = {str(t): same(fields(ref, t), fields(new, t)) for t in targets}
        untouched = all(same(fields(new, s_), fields(make(seed), s_)) for s_ in rows)  # request slots not written
        cases.append(dict(rows=rows, final_dst=final_dst, track_slots=track_slots, graph_ran=bool(ok),
                          tracked_slots_bitwise=res, request_slots_untouched=untouched))
    # (a): fast lookup replays the captured entry; same pool state as the generic lookup
    cg.OPUS_COMMIT_FAST = True
    p1, p2 = make(7), make(7)
    g1, g2 = cg.PrefillCommitGraph(), cg.PrefillCommitGraph()
    for pool, graph in ((p1, g1), (p2, g2)):
        for step in range(3):
            plan = plan_for([4], False, 50 + step)
            if graph is g2:
                graph.fast.clear()  # generic lookup every time
            graph.run(pool, plan, None, factorize=gp.factorize_layers, policy=POLICY)
    torch.cuda.synchronize()
    fast = dict(fast_entries=len(g1.fast), same_state=same(fields(p1, 4), fields(p2, 4)), stats=g1.stats)
    passed = all(all(c['tracked_slots_bitwise'].values()) and c['request_slots_untouched'] and c['graph_ran']
                 for c in cases) and fast['same_state'] and fast['fast_entries'] == 1
    print(json.dumps(dict(cases=cases, fast=fast, passed=passed)))
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
