"""Plan-local layer views: byte identity, slice isolation and host dispatch cost."""
import json
import os
import statistics
import time
from types import MethodType, SimpleNamespace

import torch
from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNPool, FactoredExtendPlan

DEVICE = os.environ.get('REPLAY_TEST_DEVICE', 'cpu')
os.environ['SGLANG_GDN_PREFILL_RING_LAYERS'] = '1'
os.environ['SGLANG_GDN_PREFILL_INITIAL_LAYERS_GRAPH'] = '1'
torch.manual_seed(297)


def pool(layers=3):
    heads, width, slots = 2, 8, 5
    p = SimpleNamespace(device=DEVICE, layer_ids=[i*3 for i in range(layers)],
        layer_map={i*3:i for i in range(layers)}, batch_prefill=True,
        hv=heads, v=width, k=width, prefill_reuse=True, prefix_dense=None,
        stats={'densified':0}, _ring_layers_logged=True)
    p.dense_ring = torch.randn(layers, 4, heads, width, width, device=DEVICE)
    p.a = torch.randn(layers, slots, heads, width, device=DEVICE)
    p.U = torch.randn(layers, slots, heads, 16, width, device=DEVICE)
    p.W = torch.randn_like(p.U)
    p.count = torch.full((layers, slots, heads), 8, dtype=torch.int32, device=DEVICE)
    p.vbar = torch.randn(layers, heads, width, device=DEVICE)
    p._initial_dense_eager = MethodType(FactoredGDNPool._initial_dense_eager, p)
    return p


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def run(p, kind, indices, enabled, first=0, last=None):
    os.environ['SGLANG_GDN_PREFILL_INITIAL_VIEWS'] = str(int(enabled))
    last = len(p.layer_ids)-1 if last is None else last
    slots = torch.tensor(indices, dtype=torch.long, device=DEVICE)
    plan = FactoredExtendPlan(slots=slots, ring_src=slots,
        ring_dst=torch.full_like(slots, -1), ring_dst_rows=slots[:0],
        use_ring=torch.full_like(slots, kind == 'ring', dtype=torch.bool),
        all_fresh=kind == 'fresh', n_ring_src=len(indices) if kind == 'ring' else 0,
        next_layer=first, last_layer=last)
    outputs=[]
    for layer in range(first,last+1):
        outputs.append(FactoredGDNPool.initial_dense(p,p.layer_ids[layer],plan))
        plan.next_layer=layer+1
    return outputs, plan


cases=[]
p=pool()
for kind in ('fresh','ring','factor'):
    for indices,first,last in (([0],0,2),([3,1],0,2),([2,2],0,2),
                               ([],0,2),([0],1,2),([0],0,1)):
        before=p.dense_ring.clone()
        count_before=p.stats['densified']
        old,_=run(p,kind,indices,False,first,last)
        count_old=p.stats['densified']-count_before
        count_before=p.stats['densified']
        new,plan=run(p,kind,indices,True,first,last)
        assert p.stats['densified']-count_before==count_old, 'densification accounting changed'
        assert all(same(a,b) for a,b in zip(old,new)), (kind,indices,first,last)
        assert same(p.dense_ring,before)
        if new:
            new[0].add_(1)
            assert same(p.dense_ring,before)
            assert all(same(a,b) for a,b in zip(old[1:],new[1:]))
            again,_=run(p,kind,indices,True,first,last)
            assert all(same(a,b) for a,b in zip(old,again))
        cases.append(dict(kind=kind,indices=indices,first=first,last=last,
                          cached=hasattr(plan,'_initial_views'),bitwise=True,
                          pool_unchanged=True,private_layers=True,private_plans=True))
# An all-layer graph owns its replay output; cached views still belong to the
# per-plan clone, including while a later plan with other slots replays it.
p=pool()
old,plan=run(p,'factor',[0],True)
snapshots=[x.clone() for x in old]
run(p,'factor',[1],True)
assert all(same(a,b) for a,b in zip(old,snapshots))

# CPU wall time is explicitly a host dispatch measurement, not GPU savings.
p=pool(36)
def bench(enabled):
    for _ in range(10):run(p,'ring',[0],enabled)
    if DEVICE=='cuda':torch.cuda.synchronize()
    t=time.perf_counter()
    for _ in range(300):run(p,'ring',[0],enabled)
    if DEVICE=='cuda':torch.cuda.synchronize()
    return (time.perf_counter()-t)*1e6/300
rows={'old':[],'new':[]}
for pair in range(6):
    for name in (('old','new') if pair%2==0 else ('new','old')):
        rows[name].append(bench(name=='new'))
print(json.dumps(dict(complete=True,passed=True,device=DEVICE,cases=cases,
    graph_plan_isolation=True,timing=dict(pairs_us=rows,
      median_us={k:statistics.median(v) for k,v in rows.items()},
      scope='Full 36-layer initial_dense host dispatch with private ring gather; small tensor shape, not production GPU timing'))))
