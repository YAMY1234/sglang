"""Real pool exact-ring gather: byte identity, ownership and graph cost."""
import json
import os
from types import SimpleNamespace

import torch
from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNPool

device = os.environ.get('REPLAY_TEST_DEVICE', 'cpu')
layers, heads, width = (36, 24, 128) if device == 'cuda' else (3, 2, 8)
torch.manual_seed(297)
ring = torch.randn(layers, 4, heads, width, width, device=device)
ring.flatten()[0] = 0.0
ring.flatten()[1] = -0.0
pool = SimpleNamespace(dense_ring=ring, layer_ids=list(range(layers)),
    layer_map={i:i for i in range(layers)}, batch_prefill=True,
    hv=heads, v=width, k=width, prefill_reuse=False, prefix_dense=None,
    stats={'densified':0}, _ring_layers_logged=True)
pool._initial_dense_eager = lambda layer, plan: FactoredGDNPool._initial_dense_eager(pool, layer, plan)


def run(flag, indices, first=0, last=None):
    os.environ['SGLANG_GDN_PREFILL_RING_LAYERS'] = str(int(flag))
    last = layers-1 if last is None else last
    index = indices if isinstance(indices,torch.Tensor) else torch.tensor(indices,device=device,dtype=torch.int64)
    plan = SimpleNamespace(slots=index, ring_src=index, all_fresh=False,
        n_ring_src=len(indices), next_layer=first, last_layer=last)
    out = []
    for layer in range(first,last+1):
        out.append(FactoredGDNPool.initial_dense(pool,layer,plan))
        plan.next_layer = layer+1
    return out, plan


def same(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(
        a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


cases = []
for indices, first, last in [([0],0,layers-1),([3,1],0,layers-1),
        ([2,2],0,layers-1),([0,1,2],0,layers-1),([],0,layers-1),
        ([3],1,layers-1),([2],0,layers-2)]:
    before = ring.clone()
    reference,_ = run(False,indices,first,last)
    actual,plan = run(True,indices,first,last)
    expected_active = bool(indices) and first==0 and last==layers-1 and (
        layers*len(indices)*heads*width*width*4 <= 128 << 20)
    assert hasattr(plan,'_ring_layers') == expected_active
    assert all(same(a,b) for a,b in zip(actual,reference))
    assert same(ring,before)
    if indices:
        # Recurrent kernels may mutate returned slices; neither the original
        # ring nor a different layer's private state may change with them.
        actual[0].add_(1)
        assert same(ring,before)
        assert all(same(a,b) for a,b in zip(actual[1:],reference[1:]))
        again,_ = run(True,indices,first,last)
        assert all(same(a,b) for a,b in zip(again,reference))
    cases.append(dict(indices=indices,first=first,last=last,slab=expected_active,
                      bitwise=True,ring_unchanged=True,private_layers=True))

result = dict(complete=True,passed=True,device=device,cuda_math_executed=device=='cuda',
              cases=cases,scope='Exact state copy and ownership only; no model numerical admission')
if device == 'cuda':
    import triton.testing
    index = torch.tensor([3],device=device,dtype=torch.int64)
    def bench(flag):
        return triton.testing.do_bench_cudagraph(lambda: run(flag,index),rep=200)
    timings = [dict(order='old-new',old_ms=bench(False),new_ms=bench(True)),
               dict(order='new-old',new_ms=bench(True),old_ms=bench(False))]
    result.update(timings=timings,delta_ms=sum(r['new_ms']-r['old_ms'] for r in timings)/2)
print(json.dumps(result))
