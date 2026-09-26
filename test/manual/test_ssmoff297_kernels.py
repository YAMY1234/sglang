"""Same-container CPU interpreter and CUDA admission for #297 candidates."""
import copy
import importlib
import json
import os
import sys

import torch
from test_factored_prefill_graph import GPU, pool, module, same

kernels = importlib.import_module('prefill_kernels.gdn_factored')
initial = importlib.import_module('prefill_owners.gdn_prefill_initial_layers_graph')


def plan(p, slot):
    return module.FactoredExtendPlan(slots=torch.tensor([slot]),
        use_ring=torch.zeros(1, dtype=torch.bool), ring_src=torch.zeros(1, dtype=torch.long),
        ring_dst=torch.tensor([-1]), ring_dst_rows=torch.empty(0, dtype=torch.long),
        last_layer=len(p.layer_ids)-1)


def restore_case(layers, heads, key):
    p = pool(layers, heads, key)
    graph = initial.PrefillInitialLayersGraph()
    retained = None
    saved = None
    for slot in (2, 5, 2):
        pp = plan(p, slot)
        expected = torch.stack([p._initial_dense_eager(lid, pp) for lid in p.layer_ids])
        out = graph.run(p, pp)
        same(out, expected, 'restored layers')
        if retained is not None:
            same(retained, saved, 'prior outputs survive replay')
        out[0].fill_(19)
        retained, saved = out, out.clone()
        same(graph.run(p, pp), expected, 'in-place recurrence cannot corrupt replay')
    if GPU:
        assert graph.stats['captured'] == 1 and graph.stats['replayed'] == 6
    # Exercise the actual pool dispatch across successive layer commits.
    os.environ['SGLANG_GDN_PREFILL_INITIAL_LAYERS_GRAPH'] = '1'
    pp = plan(p, 5)
    for lid in p.layer_ids:
        expected = p._initial_dense_eager(lid, pp)
        same(p.initial_dense(lid, pp), expected, 'pool restore dispatch')
        pp.next_layer += 1
    os.environ['SGLANG_GDN_PREFILL_INITIAL_LAYERS_GRAPH'] = '0'
    return dict(kind='restore', layers=layers, heads=heads, key=key, stats=graph.stats, bitwise=True)


def decode_case(heads, key, batch, count, dtype):
    p = pool(2, heads, key)
    p.count.fill_(count)
    old, new = copy.deepcopy(p), copy.deepcopy(p)
    old.prefix_factored_valid.fill_(1); new.prefix_factored_valid.fill_(1)
    slots = torch.tensor([2, 5, -1][:batch], dtype=dtype)
    mixed = torch.randn(batch, 3*heads*key, dtype=torch.bfloat16)
    a = torch.randn(batch, heads, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    A_log = torch.randn(heads); bias = torch.randn(heads)
    output = []
    for candidate, owner in ((False, old), (True, new)):
        if not candidate:
            owner.invalidate_prefix_dense(slots)
        output.append(kernels.factored_packed_decode(mixed, a, b,
            A_log=A_log, dt_bias=bias, scale=key**-.5, vbar=owner.vbar[0],
            fa=owner.a[0], fu=owner.U[0], fw=owner.W[0], fcount=owner.count[0],
            stale=owner.stale, ssm_state_indices=slots, num_q_heads=heads,
            num_v_heads=heads, head_k_dim=key, head_v_dim=key,
            r=8, rfull=16, truncate=True, kernel='split', post_order=True,
            prefix_valid=owner.prefix_valid if candidate else None))
    same(output[0], output[1], 'decode output')
    for name in ('a','U','W','count','stale','prefix_factored_valid'):
        same(getattr(old,name), getattr(new,name), 'decode '+name)
    return dict(kind='decode', heads=heads, key=key, batch=batch, count=count, dtype=str(dtype), bitwise=True)


def tracking_case(mask_dtype, index_dtype):
    for src, dst, mask in ((2,5,True), (2,2,True), (-1,5,True), (2,-1,True),
                           (2,5,False), (-1,-1,False)):
        old = pool(2, 2, 16); new = copy.deepcopy(old)
        old.prefix_factored_valid.fill_(1); new.prefix_factored_valid.fill_(1)
        new.decode_metadata_fused = True
        s = torch.tensor([src], dtype=index_dtype)
        d = torch.tensor([dst], dtype=index_dtype)
        m = torch.tensor([mask], dtype=mask_dtype)
        for owner in (old,new):
            owner.track_copy(s,m,d)
        for name in ('a','U','W','count','stale','prefix_factored_valid'):
            same(getattr(old,name),getattr(new,name),'track '+name)
    return dict(kind='track', mask_dtype=str(mask_dtype), index_dtype=str(index_dtype), cases=6, bitwise=True)


if __name__ == '__main__':
    torch.manual_seed(297)
    rows = [restore_case(2,2,16)]
    if GPU:
        rows.append(restore_case(36,24,128))
    for dtype in (torch.int32,torch.int64):
        for batch in (1,3):
            for count in range(8,16):
                rows.append(decode_case(24 if GPU else 2,128 if GPU else 16,batch,count,dtype))
        for mask in (torch.bool,torch.int32):
            rows.append(tracking_case(mask,dtype))
    print(json.dumps(dict(passed=True, device='CUDA' if GPU else 'CPU', cases=rows)))
