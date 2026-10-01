"""M4a integer gather and unchanged native continuation planner admission."""
import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNConfig, FactoredGDNPool
from sglang.srt.mem_cache.gdn_prefill_plan_gather import gather, reference


def same(a, b):
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype
                and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))
    return a == b


def pool(device, *, strict=1, prefix=1, exact=0, ring=16):
    return FactoredGDNPool(size=64, cache_params=NS(shape=NS(temporal=(2,16,16))),
        mamba_layer_ids=[0,1,2], device=device,
        cfg=FactoredGDNConfig(dtype=torch.float16, strict_chunk=strict,
                             factored_prefix=prefix, exact_prefix=exact, ring=ring))


def primitive(device, batch, strided, required, valid, first):
    p = pool(device)
    torch.manual_seed(0x4400+batch)
    p.stale.random_(0, 2);p.dense_of.random_(-1, 16)
    p.dense_required.random_(0, 2);p.prefix_factored_valid.random_(0, 2)
    if not required:p.dense_required=None
    if not valid:p.prefix_factored_valid=None
    slots=(torch.arange(batch*2,device=device,dtype=torch.long)%64)[::2] if strided else torch.arange(batch,device=device)
    slots[0]=-1
    owners=torch.tensor([0,7,9,33,45],device=device,dtype=torch.long)
    expected=reference(p,slots,owners,first);actual=gather(p,slots,owners,first)
    assert same(expected,actual)
    return dict(passed=True,B=batch,strided=strided,required=required,valid=valid,first=first,
                bitwise=True,dtype=str(actual.dtype),elements=actual.numel())


def native_case(device, batch, kind, *, strict=1, prefix=1, exact=0, first=0,
                abort=False, batched=False, strided=False, slot_dtype=torch.int64):
    original=pool(device,strict=strict,prefix=prefix,exact=exact)
    slots=(torch.arange(1,batch*2+1,device=device,dtype=slot_dtype)[::2]
           if strided else torch.arange(1,batch+1,device=device,dtype=slot_dtype))
    lens=[1024]*batch;prefix_lens=[0]*batch;final=[True]*batch
    if kind in ('hit','invalid','continue','lost','full'):
        prefix_lens=[1024]*batch
        if original.prefix_valid is not None:original.prefix_valid[slots]=1
    if kind=='invalid':original.prefix_valid[slots]=0
    if kind in ('continue','lost'):
        original.dense_required[slots]=1;final=[False]*batch
        if kind=='continue':
            original.dense_of[slots]=torch.arange(batch,device=device,dtype=torch.int32)
            original.stale[slots]=0;original.ring_owner[:batch]=slots.cpu().tolist()
    if kind in ('full','grow','limit','mixed'):
        original.ring_owner=list(range(33,49))
        original.stale[33:49]=int(kind=='full')
        original.dense_required[33:49]=int(kind!='full')
        if kind in ('grow','limit'):final=[False]*batch
        if kind=='mixed':final=[i%2==1 for i in range(batch)]
        if kind=='limit':original.ring_capacity_limit=len(original.ring_owner)
        original.dense_ring.copy_(torch.arange(original.dense_ring.numel(),device=device,
            dtype=original.dense_ring.dtype).view_as(original.dense_ring))
    if kind=='padded':slots[-2:]=-1;lens[-2:]=[0,0]
    states=[];outcomes=[];readbacks=[];joins=[]
    for enabled in ('0','1'):
        p=copy.deepcopy(original);calls=[];joined=[];tolist=torch.Tensor.tolist
        native_join=p.pside_join
        def join():
            joined.append(len(calls));return native_join()
        def counted(tensor,*args,**kwargs):
            assert joined, 'Metadata readback preceded the native producer join'
            calls.append(tuple(tensor.shape));return tolist(tensor,*args,**kwargs)
        with patch.dict(os.environ,SGLANG_GDN_PREFILL_PLAN_GATHER=enabled,
                        SGLANG_GDN_PREFILL_PLAN_GATHER_CHECK='0',
                        SGLANG_PFACTOR4_BATCH_METADATA=str(int(batched)),
                        SGLANG_FLASHNEXT_FACTOR_GUARD_ABORT=str(int(abort))), \
             patch.object(p,'pside_join',join),patch.object(torch.Tensor,'tolist',counted), \
             patch('sglang.srt.distributed.get_tensor_model_parallel_rank',return_value=0):
            try:
                result=vars(p.plan_extend(slots,lens,prefix_lens=prefix_lens,
                    prompt_final=final,layer_range=(first,2)))
            except RuntimeError as exc:result=dict(error=str(exc))
        outcomes.append(result);states.append(p);readbacks.append(calls);joins.append(joined)
    assert outcomes[0].keys()==outcomes[1].keys(),kind
    for key in outcomes[0]:assert same(outcomes[0][key],outcomes[1][key]),(kind,key)
    for key in ('a','U','W','count','stale','dense_of','dense_required','prefix_valid','dense_ring','prefix_dense'):
        assert same(getattr(states[0],key),getattr(states[1],key)),(kind,key)
    for key in ('ring_owner','ring_lru','ring_generation','ring_capacity_limit','stats','guard_rows'):
        assert getattr(states[0],key,None)==getattr(states[1],key,None),(kind,key)
    assert len(readbacks[1])<=len(readbacks[0])
    assert len(joins[0])==len(joins[1]) and all(j[0]==0 for j in joins)
    if kind in ('grow','mixed'):assert states[1].ring_generation==1 and len(joins[1])==2
    if kind in ('lost','limit') or (kind=='invalid' and not abort and first==0):
        assert 'error' in outcomes[1]
    else:assert 'error' not in outcomes[1]
    if kind=='invalid' and abort and first==0:assert states[1].guard_rows
    assert states[1]._plan_gather_calls==len(joins[1])
    return dict(passed=True,B=batch,kind=kind,strict=strict,prefix=prefix,exact=exact,first=first,
        abort=abort,batched=batched,strided=strided,slot_dtype=str(slot_dtype),native_plan=True,
        bitwise=True,readback_calls=[len(c) for c in readbacks],join_calls=[len(j) for j in joins],
        ring_generation=states[1].ring_generation,error=outcomes[1].get('error'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    device='cpu' if os.environ.get('TRITON_INTERPRET')=='1' else 'cuda'
    components=[primitive(device,b,strided,required,valid,first)
                for b in (1,8,48) for strided in (False,True)
                for required,valid,first in ((False,False,0),(True,False,0),(True,True,0),(True,True,1))]
    rows=[native_case(device,b,kind) for b in (1,8)
          for kind in ('fresh','hit','continue','lost','invalid','full','grow','limit','mixed')]
    rows += [native_case(device,8,'padded'),native_case(device,8,'hit',strided=True),
             native_case(device,8,'fresh',slot_dtype=torch.int32),
             native_case(device,8,'invalid',abort=True),native_case(device,8,'invalid',first=1),
             native_case(device,8,'fresh',strict=0,prefix=0),
             native_case(device,8,'fresh',prefix=0),native_case(device,8,'hit',prefix=0,exact=1),
             native_case(device,8,'full',batched=True),native_case(device,8,'grow',batched=True)]
    assert len(components)==24 and len(rows)==28
    result=dict(passed=True,complete=True,device=device,factor_dtype='float16',
        components=components,rows=rows,scope='Native planner, producer join, guards, ring growth, all state bytes. Integer Triton interpreter on CPU; CUDA integer kernel on GPU.')
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('PFACTOR4_PLAN_GATHER_GATE',json.dumps(result),flush=True)


if __name__=='__main__':main()
