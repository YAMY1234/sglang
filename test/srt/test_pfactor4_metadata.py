"""Real plan_extend parity while reducing host metadata readback calls."""
import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNConfig, FactoredGDNPool


def case(device,batch,kind,strict=1,prefix=1):
    cfg=FactoredGDNConfig(dtype=torch.float16,strict_chunk=strict,factored_prefix=prefix,ring=16)
    original=FactoredGDNPool(size=64,cache_params=NS(shape=NS(temporal=(2,16,16))),
        mamba_layer_ids=[0,1,2],device=device,cfg=cfg)
    slots=torch.arange(1,batch+1,device=device)
    lens=[1024]*batch;prefix_lens=[0]*batch;final=[True]*batch
    if kind in ('hit','invalid_prefix','continue','lost','ring_full'):
        prefix_lens=[1024]*batch
        if original.prefix_valid is not None:original.prefix_valid[slots]=1
    if kind=='invalid_prefix':original.prefix_valid[slots]=0
    if kind in ('continue','lost'):
        original.dense_required[slots]=1;final=[False]*batch
        if kind=='continue':
            original.dense_of[slots]=torch.arange(batch,device=device,dtype=torch.int32)
            original.stale[slots]=0;original.ring_owner[:batch]=list(range(1,batch+1))
    if kind=='ring_full':
        original.ring_owner=list(range(33,49))
        original.stale[33:49]=1
    if kind=='padded':slots[-2:]=-1;lens[-2:]=[0,0]
    outcomes=[];pools=[];calls=[]
    for enabled in ('0','1'):
        p=copy.deepcopy(original);count=[];tolist=torch.Tensor.tolist
        def counted(value,*a,**kw):
            count.append(tuple(value.shape));return tolist(value,*a,**kw)
        with patch.dict(os.environ,SGLANG_PFACTOR4_BATCH_METADATA=enabled,SGLANG_FLASHNEXT_FACTOR_GUARD_ABORT='0'), \
             patch.object(torch.Tensor,'tolist',counted):
            try:
                plan=p.plan_extend(slots,lens,prefix_lens=prefix_lens,prompt_final=final)
                result={k:(v.detach().cpu() if isinstance(v,torch.Tensor) else v) for k,v in vars(plan).items()}
            except RuntimeError as exc:result=dict(error=str(exc))
        outcomes.append(result);pools.append(p);calls.append(count)
    assert outcomes[0].keys()==outcomes[1].keys()
    for key in outcomes[0]:
        a,b=outcomes[0][key],outcomes[1][key]
        assert torch.equal(a,b) if isinstance(a,torch.Tensor) else a==b,key
    for key in ('a','U','W','count','stale','dense_of','dense_required','prefix_valid','dense_ring'):
        a,b=getattr(pools[0],key),getattr(pools[1],key)
        assert (a is None and b is None) or torch.equal(a,b),key
    for key in ('ring_owner','ring_lru','ring_generation','stats'):assert getattr(pools[0],key)==getattr(pools[1],key),key
    assert len(calls[1])<=len(calls[0])
    if strict and prefix and kind not in ('lost',):assert len(calls[1])<len(calls[0])
    return dict(B=batch,kind=kind,strict=strict,prefix=prefix,passed=True,readback_shapes=calls,
        readback_calls=[len(c) for c in calls],error=outcomes[0].get('error'),real_plan=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    device='cpu' if os.environ.get('TRITON_INTERPRET')=='1' else 'cuda'
    cases=[(b,k) for b in (1,8) for k in ('fresh','hit','continue','lost','invalid_prefix','ring_full')]
    cases.extend([(8,'padded'),(1,'fresh',0,0),(8,'fresh',1,0),(8,'fresh',0,1)])
    rows=[case(device,*values) for values in cases]
    result=dict(passed=True,device=device,factor_dtype='float16',gate='identical native plan, ring ownership, errors and tensor bytes',rows=rows)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('PFACTOR4_METADATA_GATE',json.dumps(result),flush=True)


if __name__=='__main__':main()
