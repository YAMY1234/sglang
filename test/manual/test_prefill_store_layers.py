"""Whole-layer publication matches frozen per-layer stores, including no-write fallback."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from sglang.srt.layers.attention.linear.kernels.gdn_factored_io import store_factored, store_factored_layers


def case(device, layers, heads, key, batch):
    rank, capacity = 16, 5
    a = torch.randn(batch, layers*heads, key, device=device)
    u = torch.randn(batch, layers*heads, rank, key, device=device).half()
    w = torch.randn_like(u)
    factors = [(a[:,i*heads:(i+1)*heads], u[:,i*heads:(i+1)*heads].contiguous(),
                w[:,i*heads:(i+1)*heads].contiguous()) for i in range(layers)]
    p = SimpleNamespace(a=torch.randn(layers,capacity,heads,key,device=device),
        U=torch.randn(layers,capacity,heads,rank,key,device=device).half(),
        W=torch.randn(layers,capacity,heads,rank,key,device=device).half(),
        count=torch.zeros(layers,capacity,heads,dtype=torch.int32,device=device),
        stale=torch.arange(capacity,dtype=torch.int32,device=device),
        dense_of=torch.arange(capacity,dtype=torch.int64,device=device),
        dense_ring=torch.randn(layers,3,heads,key,key,device=device))
    old, new = copy.deepcopy(p), copy.deepcopy(p)
    dense = torch.randn(layers,batch,heads,key,key,device=device)
    for slot, ring in ((-1,0),(2,-1),(1,2)):
        slots=torch.tensor([slot+i if slot>=0 else slot for i in range(batch)],device=device)
        rings=torch.tensor([ring]*batch,device=device)
        if batch>1:
            assert not store_factored_layers(factors,new,slots,8,stale_value=0,dense=dense,ring_dst=rings)
            assert all(torch.equal(value,getattr(new,name)) for name,value in vars(p).items())
            continue
        for stale in (0,1):
            for i in range(layers):
                store_factored(*factors[i],old.a[i],old.U[i],old.W[i],old.count[i],
                    old.stale,old.dense_of,slots,8,stale_value=stale,
                    **(dict(dense=dense[i],ring=old.dense_ring[i],ring_dst=rings) if stale==0 else {}))
            assert store_factored_layers(factors,new,slots,8,stale_value=stale,
                **(dict(dense=dense,ring_dst=rings) if stale==0 else {}))
            for name,value in vars(old).items():
                assert torch.equal(value,getattr(new,name)), (name,slot,ring,stale)
    return dict(layers=layers,heads=heads,key=key,batch=batch,bitwise=True,
                fallback_without_writes=batch>1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    torch.manual_seed(832)
    shapes=[(2,2,16)] + ([(36,24,128)] if args.device=='cuda' else [])
    rows=[case(args.device,*shape,batch) for shape in shapes for batch in (1,2)]
    result=dict(complete=True,passed=True,device=args.device,rows=rows)
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
