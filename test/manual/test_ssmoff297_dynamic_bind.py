"""Actual-length rebinding preserves bytes and reuses a compiled bucket kernel."""
import json
import os

import torch
from sglang.srt.mem_cache.gdn_prefill_block_graph import PrefillBlockGraph, bind_inputs

device=os.environ.get('REPLAY_TEST_DEVICE','cpu')
torch.manual_seed(297)
rows=[];old_hashes=set();new_hashes=set()
for n in (263,271,287,301,319,337,359,383,401,431,479,511):
    mixed=torch.randn(1,n,40*128,device=device,dtype=torch.bfloat16)
    t=dict(q=mixed[:,:,:1024].view(1,n,8,128),
        k=mixed[:,:,1024:2048].view(1,n,8,128),
        v=mixed[:,:,2048:].view(1,n,24,128),
        a=torch.randn(n,24,device=device,dtype=torch.bfloat16),
        b=torch.randn(n,24,device=device,dtype=torch.bfloat16),
        log=torch.randn(24,device=device),bias=torch.randn(24,device=device),
        state=torch.randn(1,24,128,128,device=device),
        rows=torch.tensor([0],device=device,dtype=torch.int32),
        cu=torch.tensor([0,n],device=device,dtype=torch.int32))
    def buffers():
        result={}
        for name,x in t.items():
            shape=list(x.shape)
            if name in ('q','k','v'):shape[1]=512
            if name in ('a','b'):shape[0]=512
            result[name]=torch.full(shape,7,dtype=x.dtype,device=device)
        return result
    old,new=buffers(),buffers()
    for enabled,out,hashes in ((False,old,old_hashes),(True,new,new_hashes)):
        os.environ['SGLANG_GDN_PREFILL_DYNAMIC_BIND']=str(int(enabled))
        if device=='cuda':
            kernel=bind_inputs(out,t);hashes.add(kernel.hash)
        else:
            PrefillBlockGraph(bucketed=True)._bind(out,t)
    for name in t:
        assert torch.equal(old[name].view(torch.uint8),new[name].view(torch.uint8)),name
        actual=new[name][:,:n] if name in ('q','k','v') else new[name][:n] if name in ('a','b') else new[name]
        assert torch.equal(actual.contiguous().view(torch.uint8),t[name].contiguous().view(torch.uint8)),name
    rows.append(dict(tokens=n,bitwise=True,padding_unchanged=True))
if device=='cuda':
    assert len(old_hashes)==len(rows),(len(old_hashes),len(rows))
    assert len(new_hashes)==1,len(new_hashes)
print(json.dumps(dict(complete=True,passed=True,device=device,cases=rows,
    old_compiled_kernels=len(old_hashes),new_compiled_kernels=len(new_hashes),
    scope='Binding bytes and compilation identity, not model admission')))
