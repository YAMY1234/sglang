"""Experimental shared padded buckets; CPU ownership then actual CUDA math."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

GPU=os.environ.get('REPLAY_TEST_DEVICE')=='cuda'
assert GPU or os.environ.get('CUDA_VISIBLE_DEVICES')==''
if GPU:torch.set_default_device('cuda')
p=Path(__file__).resolve().parents[2]/'python/sglang/srt/mem_cache/gdn_prefill_block_graph.py'
spec=importlib.util.spec_from_file_location('prefill_bucket_candidate',p)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


def real(t):
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    g,beta=fused_gdn_gating(t['log'],t['a'],t['b'],t['bias'])
    return chunk_gated_delta_rule(q=t['q'],k=t['k'],v=t['v'],g=g,beta=beta,
        initial_state=t['state'],initial_state_indices=t['rows'],cu_seqlens=t['cu'],
        head_first=False,use_qk_l2norm_in_kernel=True,inplace_update=True)


def fake(t):
    n=int(t['cu'][-1]);t['state'].add_(t['q'][:,:n].sum())
    out=t['q']*t['log'][0]
    checkpoint=t['state'].unsqueeze(1).repeat(1,(n+63)//64,1,1,1)
    return out,None,checkpoint


def same(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(
        a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


@torch.inference_mode()
def main():
    torch.manual_seed(779)
    graph=m.PrefillBlockGraph(bucketed=True);records=[];retained=[]
    lengths=(263,511,386,512,527,6687,8192,2285,32768,16384,24576,6687)
    for n in lengths:
        for layer in range(2):
            hq,hv,width=(8,24,128) if GPU else (1,2,4)
            mixed=torch.randn(1,n,(2*hq+hv)*width,dtype=torch.bfloat16)*.05
            t=dict(q=mixed[:,:,:hq*width].view(1,n,hq,width),
                k=mixed[:,:,hq*width:2*hq*width].view(1,n,hq,width),
                v=mixed[:,:,2*hq*width:].view(1,n,hv,width),
                a=torch.randn(n,hv,dtype=torch.bfloat16),b=torch.randn(n,hv,dtype=torch.bfloat16),
                log=torch.randn(hv),bias=torch.randn(hv,dtype=torch.bfloat16),
                state=torch.randn(1,hv,width,width)*.01,rows=torch.tensor([0],dtype=torch.int32),
                cu=torch.tensor([0,n],dtype=torch.int32))
            reference={k:v.clone() for k,v in t.items()}
            evaluate=real if GPU else fake
            expected=evaluate(reference)
            got=graph.run(t,evaluate)
            checks=m.check_result(got,expected,reference['state'])
            for value,saved in retained:assert same(value,saved),'prior return ownership'
            retained=[(got[0],got[0].clone()),(got[1],got[1].clone())]
            row=dict(tokens=n,layer=layer,bucket=max(64,1<<(n-1).bit_length()),checks=checks,
                     stats=dict(graph.stats))
            records.append(row);print(json.dumps(row),file=sys.stderr,flush=True)
    assert len(graph.entries)==6,len(graph.entries)
    if GPU:assert graph.stats['captured']==6 and graph.stats['replayed']==len(records)
    print(json.dumps(dict(complete=True,passed=True,device='CUDA' if GPU else 'CPU',
                         cases=records,stats=graph.stats,production_enabled=False)),flush=True)


if __name__=='__main__':main()
