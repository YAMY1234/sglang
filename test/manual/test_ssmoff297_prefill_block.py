"""Shared graph ownership on CPU; actual unchanged GDN kernels on CUDA."""
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
spec=importlib.util.spec_from_file_location('prefill_block_candidate',p)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


def same(a,b,name):
    assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)),name


def fake(t):
    t['state'].add_(t['q'].sum()+t['log'].sum())
    return t['q']*t['log'][0]+t['state'].sum(),None,t['state'].clone()


def real(t):
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    g,beta=fused_gdn_gating(t['log'],t['a'],t['b'],t['bias'])
    return chunk_gated_delta_rule(q=t['q'],k=t['k'],v=t['v'],g=g,beta=beta,
        initial_state=t['state'],initial_state_indices=t['rows'],cu_seqlens=t['cu'],
        head_first=False,use_qk_l2norm_in_kernel=True,inplace_update=True)


@torch.inference_mode()
def main():
    torch.manual_seed(297)
    graph=m.PrefillBlockGraph();rows=[];retained=[]
    for tokens in ((256,8192,256) if GPU else (8,16,8,32)):
        for layer in range(3):
            if GPU:
                # Match the production strided q/k views of mixed QKV.
                mixed=torch.randn(1,tokens,(2*8+24)*128,dtype=torch.bfloat16)*.05
                tensors=dict(q=mixed[:,:,:8*128].view(1,tokens,8,128),
                    k=mixed[:,:,8*128:16*128].view(1,tokens,8,128),
                    v=mixed[:,:,16*128:].view(1,tokens,24,128),
                    a=torch.randn(tokens,24,dtype=torch.bfloat16),b=torch.randn(tokens,24,dtype=torch.bfloat16),
                    log=torch.randn(24),bias=torch.randn(24,dtype=torch.bfloat16),
                    state=torch.randn(1,24,128,128)*.01,rows=torch.tensor([0],dtype=torch.int32),
                    cu=torch.tensor([0,tokens],dtype=torch.int32))
                evaluate=real
            else:
                tensors=dict(q=torch.randn(tokens,2)[:,::2],log=torch.randn(2),state=torch.randn(1,2,4,4))
                evaluate=fake
            rebound={k:torch.empty_like(v,memory_format=torch.contiguous_format)
                     for k,v in tensors.items()}
            m.bind_inputs(rebound,tensors)
            for name in tensors:same(rebound[name],tensors[name],'input binding '+name)
            print(json.dumps(dict(tokens=tokens,layer=layer,phase='reference')),file=sys.stderr,flush=True)
            reference={k:v.clone() for k,v in tensors.items()}
            out,last,h=evaluate(reference)
            final=reference['state'] if last is None else last
            if GPU:torch.cuda.synchronize()
            print(json.dumps(dict(tokens=tokens,layer=layer,phase='graph')),file=sys.stderr,flush=True)
            got,state,checkpoint=graph.run(tensors,evaluate)
            same(got,out,'output');same(state,final,'state');same(checkpoint,h,'checkpoint')
            assert all(m.check_result((got,state,checkpoint),(out,last,h),reference['state']).values())
            if not GPU:
                wrong=got.clone();wrong.flatten()[0]+=1
                try:m.check_result((wrong,state,checkpoint),(out,last,h),reference['state'])
                except RuntimeError:pass
                else:raise AssertionError('corrupt output accepted by admission checker')
            for value,saved in retained:same(value,saved,'previous layer ownership')
            retained=[(got,got.clone()),(state,state.clone())]
            rows.append(dict(tokens=tokens,layer=layer,bitwise=True))
    assert len(graph.entries)<=2
    if GPU:assert graph.stats['captured']==2 and graph.stats['replayed']==9
    print(json.dumps(dict(complete=True,passed=True,device='CUDA' if GPU else 'CPU',
                         cuda_math_executed=GPU,cases=rows,stats=graph.stats)),flush=True)


if __name__=='__main__':main()
