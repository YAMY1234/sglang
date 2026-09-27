"""Private startup bucket capture must preserve real request math and ownership."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

GPU=os.environ.get('REPLAY_TEST_DEVICE')=='cuda'
assert GPU or os.environ.get('CUDA_VISIBLE_DEVICES')==''
if GPU:torch.set_default_device('cuda')
p=Path(__file__).with_name('test_ssmoff297_prefill_buckets.py')
spec=importlib.util.spec_from_file_location('precapture_reference',p)
ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)


@torch.inference_mode()
def main():
    torch.manual_seed(297)
    hq,hv,width=(8,24,128) if GPU else (1,2,4)
    def inputs(n):
        mixed=torch.randn(1,n,(2*hq+hv)*width,dtype=torch.bfloat16)*.05
        return dict(q=mixed[:,:,:hq*width].view(1,n,hq,width),
            k=mixed[:,:,hq*width:2*hq*width].view(1,n,hq,width),
            v=mixed[:,:,2*hq*width:].view(1,n,hv,width),
            a=torch.randn(n,hv,dtype=torch.bfloat16),b=torch.randn(n,hv,dtype=torch.bfloat16),
            log=torch.randn(hv),bias=torch.randn(hv,dtype=torch.bfloat16),
            state=torch.randn(1,hv,width,width)*.01,rows=torch.tensor([0],dtype=torch.int32),
            cu=torch.tensor([0,n],dtype=torch.int32))
    graph=ref.m.PrefillBlockGraph(bucketed=True);evaluate=ref.real if GPU else ref.fake
    seed=inputs(5);saved={k:v.clone() for k,v in seed.items()}
    graph.precapture(seed,evaluate,8192)
    assert all(ref.same(seed[k],saved[k]) for k in seed), 'warmup changed live inputs'
    assert len(graph.entries)==10 and graph.prewarmed
    captures=graph.stats['captured'];records=[];previous=None
    for n in (5,15,16,31,32,33,63,64,127,128,255,256,386,511,512,1023,2047,2285,4095,6687,7936,8191,8192):
        tensors=inputs(n);reference={k:v.clone() for k,v in tensors.items()}
        expected=evaluate(reference);actual=graph.run(tensors,evaluate)
        ref.m.check_result(actual,expected,reference['state'])
        if previous is not None:
            for value,copy in previous:assert ref.same(value,copy),'returned storage reused'
        previous=[(v,v.clone()) for v in actual[:2]]
        assert graph.stats['captured']==captures,'runtime request caused another capture'
        records.append(dict(tokens=n,passed=True))
    graph.precapture(seed,evaluate,8192)
    assert graph.stats['captured']==captures
    print(json.dumps(dict(complete=True,passed=True,device='CUDA' if GPU else 'CPU',
                         cases=records,stats=graph.stats,precaptured_buckets=10)))

if __name__=='__main__':main()
