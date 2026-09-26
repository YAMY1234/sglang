"""Bitwise W8 phase admission and captured B1 recurrence timing per warp count."""
import copy
import json
import os
import sys

import torch
from test_ssmoff297_kernels import GPU, pool, kernels, decode_case


def captured_step(warps, maxnreg=0, gluon=0):
    p=pool(36,24,128)
    p.count.fill_(8)
    slots=torch.tensor([2],dtype=torch.int64)
    mixed=torch.randn(1,3*24*128,dtype=torch.bfloat16)
    a=torch.randn(1,24,dtype=torch.bfloat16);b=torch.randn_like(a)
    log=torch.randn(24);bias=torch.randn(24)
    mask=torch.tensor([False]);destination=torch.tensor([5])
    p.decode_metadata_fused=True
    kernels.STEP_WARPS=warps
    kernels.STEP_MAXNREG=maxnreg
    kernels.STEP_GLUON_WARPS=gluon
    outputs=[torch.empty(1,1,24,128,dtype=torch.bfloat16) for _ in range(36)]
    def step():
        for i in range(36):
            kernels.factored_packed_decode(mixed,a,b,A_log=log,dt_bias=bias,scale=128**-.5,
                vbar=p.vbar[i],fa=p.a[i],fu=p.U[i],fw=p.W[i],fcount=p.count[i],stale=p.stale,
                ssm_state_indices=slots,num_q_heads=24,num_v_heads=24,head_k_dim=128,head_v_dim=128,
                r=8,rfull=16,truncate=False,kernel='split',out=outputs[i],
                prefix_valid=p.prefix_valid if i==0 else None)
        kernels.factored_expiry_truncate_layers(p.U,p.W,p.count,slots,8,16)
        p.track_copy(slots,mask,destination)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(16):step()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):step()
    for _ in range(32):graph.replay()
    pairs=[]
    for _ in range(10):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(256):graph.replay()
        end.record();pairs.append((start,end))
    torch.cuda.synchronize()
    values=[start.elapsed_time(end)/256 for start,end in pairs]
    return dict(warps=warps,maxnreg=maxnreg,gluon=gluon,mean_ms=sum(values)/len(values),samples_ms=values,
                scope='36-layer B1 recurrence plus post-step expiry and inactive tracking; CUDA graph microbenchmark, not model C1')


if __name__=='__main__':
    target=int(os.environ['SSMOFF_STEP_WARPS']);assert target in (2,4)
    torch.manual_seed(297)
    rows=[]
    for dtype in (torch.int32,torch.int64):
        for batch in (1,3):
            for count in range(8,16):
                try:
                    row=decode_case(24 if GPU else 2,128 if GPU else 16,batch,count,dtype,target)
                except Exception as error:
                    row=dict(bitwise=False,count=count,batch=batch,dtype=str(dtype),error=str(error))
                rows.append(row)
    passed=all(row['bitwise'] for row in rows)
    result=dict(complete=True,passed=passed,warps=target,device='CUDA' if GPU else 'CPU',cases=rows)
    if passed and GPU:
        # Independent equal-weight forward and reverse order to expose drift.
        result['timings']=[captured_step(w) for w in (1,target,target,1)]
        base=sum(x['mean_ms'] for x in result['timings'] if x['warps']==1)/2
        candidate=sum(x['mean_ms'] for x in result['timings'] if x['warps']==target)/2
        result['timing_comparison']=dict(base_ms=base,candidate_ms=candidate,delta_ms=candidate-base,
                                         improvement=candidate<base)
    print(json.dumps(result))
    sys.exit(0 if passed else 1)
