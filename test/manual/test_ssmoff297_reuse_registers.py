"""Admission for shared prefill buffers and fixed-one-warp register scheduling."""
import json
import os
import sys
import torch
from test_ssmoff297_kernels import GPU, pool, plan, same, decode_case
from test_ssmoff297_step_warps import captured_step


def prefill_case(layers, heads, key, batch):
    p=pool(layers,heads,key);p.prefill_reuse=True;p.batch_prefill=True
    retained=[]
    for _ in range(2):
        pp=plan(p,2)
        pp.slots=torch.arange(batch,dtype=torch.int64)
        pp.all_fresh=True
        expected=torch.arange(batch,dtype=torch.int32)
        addresses=[];states=[]
        for i,lid in enumerate(p.layer_ids):
            actual=p.prefill_row_indices(pp)
            same(actual,expected,'shared row indices')
            addresses.append(actual.data_ptr())
            state=p.initial_dense(lid,pp)
            same(state,p._initial_dense_eager(lid,pp),'fresh layer')
            state.fill_(i+1)
            states.append(state)
            pp.next_layer+=1
        assert len(set(addresses))==1
        for i,state in enumerate(states):same(state,torch.full_like(state,i+1),'disjoint layer states')
        for i,state in enumerate(retained):same(state,torch.full_like(state,i+1),'prior plan retained')
        retained=states
    return dict(kind='prefill',layers=layers,heads=heads,key=key,batch=batch,bitwise=True)


if __name__=='__main__':
    target=int(os.environ.get('SSMOFF_STEP_MAXNREG','0'));assert target in (0,128,192,256)
    torch.manual_seed(297)
    rows=[prefill_case(36 if GPU else 2,24 if GPU else 2,128 if GPU else 16,b) for b in (1,3)]
    for dtype in (torch.int32,torch.int64):
        for batch in (1,3):
            for count in range(8,16):
                try:
                    row=decode_case(24 if GPU else 2,128 if GPU else 16,batch,count,dtype,1,target)
                except Exception as error:
                    row=dict(bitwise=False,count=count,batch=batch,dtype=str(dtype),error=str(error))
                rows.append(row)
    passed=all(r['bitwise'] for r in rows)
    result=dict(complete=True,passed=passed,maxnreg=target,device='CUDA' if GPU else 'CPU',cases=rows)
    if passed and GPU:
        result['timings']=[captured_step(1,n) for n in (0,target,target,0)]
        base=(result['timings'][0]['mean_ms']+result['timings'][3]['mean_ms'])/2
        candidate=(result['timings'][1]['mean_ms']+result['timings'][2]['mean_ms'])/2
        result['timing_comparison']=dict(base_ms=base,candidate_ms=candidate,delta_ms=candidate-base,improvement=candidate<base)
    print(json.dumps(result),flush=True)
    sys.exit(0 if passed else 1)
