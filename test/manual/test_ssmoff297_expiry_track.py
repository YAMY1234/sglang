"""Exact post-order expiry/publication for every W8 phase and alias boundary."""
import copy
import json
import sys
import torch
from test_ssmoff297_kernels import GPU, pool, kernels, same
from prefill_kernels.gdn_expiry_track import expiry_track
from test_ssmoff297_step_warps import captured_step


def check(count,dtype,src=2,dst=5,active=True,layers=2):
    old=pool(layers,24 if GPU else 2,128 if GPU else 16)
    old.count.fill_(count)
    old.prefix_factored_valid.fill_(1)
    old.stale.zero_()
    old.decode_metadata_fused=True
    new=copy.deepcopy(old)
    indices=torch.tensor([src],dtype=dtype)
    destinations=torch.tensor([dst],dtype=dtype)
    mask=torch.tensor([active])
    kernels.factored_expiry_truncate_layers(old.U,old.W,old.count,indices,8,16)
    old.track_copy(indices,mask,destinations)
    expiry_track(new,indices,mask,destinations)
    for name in ('a','U','W','count','stale','prefix_factored_valid'):
        same(getattr(old,name),getattr(new,name),'fused expiry '+name)
    return dict(count=count,dtype=str(dtype),src=src,dst=dst,active=active,layers=layers,bitwise=True)


torch.manual_seed(297)
rows=[]
for dtype in (torch.int32,torch.int64):
    cases=[(c,2,5,True) for c in range(8,17)]
    cases += [(16,2,2,True),(16,-1,5,True),(16,2,-1,True),
              (16,2,5,False),(16,-1,-1,False),(9,2,5,False)]
    for count,src,dst,active in cases:
        try:row=check(count,dtype,src,dst,active)
        except Exception as error:
            row=dict(count=count,dtype=str(dtype),src=src,dst=dst,active=active,
                     bitwise=False,error=str(error))
        rows.append(row)
if GPU:
    try:rows.append(check(16,torch.int64,layers=36))
    except Exception as error:rows.append(dict(layers=36,bitwise=False,error=str(error)))
passed=all(row['bitwise'] for row in rows)
result=dict(complete=True,passed=passed,device='CUDA' if GPU else 'CPU',cases=rows)
if passed and GPU:
    result['timings']=[]
    for fused in (False,True,True,False):
        torch.manual_seed(297)
        result['timings'].append(captured_step(1,expiry_track=fused))
    base=(result['timings'][0]['mean_ms']+result['timings'][3]['mean_ms'])/2
    candidate=(result['timings'][1]['mean_ms']+result['timings'][2]['mean_ms'])/2
    result['timing_comparison']=dict(base_ms=base,candidate_ms=candidate,delta_ms=candidate-base)
print(json.dumps(result),flush=True)
sys.exit(0 if passed else 1)
