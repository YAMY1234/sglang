"""CUDA-only explicit layout admission; CPU uses separate offline compilation."""
import json
import os
import sys
import torch
from test_ssmoff297_kernels import GPU, decode_case
from test_ssmoff297_step_warps import captured_step

assert GPU, 'Gluon has no CPU interpreter; use compile_ssmoff297_decode_gluon.py'
target=int(os.environ['SSMOFF_STEP_GLUON_WARPS']);assert target in (1,2,4)
torch.manual_seed(297)
rows=[]
for bias in (torch.float32,torch.bfloat16):
    for qheads in (8,24):
        for dtype in (torch.int32,torch.int64):
            for batch in (1,3):
                for count in range(8,16):
                    try:
                        row=decode_case(24,128,batch,count,dtype,1,0,target,qheads,bias)
                    except Exception as error:
                        row=dict(bitwise=False,count=count,batch=batch,dtype=str(dtype),error=str(error))
                    row.update(qheads=qheads,bias_dtype=str(bias));rows.append(row)
passed=all(r['bitwise'] for r in rows)
result=dict(complete=True,passed=passed,gluon_warps=target,device='CUDA',cases=rows)
if passed:
    result['timings']=[captured_step(1,0,n) for n in (0,target,target,0)]
    base=(result['timings'][0]['mean_ms']+result['timings'][3]['mean_ms'])/2
    candidate=(result['timings'][1]['mean_ms']+result['timings'][2]['mean_ms'])/2
    result['timing_comparison']=dict(base_ms=base,candidate_ms=candidate,delta_ms=candidate-base,improvement=candidate<base)
print(json.dumps(result),flush=True)
sys.exit(0 if passed else 1)
