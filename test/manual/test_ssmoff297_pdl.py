"""PDL step scheduling preserves W8 arithmetic; CPU omits CUDA synchronization."""
import json,sys,torch
from test_ssmoff297_kernels import GPU,decode_case
from test_ssmoff297_step_warps import captured_step
torch.manual_seed(297)
rows=[]
for bias in (torch.float32,torch.bfloat16):
    for grouped in (True,False):
        heads=24 if GPU else 6
        for dtype in (torch.int32,torch.int64):
            for batch in (1,3):
                for count in range(8,16):
                    try:
                        row=decode_case(heads,128 if GPU else 16,batch,count,dtype,
                            qheads=heads//3 if grouped else heads,bias_dtype=bias,step_pdl=True)
                    except Exception as error:
                        row=dict(bitwise=False,count=count,batch=batch,dtype=str(dtype),error=str(error))
                    row.update(grouped=grouped,bias_dtype=str(bias));rows.append(row)
for dtype in (torch.int32,torch.int64):
    for count in range(8,16):
        try:
            row=decode_case(24 if GPU else 6,128 if GPU else 16,1,count,dtype,
                qheads=8 if GPU else 2,bias_dtype=torch.bfloat16,step_pdl=True,near_span=True)
        except Exception as error:
            row=dict(bitwise=False,count=count,dtype=str(dtype),error=str(error))
        row['near_span']=True;rows.append(row)
passed=all(r['bitwise'] for r in rows)
result=dict(complete=True,passed=passed,device='CUDA' if GPU else 'CPU',cuda_synchronization_executed=GPU,cases=rows)
if passed and GPU:
    result['timings']=[]
    for enabled in (False,True,True,False):
        torch.manual_seed(297)
        result['timings'].append(captured_step(1,step_pdl=enabled))
    base=(result['timings'][0]['mean_ms']+result['timings'][3]['mean_ms'])/2
    candidate=(result['timings'][1]['mean_ms']+result['timings'][2]['mean_ms'])/2
    result['timing_comparison']=dict(base_ms=base,candidate_ms=candidate,delta_ms=candidate-base,improvement=candidate<base)
print(json.dumps(result),flush=True)
sys.exit(0 if passed else 1)
