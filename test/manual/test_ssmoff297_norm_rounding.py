"""Diagnostic stress of norm fusion rounding; no model admission/performance gate."""
import copy
import json
import os
from pathlib import Path
import sys

import torch
from test_ssmoff297_norm_step import GPU, fixture, step


def identical(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def run():
    records=[]
    seeds=range(32) if GPU else range(2)
    for seed in seeds:
        for rows in (1,4):
            torch.manual_seed(297+seed)
            count=seed%16
            original=fixture(2,count,torch.int64,2)
            original['p'].vbar.zero_()
            # Served parameters can be FP32; original tests used BF16 weights/bias.
            if seed%2:
                original['weight']=original['weight'].float()
                original['bias']=original['bias'].float()
            scale=2.**((seed%7-3)*3)
            original['x'].mul_(scale)
            a=copy.deepcopy(original);b=copy.deepcopy(original)
            for tick in range(16 if GPU else 3):
                before=copy.deepcopy(a) if GPU else None
                expected=step(a,False,rows,'sigmoid');actual=step(b,True,rows,'sigmoid')
                outputs=[identical(x,y) for x,y in zip(expected,actual)]
                states={name:identical(getattr(a['p'],name),getattr(b['p'],name))
                        for name in ('a','U','W','count','stale','prefix_factored_valid')}
                record=dict(seed=seed,rows=rows,count=count,step=tick,scale=scale,
                            weight_dtype=str(a['weight'].dtype),outputs=outputs,states=states)
                records.append(record)
                if not all(outputs) or not all(states.values()):
                    dest=Path(os.environ.get('SSMOFF_NORM_ROUNDING_OUT','norm-rounding-failure.pt'))
                    if before is not None:
                        tensors={k:v for k,v in before.items() if k!='p'}
                        tensors['pool']={k:v for k,v in vars(before['p']).items() if isinstance(v,torch.Tensor)}
                        tensors.update(expected=expected,actual=actual,case=record)
                        torch.save(tensors,dest)
                    return dict(complete=True,diagnostic_only=True,passed=False,cases=records,
                                failure=str(dest) if before is not None else None)
    return dict(complete=True,diagnostic_only=True,passed=True,cases=records,
                scope='Synthetic multi-step rounding, not full-model numerical admission')


if __name__=='__main__':
    result=run()
    print(json.dumps(result),flush=True)
    # A counterexample is a successful diagnostic collection, never admission.
    sys.exit(0)
