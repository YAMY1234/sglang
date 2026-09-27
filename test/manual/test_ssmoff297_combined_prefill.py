"""Compose the real integer plan and all-layer gather; preserve state ownership."""
import importlib.util
import json
import os
from pathlib import Path
import types

import torch

p=Path(__file__).with_name('test_ssmoff297_prefill_host.py')
spec=importlib.util.spec_from_file_location('pool_fixture',p)
host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
GPU=os.environ.get('REPLAY_TEST_DEVICE')=='cuda'
assert GPU or os.environ.get('TRITON_INTERPRET')=='1'


def run(enabled,seed,slots,first,last):
    torch.manual_seed(seed)
    pool=host.make();pool.batch_prefill=True;pool.prefill_reuse=True
    pool.hv, pool.k, pool.v=host.HV,host.K,host.V
    pool.layer_map={i:i for i in pool.layer_ids}
    pool.dense_ring=torch.randn(host.L,host.RING,host.HV,host.V,host.K,device=host.DEV)
    pool.ring_owner=[2,5,8,11];pool.stale.zero_()
    for i,s in enumerate(pool.ring_owner):pool.dense_of[s]=i
    pool.dense_required.zero_();pool._ring_layers_logged=True
    pool._initial_dense_eager=types.MethodType(host.gp.FactoredGDNPool._initial_dense_eager,pool)
    host.gp.PREFILL_HOST_TRIM=True
    os.environ['SGLANG_GDN_PREFILL_PLAN_GATHER']=str(int(enabled))
    os.environ['SGLANG_GDN_PREFILL_RING_LAYERS']=str(int(enabled))
    slots=torch.tensor(slots,device=host.DEV,dtype=torch.int32)
    plan=pool.plan_extend(slots,[32768]*len(slots),prefix_lens=[32768]*len(slots),
                         prompt_final=[False]*len(slots),layer_range=(first,last))
    fields=host.plan_fields(plan);before=pool.dense_ring.clone();outputs=[]
    for layer in range(first,last+1):
        value=host.gp.FactoredGDNPool.initial_dense(pool,layer,plan)
        outputs.append(value);plan.next_layer=layer+1
    active=hasattr(plan,'_ring_layers')
    assert active==(enabled and first==0 and last==host.L-1)
    saved=[v.clone() for v in outputs]
    outputs[0].add_(1)  # Chunk recurrence is allowed to mutate its private state.
    assert host.same(pool.dense_ring,before)
    assert host.same(outputs[1:],saved[1:])
    return fields,host.state(pool),saved


def main():
    records=[]
    for seed in range(8):
        for slots in ([2],[8,2],[5,11,8]):
            for first,last in ((0,host.L-1),(1,host.L-1)):
                reference=run(False,seed,slots,first,last)
                actual=run(True,seed,slots,first,last)
                assert host.same(reference,actual),(seed,slots,first,last)
                records.append(dict(seed=seed,slots=slots,first=first,last=last,bitwise=True,private=True))
    print(json.dumps(dict(complete=True,passed=True,device=host.DEV,cases=records,
                         scope='Real pool plan + state gather composition; model logits require separate admission')))

if __name__=='__main__':main()
