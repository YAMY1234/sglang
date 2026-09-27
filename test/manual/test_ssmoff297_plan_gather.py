"""Integer metadata kernel vs torch plus real pool plan/state equivalence."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from sglang.srt.mem_cache.gdn_prefill_plan_gather import gather, reference

GPU = os.environ.get('REPLAY_TEST_DEVICE') == 'cuda'
DEVICE = 'cuda' if GPU else 'cpu'
if not GPU: assert os.environ.get('TRITON_INTERPRET') == '1'


def main():
    torch.manual_seed(297)
    cases=[]
    for b in (1,2,3,16,32,64,96):
        for required,valid in ((False,False),(True,False),(False,True),(True,True)):
            n=128; r=96
            pool=SimpleNamespace(stale=torch.randint(0,2,(n,),device=DEVICE,dtype=torch.int32),
                dense_of=torch.randint(-1,r,(n,),device=DEVICE,dtype=torch.int32),
                dense_required=torch.randint(0,2,(n,),device=DEVICE,dtype=torch.int32) if required else None,
                prefix_valid=torch.randint(0,2,(n,),device=DEVICE,dtype=torch.int32) if valid else None)
            slots=torch.randint(0,n,(b,),device=DEVICE,dtype=torch.int64);slots[::9]=-1
            owners=torch.randint(0,n,(r,),device=DEVICE,dtype=torch.int64)
            for first in (0,1):
                actual=gather(pool,slots,owners,first);expected=reference(pool,slots,owners,first)
                assert torch.equal(actual,expected),(b,required,valid,first)
                cases.append(dict(batch=b,required=required,valid=valid,first=first,passed=True))
    # Exercise actual ring ownership, required-state validation, prefix and COW paths.
    path=Path(__file__).with_name('test_ssmoff297_prefill_host.py')
    spec=importlib.util.spec_from_file_location('host_reference',path)
    host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
    checks=0
    for tokens,prefix in ((256,64),(32768,0),(2285,32768),(6687,51392)):
        for seed in range(6):
            os.environ['SGLANG_GDN_PREFILL_PLAN_GATHER']='0'
            old=host.run(True,seed,tokens,prefix)
            os.environ['SGLANG_GDN_PREFILL_PLAN_GATHER']='1'
            new=host.run(True,seed,tokens,prefix)
            assert host.same(old,new),(tokens,prefix,seed)
            checks += len(old)
    timing=None
    if GPU:
        import time
        def bench(fn):
            for _ in range(10): fn()
            torch.cuda.synchronize(); start=time.perf_counter()
            for _ in range(100): fn()
            torch.cuda.synchronize()
            return (time.perf_counter()-start)*1e6/100
        small=slots[:1]
        timing=dict(reference_us=bench(lambda: reference(pool,small,owners,0)),
                    candidate_us=bench(lambda: gather(pool,small,owners,0)),
                    scope='B1 integer metadata operations only; excludes D2H and the rest of the planner')
    print(json.dumps(dict(complete=True,passed=True,device=DEVICE,cases=cases,pool_checks=checks,timing=timing)))

if __name__=='__main__': main()
