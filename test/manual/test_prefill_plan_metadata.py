"""Compare integer metadata and complete host ring plans across live updates."""
import argparse,copy,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
import torch
from sglang.srt.mem_cache.gdn_prefill_plan_metadata import gather_metadata
from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNPool
from types import SimpleNamespace


def check(device):
    torch.manual_seed(298832);rows=[]
    for n in (0,1,2,8,32,97,257):
        for flags in ((False,False),(True,False),(False,True),(True,True)):
            slots=torch.randint(-1,80,(max(2*n,1),),device=device,dtype=torch.int64)[::2][:n]
            stale=torch.randint(0,2,(160,),device=device,dtype=torch.int32)[::2]
            dense=torch.randint(-1,16,(160,),device=device,dtype=torch.int32)[::2]
            required=torch.randint(0,2,(160,),device=device,dtype=torch.int32)[::2] if flags[0] else None
            valid=torch.randint(0,2,(160,),device=device,dtype=torch.int32)[::2] if flags[1] else None
            for repeat in range(3):
                safe=slots.clamp(min=0)
                expected=torch.stack([slots,stale[safe].long(),dense[safe].long()]+([required[safe].long()] if flags[0] else [])+([valid[safe].long()] if flags[1] else []))
                actual=gather_metadata(slots,stale,dense,required,valid)
                assert torch.equal(actual,expected),(n,flags,repeat)
                stale.copy_(1-stale);dense.add_(1)
            rows.append(dict(batch=n,required=flags[0],prefix=flags[1],rebinds=3,bitwise=True))
    # Run the actual planner; unchanged host ownership/LRU policy and all tensor outputs.
    for b in (1,2,4):
        pool=SimpleNamespace(layer_ids=[0,1],cfg=SimpleNamespace(strict_chunk=True,ring=8,factored_prefix=True),
            device=torch.device(device),stale=torch.zeros(16,dtype=torch.int32,device=device),
            dense_of=torch.full((16,),-1,dtype=torch.int32,device=device),
            dense_required=torch.zeros(16,dtype=torch.int32,device=device),
            prefix_valid=torch.ones(16,dtype=torch.int32,device=device),
            ring_owner=[-1]*8,ring_lru=list(range(8)),stats=dict(extends=0,rows=0,ring_src=0,ring_miss=0))
        for i in range(b):pool.dense_of[i]=i;pool.ring_owner[i]=i
        plain=copy.deepcopy(pool);packed=copy.deepcopy(pool)
        for repeat in range(3):
            slots=torch.arange(b,device=device,dtype=torch.long);prefix=[256]*b
            if repeat==1:plain.stale[slots]=1;packed.stale[slots]=1
            plans=[]
            for flag,owner in ((0,plain),(1,packed)):
                os.environ['SGLANG_GDN_PREFILL_PLAN_PACKED']=str(flag)
                plan=FactoredGDNPool.plan_extend(owner,slots,[512]*b,prefix_lens=prefix,prompt_final=[repeat!=2]*b)
                plans.append(plan)
            for key,left in vars(plans[0]).items():
                right=getattr(plans[1],key)
                assert torch.equal(left,right) if isinstance(left,torch.Tensor) else left==right,(b,repeat,key)
            assert plain.ring_owner==packed.ring_owner and plain.ring_lru==packed.ring_lru and plain.stats==packed.stats
            for key in ('stale','dense_of','dense_required','prefix_valid'):assert torch.equal(getattr(plain,key),getattr(packed,key))
        rows.append(dict(planner_batch=b,rebinds=3,full_plan_bitwise=True))
    return rows

start=time.time();rows=check(a.device)
if a.device=='cuda':torch.cuda.synchronize()
r=dict(complete=True,passed=True,device=a.device,rows=rows,seconds=time.time()-start,scope='Integer gathers and host plan identity only; no model or timing admission.')
if a.device=='cuda':
    tensors=[torch.zeros(481,dtype=torch.int32,device='cuda') for _ in range(4)]
    slots=torch.tensor([2],device='cuda',dtype=torch.int64);samples={}
    for name in ('torch','packed'):
        def read():
            safe=slots.clamp(min=0)
            if name=='torch':return torch.stack([slots]+[t[safe].long() for t in tensors]).tolist()
            return gather_metadata(slots,*tensors).tolist()
        for _ in range(10):read()
        times=[]
        for _ in range(3):
            torch.cuda.synchronize();begin=time.perf_counter()
            for _ in range(100):read()
            times.append((time.perf_counter()-begin)*1000/100)
        samples[name]=times
    r['metadata_read_wall_ms']=samples
    r['timing_scope']='Three 100-read single-row batches including D2H sync; primitive only, not TTFT.'

a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
