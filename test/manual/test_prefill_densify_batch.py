"""Exact old/new prefix reconstruction and replay after slot/state changes."""
import argparse,json,os,time
from pathlib import Path
from types import SimpleNamespace
import torch
from sglang.srt.mem_cache.gdn_prefill_initial_graph import densify_all_layers,PrefillDensifyAllGraph
from sglang.srt.mem_cache.gdn_factored_pool import densify


def run(device):
    torch.manual_seed(832);rows=[]
    for layers,heads,key,value in ((2,3,32,32),(36,24,128,128)):
        slots,rank=3,16
        pool=SimpleNamespace(layer_ids=list(range(layers)),
            a=torch.randn(layers,slots,heads,key,device=device)*.03,
            U=(torch.randn(layers,slots,heads,rank,key,device=device)*.05).half(),
            W=(torch.randn(layers,slots,heads,rank,value,device=device)*.05).half(),
            count=torch.arange(layers*slots*heads,device=device,dtype=torch.int32).reshape(layers,slots,heads)%16,
            vbar=torch.randn(layers,heads,value,device=device)*.02)
        originals={n:getattr(pool,n).clone() for n in ('a','U','W','count')}
        for slot in (0,2,-1):
            index=torch.tensor([slot],device=device)
            expected=densify_all_layers(pool,index,densify)
            got=densify_all_layers(pool,index,densify,batched=True)
            assert torch.equal(expected,got), ('batched prefix restore changed FP32 state',layers,slot,(expected-got).abs().max().item())
            rows.append(dict(layers=layers,heads=heads,slot=slot,bitwise=True))
        assert all(torch.equal(getattr(pool,n),v) for n,v in originals.items())
        if device=='cuda':
            graphs=[]
            for enabled in ('0','1'):
                os.environ['SGLANG_GDN_PREFILL_DENSIFY_BATCH']=enabled
                graphs.append(PrefillDensifyAllGraph())
            for slot in (0,2,1):
                pool.a.add_(.001);pool.count.copy_((pool.count+3)%16)
                index=torch.tensor([slot],device=device)
                left=graphs[0].run(pool,index,densify);a=left.clone()
                right=graphs[1].run(pool,index,densify);b=right.clone()
                torch.cuda.synchronize();assert torch.equal(a,b), ('graph prefix restore changed state',layers,slot,(a-b).abs().max().item())
                left.zero_();right.fill_(1)  # recurrence mutates each captured output in place
            rows.append(dict(layers=layers,replay_after_mutation=True,bitwise=True))
    return dict(complete=True,passed=True,device=device,rows=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();t=time.time();r=run(a.device);r['seconds']=time.time()-t;a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
