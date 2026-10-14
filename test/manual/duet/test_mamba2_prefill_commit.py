"""Cold-factor qualification, slot reuse, FP64 optimal-cut trajectory and cost.

Arithmetic differences are recorded, not hidden behind a widened M1 tolerance.
Lifecycle/finite/ownership failures are hard assertions. No model weights.
"""
import argparse
import json
from pathlib import Path
import statistics
import torch
from sglang.srt.models.lightning_duet._common import load
from sglang.srt.layers.attention.linear.kernels.mamba2_factored import step
Pool=load('mamba2_factored_pool').Mamba2FactoredStatePool


def relative(a,b):
    return float(((a.double()-b.double()).flatten(-2).norm(dim=-1)/b.double().flatten(-2).norm(dim=-1).clamp_min(1e-30)).max())


def cut(s,d):
    coeff=(d[None,:,None,:]@s).squeeze(-2)/d.square().sum(-1)[None,:,None]
    sink=d[None,:,:,None]*coeff[...,None,:]
    u,sigma,v=torch.linalg.svd(s-sink,full_matrices=False)
    return sink+(u[...,:16]*sigma[...,None,:16])@v[...,:16,:]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--compile-only',action='store_true');args=ap.parse_args()
    torch.manual_seed(20261014);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if args.compile_only:
        # Same constexpr geometry as the service; compile before expensive model loads.
        directions=torch.nn.functional.normalize(torch.randn(1,64,64,device='cuda'),dim=-1)
        pool=Pool(41,(0,),directions,n=128,rank=16,window=16,conv_dim=4,conv_width=3)
        state=torch.randn(1,64,64,128,device='cuda');slots=torch.tensor([1],device='cuda')
        for mode in ('qr','device'):
            pool._prefill_commit_mode=mode;plan=pool.begin_extend(slots)
            pool.commit_extend(0,plan,state);pool.finish_extend(plan)
            torch.cuda.synchronize()
            assert bool(torch.isfinite(pool.materialize(0,1)).all())
        args.out.write_text(json.dumps(dict(complete=True,geometry='H64/P64/N128/slots41',modes=['qr','device']))+'\n')
        return
    result=dict(complete=False,cases=[],timings=[])
    def save():args.out.write_text(json.dumps(result,indent=2)+'\n')
    for batch in (1,8):
        heads=2;directions=torch.nn.functional.normalize(torch.randn(3,heads,64,device='cuda'),dim=-1)
        for mode in ('qr','device'):
            new=Pool(batch+2,(0,2),directions,n=128,rank=16,window=16,conv_dim=4,conv_width=3)
            old=Pool(batch+2,(0,2),directions,n=128,rank=16,window=16,conv_dim=4,conv_width=3)
            new._prefill_commit_mode=mode
            slots=torch.arange(batch,0,-1,device='cuda',dtype=torch.int64)
            for repeat in range(2):
                plan=new.begin_extend(slots);states={}
                for lid in (0,2):
                    state=torch.randn(batch,heads,64,128,device='cuda');states[lid]=state
                    # Reverse producer rows also exercises pointer-row mapping.
                    for row in reversed(range(batch)):
                        new.commit_extend(lid,plan,state[row:row+1],rows=(row,))
                        old.initialize(lid,int(slots[row]),state[row],torch.zeros(4,3,device='cuda'))
                new.finish_extend(plan)
                assert new.dense.numel()==0 and torch.equal(new.valid[:,slots],torch.ones_like(new.valid[:,slots]))
                assert not bool(new.u[:,0].any()) and not bool(new.v[:,0].any())
                changes=[]
                for i,lid in enumerate((0,2)):
                    got=torch.stack([new.materialize(lid,int(s)) for s in slots]).double().cpu()
                    reference=torch.stack([old.materialize(lid,int(s)) for s in slots]).double().cpu()
                    optimal=cut(states[lid].double().cpu(),new.directions[i].double().cpu())
                    error=relative(got,optimal);threshold=max(relative(optimal.bfloat16().float(),optimal),1e-5)
                    q=new.u[i,slots,:,:16].transpose(-1,-2)
                    assert bool(torch.isfinite(got).all())
                    ortho=float((q.transpose(-1,-2)@q-torch.eye(16,device='cuda')).abs().max())
                    assert ortho<.01,ortho
                    changes.append(dict(layer=lid,vs_release_rel_f=relative(got,reference),
                        vs_best_rank_rel_f=error,bf16_threshold=threshold,cold_best_rank_gate=error<=threshold,
                        orthogonality_max_abs=ortho))
                row=dict(B=batch,mode=mode,repeat=repeat,commit_groups=plan.commits,comparisons=changes)
                result['cases'].append(row);save()
            # Test the actual initialized factors for64 steps, FP64+same-period best rank.
            dense=cut(states[0].double().cpu(),new.directions[0].double().cpu())
            d=torch.randn(heads,device='cuda');d_cpu=d.double().cpu();records=[]
            for t in range(64):
                x=torch.randn(batch,heads,64,device='cuda');b=torch.randn(batch,1,128,device='cuda')
                c=torch.randn_like(b);dt=torch.nn.functional.softplus(torch.randn(batch,heads,device='cuda'))
                decay=(-.15*dt).exp()
                got=step(x,b,c,dt,decay,d,new.directions[0],new.coeff[0],new.u[0],new.v[0],new.fcount[0],slots)
                bh=b.double().cpu().expand(batch,heads,128);ch=c.double().cpu().expand_as(bh)
                dense=decay.double().cpu()[...,None,None]*dense+(dt.double().cpu()[...,None]*x.double().cpu())[...,None]*bh[...,None,:]
                expected=(dense@ch[...,None]).squeeze(-1)+d_cpu[None,:,None]*x.double().cpu()
                # Per-head vector relative Frobenius; same envelope as M1FP32.
                err=float(((got.double().cpu()-expected).norm(dim=-1)/expected.norm(dim=-1).clamp_min(1e-30)).max())
                eps=float(((expected.bfloat16().float()-expected).norm(dim=-1)/expected.norm(dim=-1).clamp_min(1e-30)).max())
                records.append(dict(step=t+1,rel_f=err,threshold=max(eps,1e-5),passed=err<=max(eps,1e-5)))
                if (t+1)%16==0:dense=cut(dense,new.directions[0].double().cpu())
            row['cold_64_steps']=records;row['cold_trajectory_pass']=all(r['passed'] for r in records);save()
    # Full production geometry, steady-state whole begin/commit/finish wall+events.
    for batch in (1,8):
        directions=torch.nn.functional.normalize(torch.randn(23,64,64,device='cuda'),dim=-1)
        states=[torch.randn(batch,64,64,128,device='cuda') for _ in range(23)]
        pool=Pool(batch+1,tuple(range(23)),directions,n=128,rank=16,window=16,conv_dim=4,conv_width=3)
        slots=torch.arange(1,batch+1,device='cuda')
        import time
        for mode in ('reference','qr','device'):
            pool._prefill_commit_mode=mode;spans=[];walls=[]
            for repeat in range(4):
                torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                t=time.perf_counter();start.record();plan=pool.begin_extend(slots)
                for lid,state in enumerate(states):pool.commit_extend(lid,plan,state)
                pool.finish_extend(plan);end.record();end.synchronize()
                if repeat:spans.append(start.elapsed_time(end));walls.append((time.perf_counter()-t)*1000)
            result['timings'].append(dict(B=batch,mode=mode,device_span_ms=statistics.median(spans),
                completed_wall_ms=statistics.median(walls),commit_groups=plan.commits,
                peak_pending_bytes=plan.peak_pending_bytes,scope='23layers x64heads full begin/commit/finish; per batch'))
            save()
    result['complete']=True;save()

if __name__=='__main__':main()
