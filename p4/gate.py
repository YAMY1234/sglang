"""P4 strict bitwise gate. A mismatch exits nonzero; it never starts timing."""
import argparse
import json
from pathlib import Path
import torch
from common import FIELDS,bitwise,clone,fixture,norm,step,weights,diff
from inspect_norm import inspect
from sglang.srt.layers.attention.linear.kernels import gdn_factored as gf
from sglang.srt.layers.attention.linear.kernels.gdn_factored_cuda import load_extension


def case(ext,tp,b,inv,graph,w,gen,steps):
    base,i,slots,z=fixture(tp,b,gen)
    pa,pb=clone(base),clone(base)
    shape=(b,1,48//tp,128)
    raw=torch.empty(shape,device='cuda',dtype=torch.bfloat16)
    out=torch.empty_like(raw)
    def a():
        step(ext,pa,i,slots,raw,0,tp,invalidate=inv)
    def c():
        step(ext,pb,i,slots,out,0,tp,z,w,invalidate=inv)
    if graph:
        a(); c(); norm(raw,z,w); torch.cuda.synchronize()
        for k in FIELDS:
            pa[k].copy_(base[k]);pb[k].copy_(base[k])
        ga,gb=torch.cuda.CUDAGraph(),torch.cuda.CUDAGraph()
        with torch.cuda.graph(ga):
            a()
        with torch.cuda.graph(gb):
            c()
        a,c=ga.replay,gb.replay
    due=torch.zeros_like(base['fcount'])
    for n in range(steps):
        slots.copy_(torch.arange(1,b+1,device='cuda',dtype=slots.dtype))
        if n%8==7:
            slots[-1]=-1
        if n==32:
            for p in (pa,pb):
                for k in ('fa','fu','fw','fcount'):
                    p[k][:,0].copy_(base[k][:,1])
                p['prefix_valid'][0]=1
            slots[0]=0
        i['mixed'].copy_((torch.randn(i['mixed'].shape,device='cuda',generator=gen)*.5).bfloat16())
        i['a'].copy_(torch.randn(i['a'].shape,device='cuda',generator=gen).bfloat16())
        i['b'].copy_(torch.randn(i['b'].shape,device='cuda',generator=gen).bfloat16())
        z.copy_((torch.randn(z.shape,device='cuda',generator=gen)*[1.,8.,.01][n%3]).bfloat16())
        live=slots[slots>=0].long()
        due[:,live]+=pa['fcount'][:,live]==31
        a();c()
        ref=norm(raw,z,w)
        checks={k:bitwise(pa[k],pb[k]) for k in FIELDS}
        checks['y']=bitwise(ref,out)
        checks['finite']=bool(torch.isfinite(out).all())
        if not all(checks.values()):
            bad={k:diff(pa[k],pb[k]) for k in FIELDS if not checks[k]}
            if not checks['y']:
                bad['y']=diff(ref,out)
                ix=torch.nonzero(ref!=out)[:8]
                bad['y']['ref']=[float(ref[tuple(j)]) for j in ix]
                bad['y']['fused']=[float(out[tuple(j)]) for j in ix]
            return dict(pass_=False,step=n,checks=checks,differences=bad)
        gf.factored_expiry_truncate(pa['fu'][0],pa['fw'][0],pa['fcount'][0],slots,16,32)
        gf.factored_expiry_truncate(pb['fu'][0],pb['fw'][0],pb['fcount'][0],slots,16,32)
    minimum=int(due[0,1:b+1].min())
    assert steps<64 or minimum>=3, minimum
    return dict(pass_=True,steps=steps,min_expiries=minimum,fields=[*FIELDS,'y'])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    ap.add_argument('--model',default='/model')
    ap.add_argument('--steps',type=int,default=64)
    ap.add_argument('--batches',default='')
    args=ap.parse_args()
    dst=Path(args.out);dst.parent.mkdir(parents=True,exist_ok=True)
    result=dict(complete=False,cases={},runtime=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name()),
                service_dump='not supplied; synthetic WS-1b distribution only')
    def save():
        dst.write_text(json.dumps(result,indent=2)+'\n')
    save()
    inspect(dst.parent/'norm-asm')
    ext=load_extension();assert ext is not None,'extension build failed'
    result['extension']=ext.__file__
    ws,meta=weights(args.model);result['norm_weights']=meta
    gen=torch.Generator(device='cuda').manual_seed(221)
    for tp in (2,1):
        for b in ([int(v) for v in args.batches.split(',')] if args.batches else range(1,9 if tp==2 else 4)):
            if tp==1 and b>3:
                continue
            for inv in (False,True):
                for graph in (False,True):
                    key=f'tp{tp}_B{b}_inv{int(inv)}_graph{int(graph)}'
                    w=ws[len(result['cases'])%len(ws)]
                    rec=case(ext,tp,b,inv,graph,w,gen,args.steps)
                    result['cases'][key]=rec;save()
                    print(key,json.dumps(rec),flush=True)
                    if not rec['pass_']:
                        raise SystemExit(1)
    # Each actual layer weight is exercised, including weights with negative
    # components; this also checks signed zeros for padded rows.
    for n,w in enumerate(ws):
        key=f'weight{n}'
        rec=case(ext,2,1,True,False,w,gen,8)
        result['cases'][key]=rec;save()
        if not rec['pass_']:
            raise SystemExit(1)
    result['complete']=True;result['pass']=True;save()
    print('BITWISE_PASS',len(result['cases']),flush=True)

if __name__=='__main__':
    main()
