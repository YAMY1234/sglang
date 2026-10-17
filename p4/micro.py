"""Matched 36-layer K1/norm/out-proj graphs. Run only after strict gate PASS."""
import argparse
import json
from pathlib import Path
import statistics
import torch
from common import FIELDS,clone,fixture,norm,step,weights,z_metadata
from sglang.srt.layers.attention.linear.kernels.gdn_factored_cuda import load_extension


def point(ext,tp,b,ws,gen,reps,layout):
    layers,hv,hidden=36,48//tp,4096
    base,i,slots,z=fixture(tp,b,gen,layers,layout)
    p=clone(base)
    flat=z.reshape(-1,128)
    separate_copy_bytes=layers*z.numel()*z.element_size() if flat.data_ptr()!=z.data_ptr() else 0
    del flat
    raw=torch.empty(layers,b,1,hv,128,device='cuda',dtype=torch.bfloat16)
    fused=torch.empty_like(raw)
    matrices=[torch.randn(hv*128,hidden,device='cuda',dtype=torch.bfloat16,generator=gen)*.01 for _ in range(layers)]
    gemm=torch.empty(layers,b,hidden,device='cuda',dtype=torch.bfloat16)
    def reset():
        for k in FIELDS:
            p[k].copy_(base[k])
    def call(label):
        for l in range(layers):
            f=label in ('fused_chain','fused_only')
            out=fused[l] if f else raw[l]
            step(ext,p,i,slots,out,l,tp,z if f else None,ws[l%len(ws)] if f else None)
            if label=='separate_chain':
                out=norm(out,z,ws[l%len(ws)])
            if label.endswith('_chain'):
                torch.mm(out.view(b,hv*128),matrices[l],out=gemm[l])
    labels=('separate_chain','fused_chain','k1_chain','k1_only','fused_only')
    graphs={}
    for label in labels:
        reset();call(label);torch.cuda.synchronize();reset()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call(label)
        graphs[label]=g
    flush=torch.empty(512*1024*1024//4,device='cuda',dtype=torch.float32)
    samples={label:[] for label in labels}
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    # 50 samples per arm. Both orders, with each paired arm measured on the same
    # GPU and reset state. Capture/warmup/flush/copies are outside the interval.
    for n in range(reps):
        for label in (labels if n%2==0 else labels[::-1]):
            reset();flush.zero_();start.record();graphs[label].replay();end.record()
            end.synchronize();samples[label].append(start.elapsed_time(end)*1000)
    med={label:statistics.median(x) for label,x in samples.items()}
    delta=(med['fused_chain']-med['separate_chain'])/layers
    kernel_delta=(med['fused_only']-med['k1_only'])/layers
    return dict(tp=tp,batch=b,layers=layers,z_layout=layout,z_metadata=z_metadata(z),
        layout_copy_bytes_per_step=dict(separate=separate_copy_bytes,fused=0),
        layout_copy_us_per_step=dict(separate=None,fused=0),
        separate_copy_timing='included in separate_chain; not isolated',
        producer_layout_change=False,
        measurement_scope='measured-layout' if layout['measured'] else 'synthetic-layout-preflight',
        spacer_gemm=[b,hv*128,hidden],
        samples_us=samples,median_step_us=med,median_layer_us={k:v/layers for k,v in med.items()},
        fused_minus_separate_layer_us=delta,fused_minus_separate_step_us=delta*layers,
        fused_minus_k1_layer_us=kernel_delta,
        decision='pass' if delta<=-1 and kernel_delta<=.5 else 'kill' if delta>-.5 else 'not_admitted')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True);ap.add_argument('--gate',required=True)
    ap.add_argument('--model',default='/model');ap.add_argument('--reps',type=int,default=50)
    args=ap.parse_args()
    gate=json.loads(Path(args.gate).read_text())
    assert gate.get('complete') and gate.get('pass'), 'strict bitwise gate required before timing'
    assert args.reps>=50
    ext=load_extension();assert ext is not None
    assert gate['extension']==ext.__file__, 'gate must qualify this exact source binary'
    ws,meta=weights(args.model)
    gen=torch.Generator(device='cuda').manual_seed(231)
    result=dict(complete=False,gate=args.gate,extension=ext.__file__,norm_weights=meta,
                l2_flush_bytes=512*1024*1024,estimator='median of 50 per arm; alternating forward/reverse order',points={})
    dst=Path(args.out)
    def save():
        dst.write_text(json.dumps(result,indent=2)+'\n')
    save()
    catalog=gate['z_layouts']
    result['z_layouts']=catalog
    for tp,batches in ((2,(1,2,4,8)),(1,(1,2,3))):
        for layout in catalog[str(tp)]:
            if not layout['primary']: continue
            for b in batches:
                rec=point(ext,tp,b,ws,gen,args.reps,layout)
                result['points'][f"tp{tp}_B{b}_{layout['name']}"]=rec;save()
                print('MICRO',json.dumps({k:v for k,v in rec.items() if k!='samples_us'}),flush=True)
                torch.cuda.empty_cache()
    result['complete']=True;save()

if __name__=='__main__':
    main()
