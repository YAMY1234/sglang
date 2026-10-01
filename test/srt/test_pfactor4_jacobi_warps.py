"""Default-off Jacobi launch-width experiment; exact D/Z before event timing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from sglang.srt.layers.attention.linear.kernels.gdn_k31_eigh import eigh, _k31_eigh_kernel


def compile_gate():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    for warps in (1, 2, 4):
        compiled = triton.compile(ASTSource(fn=_k31_eigh_kernel,
            signature={'G_ptr':'*fp64','D_ptr':'*fp64','Z_ptr':'*fp64'},
            constexprs={'N':16,'SWEEPS':12,'EARLY_EXIT':True}),
            target=GPUTarget('cuda',103,32),options={'num_warps':warps})
        assert compiled.asm['ptx'] and compiled.asm['cubin']
    print('PFACTOR4_JACOBI_WIDTH_COMPILE PASS sm103 1/2/4',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--compile-only',action='store_true')
    parser.add_argument('--no-timing',action='store_true');parser.add_argument('--output');args=parser.parse_args()
    if args.compile_only:compile_gate();return
    device='cpu' if os.environ.get('TRITON_INTERPRET')=='1' else 'cuda'
    if device=='cpu':
        subprocess.run([sys.executable,__file__,'--compile-only'],check=True,
            env=dict(os.environ,TRITON_INTERPRET='0'),timeout=180)
    torch.manual_seed(0x504634)
    rows=[]
    for batch in ((1,) if device=='cpu' else (1,2,8,16)):
        heads=2 if device=='cpu' else 24
        x=torch.randn(batch,heads,16,32,dtype=torch.float64,device=device)
        for kind in ('full','rank4','diagonal','zero'):
            y=x.clone()
            if kind=='rank4':y[...,4:,:]*=1.e-8
            g=y@y.transpose(-1,-2)
            identity=torch.eye(16,dtype=torch.float64,device=device)
            if kind=='diagonal':g=identity.expand(batch,heads,16,16).clone()
            if kind=='zero':g.zero_()
            d,z=eigh(g)
            for warps in (1,2,4):
                new_d,new_z=eigh(g,num_warps=warps)
                exact=torch.equal(d,new_d) and torch.equal(z,new_z)
                row=dict(B=batch,heads=heads,kind=kind,warps=warps,passed=exact,
                    d_max_abs=float((d-new_d).abs().max()),z_max_abs=float((z-new_z).abs().max()))
                rows.append(row)
    admitted=[w for w in (1,2,4) if all(r['passed'] for r in rows if r['warps']==w)]
    # Time only launch widths whose entire matrix suite passed, including the
    # joined normal+tracked B2/B16 service shapes. Never alter convergence.
    if device=='cuda' and not args.no_timing:
        for batch in (1,2,8,16):
            x=torch.randn(batch,24,16,32,dtype=torch.float64,device=device)
            g=x@x.transpose(-1,-2)
            for warps in admitted:
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):eigh(g,num_warps=warps)
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                pairs=[(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)) for _ in range(20)]
                with torch.cuda.graph(graph,stream=stream):
                    for begin,end in pairs:
                        begin.record();eigh(g,num_warps=warps);end.record()
                graph.replay();torch.cuda.synchronize()
                values=[a.elapsed_time(b) for a,b in pairs]
                rows.append(dict(B=batch,heads=24,kind='timing_full',warps=warps,
                    mean_ms=sum(values)/len(values),min_ms=min(values),max_ms=max(values)))
    result=dict(passed=1 in admitted,device=device,production_warps=1,
        gate='D and Z bitwise equal to unchanged one-warp early-exit reference',
        numerically_admitted=admitted,rows=rows)
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('PFACTOR4_JACOBI_WIDTH_GATE',json.dumps(result),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
