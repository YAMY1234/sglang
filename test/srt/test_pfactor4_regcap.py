"""Default-off one-warp register-cap experiment; preserve every state/output bit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from test_pfactor4_packed_step import inputs, launch as original
from sglang.srt.layers.attention.linear.kernels.gdn_factored import _factored_packed_step_kernel

CAPS = (0, 128, 160, 192, 224)


def launch(work, constants, cap):
    options = dict(num_warps=1)
    if cap:
        options['maxnreg'] = cap
    return _factored_packed_step_kernel[(work[0].shape[0]*24,)](
        *work, 128**-.5, 1.e-4, **constants, LATE_W_LOAD=False, **options)


def compile_gate():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    names = ('mixed_qkv','a_gate','b_gate','A_log','dt_bias','vbar','a_ptr','u_ptr','w_ptr',
             'cnt_ptr','stale_ptr','ssm_state_indices','o','scale','gs_eps')
    types = ('*bf16','*bf16','*bf16','*fp32','*fp32','*fp32','*fp32','*bf16','*bf16',
             '*i32','*i32','*i32','*bf16','fp32','fp32')
    constants = dict(stride_mixed_tok=5120,stride_a_tok=24,stride_b_tok=24,stride_idx=1,
                     H=8,HV=24,K=128,V=128,RMAX=16,SOFTPLUS_THRESHOLD=20.,LATE_W_LOAD=False)
    for cap in CAPS:
        options = dict(num_warps=1)
        if cap:options['maxnreg']=cap
        compiled = triton.compile(ASTSource(fn=_factored_packed_step_kernel,
            signature=dict(zip(names,types)),constexprs=constants),
            target=GPUTarget('cuda',103,32),options=options)
        assert compiled.asm['ptx'] and compiled.asm['cubin']
    print('PFACTOR4_REGCAP_COMPILE PASS sm103', CAPS, flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--compile-only',action='store_true')
    parser.add_argument('--no-timing',action='store_true')
    parser.add_argument('--output')
    args=parser.parse_args()
    if args.compile_only:compile_gate();return
    device='cpu' if os.environ.get('TRITON_INTERPRET')=='1' else 'cuda'
    if device=='cpu':
        subprocess.run([sys.executable,__file__,'--compile-only'],check=True,
                       env=dict(os.environ,TRITON_INTERPRET='0'),timeout=180)
    records=[]
    shapes=((1,False),(8,False),(8,True))
    for batch,padded in shapes:
        for near in (False,True):
            source,constants=inputs(batch,device,near)
            if not padded:source[11]=torch.arange(batch,device=device,dtype=torch.int32)
            expected=[v.clone() for v in source];original(expected,constants,False,1)
            for cap in CAPS:
                work=[v.clone() for v in source];launch(work,constants,cap)
                failures=[dict(tensor=i,max_abs=float((work[i].float()-expected[i].float()).abs().max()))
                          for i in (6,7,8,9,10,12) if not torch.equal(work[i],expected[i])]
                records.append(dict(B=batch,padded=padded,near_span=near,cap=cap,
                                    passed=not failures,failures=failures))
    admitted=[cap for cap in CAPS if all(v['passed'] for v in records if v['cap']==cap)]
    if device=='cuda' and not args.no_timing:
        for batch,padded in shapes:
            for near in (False,True):
                source,constants=inputs(batch,device,near)
                if not padded:source[11]=torch.arange(batch,device=device,dtype=torch.int32)
                for cap in admitted:
                    work=[v.clone() for v in source];compiled=launch(work,constants,cap)
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    graph=torch.cuda.CUDAGraph()
                    pairs=[(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)) for _ in range(20)]
                    with torch.cuda.graph(graph,stream=stream):
                        for begin,end in pairs:
                            for dst,src in zip(work,source):dst.copy_(src)
                            begin.record();launch(work,constants,cap);end.record()
                    graph.replay();torch.cuda.synchronize()
                    times=[a.elapsed_time(b) for a,b in pairs]
                    records.append(dict(B=batch,padded=padded,near_span=near,cap=cap,timing=True,
                        mean_ms=sum(times)/len(times),min_ms=min(times),max_ms=max(times),
                        registers=compiled.n_regs,spills=compiled.n_spills,shared=compiled.metadata.shared))
    result=dict(passed=0 in admitted,production_default=False,device=device,records=records,
                numerically_admitted=admitted,gate='all state/output tensors bitwise equal for every shape',
                timing='step only; includes compiler spills; production launch unchanged')
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('PFACTOR4_REGCAP_GATE',json.dumps(result),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
