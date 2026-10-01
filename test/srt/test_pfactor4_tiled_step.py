"""Unused V-partition candidate: immutable a/count inputs and exact byte gate.

Microseconds cover the step kernel only. Snapshot preparation is excluded and
must be measured/charged before this candidate can enter a production path.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from test_pfactor4_packed_step import inputs, launch as original
from sglang.srt.layers.attention.linear.kernels.gdn_factored_tiled_step import _factored_tiled_v_step_kernel


def launch(work, constants, tile, a_snapshot, count_snapshot):
    values = work[:7]+[a_snapshot]+work[7:10]+[count_snapshot]+work[10:]
    return _factored_tiled_v_step_kernel[(work[0].shape[0]*24*(128//tile),)](
        *values, 128**-.5, 1e-4, **constants, V_TILE=tile, PRESERVE_V_LAYOUT=True, num_warps=1)


def compile_gate():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    signature = dict(zip(('mixed_qkv','a_gate','b_gate','A_log','dt_bias','vbar',
        'a_ptr','a_snapshot_ptr','u_ptr','w_ptr','cnt_ptr','count_snapshot_ptr',
        'stale_ptr','ssm_state_indices','o','scale','gs_eps'),
        ('*bf16','*bf16','*bf16','*fp32','*fp32','*fp32','*fp32','*fp32','*bf16',
         '*bf16','*i32','*i32','*i32','*i32','*bf16','fp32','fp32')))
    constants = dict(stride_mixed_tok=5120,stride_a_tok=24,stride_b_tok=24,stride_idx=1,
        H=8,HV=24,K=128,V=128,RMAX=16,SOFTPLUS_THRESHOLD=20.,LATE_W_LOAD=False)
    for tile in (128,64,32):
        compiled = triton.compile(ASTSource(fn=_factored_tiled_v_step_kernel,
            signature=signature,constexprs=dict(constants,V_TILE=tile,PRESERVE_V_LAYOUT=True)),
            target=GPUTarget('cuda',103,32),options={'num_warps':1})
        assert compiled.asm['ptx'] and compiled.asm['cubin']
    print('PFACTOR4_TILED_COMPILE PASS sm103 tiles=128/64/32',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--compile-only',action='store_true')
    parser.add_argument('--no-timing',action='store_true');parser.add_argument('--output');args=parser.parse_args()
    if args.compile_only:compile_gate();return
    device='cpu' if os.environ.get('TRITON_INTERPRET')=='1' else 'cuda'
    if device=='cpu':
        subprocess.run([sys.executable,__file__,'--compile-only'],check=True,
            env=dict(os.environ,TRITON_INTERPRET='0'),timeout=180)
    records=[]
    for batch,padded in ((1,False),(8,False),(8,True)):
        for near in (False,True):
            source,constants=inputs(batch,device,near)
            if not padded:source[11]=torch.arange(batch,device=device,dtype=torch.int32)
            expected=[v.clone() for v in source];original(expected,constants,False,1)
            a_snapshot,count_snapshot=source[6].clone(),source[9].clone()
            for tile in (128,64,32):
                work=[v.clone() for v in source]
                compiled=launch(work,constants,tile,a_snapshot,count_snapshot)
                failed=[]
                for i in (6,7,8,9,10,12):
                    if not torch.equal(work[i],expected[i]):
                        failed.append(dict(tensor=i,max_abs=float((work[i].float()-expected[i].float()).abs().max())))
                row=dict(B=batch,padded=padded,near_span=near,tile=tile,passed=not failed,failures=failed)
                records.append(row)
    admitted=[tile for tile in (128,64,32) if all(r['passed'] for r in records if r['tile']==tile)]
    if device=='cuda' and not args.no_timing:
        for batch,padded in ((1,False),(8,False),(8,True)):
            for near in (False,True):
                source,constants=inputs(batch,device,near)
                if not padded:source[11]=torch.arange(batch,device=device,dtype=torch.int32)
                a_snapshot,count_snapshot=source[6].clone(),source[9].clone()
                for tile in admitted:
                    work=[v.clone() for v in source]
                    compiled=launch(work,constants,tile,a_snapshot,count_snapshot)
                    graph=torch.cuda.CUDAGraph();stream=torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    pairs=[(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)) for _ in range(20)]
                    with torch.cuda.graph(graph,stream=stream):
                        for begin,end in pairs:
                            for dst,src in zip(work,source):dst.copy_(src)
                            begin.record();launch(work,constants,tile,a_snapshot,count_snapshot);end.record()
                    graph.replay();torch.cuda.synchronize()
                    values=[a.elapsed_time(b) for a,b in pairs]
                    records.append(dict(B=batch,padded=padded,near_span=near,tile=tile,
                        timing=True,mean_ms=sum(values)/len(values),max_ms=max(values),min_ms=min(values),
                        registers=compiled.n_regs,spills=compiled.n_spills,shared=compiled.metadata.shared))
    passed=128 in admitted
    result=dict(passed=passed,production_default=False,device=device,records=records,
        numerically_admitted=admitted,preserve_v_layout=True,
        gate='all state/output tensors bitwise equal across all shapes; any failing tile disabled everywhere',
        timing='step only, excludes snapshots; not a production performance claim')
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('PFACTOR4_TILED_GATE',json.dumps(result),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
