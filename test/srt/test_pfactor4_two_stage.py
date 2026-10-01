"""Exact FP16-state gate and two-repeat timing of both R3b launches.

Production remains unconnected. The timer includes producer, scratch writes,
consumer reads and consumer; only identical fixture restoration is excluded.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from test_pfactor4_packed_step import inputs, launch as original
from sglang.srt.layers.attention.linear.kernels.gdn_factored_two_stage import (
    _factored_k_producer, _factored_v_consumer,
)


def launch(work, constants, scratch, tile):
    assert scratch.shape == (work[0].shape[0], 24, 64)
    assert scratch.dtype == torch.float32 and scratch.is_contiguous()
    c = dict(constants)
    c.pop('V')
    producer = _factored_k_producer[(work[0].shape[0]*24,)](
        *work[:5], work[6], work[7], work[9], work[10], work[11], scratch,
        128**-.5, 1e-4, **c, SCRATCH_WORDS=64, num_warps=1)
    consumer = _factored_v_consumer[(work[0].shape[0]*24*(128//tile),)](
        work[0], work[5], work[8], work[11], work[12], scratch,
        **{k:v for k,v in constants.items() if k in ('stride_mixed_tok','stride_idx','H','HV','K','V','RMAX')},
        SCRATCH_WORDS=64, V_TILE=tile, num_warps=1)
    return producer, consumer


def compile_gate():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    pnames = ('mixed_qkv','a_gate','b_gate','A_log','dt_bias','a_ptr','u_ptr',
              'cnt_ptr','stale_ptr','ssm_state_indices','scratch','scale','gs_eps')
    ptypes = ('*bf16','*bf16','*bf16','*fp32','*fp32','*fp32','*fp16',
              '*i32','*i32','*i32','*fp32','fp32','fp32')
    cnames = ('mixed_qkv','vbar','w_ptr','ssm_state_indices','o','scratch')
    ctypes = ('*bf16','*fp32','*fp16','*i32','*bf16','*fp32')
    shared = dict(stride_mixed_tok=5120,stride_idx=1,H=8,HV=24,K=128,RMAX=16,SCRATCH_WORDS=64)
    variants = [(_factored_k_producer, dict(zip(pnames,ptypes)),
                 dict(shared,stride_a_tok=24,stride_b_tok=24,SOFTPLUS_THRESHOLD=20.))]
    variants += [(_factored_v_consumer, dict(zip(cnames,ctypes)),dict(shared,V=128,V_TILE=tile))
                 for tile in (64,32)]
    for fn,signature,constants in variants:
        compiled = triton.compile(ASTSource(fn=fn,signature=signature,constexprs=constants),
            target=GPUTarget('cuda',103,32), options={'num_warps':1})
        assert compiled.asm['cubin'] and compiled.asm['ptx']
    print('PFACTOR4_TWO_STAGE_COMPILE PASS sm103 producer/consumer64/consumer32',flush=True)


def compare(work, expected):
    return [dict(tensor=i,max_abs=float((work[i].float()-expected[i].float()).abs().max()))
            for i in (6,7,8,9,10,12) if not torch.equal(work[i],expected[i])]


def dynamic_replay(tile):
    source,constants = inputs(8,'cuda',True)
    source[11] = source[11].long()
    work = [x.clone() for x in source]
    scratch = torch.empty((8,24,64),device='cuda')
    launch(work,constants,scratch,tile)
    stream = torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        for dst,src in zip(work,source):dst.copy_(src)
        launch(work,constants,scratch,tile)
    for repeat in range(3):
        source[0].mul_(.75)
        source[11].copy_(torch.arange(7,-1,-1,device='cuda'))
        source[11][repeat] = -1
        expected = [x.clone() for x in source];original(expected,constants,False,1)
        graph.replay();torch.cuda.synchronize()
        failures = compare(work,expected)
        if failures:return dict(passed=False,repeat=repeat,failures=failures)
    return dict(passed=True,replays=3,dynamic_inputs=True,dynamic_i64_slots=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--compile-only',action='store_true')
    p.add_argument('--no-timing',action='store_true');p.add_argument('--output');args=p.parse_args()
    if args.compile_only:compile_gate();return
    device='cpu' if os.environ.get('TRITON_INTERPRET')=='1' else 'cuda'
    if device=='cpu':
        subprocess.run([sys.executable,__file__,'--compile-only'],check=True,
            env=dict(os.environ,TRITON_INTERPRET='0'),timeout=180)
    rows=[];timings=[];dynamic={}
    cases=[(1,False),(8,False),(8,True)]
    if device=='cuda':cases += [(32,False),(96,False)]
    for batch,padded in cases:
        for near in (False,True):
            source,constants=inputs(batch,device,near)
            if not padded:source[11]=torch.arange(batch,device=device,dtype=torch.int32)
            expected=[x.clone() for x in source];original(expected,constants,False,1)
            for tile in (64,32):
                work=[x.clone() for x in source]
                scratch=torch.empty((batch,24,64),device=device)
                compiled=launch(work,constants,scratch,tile)
                failures=compare(work,expected)
                row=dict(B=batch,padded=padded,near_span=near,tile=tile,passed=not failures,failures=failures,
                    scratch_bytes=scratch.numel()*scratch.element_size())
                if device=='cuda':row['resources']=[dict(registers=k.n_regs,spills=k.n_spills,shared=k.metadata.shared) for k in compiled]
                rows.append(row)
    admitted=[t for t in (64,32) if all(r['passed'] for r in rows if r['tile']==t)]
    if device=='cuda':
        for tile in admitted:dynamic[str(tile)]=dynamic_replay(tile)
        admitted=[t for t in admitted if dynamic[str(t)]['passed']]
    if device=='cuda' and not args.no_timing:
        for batch,padded in cases:
            for near in (False,True):
                source,constants=inputs(batch,device,near)
                if not padded:source[11]=torch.arange(batch,device=device,dtype=torch.int32)
                for repeat in range(2):
                    for tile in (0,*admitted):
                        work=[x.clone() for x in source];scratch=torch.empty((batch,24,64),device=device)
                        def step():
                            return launch(work,constants,scratch,tile) if tile else original(work,constants,False,1)
                        step()
                        pairs=[(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)) for _ in range(20)]
                        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,stream=stream):
                            for begin,end in pairs:
                                for dst,src in zip(work,source):dst.copy_(src)
                                begin.record();step();end.record()
                        graph.replay();torch.cuda.synchronize()
                        values=[a.elapsed_time(b) for a,b in pairs]
                        timings.append(dict(B=batch,padded=padded,near_span=near,repeat=repeat,tile=tile,
                            mean_ms=sum(values)/len(values),min_ms=min(values),max_ms=max(values),
                            samples=20,launches=2 if tile else 1))
    result=dict(passed=bool(admitted),complete=True,production_default=False,device=device,
        model_dtype='bfloat16',factor_dtype='float16',rows=rows,dynamic_replay=dynamic,
        numerically_admitted=admitted,timings=timings,timing_repeats=2 if timings else 0,
        gate='all a/U/W/count/stale/output bytes exact; failed tiles disabled for all shapes',
        timing_scope='producer+consumer including scratch; identical fixture reset excluded; step only, no expiry truncation')
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('PFACTOR4_TWO_STAGE_GATE',json.dumps(result),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
