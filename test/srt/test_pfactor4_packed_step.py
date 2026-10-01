"""Served bf16-QKV/fp16-state packed-step parity and bounded event timing after the service is idle."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from sglang.srt.layers.attention.linear.kernels.gdn_factored import _factored_packed_step_kernel


def inputs(batch, device, near_span=False):
    torch.manual_seed(0x504634)
    heads, qheads, width, rank = 24, 8, 128, 16
    slots = batch+1
    mixed = torch.randn(batch, 2*qheads*width+heads*width, device=device).bfloat16()
    gates = [torch.randn(batch, heads, device=device).bfloat16() for _ in range(2)]
    if near_span:
        basis = torch.eye(width, device=device)[:rank].expand(slots, heads, rank, width).contiguous().half()
    else:
        basis = torch.linalg.qr(torch.randn(slots, heads, width, rank, device=device)).Q.transpose(-1,-2).contiguous().half()
    if near_span:
        mixed[:, qheads*width:2*qheads*width].zero_()
        mixed[:, qheads*width:qheads*width+8] = .25
    count = torch.arange(slots*heads, device=device, dtype=torch.int32).reshape(slots, heads)%8+8
    indices = torch.arange(batch, device=device, dtype=torch.int32)
    if batch>1:indices[-1]=-1
    tensors = [mixed, *gates, torch.randn(heads, device=device), torch.randn(heads, device=device),
       torch.randn(heads, width, device=device), torch.randn(slots, heads, width, device=device),
       basis, torch.randn(slots, heads, rank, width, device=device).half(), count,
       torch.zeros(slots, dtype=torch.int32, device=device), indices,
       torch.zeros(batch, 1, heads, width, device=device, dtype=torch.bfloat16)]
    constants = dict(stride_mixed_tok=mixed.stride(0), stride_a_tok=heads,
        stride_b_tok=heads, stride_idx=1, H=qheads, HV=heads, K=width, V=width,
        RMAX=rank, SOFTPLUS_THRESHOLD=20.)
    return tensors, constants


def launch(tensors, constants, late, warps):
    return _factored_packed_step_kernel[(tensors[0].shape[0]*24,)](
        *tensors, 128**-.5, 1e-4, **constants, LATE_W_LOAD=late, num_warps=warps)


def compile_gate():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    signature={name:kind for name,kind in zip(
       ('mixed_qkv','a_gate','b_gate','A_log','dt_bias','vbar','a_ptr','u_ptr','w_ptr',
        'cnt_ptr','stale_ptr','ssm_state_indices','o','scale','gs_eps'),
       ('*bf16','*bf16','*bf16','*fp32','*fp32','*fp32','*fp32','*fp16','*fp16',
        '*i32','*i32','*i32','*bf16','fp32','fp32'))}
    constants=dict(stride_mixed_tok=5120,stride_a_tok=24,stride_b_tok=24,stride_idx=1,
                   H=8,HV=24,K=128,V=128,RMAX=16,SOFTPLUS_THRESHOLD=20.)
    for late,warps in ((False,1),(True,1),(True,2),(True,4)):
        result=triton.compile(ASTSource(fn=_factored_packed_step_kernel,signature=signature,
            constexprs=dict(constants,LATE_W_LOAD=late)),target=GPUTarget('cuda',103,32),
            options={'num_warps':warps})
        assert result.asm['ptx'] and result.asm['cubin']
    print('PFACTOR4_STEP_COMPILE PASS sm103 late=False/True warps=1/2/4',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--compile-only',action='store_true')
    parser.add_argument('--output');parser.add_argument('--no-timing',action='store_true');args=parser.parse_args()
    if args.compile_only:compile_gate();return
    device='cpu' if os.environ.get('TRITON_INTERPRET')=='1' else 'cuda'
    if device=='cpu':
        subprocess.run([sys.executable,__file__,'--compile-only'],check=True,
                       env=dict(os.environ,TRITON_INTERPRET='0'),timeout=180)
    # Captured events must create record nodes, otherwise elapsed_time has
    # no timestamp. Check this before starting a costly persistent service.
    import inspect
    inspect.signature(torch.cuda.Event).bind(enable_timing=True, external=True)
    if device == 'cuda':
        sample = torch.empty(64, device=device)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        begin = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph, stream=stream):
            begin.record(); sample.zero_(); end.record()
        graph.replay(); torch.cuda.synchronize()
        assert begin.elapsed_time(end) >= 0
    records=[]
    for batch in (1,8):
        for near in (False,True):
            source,constants=inputs(batch,device,near)
            old=[v.clone() for v in source];launch(old,constants,False,1)
            for late,warps in ((False,1),(True,1),(True,2),(True,4)):
                work=[v.clone() for v in source];compiled=launch(work,constants,late,warps)
                failures=[]
                for index in (6,7,8,9,10,12):
                    # Same-warp scheduling must retain exact state/output.
                    try:
                        if warps==1:assert torch.equal(old[index],work[index]),(batch,near,index)
                        else:torch.testing.assert_close(work[index],old[index],atol=2e-6,rtol=2e-6)
                    except AssertionError as exc:
                        failures.append(dict(tensor=index,message=str(exc),
                            max_abs=float((work[index].float()-old[index].float()).abs().max())))
                record=dict(B=batch,near_span=near,late=late,warps=warps,
                    passed=not failures,failures=failures,production_variant=warps==1)
                # Exploratory variants keep the exact original gate and their
                # failures. They are never timed or admitted after a failure.
                if device=='cuda' and not args.no_timing and not failures and warps==1:
                    # Each replay includes a separate reset so count never
                    # advances beyond RMAX. Events enclose only the step.
                    graph=torch.cuda.CUDAGraph();stream=torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    times=[(torch.cuda.Event(enable_timing=True, external=True),torch.cuda.Event(enable_timing=True, external=True)) for _ in range(20)]
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            for dst,src in zip(work,source):dst.copy_(src)
                            launch(work,constants,late,warps)
                    torch.cuda.current_stream().wait_stream(stream)
                    with torch.cuda.graph(graph,stream=stream):
                        for begin,end in times:
                            for dst,src in zip(work,source):dst.copy_(src)
                            begin.record();launch(work,constants,late,warps);end.record()
                    graph.replay();torch.cuda.synchronize()
                    values=[a.elapsed_time(b) for a,b in times]
                    record.update(mean_ms=sum(values)/len(values),min_ms=min(values),max_ms=max(values),
                        registers=compiled.n_regs,spills=compiled.n_spills,shared=compiled.metadata.shared)
                records.append(record)
    passed=all(r['passed'] for r in records if r['production_variant'])
    result=dict(passed=passed,device=device,model_dtype='bfloat16',factor_dtype='float16',records=records,performance=device=='cuda' and not args.no_timing,
        required='late=False/True, warps=1, exact state/output; every B/near-span case',
        exploratory_gate='warps=2/4: unchanged atol=rtol=2e-6; failures disabled, not timed')
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('PFACTOR4_STEP_GATE',json.dumps(result),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
