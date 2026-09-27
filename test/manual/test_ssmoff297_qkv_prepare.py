"""Exact comparison with production split and L2, at both post-conv layouts."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import torch
import triton
import triton.language as tl
from typing import Optional

GPU = os.environ.get('REPLAY_TEST_DEVICE') == 'cuda'
assert GPU or os.environ.get('CUDA_VISIBLE_DEVICES') == ''
if GPU:torch.set_default_device('cuda')
root = Path(__file__).resolve().parents[2] / 'python'
path = root / 'sglang/srt/mem_cache/gdn_prefill_qkv_prepare.py'
spec = importlib.util.spec_from_file_location('qkv_prepare_component', path)
candidate = importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
reference = dict(torch=torch, triton=triton, tl=tl, Optional=Optional, _is_hip=False)
references = {}
for file,names in [('sglang/kernels/ops/attention/fla/l2norm.py',
                   {'l2norm_fwd_kernel','l2norm_fwd_kernel1','l2norm_fwd'}),
                  ('sglang/kernels/ops/attention/triton_gdn_fused_proj.py',
                   {'fused_qkv_split_gdn_prefill_kernel','fused_qkv_split_gdn_prefill'})]:
    p=root/file;tree=ast.parse(p.read_text())
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in tree.body}==names
    exec(compile(tree,str(p),'exec'),reference)
    references[file]=hashlib.sha256(p.read_bytes()).hexdigest()


def make(tokens, heads, vh, width, layout, dtype):
    cols=(2*heads+vh)*width
    if layout=='token-major':return torch.randn(tokens,cols,dtype=dtype)
    return torch.randn(cols,tokens,dtype=dtype).transpose(0,1)


def same(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def baseline(mixed,h,hv,d):
    q,k,v=reference['fused_qkv_split_gdn_prefill'](mixed,h,h,hv,d,d,d)
    return reference['l2norm_fwd'](q),reference['l2norm_fwd'](k),v


def check(tokens,h,hv,d,layout,dtype,special=False):
    mixed=make(tokens,h,hv,d,layout,dtype)
    if special:
        mixed.zero_();mixed[:,::3]=-0.0;mixed[:,::7]=1e-12
    saved=mixed.clone();expected=baseline(mixed,h,hv,d);got=candidate.prepare(mixed,h,hv,d)
    assert all(same(x,y) for x,y in zip(expected,got)),('bytes',tokens,h,hv,d,layout,dtype)
    assert same(mixed,saved),'input modified'
    kept=tuple(x.clone() for x in got);mixed.fill_(3);other=candidate.prepare(mixed,h,hv,d)
    assert all(same(x,y) for x,y in zip(kept,got)),'previous output aliased'
    assert all(x.is_contiguous() for x in got+other)
    return dict(tokens=tokens,heads=h,value_heads=hv,width=d,layout=layout,dtype=str(dtype),
                bitwise=True,private_output=True,special=special)


def timing(tokens,layout):
    mixed=make(tokens,8,24,128,layout,torch.bfloat16)
    graphs={}
    for name,fn in [('reference',baseline),('candidate',candidate.prepare)]:
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(4):fn(mixed,8,24,128)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):out=fn(mixed,8,24,128)
        graphs[name]=(graph,out)
    pairs={k:[] for k in graphs}
    for pair in range(6):
        for name in (('reference','candidate') if pair%2==0 else ('candidate','reference')):
            graph=graphs[name][0]
            for _ in range(8):graph.replay()
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(64):graph.replay()
            b.record();b.synchronize();pairs[name].append(a.elapsed_time(b)/64)
    return dict(tokens=tokens,heads=8,value_heads=24,width=128,layout=layout,pairs_ms=pairs,
                medians_ms={k:statistics.median(v) for k,v in pairs.items()},
                scope='one-layer post-conv split plus Q/K L2 only; no model claim')


def main():
    torch.manual_seed(839)
    cases=[check(t,8 if GPU else 2,24 if GPU else 6,128 if GPU else 16,layout,torch.bfloat16)
           for t in (1,15,16,17,31,64,65,256) for layout in ('token-major','channel-major')]
    cases += [check(17,2,6,d,layout,dt,True) for d in (16,128)
              for layout in ('token-major','channel-major') for dt in (torch.float16,torch.float32)]
    if GPU:cases += [check(t,8,24,128,layout,torch.bfloat16) for t in (8192,16384,24577,32768)
                    for layout in ('token-major','channel-major')]
    result=dict(complete=True,passed=True,device='CUDA' if GPU else 'CPU',cases=cases,
                reference_sha256=references,production_default_enabled=False)
    if GPU:result['timings']=[timing(t,layout) for t in (256,8192,16384,32768)
                            for layout in ('token-major','channel-major')]
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
