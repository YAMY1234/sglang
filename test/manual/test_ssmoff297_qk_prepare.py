"""Compare against the unmodified production L2 kernel, including byte ownership."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Optional

import torch
import triton
import triton.language as tl

GPU = os.environ.get('REPLAY_TEST_DEVICE') == 'cuda'
assert GPU or os.environ.get('CUDA_VISIBLE_DEVICES') == ''
if GPU:
    torch.set_default_device('cuda')
root = Path(__file__).resolve().parents[2] / 'python'
path = root / 'sglang/srt/mem_cache/gdn_prefill_qk_prepare.py'
spec = importlib.util.spec_from_file_location('qk_prepare_candidate', path)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
# Compile the actual reference function AST unchanged, avoiding unrelated
# hardware imports during the same-image CPU interpreter check.
refpath = root / 'sglang/kernels/ops/attention/fla/l2norm.py'
tree = ast.parse(refpath.read_text())
names = {'l2norm_fwd_kernel', 'l2norm_fwd_kernel1', 'l2norm_fwd'}
tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
assert {n.name for n in tree.body} == names
reference = dict(torch=torch, triton=triton, tl=tl, Optional=Optional)
exec(compile(tree, str(refpath), 'exec'), reference)


def inputs(tokens, heads, width, layout, dtype):
    if layout == 'contiguous':
        return (torch.randn(1, tokens, heads, width, dtype=dtype),
                torch.randn(1, tokens, heads, width, dtype=dtype))
    if layout == 'packed':
        mixed = torch.randn(1, tokens, heads * 5, width, dtype=dtype)
        return mixed[:, :, :heads], mixed[:, :, heads:2*heads]
    if layout == 'transpose':
        return tuple(torch.randn(1, heads, tokens, width, dtype=dtype).transpose(1, 2)
                     for _ in range(2))
    return tuple(torch.randn(1, tokens, heads, width * 2, dtype=dtype)[..., ::2]
                 for _ in range(2))


def same(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype and
            torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def baseline(q, k):
    return tuple(reference['l2norm_fwd'](x.contiguous()) for x in (q, k))


def check(tokens, heads, width, layout, dtype, special=False):
    q, k = inputs(tokens, heads, width, layout, dtype)
    if special:
        q.zero_(); k.fill_(-0.0)
        q[..., 0] = 1e-12; k[..., -1] = -1e-12
    saved = (q.clone(), k.clone())
    expected = baseline(q, k)
    got = candidate.prepare(q, k)
    assert all(same(a, b) for a, b in zip(got, expected)), ('Q/K bytes differ', tokens, layout, dtype)
    assert same(q, saved[0]) and same(k, saved[1]), 'source modified'
    kept = tuple(x.clone() for x in got)
    q.fill_(3); k.fill_(5)
    other = candidate.prepare(q, k)
    assert all(same(a, b) for a, b in zip(got, kept)), 'previous output aliased'
    assert all(x.is_contiguous() for x in got + other)
    return dict(tokens=tokens, heads=heads, width=width, layout=layout,
                dtype=str(dtype), special=special, bitwise=True, private_output=True)


def timing(tokens, layout):
    q, k = inputs(tokens, 8, 128, layout, torch.bfloat16)
    functions = dict(reference=baseline, candidate=candidate.prepare)
    graphs = {}
    for name, fn in functions.items():
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(4): fn(q, k)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream): out = fn(q, k)
        graphs[name] = (graph, out)
    pairs = {name: [] for name in functions}
    for pair in range(6):
        order = ('reference', 'candidate') if pair % 2 == 0 else ('candidate', 'reference')
        for name in order:
            graph = graphs[name][0]
            for _ in range(8): graph.replay()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(64): graph.replay()
            b.record(); b.synchronize()
            pairs[name].append(a.elapsed_time(b) / 64)
    return dict(tokens=tokens, layout=layout, heads=8, width=128, pairs_ms=pairs,
                medians_ms={k:statistics.median(v) for k,v in pairs.items()},
                scope='one-layer Q/K preparation at stated layout; no model claim')


def dispatch_check():
    """Execute the actual dispatcher method and inspect its chunk-kernel contract."""
    path = root / 'sglang/srt/layers/attention/linear/kernels/gdn_triton.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TritonGDNKernel')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'extend')
    calls = []
    def chunk(**kwargs):
        calls.append(kwargs)
        return None, None, None
    ns = dict(torch=torch, is_cpu=lambda:False, is_npu=lambda:False,
              chunk_gated_delta_rule=chunk)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
    name = 'sglang.srt.mem_cache.gdn_prefill_qk_prepare'
    old = sys.modules.get(name)
    sys.modules[name] = candidate
    try:
        q, k = inputs(17, 2, 16, 'packed', torch.bfloat16)
        args = dict(q=q, k=k, v=q, g=q, beta=q, ssm_states=q,
                    cache_indices=None, query_start_loc=None)
        ns['extend'](None, **args)
        assert calls[-1]['q'] is q and calls[-1]['k'] is k
        assert calls[-1]['use_qk_l2norm_in_kernel'] is True
        ns['extend'](None, **args, factored_qk_prepare=True)
        assert calls[-1]['use_qk_l2norm_in_kernel'] is False
        expected = baseline(q, k)
        assert same(calls[-1]['q'], expected[0]) and same(calls[-1]['k'], expected[1])
        assert calls[-1]['initial_state'] is q and calls[-1]['inplace_update'] is True
    finally:
        if old is None:sys.modules.pop(name, None)
        else:sys.modules[name] = old
    return dict(passed=True, cases=2, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    torch.manual_seed(838)
    cases = [check(n, 8 if GPU else 2, 128 if GPU else 16, layout, torch.bfloat16)
             for n in (1, 15, 16, 17, 31, 64, 65, 256)
             for layout in ('contiguous', 'packed', 'transpose', 'slice')]
    cases += [check(17, 3, d, 'packed', dt, special=True)
              for d in (16, 128) for dt in (torch.float16, torch.float32)]
    if GPU:
        cases += [check(n, 8, 128, 'packed', torch.bfloat16)
                  for n in (8192, 16384, 24577, 32768)]
    result = dict(complete=True, passed=True, device='CUDA' if GPU else 'CPU', cases=cases,
                  reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),
                  production_default_enabled=False, dispatch=dispatch_check())
    if GPU: result['timings'] = [timing(n, layout) for layout in ('contiguous', 'packed') for n in (256, 8192, 16384, 32768)]
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
