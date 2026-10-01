"""Fused orth candidate: real shape, adversarial ranks, fixed numerical gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from sglang.srt.layers.attention.linear.kernels.gdn_k31_orth import orth_cholqr2, _orth_cholqr2_kernel
from sglang.srt.layers.attention.linear.kernels.gdn_prefill_reference import _orth_cholqr2


def compile_gate():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    result = triton.compile(ASTSource(fn=_orth_cholqr2_kernel,
        signature={'Y_ptr': '*fp32', 'Q_ptr': '*fp32'}, constexprs={'ROWS': 128, 'COLS': 16}),
        target=GPUTarget('cuda', 103, 32), options={'num_warps': 4, 'enable_fp_fusion': False})
    assert result.asm['ptx'] and result.asm['cubin']
    print('PFACTOR4_ORTH_COMPILE PASS sm103', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--no-timing', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.compile_only:
        compile_gate(); return
    device = 'cpu' if os.environ.get('TRITON_INTERPRET') == '1' else 'cuda'
    if device == 'cpu':
        subprocess.run([sys.executable, __file__, '--compile-only'], check=True,
                       env=dict(os.environ, TRITON_INTERPRET='0'), timeout=180)
    torch.manual_seed(0x504634)
    rows = []
    for batch in ((1,) if device == 'cpu' else (1, 8)):
        heads = 2 if device == 'cpu' else 24
        original = torch.randn(batch, heads, 128, 16, device=device)
        for kind in ('full', 'rank4', 'zero', 'scaled'):
            y = original.clone()
            if kind == 'rank4': y[..., 4:] = y[..., :1] * 1.e-8
            if kind == 'zero': y.zero_()
            if kind == 'scaled': y *= 1.e-12
            expected = _orth_cholqr2(y)
            actual = orth_cholqr2(y)
            torch.testing.assert_close(actual, expected, atol=2.e-6, rtol=2.e-6)
            row = dict(B=batch, heads=heads, kind=kind, passed=True,
                       max_abs=float((actual-expected).abs().max()))
            if device == 'cuda' and not args.no_timing:
                for label, fn in (('reference', _orth_cholqr2), ('candidate', orth_cholqr2)):
                    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3): fn(y)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    pairs = [(torch.cuda.Event(enable_timing=True, external=True), torch.cuda.Event(enable_timing=True, external=True)) for _ in range(20)]
                    with torch.cuda.graph(graph, stream=stream):
                        for begin, end in pairs:
                            begin.record(); fn(y); end.record()
                    graph.replay(); torch.cuda.synchronize()
                    values = [a.elapsed_time(b) for a,b in pairs]
                    row[label+'_ms'] = dict(mean=sum(values)/len(values), min=min(values), max=max(values))
            rows.append(row)
    result = dict(passed=True, device=device, production_default=False, rows=rows)
    if args.output: Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print('PFACTOR4_ORTH_GATE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
