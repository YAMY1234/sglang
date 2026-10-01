"""Fused orth candidate: real shape, adversarial ranks, fixed numerical gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import torch
from sglang.srt.layers.attention.linear.kernels.gdn_k31_orth import orth_cholqr2, _orth_cholqr2_kernel
from sglang.srt.layers.attention.linear.kernels.gdn_prefill_reference import _orth_cholqr2_reference as _orth_cholqr2


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
    # Test the actual consumer, including fp32 matmuls, Jacobi on CUDA and
    # final bf16 factor stores. A small Q error alone is insufficient evidence.
    from sglang.srt.layers.attention.linear.kernels import gdn_prefill_reference as reference
    factors = []
    for batch in ((1,) if device == 'cpu' else (1, 8)):
        heads = 2 if device == 'cpu' else 24
        dense = torch.randn(batch, heads, 128, 128, device=device)
        vbar = torch.randn(heads, 128, device=device)
        omega = torch.randn(batch, heads, 128, 16, device=device)
        for kind in ('full', 'rank4', 'zero', 'scaled'):
            state = dense.clone()
            if kind == 'rank4': state = state[..., :4] @ state[..., :4, :]
            if kind == 'zero': state.zero_()
            if kind == 'scaled': state *= 1.e-12
            with patch.object(reference, 'K31_FUSED_ORTH', False):
                expected = reference.factorize_prefill_k31(state, vbar, 8, 16, torch.float16, omega)
            if device == 'cuda':
                # Exercise the production dispatcher, not a substitute caller.
                with patch.object(reference, 'K31_FUSED_ORTH', True):
                    actual = reference.factorize_prefill_k31(state, vbar, 8, 16, torch.float16, omega)
            else:
                with patch.object(reference, '_orth_cholqr2', orth_cholqr2):
                    actual = reference.factorize_prefill_k31(state, vbar, 8, 16, torch.float16, omega)
            errors = {}
            for label, new, old in zip(('a', 'U', 'W'), actual, expected):
                close = torch.isclose(new, old, atol=2.e-6, rtol=2.e-6)
                errors[label] = dict(max_abs=float((new.float()-old.float()).abs().max()),
                                     exact=torch.equal(new, old), passed=bool(close.all()),
                                     mismatched=int((~close).sum()), elements=new.numel(),
                                     dtype=str(new.dtype), shape=list(new.shape))
            factors.append(dict(B=batch, heads=heads, kind=kind,
                                passed=all(v['passed'] for v in errors.values()), tensors=errors))
    # Persist every consumer failure before rejecting the candidate. In
    # particular, a Q-only pass must not hide a changed retained factor basis.
    result = dict(passed=all(v['passed'] for v in factors), device=device, production_default=False, rows=rows,
                  factorization_rows=factors, factorization_gate='a/U/W atol=rtol=2e-6')
    if args.output: Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print('PFACTOR4_ORTH_GATE', json.dumps(result), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
