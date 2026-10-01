"""CPU interpreter state/scratch checks and offline sm103 ptxas resource gate.

Pass --mode arithmetic with TRITON_INTERPRET=1, then --mode compile with 0.
All generated code/PTX/cubin/receipts stay in the explicit private --out path.
"""
import argparse
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


def load(args):
    original = (args.source/'gdn_factored.py').read_text()
    nodes = ast.parse(original).body
    def extract(name):
        n = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
        return '\n'.join(original.splitlines()[n.decorator_list[0].lineno-1:n.end_lineno])
    candidate = (args.source/'gdn_expiry_two_stage.py').read_text()
    candidate = candidate.replace('from .gdn_factored import _mgs', extract('_mgs'))
    path = args.out/'actual_kernels.py'
    path.write_text(candidate+'\n'+extract('_factored_expiry_truncate_kernel')+'\n')
    spec = importlib.util.spec_from_file_location('r6_actual_kernels', path)
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def arithmetic(a, m):
    import torch
    assert os.environ.get('TRITON_INTERPRET') == '1' and not torch.cuda.is_available()
    torch.manual_seed(1619); records = []
    for kind in ('random', 'rank_deficient', 'tied', 'zero'):
        for stride in (1, 2):
            heads = 2
            u = torch.randn(6, heads, 32, 128); w = torch.randn_like(u)
            if kind == 'rank_deficient': w[:, :, 12:] = w[:, :, :1]
            if kind == 'tied':
                w.zero_(); w[:, :, :, :32] = torch.eye(32)
            if kind == 'zero': w.zero_()
            count = torch.tensor([[32, 16], [31, 33], [16, 32], [32, 32], [0, 31], [32, 32]], dtype=torch.int32)
            index = torch.tensor([0, -1, 2, 4, 1], dtype=torch.int64)
            if stride == 2:
                full = torch.zeros(10, dtype=torch.int64); full[::2] = index; index = full[::2]
            ref_u, ref_w, ref_c = u.clone(), w.clone(), count.clone()
            m._factored_expiry_truncate_kernel[(index.numel()*heads,)](
                ref_u, ref_w, ref_c, index, stride_idx=stride, HV=heads,
                K=128, V=128, RMAX=32, R=16, RFULL=32, ITERS=3, REL_TOL=1e-4,
                RECT=False, VERIFY_GATHER=False, num_warps=4)
            bank = m.ExpiryWorkspaceBank()
            scratch = bank.get(0, index.numel(), heads, 'cpu', 101, capturing=False)
            scratch[0].fill_(float('nan')); scratch[1].fill_(6)
            m.truncate_two_stage(u, w, count, index, scratch)
            assert torch.equal(count, ref_c)
            torch.testing.assert_close(u, ref_u, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(w, ref_w, rtol=2e-5, atol=2e-6)
            assert torch.equal(scratch[1][1], torch.zeros(heads, dtype=torch.int32)), 'padding reused scratch'
            assert torch.equal(scratch[1][3], torch.zeros(heads, dtype=torch.int32)), 'nonexpiry reused scratch'
            # Slots 3/5 never selected; rows >=16 and non-expiry heads are byte-exact.
            assert torch.equal(u[3], ref_u[3]) and torch.equal(w[5], ref_w[5])
            assert torch.equal(u[:, :, 16:], ref_u[:, :, 16:])
            assert torch.equal(w[:, :, 16:], ref_w[:, :, 16:])
            records.append(dict(kind=kind, index_stride=stride, max_abs_u=float((u-ref_u).abs().max()),
                                max_abs_w=float((w-ref_w).abs().max())))
    bank = m.ExpiryWorkspaceBank(); live = []
    for layer in (0, 1, 35):
        for batch in (1, 64):
            for stream in (101, 202):
                z, active = bank.get(layer, batch, 24, 'cpu', stream, capturing=False)
                live.extend([z, active])
                again = bank.get(layer, batch, 24, 'cpu', stream, capturing=True)
                assert again[0] is z and again[1] is active
    assert len({x.untyped_storage().data_ptr() for x in live}) == len(live), 'scratch aliases across owners'
    try: bank.get(0, 8, 24, 'cpu', 101, capturing=True)
    except RuntimeError: pass
    else: raise AssertionError('new capture shape allocated scratch')
    # Real interleaving: directions A, directions B, project A, project B.
    states = []; expected = []
    for layer in (0, 1):
        u = torch.randn(2, 1, 32, 128); w = torch.randn_like(u)
        c = torch.full((2, 1), 32, dtype=torch.int32); index = torch.tensor([0, 1])
        z, active = bank.get(layer, 2, 1, 'cpu', 303, capturing=False)
        ru, rw, rc = u.clone(), w.clone(), c.clone()
        m._factored_expiry_truncate_kernel[(2,)](ru, rw, rc, index, stride_idx=1, HV=1,
            K=128, V=128, RMAX=32, R=16, RFULL=32, ITERS=3, REL_TOL=1e-4,
            RECT=False, VERIFY_GATHER=False, num_warps=4)
        m._expiry_directions_kernel[(2,)](w, c, index, z, active, HV=1, STRIDE_IDX=1, num_warps=2)
        states.append((u, w, c, z, active)); expected.append((ru, rw, rc))
    for state, ref in zip(states, expected):
        m._expiry_project_kernel[(2,)](*state, HV=1, num_warps=4)
        for x, y in zip(state[:3], ref): torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-6)
    return dict(passed=True, cases=records, disjoint_buffers=len(live), interleaved_layers=2,
                captured_shape_allocation_rejected=True, gpu_math_executed=False)


def compile_gate(a, m):
    import torch
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    assert os.environ.get('TRITON_INTERPRET') == '0' and not torch.cuda.is_available()
    # Triton ships separate assemblers: the generic one may predate sm103a.
    # Its Blackwell binary is also used by the successful Triton compile above.
    binaries = Path(triton.__file__).parent/'backends/nvidia/bin'
    ptxas = next((p for p in (binaries/'ptxas-blackwell', Path('/usr/local/cuda/bin/ptxas'),
                              binaries/'ptxas') if p.exists()), binaries/'ptxas')
    assert ptxas.exists(), ptxas
    rows = []
    for name, kernel, warps, constants, types, reg_limit, blocks_limit in (
        ('A', m._expiry_directions_kernel, 2, dict(HV=24, STRIDE_IDX=1),
         ['*fp32', '*i32', '*i64', '*fp32', '*i32'], 192, 5),
        ('B', m._expiry_project_kernel, 4, dict(HV=24),
         ['*fp32', '*fp32', '*i32', '*fp32', '*i32'], 128, 4)):
        signature = dict(zip(kernel.arg_names, types))
        signature.update({k:'constexpr' for k in constants})
        compiled = triton.compile(ASTSource(kernel, signature=signature, constexprs=constants),
            target=GPUTarget('cuda', 103, 32), options={'num_warps':warps})
        ptx = a.out/f'{name}.ptx'; ptx.write_text(compiled.asm['ptx'])
        arch = re.search(r'\.target\s+(\w+)', compiled.asm['ptx']).group(1)
        cmd = [str(ptxas), '-v', '--gpu-name='+arch, str(ptx), '-o', str(a.out/f'{name}.cubin')]
        result = subprocess.run(cmd, text=True, capture_output=True)
        (a.out/f'{name}-ptxas.txt').write_text(result.stdout+result.stderr); result.check_returncode()
        regs = int(re.search(r'Used (\d+) registers', result.stderr).group(1))
        shared = int(compiled.metadata.shared)
        alloc_regs = math.ceil(regs*32/256)*256*warps
        alloc_shared = math.ceil(shared/256)*256+1024
        blocks = min(65536//alloc_regs, 233472//alloc_shared, 64//warps, 32)
        rows.append(dict(kernel=name, registers=regs, shared=shared, allocated_regs_estimate=alloc_regs,
            allocated_shared_estimate=alloc_shared, blocks_per_SM_estimate=blocks,
            resident_warps_estimate=blocks*warps, ptxas_command=cmd, ptxas=result.stderr,
            passed=regs<=reg_limit and blocks>=blocks_limit))
        print(json.dumps(rows[-1]), flush=True)
    return dict(passed=all(r['passed'] for r in rows), rows=rows, gpu_math_executed=False,
                target='sm103', rule='A<=192regs and >=5blocks; B<=128regs and >=4blocks; no GPU if failed')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=('arithmetic', 'compile'), required=True)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(exist_ok=True, parents=True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    result = {'arithmetic':arithmetic, 'compile':compile_gate}[a.mode](a, load(a))
    (a.out/(a.mode+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
