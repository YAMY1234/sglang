"""CPU-only counterexample audit; no production operator is changed.

Compare original W8 post-order cuts with four untruncated candidate updates.
RMAX=32 on both sides isolates cut semantics without an out-of-bounds append.
RMAX16 capacity is audited separately. This is not a full-model quality gate.
"""
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType

if os.environ.get('TRITON_INTERPRET') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('CPU interpreter with CUDA hidden required')

import torch


def main():
    root = Path(__file__).resolve().parents[2]
    pkg = ModuleType('audit298_kernels')
    pkg.__path__ = [str(root/'python/sglang/srt/layers/attention/linear/kernels')]
    sys.modules[pkg.__name__] = pkg
    kernel = importlib.import_module(pkg.__name__+'.gdn_factored')
    torch.manual_seed(298)
    key = value = 128
    heads = 1
    qkv = torch.randn(4, 1, 2*key+value, dtype=torch.bfloat16)
    ga = torch.randn(4, 1, heads, dtype=torch.bfloat16)
    gb = torch.randn_like(ga)
    const = dict(A_log=torch.randn(heads), dt_bias=torch.randn(heads),
        vbar=torch.randn(heads, value)*.01, num_q_heads=1, num_v_heads=heads,
        head_k_dim=key, head_v_dim=value, r=8, rfull=16, scale=key**-.5,
        kernel='split', post_order=True, ssm_state_indices=torch.tensor([0]))
    rows = []
    for count in range(8, 16):
        # Identical padded layouts keep reduction shape fixed within each pair.
        # Coordinate basis gives a valid orthonormal initial key-side state.
        initial = dict(fa=torch.randn(1,heads,key)*.01,
            fu=torch.eye(key, dtype=torch.float16)[:32].reshape(1,heads,32,key).clone(),
            fw=torch.randn(1,heads,32,value,dtype=torch.float16)*.1,
            fcount=torch.full((1,heads),count,dtype=torch.int32),
            stale=torch.zeros(1,dtype=torch.int32))
        def clone(): return {k:v.clone() for k,v in initial.items()}
        def run(state, consumed, truncate):
            outputs=[]
            for step in range(consumed):
                outputs.append(kernel.factored_packed_decode(qkv[step],ga[step],gb[step],
                    **state, **const, truncate=truncate).clone())
            return outputs
        old, deferred = clone(), clone()
        expected = run(old, 4, True)
        actual = run(deferred, 4, False)
        matches = [torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(expected,actual)]
        row=dict(initial_count=count, required_rows=count+4,
            fits_rmax16=count+4<=16, cut_after_input_1based=16-count if count>=12 else None,
            output_bitwise_by_input=matches,
            first_output_difference_1based=next((i+1 for i,v in enumerate(matches) if not v),None),
            output_max_abs_by_input=[float((a.float()-b.float()).abs().max()) for a,b in zip(expected,actual)],
            original_final_count=int(old['fcount'].item()),
            deferred_final_count=int(deferred['fcount'].item()))
        rows.append(row)
        print(json.dumps(row), file=sys.stderr, flush=True)
    assert all(all(r['output_bitwise_by_input']) for r in rows[:5])
    assert all(not r['fits_rmax16'] for r in rows[5:])
    assert all(r['first_output_difference_1based'] is not None for r in rows[5:])
    print(json.dumps(dict(complete=True, device='CPU', triton_interpret=True,
        comparison_rmax=32, production_rmax=16, rows=rows,
        scope='actual recurrence/cut kernels on identical RMAX32 layouts; capacity proof for production RMAX16; not full-model logits or quality admission')))


if __name__ == '__main__':
    main()
