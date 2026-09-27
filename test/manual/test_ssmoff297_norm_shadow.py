"""CPU interpreter checks for full-model shadow ownership and byte comparisons."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from test_ssmoff297_norm_step import fixture, kernels
from sglang.srt.layers.attention.linear.kernels import gdn_norm_diagnostic as diagnostic
from sglang.kernels.ops.attention.fla import layernorm_gated as norm


def main():
    records = []
    for rows in (1, 4):
        for count in (0, 8, 15):
            f = fixture(1, count, torch.int64, 2); p = f['p']
            kwargs = dict(A_log=f['log'][0], dt_bias=f['bias'][0], scale=f['k']**-.5,
                vbar=p.vbar[0], fa=p.a[0], fu=p.U[0], fw=p.W[0], fcount=p.count[0],
                stale=p.stale, ssm_state_indices=f['slots'], num_q_heads=f['h'],
                num_v_heads=f['hv'], head_k_dim=f['k'], head_v_dim=f['k'], r=8,
                rfull=16, truncate=False, kernel='split', prefix_valid=p.prefix_valid,
                conv_context=None, async_stream=None,
                norm_context=(f['z'][0], f['weight'][0], 1e-6, rows, 'sigmoid'))
            old = {k: kwargs[k].clone() for k in ('fa','fu','fw','fcount','stale','prefix_valid')}
            with patch.object(norm, 'calc_rows_per_block', return_value=rows), \
                 patch.object(norm, 'is_arch_support_pdl', return_value=False):
                record = diagnostic.before(0, f['x'][0], f['gates'][0,0], f['gates'][0,1], kwargs)
            assert all(torch.equal(v, kwargs[k]) for k,v in old.items())
            out = kernels.factored_packed_decode(f['x'][0], f['gates'][0,0], f['gates'][0,1], **kwargs)
            diagnostic.after(record, out, kwargs)
            assert not any(record['differences'].tolist())
            # Check mismatch reporting and capture use their own output storage.
            out.zero_()
            record['actual'].view(torch.uint8).flatten()[0] ^= 1
            diagnostic.after(record, record['actual'], kwargs)
            assert record['differences'][0].item() == 1
            with tempfile.TemporaryDirectory() as directory:
                diagnostic._saved.clear()
                diagnostic.save_step(directory, 0, 1)
                assert len(list(Path(directory).glob('*.pt'))) == 1
            records.append(dict(rows=rows, count=count, passed=True))
    print(json.dumps(dict(complete=True, diagnostic_only=True, passed=True, cases=records)))


if __name__ == '__main__': main()
