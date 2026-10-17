"""Synthetic GB300 primitive: preserve the old fp32 aligned checkpoint.

This qualifies the existing chunk kernel's fp32 snapshot mechanism; it does
not replace the actual P/radix gate or any service numerical gate.
"""
import argparse
import json
from pathlib import Path


def main():
    import torch
    from sglang.kernels.ops.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from sglang.kernels.ops.attention.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from sglang.kernels.ops.attention.fla.chunk_o import chunk_fwd_o
    from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
    from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
    from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(232)
    records = []
    for length in (65, 129, 4097):
        q, k, v = [torch.randn(1, length, 24, 128, device='cuda', dtype=torch.bfloat16)
                   for _ in range(3)]
        g = -torch.rand(1, length, 24, device='cuda') * .02
        beta = torch.rand(1, length, 24, device='cuda', dtype=torch.bfloat16)

        def run(tokens, snapshot=False):
            cu = torch.tensor([0, tokens], device='cuda', dtype=torch.int32)
            indices = prepare_chunk_indices(cu, 64)
            qn, kn = l2norm_fwd(q[:, :tokens].contiguous()), l2norm_fwd(k[:, :tokens].contiguous())
            decay = chunk_local_cumsum(g[:, :tokens].contiguous(), chunk_size=64,
                                       cu_seqlens=cu, chunk_indices=indices)
            w, u, _ = chunk_gated_delta_rule_fwd_intra(
                k=kn, v=v[:, :tokens].contiguous(), g=decay,
                beta=beta[:, :tokens].contiguous(), cu_seqlens=cu, chunk_indices=indices)
            state = torch.zeros(1, 24, 128, 128, device='cuda')
            tracked = torch.empty_like(state) if snapshot else None
            h, vn = chunk_gated_delta_rule_fwd_h(
                k=kn, w=w, u=u, g=decay, initial_state=state,
                initial_state_indices=torch.zeros(1, device='cuda', dtype=torch.int32),
                cu_seqlens=cu, chunk_indices=indices,
                track_state=tracked, track_chunk_idx=(
                    torch.tensor([(length - 1) // 64], device='cuda', dtype=torch.int32)
                    if snapshot else None))
            out = chunk_fwd_o(q=qn, k=kn, v=vn, h=h, g=decay,
                              scale=128 ** -.5, cu_seqlens=cu)
            return state, h, out, tracked

        old, _, _, _ = run(length - 1)
        final, h, out, _ = run(length)
        final2, h2, out2, tracked = run(length, snapshot=True)
        torch.cuda.synchronize()
        row = dict(length=length, old_dtype=str(old.dtype), h_dtype=str(h.dtype),
                   snapshot_bitwise=torch.equal(tracked, old),
                   rounded_h_negative_detected=not torch.equal(h[:, (length - 1) // 64].float(), old),
                   final_unchanged=torch.equal(final, final2), h_unchanged=torch.equal(h, h2),
                   output_unchanged=torch.equal(out, out2))
        assert all(row[key] for key in ('snapshot_bitwise', 'rounded_h_negative_detected',
                                        'final_unchanged', 'h_unchanged', 'output_unchanged')), row
        records.append(row)
        print('P3_CHECKPOINT_SNAPSHOT', json.dumps(row), flush=True)
    args.out.write_text(json.dumps(dict(passed=True, scope='synthetic primitive only',
                                      device=torch.cuda.get_device_name(), cases=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
