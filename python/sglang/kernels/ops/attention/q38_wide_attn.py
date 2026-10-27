"""M8/M12 wrapper for the selected BF16 QSA core and gated merge.

Each request still has WIDTH=4. The original partial kernel resolves fresh
KV with row//WIDTH and row%WIDTH, so two or three requests remain isolated.
M4 continues to use the original run_gated wrapper.
"""

import torch
import triton

from sglang.kernels.ops.attention.qsa_gate_epilogue_v2 import _merge_gate_pdl_v2
from sglang.kernels.ops.attention.qsa_tcgen05_v1 import _store, partial_tcgen05


def run_gated_wide(q, k_pool, v_pool, k_new, v_new, loc, req_to_token,
                   row_requests, block_indices, lengths, qpos, scale, gate,
                   width=4):
    m, nh, d = q.shape
    h = k_pool.shape[1]
    assert m in (8, 12) and width == 4 and nh == 24 and h == 2 and d == 256
    assert gate.dtype == torch.bfloat16 and gate.stride(-1) == 1
    assert tuple(gate.shape) in ((m, 24, 256), (m, 6144))
    assert block_indices.shape == (m, 512) and block_indices.stride(1) == 1
    assert all(t.dtype == torch.bfloat16 for t in (q, k_pool, v_pool, k_new, v_new))
    assert all(t.stride(-1) == 1 and t.stride(-2) == d
               for t in (q, k_pool, v_pool, k_new, v_new))
    nt = triton.cdiv(512 * 4 + 3, 64)
    partial = torch.empty((m, h, nt, 16, d), dtype=torch.float32, device=q.device)
    maximum = torch.empty((m, h, nt, 16), dtype=torch.float32, device=q.device)
    sums = torch.empty_like(maximum)
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    _store[(m, h)](
        k_pool, v_pool, k_new, v_new, loc,
        k_pool.stride(0), v_pool.stride(0), k_new.stride(0), v_new.stride(0), h, d,
        num_warps=4,
    )
    partial_tcgen05[(m, nt, h)](
        q, k_pool, v_pool, k_new, v_new, loc, req_to_token, row_requests,
        block_indices, lengths, qpos, partial, maximum, sums,
        QS=q.stride(0), QH=q.stride(1), KS=k_pool.stride(0), VS=v_pool.stride(0),
        NKS=k_new.stride(0), NVS=v_new.stride(0), MS=req_to_token.stride(0),
        IS=block_indices.stride(0), NT=nt, GROUP=12, WIDTH=width, SCALE=float(scale),
        num_warps=4,
    )
    _merge_gate_pdl_v2[(m, nh, triton.cdiv(d, 64))](
        partial, maximum, sums, output, gate, gate.stride(0),
        gate.stride(1) if gate.ndim == 3 else 256, nt, triton.next_power_of_2(nt),
        num_warps=4, launch_pdl=True,
    )
    return output
