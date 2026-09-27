"""Opt-in exact QKV split plus original Q/K L2 normalization.

The backend enables this only through PREFILL_QKV_PREPARE (default off). Q/K retain the original BT16,
8-warp L2 expression; V is copied without arithmetic. Both token-major and
channel-major post-convolution tensors are supported without a temporary split.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def _prepare(MIXED, Q, K, V, T, ST: tl.constexpr, SD: tl.constexpr,
             H: tl.constexpr, HV: tl.constexpr, D: tl.constexpr,
             BT: tl.constexpr, BD: tl.constexpr, EPS: tl.constexpr):
    pid = tl.program_id(0)
    q_blocks = tl.cdiv(T * H, BT)
    if pid < q_blocks * 2:
        if pid < q_blocks:
            block, output, begin = pid, Q, 0
        else:
            block, output, begin = pid - q_blocks, K, H * D
        row = block * BT + tl.arange(0, BT)
        col = tl.arange(0, BD)
        address = (row // H)[:, None] * ST + (begin + (row % H)[:, None] * D + col[None, :]) * SD
        b_x = tl.load(MIXED + address,
                      mask=(row[:, None] < T * H) & (col[None, :] < D), other=0).to(tl.float32)
        b_var = tl.sum(b_x * b_x, axis=1)
        b_y = b_x / tl.sqrt(b_var + EPS)[:, None]
        p_y = tl.make_block_ptr(output, (T * H, D), (D, 1), (block * BT, 0),
                               (BT, BD), (1, 0))
        tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
    else:
        block = pid - q_blocks * 2
        row = block * BT + tl.arange(0, BT)
        col = tl.arange(0, BD)
        address = (row // HV)[:, None] * ST + (2 * H * D + (row % HV)[:, None] * D + col[None, :]) * SD
        value = tl.load(MIXED + address,
                        mask=(row[:, None] < T * HV) & (col[None, :] < D), other=0)
        out = V + row[:, None] * D + col[None, :]
        tl.store(out, value, mask=(row[:, None] < T * HV) & (col[None, :] < D))


def prepare(mixed, heads, value_heads, width, eps=1e-6):
    if (mixed.ndim != 2 or mixed.shape[0] <= 0 or heads <= 0 or value_heads <= 0
            or not 0 < width <= 512 or mixed.shape[1] != (2 * heads + value_heads) * width
            or mixed.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or any(s < 0 for s in mixed.stride())):
        raise ValueError('QKV preparation requires T>0, matching Q/K heads, and width<=512')
    tokens = mixed.shape[0]
    q, k = (torch.empty((1, tokens, heads, width), dtype=mixed.dtype, device=mixed.device)
            for _ in range(2))
    v = torch.empty((1, tokens, value_heads, width), dtype=mixed.dtype, device=mixed.device)
    blocks = 2 * triton.cdiv(tokens * heads, 16) + triton.cdiv(tokens * value_heads, 16)
    _prepare[(blocks,)](mixed, q, k, v, tokens, *mixed.stride(), heads, value_heads,
                       width, 16, triton.next_power_of_2(width), eps,
                       num_warps=8, num_stages=3)
    return q, k, v
