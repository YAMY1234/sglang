"""Exact row-wise Q/K L2 preparation without intermediate contiguous copies.

The backend enables this only through PREFILL_QK_PREPARE (default off). It uses the reference
l2norm_fwd_kernel's BT/BD, warps, stages and floating-point expression.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def _prepare(Q, K, Q_OUT, K_OUT, QT, QH, QD, KT, KH, KD, T,
             H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
             BD: tl.constexpr, EPS: tl.constexpr):
    block, side = tl.program_id(0), tl.program_id(1)
    if side == 0:
        x, y, st, sh, sd = Q, Q_OUT, QT, QH, QD
    else:
        x, y, st, sh, sd = K, K_OUT, KT, KH, KD
    row = block * BT + tl.arange(0, BT)
    col = tl.arange(0, BD)
    address = (row // H)[:, None] * st + (row % H)[:, None] * sh + col[None, :] * sd
    b_x = tl.load(x + address, mask=(row[:, None] < T * H) & (col[None, :] < D),
                  other=0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + EPS)[:, None]
    p_y = tl.make_block_ptr(y, (T * H, D), (D, 1), (block * BT, 0),
                           (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


def prepare(q, k, eps=1e-6):
    """Return private contiguous normalized tensors for one flattened sequence."""
    if (q.ndim != 4 or q.shape[0] != 1 or k.shape != q.shape
            or q.dtype != k.dtype or q.device != k.device
            or q.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or not 0 < q.shape[-1] <= 512 or min(q.shape[1:]) <= 0
            or any(s < 0 for x in (q, k) for s in x.stride())):
        raise ValueError('Q/K preparation requires matching B=1 floating tensors, 0<D<=512')
    _, tokens, heads, width = q.shape
    out_q = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    out_k = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    _prepare[(triton.cdiv(tokens * heads, 16), 2)](
        q, k, out_q, out_k, *q.stride()[1:], *k.stride()[1:], tokens,
        heads, width, 16, triton.next_power_of_2(width), eps,
        num_warps=8, num_stages=3)
    return out_q, out_k
