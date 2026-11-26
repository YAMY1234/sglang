"""MLA's q_a and kv_a RMSNorms in one launch.

The fused q/kv-a projection gives each token one row ``[q | k_nope | k_pe]``.
``mla_qkv_a_rmsnorm`` normalizes ``q`` and ``k_nope``, each with its own weight
and epsilon, in one launch: one program per row and part.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl


@triton.jit
def _rmsnorm_store(x, w_ptr, out_ptr, cols, mask, width, eps):
    xf = x.to(tl.float32)
    rstd = tl.math.rsqrt(tl.sum(xf * xf, axis=0) / width + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # x * w is exact in fp32; scaling it rounds once, as RMSNorm's JIT kernel does.
    tl.store(out_ptr + cols, (xf * w * rstd).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mla_qkv_a_rmsnorm_kernel(
    src_ptr,
    q_out_ptr,
    kv_out_ptr,
    q_weight_ptr,
    kv_weight_ptr,
    src_row_stride,
    q_eps,
    kv_eps,
    Q: tl.constexpr,
    KV: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    # Program (row, 0) owns columns [0, Q) of the row, (row, 1) owns [Q, Q + KV).
    row = tl.program_id(0)
    is_kv = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < Q + is_kv * (KV - Q)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(src_ptr + row * src_row_stride + is_kv * Q + cols, mask=mask, other=0.0)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    if is_kv == 0:
        _rmsnorm_store(x, q_weight_ptr, q_out_ptr + row * Q, cols, mask, Q, q_eps)
    else:
        _rmsnorm_store(x, kv_weight_ptr, kv_out_ptr + row * KV, cols, mask, KV, kv_eps)


def mla_qkv_a_rmsnorm(
    latent: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    kv_weight: torch.Tensor,
    kv_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``rmsnorm(latent[:, :Q])`` and ``rmsnorm(latent[:, Q:Q + KV])``,
    both contiguous, for the projection output ``latent`` ``[M, >= Q + KV]``
    with unit column stride. Q and KV are the weights' sizes."""
    q_size, kv_size = q_weight.shape[0], kv_weight.shape[0]
    num_rows = latent.shape[0]
    assert latent.stride(1) == 1 and q_size + kv_size <= latent.shape[1]
    q = latent.new_empty((num_rows, q_size))
    kv = latent.new_empty((num_rows, kv_size))
    if num_rows == 0:
        return q, kv
    block = triton.next_power_of_2(max(q_size, kv_size))
    use_pdl = is_arch_support_pdl()
    _mla_qkv_a_rmsnorm_kernel[(num_rows, 2)](
        latent,
        q,
        kv,
        q_weight,
        kv_weight,
        latent.stride(0),
        q_eps,
        kv_eps,
        Q=q_size,
        KV=kv_size,
        BLOCK=block,
        USE_PDL=use_pdl,
        num_warps=max(1, min(16, block // 256)),
        **({"launch_pdl": True} if use_pdl else {}),
    )
    return q, kv
