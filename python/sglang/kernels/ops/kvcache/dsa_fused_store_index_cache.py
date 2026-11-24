"""
This module provides JIT-compiled CUDA kernels for fusing multiple tensor
copy operations into single kernel launches, reducing kernel launch overhead
and improving CUDA graph replay performance.

The kernels are compiled on-demand using TVM FFI and cached for subsequent use.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)


@cache_once
def _jit_dsa_fused_store_module(
    key_dtype: torch.dtype,
    indices_dtype: torch.dtype,
    page_size: int,
    pow2_scale: bool = False,
) -> Module:
    """
    Build a JIT module that exposes:
      module.fused_store_index_k_cache(
          input_bf16, index_k_with_scale_u8, loc_i64, dcp_size, dcp_rank
      )
    pow2_scale rounds each row's scale up to a power of two (kPow2Scale); off keeps
    the original template arguments, so the default module is unchanged.
    """
    args = make_cpp_args(key_dtype, indices_dtype, page_size, is_arch_support_pdl())
    if pow2_scale:
        args = make_cpp_args(
            key_dtype, indices_dtype, page_size, is_arch_support_pdl(), True
        )
    return load_jit(
        "fused_store_index_k_cache",
        *args,
        cuda_files=["dsa/fused_store_index_cache.cuh"],
        cuda_wrappers=[
            (
                "fused_store_index_k_cache",
                # - Float  = bf16_t (sgl_kernel/type.cuh)
                # - IndicesT = int64_t (out_cache_loc is int64 in SGLang SetKAndS)
                # - kPageSize = 64 (CUDA DSA)
                f"FusedStoreCacheIndexerKernel<{args}>::run",
            )
        ],
    )


@cache_once
def can_use_dsa_fused_store(
    key_dtype: torch.dtype, indices_dtype: torch.dtype, page_size: int
) -> bool:
    logger = logging.getLogger(__name__)
    try:
        _jit_dsa_fused_store_module(key_dtype, indices_dtype, page_size)
        return True
    except Exception as e:
        logger.warning(f"Failed to load dsa fused store JIT kernel: {e}")
        return False


@debug_kernel_api
def fused_store_index_k_cache(
    key: torch.Tensor,
    index_k_with_scale: torch.Tensor,
    out_cache_loc: torch.Tensor,
    page_size: int = 64,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    pow2_scale: bool = False,
) -> None:
    """
    Fused: quantize bf16 key (N,128) -> fp8 + fp32 scale and write into DSATokenToKVPool.index_k_with_scale_buffer.

    key:            (num_tokens, 128) bf16 (or reshapeable to it)
    index_k_with_scale:  (num_pages, 64*(128+4)) uint8
    out_cache_loc:       (num_tokens,) int64 token indices in TokenToKVPool
    dcp_size, dcp_rank:  under DCP, out_cache_loc holds widened slots; only the
                         rank's own slots are stored, at row slot // dcp_size
    pow2_scale:          round each row's scale up to a power of two, so a key on
                         the MXFP4 grid (mxfp4_roundtrip_index_k) stores exactly
    """
    assert key.is_cuda
    assert index_k_with_scale.is_cuda
    assert out_cache_loc.is_cuda

    # 1) normalize shapes
    if key.dim() != 2:
        key = key.view(-1, key.shape[-1])
    assert key.shape[1] == 128, f"expected key last-dim=128, got {key.shape}"

    # 2) dtypes
    assert key.dtype == torch.bfloat16, f"{key.dtype=}"
    assert index_k_with_scale.dtype == torch.uint8, f"{index_k_with_scale.dtype=}"
    assert out_cache_loc.dtype == torch.int64, f"{out_cache_loc.dtype=}"

    # 3) contiguity
    if not key.is_contiguous():
        key = key.contiguous()
    if not out_cache_loc.is_contiguous():
        out_cache_loc = out_cache_loc.contiguous()
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()

    module = _jit_dsa_fused_store_module(
        key.dtype, out_cache_loc.dtype, page_size, pow2_scale
    )
    module.fused_store_index_k_cache(
        key, index_k_with_scale, out_cache_loc, dcp_size, dcp_rank
    )


# E2M1 boundaries between {0, .5, 1, 1.5, 2, 3, 4, 6}; True: a tie rounds up (to the even code)
_E2M1_BOUNDARIES = (
    (0.25, False),
    (0.75, True),
    (1.25, False),
    (1.75, True),
    (2.5, False),
    (3.5, True),
    (5.0, False),
)


def mxfp4_roundtrip_index_k(key: torch.Tensor) -> torch.Tensor:
    """Index key onto the MXFP4 grid and back, bit-equal to DeepGEMM's
    per_token_cast_to_fp4(use_ue8m0=True, gran_k=32) + cast_back_from_fp4
    (deep_gemm/utils/math.py, 0.2.3.post2 and 0.2.8); idempotent, exact in bf16."""
    head_dim = key.shape[-1]
    assert head_dim % 32 == 0, f"{key.shape=}"
    x = key.reshape(-1, head_dim // 32, 32).float()
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    bits = (amax / 6.0).view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).ne(0).to(torch.int32)
    sf = (exp.clamp(1, 254) << 23).view(torch.float32)
    scaled = x * (1.0 / sf)
    mag = scaled.abs()
    code = torch.zeros_like(mag, dtype=torch.int32)
    for boundary, tie_up in _E2M1_BOUNDARIES:
        code += (mag >= boundary if tie_up else mag > boundary).to(torch.int32)
    # code -> value without a host table, so the call can be graph-captured
    value = torch.where(
        code < 4, code * 0.5, torch.exp2((code >> 1) - 1.0) * (1.0 + 0.5 * (code & 1))
    )
    value = torch.where((scaled < 0) & (code != 0), -value, value)
    return (value * sf).to(key.dtype).reshape(key.shape)
