from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _module(k: int, warps: int, rows_per_warp: int) -> Module:
    args = make_cpp_args(k, warps, rows_per_warp, is_arch_support_pdl())
    return load_jit(
        "mkd_draft_qkv",
        *args,
        cuda_files=["speculative/mkd_draft_qkv.cuh"],
        cuda_wrappers=[("run", f"MkdQkv<{args}>::run")],
    )


def mkd_draft_qkv(
    h: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    weight: torch.Tensor,
    out: torch.Tensor,
    warps: int = 8,
    rows_per_warp: int = 2,
) -> torch.Tensor:
    """``out`` [1, N] bf16 = GemmaRMSNorm(``h`` [1, K]; ``gamma`` [K], ``eps``) @ ``weight`` [N, K]^T: the norm in fp32
    rounded to bf16, then fp32 accumulation; the weight rows are prefetched into L2 before the PDL wait."""
    _module(h.shape[-1], warps, rows_per_warp).run(h, gamma, weight, out, eps)
    return out
