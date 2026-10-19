"""Small-M (<= 64 tokens) per-tensor FP8 projection GEMM for decode (see csrc/gemm/q35_smallm_fp8_gemm.cuh)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_ROWS_PER_CTA = 64
_counts: dict[tuple[torch.device, int], torch.Tensor] = {}


@cache_once
def _jit_module(nt: int, split: int, stages: int) -> Module:
    args = make_cpp_args(nt, split, stages)
    return load_jit(
        "q35_smallm_fp8_gemm",
        *args,
        cuda_files=["gemm/q35_smallm_fp8_gemm.cuh"],
        cuda_wrappers=[("run", f"sglang::SmallmFp8Gemm<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def smallm_fp8_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    alpha: torch.Tensor,
    *,
    split: int = 1,
    stages: int = 4,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """out[M, N] bf16 = (x[M, K] @ w[N, K]^T) * alpha; x / w e4m3 (row-major, K contiguous), alpha fp32 [1]."""
    m, k = x.shape
    n = w.shape[0]
    nt = 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    part = (
        torch.empty((split * nt * 8 * n,), dtype=torch.float32, device=x.device)
        if split > 1
        else out
    )
    key = (x.device, n)
    count = _counts.get(key)
    if count is None:
        count = _counts[key] = torch.zeros((n // _ROWS_PER_CTA,), dtype=torch.int32, device=x.device)
    _jit_module(nt, split, stages).run(
        x.view(torch.uint8), w.view(torch.uint8), alpha, out, part, count
    )
    return out
