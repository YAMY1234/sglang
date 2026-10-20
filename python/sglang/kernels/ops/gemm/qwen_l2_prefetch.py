"""Bulk L2 prefetch of a tensor's bytes on the current stream (csrc/gemm/qwen_l2_prefetch.cuh)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_side_stream: Optional[torch.cuda.Stream] = None


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "qwen_l2_prefetch",
        cuda_files=["gemm/qwen_l2_prefetch.cuh"],
        cuda_wrappers=[("run", "sglang::L2Prefetch::run")],
        extra_cuda_cflags=["-O3"],
    )


def l2_prefetch(tensor: torch.Tensor, ctas: int = 148) -> None:
    """Prefetch the bytes a gap-free tensor (a weight or its transposed view) spans into L2."""
    span = torch.as_strided(tensor, (tensor.numel(),), (1,)).view(torch.uint8)
    _jit_module().run(span, ctas)


def l2_prefetch_stream() -> torch.cuda.Stream:
    """One side stream per process for the prefetch launches."""
    global _side_stream
    if _side_stream is None:
        _side_stream = torch.cuda.Stream()
    return _side_stream
