from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _module() -> Module:
    return load_jit(
        "l2_line_prefetch",
        cuda_files=["elementwise/l2_line_prefetch.cuh"],
        cuda_wrappers=[("run", "L2LinePrefetch::run")],
    )


def l2_line_prefetch(weight: torch.Tensor, ctas: int = 592) -> None:
    """Prefetch a 2-D weight's bytes into L2 on the current stream (no values change)."""
    # Projection weights are often stored as transposed views; the prefetch only needs the bytes.
    if not weight.is_contiguous():
        weight = weight.t()
    if weight.is_contiguous():
        _module().run(weight, weight.numel() * weight.element_size(), ctas)
