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


# Side streams by tag (SGLANG_Q35MF_KRNC_L2PF_LL / _OPROJ_PF) and the tags forked since the last join.
_deferred: dict = {}
_pending: set = set()


def l2_prefetch_deferred(tensors, tag: str, ctas: int = 148) -> None:
    """Prefetch tensors into L2 on the tag's side stream after the work already queued on the current stream, and
    leave it unjoined: l2_prefetch_join_deferred() at the end of the forward joins it, so no kernel waits on it."""
    side = _deferred.get(tag)
    if side is None:
        side = _deferred[tag] = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for tensor in tensors:
            l2_prefetch(tensor, ctas)
    _pending.add(tag)


def l2_prefetch_join_deferred() -> None:
    """Join every side stream forked by l2_prefetch_deferred since the last join (a CUDA graph capture must end with
    all forked streams joined)."""
    current = torch.cuda.current_stream()
    for tag in sorted(_pending):
        current.wait_stream(_deferred[tag])
    _pending.clear()
