"""Triton scatter for SGLANG_LINKER_EARLY_READS: copy ranges given as absolute
addresses (device destinations, pinned-host sources reached through UVA)."""

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_kernel(dst_ptr, src_ptr, size_ptr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    dst = tl.load(dst_ptr + i).to(tl.pointer_type(tl.int8))
    src = tl.load(src_ptr + i).to(tl.pointer_type(tl.int8))
    n = tl.load(size_ptr + i)
    for off in range(0, n, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < n
        tl.store(dst + idx, tl.load(src + idx, mask=mask), mask=mask)


def scatter_copy(dst: np.ndarray, src: np.ndarray, sizes: np.ndarray, block: int = 4096) -> None:
    """One program per range, on the current stream."""
    args = [torch.from_numpy(np.ascontiguousarray(a, dtype=np.int64)).pin_memory().to("cuda", non_blocking=True)
            for a in (dst, src, sizes)]
    _scatter_kernel[(len(dst),)](*args, BLOCK=block)
