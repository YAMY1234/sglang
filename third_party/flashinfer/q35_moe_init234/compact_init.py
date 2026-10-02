"""Default-off initialization helpers for the pinned SM100 MoE wrapper."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fill_metadata(A, B, NA: tl.constexpr, NB: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(A + i, 0, i < NA)
    tl.store(B + i, 0, i < NB)


def fill_tile_metadata(expert, limit):
    assert expert.dtype == limit.dtype == torch.int32
    assert expert.is_contiguous() and limit.is_contiguous()
    assert expert.device == limit.device
    n = max(expert.numel(), limit.numel())
    if n:
        _fill_metadata[(triton.cdiv(n, 256),)](
            expert, limit, expert.numel(), limit.numel(), 256
        )


@triton.jit
def _zero_routed_rows(
    OUT,
    IDS,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    STRIDE: tl.constexpr,
    FIRST: tl.constexpr,
    LOCAL: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, K)
    ids = tl.load(IDS + row * STRIDE + k, k < TOPK, other=-1)
    routed = tl.sum(((ids >= FIRST) & (ids < FIRST + LOCAL)).to(tl.int32), 0) > 0
    if routed:
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        tl.store(OUT + row * H + col, 0, col < H)


def sparse_output_zero(output, ids, first, local):
    # Caller guarantees that MNNVL combine never consumes unrouted rows.
    assert output.is_contiguous() and output.ndim == ids.ndim == 2
    assert ids.stride(1) == 1 and output.shape[0] == ids.shape[0]
    assert output.device == ids.device and ids.dtype == torch.int32
    n, h = output.shape
    topk = ids.shape[1]
    if n and h:
        _zero_routed_rows[(n, triton.cdiv(h, 1024))](
            output,
            ids,
            h,
            topk,
            ids.stride(0),
            first,
            local,
            triton.next_power_of_2(topk),
            1024,
        )
