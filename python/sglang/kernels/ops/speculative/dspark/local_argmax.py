# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Greedy verify argmax of a vocab-sharded LM head without gathering the logits.

Developed at NVIDIA. Each rank reduces chunks of its [rows, vocab_local] logits shard
to 64-bit (value, index) keys, posts them into every rank's symmetric-memory mailbox
and reduces the keys of its rows itself, all in one cooperative launch per rank. The
CUDA source is jit/csrc/speculative/dspark_local_argmax.cuh; the mailbox and the
process group belong to srt/speculative/dspark_components/dspark_local_argmax.py.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import (
    cache_once,
    load_jit,
    make_cpp_args,
    parallel_compile,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Must match the constants of jit/csrc/speculative/dspark_local_argmax.cuh.
THREADS = 256
MAX_RANKS = 32
# The key's 17-bit index field.
MAX_VOCAB = 1 << 17
LOGIT_DTYPES = (torch.bfloat16, torch.float32)
# Chunks per row; arbitrary, enough to spread 6 rows over about 100 CTAs.
MAX_SPLITS = 16
# Columns per chunk are a multiple of this, so a 16-byte vector never spans two chunks.
CHUNK_ALIGN = 8


@cache_once
def _jit_module(logit_dtype: torch.dtype) -> Module:
    args = make_cpp_args(logit_dtype)
    return load_jit(
        "dspark_local_argmax",
        *args,
        cuda_files=["speculative/dspark_local_argmax.cuh"],
        cuda_wrappers=[
            ("run", f"DSparkLocalArgmax<{args}>::run"),
            ("max_ctas_per_sm", f"DSparkLocalArgmax<{args}>::max_ctas_per_sm"),
        ],
    )


def precompile() -> None:
    """Build the kernel for every logits dtype, concurrently."""
    parallel_compile(functools.partial(_jit_module, dtype) for dtype in LOGIT_DTYPES)


def max_ctas_per_sm(*, device: torch.device, logit_dtype: torch.dtype) -> int:
    """Co-resident CTAs per SM of one kernel; compiles it on first use."""
    module = _jit_module(logit_dtype)
    with torch.cuda.device(device):
        return int(module.max_ctas_per_sm())


def exchange_splits(rows: int, ctas: int) -> int:
    """Chunks per row: about one work item per CTA, one chunk per row at >= ctas rows."""
    return max(1, min(MAX_SPLITS, ctas // max(rows, 1)))


def exchange_chunk(vocab_local: int, splits: int) -> int:
    return triton.cdiv(triton.cdiv(vocab_local, splits), CHUNK_ALIGN) * CHUNK_ALIGN


def mailbox_bytes(*, max_items: int, world: int) -> int:
    """Records u64 [parity 2][work item][source rank]."""
    return 2 * max_items * world * 8


def local_argmax_exchange(
    *,
    logits: torch.Tensor,
    valid_local: int,
    peers: torch.Tensor,
    ids: torch.Tensor,
    counters: torch.Tensor,
    rank: int,
    world: int,
    max_items: int,
    timeout_ns: int,
    ctas: int,
) -> None:
    """Global argmax ids of every row of ``logits`` into ``ids`` on the current stream.

    ``logits`` [rows, vocab_local] (BF16/FP32, unit column stride, 16-byte aligned rows)
    is this rank's shard, columns [rank * vocab_local, (rank + 1) * vocab_local); only
    its first ``valid_local`` columns take part. ``peers`` int64 [world] holds every
    rank's mailbox base; ``counters`` int32 [ctas] the per-CTA launch counts. Every
    cross-rank wait traps after ``timeout_ns``.
    """
    rows, vocab_local = logits.shape
    splits = exchange_splits(rows, ctas)
    _jit_module(logits.dtype).run(
        logits,
        peers,
        ids,
        counters,
        valid_local,
        splits,
        exchange_chunk(vocab_local, splits),
        rank,
        world,
        max_items,
        timeout_ns,
        ctas,
    )


@triton.jit
def _predicated_copy_kernel(
    dst_ptr, src_ptr, enable_ptr, dst_stride, src_stride, cols, BLOCK: tl.constexpr
):
    if tl.load(enable_ptr) != 0:
        row = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < cols
        x = tl.load(src_ptr + row * src_stride + offs, mask=mask)
        tl.store(
            dst_ptr + row * dst_stride + offs,
            x.to(dst_ptr.dtype.element_ty),
            mask=mask,
        )


def predicated_copy_(
    dst: torch.Tensor, src: torch.Tensor, enable: torch.Tensor
) -> torch.Tensor:
    """``dst.copy_(src)`` for [rows, cols] tensors with unit column stride, done only
    while the int32 at ``enable`` is nonzero (read on the device, graph-safe)."""
    assert dst.shape == src.shape and dst.dim() == 2
    assert dst.stride(1) == 1 and src.stride(1) == 1
    rows, cols = src.shape
    block = 4096
    if rows:
        _predicated_copy_kernel[(rows, triton.cdiv(cols, block))](
            dst,
            src,
            enable,
            dst.stride(0),
            src.stride(0),
            cols,
            BLOCK=block,
            num_warps=8,
        )
    return dst
