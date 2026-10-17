# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DSpark Markov draft chain in one cooperative kernel per rank, vocab-sharded.

Developed at NVIDIA. Rank k keeps rows [k V / world, (k + 1) V / world) of the Markov
head (markov_w2) in registers, adds the bias of the previous token's markov_w1 row to
its shard of the base logits, picks its best (score, index) key and exchanges one 64-bit
key per rank, request row and draft step through a mailbox in symmetric memory; the kept
step logits reach every rank through peer staging. All draft steps run in one launch.
The CUDA source is jit/csrc/speculative/dspark_sharded_markov.cuh; the peer buffers and
the process group belong to srt/speculative/dspark_components/dspark_sharded_markov.py.

The Gumbel noise is SGLang's Philox draft noise (markov_sample.py, same key, counters
and offset), drawn for this rank's shard before the chain.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Iterable

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.jit.utils import (
    cache_once,
    load_jit,
    make_cpp_args,
    parallel_compile,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Must match the constants of jit/csrc/speculative/dspark_sharded_markov.cuh.
RANK_DIM = 256
THREADS = 512
WARPS = THREADS // 32
MAX_STEPS = 8
MAX_RANKS = 8
# MMA columns: request rows of the narrow kernel.
MAX_ROWS = 8
# Request rows of the wide kernel.
WIDE_ROWS = 64
# The candidate key's 17-bit index field.
MAX_VOCAB = (1 << 17) - 1
# Element types of the base logits and of the step-logits cache.
LOGIT_DTYPES = (torch.bfloat16, torch.float32)
# Peer buffer bytes before the staging logits: the fold's mailbox records.
STAGING_OFFSET = 16384
# Per-CTA launch counters at words [256, 512) (fold) and [512, 768) (direct).
CONTROL_WORDS = 768
MAX_CTAS = 256
# The wide kernel's control words: combine, arrivals, per-CTA counters at 1280.
WIDE_CONTROL_WORDS = 1280 + MAX_CTAS
WIDE_RECORD_BYTES = 2 * MAX_STEPS * WIDE_ROWS * MAX_RANKS * 8
NOISE_BLOCK = 1024
# markov_sample.py's Philox draft noise.
_UNIFORM_SCALE = tl.constexpr(2.3283064365386963e-10)
_PHILOX_ROUNDS = tl.constexpr(10)


@triton.jit
def _shard_noise_kernel(
    NOISE,
    OFFSET,
    KEY,
    VOCAB_START,
    VOCAB_LOCAL,
    STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gumbel noise of (row, step) for this rank's vocab shard, as _select_block draws it."""
    sample = tl.program_id(0)
    row = sample // STEPS
    step = sample % STEPS
    part = tl.program_id(1)
    quad = (part * BLOCK + VOCAB_START) // 4 + tl.arange(0, BLOCK // 4)
    lane = tl.zeros([BLOCK // 4], tl.int64)
    w0, w1, w2, w3 = tl.philox(
        KEY,
        quad.to(tl.uint32),
        (lane + row).to(tl.uint32),
        (lane + step).to(tl.uint32),
        ((lane + tl.load(OFFSET)) & 0xFFFFFFFF).to(tl.uint32),
        _PHILOX_ROUNDS,
    )
    bits = tl.interleave(tl.interleave(w0, w2), tl.interleave(w1, w3))
    uniform = (bits.to(tl.float32) + 0.5) * _UNIFORM_SCALE
    exp_noise = -libdevice.log1p(-uniform)
    gumbel = -tl.log(exp_noise)
    cols = part * BLOCK + tl.arange(0, BLOCK)
    tl.store(NOISE + sample * VOCAB_LOCAL + cols, gumbel, mask=cols < VOCAB_LOCAL)


def launch_shard_noise(
    *,
    noise: torch.Tensor,
    offset: torch.Tensor,
    key: int,
    vocab_start: int,
    vocab_local: int,
    rows: int,
    steps: int,
) -> None:
    """Gumbel noise of ``rows`` request rows into ``noise`` [>= rows * steps, vocab_local]."""
    grid = (rows * steps, triton.cdiv(vocab_local, NOISE_BLOCK))
    _shard_noise_kernel[grid](
        noise, offset, key, vocab_start, vocab_local, STEPS=steps, BLOCK=NOISE_BLOCK
    )


def direct_mailbox_bytes(*, steps: int, world: int, ctas: int, max_rows: int) -> int:
    """Direct-exchange records u64 [parity 2][step][source rank][cta][row]."""
    return 2 * steps * world * ctas * max_rows * 8


@cache_once
def _jit_module(
    steps: int,
    base_dtype: torch.dtype,
    cache_dtype: torch.dtype,
    direct: bool,
    wide: bool,
) -> Module:
    args = make_cpp_args(steps, base_dtype, cache_dtype, direct, wide)
    return load_jit(
        "dspark_sharded_markov",
        *args,
        cuda_files=["speculative/dspark_sharded_markov.cuh"],
        cuda_wrappers=[
            ("sample", f"DSparkShardedMarkov<{args}>::run"),
            ("max_ctas", f"DSparkShardedMarkov<{args}>::max_ctas"),
        ],
    )


def precompile(
    *,
    steps: int,
    cache_dtype: torch.dtype,
    builds: Iterable[tuple[torch.dtype, bool, bool]],
) -> None:
    """Build every (base dtype, direct, wide) kernel a dispatch may reach, concurrently."""
    parallel_compile(
        functools.partial(_jit_module, steps, base_dtype, cache_dtype, direct, wide)
        for base_dtype, direct, wide in builds
    )


def max_coresident_ctas(
    *,
    device: torch.device,
    steps: int,
    base_dtype: torch.dtype,
    cache_dtype: torch.dtype,
    direct: bool,
    wide: bool,
) -> int:
    """CTAs of one kernel the device holds at once (a cooperative grid's bound);
    compiles the kernel on first use."""
    module = _jit_module(steps, base_dtype, cache_dtype, direct, wide)
    with torch.cuda.device(device):
        return int(module.max_ctas())


def dspark_sharded_markov_sample(
    *,
    base_logits: torch.Tensor,
    embed_table: torch.Tensor,
    head_shard: torch.Tensor,
    anchor: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperatures: torch.Tensor,
    noise: torch.Tensor,
    logits_cache: torch.Tensor,
    cache_stride0: int,
    cache_stride1: int,
    draft_tokens: torch.Tensor,
    peers: torch.Tensor,
    control: torch.Tensor,
    base_column: int,
    rank: int,
    world: int,
    vocab_local: int,
    rows: int,
    max_rows: int,
    timeout_ns: int,
    mailbox_offset: int,
    ctas: int,
    cache_dtype: torch.dtype,
    direct: bool,
    wide: bool,
) -> None:
    """One cooperative launch of the chain on the current stream.

    ``base_logits`` [rows, steps, vocab or vocab_local] (BF16/FP32, unit last stride);
    ``embed_table`` [vocab, 256] and ``head_shard`` [vocab_local, 256] BF16; ``anchor``
    int32 [rows]; ``idx_mapping`` int32 [rows * steps], a negative entry marks an inert
    row (argmax, no cache write); ``noise`` FP32 [rows * steps, vocab_local];
    ``draft_tokens`` int64 [rows, steps] (unit last stride). ``peers`` int64 [world]
    holds the peer buffer base of every rank; ``control`` is this kernel's int32 control
    words. Each cross-rank wait traps after ``timeout_ns``.
    """
    module = _jit_module(
        base_logits.shape[1], base_logits.dtype, cache_dtype, direct, wide
    )
    module.sample(
        base_logits,
        embed_table,
        head_shard,
        anchor,
        idx_mapping,
        temperatures,
        noise,
        logits_cache,
        cache_stride0,
        cache_stride1,
        draft_tokens,
        peers,
        control,
        base_column,
        rank,
        world,
        vocab_local,
        rows,
        max_rows,
        timeout_ns,
        mailbox_offset,
        ctas,
    )
