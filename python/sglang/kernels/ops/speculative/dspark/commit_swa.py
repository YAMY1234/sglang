"""Translate committed cache locations to the draft SWA pool."""

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# N = batch size x verify width. As a constexpr it compiles a new kernel for every decode batch size, outside CUDA
# graphs, and one compiling DP-attention rank holds every rank at the next sync (sgl-project/sglang#42394).
_RUNTIME_N = os.environ.get("SGLANG_DSPARK_COMMIT_SWA_RUNTIME_N", "0").lower() in ("1", "true")
_logged = False


@triton.jit
def _committed_swa_locations(
    LOC,
    MAP,
    LENS,
    OUT,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    MAP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(LENS + i // WIDTH, i < N, other=0)
    committed = (i < N) & (i % WIDTH < length)
    loc = tl.load(LOC + i, committed, other=0).to(tl.int64)
    # Preserve torch indexing for an unused negative location.
    loc = tl.where(loc < 0, loc + MAP_SIZE, loc)
    swa = tl.load(MAP + loc, committed, other=-1).to(tl.int32)
    tl.store(OUT + i, swa, i < N)


@triton.jit
def _committed_swa_locations_runtime_n(
    LOC,
    MAP,
    LENS,
    OUT,
    N,
    WIDTH: tl.constexpr,
    MAP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(LENS + i // WIDTH, i < N, other=0)
    committed = (i < N) & (i % WIDTH < length)
    loc = tl.load(LOC + i, committed, other=0).to(tl.int64)
    # Preserve torch indexing for an unused negative location.
    loc = tl.where(loc < 0, loc + MAP_SIZE, loc)
    swa = tl.load(MAP + loc, committed, other=-1).to(tl.int32)
    tl.store(OUT + i, swa, i < N)


def committed_swa_locations(cache_loc, full_to_swa_mapping, commit_lens, width):
    global _logged
    assert cache_loc.numel() == commit_lens.numel() * width
    out = torch.empty_like(cache_loc, dtype=torch.int32)
    if cache_loc.numel():
        kernel = _committed_swa_locations_runtime_n if _RUNTIME_N else _committed_swa_locations
        if _RUNTIME_N and not _logged:
            _logged = True
            logger.info("DSpark committed_swa_locations: N is a runtime argument")
        kernel[(triton.cdiv(cache_loc.numel(), 256),)](
            cache_loc,
            full_to_swa_mapping,
            commit_lens,
            out,
            cache_loc.numel(),
            width,
            full_to_swa_mapping.numel(),
            256,
        )
    return out
