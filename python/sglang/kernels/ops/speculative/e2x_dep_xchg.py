from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.speculative.draft_vocab_argmax import (
    NAN_KEY_SPLIT_ARGMAX,
    NAN_KEY_TORCH_ARGMAX,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Must match e2x_dep::kMaxBlocks / kMarker in e2x_dep_xchg.cuh.
MAX_BLOCKS = 32
SLOT_MARKER = -1


@cache_once
def _module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "e2x_dep_xchg",
        *args,
        cuda_files=["speculative/e2x_dep_xchg.cuh"],
        cuda_wrappers=[("run", f"E2xDepXchg<{args}>::run")],
    )


def dep_slots_shape(max_gathered_rows: int, world: int) -> tuple:
    """[2 halves, gathered rows, sender rank, 4] int32: one 16B candidate per sender."""
    return (2, max_gathered_rows, world, 4)


def dep_data_shape(max_gathered_rows: int, world: int, hidden: int) -> tuple:
    """[2 halves, gathered rows, sender rank, hidden] bf16: the memo row behind each candidate."""
    return (2, max_gathered_rows, world, hidden)


def dep_xchg_blocks(gathered_rows: int) -> int:
    # Arbitrary: about 256 CTAs in flight, at least 4 per row.
    return max(4, min(MAX_BLOCKS, 256 // gathered_rows))


def e2x_dep_xchg(
    logits: torch.Tensor,
    part_key: torch.Tensor,
    part_idx: torch.Tensor,
    counter: torch.Tensor,
    slots: torch.Tensor,
    peer_slots: torch.Tensor,
    epoch: torch.Tensor,
    tokens: torch.Tensor,
    topk_p: torch.Tensor,
    draft_tokens: torch.Tensor,
    positions: torch.Tensor,
    mrope_positions: torch.Tensor,
    memo: torch.Tensor,
    data: torch.Tensor,
    peer_data: torch.Tensor,
    fc_out: torch.Tensor,
    vocab_start: int,
    torch_nan: bool,
    rank: int,
    rows: int,
    draft_column: int,
    blocks: Optional[int] = None,
) -> None:
    """Greedy draft tokens [rows, 1] of this rank's rows from the shard ``logits``
    [world * rows, V_local] of every gathered row (rank-major): the argmax over the full vocab,
    ties to the lowest id, NaN as torch.argmax (``torch_nan``) or as -1e30. Non-empty ``topk_p``,
    ``draft_tokens`` [rows, steps], ``positions`` [rows] and ``mrope_positions`` [3, rows] get
    k75p_topk1_postprocess's writes (topk_p = 1, column ``draft_column``, + 1) and mrope + 1. A
    non-empty ``memo`` [V_local, H] (normalized embedding rows of this shard) sends each candidate's row
    through ``data``; the owner writes the winner's row into ``fc_out`` [rows, H] (row stride free)."""
    if blocks is None:
        blocks = dep_xchg_blocks(logits.shape[0])
    _module().run(
        logits,
        part_key,
        part_idx,
        counter,
        slots,
        peer_slots,
        epoch,
        tokens,
        topk_p,
        draft_tokens,
        positions,
        mrope_positions,
        memo,
        data,
        peer_data,
        fc_out,
        vocab_start,
        NAN_KEY_TORCH_ARGMAX if torch_nan else NAN_KEY_SPLIT_ARGMAX,
        rank,
        rows,
        blocks,
        draft_column,
    )
