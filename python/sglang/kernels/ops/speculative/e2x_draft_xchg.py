from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.speculative.draft_vocab_argmax import PACK_BLOCKS

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Arrival marker the slots hold until the peer's store lands (see e2x_draft_xchg.cuh).
FP32_MARKER = 0xFFFFFFFF


@cache_once
def _module(hidden: int, blocks: int) -> Module:
    args = make_cpp_args(hidden, blocks, is_arch_support_pdl())
    return load_jit(
        "e2x_draft_xchg",
        *args,
        cuda_files=["speculative/e2x_draft_xchg.cuh"],
        cuda_wrappers=[("run", f"E2xDraftXchg<{args}>::run")],
    )


def e2x_slots_shape(max_rows: int, hidden: int) -> tuple:
    """[2 halves, rows, 2 ranks, memo H + hidden H + meta 4] fp32."""
    return (2, max_rows, 2, 2 * hidden + 4)


def e2x_draft_xchg(
    logits: torch.Tensor,
    memo: torch.Tensor,
    hidden_partial: torch.Tensor,
    part_key: torch.Tensor,
    part_idx: torch.Tensor,
    counter: torch.Tensor,
    slots: torch.Tensor,
    peer_slots: int,
    epoch: torch.Tensor,
    topk_index: torch.Tensor,
    topk_p: torch.Tensor,
    draft_tokens: torch.Tensor,
    positions: torch.Tensor,
    mrope_positions: torch.Tensor,
    fc_out: torch.Tensor,
    vocab_start: int,
    nan_key: int,
    rank: int,
    hidden_rank: int,
    draft_column: int,
) -> None:
    """Two-rank draft-step boundary: shard argmax of ``logits`` [M, V_local], exchange of each
    rank's candidate (token, ``memo`` row [V_local, H] fp32) and the hidden rank's
    ``hidden_partial`` [M, H] fp32, then ``fc_out`` [M, H] = bf16(memo[winner] + hidden partial).
    An empty ``fc_out`` exchanges the token only. ``slots`` (filled with FP32_MARKER before first
    use) is this rank's symmetric buffer, ``peer_slots`` the peer's data pointer. Empty tensors
    skip ``draft_tokens`` / ``positions`` / ``mrope_positions``."""
    _module(memo.shape[-1], PACK_BLOCKS).run(
        logits,
        memo,
        hidden_partial,
        part_key,
        part_idx,
        counter,
        slots,
        peer_slots,
        epoch,
        topk_index,
        topk_p,
        draft_tokens,
        positions,
        mrope_positions,
        fc_out,
        vocab_start,
        nan_key,
        rank,
        hidden_rank,
        draft_column,
    )
