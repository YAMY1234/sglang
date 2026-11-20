"""Exact global top-k merge for DSA decode context parallelism; the algorithm
is described in ``jit/csrc/dsa/dcp_topk_merge.cuh``."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _staged_launch() -> tuple[int, int]:
    """(kStaged, kMinBlocks) of merge_kernel: SGLANG_K31_DCP_MERGE_STAGED (default
    off) and, when staged, SGLANG_K31_DCP_MERGE_MIN_BLOCKS (default 1)."""
    if not envs.SGLANG_K31_DCP_MERGE_STAGED.get():
        return 0, 1
    min_blocks = envs.SGLANG_K31_DCP_MERGE_MIN_BLOCKS.get()
    if min_blocks < 1:
        raise ValueError(f"SGLANG_K31_DCP_MERGE_MIN_BLOCKS must be >= 1, got {min_blocks}")
    logger.info(
        "DSA DCP top-k merge: staged push variant on (SGLANG_K31_DCP_MERGE_STAGED=1), "
        "launch bounds min blocks %d (SGLANG_K31_DCP_MERGE_MIN_BLOCKS)",
        min_blocks,
    )
    return 1, min_blocks


@cache_once
def _jit_dcp_topk_merge_module(
    num_ranks: int, topk: int, cand_k: int, split: bool = False
) -> Module:
    staged, min_blocks = _staged_launch()
    if staged:
        args = make_cpp_args(num_ranks, topk, cand_k, int(split), staged, min_blocks)
    else:
        args = make_cpp_args(num_ranks, topk, cand_k, int(split))
    return load_jit(
        "dcp_topk_merge",
        *args,
        cuda_files=["dsa/dcp_topk_merge.cuh"],
        cuda_wrappers=[("dcp_topk_merge", f"DcpTopKMergeKernel<{args}>::run")],
    )


# Running totals a merge adds to (int64 ``stats``): rows whose certificate
# failed, of them the rows this rank published its keys for, of those the rows
# where it had to select among more than top-k keys, and all merged rows.
DCP_TOPK_MERGE_NUM_STATS = 4
_SCRATCH_STATS: dict[torch.device, torch.Tensor] = {}


def _stats_or_scratch(stats: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
    if stats is not None:
        return stats
    if device not in _SCRATCH_STATS:
        _SCRATCH_STATS[device] = torch.zeros(
            DCP_TOPK_MERGE_NUM_STATS, dtype=torch.int64, device=device
        )
    return _SCRATCH_STATS[device]


# Entries after the full slot of a pushed-merge row slot: its stamp (kPushExtra in dcp_topk_merge.cuh).
DCP_TOPK_MERGE_PUSH_EXTRA = 1


def dcp_topk_merge_buffer_numel(
    num_ranks: int, max_rows: int, topk: int, cand_k: int, push: bool = False
) -> int:
    """int64 entries of the symmetric buffer: two parities of max_rows row slots,
    each holding every rank's candidates and this rank's full local top-k (and,
    for the pushed merge, the full slot's stamp)."""
    return 2 * max_rows * dcp_topk_merge_row_slot(num_ranks, topk, cand_k, push)


def dcp_topk_merge_row_slot(num_ranks: int, topk: int, cand_k: int, push: bool) -> int:
    return num_ranks * cand_k + topk + (DCP_TOPK_MERGE_PUSH_EXTRA if push else 0)


def dcp_topk_merge(
    candidates: torch.Tensor,
    global_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    buffer: torch.Tensor,
    multicast_ptr: int,
    peer_ptrs: torch.Tensor,
    counters: torch.Tensor,
    union_scratch: torch.Tensor,
    out_slots: torch.Tensor,
    out_fallback: torch.Tensor,
    rank: int,
    span: int,
    cand_k: int,
    split: bool = False,
    stats: Optional[torch.Tensor] = None,
) -> None:
    """Merge one batch of rows; every rank of the group must call it with the
    same rows, in the same order.

    ``candidates``: [rows, topk, 2] int32 (fp32 score bits, global position or
    -1), this rank's local top-k. ``global_lens``: [rows] int32 visible keys of
    each row. ``page_table``: [>= rows, pages] int32 widened page ids, row i
    for row i. ``buffer``: the int64 symmetric buffer
    (``dcp_topk_merge_buffer_numel``), ``multicast_ptr`` its multicast address,
    ``peer_ptrs``: [num_ranks] int64 its address on every rank. ``counters``:
    [max_rows] int64 calls per row slot, zero-initialized together with the
    buffer on every rank. ``union_scratch``: [max_rows, num_ranks * topk] int64
    working space of the fallback. ``out_slots``: [rows, topk] int32, this
    rank's selected keys as widened slots ``page_table[p // span] * span + p %
    span``, -1 after them. ``out_fallback``: [rows] int32, 1 where a
    certificate failed.

    ``split`` (KV replicated): rank q scored the 64-token pages p with
    p % num_ranks == q, and every rank writes all selected keys, not only its own.
    ``stats``: [DCP_TOPK_MERGE_NUM_STATS] int64 running totals the merge adds to.
    """
    num_ranks = peer_ptrs.shape[0]
    module = _jit_dcp_topk_merge_module(num_ranks, candidates.shape[1], cand_k, split)
    module.dcp_topk_merge(
        candidates,
        buffer,
        multicast_ptr,
        peer_ptrs,
        counters,
        global_lens,
        page_table,
        out_slots,
        out_fallback,
        union_scratch,
        _stats_or_scratch(stats, out_slots.device),
        rank,
        span,
    )


@cache_once
def _jit_dcp_topk_merge_pushed_module(num_ranks: int, topk: int, cand_k: int) -> Module:
    args = make_cpp_args(num_ranks, topk, cand_k, is_arch_support_pdl())
    return load_jit(
        "dcp_topk_merge_pushed",
        *args,
        cuda_files=["dsa/dcp_topk_merge.cuh"],
        cuda_wrappers=[
            ("dcp_topk_merge_pushed", f"DcpTopKMergePushedKernel<{args}>::run")
        ],
    )


def dcp_topk_merge_pushed(
    logits: torch.Tensor,
    local_lens: torch.Tensor,
    global_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    buffer: torch.Tensor,
    peer_ptrs: torch.Tensor,
    counters: torch.Tensor,
    out_slots: torch.Tensor,
    out_local: torch.Tensor,
    out_lengths: torch.Tensor,
    out_fallback: torch.Tensor,
    rank: int,
    span: int,
    cand_k: int,
    stats: Optional[torch.Tensor] = None,
) -> None:
    """``dcp_topk_merge`` for candidates the local top-k already pushed
    (``topk_transform_candidates_dcp_push`` with the same ``buffer``,
    ``counters`` and ``cand_k``); token interleave only. Every rank must call it
    with the same rows, in the same order, right after its push.

    ``logits``: [rows, >= local_lens.max()] fp32, this rank's scores, read only
    by a row whose certificate failed on this rank. ``out_slots`` as
    ``dcp_topk_merge``; ``out_local``: [rows, topk] the same keys as this rank's
    local KV rows (slot // num_ranks), -1 after them, and ``out_lengths``: [rows]
    their count (what ``dcp_localize_compact`` returns for ``out_slots``).
    ``stats`` as ``dcp_topk_merge``.
    """
    num_ranks = peer_ptrs.shape[0]
    module = _jit_dcp_topk_merge_pushed_module(num_ranks, out_slots.shape[1], cand_k)
    module.dcp_topk_merge_pushed(
        logits,
        local_lens,
        buffer,
        peer_ptrs,
        counters,
        global_lens,
        page_table,
        out_slots,
        out_local,
        out_lengths,
        out_fallback,
        _stats_or_scratch(stats, out_slots.device),
        rank,
        span,
    )
