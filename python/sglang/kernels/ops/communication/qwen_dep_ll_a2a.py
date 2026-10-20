"""Fence-free MoE all-to-all for attention-DP + EP decode (csrc/distributed/qwen_dep_ll_a2a.cuh)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# csrc kMaxTokens: tokens per source rank one exchange slot table holds.
LL_A2A_MAX_TOKENS = 128
# csrc kCountLines: per-expert count lines per (stage, source) behind the dispatch slots.
LL_A2A_COUNT_LINES = 64


def ll_a2a_region_bytes(
    ranks: int, top_k: int, hidden: int, slots: int, x_bytes: int, sf_bytes: int
) -> tuple[int, int]:
    """Bytes of the dispatch and combine regions on every rank (csrc Geometry::dispatch_bytes / combine_bytes)."""
    lines = top_k + (x_bytes + sf_bytes) // 8
    return (
        2 * ranks * 16 * (1 + slots * lines + LL_A2A_COUNT_LINES),
        3 * ranks * slots * hidden * 2,
    )


@cache_once
def _jit_module(
    ranks: int, top_k: int, hidden: int, x_bytes: int, sf_bytes: int
) -> Module:
    args = make_cpp_args(ranks, top_k, hidden, x_bytes, sf_bytes, is_arch_support_pdl())
    return load_jit(
        "qwen_dep_ll_a2a",
        *args,
        cuda_files=["distributed/qwen_dep_ll_a2a.cuh"],
        cuda_wrappers=[
            ("dispatch", f"sglang::LlA2A<{args}>::dispatch"),
            ("dispatch_tables", f"sglang::LlA2A<{args}>::dispatch_tables"),
            ("combine", f"sglang::LlA2A<{args}>::combine"),
        ],
        extra_cuda_cflags=["-O3"],
    )


def ll_a2a_dispatch(
    peers: torch.Tensor,
    x: torch.Tensor,
    sf: torch.Tensor,
    ids: torch.Tensor,
    wts: torch.Tensor,
    ctr: torch.Tensor,
    out_x: torch.Tensor,
    out_sf: torch.Tensor,
    out_ids: torch.Tensor,
    out_wts: torch.Tensor,
    recv_tokens: torch.Tensor,
    err: torch.Tensor,
    *,
    tmax: int,
    slots: int,
    rank: int,
    experts_per_rank: int,
    invalid_id: int,
    hidden: int,
) -> None:
    """peers: CPU int64 [ranks] dispatch-region bases; x / sf uint8 [tokens, bytes] (sf [tokens, 0] for BF16);
    out_* rows [ranks * tmax]; one CTA per ctr entry."""
    _jit_module(peers.shape[0], ids.shape[1], hidden, x.shape[1], sf.shape[1]).dispatch(
        peers,
        x,
        sf,
        ids,
        wts,
        ctr,
        out_x,
        out_sf,
        out_ids,
        out_wts,
        recv_tokens,
        err,
        tmax,
        slots,
        rank,
        experts_per_rank,
        invalid_id,
    )


def ll_a2a_dispatch_tables(
    peers: torch.Tensor,
    x: torch.Tensor,
    sf: torch.Tensor,
    ids: torch.Tensor,
    wts: torch.Tensor,
    ctr: torch.Tensor,
    out_x: torch.Tensor,
    out_sf: torch.Tensor,
    out_ids: torch.Tensor,
    out_wts: torch.Tensor,
    recv_tokens: torch.Tensor,
    err: torch.Tensor,
    tables,
    *,
    tmax: int,
    slots: int,
    rank: int,
    experts_per_rank: int,
    invalid_id: int,
    hidden: int,
    local_offset: int,
    num_local: int,
) -> None:
    """ll_a2a_dispatch that also fills `tables` (an EpTablesBuffers over out_ids) with this rank's routing tables;
    every rank of the exchange must take this path for the same call."""
    _jit_module(
        peers.shape[0], ids.shape[1], hidden, x.shape[1], sf.shape[1]
    ).dispatch_tables(
        peers,
        x,
        sf,
        ids,
        wts,
        ctr,
        out_x,
        out_sf,
        out_ids,
        out_wts,
        recv_tokens,
        err,
        tables.tile_idx_to_expert_idx,
        tables.tile_idx_to_mn_limit,
        tables.expanded_idx_to_permuted_idx,
        tables.permuted_idx_to_expanded_idx,
        tables.total_num_padded_tokens,
        tables.num_non_exiting_tiles,
        tmax,
        slots,
        rank,
        experts_per_rank,
        invalid_id,
        local_offset,
        num_local,
        tables.tile_size.bit_length() - 1,
    )


def ll_a2a_combine(
    peers: torch.Tensor,
    partial: torch.Tensor,
    rids: torch.Tensor,
    recv_tokens: torch.Tensor,
    ids: torch.Tensor,
    out: torch.Tensor,
    ctr: torch.Tensor,
    err: torch.Tensor,
    *,
    tmax: int,
    slots: int,
    rank: int,
    experts_per_rank: int,
    num_experts: int,
    x_bytes: int,
    sf_bytes: int,
) -> None:
    """peers: CPU int64 [ranks] combine-region bases; partial / rids rows [ranks * tmax] as dispatch wrote them;
    x_bytes / sf_bytes as in the matching dispatch."""
    _jit_module(
        peers.shape[0], ids.shape[1], partial.shape[1], x_bytes, sf_bytes
    ).combine(
        peers,
        partial,
        rids,
        recv_tokens,
        ids,
        out,
        ctr,
        err,
        tmax,
        slots,
        rank,
        experts_per_rank,
        num_experts,
    )
