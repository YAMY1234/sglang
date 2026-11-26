"""A draft block's sliding-window MLA attention with each group of rows sharing one
window read (``sglang.kernels.ops.attention.mla_shared_window``)."""

import torch

from sglang.kernels.ops.attention.mla_shared_window import (
    merge_shared_window,
    mla_shared_window_indices,
)
from sglang.srt.layers.attention.dsa.dsa_dcp import dcp_trtllm_sparse_attention

# trtllm-gen MLA serves at most 64 heads below its 128-head variant.
_MAX_UNPADDED_HEADS = 64


def shared_window_group(*, rows_per_req: int, heads: int, group: int) -> int:
    """``group`` when the folded heads need no padding and divide the block, else 0."""
    if group < 2 or rows_per_req % group or group * heads > _MAX_UNPADDED_HEADS:
        return 0
    return group


def shared_window_indices(
    *,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    min_pos: torch.Tensor,
    window_left: int,
    ring_rows_per_req: int,
    group: int,
) -> tuple[torch.Tensor, ...]:
    """The index lists one forward's sliding layers share."""
    return mla_shared_window_indices(
        req_pool_indices=req_pool_indices,
        positions=positions,
        min_pos=min_pos,
        window_left=window_left,
        ring_rows_per_req=ring_rows_per_req,
        group=group,
    )


def shared_window_sparse_attention(
    *,
    query: torch.Tensor,
    indices: tuple[torch.Tensor, ...],
    group: int,
    row_indptr: torch.Tensor,
    **attention_kwargs,
) -> torch.Tensor:
    """[bs * q_len, heads, kv_lora_rank]: each row's window attention, from the
    group's common window (rows folded into heads) and the row's edge keys.

    ``query`` is [bs, q_len, heads, head_dim]; ``indices`` come from
    ``shared_window_indices``; ``attention_kwargs`` are the trtllm-gen sparse MLA
    arguments the per-row path passes (kv_cache, workspace, scales, ...).
    """
    bs, q_len, heads, head_dim = query.shape
    rows, groups = bs * q_len, bs * q_len // group
    common, common_counts, edge, edge_counts = indices
    # Row k of a group is heads k * heads .. (k + 1) * heads of the group's query.
    common_out, common_lse = dcp_trtllm_sparse_attention(
        query=query.view(groups, 1, group * heads, head_dim),
        block_tables=common.unsqueeze(1),
        seq_lens=common_counts,
        cu_seqlens_q=row_indptr[: groups + 1],
        sparse_mla_top_k=common.shape[1],
        need_lse=True,
        **attention_kwargs,
    )
    edge_out, edge_lse = dcp_trtllm_sparse_attention(
        query=query.view(rows, 1, heads, head_dim),
        block_tables=edge.unsqueeze(1),
        seq_lens=edge_counts,
        cu_seqlens_q=row_indptr[: rows + 1],
        sparse_mla_top_k=edge.shape[1],
        need_lse=True,
        **attention_kwargs,
    )
    return merge_shared_window(
        common_out.view(groups, group * heads, -1),
        common_lse,
        edge_out.view(rows, heads, -1),
        edge_lse,
        group=group,
        # trtllm-gen returns log2-based LSEs.
        base2=True,
    )
