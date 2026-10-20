"""Cute-DSL MoE routing tables for one EP rank's experts from A2A-received top-k ids (csrc/moe/qwen_ep_tables.cuh)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# csrc kThreads x kMaxPerThread: DEP16 decode at 10 requests per rank (1120 rows x top-10) fits; prefill does not.
EP_TABLES_MAX_EXPANDED = 1024 * 11


@cache_once
def _jit_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "qwen_ep_tables",
        *args,
        cuda_files=["moe/qwen_ep_tables.cuh"],
        cuda_wrappers=[("run", f"sglang::EpTables<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


class EpTablesBuffers:
    """moe_sort's six outputs plus the fields _run_experts_from_tables reads (x_fp4, topk_weights, num_tokens)."""

    def __init__(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        x_fp4: torch.Tensor,
        num_local: int,
        tile_size: int,
    ) -> None:
        rows, top_k = topk_ids.shape
        expanded = rows * top_k
        filled = min(num_local, expanded)
        max_tiles = filled + (expanded - filled) // tile_size
        kw = dict(dtype=torch.int32, device=topk_ids.device)
        self.num_tokens = rows
        self.tile_size = tile_size
        self.x_fp4 = x_fp4
        self.topk_weights = topk_weights
        self.tile_idx_to_expert_idx = torch.empty((max_tiles,), **kw)
        self.tile_idx_to_mn_limit = torch.empty((max_tiles,), **kw)
        self.expanded_idx_to_permuted_idx = torch.empty((rows, top_k), **kw)
        self.permuted_idx_to_expanded_idx = torch.empty((max_tiles * tile_size,), **kw)
        self.total_num_padded_tokens = torch.empty((1,), **kw)
        self.num_non_exiting_tiles = torch.empty((1,), **kw)


def ep_tables(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    x_fp4: torch.Tensor,
    *,
    local_offset: int,
    num_local: int,
    tile_size: int,
) -> EpTablesBuffers:
    """topk_ids int32 [R, top_k] (global ids; ids outside [local_offset, local_offset + num_local) are skipped)."""
    bufs = EpTablesBuffers(topk_ids, topk_weights, x_fp4, num_local, tile_size)
    _jit_module().run(
        topk_ids,
        bufs.tile_idx_to_expert_idx,
        bufs.tile_idx_to_mn_limit,
        bufs.expanded_idx_to_permuted_idx,
        bufs.permuted_idx_to_expanded_idx,
        bufs.total_num_padded_tokens,
        bufs.num_non_exiting_tiles,
        local_offset,
        num_local,
        tile_size.bit_length() - 1,
    )
    return bufs
