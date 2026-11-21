"""JIT-compiled Mamba KV cache transfer kernel.

Provides ``transfer_kv_mamba_pf_lf`` (load: page_first -> layer_first)
and ``transfer_kv_mamba_lf_pf`` (backup: layer_first -> page_first).

Uses the shared ``load_jit`` + ``cache_once`` infrastructure from
``sglang.kernels.jit.utils`` — the same mechanism used by ``hicache.py``
for MHA/MLA staged write-back kernels.  This ensures consistent
content-addressed caching, CUDA arch detection, and multi-worker
JIT compilation behavior across all JIT kernels.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    import torch
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)


def precompile_transfer_mamba() -> None:
    """Build and load the JIT module now instead of at the first host transfer."""
    _jit_transfer_mamba_module()


@cache_once
def _jit_transfer_mamba_module() -> Module:
    return load_jit(
        "transfer_mamba",
        cuda_files=["kvcacheio/transfer_mamba.cuh"],
        cuda_wrappers=[
            ("transfer_kv_mamba_pf_lf", "&TransferMambaKernel::run_pf_lf"),
            ("transfer_kv_mamba_lf_pf", "&TransferMambaKernel::run_lf_pf"),
            (
                "transfer_kv_mamba_pf_lf_chunked",
                "&TransferMambaKernel::run_pf_lf_chunked",
            ),
            (
                "transfer_kv_mamba_lf_pf_chunked",
                "&TransferMambaKernel::run_lf_pf_chunked",
            ),
        ],
    )


@debug_kernel_api
def transfer_kv_mamba_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    layer_id: int,
    item_size: int,
    src_layout_dim: int,
    num_warps_per_item: int = 32,
    block_quota: int = 0,
):
    module = _jit_transfer_mamba_module()
    if block_quota > 0:
        # Grid sized to the transfer (chunks of each item), capped at block_quota.
        module.transfer_kv_mamba_pf_lf_chunked(
            src,
            dst,
            src_indices,
            dst_indices,
            layer_id,
            item_size,
            src_layout_dim,
            block_quota,
        )
        return
    module.transfer_kv_mamba_pf_lf(
        src,
        dst,
        src_indices,
        dst_indices,
        layer_id,
        item_size,
        src_layout_dim,
    )


@debug_kernel_api
def transfer_kv_mamba_lf_pf(
    src_ptrs: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    num_warps_per_item: int = 32,
    block_quota: int = 0,
):
    module = _jit_transfer_mamba_module()
    if block_quota > 0:
        module.transfer_kv_mamba_lf_pf_chunked(
            src_ptrs,
            dst,
            src_indices,
            dst_indices,
            item_size,
            dst_layout_dim,
            num_layers,
            block_quota,
        )
        return
    module.transfer_kv_mamba_lf_pf(
        src_ptrs,
        dst,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
    )
