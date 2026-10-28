"""Cached-descriptor CUDA kernels for EAGLE decode index preparation."""

from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.common import is_cuda

_INDEX_DTYPES = (torch.int32, torch.int64)
_is_cuda = is_cuda()
_descriptors: dict[str, tuple[torch.Tensor, object]] = {}


def _runtime_optimization_enabled() -> bool:
    """Keep the generic index kernels behind the runtime master switch."""
    from sglang.srt.kernel_plugins.nvidia.binding import (
        nv_mega_runtime_optimization_enabled,
    )

    return nv_mega_runtime_optimization_enabled()


@cache_once
def _module():
    return load_jit(
        "speculative_index",
        cuda_files=["elementwise/speculative_index.cuh"],
        cuda_wrappers=[
            ("chain_tree", "SpeculativeIndex::chain_tree"),
            ("extend_layout", "SpeculativeIndex::extend_layout"),
            ("draft_locs", "SpeculativeIndex::draft_locs"),
        ],
    )


def _descriptor(name: str, num_entries: int):
    entry = _descriptors.get(name)
    if entry is None:
        tensor = torch.zeros((num_entries,), dtype=torch.int64)
        entry = _descriptors[name] = (tensor, tensor.numpy())
    return entry


def chain_tree(
    *,
    bonus,
    draft,
    seq,
    mask,
    positions,
    retrieve,
    tokens,
    req,
    table,
    out_cache_loc,
    mrope,
) -> bool:
    if not _is_cuda or not _runtime_optimization_enabled() or not (
        bonus.is_cuda
        and bonus.dtype in _INDEX_DTYPES
        and draft.dtype in _INDEX_DTYPES
        and seq.dtype in _INDEX_DTYPES
        and tokens.dtype in _INDEX_DTYPES
        and (req is None or req.dtype in _INDEX_DTYPES)
        and (table is None or table.dtype in _INDEX_DTYPES)
    ):
        return False
    batch_size, depth = draft.shape
    descriptor, values = _descriptor("chain_tree", 27)
    values[0] = bonus.data_ptr()
    values[1] = draft.data_ptr()
    values[2] = seq.data_ptr()
    values[3] = mask.data_ptr()
    values[4] = positions.data_ptr()
    values[5] = retrieve.data_ptr()
    values[6] = tokens.data_ptr()
    values[7] = req.data_ptr() if req is not None else 0
    values[8] = table.data_ptr() if table is not None else 0
    values[9] = out_cache_loc.data_ptr() if out_cache_loc is not None else 0
    values[10] = mrope.data_ptr() if mrope is not None else 0
    values[11] = bonus.dtype is torch.int64
    values[12] = bonus.stride(0)
    values[13] = draft.dtype is torch.int64
    values[14] = draft.stride(0)
    values[15] = draft.stride(1)
    values[16] = seq.dtype is torch.int64
    values[17] = seq.stride(0)
    values[18] = tokens.dtype is torch.int64
    values[19] = req is not None and req.dtype is torch.int64
    values[20] = req.stride(0) if req is not None else 0
    values[21] = table is not None and table.dtype is torch.int64
    values[22] = table.stride(0) if table is not None else 0
    values[23] = table.stride(1) if table is not None else 0
    values[24] = batch_size
    values[25] = depth + 1
    values[26] = (table is not None) | ((mrope is not None) << 1)
    _module().chain_tree(seq, descriptor)
    return True


def extend_layout(
    *, seq, width, prefix, extend, post, positions, starts, mrope
) -> bool:
    if (
        not _is_cuda
        or not _runtime_optimization_enabled()
        or not seq.is_cuda
        or seq.dtype not in _INDEX_DTYPES
    ):
        return False
    descriptor, values = _descriptor("extend_layout", 13)
    values[0] = seq.data_ptr()
    values[1] = prefix.data_ptr()
    values[2] = extend.data_ptr()
    values[3] = post.data_ptr()
    values[4] = positions.data_ptr()
    values[5] = starts.data_ptr()
    values[6] = mrope.data_ptr() if mrope is not None else 0
    values[7] = seq.dtype is torch.int64
    values[8] = seq.stride(0)
    values[9] = post.dtype is torch.int64
    values[10] = seq.numel()
    values[11] = width
    values[12] = mrope is not None
    _module().extend_layout(seq, descriptor)
    return True


def draft_locs(
    *,
    req,
    table,
    seq,
    out_cache_loc,
    pool_len,
    topk,
    steps,
    batch_size,
    positions,
    mrope,
) -> bool:
    if not _is_cuda or not _runtime_optimization_enabled() or not (
        req.is_cuda
        and req.dtype in _INDEX_DTYPES
        and table.dtype in _INDEX_DTYPES
        and seq.dtype in _INDEX_DTYPES
        and out_cache_loc.dtype is torch.int64
        and (positions is None or positions.dtype in _INDEX_DTYPES)
        and (mrope is None or mrope.dtype in _INDEX_DTYPES)
        and req.is_contiguous()
        and seq.is_contiguous()
    ):
        return False
    descriptor, values = _descriptor("draft_locs", 17)
    values[0] = req.data_ptr()
    values[1] = table.data_ptr()
    values[2] = seq.data_ptr()
    values[3] = out_cache_loc.data_ptr()
    values[4] = positions.data_ptr() if positions is not None else 0
    values[5] = mrope.data_ptr() if mrope is not None else 0
    values[6] = req.dtype is torch.int64
    values[7] = table.dtype is torch.int64
    values[8] = seq.dtype is torch.int64
    values[9] = positions is not None and positions.dtype is torch.int64
    values[10] = mrope is not None and mrope.dtype is torch.int64
    values[11] = pool_len
    values[12] = topk
    values[13] = steps
    values[14] = batch_size
    values[15] = positions is not None
    values[16] = mrope is not None
    _module().draft_locs(seq, descriptor)
    return True
