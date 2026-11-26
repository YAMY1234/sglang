"""Page-wise HiCache L2 transfers for ``page_first_direct`` host pools.

A ``page_first_direct`` host page holds its layers back to back, so a run of
consecutive layers of one page is one contiguous copy. With
SGLANG_HICACHE_PAGEWISE_LOAD, a load copies such runs into a device stage (one
copy per page and run instead of one per page and layer) and places each layer
from the stage with device-side indices; a raw DSA index write-back gathers
each page's layers on the device and copies whole pages. The bytes are
unchanged.
"""

from __future__ import annotations

from typing import Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda

if is_cuda():
    from sgl_kernel.kvcacheio import (
        transfer_embedding_ranges_direct,
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_per_layer_direct_pf_lf,
    )

# Device stage one page-wise copy step may take on its transfer stream; the
# step's page count follows from it. Arbitrary.
PAGEWISE_STAGE_BYTES = 256 << 20


def uses_pagewise_load(host_pool, io_backend: str) -> bool:
    """Whether L2 transfers of ``host_pool`` take the page-wise path."""
    supported = getattr(host_pool, "pagewise_load_ok", None)
    return (
        envs.SGLANG_HICACHE_PAGEWISE_LOAD.get()
        and supported is not None
        and supported(io_backend)
    )


def chunk_pages(stage_bytes_per_page: int) -> int:
    return max(1, PAGEWISE_STAGE_BYTES // stage_bytes_per_page)


def split_layer_groups(layer_ids: list[int], groups: int) -> list[list[int]]:
    """``layer_ids`` in consecutive groups of ceil(len / groups) layers."""
    size = max(1, -(-len(layer_ids) // max(1, groups)))
    return [layer_ids[i : i + size] for i in range(0, len(layer_ids), size)]


def layer_runs(host_layers: Sequence[int]) -> list[tuple[int, int, list[int]]]:
    """Runs of consecutive host layers in host-layer order, as (first layer,
    length, positions of the run's layers in ``host_layers``)."""
    runs = []
    for pos in sorted(range(len(host_layers)), key=host_layers.__getitem__):
        layer = host_layers[pos]
        if runs and layer == runs[-1][0] + runs[-1][1]:
            runs[-1][1] += 1
            runs[-1][2].append(pos)
        else:
            runs.append([layer, 1, [pos]])
    return [tuple(run) for run in runs]


def contiguous_runs(ids: torch.Tensor) -> tuple[list[int], list[int]]:
    """(start positions, lengths) of the runs of consecutive values in a
    non-empty 1-D CPU tensor."""
    starts = torch.cat((ids.new_zeros(1), torch.nonzero(ids.diff() != 1)[:, 0] + 1))
    lengths = starts.diff(append=starts.new_tensor([ids.numel()]))
    return starts.tolist(), lengths.tolist()


def _layer_run_view(host_buffer: torch.Tensor, nseg: int) -> torch.Tensor:
    """The bytes of a page-first host buffer as [pages, layers - nseg + 1, 1,
    nseg * layer_bytes]: element [p, l] is layers l .. l + nseg - 1 of page p.
    Pages keep the buffer's stride, storage padding included."""
    u8 = host_buffer.view(torch.uint8)
    assert u8[0].is_contiguous(), "a host page must hold its layers back to back"
    layer_bytes = u8.stride(1)
    return u8.as_strided(
        (u8.shape[0], u8.shape[1] - nseg + 1, 1, nseg * layer_bytes),
        (u8.stride(0), layer_bytes, nseg * layer_bytes, 1),
    )


def copy_page_runs_to_stage(
    host_buffer: torch.Tensor,
    host_pages: torch.Tensor,
    first_layer: int,
    nseg: int,
    stage: torch.Tensor,
) -> None:
    """stage[i] = host layers first_layer .. first_layer + nseg - 1 of page
    host_pages[i] (CPU ids): one copy per page, or one per run of consecutive
    pages when the layers are a whole page of an unpadded buffer."""
    n = host_pages.numel()
    dst = stage.view(n, -1)
    if nseg == host_buffer.shape[1] and host_buffer.is_contiguous():
        starts, lengths = contiguous_runs(host_pages)
        transfer_embedding_ranges_direct(
            host_buffer.view(torch.uint8).view(host_buffer.shape[0], -1),
            dst,
            host_pages[starts].tolist(),
            starts,
            lengths,
        )
        return
    transfer_kv_per_layer_direct_pf_lf(
        src_ptrs=[_layer_run_view(host_buffer, nseg)],
        dst_ptrs=[dst],
        src_indices=host_pages,
        dst_indices=torch.arange(n),
        layer_id=first_layer,
        page_size=1,
    )


def copy_stage_to_host_pages(
    stage: torch.Tensor, host_buffer: torch.Tensor, host_pages: torch.Tensor
) -> None:
    """Whole host pages host_pages[i] (CPU ids) = stage[i]: one copy per page."""
    n = host_pages.numel()
    transfer_kv_all_layer_direct_lf_pf(
        src_ptrs=[stage.view(n, -1)],
        dst_ptrs=[_layer_run_view(host_buffer, host_buffer.shape[1])],
        src_indices=torch.arange(n),
        dst_indices=host_pages,
        page_size=1,
    )
