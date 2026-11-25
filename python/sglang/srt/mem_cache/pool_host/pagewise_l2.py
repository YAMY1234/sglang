"""Page-wise HiCache L2 transfers for page_first_direct host pools (default off).

SGLANG_HICACHE_PAGEWISE_LOAD=1: a host page holds every layer contiguously, so a
run of consecutive layers of one page is one copy entry. Loads copy whole page
runs into a device stage and place each layer from there with device-side
indices; DSA write-back gathers on the device and copies whole pages. KV bytes
are unchanged: only the copy route differs.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, List, Tuple

import numpy as np
import torch

from sglang.srt.utils import is_cuda

if is_cuda():
    from sgl_kernel.kvcacheio import (
        transfer_embedding_ranges_direct,
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_per_layer_direct_pf_lf,
    )

_MERGE_RUNS = os.environ.get("SGLANG_HICACHE_PAGEWISE_MERGE_RUNS", "0") == "1"
_merge_logged = [False]


def pagewise_enabled() -> bool:
    return os.environ.get("SGLANG_HICACHE_PAGEWISE_LOAD", "0") == "1"


def pagewise_groups() -> int:
    return max(1, int(os.environ.get("SGLANG_HICACHE_PAGEWISE_LOAD_GROUPS", "4")))


def pagewise_scratch_bytes() -> int:
    return max(1, int(os.environ.get("SGLANG_HICACHE_PAGEWISE_SCRATCH_MIB", "256"))) << 20


def layer_runs(host_layers: Iterable[int]) -> List[Tuple[int, int, List[int]]]:
    """[(first host layer, count, positions in the input)] for runs of consecutive host layers."""
    order = sorted(enumerate(host_layers), key=lambda x: x[1])
    runs: List[Tuple[int, int, List[int]]] = []
    for pos, layer in order:
        if runs and layer == runs[-1][0] + runs[-1][1]:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1, runs[-1][2] + [pos])
        else:
            runs.append((layer, 1, [pos]))
    return runs


def _segment_view(host_buffer: torch.Tensor, nseg: int) -> torch.Tensor:
    """[pages, layers - nseg + 1, 1, nseg * layer_bytes] uint8 view: element [p, l] is
    layers l .. l + nseg - 1 of page p, which are contiguous in a page-first buffer."""
    flat = host_buffer.view(torch.uint8) if host_buffer.dtype != torch.uint8 else host_buffer
    pages, layers = host_buffer.shape[0], host_buffer.shape[1]
    page_bytes = host_buffer.stride(0) * host_buffer.element_size()
    layer_bytes = host_buffer.stride(1) * host_buffer.element_size()
    assert host_buffer.is_contiguous() and page_bytes == layers * layer_bytes
    return torch.as_strided(
        flat.view(-1),
        (pages, layers - nseg + 1, 1, nseg * layer_bytes),
        (page_bytes, layer_bytes, nseg * layer_bytes, 1),
    )


def copy_page_runs_to_stage(
    host_buffer: torch.Tensor,
    host_pages_cpu: torch.Tensor,
    first_layer: int,
    nseg: int,
    stage: torch.Tensor,
) -> None:
    """stage[i] = layers first_layer .. +nseg of host page host_pages_cpu[i]: one entry per page."""
    n = host_pages_cpu.numel()
    if _MERGE_RUNS and n and first_layer == 0 and nseg == host_buffer.shape[1]:
        _copy_whole_page_runs(host_buffer, host_pages_cpu, stage)
        return
    transfer_kv_per_layer_direct_pf_lf(
        src_ptrs=[_segment_view(host_buffer, nseg)],
        dst_ptrs=[stage.view(n, -1)],
        src_indices=host_pages_cpu,
        dst_indices=torch.arange(n, dtype=host_pages_cpu.dtype),
        layer_id=first_layer,
        page_size=1,
    )


def _copy_whole_page_runs(host_buffer, host_pages_cpu, stage) -> None:
    """Whole-page group: one copy range per run of consecutive host pages."""
    n = host_pages_cpu.numel()
    flat = host_buffer.view(torch.uint8) if host_buffer.dtype != torch.uint8 else host_buffer
    page_bytes = host_buffer.stride(0) * host_buffer.element_size()
    src = flat.view(host_buffer.shape[0], page_bytes)
    dst = (stage.view(torch.uint8) if stage.dtype != torch.uint8 else stage).view(n, -1)
    assert dst.shape[1] == page_bytes, (dst.shape, page_bytes)
    hp = host_pages_cpu.numpy()
    starts = np.concatenate(([0], np.flatnonzero(np.diff(hp) != 1) + 1))
    lengths = np.diff(np.concatenate((starts, [n])))
    if not _merge_logged[0]:
        _merge_logged[0] = True
        logging.getLogger(__name__).info(
            "HiCache page-wise load: merged page runs on (SGLANG_HICACHE_PAGEWISE_MERGE_RUNS), first load %d pages in %d runs",
            n, len(starts),
        )
    transfer_embedding_ranges_direct(src, dst, hp[starts].tolist(), starts.tolist(), lengths.tolist())


def copy_stage_to_host_pages(
    stage: torch.Tensor, host_buffer: torch.Tensor, host_pages_cpu: torch.Tensor
) -> None:
    """Whole host pages host_pages_cpu[i] = stage[i]: one entry per page."""
    n = host_pages_cpu.numel()
    host_u8 = host_buffer.view(torch.uint8) if host_buffer.dtype != torch.uint8 else host_buffer
    transfer_kv_all_layer_direct_lf_pf(
        src_ptrs=[stage.view(n, -1)],
        dst_ptrs=[host_u8.view(host_buffer.shape[0], 1, 1, -1)],
        src_indices=torch.arange(n, dtype=host_pages_cpu.dtype),
        dst_indices=host_pages_cpu,
        page_size=1,
    )


def chunk_pages(bytes_per_page: int) -> int:
    return max(1, pagewise_scratch_bytes() // max(1, bytes_per_page))
