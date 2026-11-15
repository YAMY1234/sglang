"""Buffers and index rules both ends of a tail-stage transfer call alike.

Stage 0 -> stage 1: the ratio-1 kv_source pools (c1 KV and c1 index K of layer 20),
one item per full page, as get_contiguous_buf_infos registers them for PD.
Stage 1 -> decode: DSV41_TAIL_SWA, one item per logical SWA page of one late layer,
as get_state_buf_infos registers the window, on the paged SWA pool both run.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import msgspec
import numpy as np
import torch

C1_RATIO = 1


class BufInfos(msgspec.Struct, frozen=True):
    """As the KVArgs fields hold them."""

    ptrs: List[int]
    lens: List[int]
    item_lens: List[int]


def c1_buf_infos(pool) -> Tuple[BufInfos, BufInfos]:
    """The c1 KV and c1 index buffers of a DeepSeekV4TokenToKVPool."""
    kv_bufs = pool.kv_pools[C1_RATIO].kv_buffer
    kv = BufInfos(
        ptrs=[buf.data_ptr() for buf in kv_bufs],
        lens=[buf.nbytes for buf in kv_bufs],
        item_lens=[buf[0].nbytes for buf in kv_bufs],
    )
    index_pool = pool.index_pools[C1_RATIO]
    slots_per_full_page = pool.page_size // C1_RATIO
    assert slots_per_full_page % index_pool.page_size == 0, (
        slots_per_full_page,
        index_pool.page_size,
    )
    index_pages_per_full_page = slots_per_full_page // index_pool.page_size
    index_bufs = index_pool.contiguous_page_row_buffers()
    index = BufInfos(
        ptrs=[buf.data_ptr() for buf in index_bufs],
        lens=[buf.nbytes for buf in index_bufs],
        item_lens=[buf[0].nbytes * index_pages_per_full_page for buf in index_bufs],
    )
    return kv, index


def c1_page_ids(
    req_to_token_row: torch.Tensor, *, e: int, page_size: int
) -> np.ndarray:
    """Full-page ids of [0, e), one per started page, in position order."""
    return (req_to_token_row[0:e:page_size] // page_size).cpu().numpy().astype(np.int32)


def tail_swa_buf_infos(pool, layer_ids: Sequence[int]) -> BufInfos:
    """The paged SWA buffers of `layer_ids`, one item per logical SWA page."""
    assert pool.request_window is None and pool.swa_kv_pool is not None, (
        "DSV41_TAIL_SWA needs the paged SWA pool (no encoder SWA replay)"
    )
    swa_pool = pool.swa_kv_pool
    physical_pages_per_logical_page = pool.swa_page_size // swa_pool.page_size
    own = pool.swa_layer_ids()
    bufs = [swa_pool.kv_buffer[own.index(layer_id)] for layer_id in layer_ids]
    return BufInfos(
        ptrs=[buf.data_ptr() for buf in bufs],
        lens=[buf.nbytes for buf in bufs],
        item_lens=[buf[0].nbytes * physical_pages_per_logical_page for buf in bufs],
    )


def tail_swa_page_ids(
    pool, req_to_token_row: torch.Tensor, *, start: int, end: int
) -> np.ndarray:
    """Logical SWA page ids holding positions [start, end), in position order."""
    swa_locs = pool.translate_loc_from_full_to_swa(req_to_token_row[start:end])
    pages = (swa_locs // pool.swa_page_size).cpu().numpy()
    first_of_run = np.r_[True, pages[1:] != pages[:-1]]
    return pages[first_of_run].astype(np.int32)


def item_lens_error(
    name: str, src_item_lens: Sequence[int], dst_item_lens: Sequence[int]
) -> Optional[str]:
    if list(src_item_lens) != list(dst_item_lens):
        return (
            f"{name} item lengths differ: tail stage {list(src_item_lens)}, "
            f"peer {list(dst_item_lens)}"
        )
    return None
