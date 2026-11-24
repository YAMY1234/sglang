"""PD debug dump of the DSA index-K rows (SGLANG_DEBUG_K31_IDXK_PD_DUMP_DIR).

Prefill writes the rows it sends and decode the rows it received, for the same
sampled token positions of the first few requests whose bootstrap room is a
multiple of 4 (both sides pick the same requests), one file per process and
request; an offline join on (bootstrap room, layer, position) checks that the
PD path (including the DCP relayout) moves every byte. Debug only: it
synchronizes the device.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch

from sglang.srt.disaggregation.checksum import is_health_check_req
from sglang.srt.environ import envs

_dumped_rooms = set()


def _positions(start: int, end: int) -> torch.Tensor:
    """First and last 64 positions of [start, end) and 128 spread between them."""
    mid = torch.linspace(start + 64, end - 65, 128).long()
    pos = torch.cat([torch.arange(start, start + 64), mid, torch.arange(end - 64, end)])
    return torch.unique(pos)


def maybe_dump_index_k(scheduler, reqs: List, side: str) -> None:
    out_dir = envs.SGLANG_DEBUG_K31_IDXK_PD_DUMP_DIR.get()
    if (
        not out_dir
        or len(_dumped_rooms) >= envs.SGLANG_DEBUG_K31_IDXK_PD_DUMP_MAX.get()
    ):
        return
    pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    if not getattr(pool, "use_dsa", False):
        return
    dsa = pool.full_kv_pool
    page = dsa.index_page_size
    vb, sb = dsa.index_k_value_bytes, dsa.index_k_scale_bytes
    dcp_size, dcp_rank = 1, 0
    if side == "decode":
        from sglang.srt.layers.attention.dsa.dsa_indexer import _index_k_dcp

        dcp_size, dcp_rank = _index_k_dcp()
    layer_ids = sorted(pool.full_attention_layer_id_mapping)
    for req in reqs:
        if len(_dumped_rooms) >= envs.SGLANG_DEBUG_K31_IDXK_PD_DUMP_MAX.get():
            return
        room = req.bootstrap_room
        if (
            room is None
            or room % 4
            or room in _dumped_rooms
            or is_health_check_req(req)
        ):
            continue
        end = len(req.origin_input_ids)
        # positions decode already held (its promised prefix) never cross PD
        start = (
            req.disagg_decode_prefix_len
            if side == "prefill"
            else req.kv.cache_protected_len
        )
        if end - start < 4096:
            continue
        _dumped_rooms.add(room)
        torch.cuda.synchronize()
        pos = _positions(start, end)
        row = scheduler.req_to_token_pool.req_to_token[req.kv.req_pool_idx]
        locs = row[pos.to(row.device)].long()
        # decode DCP: req_to_token holds widened slots; rank r owns slot % size == r at slot // size
        own = (locs % dcp_size) == dcp_rank
        slots = locs // dcp_size
        pg, off = slots // page, slots % page
        vcol = off[:, None] * vb + torch.arange(vb, device=off.device)
        scol = page * vb + off[:, None] * sb + torch.arange(sb, device=off.device)
        values, scales = [], []
        for layer_id in layer_ids:
            buf = pool.get_index_k_with_scale_buffer(layer_id)
            buf = buf.view(torch.uint8).reshape(buf.shape[0], -1)
            assert buf.shape[1] == page * (vb + sb), (buf.shape, page, vb, sb)
            values.append(buf[pg[:, None], vcol])
            scales.append(buf[pg[:, None], scol])
        tag = f"{side}-room{room}-pid{os.getpid()}"
        os.makedirs(out_dir, exist_ok=True)
        np.savez(
            os.path.join(out_dir, tag + ".npz"),
            layers=np.array(layer_ids),
            pos=pos.numpy(),
            own=own.cpu().numpy(),
            values=torch.stack(values).cpu().numpy(),
            scales=torch.stack(scales).cpu().numpy(),
            meta=np.array([start, end, vb, sb, dcp_size, dcp_rank]),
        )
