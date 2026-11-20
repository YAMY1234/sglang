"""Debug only (not for merge): bitwise shadow compare of DeepGEMM's paged MQA logits against the one-block kernel
on every DG-native target-verify call (SGLANG_K31_DSA_PAGED_MQA_ONEBLOCK_SHADOW=1 serves DeepGEMM's logits, =2 the
one-block kernel's). Device-only ops, so it records inside CUDA graphs; per layer it keeps
[calls, mismatching in-length columns, compared columns, first mismatch row, col, row length, last-row length,
mismatch rows that are padded requests (length 0 last row)] and copies them to pinned host memory every call; a
daemon thread logs the totals every 15 s when they change.
"""

import logging
import threading
import time

import torch

logger = logging.getLogger(__name__)

_dev = {}
_host = {}
_lock = threading.Lock()
_thread = None


def _logger_loop():
    last = None
    while True:
        time.sleep(15)
        with _lock:
            snap = {k: v.tolist() for k, v in _host.items()}
        if not snap:
            continue
        calls = sum(v[0] for v in snap.values())
        mism = sum(v[1] for v in snap.values())
        comp = sum(v[2] for v in snap.values())
        cur = (calls, mism, comp)
        if cur == last:
            continue
        last = cur
        bad = {k: v[3:8] for k, v in sorted(snap.items()) if v[1] > 0}
        logger.info(
            "MQA1BLK_SHADOW calls=%d mismatching_cols=%d compared_cols=%d layers_with_mismatch=%d first(row,col,len,lastlen,padrows)=%s",
            calls, mism, comp, len(bad), dict(list(bad.items())[:6]),
        )


def compare(layer_id: int, ref: torch.Tensor, new: torch.Tensor, lens2d: torch.Tensor, q_offset: int) -> None:
    global _thread
    st = _dev.get(layer_id)
    if st is None:
        st = torch.zeros(8, dtype=torch.int64, device=ref.device)
        _dev[layer_id] = st
        with _lock:
            _host[layer_id] = torch.zeros(8, dtype=torch.int64).pin_memory()
        if _thread is None:
            _thread = threading.Thread(target=_logger_loop, daemon=True)
            _thread.start()
    rows, cols = q_offset, ref.shape[1]
    nn = lens2d.shape[1]
    lens = lens2d.reshape(-1)[:rows].to(torch.int64).clamp(min=0)
    last = lens2d[:, -1].to(torch.int64).clamp(min=0).repeat_interleave(nn)[:rows]
    in_len = torch.arange(cols, device=ref.device)[None, :] < lens[:, None]
    neq = (ref[:rows].view(torch.int32) != new[:rows].view(torch.int32)) & in_len
    n = neq.sum()
    first = torch.argmax(neq.reshape(-1).to(torch.int8))
    r, c = first // cols, first % cols
    pad_rows = ((neq.any(dim=1)) & (last <= 1)).sum()  # graph-padded requests (fill 1)
    is_first = (st[1] == 0) & (n > 0)
    info = torch.stack([r, c, lens[r], last[r], pad_rows])
    st[0] += 1
    st[1] += n
    st[2] += in_len.sum()
    st[3:8].copy_(torch.where(is_first, info, st[3:8]))
    _host[layer_id].copy_(st, non_blocking=True)
