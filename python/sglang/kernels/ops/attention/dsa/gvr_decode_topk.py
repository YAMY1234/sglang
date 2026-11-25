"""Experimental existing GVR selector and selected-only page mapping."""
import importlib.util
import logging
import os
import threading

import torch

# vendored = gvr_vendor/, which not every tree ships; public = public TensorRT-LLM
# 4ccdca1e, unmodified. Unset: vendored when gvr_vendor/ is present, else public.
_VENDORED = "sglang.kernels.ops.attention.dsa.gvr_vendor"
_SRC = os.environ.get("SGLANG_DSA_DECODE_GVR_TOPK_SRC") or (
    "vendored" if importlib.util.find_spec(_VENDORED) is not None else "public"
)
if _SRC == "public":
    from sglang.kernels.ops.attention.dsa.gvr_public import (
        gvr_topk_decode_self_sampling_host as gvr,
    )
elif _SRC == "vendored":
    try:
        from sglang.kernels.ops.attention.dsa.gvr_vendor import (
            gvr_topk_decode_self_sampling_host as gvr,
        )
    except ModuleNotFoundError as error:
        if error.name != _VENDORED:
            raise
        raise RuntimeError(
            f"SGLANG_DSA_DECODE_GVR_TOPK_SRC=vendored needs {_VENDORED}, which this "
            "tree does not ship; use public"
        ) from error
else:
    raise ValueError(f"SGLANG_DSA_DECODE_GVR_TOPK_SRC must be vendored or public, got {_SRC!r}")
from sglang.kernels.ops.attention.dsa.gvr_page_map import page_map

# Each stream owns its slab; graphs and outputs never borrow another stream's
# scratch. Strong references keep the allocator from recycling graph pointers.
_workspaces = {}
_workspace_lock = threading.Lock()
_logger = logging.getLogger(__name__)
_logged = False
# One deferred callable (forward_mla's verify q-prep fork), run right before the GVR launch
# so its side-stream work overlaps GVR; run_deferred() is the caller's fallback.
_deferred = None


def defer_until_topk(fn):
    global _deferred
    assert _deferred is None, "a deferred q-prep fork is already pending"
    _deferred = fn


def run_deferred():
    global _deferred
    fn, _deferred = _deferred, None
    if fn is not None:
        fn()


def _workspace(logits):
    key = (logits.device.index, torch.cuda.current_stream(logits.device).cuda_stream)
    cached = _workspaces.get(key)
    if cached is not None:
        return cached
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("GVR stream workspace must be warmed before capture")
    with _workspace_lock:
        if key not in _workspaces:
            _workspaces[key] = torch.zeros(
                gvr.workspace_bytes(), dtype=torch.uint8, device=logits.device
            )
    return _workspaces[key]


def gvr_decode_topk(logits, lengths, page_table, cache_lengths):
    global _logged
    run_deferred()
    rows = logits.shape[0]
    raw = logits.new_empty((rows, 2048), dtype=torch.int32)
    output = torch.empty_like(raw)
    if _SRC == "public":  # public entry is hint-free only: no pre_idx argument
        gvr.run_varlen(
            logits, cache_lengths, raw, next_n=8, compress_ratio=1,
            max_seq_len=logits.shape[1], workspace=_workspace(logits),
        )
    else:
        gvr.run_varlen(
            logits, None, cache_lengths, raw, next_n=8, compress_ratio=1,
            max_seq_len=logits.shape[1], workspace=_workspace(logits),
        )
    page_map(raw, page_table, lengths, output, block=256, page=64)
    if not _logged:
        _logger.info(
            "DSA_GVR_DECODE_TOPK_ENQUEUED rows=%s cols=%s next_n=8 k=2048 page=64 src=%s",
            rows, logits.shape[1], _SRC,
        )
        _logged = True
    return output
