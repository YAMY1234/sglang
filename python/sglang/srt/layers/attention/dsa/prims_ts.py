"""FlashInfer Prims-TS sparse MLA for the DSA backend (``prims_ts``).

The kernel reads the raw FP8 MLA cache rows that the ``trtllm`` DSA backend writes
(576 B/token: 512 NoPE + 64 RoPE E4M3, one per-tensor scale) with an E4M3 query
and FP8 QK/PV MMA, and returns BF16 output with a natural-log LSE. It shares the
``trtllm`` route's query preparation, KV store, top-k transforms and DCP slot
localization; only the attention kernel differs.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.flashinfer_vendored import load_vendored

logger = logging.getLogger(__name__)

# SGLANG_ENABLE_PRIMS_TS_SM107 runs FlashInfer prims_ts at 784bfaf5 (#8186). The
# image overlays main 26bb6dc9's prims_ts/ (docker/k31-image); 784bfaf5 differs in
# three files, vendored verbatim beside this module and loaded under their own names.
_FI_784BFAF5_DIR = Path(__file__).with_name("flashinfer_prims_ts_784bfaf5")
# module under flashinfer.attention.prims_ts: its git blob at 784bfaf5
_FI_784BFAF5_FILES = {
    "mla_decode": "8101912f87e572cca263092e8d69d40cb2a49532",
    "kernels.mla_decode.helpers.nvfp4_kv": "194d608e0901e483f61aa74ef653a4182deb3887",
    "kernels.mla_decode.throughput_2cta.nvfp4_kv_load": (
        "12befbaebb3b1d74ce2ef6c48f2e0b411b8eca56"
    ),
}
# sha256 over every *.py (relative path, NUL, bytes; sorted), as the image's marker
_PRIMS_TS_TREES = {
    "31ce036cc1e27ab7c4372b2b07bb0755a63d7590e579931df28cc6b84de15633": "26bb6dc9",
    "ac72191a4a6db76cdaa8768e758ba87683f938e5021f1cbc62bc56a0d1fa0d97": "784bfaf5",
}


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


@functools.cache
def _use_flashinfer_prims_ts_784bfaf5() -> str:
    """Make flashinfer.attention.prims_ts FlashInfer 784bfaf5; returns the tree."""
    package = "flashinfer.attention.prims_ts"
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(f"SGLANG_ENABLE_PRIMS_TS_SM107: {package} is not installed")
    root = Path(next(iter(spec.submodule_search_locations)))
    tree = _PRIMS_TS_TREES.get(_tree_sha256(root))
    if tree == "784bfaf5":
        return tree
    if tree != "26bb6dc9":
        raise RuntimeError(
            "SGLANG_ENABLE_PRIMS_TS_SM107 needs FlashInfer prims_ts at 26bb6dc9 (the "
            "vendored 784bfaf5 files replace three of them) or 784bfaf5; "
            f"{root} is neither"
        )
    if package in sys.modules:
        raise RuntimeError(
            f"SGLANG_ENABLE_PRIMS_TS_SM107: {package} (26bb6dc9) was imported before "
            "the 784bfaf5 files could be loaded"
        )
    load_vendored(
        {
            f"{package}.{name}": (
                _FI_784BFAF5_DIR.joinpath(*name.split(".")).with_suffix(".py"),
                blob,
            )
            for name, blob in _FI_784BFAF5_FILES.items()
        }
    )
    logger.info(f"FlashInfer prims_ts: 784bfaf5 (#8186 files from {_FI_784BFAF5_DIR})")
    return tree


def _flashinfer_prims_ts():
    if envs.SGLANG_ENABLE_PRIMS_TS_SM107.get():
        _use_flashinfer_prims_ts_784bfaf5()
    import flashinfer.attention.prims_ts as prims_ts

    return prims_ts


# Decode/verify graphs plan their exact row count; eager batches above this share
# power-of-two capacity plans, so prefill chunks do not plan one kernel per size.
_EXACT_PLAN_MAX_ROWS = 1024


def _plan_capacity(rows: int) -> Optional[int]:
    """None plans exactly ``rows`` fixed rows; otherwise the packed row capacity."""
    if rows <= _EXACT_PLAN_MAX_ROWS:
        return None
    return 1 << (rows - 1).bit_length()


def check_prims_ts_sparse_mla_support(
    *, device: torch.device, num_heads: int, kv_lora_rank: int, qk_rope_head_dim: int
) -> None:
    """Fail at startup, not at the first forward, when the installed FlashInfer
    cannot plan the D576 native-E4M3 sparse MLA this backend needs."""
    if (kv_lora_rank, qk_rope_head_dim) != (512, 64):
        raise ValueError(
            "--dsa-*-backend prims_ts supports D576 MLA "
            f"(kv_lora_rank 512, qk_rope_head_dim 64); got {kv_lora_rank}, "
            f"{qk_rope_head_dim}."
        )
    allowed = ((10, 0), (10, 3))
    if envs.SGLANG_ENABLE_PRIMS_TS_SM107.get():
        allowed += ((10, 7),)
    if torch.cuda.get_device_capability(device) not in allowed:
        raise ValueError(
            "Prims-TS sparse MLA requires SM100 or SM103 "
            "(SM107 with SGLANG_ENABLE_PRIMS_TS_SM107=1)."
        )
    try:
        _flashinfer_prims_ts().get_prims_ts_sparse_mla_decode_workspace_size(
            device,
            1,
            num_heads,
            max_topk=2048,
            q_dtype=torch.float8_e4m3fn,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_cache_format="native",
            return_lse=True,
        )
    except (ImportError, TypeError, ValueError, NotImplementedError) as exc:
        raise RuntimeError(
            "--dsa-*-backend prims_ts needs FlashInfer Prims-TS with D576 native-FP8 "
            "sparse MLA (flashinfer-ai/flashinfer#7410 and #7984; SM107 also #8186): "
            f"{exc}"
        ) from exc


class PrimsTSSparseMLA:
    """Planned Prims-TS sparse MLA launches, keyed by row plan, heads and top-k width.

    Plans compile and allocate, so they are made on eager calls; the CUDA graph
    runner's eager warmups plan every captured row count before its capture.
    """

    def __init__(self, device: torch.device):
        # Plans compare tensor devices with this one, so it carries the index.
        if device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        self.device = device
        self._plans: dict[tuple[int, int, int, bool], object] = {}
        # packed plans' [0, rows] query offsets, keyed like _plans
        self._qo_indptr: dict[tuple[int, int, int, bool], torch.Tensor] = {}

    def _wrapper(self, *, rows: int, heads: int, width: int):
        BatchSparseMLADecodePagedTSWrapper = (
            _flashinfer_prims_ts().BatchSparseMLADecodePagedTSWrapper
        )
        capacity = _plan_capacity(rows)
        packed = capacity is not None
        key = (capacity if packed else rows, heads, width, packed)
        wrapper = self._plans.get(key)
        if wrapper is not None:
            return wrapper, self._qo_indptr.get(key)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Prims-TS sparse MLA has no plan for {rows} rows x {heads} heads; "
                "graph capture must follow an eager run of the same shape."
            )
        wrapper = BatchSparseMLADecodePagedTSWrapper()
        wrapper.plan(
            self.device,
            1 if packed else rows,
            heads,
            max_topk=width,
            max_q_len=capacity if packed else 1,
            packed_query=packed,
            q_dtype=torch.float8_e4m3fn,
            qk_rope_head_dim=64,
            kv_cache_format="native",
            return_lse=True,
        )
        self._plans[key] = wrapper
        if packed:
            self._qo_indptr[key] = torch.zeros(2, dtype=torch.int32, device=self.device)
        return wrapper, self._qo_indptr.get(key)

    def forward(
        self,
        *,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        sm_scale: float,
        kv_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q [rows, H, 576] E4M3; kv_cache raw rows viewable as [N, 576] E4M3;
        indices [rows, K] int32 cache rows (-1 holes); lengths [rows] int32 <= K.
        Returns BF16 [rows, H, 512] and natural-log LSE [rows, H] (-inf, zero
        output for a row with no keys)."""
        rows, heads, head_dim = q.shape
        out = torch.empty((rows, heads, 512), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((rows, heads), dtype=torch.float32, device=q.device)
        if rows == 0:
            return out, lse
        width = indices.shape[1]
        wrapper, qo_indptr = self._wrapper(rows=rows, heads=heads, width=width)
        kv_rows = kv_cache.view(torch.float8_e4m3fn).view(-1, head_dim)
        # The per-tensor KV descale multiplies both the QK logits and the
        # weighted V sum; the query is cast unscaled.
        kwargs = dict(
            softmax_scale=sm_scale * kv_scale,
            output_scale=kv_scale,
            validate=False,
        )
        if qo_indptr is not None:
            qo_indptr[1:].fill_(rows)
            wrapper.run(
                q,
                kv_rows,
                indices,
                lengths,
                qo_indptr=qo_indptr,
                out=out,
                lse=lse,
                **kwargs,
            )
        else:
            wrapper.run(
                q.view(rows, 1, heads, head_dim),
                kv_rows,
                indices,
                lengths,
                out=out.view(rows, 1, heads, 512),
                lse=lse.view(rows, 1, heads),
                **kwargs,
            )
        return out, lse
