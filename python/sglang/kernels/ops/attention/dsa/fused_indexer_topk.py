"""Fused DSA indexer scorer + threshold top-k (FSTK) for the packed PAGED prefill.

Scores are bitwise DeepGEMM's legacy ``fp8_mqa_logits`` (32 heads x 128, FP8, FP32
reduction); the [rows, num_kv] score matrix is never written. Each row first gets a
threshold ``tau`` from every ``sample_stride``-th key of its window (the ``rank``-th
largest sample score), the main pass appends the scores >= tau of the row's window to
a per-row candidate list, and the exact top-k (equal scores: smaller column first) is
selected from it and mapped through the page table. A row whose list holds fewer than
``topk`` entries (tau too high) or overflows ``cand_cap`` is reported for the caller's
fallback path, so the output set is exact for every row.
"""

from __future__ import annotations

import logging
import os
from typing import NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import cache_once, cuda_stubs_dir, load_jit

logger = logging.getLogger(__name__)

MODE_LOGITS, MODE_NO_EPI, MODE_NO_STORE, MODE_FILTER = 0, 1, 2, 3
# layout -> (rows per q-block, keys per split); see LayoutTraits in the kernel.
LAYOUTS = {0: (4, 256), 1: (4, 256), 2: (8, 128), 3: (4, 256), 4: (8, 128), 5: (4, 256)}
DEFAULT_LAYOUT = 5  # measured fastest for the filter pass (VR, 11-20)
MATH_WARPS = 8  # candidate sub-lists per row: one per math warp of the row's CTA


def _deep_gemm_include() -> tuple[str, bool]:
    """DeepGEMM's header root, and whether the MQA logits kernel sits under legacy/."""
    import deep_gemm

    root = os.path.join(os.path.dirname(deep_gemm.__file__), "include")
    legacy = os.path.exists(os.path.join(root, "deep_gemm/legacy/impls/sm100_mqa_logits.cuh"))
    if not legacy and not os.path.exists(os.path.join(root, "deep_gemm/impls/sm100_mqa_logits.cuh")):
        raise RuntimeError(f"fused indexer top-k: no sm100_mqa_logits.cuh under {root}")
    return root, legacy


@cache_once
def _jit_module(cuh_path: Optional[str] = None, serving: bool = False):
    """serving builds only what the prefill path runs (DEFAULT_LAYOUT, scores and filter)."""
    dg_root, dg_legacy = _deep_gemm_include()
    return load_jit(
        "dsa_fused_indexer_topk",
        int(dg_legacy),
        int(serving),
        cuda_files=[cuh_path or "dsa/fused_indexer_topk.cuh"],
        cuda_wrappers=[
            ("score", "fstk::FusedIndexerTopK::score"),
            ("select", "fstk::FusedIndexerTopK::select"),
            ("tau", "fstk::FusedIndexerTopK::tau"),
        ],
        # DeepGEMM's own JIT flags; no fast-math, the bitwise contract depends on it.
        extra_cuda_cflags=[
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--compiler-options=-fconcepts",
            "--ptxas-options=--register-usage-level=10",
            f"-DFSTK_DG_LEGACY={int(dg_legacy)}",
            f"-DFSTK_SERVING={int(serving)}",
            f"-DFSTK_SERVING_LAYOUT={DEFAULT_LAYOUT}",
        ],
        extra_include_paths=[dg_root],
        extra_ldflags=[f"-L{cuda_stubs_dir()}", "-lcuda"],
    )


def _empty(device, dtype=torch.float32):
    return torch.empty(0, device=device, dtype=dtype)


def lpt_order(ks: torch.Tensor, ke: torch.Tensor, num_kv: int, layout: int = DEFAULT_LAYOUT) -> torch.Tensor:
    """q-blocks by descending KV span (the kernel deals them out snake-wise)."""
    block_q, split_kv = LAYOUTS[layout]
    t = ks.shape[0]
    nb = (t + block_q - 1) // block_q
    pad = nb * block_q - t
    s = ks.clamp(max=num_kv).to(torch.int64)
    e = ke.clamp(max=num_kv).to(torch.int64)
    if pad:
        s = torch.cat([s, s[-1:].expand(pad)])
        e = torch.cat([e, e[-1:].expand(pad)])
    s = s.view(nb, block_q).amin(1) // 4 * 4
    e = e.view(nb, block_q).amax(1)
    span = (e - s + split_kv - 1) // split_kv
    return torch.argsort(span, descending=True, stable=True).to(torch.int32)


def score(
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scale: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    *,
    mode: int,
    order: Optional[torch.Tensor] = None,
    logits: Optional[torch.Tensor] = None,
    tau: Optional[torch.Tensor] = None,
    cand_key: Optional[torch.Tensor] = None,
    cand_count: Optional[torch.Tensor] = None,
    reuse_kv: bool = False,
    layout: int = DEFAULT_LAYOUT,
    alu_heads: int = 0,
    num_ctas: Optional[int] = None,
    cuh_path: Optional[str] = None,
    serving: bool = False,
) -> None:
    dev = q_fp8.device
    if num_ctas is None:
        num_ctas = torch.cuda.get_device_properties(dev).multi_processor_count
    _jit_module(cuh_path, serving).score(
        q_fp8,
        kv_fp8,
        kv_scale,
        weights,
        ks,
        ke,
        order if order is not None else _empty(dev, torch.int32),
        logits if logits is not None else _empty(dev),
        tau if tau is not None else _empty(dev),
        cand_key if cand_key is not None else _empty(dev, torch.int64),
        cand_count if cand_count is not None else _empty(dev, torch.int32),
        mode,
        int(reuse_kv),
        layout,
        alu_heads,
        num_ctas,
    )


def _sample_stride_elems(num_samples: int) -> int:
    return (num_samples + 512 + 255) // 256 * 256


def scratch_bytes_per_row(num_kv: int, sample_stride: int = 16, cand_cap: int = 16384) -> int:
    """Device scratch of fused_indexer_topk per query row (sample scores, candidates)."""
    num_samples = (num_kv + sample_stride - 1) // sample_stride
    return 4 * _sample_stride_elems(num_samples) + 8 * cand_cap + 4 * MATH_WARPS + 16


class FusedTopKResult(NamedTuple):
    out: torch.Tensor  # [rows, topk] int32 physical KV slots, -1 padded
    num_fallback: torch.Tensor  # [1] int32
    fallback_rows: torch.Tensor  # [rows] int32, first num_fallback valid
    tau: torch.Tensor
    cand_count: torch.Tensor


def fused_indexer_topk(
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scale: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    lengths: torch.Tensor,
    row_to_batch: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    topk: int,
    *,
    sample_stride: int = 16,
    rank: int = 192,
    cand_cap: int = 16384,
    keep_all: Optional[int] = None,
    order: Optional[torch.Tensor] = None,
    layout: int = DEFAULT_LAYOUT,
    alu_heads: int = 0,
    out: Optional[torch.Tensor] = None,
    cuh_path: Optional[str] = None,
    serving: bool = False,
) -> FusedTopKResult:
    """Exact top-k of every packed row; rows listed in ``fallback_rows`` were not written."""
    dev = q_fp8.device
    rows = q_fp8.shape[0]
    num_kv = kv_fp8.shape[0]
    # Rows this short keep every score; the 8 sub-lists of such a row stay within the cap.
    keep_all = cand_cap // 2 if keep_all is None else keep_all
    if order is None:
        order = lpt_order(ks, ke, num_kv, layout)

    # Sample keys: every sample_stride-th flattened position; a row's window [ks, ke)
    # holds the samples [ceil(ks / s), ceil(ke / s)).
    kv_s = kv_fp8[::sample_stride].contiguous()
    sc_s = kv_scale[::sample_stride].contiguous()
    sks = (ks + sample_stride - 1) // sample_stride
    ske = (ke + sample_stride - 1) // sample_stride
    n_s = kv_s.shape[0]
    # Room for a q-block's span past the last key and for its padded rows.
    stride_s = _sample_stride_elems(n_s)
    sample_logits = torch.empty(rows + 8, stride_s, device=dev, dtype=torch.float32)
    score(
        q_fp8,
        kv_s,
        sc_s,
        weights,
        sks,
        ske,
        mode=MODE_LOGITS,
        order=lpt_order(sks, ske, n_s, layout),
        logits=sample_logits,
        layout=layout,
        alu_heads=alu_heads,
        cuh_path=cuh_path,
        serving=serving,
    )
    tau = torch.empty(rows, device=dev, dtype=torch.float32)
    module = _jit_module(cuh_path, serving)
    module.tau(sample_logits, sks, ske, lengths, tau, rank, topk, keep_all)
    del sample_logits

    # Select keys: (order-preserving score bits << 32) | ~window column, one sub-list per
    # math warp; a warp that never sees a row leaves its zero count.
    cand_key = torch.empty(rows, MATH_WARPS, cand_cap // MATH_WARPS, device=dev, dtype=torch.int64)
    cand_count = torch.zeros(rows, MATH_WARPS, device=dev, dtype=torch.int32)
    score(
        q_fp8,
        kv_fp8,
        kv_scale,
        weights,
        ks,
        ke,
        mode=MODE_FILTER,
        order=order,
        tau=tau,
        cand_key=cand_key,
        cand_count=cand_count,
        layout=layout,
        alu_heads=alu_heads,
        cuh_path=cuh_path,
        serving=serving,
    )
    # The select kernel writes rows of exactly topk entries.
    dst = out if out is not None and out.is_contiguous() else None
    if dst is None:
        dst = torch.empty(rows, topk, device=dev, dtype=torch.int32)
    fallback_rows = torch.empty(rows, device=dev, dtype=torch.int32)
    num_fallback = torch.zeros(1, device=dev, dtype=torch.int32)
    module.select(
        cand_key,
        cand_count,
        lengths,
        row_to_batch,
        page_table,
        dst,
        fallback_rows,
        num_fallback,
        page_size,
    )
    if out is None:
        out = dst
    elif dst is not out:
        out.copy_(dst)
    return FusedTopKResult(out, num_fallback, fallback_rows, tau, cand_count)


class FusedTopKStats:
    """Process-wide counters of the serving path, logged every ``log_interval`` calls."""

    def __init__(self, log_interval: int, check_interval: int):
        self.log_interval = log_interval
        self.check_interval = check_interval
        self.total = dict(calls=0, rows=0, long_rows=0, fallback_rows=0, max_fallback=0, chunks=0,
                          budget_mb=0, check_rows=0, check_differ=0, check_tie_explained=0, check_tie_checked=0)
        self.window = dict(self.total)

    def record(self, *, rows: int, long_rows: int, fallback_rows: int, chunks: int = 1, budget: int = 0) -> bool:
        """Counts one call; True when this call should also run the stock-path check."""
        for d in (self.total, self.window):
            d["calls"] += 1
            d["rows"] += rows
            d["long_rows"] += long_rows
            d["fallback_rows"] += fallback_rows
            d["max_fallback"] = max(d["max_fallback"], fallback_rows)
            d["chunks"] += chunks
            d["budget_mb"] = budget >> 20
        calls = self.total["calls"]
        if self.log_interval and calls % self.log_interval == 0:
            logger.info("FSTK window %s total %s", self.window, self.total)
            self.window = {k: 0 for k in self.window}
        return bool(self.check_interval) and calls % self.check_interval == 0

    def record_check(self, *, rows: int, differ: int, tie_explained: int, tie_checked: int) -> None:
        for d in (self.total, self.window):
            d["check_rows"] += rows
            d["check_differ"] += differ
            d["check_tie_explained"] += tie_explained
            d["check_tie_checked"] += tie_checked
        logger.info("FSTK check rows=%d differ=%d tie_explained=%d/%d", rows, differ, tie_explained, tie_checked)
