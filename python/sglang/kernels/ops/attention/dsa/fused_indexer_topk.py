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

import os
from typing import NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import cache_once, cuda_stubs_dir, load_jit

BLOCK_Q = 4
SPLIT_KV = 256
MODE_LOGITS, MODE_NO_EPI, MODE_NO_STORE, MODE_FILTER = 0, 1, 2, 3


def _deep_gemm_include() -> tuple[str, bool]:
    """DeepGEMM's header root, and whether the MQA logits kernel sits under legacy/."""
    import deep_gemm

    root = os.path.join(os.path.dirname(deep_gemm.__file__), "include")
    legacy = os.path.exists(os.path.join(root, "deep_gemm/legacy/impls/sm100_mqa_logits.cuh"))
    if not legacy and not os.path.exists(os.path.join(root, "deep_gemm/impls/sm100_mqa_logits.cuh")):
        raise RuntimeError(f"fused indexer top-k: no sm100_mqa_logits.cuh under {root}")
    return root, legacy


@cache_once
def _jit_module(cuh_path: Optional[str] = None):
    dg_root, dg_legacy = _deep_gemm_include()
    return load_jit(
        "dsa_fused_indexer_topk",
        int(dg_legacy),
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
        ],
        extra_include_paths=[dg_root],
        extra_ldflags=[f"-L{cuda_stubs_dir()}", "-lcuda"],
    )


def _empty(device, dtype=torch.float32):
    return torch.empty(0, device=device, dtype=dtype)


def lpt_order(ks: torch.Tensor, ke: torch.Tensor, num_kv: int) -> torch.Tensor:
    """q-blocks by descending KV span (the kernel deals them out snake-wise)."""
    t = ks.shape[0]
    nb = (t + BLOCK_Q - 1) // BLOCK_Q
    pad = nb * BLOCK_Q - t
    s = ks.clamp(max=num_kv).to(torch.int64)
    e = ke.clamp(max=num_kv).to(torch.int64)
    if pad:
        s = torch.cat([s, s[-1:].expand(pad)])
        e = torch.cat([e, e[-1:].expand(pad)])
    s = s.view(nb, BLOCK_Q).amin(1) // 4 * 4
    e = e.view(nb, BLOCK_Q).amax(1)
    span = (e - s + SPLIT_KV - 1) // SPLIT_KV
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
    cand_score: Optional[torch.Tensor] = None,
    cand_col: Optional[torch.Tensor] = None,
    cand_count: Optional[torch.Tensor] = None,
    reuse_kv: bool = False,
    row_split: bool = True,
    alu_heads: int = 0,
    num_ctas: Optional[int] = None,
    cuh_path: Optional[str] = None,
) -> None:
    dev = q_fp8.device
    if num_ctas is None:
        num_ctas = torch.cuda.get_device_properties(dev).multi_processor_count
    _jit_module(cuh_path).score(
        q_fp8,
        kv_fp8,
        kv_scale,
        weights,
        ks,
        ke,
        order if order is not None else _empty(dev, torch.int32),
        logits if logits is not None else _empty(dev),
        tau if tau is not None else _empty(dev),
        cand_score if cand_score is not None else _empty(dev),
        cand_col if cand_col is not None else _empty(dev, torch.int32),
        cand_count if cand_count is not None else _empty(dev, torch.int32),
        mode,
        int(reuse_kv),
        int(row_split),
        alu_heads,
        num_ctas,
    )


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
    cand_cap: int = 8192,
    keep_all: Optional[int] = None,
    order: Optional[torch.Tensor] = None,
    cuh_path: Optional[str] = None,
) -> FusedTopKResult:
    """Exact top-k of every packed row; rows listed in ``fallback_rows`` were not written."""
    dev = q_fp8.device
    rows = q_fp8.shape[0]
    num_kv = kv_fp8.shape[0]
    keep_all = cand_cap if keep_all is None else keep_all
    if order is None:
        order = lpt_order(ks, ke, num_kv)

    # Sample keys: every sample_stride-th flattened position; a row's window [ks, ke)
    # holds the samples [ceil(ks / s), ceil(ke / s)).
    kv_s = kv_fp8[::sample_stride].contiguous()
    sc_s = kv_scale[::sample_stride].contiguous()
    sks = (ks + sample_stride - 1) // sample_stride
    ske = (ke + sample_stride - 1) // sample_stride
    n_s = kv_s.shape[0]
    stride_s = (n_s + 2 * SPLIT_KV + 255) // 256 * 256
    sample_logits = torch.empty(rows + BLOCK_Q, stride_s, device=dev, dtype=torch.float32)
    score(
        q_fp8,
        kv_s,
        sc_s,
        weights,
        sks,
        ske,
        mode=MODE_LOGITS,
        order=lpt_order(sks, ske, n_s),
        logits=sample_logits,
        cuh_path=cuh_path,
    )
    tau = torch.empty(rows, device=dev, dtype=torch.float32)
    _jit_module(cuh_path).tau(sample_logits, sks, ske, lengths, tau, rank, topk, keep_all)

    cand_score = torch.empty(rows, cand_cap, device=dev, dtype=torch.float32)
    cand_col = torch.empty(rows, cand_cap, device=dev, dtype=torch.int32)
    cand_count = torch.zeros(rows, device=dev, dtype=torch.int32)
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
        cand_score=cand_score,
        cand_col=cand_col,
        cand_count=cand_count,
        cuh_path=cuh_path,
    )
    out = torch.empty(rows, topk, device=dev, dtype=torch.int32)
    fallback_rows = torch.empty(rows, device=dev, dtype=torch.int32)
    num_fallback = torch.zeros(1, device=dev, dtype=torch.int32)
    _jit_module(cuh_path).select(
        cand_score,
        cand_col,
        cand_count,
        lengths,
        row_to_batch,
        page_table,
        out,
        fallback_rows,
        num_fallback,
        page_size,
    )
    return FusedTopKResult(out, num_fallback, fallback_rows, tau, cand_count)
