"""DSA decode indexer logits over the paged FP8 index-K cache with one Q block per request.

Drop-in for ``deep_gemm.fp8_paged_mqa_logits`` (sgl-deep-gemm 0.2.8, legacy SM100 path) at 32
heads x 128, 64-key pages, 2-D context lengths and next_n 1..8. DeepGEMM's Q block is 4 rows, so
next_n 5..8 splits each request into two blocks that each stream the request's index K; this
kernel (``jit/csrc/dsa/paged_mqa_logits_1blk.cuh``, adapted from DeepGEMM, MIT) keeps all next_n
rows in one block (UMMA N = next_n x 32) and reads the index K once. It takes DeepGEMM's own
schedule metadata (the work split is per request and 256-key split, independent of the Q block)
and writes bitwise the same logits, including which columns stay unwritten.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, cuda_stubs_dir, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

NUM_HEADS, HEAD_DIM, PAGE_KV, SPLIT_KV = 32, 128, 64, 256
MAX_NEXT_N = 8


def default_config(next_n: int) -> tuple[int, int, int, bool]:
    """(Q stages, KV stages, TMEM stages, weights in shared memory) for next_n.

    TMEM: 3 accumulator stages while 3 x next_n x 32 <= 512 columns, else 2. Shared memory
    (227 KB): DeepGEMM's 5 KV stages are kept; Q stages drop to 2 (1 at next_n 8). Weights stay in
    registers (next_n x 32 of them) up to next_n 5 and are read from the Q stage beyond.
    """
    tmem_stages = 3 if 3 * next_n * NUM_HEADS <= 512 else 2
    q_stages = 2 if next_n <= 7 else 1
    return q_stages, 5, tmem_stages, next_n > 5


def _deep_gemm_include() -> str:
    import deep_gemm

    root = os.path.join(os.path.dirname(deep_gemm.__file__), "include")
    if not os.path.exists(os.path.join(root, "deep_gemm/legacy/layout/mqa_logits.cuh")):
        raise RuntimeError(
            f"paged MQA logits 1blk: needs DeepGEMM >= 0.2.8 headers under {root}"
        )
    return root


@cache_once
def _jit_module(
    next_n: int,
    q_stages: int,
    kv_stages: int,
    tmem_stages: int,
    weights_in_smem: bool,
    cuh_path: Optional[str] = None,
) -> Module:
    args = make_cpp_args(next_n, q_stages, kv_stages, tmem_stages, weights_in_smem)
    return load_jit(
        "dsa_paged_mqa_logits_1blk",
        *args,
        cuda_files=[cuh_path or "dsa/paged_mqa_logits_1blk.cuh"],
        cuda_wrappers=[("run", f"sgl_mqa1b::PagedMQALogits1Blk<{args}>::run")],
        # DeepGEMM's own JIT flags; no fast-math, the bitwise contract depends on it.
        extra_cuda_cflags=[
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--compiler-options=-fconcepts",
            "--ptxas-options=--register-usage-level=10",
        ],
        extra_include_paths=[_deep_gemm_include()],
        extra_ldflags=[f"-L{cuda_stubs_dir()}", "-lcuda"],
    )


def supports(next_n: int, num_heads: int, head_dim: int) -> bool:
    return 1 <= next_n <= MAX_NEXT_N and num_heads == NUM_HEADS and head_dim == HEAD_DIM


def fp8_paged_mqa_logits_1blk(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    clean_logits: bool = False,
    *,
    config: Optional[tuple[int, int, int, bool]] = None,
    cuh_path: Optional[str] = None,
) -> torch.Tensor:
    """Same arguments and result as ``deep_gemm.fp8_paged_mqa_logits(..., clean_logits=False)``.

    q [B, next_n, 32, 128] e4m3; kv_cache [blocks, 64, 1, 132] uint8; weights [B * next_n, 32]
    fp32; context_lens [B, next_n] int32; block_table [B, pages] int32 (row stride free);
    schedule_meta from ``deep_gemm.get_paged_mqa_logits_metadata(context_lens, 64, num_sms)``.
    Returns [B * next_n, max_context_len] fp32 with a 256-aligned row stride; columns past the
    request's last 256-key split are left unwritten, as DeepGEMM leaves them.
    """
    assert not clean_logits, (
        "paged MQA logits 1blk: clean_logits is not supported (as in DeepGEMM)"
    )
    batch_size, next_n = q.shape[0], q.shape[1]
    assert supports(next_n, q.shape[2], q.shape[3]), (
        f"unsupported q shape {tuple(q.shape)}"
    )
    aligned = (max_context_len + SPLIT_KV - 1) // SPLIT_KV * SPLIT_KV
    logits = torch.empty(
        (batch_size * next_n, aligned), dtype=torch.float32, device=q.device
    )
    module = _jit_module(next_n, *(config or default_config(next_n)), cuh_path)
    module.run(q, kv_cache, weights, context_lens, block_table, schedule_meta, logits)
    return logits[:, :max_context_len]


# ---------------------------------------------------------------- MXFP4 Q x MXFP4 K
# jit/csrc/dsa/paged_mqa_logits_1blk_mxfp4.cuh: DeepGEMM's MXFP4 paged path (kind::mxf4 block32 UMMA,
# K 64 x 2, scale factors in TMEM) with one Q block per request; bitwise
# ``deep_gemm.fp8_fp4_paged_mqa_logits((q_fp4, q_sf), ..., False, torch.float)``.
MXFP4_BYTES_PER_KEY = HEAD_DIM // 2 + 4


def default_config_mxfp4(next_n: int) -> tuple[int, int, int, bool]:
    """(Q stages, KV stages, TMEM stages, weights in shared memory) for next_n.

    DeepGEMM's FP4 stages (3 Q, 10 KV) fit 227 KB up to next_n 8. TMEM holds the accumulators plus
    the Q and KV scale factors (align(next_n x 32, 128) / 32 + 8 columns): 3 stages up to next_n 5.
    """
    n = next_n * NUM_HEADS
    sf_cols = -(-n // 128) * 128 // 32 + SPLIT_KV // 32
    tmem_stages = 3 if 3 * n + sf_cols <= 512 else 2
    return 3, 10, tmem_stages, next_n > 5


@cache_once
def _jit_module_mxfp4(
    next_n: int,
    q_stages: int,
    kv_stages: int,
    tmem_stages: int,
    weights_in_smem: bool,
    cuh_path: Optional[str] = None,
) -> Module:
    args = make_cpp_args(next_n, q_stages, kv_stages, tmem_stages, weights_in_smem)
    return load_jit(
        "dsa_paged_mqa_logits_1blk_mxfp4",
        *args,
        cuda_files=[cuh_path or "dsa/paged_mqa_logits_1blk_mxfp4.cuh"],
        cuda_wrappers=[("run", f"sgl_mqa1b_mx4::PagedMQALogits1BlkMXFP4<{args}>::run")],
        # DeepGEMM's own JIT flags; no fast-math, the bitwise contract depends on it.
        extra_cuda_cflags=[
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--compiler-options=-fconcepts",
            "--ptxas-options=--register-usage-level=10",
        ],
        extra_include_paths=[_deep_gemm_include()],
        extra_ldflags=[f"-L{cuda_stubs_dir()}", "-lcuda"],
    )


def fp4_paged_mqa_logits_1blk(
    q: torch.Tensor,
    q_sf: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    *,
    config: Optional[tuple[int, int, int, bool]] = None,
    cuh_path: Optional[str] = None,
) -> torch.Tensor:
    """Same result as ``deep_gemm.fp8_fp4_paged_mqa_logits((q, q_sf), kv_cache, ..., False, torch.float)``.

    q [B, next_n, 32, 64] int8 (E2M1 pairs); q_sf [B, next_n, 32] int32 (UE8M0 x4); kv_cache
    [blocks, 64, 1, 68] uint8; weights [B * next_n, 32] fp32; context_lens [B, next_n] int32;
    block_table [B, pages] int32; schedule_meta from DeepGEMM's metadata for (context_lens, 64, num_sms).
    Returns [B * next_n, max_context_len] fp32 with a 256-aligned row stride; columns past the
    request's last 256-key split are left unwritten, as DeepGEMM leaves them.
    """
    batch_size, next_n = q.shape[0], q.shape[1]
    assert supports(next_n, q.shape[2], q.shape[3] * 2), (
        f"unsupported q shape {tuple(q.shape)}"
    )
    aligned = (max_context_len + SPLIT_KV - 1) // SPLIT_KV * SPLIT_KV
    logits = torch.empty(
        (batch_size * next_n, aligned), dtype=torch.float32, device=q.device
    )
    module = _jit_module_mxfp4(
        next_n, *(config or default_config_mxfp4(next_n)), cuh_path
    )
    module.run(
        q, q_sf, kv_cache, weights, context_lens, block_table, schedule_meta, logits
    )
    return logits[:, :max_context_len]
