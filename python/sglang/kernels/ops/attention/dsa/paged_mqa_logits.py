# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable

import torch


def can_use_cutedsl_cap8(
    q, kv, weights, context_lens, schedule, block_tables, max_seq_len,
    *, B, next_n, q_offset, extend_lens, graph_query_group_size=None,
) -> bool:
    """Only the fixed-eight 32-head FP8 ABI; no tensor-value/host reads."""
    return (
        next_n == 8
        and 0 < B <= 32
        and q_offset == B * 8
        and (
            (
                graph_query_group_size is None
                and len(extend_lens) == B
                and all(n == 8 for n in extend_lens)
            )
            or (
                graph_query_group_size == 8
                and len(extend_lens) == B * 8
                and all(n == 1 for n in extend_lens)
            )
        )
        and q.shape == (q_offset, 32, 128)
        and q.dtype == torch.float8_e4m3fn
        and q.is_contiguous()
        and len(kv.shape) == 4
        and kv.shape[1:] == (64, 1, 132)
        and kv.dtype == torch.uint8
        and kv.is_contiguous()
        and weights.shape == (q_offset, 32)
        and weights.dtype == torch.float32
        and weights.is_contiguous()
        and context_lens is not None
        and context_lens.shape == (B,)
        and context_lens.dtype == torch.int32
        and context_lens.is_contiguous()
        and schedule is not None
        and len(schedule.shape) == 2
        and schedule.shape[1] == 2
        and schedule.dtype == torch.int32
        and schedule.is_contiguous()
        and 64 <= max_seq_len <= 1048704
        and max_seq_len % 64 == 0
        and block_tables.shape == (q_offset, max_seq_len // 64)
        and block_tables.dtype == torch.int32
        and block_tables.stride(1) == 1
    )


def cutedsl_paged_mqa_logits_cap8(
    q, kv, weights, context_lens, block_tables, schedule, max_seq_len, *, B,
):
    from sglang.kernels.ops.attention.dsa.cutedsl_paged_mqa_logits import (
        CuteDSLPagedMQALogitsRunner,
    )

    return CuteDSLPagedMQALogitsRunner.forward(
        q.view(B, 8, 32, 128), kv, weights, context_lens,
        block_tables[::8], schedule, max_seq_len,
        num_epi_subtiles=1, pipeline_flags={"max_w_in_reg": 8},
        block_table_strided=True,
    )


def _restore_row_stride(logits: torch.Tensor) -> torch.Tensor:
    """Undo PyTorch's DLPack stride normalization on single-row DeepGEMM logits.

    DeepGEMM returns paged-MQA logits as a row-padded view:
    ``torch.empty(num_rows, aligned_len)[:, :max_len]`` with ``aligned_len``
    256-element (1024-byte) aligned. tvm-ffi builds of DeepGEMM (sgl-deep-gemm
    >= 0.1.x) round-trip that view through DLPack on return, and PyTorch's
    DLPack *exporter* rewrites the stride of every size<2 dim to 1 whenever it
    differs from the packed expectation (pytorch/pytorch#83158). A one-row
    result whose row is actually padded (``max_len % 256 != 0``, e.g. any
    ``model context_len + 4`` page-table width at bs=1 decode capture) therefore
    arrives with ``stride() == (1, 1)`` instead of ``(aligned_len, 1)``, which
    violates the fused top-k v2 kernel ABI (``score_stride % 4 == 0``, enforced
    both in ``dsa_topk_backend._topk_transform_v2_paged`` and by the kernel's
    own RuntimeCheck).

    For ``num_rows <= 1`` the row stride is semantically arbitrary (row 0 is
    the only row ever addressed and ``(num_rows - 1) * stride(0)`` contributes
    nothing to the storage extent), so restoring a 16-byte-aligned value is a
    pure metadata rewrite: same storage, same data pointer, no copy and no
    kernel launch -- trivially CUDA-graph-capture-safe. Multi-row results keep
    their true strides through DLPack (no size<2 dim) and pass through
    untouched.
    """
    if logits.shape[0] <= 1 and logits.stride(0) % 4 != 0:
        width = logits.shape[1]
        logits = logits.as_strided((logits.shape[0], width), ((width + 3) // 4 * 4, 1))
    return logits


def deepgemm_paged_mqa_logits_native(
    fp8_paged_mqa_logits_fn: Callable[..., torch.Tensor],
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    ctx_lens_2d: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_seq_len: int,
    *,
    q_offset: int,
    B: int,
    next_n: int,
) -> torch.Tensor:
    # block_tables[::next_n] de-expands the caller's repeat_interleave without a
    # copy (DeepGEMM only checks `stride(1) == 1`).
    logits = fp8_paged_mqa_logits_fn(
        q_fp8[:q_offset].view(B, next_n, q_fp8.shape[1], q_fp8.shape[2]),
        kv_cache_fp8,
        weights[:q_offset],
        ctx_lens_2d,
        block_tables[::next_n],
        schedule_metadata,
        max_seq_len,
        clean_logits=False,
    )
    return _restore_row_stride(logits)


def deepgemm_paged_mqa_logits_split(
    fp8_paged_mqa_logits_fn: Callable[..., torch.Tensor],
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    ctx_lens_2d: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_seq_len: int,
    *,
    q_offset: int,
) -> torch.Tensor:
    q_fp8 = q_fp8.unsqueeze(1)
    logits = fp8_paged_mqa_logits_fn(
        q_fp8[:q_offset],
        kv_cache_fp8,
        weights[:q_offset],
        ctx_lens_2d,
        block_tables,
        schedule_metadata,
        max_seq_len,
        clean_logits=False,
    )
    return _restore_row_stride(logits)


def aiter_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_seq_len: int,
    *,
    preshuffle: bool,
    kv_block_size: int,
) -> torch.Tensor:
    from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

    q_fp8 = q_fp8.unsqueeze(1)
    batch_size, next_n, _, _ = q_fp8.shape
    logits = torch.empty(
        (batch_size * next_n, max_seq_len),
        device=q_fp8.device,
        dtype=torch.float32,
    )
    deepgemm_fp8_paged_mqa_logits(
        q_fp8,
        kv_cache_fp8,
        weights,
        logits,
        seq_lens,
        block_tables,
        max_seq_len,
        Preshuffle=preshuffle,
        KVBlockSize=kv_block_size,
    )
    return logits


def cutedsl_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    ctx_lens_1d: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor | None,
    max_seq_len: int,
    *,
    q_offset: int,
    B: int,
    next_n: int,
    is_target_verify: bool,
    dsl_expand_factor: int,
    dsl_atom: int,
    blocksize: int,
    sm_count: int,
    get_paged_mqa_logits_metadata_fn: Callable[..., torch.Tensor],
) -> torch.Tensor:
    from sglang.kernels.ops.attention.dsa.cutedsl_paged_mqa_logits import (
        CuteDSLPagedMQALogitsRunner,
    )

    dsl_atom_split = dsl_expand_factor > 1 and next_n == dsl_expand_factor * dsl_atom
    if is_target_verify and dsl_atom_split:
        exp_B = B * dsl_expand_factor
        q_dsl = q_fp8[:q_offset].view(exp_B, dsl_atom, q_fp8.shape[1], q_fp8.shape[2])
        ctx_lens_1d = ctx_lens_1d.repeat_interleave(dsl_expand_factor)
        block_tables_dsl = block_tables[::next_n].repeat_interleave(
            dsl_expand_factor, dim=0
        )
        schedule_metadata = get_paged_mqa_logits_metadata_fn(
            ctx_lens_1d.unsqueeze(-1), blocksize, sm_count
        )
    elif is_target_verify and next_n >= 2:
        # Native single-launch: one task per batch entry (the kernel iterates
        # next_n internally), so the schedule must be built from B-length
        # context lens, not the caller's [B, next_n] or per-token layout.
        q_dsl = q_fp8[:q_offset].view(B, next_n, q_fp8.shape[1], q_fp8.shape[2])
        block_tables_dsl = block_tables[::next_n]
        schedule_metadata = get_paged_mqa_logits_metadata_fn(
            ctx_lens_1d.unsqueeze(-1), blocksize, sm_count
        )
    else:
        q_dsl = q_fp8[:q_offset].unsqueeze(1)
        block_tables_dsl = block_tables[:B]

    return CuteDSLPagedMQALogitsRunner.forward(
        q_dsl,
        kv_cache_fp8.view(torch.uint8),
        weights[:q_offset],
        ctx_lens_1d,
        block_tables_dsl,
        schedule_metadata,
        max_seq_len,
    )
