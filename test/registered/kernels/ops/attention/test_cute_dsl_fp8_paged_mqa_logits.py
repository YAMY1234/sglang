# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsa import (
    can_use_cutedsl_paged_mqa_logits_cap8,
    cutedsl_paged_mqa_logits,
    cutedsl_paged_mqa_logits_cap8,
    deepgemm_paged_mqa_logits_native,
    pick_dsl_expand,
)
from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.paged_mqa import (
    BLOCK_KV,
    HEAD_DIM,
    assert_paged_mqa_matches_ref,
    generate_paged_mqa_test_data,
    ref_fp8_paged_mqa_logits,
)

register_cuda_ci(est_time=240, stage="nightly", runner_config="4-gpu-b200")


def _run_cutedsl_paged_mqa_logits(
    data, batch_size, next_n, num_heads, max_model_len, is_target_verify
):
    """Mirrors the CUTEDSL dispatch in
    sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._get_topk_paged."""
    import deep_gemm

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    if is_target_verify and next_n >= 2:
        dsl_expand_factor, dsl_atom = pick_dsl_expand(
            next_n,
            batch_size=batch_size,
            max_ctx=max_model_len,
            num_sms=num_sms,
            kernel_atoms=(1, 2, 3, 4),
            num_heads=num_heads,
        )
    else:
        dsl_expand_factor, dsl_atom = 1, 1

    context_lens = data["context_lens"]
    expanded_ctx = (
        context_lens.unsqueeze(-1)
        - next_n
        + torch.arange(1, next_n + 1, device=context_lens.device, dtype=torch.int32)
    ).flatten()
    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
        expanded_ctx.unsqueeze(-1), BLOCK_KV, num_sms
    )
    block_tables_expanded = data["block_table"].repeat_interleave(next_n, dim=0)

    return cutedsl_paged_mqa_logits(
        data["q_fp8"].view(batch_size * next_n, num_heads, HEAD_DIM),
        data["kv_fused"],
        data["weights"],
        context_lens,
        block_tables_expanded,
        schedule_metadata,
        max_model_len,
        q_offset=batch_size * next_n,
        B=batch_size,
        next_n=next_n,
        is_target_verify=is_target_verify,
        dsl_expand_factor=dsl_expand_factor,
        dsl_atom=dsl_atom,
        blocksize=BLOCK_KV,
        sm_count=num_sms,
        get_paged_mqa_logits_metadata_fn=deep_gemm.get_paged_mqa_logits_metadata,
    )


@pytest.mark.skipif(
    not is_sm100_supported(),
    reason="CuTe DSL FP8 Paged MQA Logits only supports SM 100 family.",
)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("next_n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("avg_ctx", [128, 1024, 4096, 16384])
def test_cutedsl_paged_mqa_logits(batch_size, next_n, num_heads, avg_ctx):
    max_model_len = max(avg_ctx * 2, 2048)
    data = generate_paged_mqa_test_data(
        batch_size, next_n, num_heads, avg_ctx, max_model_len
    )

    logits = _run_cutedsl_paged_mqa_logits(
        data,
        batch_size,
        next_n,
        num_heads,
        max_model_len,
        is_target_verify=next_n >= 2,
    )

    ref_logits = ref_fp8_paged_mqa_logits(
        data["q_fp8"],
        data["kv_fp8"],
        data["kv_scales"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        max_model_len,
        BLOCK_KV,
    )
    assert_paged_mqa_matches_ref(
        logits, ref_logits, data["context_lens"], batch_size, next_n, max_model_len
    )


def _run_cutedsl_paged_mqa_logits_cap8(data, batch_size, max_model_len):
    """Mirrors the SGLANG_DSA_DECODE_CAP8_LOGITS dispatch in
    sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._get_topk_paged and its
    metadata from DeepseekSparseAttnBackend._build_cap8_paged_mqa_metadata."""
    import deep_gemm

    next_n = 8
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    q = data["q_fp8"].view(batch_size * next_n, -1, HEAD_DIM)
    # One block-table row per query row, as the indexer receives it in verify.
    block_tables_expanded = data["block_table"].repeat_interleave(next_n, dim=0)
    context_lens = data["context_lens"]
    assert can_use_cutedsl_paged_mqa_logits_cap8(
        q,
        data["kv_fused"],
        data["weights"],
        context_lens,
        block_tables_expanded,
        q_offset=batch_size * next_n,
        B=batch_size,
        next_n=next_n,
    )
    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens.unsqueeze(-1), BLOCK_KV, num_sms
    )
    cap8_logits = cutedsl_paged_mqa_logits_cap8(
        q,
        data["kv_fused"],
        data["weights"],
        context_lens,
        block_tables_expanded,
        schedule_metadata,
        max_model_len,
        q_offset=batch_size * next_n,
        B=batch_size,
    )

    # The DeepGEMM route it replaces, with the backend's [B, next_n] verify lengths.
    ctx_lens_2d = context_lens.unsqueeze(-1).expand(-1, next_n).contiguous()
    dg_logits = deepgemm_paged_mqa_logits_native(
        deep_gemm.fp8_paged_mqa_logits,
        q,
        data["kv_fused"],
        data["weights"],
        ctx_lens_2d,
        block_tables_expanded,
        deep_gemm.get_paged_mqa_logits_metadata(ctx_lens_2d, BLOCK_KV, num_sms),
        max_model_len,
        q_offset=batch_size * next_n,
        B=batch_size,
        next_n=next_n,
    )
    return cap8_logits, dg_logits


@pytest.mark.skipif(
    not is_sm100_supported(),
    reason="CuTe DSL FP8 Paged MQA Logits only supports SM 100 family.",
)
@pytest.mark.parametrize("batch_size", [1, 3, 8, 32])
@pytest.mark.parametrize("avg_ctx", [128, 1024, 4096, 16384])
def test_cutedsl_paged_mqa_logits_cap8(batch_size, avg_ctx):
    next_n, num_heads = 8, 32
    max_model_len = max(avg_ctx * 2, 2048)
    data = generate_paged_mqa_test_data(
        batch_size, next_n, num_heads, avg_ctx, max_model_len
    )

    logits, dg_logits = _run_cutedsl_paged_mqa_logits_cap8(
        data, batch_size, max_model_len
    )

    ref_logits = ref_fp8_paged_mqa_logits(
        data["q_fp8"],
        data["kv_fp8"],
        data["kv_scales"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        max_model_len,
        BLOCK_KV,
    )
    assert_paged_mqa_matches_ref(
        logits, ref_logits, data["context_lens"], batch_size, next_n, max_model_len
    )
    assert_paged_mqa_matches_ref(
        logits, dg_logits, data["context_lens"], batch_size, next_n, max_model_len
    )


def _cap8_gate_inputs(
    batch_size=4,
    next_n=8,
    num_heads=32,
    page_size=BLOCK_KV,
    q_dtype=torch.float8_e4m3fn,
):
    rows = batch_size * next_n
    return dict(
        q_fp8=torch.empty(rows, num_heads, HEAD_DIM, dtype=q_dtype),
        kv_cache_fp8=torch.empty(16, page_size, 1, HEAD_DIM + 4, dtype=torch.uint8),
        weights=torch.empty(rows, num_heads, dtype=torch.float32),
        context_lens=torch.empty(batch_size, dtype=torch.int32),
        block_tables=torch.empty(rows, 4, dtype=torch.int32),
        q_offset=rows,
        B=batch_size,
        next_n=next_n,
    )


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, True),
        ({"next_n": 6}, False),
        ({"num_heads": 64}, False),
        ({"batch_size": 33}, False),
        ({"page_size": 32}, False),
        ({"q_dtype": torch.bfloat16}, False),
    ],
)
def test_cutedsl_paged_mqa_logits_cap8_gate(overrides, expected):
    """Shapes outside the fixed-eight kernel keep the DeepGEMM route."""
    inputs = _cap8_gate_inputs(**overrides)
    assert can_use_cutedsl_paged_mqa_logits_cap8(**inputs) is expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
