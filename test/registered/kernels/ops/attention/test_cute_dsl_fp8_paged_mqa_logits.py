# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsa import cutedsl_paged_mqa_logits, pick_dsl_expand
from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.paged_mqa import (
    BLOCK_KV,
    HEAD_DIM,
    assert_paged_mqa_matches_ref,
    generate_paged_mqa_test_data,
    ref_fp8_paged_mqa_logits,
)

register_cuda_ci(est_time=180, stage="nightly", runner_config="4-gpu-b200")


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


@pytest.mark.skipif(
    not is_sm100_supported(),
    reason="CuTe DSL FP8 Paged MQA Logits only supports SM 100 family.",
)
@pytest.mark.parametrize("next_n", [1, 4])
def test_cutedsl_paged_mqa_logits_one_page_block_table(next_n):
    """Every request fits in one 64-token page, so the block table is one column
    wide while each 128-token compute tile prefetches two page slots. The kernel
    must clamp the second slot to the row end instead of reading past the table
    (an OOB read on the last row, hit by every EAGLE autotune dummy forward), and
    the clamped slot must not change any logit at a position < context_len."""
    import deep_gemm

    batch_size, num_heads = 128, 64
    narrow_len, wide_len = BLOCK_KV, 2 * BLOCK_KV
    data = generate_paged_mqa_test_data(
        batch_size,
        next_n,
        num_heads,
        avg_context_len=48,
        max_model_len=narrow_len,
        min_context_len=next_n + 1,
    )
    assert data["block_table"].shape[1] == 1
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    is_target_verify = next_n >= 2
    if is_target_verify:
        dsl_expand_factor, dsl_atom = pick_dsl_expand(
            next_n,
            batch_size=batch_size,
            max_ctx=wide_len,
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

    def run(block_table, max_seq_len):
        return cutedsl_paged_mqa_logits(
            data["q_fp8"].view(batch_size * next_n, num_heads, HEAD_DIM),
            data["kv_fused"],
            data["weights"],
            context_lens,
            block_table.repeat_interleave(next_n, dim=0),
            schedule_metadata,
            max_seq_len,
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

    # Narrow table: the second page slot of every row is clamped.
    logits_narrow = run(data["block_table"], narrow_len)
    ref_logits = ref_fp8_paged_mqa_logits(
        data["q_fp8"],
        data["kv_fp8"],
        data["kv_scales"],
        data["weights"],
        context_lens,
        data["block_table"],
        narrow_len,
        BLOCK_KV,
    )
    assert_paged_mqa_matches_ref(
        logits_narrow, ref_logits, context_lens, batch_size, next_n, narrow_len
    )

    # Wide table: the second slot points at a real, distinct random page per row
    # (unclamped read). Logits at positions < context_len must be identical.
    num_blocks = data["kv_fused"].shape[0]
    extra_pages = torch.arange(
        num_blocks - batch_size, num_blocks, dtype=torch.int32, device="cuda"
    )
    block_table_wide = torch.stack([data["block_table"][:, 0], extra_pages], dim=1)
    logits_wide = run(block_table_wide, wide_len)[:, :narrow_len]
    positions = torch.arange(narrow_len, device="cuda").unsqueeze(0)
    rows = torch.arange(batch_size * next_n, device="cuda")
    end_pos = context_lens[rows // next_n] - next_n + rows % next_n
    valid = positions <= end_pos.unsqueeze(1)
    assert torch.equal(
        logits_narrow.masked_fill(~valid, 0), logits_wide.masked_fill(~valid, 0)
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
