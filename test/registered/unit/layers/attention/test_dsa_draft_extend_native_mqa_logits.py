"""SGLANG_OPT_DSA_DRAFT_EXTEND_NATIVE_MQA_LOGITS: MTP draft extend scores each
request's next_n rows as one DeepGEMM paged-MQA request, with target verify's
[bs, next_n] context-length layout, and keeps the per-row logits."""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsa.paged_mqa_logits import (
    deepgemm_paged_mqa_logits_native,
    deepgemm_paged_mqa_logits_split,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PLATFORM = "sglang.srt.layers.attention.dsa_backend.get_platform"
_SWITCH = envs.SGLANG_OPT_DSA_DRAFT_EXTEND_NATIVE_MQA_LOGITS
BS, NEXT_N, BLOCK_KV = 3, 4, 4
SEQ_LENS = [7, 13, 4]


def _ctx_lens(forward_mode, seqlens_expanded, *, switch, is_sm100=True):
    """``_build_paged_mqa_schedule_2d_ctx_lens`` on a CPU backend stand-in."""
    backend = SimpleNamespace(
        speculative_num_draft_tokens=NEXT_N, dsa_shards_index_k=False
    )
    build = MethodType(
        DeepseekSparseAttnBackend._build_paged_mqa_schedule_2d_ctx_lens, backend
    )
    cache_seqlens = torch.tensor(SEQ_LENS, dtype=torch.int32)
    with (
        patch(_PLATFORM, return_value=SimpleNamespace(is_sm100=is_sm100)),
        _SWITCH.override(switch),
    ):
        return build(forward_mode, cache_seqlens, seqlens_expanded, BS)


def _expanded(rows_per_req=NEXT_N):
    """Each request's per-row visible length, ending at its full length."""
    return torch.cat(
        [torch.arange(n - rows_per_req + 1, n + 1, dtype=torch.int32) for n in SEQ_LENS]
    )


def _reference_paged_mqa_logits(
    q, kv_cache, weights, context_lens, block_table, schedule, max_len, clean_logits
):
    """CPU stand-in for DeepGEMM's paged MQA logits: row (b, j) scores the first
    context_lens[b, j] keys of request b's pages; later columns stay -inf."""
    batch, next_n = q.shape[:2]
    logits = torch.full((batch * next_n, max_len), float("-inf"))
    for b in range(batch):
        keys = kv_cache[block_table[b]].reshape(-1, kv_cache.shape[-1])
        for j in range(next_n):
            n = int(context_lens[b, j])
            row = b * next_n + j
            logits[row, :n] = weights[row] @ torch.relu(q[b, j] @ keys[:n].T)
    return logits


class TestDSADraftExtendNativeMQALogits(CustomTestCase):
    def test_draft_extend_takes_target_verify_layout(self):
        verify = _ctx_lens(ForwardMode.TARGET_VERIFY, _expanded(), switch=True)
        draft = _ctx_lens(ForwardMode.DRAFT_EXTEND_V2, _expanded(), switch=True)
        self.assertEqual(tuple(draft.shape), (BS, NEXT_N))
        self.assertTrue(torch.equal(draft, verify))

    def test_other_cases_keep_per_row_lens(self):
        cases = (
            # (switch, is_sm100, rows per request)
            (False, True, NEXT_N),  # switch off: main's layout
            (True, False, NEXT_N),  # DeepGEMM-native next_n is SM100-only
            (True, True, NEXT_N + 1),  # widened draft window
        )
        for switch, is_sm100, rows in cases:
            with self.subTest(switch=switch, is_sm100=is_sm100, rows=rows):
                expanded = _expanded(rows)
                lens = _ctx_lens(
                    ForwardMode.DRAFT_EXTEND_V2,
                    expanded,
                    switch=switch,
                    is_sm100=is_sm100,
                )
                self.assertTrue(torch.equal(lens, expanded.view(-1, 1)))

    def test_native_logits_match_per_row_logits(self):
        num_heads, head_dim, pad_rows = 2, 8, 2
        max_blocks = -(-max(SEQ_LENS) // BLOCK_KV)
        max_len = max_blocks * BLOCK_KV
        rows = BS * NEXT_N
        gen = torch.Generator().manual_seed(0)

        def small_ints(*shape):
            # Integer-valued floats keep both paths' sums exact.
            return torch.randint(-2, 3, shape, generator=gen).float()

        kv_cache = small_ints(BS * max_blocks, BLOCK_KV, head_dim)
        # DP padding rows past the real q rows are ignored by both paths.
        q = small_ints(rows + pad_rows, num_heads, head_dim)
        weights = small_ints(rows + pad_rows, num_heads)
        # Draft extend's per-row page table repeats each request's row next_n times.
        block_table = torch.randperm(BS * max_blocks, generator=gen).view(BS, -1)
        block_table = block_table.repeat_interleave(NEXT_N, dim=0)

        expanded = _expanded()
        per_row = deepgemm_paged_mqa_logits_split(
            _reference_paged_mqa_logits,
            q,
            kv_cache,
            weights,
            _ctx_lens(ForwardMode.DRAFT_EXTEND_V2, expanded, switch=False),
            block_table,
            None,
            max_len,
            q_offset=rows,
        )
        native = deepgemm_paged_mqa_logits_native(
            _reference_paged_mqa_logits,
            q,
            kv_cache,
            weights,
            _ctx_lens(ForwardMode.DRAFT_EXTEND_V2, expanded, switch=True),
            block_table,
            None,
            max_len,
            q_offset=rows,
            B=BS,
            next_n=NEXT_N,
        )
        self.assertEqual(tuple(native.shape), tuple(per_row.shape))
        # The top-k masks each row past its visible length.
        for row, n in enumerate(expanded.tolist()):
            self.assertTrue(torch.equal(native[row, :n], per_row[row, :n]))


if __name__ == "__main__":
    unittest.main()
