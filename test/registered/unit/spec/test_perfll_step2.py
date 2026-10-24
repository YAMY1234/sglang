"""Unit tests for SGLANG_PERFLL_STEP2's counting mode and its three-window decode reserve."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_len_per_decode,
    get_alloc_reserve_per_decode,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.perfll_step2 import NextDraft, Step2Check, Step2Counter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(slot: int, allocated: int, committed: int) -> Req:
    sp = SamplingParams(max_new_tokens=16, temperature=0)
    sp.normalize(None)
    req = Req(
        rid=f"r{slot}",
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=sp,
    )
    req.kv.req_pool_idx = slot
    req.kv.kv_committed_len = committed
    req.kv.kv_allocated_len = allocated
    return req


def _batch(slots, allocated, committed=100):
    """A decode batch of real Req objects (the scheduler's request type)."""
    reqs = [_req(slot, alloc, committed) for slot, alloc in zip(slots, allocated)]
    return SimpleNamespace(reqs=reqs)


class TestPerfllStep2(CustomTestCase):
    def _pass(self, counter, batch):
        counter.count_draft(batch)
        counter.note_extend(batch)

    def test_a_decode_pass_is_staged_only_after_the_same_unpadded_allocated_batch(self):
        """On the scheduler's Req (slot and lengths under req.kv); reading a field Req does not
        have crashed every rank of the first serving run."""
        counter = Step2Counter(width=7, graph_sizes=[1, 2, 4])
        whole = 100 + 21  # committed + three windows
        self._pass(counter, _batch([5, 9], [whole, whole]))
        self._pass(counter, _batch([5, 9], [whole, whole]))
        self._pass(counter, _batch([9, 5], [whole, whole]))
        self._pass(counter, _batch([9, 5, 3], [whole] * 3))
        self._pass(counter, _batch([9, 5, 3], [whole, whole, whole - 1]))
        self._pass(counter, _batch([9, 5], [whole, whole - 1]))
        self._pass(counter, _batch([9, 5], [whole, whole]))
        counter.forget()
        self._pass(counter, _batch([9, 5], [whole, whole]))
        self.assertEqual(
            dict(counter.counts),
            {
                "no_previous_decode": 2,
                "staged": 1,
                "batch_changed": 3,
                "padded": 1,
                "window_not_allocated": 1,
            },
        )
        self.assertEqual(counter.decode_pass_ct, 8)

    def test_mode_two_hands_over_only_inputs_a_post_verify_graph_staged(self):
        counter = Step2Counter(width=7, graph_sizes=[2], needs_next_draft=True)
        whole = 100 + 21
        staged = object()
        counter.note_extend(_batch([5, 9], [whole, whole]), next_draft=staged)
        self.assertIs(counter.count_draft(_batch([5, 9], [whole, whole])), staged)
        counter.note_extend(_batch([5, 9], [whole, whole]), next_draft=None)
        self.assertIsNone(counter.count_draft(_batch([5, 9], [whole, whole])))
        counter.note_extend(_batch([5, 9], [whole, whole]), next_draft=staged)
        self.assertIsNone(counter.count_draft(_batch([9, 5], [whole, whole])))
        self.assertEqual(
            dict(counter.counts),
            {"staged": 1, "not_staged_by_post_verify_graph": 1, "batch_changed": 1},
        )

    def test_the_check_compares_the_draft_slots_with_the_windows_first_columns(self):
        def i64(*v):
            return torch.tensor(v, dtype=torch.int64)

        staged = NextDraft(
            num_requests=2,
            seq_lens=i64(10, 20),
            positions=i64(10, 20),
            mrope=None,
            draft_out_cache_loc=i64(100, 101, 200, 201),
            window=i64(100, 101, 102, 200, 201, 202),
        )
        check = Step2Check(device="cpu")
        seq_lens = torch.tensor([10, 20], dtype=torch.int32)
        window = i64(100, 101, 102, 200, 201, 202)
        check.compare(staged, seq_lens=seq_lens, positions=i64(10, 20), mrope=None, window=window, num_steps=2)
        self.assertEqual(check.mismatch.tolist(), [0, 0, 0, 0, 0])
        window[5] = 7  # past the draft steps' columns: only the window differs
        check.compare(staged, seq_lens=seq_lens, positions=i64(10, 20), mrope=None, window=window, num_steps=2)
        window[3] = 7  # a draft step's slot: both differ
        check.compare(staged, seq_lens=seq_lens, positions=i64(11, 20), mrope=None, window=window, num_steps=2)
        self.assertEqual(check.mismatch.tolist(), [0, 1, 0, 1, 2])
        self.assertEqual(check.check_ct, 3)

    def test_the_switch_reserves_three_windows(self):
        with (
            get_context().override_server_args(
                speculative_algorithm="EAGLE",
                speculative_num_steps=6,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=7,
                page_size=1,
            ),
            get_parallel().override(attn_dcp_size=1),
        ):
            window = get_alloc_len_per_decode()
            self.assertEqual(get_alloc_reserve_per_decode(), 2 * window)
            with envs.SGLANG_PERFLL_STEP2.override(1):
                self.assertEqual(get_alloc_reserve_per_decode(), 3 * window)


if __name__ == "__main__":
    unittest.main()
