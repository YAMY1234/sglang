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
from sglang.srt.speculative.perfll_step2 import (
    NextDraft,
    Step2Check,
    Step2Counter,
    fixed_draft_variant,
    takes_staging,
)
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
        self.assertEqual(check.mismatch_by_bs[2].tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(check.first.tolist()[0], -1)
        window[5] = 7  # past the draft steps' columns: only the window differs
        check.compare(staged, seq_lens=seq_lens, positions=i64(10, 20), mrope=None, window=window, num_steps=2)
        window[3] = 7  # a draft step's slot: both differ
        check.compare(staged, seq_lens=seq_lens, positions=i64(11, 20), mrope=None, window=window, num_steps=2)
        self.assertEqual(check.mismatch_by_bs[2].tolist(), [0, 1, 0, 1, 2])
        self.assertEqual(check.check_ct_by_bs[2], 3)
        # the first difference: the window's last slot in the second comparison
        self.assertEqual(check.first.tolist(), [4, 5, 2, 202, 7])
        self.assertIn("first difference window[5] at B2: staged 202, eager 7", check.summary())

    def test_a_draft_declines_a_staging_of_another_size_plan_or_mrope_choice(self):
        """Mode 3 takes, and mode 2 compares, only a staging the eager path would match: a batch
        with multimodal inputs computes M-RoPE rows with per-request offsets the staging lacks."""
        rows = torch.zeros(2, dtype=torch.int64)
        staged = NextDraft(
            num_requests=2,
            seq_lens=rows,
            positions=rows,
            mrope=torch.zeros(3, 2, dtype=torch.int64),
            draft_out_cache_loc=rows,
            window=rows,
        )
        take = dict(num_requests=2, plans_window=True, with_mrope=True)
        self.assertTrue(takes_staging(staged, **take))
        for field, value in (("num_requests", 3), ("plans_window", False), ("with_mrope", False)):
            self.assertFalse(takes_staging(staged, **{**take, field: value}), field)

    def test_a_draft_variant_is_fixed_a_round_early_only_when_one_window_cannot_cross_a_threshold(
        self,
    ):
        """SGLANG_PERFLL_STEP5B runs the next round's draft a round early: its graph variant must be
        the one the next round would pick, whose host length may be up to one window longer."""

        def select(fb):  # the plugin's Q1Variants shape: by length at batch size 1 only
            if fb.batch_size != 1 or fb.attention_kv_len_lower_bound is None:
                return "stock"
            return "long" if fb.attention_kv_len_lower_bound >= 16384 else "stock"

        def fixed(bs, length):
            return fixed_draft_variant(select, batch_size=bs, committed_len=length, window=7)

        self.assertEqual(fixed(1, 16000), (True, "stock"))
        self.assertEqual(fixed(1, 16376), (True, "stock"))  # 16376 + 7 stays below the threshold
        self.assertEqual(fixed(1, 16377), (False, None))  # 16377 + 7 reaches it
        self.assertEqual(fixed(1, 16384), (True, "long"))
        self.assertEqual(fixed(4, 16380), (True, "stock"))  # lengths do not pick at B > 1
        self.assertEqual(fixed_draft_variant(None, batch_size=1, committed_len=1, window=7), (True, None))

    def test_a_draft_run_a_round_early_is_taken_only_by_the_same_staged_batch(self):
        """SGLANG_PERFLL_STEP5B: the next pass takes the early draft only when it is staged and
        its request slots are the ones the draft ran for; a changed or padded batch drops it."""
        counter = Step2Counter(width=7, graph_sizes=[1, 2, 4], needs_next_draft=True)
        whole = 100 + 21
        staged = object()

        def run_pass(slots, allocated, next_slots=None, tail=True):
            batch = _batch(slots, allocated)
            counter.count_draft(batch)
            taken = counter.tail_taken
            if tail and counter.can_stage(batch):
                counter.note_tail(batch, variant="stock", outputs=("tail", tuple(slots)))
            counter.note_extend(batch, next_draft=staged)
            return taken

        self.assertIsNone(run_pass([5, 9], [whole, whole]))  # no previous pass
        self.assertEqual(run_pass([5, 9], [whole, whole]).outputs, ("tail", (5, 9)))
        self.assertIsNone(run_pass([9, 5], [whole, whole]))  # order changed: dropped
        self.assertIsNone(run_pass([9, 5, 3], [whole] * 3))  # batch changed; 3 pads to 4: no tail
        self.assertIsNone(run_pass([9, 5, 3], [whole] * 3))  # padded: not staged, no tail to take
        self.assertEqual(run_pass([9, 5, 3, 1], [whole] * 4, tail=True), None)  # batch changed
        self.assertEqual(run_pass([9, 5, 3, 1], [whole] * 4).outputs, ("tail", (9, 5, 3, 1)))
        counter.forget()
        self.assertIsNone(run_pass([9, 5, 3, 1], [whole] * 4))  # a prefill pass in between
        self.assertEqual(counter.counts["tail_taken"], 2)
        self.assertEqual(counter.counts["tail_dropped"], 2)

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
