"""Unit tests for SGLANG_PERFLL_STEP2's counting mode and its three-window decode reserve."""

import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_len_per_decode,
    get_alloc_reserve_per_decode,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.perfll_step2 import Step2Counter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch(slots, allocated, committed=100):
    reqs = [
        SimpleNamespace(
            req_pool_idx=slot,
            kv=SimpleNamespace(kv_allocated_len=alloc, kv_committed_len=committed),
        )
        for slot, alloc in zip(slots, allocated)
    ]
    return SimpleNamespace(reqs=reqs)


class TestPerfllStep2(CustomTestCase):
    def _pass(self, counter, batch):
        counter.count_draft(batch)
        counter.note_extend(batch)

    def test_a_decode_pass_is_staged_only_after_the_same_unpadded_allocated_batch(self):
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
