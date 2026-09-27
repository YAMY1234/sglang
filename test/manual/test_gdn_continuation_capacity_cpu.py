"""Real host-planner regressions for high-hit batches and exact continuations."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNConfig, FactoredGDNPool


def pool(*, ring=16, limit=64, strict=1):
    return FactoredGDNPool(
        size=64, max_running_requests=limit,
        cache_params=SimpleNamespace(shape=SimpleNamespace(temporal=(1, 2, 2))),
        mamba_layer_ids=[0, 1], device="cpu",
        cfg=FactoredGDNConfig(ring=ring, strict_chunk=strict),
    )


def plan(p, slots, lengths, final):
    return p.plan_extend(torch.tensor(slots), lengths, prompt_final=final)


def own(p, slots):
    for position, slot in enumerate(slots):
        p.ring_owner[position] = slot
        p.dense_of[slot] = position
        p.stale[slot] = 0
        p.dense_required[slot] = 1
    p.dense_ring.copy_(torch.arange(p.dense_ring.numel()).reshape(p.dense_ring.shape))


class ContinuationCapacityTest(unittest.TestCase):
    def test_real_twenty_fragment_failure_needs_no_growth_after_priority_fix(self):
        for unfinished_first in (False, True):
            p = pool()
            lengths = [410] * 19 + [402]
            final = [True] * 19 + [False]
            if unfinished_first:
                lengths.reverse(); final.reverse()
            before = p.dense_ring.data_ptr()
            q = plan(p, list(range(1, 21)), lengths, final)
            self.assertGreaterEqual(q.ring_dst[final.index(False)].item(), 0)
            self.assertEqual(len(p.ring_owner), 16)
            self.assertEqual(p.dense_ring.data_ptr(), before)

    def test_seventeen_unfinished_rows_grow_at_real_entry_with_identical_states(self):
        p = pool(); own(p, list(range(1, 17)))
        exact = p.dense_ring.clone()
        factors = [x.clone() for x in (p.a, p.U, p.W, p.count)]
        before_bytes = p.mem_usage_bytes()
        q = plan(p, list(range(1, 18)), [64] * 17, [False] * 17)
        self.assertEqual(len(p.ring_owner), 17)
        self.assertEqual(p.ring_generation, 1)
        self.assertEqual(q.n_ring_src, 16)
        self.assertEqual(q.ring_dst.tolist(), list(range(17)))
        self.assertTrue(torch.equal(p.dense_ring[:, :16], exact))
        self.assertEqual(torch.count_nonzero(p.dense_ring[:, 16]).item(), 0)
        self.assertEqual(p.mem_usage_bytes() - before_bytes, exact[:, 0].nbytes)
        for before, after in zip(factors, (p.a, p.U, p.W, p.count)):
            self.assertTrue(torch.equal(before, after))
        # No allocation and no source displacement on the next chunk.
        p.stale[17] = 0; p.dense_required[17] = 1
        pointer = p.dense_ring.data_ptr()
        q = plan(p, list(range(1, 18)), [64] * 17, [False] * 17)
        self.assertEqual(q.n_ring_src, 17)
        self.assertEqual(p.dense_ring.data_ptr(), pointer)
        self.assertEqual(p.ring_generation, 1)

    def test_external_unfinished_states_are_not_evicted_to_fit_new_batch(self):
        p = pool(); own(p, list(range(1, 17)))
        before = p.dense_ring.clone()
        q = plan(p, [17, 18], [64, 64], [False, False])
        self.assertEqual(q.ring_dst.tolist(), [16, 17])
        self.assertEqual(p.ring_owner[:16], list(range(1, 17)))
        self.assertTrue(torch.equal(p.dense_ring[:, :16], before))

    def test_completed_rows_do_not_trigger_growth(self):
        p = pool()
        q = plan(p, list(range(1, 33)), [64] * 32, [True] * 32)
        self.assertEqual(q.n_ring_miss, 16)
        self.assertEqual(p.ring_generation, 0)

    def test_limit_is_request_capacity_not_state_checkpoint_slots(self):
        p = pool(limit=17); own(p, list(range(1, 17)))
        before = p.dense_ring
        with self.assertRaisesRegex(RuntimeError, "request capacity=17"):
            plan(p, [17, 18], [64, 64], [False, False])
        self.assertIs(p.dense_ring, before)
        self.assertEqual(p.ring_owner, list(range(1, 17)))
        self.assertEqual(p.dense_of[17:19].tolist(), [-1, -1])

    def test_allocation_failure_does_not_publish_owners_or_indices(self):
        p = pool(); own(p, list(range(1, 17)))
        before = p.dense_ring
        with patch.object(torch.Tensor, "new_zeros", side_effect=RuntimeError("allocation failed")):
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                plan(p, [17], [64], [False])
        self.assertIs(p.dense_ring, before)
        self.assertEqual(p.ring_owner, list(range(1, 17)))
        self.assertEqual(p.dense_of[17].item(), -1)
        self.assertEqual(p.ring_generation, 0)

    def test_non_strict_path_keeps_original_optional_policy(self):
        p = pool(strict=0)
        q = plan(p, list(range(1, 18)), [64] * 17, None)
        self.assertEqual(q.n_ring_miss, 1)
        self.assertEqual(p.ring_generation, 0)

    def test_padding_and_layer_range_survive_retry(self):
        p = pool(); own(p, list(range(1, 17)))
        q = p.plan_extend(torch.tensor([17, 18, -1]), [64, 64],
                          prefix_lens=[0, 0], prompt_final=[False, False], layer_range=(1, 1))
        self.assertEqual(q.ring_dst.tolist(), [16, 17, -1])
        self.assertEqual((q.next_layer, q.last_layer), (1, 1))
        self.assertEqual(q.dense_required_after_commit.tolist(), [1, 1, 0])


if __name__ == "__main__":
    unittest.main()
