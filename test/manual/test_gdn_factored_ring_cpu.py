"""K5-ring (docs/224): a 2-position initial ring grows on demand; captured ring pointers follow."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache import gdn_prefill_batch_graph as batch_graph
from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNConfig, FactoredGDNPool
from sglang.srt.model_executor import fullstack_policy

LAYERS, HEADS, WIDTH = 3, 2, 4


def pool(*, ring=2, limit=64):
    return FactoredGDNPool(
        size=64, max_running_requests=limit,
        cache_params=SimpleNamespace(shape=SimpleNamespace(temporal=(HEADS, WIDTH, WIDTH))),
        mamba_layer_ids=list(range(LAYERS)), device="cpu",
        cfg=FactoredGDNConfig(ring=ring, strict_chunk=1),
    )


def unfinished(p, slots):
    plan = p.plan_extend(torch.tensor(slots), [64] * len(slots), prompt_final=[False] * len(slots))
    for slot in slots:  # what store_factored publishes for a mandatory row
        p.stale[slot] = 0
        p.dense_required[slot] = 1
    p.dense_of[torch.tensor(slots)] = plan.ring_dst.to(torch.int32)
    return plan


def kernel_address(buffers, layer, ring_slot, head):
    """_store_factored_kernel with RING_INDIRECT: ring + (ring_slot * H + head) * V * K (fp32)."""
    base = int(buffers.ring_pointers[layer])
    return base + (ring_slot * HEADS + head) * WIDTH * WIDTH * 4


class FactoredRingTest(unittest.TestCase):
    def test_policy_default_is_unchanged_and_override_logs_once(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_GDN_FACTORED_RING", None)
            self.assertEqual(fullstack_policy.factored_ring_size(), 16)
        policy = fullstack_policy
        policy._factored_ring_logged = False
        with patch.dict(os.environ, {"SGLANG_GDN_FACTORED_RING": "2"}):
            with self.assertLogs(policy.__name__, "INFO") as logs:
                self.assertEqual(policy.factored_ring_size(), 2)
                self.assertEqual(policy.factored_ring_size(), 2)
            self.assertEqual(len([m for m in logs.output if "FACTORED_RING_ACTIVE ring=2" in m]), 1)
        with patch.dict(os.environ, {"SGLANG_GDN_FACTORED_RING": "16"}):
            self.assertEqual(policy.factored_ring_size(), 16)
        for bad in ("0", "-1", "two", ""):
            with patch.dict(os.environ, {"SGLANG_GDN_FACTORED_RING": bad}):
                with self.assertRaises(ValueError):
                    policy.factored_ring_size()
        self.assertEqual(FactoredGDNConfig.parse("r=16,m=16,ring=2,strict_chunk=1").ring, 2)

    def test_ring_two_grows_to_three_and_seventeen_with_live_batch_pointers(self):
        p = pool()
        self.assertEqual(tuple(p.dense_ring.shape[:2]), (LAYERS, 2))
        buffers = batch_graph.BatchBuffers(p, 32, None, include_tail=False)
        self.assertEqual(buffers.ring_pointers.numel(), LAYERS)  # one base per layer, not per position

        held = [1, 2]
        unfinished(p, held)  # fills the initial reservation, no growth
        self.assertEqual((len(p.ring_owner), p.ring_generation), (2, 0))
        for target, new in ((3, [3]), (17, list(range(4, 18)))):
            p.dense_ring.copy_(torch.arange(p.dense_ring.numel(), dtype=torch.float32).reshape(p.dense_ring.shape))
            exact = p.dense_ring.clone()
            held += new
            plan = unfinished(p, held)
            self.assertEqual(len(p.ring_owner), target)
            self.assertEqual(p.dense_ring.shape[1], target)
            self.assertTrue(torch.equal(p.dense_ring[:, :exact.shape[1]], exact))
            self.assertLessEqual(max(plan.ring_dst.tolist()), target - 1)
            buffers.bind(plan, [(torch.zeros(len(held), HEADS, WIDTH, WIDTH), None)] * LAYERS,
                         None, None, None)
            self.assertEqual(buffers.ring_generation, p.ring_generation)
            self.assertEqual(buffers.ring_pointers.tolist(), [row.data_ptr() for row in p.dense_ring])
            for layer in range(LAYERS):
                for ring_slot in plan.ring_dst.tolist():
                    for head in range(HEADS):
                        self.assertEqual(kernel_address(buffers, layer, ring_slot, head),
                                         p.dense_ring[layer, ring_slot, head].data_ptr())
        self.assertEqual(p.ring_generation, 2)
        self.assertEqual(p.stats["ring_grown"], 2)
        # The next chunk of the same seventeen prompts reuses every position.
        pointer = p.dense_ring.data_ptr()
        plan = unfinished(p, held)
        self.assertEqual(plan.n_ring_src, 17)
        self.assertEqual(p.dense_ring.data_ptr(), pointer)
        self.assertEqual(p.stats["ring_grown"], 2)

    def test_completed_rows_never_grow_a_two_position_ring(self):
        p = pool()
        q = p.plan_extend(torch.arange(1, 33), [64] * 32, prompt_final=[True] * 32)
        self.assertEqual(q.n_ring_miss, 30)
        self.assertEqual((p.ring_generation, p.stats.get("ring_grown", 0)), (0, 0))

    def test_one_chunked_prompt_plus_completions_fits_without_growth(self):
        p = pool()
        unfinished(p, [1])
        for step in range(5):  # chunk k of slot 1 alongside newly completing prompts
            slots = [1] + list(range(10 + 4 * step, 14 + 4 * step))
            q = p.plan_extend(torch.tensor(slots), [64] * 5, prompt_final=[False] + [True] * 4)
            self.assertGreaterEqual(q.ring_dst[0].item(), 0)
            self.assertEqual(q.n_ring_src, 1)
        self.assertEqual(p.ring_generation, 0)

    def test_growth_respects_request_limit(self):
        p = pool(limit=3)
        unfinished(p, [1, 2, 3])
        with self.assertRaisesRegex(RuntimeError, "request capacity=3"):
            unfinished(p, [1, 2, 3, 4])
        self.assertEqual(p.stats.get("ring_grown", 0), 1)


if __name__ == "__main__":
    unittest.main()
