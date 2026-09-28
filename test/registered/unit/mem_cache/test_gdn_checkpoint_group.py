"""Checkpoint deferral must preserve live-state dependencies and publication."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache import gdn_factored_pool as native
from sglang.srt.mem_cache.gdn_prefill_checkpoint_graph import (
    CheckpointGroup, GroupBuffers, independent_slots,
)


class CheckpointGroupTest(unittest.TestCase):
    def test_aliasing_checkpoints_cannot_be_deferred(self):
        self.assertTrue(independent_slots([1, 2, -1], [3, 4], [5]))
        for slots in ([2, 3], [3, 3], [3, 5], []):
            self.assertFalse(independent_slots([1, 2], slots, [5]))

    def test_six_layer_groups_complete_before_forward_returns(self):
        graph = Mock()
        slots = torch.tensor([3])
        group = CheckpointGroup(NS(), graph, slots)
        for i in range(36):
            group.add(i, torch.full((1, 2, 16, 16), float(i)), slots,
                      eager=None, policy=())
            self.assertEqual(graph.run.call_count, (i + 1) // 6)
        self.assertEqual(group.states, [])
        self.assertEqual([c.args[1] for c in graph.run.call_args_list], [0, 6, 12, 18, 24, 30])
        with self.assertRaisesRegex(RuntimeError, "order"):
            group.add(38, None, slots, eager=None, policy=())

    def test_group_padding_cannot_republish_previous_batch_slots(self):
        cfg = native.FactoredGDNConfig(init_method="k31", factored_prefix=1)
        pool = NS(cfg=cfg, vbar=torch.ones(6, 2, 16), init_omega=lambda b: None)
        states = [torch.ones(8, 2, 16, 16) for _ in range(6)]
        buffers = GroupBuffers(pool, 0, states, torch.arange(8))
        buffers.bind([s[:3] * 2 for s in states], torch.tensor([21, 22, 23]))
        self.assertEqual(buffers.slots.tolist(), [21, 22, 23, -1, -1, -1, -1, -1])
        self.assertTrue(all(torch.count_nonzero(s[3:]) == 0 for s in buffers.states))

    def test_nonfinal_native_batched_commits_keep_the_original_path(self):
        group = CheckpointGroup(NS(), Mock(), torch.tensor([3]))
        self.assertFalse(group.accepts(35, NS(pending=[None] * 36, last_layer=35)))
        self.assertFalse(group.accepts(0, NS(pending=[None], last_layer=0)))
        split = CheckpointGroup(NS(), Mock(), torch.tensor([3]))
        self.assertTrue(split.accepts(0, NS(pending=[None], last_layer=0)))
        with self.assertRaisesRegex(RuntimeError, "split"):
            split.accepts(1, NS(pending=[None] * 2, last_layer=35))

    def test_tracking_keeps_normal_bucket_and_singleton_arithmetic(self):
        grouped = CheckpointGroup(NS(), Mock(), torch.tensor([3, 4, 5]), normal_batch=16)
        self.assertEqual(grouped.batch, 16)
        singleton = CheckpointGroup(NS(), Mock(), torch.tensor([3]), normal_batch=16)
        self.assertEqual(singleton.batch, 1)

    def test_normal_commit_finishes_before_checkpoint_is_queued(self):
        from sglang.srt.mem_cache.gdn_prefill_commit_graph import PrefillCommitGraph
        pool = native.FactoredGDNPool.__new__(native.FactoredGDNPool)
        pool.cfg = native.FactoredGDNConfig(init_method="k31")
        pool.layer_map = {0: 0}
        pool.vbar = torch.zeros(1, 2, 16)
        pool.prefill_factor_graph = None
        pool.dense_required = None
        pool.batch_prefill_final_copy = True
        dense, tracked, slots = NS(device="cuda"), object(), torch.tensor([8])
        order = []
        checkpoint = NS(add=lambda *a, **kw: order.append("checkpoint"), accepts=lambda *a: True)
        plan = NS(pending=[(dense, tracked)], last_layer=0, checkpoint_group=checkpoint)
        def replay(*args, **kwargs):
            self.assertIsNone(args[4])
            self.assertEqual(plan.pending, [(dense, None)])
            order.append("normal")
            return True
        with patch.dict("os.environ", SGLANG_GDN_PREFILL_COMMIT_GRAPH="1"), \
             patch.object(native, "k31_graph_safe", return_value=True), \
             patch.object(PrefillCommitGraph, "run", side_effect=replay):
            pool._commit_extend_group(0, plan, dense, tracked, slots, None, None)
        self.assertEqual(order, ["normal", "checkpoint"])
        self.assertEqual(plan.pending, [])

    def test_grouped_k31_reconstructs_same_states_as_each_layer(self):
        torch.set_num_threads(1)
        torch.manual_seed(1007)
        cfg = native.FactoredGDNConfig(init_method="k31", dtype=torch.float32)
        states = [torch.randn(1, 2, 16, 16) for _ in range(6)]
        vbar, omega = torch.randn(6, 2, 16), torch.randn(1, 2, 16, 16)
        combined = native.factorize_layers(states, vbar, cfg, omega=omega)
        for i, state in enumerate(states):
            separate = native.factorize_layers([state], vbar[i:i+1], cfg, omega=omega)[0]
            for a, b in zip(combined[i], separate):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
