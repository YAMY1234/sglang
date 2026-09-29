"""Both serving commit paths preserve independent states and fixed directions."""
from contextlib import ExitStack, nullcontext
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache import gdn_factored_pool as native
from sglang.srt.mem_cache import gdn_prefill_batch_graph as batch_module
from sglang.srt.mem_cache import gdn_prefill_commit_graph as commit_module
from sglang.srt.mem_cache import gdn_prefill_joint as joint
from test_gdn_prefill_batch_graph import fake_pool, make_plan


class JointFactorizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_directions_and_two_distinct_states_match_separate_factorization(self):
        torch.manual_seed(1863)
        for layers in (1, 36):
            pool = fake_pool(layers=layers, width=16)
            normal = [torch.randn(1, 2, 16, 16) for _ in range(layers)]
            tracked = [torch.randn(1, 2, 16, 16).bfloat16().float() for _ in range(layers)]
            omega = torch.randn(1, 2, 16, 16)
            track_omega = torch.randn_like(omega)  # Deliberately different seeds.
            inputs = joint.JointInputs(normal, tracked, omega, track_omega)
            for dst, src in zip(inputs.normal + inputs.tracked, normal + tracked):
                dst.copy_(src)
            torch.testing.assert_close(inputs.omega[:1], omega, atol=0, rtol=0)
            torch.testing.assert_close(inputs.omega[1:], track_omega, atol=0, rtol=0)
            actual = inputs.evaluate(native.factorize_layers, pool.vbar, pool.cfg)
            for states, directions, values in zip((normal, tracked), (omega, track_omega), actual):
                expected = native.factorize_layers(states, pool.vbar, pool.cfg, omega=directions)
                for i, (got, ref) in enumerate(zip(values, expected)):
                    # Eigenvector signs are not canonical under a changed batch
                    # shape. U and W may flip together; compare the represented
                    # state and its application, not arbitrary basis entries.
                    torch.testing.assert_close(got[0], ref[0], atol=2e-5, rtol=2e-5)
                    count = torch.full((1, 2), pool.cfg.r)
                    torch.testing.assert_close(native.densify(*got, count, pool.vbar[i]),
                                               native.densify(*ref, count, pool.vbar[i]),
                                               atol=2e-5, rtol=2e-5)
                    query = torch.randn(1, 2, 16, 1)
                    torch.testing.assert_close(native.densify(*got, count, pool.vbar[i]) @ query,
                                               native.densify(*ref, count, pool.vbar[i]) @ query,
                                               atol=2e-5, rtol=2e-5)

    def test_batch_and_layer_buffers_rebind_both_states_without_seed_or_slot_alias(self):
        pool = fake_pool(layers=6)
        plan = make_plan(pool)
        for joined in (False, True):
            buffers = batch_module.BatchBuffers(pool, 1, 1, include_tail=False,
                                                join_branches=joined)
            layer = commit_module.CommitBuffers(pool, 0, plan, torch.zeros(1, 2, 16, 16),
                torch.zeros(1, 2, 16, 16), torch.tensor([20]), join_branches=joined)
            first_ptr = buffers.normal[0].data_ptr()
            for normal_value, track_value in ((3., 7.), (-2., 11.)):
                normal = torch.full((1, 2, 16, 16), normal_value)
                tracked = torch.full_like(normal, track_value).bfloat16()
                buffers.bind(plan, [(normal, tracked)] * 6, torch.tensor([20]), None, None)
                layer.bind(plan, normal, tracked, torch.tensor([20]))
                self.assertEqual(buffers.normal[0].data_ptr(), first_ptr)
                for aa, bb in ((buffers.normal[0], buffers.tracked[0]), (layer.dense, layer.track_dense)):
                    self.assertTrue(torch.equal(aa, normal))
                    self.assertTrue(torch.equal(bb, tracked.float()))
                    self.assertNotEqual(aa.data_ptr(), bb.data_ptr())
                self.assertEqual(buffers.slots.tolist(), [1])
                self.assertEqual(buffers.track_slots.tolist(), [20])
            other = commit_module.CommitBuffers(pool, 1, plan, normal, tracked,
                torch.tensor([21]), shared=layer, join_branches=joined)
            self.assertIs(other.joint, layer.joint)
            self.assertNotEqual(other.vbar.data_ptr(), layer.vbar.data_ptr())
            self.assertEqual(other.li, 1)

    def test_only_singleton_checkpoint_bucket_changes_and_marker_is_runtime_selection(self):
        cfg = native.FactoredGDNConfig(init_method='k31')
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {joint.FLAG: '1'}):
            marker = Path(tmp)/'enabled'
            with patch.dict(os.environ, {joint.FLAG+'_FILE': str(marker)}):
                self.assertFalse(joint.enabled())
                marker.touch(); self.assertTrue(joint.enabled())
                marker.unlink(); self.assertFalse(joint.enabled())
            for normal, tracked in commit_module.prewarm_shapes():
                self.assertEqual(joint.modes(cfg, normal, tracked),
                                 (False, True) if normal == tracked == 1 else (False,))
        self.assertFalse(joint.eligible(native.FactoredGDNConfig(init_method='iter'), 1, 1))

    def test_both_graphs_prewarm_old_and_joint_before_any_marker_switch(self):
        pool = fake_pool(layers=6)
        pool.device = 'cpu'
        batch_graph = batch_module.PrefillBatchGraph(include_tail=False)
        layer_graph = commit_module.PrefillCommitGraph()
        graph, stream = Mock(), Mock()
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {joint.FLAG: '1'}))
            stack.enter_context(patch.object(torch.Tensor, 'is_cuda', property(lambda _: True)))
            stack.enter_context(patch.object(commit_module, 'k31_graph_safe', return_value=True))
            stack.enter_context(patch.object(batch_module.BatchBuffers, 'evaluate'))
            stack.enter_context(patch.object(commit_module.CommitBuffers, 'evaluate'))
            for name, value in (('is_current_stream_capturing', False), ('current_stream', stream),
                ('Stream', stream), ('CUDAGraph', graph), ('memory_allocated', 0),
                ('memory_reserved', 0), ('synchronize', None), ('graph_pool_handle', object())):
                stack.enter_context(patch.object(torch.cuda, name, return_value=value))
            for name in ('stream', 'graph'):
                stack.enter_context(patch.object(torch.cuda, name, side_effect=lambda *a, **kw: nullcontext()))
            for cache in (batch_graph, layer_graph):
                cache.prewarm(pool, eager=native.factorize_layers, policy=())
            self.assertEqual(batch_graph.stats['captured'], 19)
            self.assertEqual(layer_graph.stats['captured'], 19*6)
            self.assertEqual(sum(key[-1] for key in batch_graph.entries), 1)
            self.assertEqual(sum(key[-1] for key in layer_graph.entries), 6)
            normal = torch.ones(1, 2, 16, 16); track = 2*normal
            plan = make_plan(pool); plan.pending = [(normal, track)]
            for mode in (False, True, False, True):
                batch_graph.run(pool, plan, [(normal, track)]*6, torch.tensor([20]), None, None,
                    eager=native.factorize_layers, policy=(), join_branches=mode)
                for lid in pool.layer_ids:
                    self.assertTrue(layer_graph.run(pool, lid, plan, normal, track, torch.tensor([20]),
                        eager=native.factorize_layers, policy=(), join_branches=mode))
            self.assertEqual(batch_graph.stats['captured'], 19)
            self.assertEqual(layer_graph.stats['captured'], 114)
            self.assertEqual(batch_graph.stats['joint_replayed'], 3)
            self.assertEqual(layer_graph.stats['joint_replayed'], 18)


if __name__ == '__main__': unittest.main()
