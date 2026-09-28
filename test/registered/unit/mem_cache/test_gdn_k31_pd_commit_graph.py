"""PD k31 commits must reach the graph, with an eager torch reference override."""
import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.linear.kernels import gdn_prefill_reference as ref
from sglang.srt.mem_cache import gdn_factored_pool as pool_module
from sglang.srt.mem_cache import gdn_prefill_commit_graph as graph_module


class K31PDCommitGraphTest(unittest.TestCase):
    def test_pd_commit_dispatches_capturable_k31(self):
        pool = pool_module.FactoredGDNPool.__new__(pool_module.FactoredGDNPool)
        pool.cfg = pool_module.FactoredGDNConfig(init_method="k31")
        pool.layer_map = {0: 0}
        pool.vbar = torch.zeros(1, 2, 16)
        pool.prefill_factor_graph = None
        pool.dense_required = None
        pool.batch_prefill_final_copy = True
        dense = NS(device="cuda")
        plan = NS(pending=[(dense, None)], last_layer=0)
        with patch.dict(os.environ, SGLANG_GDN_PREFILL_COMMIT_GRAPH="1"), \
             patch.object(ref, "K31_EIGH", "auto"), \
             patch.object(graph_module.PrefillCommitGraph, "run", return_value=True) as run, \
             patch.object(pool_module, "factorize_layers", side_effect=AssertionError("unexpected eager commit")):
            pool._commit_extend_group(0, plan, dense, None, None, None, None)
        run.assert_called_once()
        self.assertEqual(plan.pending, [])

    def test_graph_replays_live_controls_and_torch_falls_back(self):
        cfg = pool_module.FactoredGDNConfig(init_method="k31", factored_prefix=1)
        pool = NS(cfg=cfg, prefix_dense=None, layer_ids=[0])
        dense = Mock(is_cuda=True, device=torch.device("cuda"), dtype=torch.float32,
                     shape=(1, 24, 128, 128))
        dense.stride.return_value = (393216, 16384, 128, 1)
        plan = NS(pending=[(dense, None)], slots=torch.tensor([1]), ring_dst=torch.tensor([0]))
        graph = graph_module.PrefillCommitGraph()
        buffers, cuda_graph, stream = Mock(), Mock(), Mock()
        eager = Mock()
        with patch.object(ref, "K31_EIGH", "auto"), \
             patch.object(torch.cuda, "is_current_stream_capturing", return_value=False), \
             patch.object(graph_module, "CommitBuffers", return_value=buffers), \
             patch.object(torch.cuda, "current_stream", return_value=stream), \
             patch.object(torch.cuda, "Stream", return_value=stream), \
             patch.object(torch.cuda, "stream", side_effect=lambda _: nullcontext()), \
             patch.object(torch.cuda, "CUDAGraph", return_value=cuda_graph), \
             patch.object(torch.cuda, "graph", side_effect=lambda *a, **kw: nullcontext()):
            self.assertTrue(graph.run(pool, 0, plan, dense, None, None, eager=eager, policy=()))
            plan.slots = torch.tensor([7])
            self.assertTrue(graph.run(pool, 0, plan, dense, None, None, eager=eager, policy=()))
            buffers.bind.assert_called_once_with(plan, dense, None, None)
            self.assertEqual(cuda_graph.replay.call_count, 2)
            with patch.object(ref, "K31_EIGH", "torch"):
                self.assertFalse(graph.run(pool, 0, plan, dense, None, None, eager=eager, policy=()))
            self.assertEqual(cuda_graph.replay.call_count, 2)
        self.assertEqual(graph.stats, dict(captured=1, replayed=2, fallback=1))

    def test_batched_bucket_rebind_clears_previous_rows(self):
        cfg = pool_module.FactoredGDNConfig(init_method="k31", factored_prefix=1)
        pool = NS(cfg=cfg, layer_map={0: 0}, vbar=torch.zeros(1, 2, 16),
                  init_omega=lambda b: None)
        dense = torch.ones(8, 2, 16, 16)
        plan = NS(slots=torch.arange(1, 9), ring_dst=torch.arange(8))
        buffers = graph_module.CommitBuffers(pool, 0, plan, dense, dense, plan.slots)
        plan.slots, plan.ring_dst = torch.arange(11, 17), torch.arange(6)
        buffers.bind(plan, dense[:6] * 2, dense[:3] * 3, torch.arange(21, 24))
        self.assertEqual(buffers.slots.tolist(), [11, 12, 13, 14, 15, 16, -1, -1])
        self.assertEqual(buffers.track_slots.tolist(), [21, 22, 23, -1, -1, -1, -1, -1])
        self.assertEqual(buffers.ring_dst.tolist(), [0, 1, 2, 3, 4, 5, -1, -1])
        self.assertTrue(torch.equal(buffers.dense[6:], torch.zeros_like(dense[6:])))
        self.assertTrue(torch.equal(buffers.track_dense[3:], torch.zeros_like(dense[3:])))
        self.assertEqual(graph_module.batch_bucket(dense[:3], None), 4)
        self.assertEqual(graph_module.batch_bucket(dense[:6], None), 8)
        self.assertIsNone(graph_module.batch_bucket(torch.zeros(17, 2, 16, 16), None))

    def test_batched_graph_reuses_bucket_and_keeps_shapes_bounded(self):
        cfg = pool_module.FactoredGDNConfig(init_method="k31", factored_prefix=1)
        pool = NS(cfg=cfg, prefix_dense=None, layer_ids=[0])
        plan = NS(pending=[None], slots=torch.arange(4), ring_dst=torch.arange(4))
        graph = graph_module.PrefillCommitGraph()
        buffers, cuda_graph, stream, eager = Mock(), Mock(), Mock(), Mock()
        with patch.object(ref, "K31_EIGH", "auto"), \
             patch.object(torch.cuda, "is_current_stream_capturing", return_value=False), \
             patch.object(graph_module, "CommitBuffers", return_value=buffers), \
             patch.object(torch.cuda, "current_stream", return_value=stream), \
             patch.object(torch.cuda, "Stream", return_value=stream), \
             patch.object(torch.cuda, "stream", side_effect=lambda _: nullcontext()), \
             patch.object(torch.cuda, "CUDAGraph", return_value=cuda_graph), \
             patch.object(torch.cuda, "graph", side_effect=lambda *a, **kw: nullcontext()):
            for b in (4, 3, 8, 6):
                dense = NS(is_cuda=True, device=torch.device("cuda"), dtype=torch.float32,
                           shape=(b, 24, 128, 128))
                plan.slots = plan.ring_dst = torch.arange(b)
                self.assertTrue(graph.run(pool, 0, plan, dense, None, None, eager=eager, policy=()))
            self.assertEqual(graph.stats, dict(captured=2, replayed=4, fallback=0))
            dense.shape = (17, 24, 128, 128)
            self.assertFalse(graph.run(pool, 0, plan, dense, None, None, eager=eager, policy=()))


if __name__ == "__main__":
    unittest.main()
