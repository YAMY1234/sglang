"""Startup captures every reachable commit shape without publishing live slots."""
import os
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache import gdn_factored_pool as pool_module
from sglang.srt.mem_cache import gdn_prefill_commit_graph as graph_module


class CommitPrewarmTest(unittest.TestCase):
    def test_startup_captures_all_layers_and_singleton_variants_once(self):
        cfg = pool_module.FactoredGDNConfig(init_method="k31", factored_prefix=1)
        pool = NS(cfg=cfg, prefix_dense=None, layer_ids=[i for i in range(48) if i % 4 != 3],
                  device="cpu", hv=2, v=16, k=16)
        graph = graph_module.PrefillCommitGraph()
        buffers, cuda_graph, stream, eager = Mock(), Mock(), Mock(), Mock()
        with ExitStack() as stack:
            stack.enter_context(patch.object(torch.Tensor, "is_cuda", property(lambda _: True)))
            stack.enter_context(patch.object(graph_module, "k31_graph_safe", return_value=True))
            ctor = stack.enter_context(patch.object(graph_module, "CommitBuffers", return_value=buffers))
            for name, value in (("is_current_stream_capturing", False),
                                ("current_stream", stream), ("Stream", stream),
                                ("CUDAGraph", cuda_graph), ("memory_allocated", 0),
                                ("synchronize", None)):
                stack.enter_context(patch.object(torch.cuda, name, return_value=value))
            for name in ("stream", "graph"):
                stack.enter_context(patch.object(torch.cuda, name,
                                                side_effect=lambda *a, **kw: nullcontext()))
            graph.prewarm(pool, eager=eager, policy=())
            signatures = {(key[0], key[1][0][0][0],
                           None if key[1][3] is None else key[1][3][0][0])
                          for key in graph.entries}
            expected = set()
            for lid in pool.layer_ids:
                for batch in (1, 2, 4, 8, 16):
                    expected.update(((lid, batch, None), (lid, batch, batch)))
                    if batch > 1:
                        expected.update(((lid, 1, batch), (lid, batch, 1)))
            self.assertEqual(signatures, expected)
            self.assertEqual(len(signatures), 648)
            self.assertIn((46, 16, 16), signatures)
            self.assertEqual(graph.stats["captured"], 648)
            self.assertEqual(len(graph.shared_buffers), 18)
            graph.prewarm(pool, eager=eager, policy=())
            self.assertEqual(graph.stats["captured"], 648)
            for call in ctor.call_args_list:
                plan, track_slots = call.args[2], call.args[5]
                self.assertTrue(torch.all(plan.slots == -1))
                self.assertTrue(torch.all(plan.ring_dst == -1))
                if track_slots is not None:
                    self.assertTrue(torch.all(track_slots == -1))

    def test_shared_inputs_keep_layer_specific_publication_targets(self):
        cfg = pool_module.FactoredGDNConfig(init_method="k31", factored_prefix=1)
        pool = NS(cfg=cfg, layer_map={0: 0, 3: 1}, vbar=torch.zeros(2, 2, 16),
                  dense_ring=torch.zeros(2, 16, 2, 16, 16),
                  init_omega=lambda b: None)
        dense = torch.ones(4, 2, 16, 16)
        plan = NS(slots=torch.arange(4), ring_dst=torch.arange(4))
        first = graph_module.CommitBuffers(pool, 0, plan, dense, dense, plan.slots)
        second = graph_module.CommitBuffers(pool, 3, plan, dense, dense, plan.slots,
                                           shared=first)
        self.assertIs(first.dense, second.dense)
        self.assertIs(first.track_slots, second.track_slots)
        self.assertIs(first.omega, second.omega)
        self.assertNotEqual(first.vbar.data_ptr(), second.vbar.data_ptr())
        self.assertEqual((first.li, second.li), (0, 1))

    def test_ring_growth_updates_captured_pointer_without_new_graph(self):
        cfg = pool_module.FactoredGDNConfig(init_method="k31", factored_prefix=1)
        pool = NS(cfg=cfg, layer_map={3: 1}, vbar=torch.zeros(2, 2, 16),
                  dense_ring=torch.zeros(2, 2, 2, 16, 16), ring_generation=0,
                  init_omega=lambda b: None)
        dense = torch.ones(1, 2, 16, 16)
        plan = NS(slots=torch.tensor([1]), ring_dst=torch.tensor([0]))
        buffers = graph_module.CommitBuffers(pool, 3, plan, dense, None, None)
        pointer_control = buffers.ring_pointer.data_ptr()
        old_address = buffers.ring_pointer.item()
        pool.dense_ring = torch.zeros(2, 4, 2, 16, 16)
        pool.ring_generation += 1
        plan.ring_dst = torch.tensor([3])
        buffers.bind(plan, dense, None, None)
        self.assertEqual(buffers.ring_pointer.data_ptr(), pointer_control)
        self.assertNotEqual(buffers.ring_pointer.item(), old_address)
        self.assertEqual(buffers.ring_pointer.item(), pool.dense_ring[1].data_ptr())
        self.assertEqual(buffers.ring_dst.item(), 3)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_captured_store_replays_into_grown_ring(self):
        cfg = pool_module.FactoredGDNConfig(init_method="k31", factored_prefix=1)
        pool = NS(cfg=cfg, layer_map={0: 0}, vbar=torch.zeros(1, 2, 16, device="cuda"),
                  dense_ring=torch.zeros(1, 2, 2, 16, 16, device="cuda"), ring_generation=0,
                  init_omega=lambda b: None, prefix_valid=None,
                  a=torch.zeros(1, 4, 2, 16, device="cuda"),
                  U=torch.zeros(1, 4, 2, 16, 16, device="cuda"),
                  W=torch.zeros(1, 4, 2, 16, 16, device="cuda"),
                  count=torch.zeros(1, 4, 2, dtype=torch.int32, device="cuda"),
                  stale=torch.zeros(4, dtype=torch.int32, device="cuda"),
                  dense_of=torch.zeros(4, dtype=torch.int32, device="cuda"))
        dense = torch.ones(1, 2, 16, 16, device="cuda")
        plan = NS(slots=torch.tensor([1], device="cuda"), ring_dst=torch.tensor([0], device="cuda"))
        buffers = graph_module.CommitBuffers(pool, 0, plan, dense, None, None)
        factors = (torch.ones(1, 2, 16, device="cuda"),
                   torch.ones(1, 2, 16, 16, device="cuda"),
                   torch.ones(1, 2, 16, 16, device="cuda"))
        eager = lambda *a, **kw: [factors]
        buffers.evaluate(eager)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            buffers.evaluate(eager)
        old_ring, old_values = pool.dense_ring, pool.dense_ring.clone()
        pool.dense_ring = torch.zeros(1, 4, 2, 16, 16, device="cuda")
        pool.ring_generation += 1
        plan.ring_dst.fill_(3)
        dense.fill_(7)
        buffers.bind(plan, dense, None, None)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(pool.dense_ring[0, 3], dense[0])
        torch.testing.assert_close(old_ring, old_values)

    def test_pool_uses_runtime_eager_identity_and_skips_decode(self):
        pool = pool_module.FactoredGDNPool.__new__(pool_module.FactoredGDNPool)
        pool.cfg = pool_module.FactoredGDNConfig(init_method="k31", factored_prefix=1)
        pool.prefix_dense, pool.device, pool.a = None, "cuda", NS(is_cuda=True)
        graph = Mock()
        with patch.dict(os.environ, SGLANG_GDN_PREFILL_COMMIT_GRAPH="1",
                        SGLANG_GDN_PREFILL_CHECKPOINT_GRAPH="0"), \
             patch.object(pool_module, "k31_graph_safe", return_value=True), \
             patch.object(graph_module, "PrefillCommitGraph", return_value=graph):
            pool.prewarm_commit_graph()
            graph.prewarm.assert_called_once_with(
                pool, eager=pool_module.factorize_layers,
                policy=(pool_module.ORTH_METHOD, pool_module.ORTH_WARPS_OVERRIDE,
                        pool_module.factorize_dense))
            pool.cfg.factored_prefix = 0
            pool.prewarm_commit_graph()
            self.assertEqual(graph.prewarm.call_count, 1)

    def test_pool_prewarm_includes_enabled_checkpoint_groups(self):
        from sglang.srt.mem_cache import gdn_prefill_checkpoint_graph as checkpoints

        pool = pool_module.FactoredGDNPool.__new__(pool_module.FactoredGDNPool)
        pool.cfg = pool_module.FactoredGDNConfig(init_method="k31", factored_prefix=1)
        pool.prefix_dense, pool.device, pool.a = None, "cuda", NS(is_cuda=True)
        graph, checkpoint_graph = Mock(), Mock()
        with patch.dict(os.environ, SGLANG_GDN_PREFILL_COMMIT_GRAPH="1",
                        SGLANG_GDN_PREFILL_CHECKPOINT_GRAPH="1"), \
             patch.object(pool_module, "k31_graph_safe", return_value=True), \
             patch.object(graph_module, "PrefillCommitGraph", return_value=graph), \
             patch.object(checkpoints, "CheckpointGraph", return_value=checkpoint_graph):
            pool.prewarm_commit_graph()
            checkpoint_graph.prewarm.assert_called_once_with(
                pool, eager=pool_module.factorize_layers,
                policy=(pool_module.ORTH_METHOD, pool_module.ORTH_WARPS_OVERRIDE,
                        pool_module.factorize_dense))


if __name__ == "__main__":
    unittest.main()
