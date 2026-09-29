"""Checkpoint deferral must preserve live-state dependencies and publication."""
import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache import gdn_factored_pool as native
from sglang.srt.mem_cache.gdn_prefill_checkpoint_graph import (
    CheckpointGraph, CheckpointGroup, GroupBuffers, independent_slots, prepare,
)


class CheckpointGroupTest(unittest.TestCase):
    def test_full_depth_adapter_can_defer_its_full_batch_plan(self):
        with patch.dict("os.environ", SGLANG_GDN_PREFILL_CHECKPOINT_GRAPH="1"):
            prepare(NS(), NS(factored_extend=None))

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

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_group_replay_matches_layerwise_k31_and_publication(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_factored_io import store_factored

        layers, heads, width, capacity = 12, 24, 128, 48
        cfg = native.FactoredGDNConfig(init_method="k31", dtype=torch.float16, factored_prefix=1)
        generator = torch.Generator(device="cuda").manual_seed(1007)
        vbar = torch.randn(layers, heads, width, device="cuda", generator=generator)
        vbar = torch.nn.functional.normalize(vbar, dim=-1)
        omega = torch.randn(1, heads, width, cfg.r + 8, device="cuda", generator=generator)
        initial = dict(a=0.375, U=-0.5, W=0.25, count=3, stale=0, dense_of=7, prefix_valid=0)

        def make_pool():
            return NS(cfg=cfg, vbar=vbar, init_omega=lambda b: omega.expand(b, -1, -1, -1),
                      prefix_layer_count=lambda: layers,
                      a=torch.empty(layers, capacity, heads, width, device="cuda"),
                      U=torch.empty(layers, capacity, heads, cfg.rmax, width, device="cuda", dtype=cfg.dtype),
                      W=torch.empty(layers, capacity, heads, cfg.rmax, width, device="cuda", dtype=cfg.dtype),
                      count=torch.empty(layers, capacity, heads, device="cuda", dtype=torch.int32),
                      stale=torch.empty(capacity, device="cuda", dtype=torch.int32),
                      dense_of=torch.empty(capacity, device="cuda", dtype=torch.int32),
                      prefix_valid=torch.empty(capacity, device="cuda", dtype=torch.int32))

        candidate, reference = make_pool(), make_pool()
        graph = CheckpointGraph()
        policy = (native.ORTH_METHOD, native.ORTH_WARPS_OVERRIDE, native.factorize_dense)
        for bucket in (1, 8, 16):
            for repetition, rows in enumerate((bucket, {1: 1, 8: 3, 16: 5}[bucket])):
                for pool in (candidate, reference):
                    for name, fill in initial.items():
                        getattr(pool, name).fill_(fill)
                start = 2 if repetition == 0 else 24
                slots = torch.arange(start, start + rows, device="cuda", dtype=torch.long)
                padded_slots = torch.full((bucket,), -1, device="cuda", dtype=torch.long)
                padded_slots[:rows].copy_(slots)
                states = [torch.randn(rows, heads, width, width, device="cuda", generator=generator)
                          for _ in range(layers)]
                for layer, state in enumerate(states):
                    padded = state.new_zeros((bucket, heads, width, width))
                    padded[:rows].copy_(state)
                    values = native.factorize_layers([padded], vbar[layer:layer + 1], cfg,
                                                     omega=reference.init_omega(bucket))[0]
                    store_factored(*values, reference.a[layer], reference.U[layer], reference.W[layer],
                                   reference.count[layer], reference.stale, reference.dense_of,
                                   padded_slots, cfg.r, stale_value=1)
                reference.prefix_valid[slots] = 1
                captures = graph.stats["captured"]
                for first in (0, 6):
                    graph.run(candidate, first, states[first:first + 6], slots,
                              eager=native.factorize_layers, policy=policy, batch=bucket)
                    if first == 0:
                        self.assertEqual(torch.count_nonzero(candidate.prefix_valid).item(), 0)
                torch.cuda.synchronize()
                self.assertEqual(graph.stats["captured"] - captures, 2 if repetition == 0 else 0)
                untouched = torch.ones(capacity, device="cuda", dtype=torch.bool)
                untouched[slots] = False
                differences = {}
                for name in initial:
                    actual, expected = getattr(candidate, name), getattr(reference, name)
                    unused = actual[:, untouched] if name in ("a", "U", "W", "count") else actual[untouched]
                    self.assertTrue(torch.all(unused == initial[name]).item(), name)
                    if name in ("a", "U", "W"):
                        differences[name] = dict(nonzero=int(torch.count_nonzero(actual != expected).item()),
                                                 max_abs=float((actual.float() - expected.float()).abs().max().item()))
                        print("PFACTOR checkpoint CUDA factors " + json.dumps(dict(
                            bucket=bucket, rows=rows, field=name, **differences[name])), flush=True)
                        tolerance = 5e-6 if name == "a" else 2e-3
                        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                    else:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                reconstructed = []
                for pool in (candidate, reference):
                    a = pool.a[:, slots].permute(1, 0, 2, 3).reshape(rows, layers * heads, width)
                    u = pool.U[:, slots].permute(1, 0, 2, 3, 4).reshape(rows, layers * heads, cfg.rmax, width)
                    w = pool.W[:, slots].permute(1, 0, 2, 3, 4).reshape(rows, layers * heads, cfg.rmax, width)
                    count = pool.count[:, slots].permute(1, 0, 2).reshape(rows, layers * heads)
                    reconstructed.append(native.densify(a, u, w, count, vbar.reshape(layers * heads, width)))
                torch.testing.assert_close(*reconstructed, rtol=3e-3, atol=3e-3)
                relative = (reconstructed[0] - reconstructed[1]).norm() / reconstructed[1].norm()
                self.assertLess(float(relative.item()), 1e-3)
                print("PFACTOR checkpoint CUDA parity " + json.dumps(dict(
                    bucket=bucket, rows=rows, repetition=repetition, factors=differences,
                    allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    reconstruction_relative_l2=float(relative.item()),
                    captures=graph.stats["captured"])), flush=True)


if __name__ == "__main__":
    unittest.main()
