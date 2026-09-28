"""Whole-prefix publication waits for every normal and tracked layer."""
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache import gdn_factored_pool as native
from sglang.srt.mem_cache import gdn_prefill_batch_graph as module


def fake_pool(layers=6, device="cpu", width=16, heads=2, capacity=40):
    cfg = native.FactoredGDNConfig(init_method="k31", dtype=torch.float16 if str(device).startswith("cuda") else torch.float32, factored_prefix=1)
    omega = torch.randn(1, heads, width, cfg.r + cfg.init_oversample, device=device)
    pool = NS(cfg=cfg, layer_ids=list(range(layers)), layer_map={i: i for i in range(layers)},
              hv=heads, v=width, k=width, prefix_layer_count=lambda: layers,
              init_omega=lambda b: omega.expand(b, -1, -1, -1),
              a=torch.full((layers, capacity, heads, width), .5, device=device),
              U=torch.zeros(layers, capacity, heads, cfg.rmax, width, device=device, dtype=cfg.dtype),
              W=torch.zeros(layers, capacity, heads, cfg.rmax, width, device=device, dtype=cfg.dtype),
              count=torch.full((layers, capacity, heads), 3, dtype=torch.int32, device=device),
              stale=torch.zeros(capacity, dtype=torch.int32, device=device),
              dense_of=torch.full((capacity,), -1, dtype=torch.int32, device=device),
              dense_required=torch.zeros(capacity, dtype=torch.int32, device=device),
              prefix_valid=torch.zeros(capacity, dtype=torch.int32, device=device),
              dense_ring=torch.zeros(layers, 16, heads, width, width, device=device),
              vbar=torch.nn.functional.normalize(torch.randn(layers, heads, width, device=device), dim=-1),
              pside_join=Mock(), invalidate_prefix_dense=Mock(), ring_generation=0, prefix_dense=None)
    return pool


def make_plan(pool, rows=1):
    slots = torch.arange(1, rows+1, device=pool.a.device)
    return native.FactoredExtendPlan(slots=slots, use_ring=slots.bool(), ring_src=slots-1,
        ring_dst=slots-1, ring_dst_rows=slots-1, last_layer=len(pool.layer_ids)-1,
        dense_required_after_commit=torch.ones(rows, dtype=torch.int32, device=pool.a.device))


class PrefillBatchGraphTest(unittest.TestCase):
    def test_no_partial_publication_and_both_branches_flush_before_tail(self):
        for layers in (36, 48):
            pool, order = fake_pool(layers), []
            pool._prefill_batch_graph = NS(warmed=True, run=Mock(side_effect=lambda *a, **kw: order.append("commit")))
            plan = make_plan(pool)
            track_slots = torch.tensor([9])
            with module.BatchCollector(pool, plan):
                for layer in pool.layer_ids:
                    dense = torch.full((1, 2, 16, 16), float(layer))
                    native.FactoredGDNPool.commit_extend_batched(pool, layer, plan, dense, dense, track_slots)
                    self.assertEqual(order, [])
                retained = tuple(plan.pending)
            order.append("tail")
            self.assertEqual(order, ["commit", "tail"])
            self.assertEqual(len(retained), layers)
            self.assertEqual(plan.pending, [])
            self.assertIsNone(plan.batch_collector)
            self.assertEqual(pool._prefill_batch_graph.run.call_count, 1)

    def test_missing_layer_or_changed_checkpoint_cannot_publish(self):
        pool = fake_pool()
        pool._prefill_batch_graph = NS(warmed=True, run=Mock())
        plan = make_plan(pool)
        dense = torch.zeros(1, 2, 16, 16)
        with self.assertRaisesRegex(RuntimeError, "before every layer"):
            with module.BatchCollector(pool, plan) as collect:
                collect.add(0, dense, None, None, None, None)
        pool._prefill_batch_graph.run.assert_not_called()
        plan = make_plan(pool)
        with self.assertRaisesRegex(RuntimeError, "destinations changed"):
            with module.BatchCollector(pool, plan) as collect:
                collect.add(0, dense, dense, torch.tensor([20]), None, None)
                collect.add(1, dense, dense, torch.tensor([21]), None, None)
        pool._prefill_batch_graph.run.assert_not_called()

    def test_bf16_padding_and_ring_generation_keep_live_bindings(self):
        pool = fake_pool()
        buffers = module.BatchBuffers(pool, 8, 8)
        plan = make_plan(pool, 3)
        normal = torch.randn(3, 2, 16, 16)
        tracked = torch.randn(1, 2, 16, 16).bfloat16()
        track_slots = torch.tensor([20])
        states = [(normal, tracked)] * 6
        buffers.bind(plan, states, track_slots, torch.tensor([1]), torch.tensor([21]))
        self.assertEqual(buffers.slots.tolist(), [1, 2, 3, -1, -1, -1, -1, -1])
        self.assertEqual(buffers.track_slots.tolist(), [20, -1, -1, -1, -1, -1, -1, -1])
        self.assertTrue(torch.equal(buffers.tracked[0][:1], tracked.float()))
        self.assertEqual(torch.count_nonzero(buffers.tracked[0][1:]).item(), 0)
        pool.dense_ring = torch.ones_like(pool.dense_ring)
        pool.ring_generation += 1
        buffers.bind(plan, states, track_slots, None, None)
        self.assertEqual(buffers.ring_pointers.tolist(), [row.data_ptr() for row in pool.dense_ring])
        self.assertTrue(torch.all(buffers.final_dst == -1))

    def test_prewarm_covers_all_buckets_and_never_captures_in_request(self):
        pool = fake_pool()
        graph, stream, cuda_graph = module.PrefillBatchGraph(), Mock(), Mock()
        with ExitStack() as stack:
            stack.enter_context(patch.object(module.BatchBuffers, "evaluate"))
            for name, value in (("current_stream", stream), ("Stream", stream),
                                ("CUDAGraph", cuda_graph), ("memory_allocated", 0),
                                ("memory_reserved", 0),
                                ("graph_pool_handle", object()), ("synchronize", None)):
                stack.enter_context(patch.object(torch.cuda, name, return_value=value))
            for name in ("stream", "graph"):
                stack.enter_context(patch.object(torch.cuda, name, side_effect=lambda *a, **kw: nullcontext()))
            graph.prewarm(pool, eager=native.factorize_layers, policy=())
            self.assertEqual({key[:2] for key in graph.entries},
                             set(module.prewarm_shapes()))
            dense = torch.randn(3, 2, 16, 16)
            for dtype in (torch.bfloat16, torch.float32):
                graph.run(pool, make_plan(pool, 3), [(dense, dense.to(dtype))] * 6,
                          torch.tensor([20, 21, 22]), None, None,
                          eager=native.factorize_layers, policy=())
            self.assertEqual(graph.stats["captured"], 18)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_whole_graph_matches_layerwise_and_survives_ring_growth(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_factored_io import store_factored

        import copy
        import json

        torch.manual_seed(1007)
        pool = fake_pool(layers=36, device="cuda", width=128, heads=24)
        graph = module.PrefillBatchGraph()
        fields = ("a", "U", "W", "count", "stale", "dense_of", "dense_required", "prefix_valid")
        original = {name: getattr(pool, name).clone() for name in fields}
        graph.prewarm(pool, eager=native.factorize_layers, policy=())
        for name in fields:
            torch.testing.assert_close(getattr(pool, name), original[name], atol=0, rtol=0)
        for repetition, (rows, track_rows) in enumerate(((16, 16), (1, 0), (3, 1), (1, 8))):
            for name in fields:
                getattr(pool, name).copy_(original[name])
            pool.dense_ring.zero_()
            if repetition == 2:
                pool.dense_ring = torch.zeros(36, 32, 24, 128, 128, device="cuda")
                pool.ring_generation += 1
            reference = copy.copy(pool)
            for name in (*fields, "dense_ring"):
                setattr(reference, name, getattr(pool, name).clone())
            plan = make_plan(pool, rows)
            dense = [torch.randn(rows, 24, 128, 128, device="cuda") for _ in pool.layer_ids]
            tracked = ([torch.randn(track_rows, 24, 128, 128, device="cuda").bfloat16()
                        for _ in pool.layer_ids] if track_rows else [None] * len(pool.layer_ids))
            track_slots = (torch.arange(20, 20 + track_rows, device="cuda") if track_rows else None)
            bucket = next(b for b in module.BATCH_BUCKETS if b >= max(rows, track_rows))
            normal_bucket = 1 if rows == 1 else bucket
            track_bucket = 1 if track_rows == 1 else bucket
            for i in pool.layer_ids:
                for states, slots, count, padded, stale in (
                        (dense, plan.slots, rows, normal_bucket, 0),
                        (tracked, track_slots, track_rows, track_bucket, 1)):
                    if not count:
                        continue
                    state = torch.zeros(padded, 24, 128, 128, device="cuda", dtype=states[i].dtype)
                    state[:count].copy_(states[i])
                    controls = torch.full((padded,), -1, dtype=torch.long, device="cuda")
                    controls[:count].copy_(slots)
                    values = native.factorize_layers([state], pool.vbar[i:i+1], pool.cfg,
                                                       omega=pool.init_omega(padded))[0]
                    kwargs = {}
                    if not stale:
                        ring_dst = controls.clone()
                        ring_dst[:count].copy_(plan.ring_dst)
                        kwargs = dict(dense=state.float(), ring=reference.dense_ring[i], ring_dst=ring_dst)
                    store_factored(*values, reference.a[i], reference.U[i], reference.W[i], reference.count[i],
                                   reference.stale, reference.dense_of, controls, pool.cfg.r, stale_value=stale, **kwargs)
            reference.prefix_valid[plan.slots] = 1
            reference.dense_required[plan.slots] = plan.dense_required_after_commit
            if track_rows:
                reference.prefix_valid[track_slots] = 1
            src, dst = torch.tensor([1], device="cuda"), torch.tensor([38], device="cuda")
            native.FactoredGDNPool.copy_slots(reference, src, dst)
            graph.run(pool, plan, list(zip(dense, tracked)), track_slots, src, dst,
                      eager=native.factorize_layers, policy=())
            torch.cuda.synchronize()
            errors = {}
            for name in fields:
                actual, expected = getattr(pool, name), getattr(reference, name)
                torch.testing.assert_close(actual, expected, atol=3e-3 if name in ("U", "W") else 5e-6,
                                           rtol=3e-3 if name in ("U", "W") else 5e-6)
                sentinel = actual[:, 0] if name in ("a", "U", "W", "count") else actual[0]
                previous = original[name][:, 0] if name in ("a", "U", "W", "count") else original[name][0]
                torch.testing.assert_close(sentinel, previous, atol=0, rtol=0)
                if name in ("a", "U", "W"):
                    errors[name] = float((actual.float() - expected.float()).abs().max())
            torch.testing.assert_close(pool.dense_ring, reference.dense_ring, atol=0, rtol=0)
            self.assertEqual(graph.stats["captured"], 18)
            print("PFACTOR batch CUDA parity " + json.dumps(dict(normal=rows, tracked=track_rows,
                layers=36, dtype=str(pool.cfg.dtype), ring_capacity=pool.dense_ring.shape[1],
                captures=graph.stats["captured"], max_abs=errors)), flush=True)


if __name__ == "__main__":
    unittest.main()
