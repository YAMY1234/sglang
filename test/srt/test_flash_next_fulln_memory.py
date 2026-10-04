"""Bounded full-N storage and counters; CUDA allocation is modeled, not executed."""
import gc
import json
import os
import weakref
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from test_flash_next_agg_fulln import worker, schedule, init_graphs, torch, ForwardMode
from test_gdn_prefill_batch_graph import fake_pool, make_plan
from sglang.srt.mem_cache import gdn_prefill_agg_contract as agg
from sglang.srt.mem_cache import gdn_prefill_batch_graph as bg
from sglang.srt.mem_cache import gdn_factored_pool as native
from sglang.srt.mem_cache.gdn_fulln_workspace import (
    FullNWorkspace, row_capacity, state_memory_bytes, unique_state_bytes,
)

FLAG = 'SGLANG_GDN_AGG_FULLN_COMPACT_BUFFERS'


class FullNMemoryTest(unittest.TestCase):
    def test_capacity_uses_chunk_and_request_bound_not_trunk_bucket(self):
        self.assertEqual(row_capacity(32768), 16)
        self.assertEqual(row_capacity(16384), 16)
        self.assertEqual(row_capacity(32768, 8), 8)
        self.assertEqual(row_capacity(6), 4)
        self.assertEqual(row_capacity(2), 1)
        for chunk in (None, -1, 0, 1):
            with self.assertRaises(ValueError): row_capacity(chunk)
        with self.assertRaises(ValueError): row_capacity(32768, 0)

    def test_BF16_32768_trunk_OOM_allocation_ledger(self):
        # Exact 36-layer/TP2 geometry from the BF16 receipt. No 1.69 GiB CPU
        # allocation: meta tensors exercise the real constructor's shapes.
        pool = NS(a=torch.empty(0, device='meta'), layer_ids=list(range(36)),
                  hv=24, v=128, k=128)
        ws = FullNWorkspace(pool, row_capacity(32768), 32768)
        actual = sum(x.numel() * x.element_size() for x in ws.slabs.values())
        self.assertEqual(actual, 1811939328)
        self.assertEqual(actual, state_memory_bytes(36, 24, 128, 128, 16))
        old = 3510632448  # old log: 2 * (1+2+4+8+16) * 36 * 24 * 128**2 * 4
        self.assertEqual(old, actual * 31 // 16)
        requested = 32768 * 10240 * 2
        self.assertEqual(requested, 640 * 1024**2)
        observed_free = 31522816
        self.assertLess(observed_free, requested)
        # Allocation arithmetic only. Private-pool/allocator fragmentation and
        # the remaining graph-capture peaks still require the hardware rerun.
        self.assertGreater(observed_free + old - actual, requested)
        print('FULLN_MEMORY_LEDGER', json.dumps(dict(
            chunk_tokens=32768, trunk_graph_GiB=22.9, old_state_bytes=old,
            compact_state_bytes=actual, saved_bytes=old-actual,
            failed_activation_bytes=requested, old_free_bytes=observed_free,
            hardware_OOM_resolved=False)))

    def test_all_bucket_inputs_share_two_bounded_storages(self):
        pool = fake_pool(3, width=16)
        ws = FullNWorkspace(pool, 16, 32768)
        self.assertEqual(unique_state_bytes(ws.shared), state_memory_bytes(3, 2, 16, 16, 16))
        for role in ('normal', 'tracked'):
            for size in (1, 2, 4, 8, 16):
                for i, view in enumerate(ws.shared[role, size]):
                    self.assertTrue(view.is_set_to(ws.slabs[role][i, :size]))
        # Constructing every publication shape does not allocate state per shape.
        for batch, tracked in bg.prewarm_shapes():
            buffers = bg.BatchBuffers(pool, batch, tracked, ws.shared, include_tail=False)
            self.assertEqual(buffers.normal[0].untyped_storage().data_ptr(),
                             ws.slabs['normal'].untyped_storage().data_ptr())
        self.assertEqual(unique_state_bytes(ws.shared), state_memory_bytes(3, 2, 16, 16, 16))

    def test_snapshot_releases_producer_storage_and_preserves_BF16_values(self):
        pool = fake_pool(2)
        ws = FullNWorkspace(pool, 4, 32768)
        producer = torch.randn(80, 2, 16, 16)
        reference = weakref.ref(producer)
        normal = producer[:3]
        tracked = torch.randn(1, 2, 16, 16).bfloat16()
        expected = (normal.clone(), tracked.float())
        got = ws.snapshot(0, normal, tracked, 8)
        del normal, producer, tracked
        gc.collect()
        self.assertIsNone(reference())
        self.assertTrue(torch.equal(got[0], expected[0]))
        self.assertTrue(torch.equal(got[1], expected[1]))
        for tokens, rows, track_rows in ((32769, 1, 0), (2, 2, 0), (8, 5, 0), (8, 1, 2)):
            with self.assertRaises(ValueError):
                ws.snapshot(0, torch.zeros(rows, 2, 16, 16),
                            None if not track_rows else torch.zeros(track_rows, 2, 16, 16), tokens)

    def test_reuse_and_padding_match_old_graph_inputs_bitwise(self):
        pool = fake_pool(3)
        ws = FullNWorkspace(pool, 16, 32768)
        for rows, tracks in ((16, 16), (3, 1), (1, 0), (8, 8)):
            bucket = next(b for b in (1, 2, 4, 8, 16) if b >= rows)
            track_bucket = None if not tracks else (1 if tracks == 1 else bucket)
            before = bg.BatchBuffers(pool, bucket, track_bucket, include_tail=False)
            after = bg.BatchBuffers(pool, bucket, track_bucket, ws.shared, include_tail=False)
            states = [(torch.randn(rows, 2, 16, 16),
                       torch.randn(tracks, 2, 16, 16).bfloat16() if tracks else None)
                      for _ in pool.layer_ids]
            copied = [ws.snapshot(i, *state, 32768) for i, state in enumerate(states)]
            plan = make_plan(pool, rows)
            slots = torch.arange(20, 20 + tracks) if tracks else None
            before.bind(plan, states, slots, None, None)
            after.bind(plan, copied, slots, None, None)
            for name in ('normal', 'tracked'):
                if getattr(before, name) is not None:
                    for old, new in zip(getattr(before, name), getattr(after, name), strict=True):
                        self.assertTrue(torch.equal(old, new))
            for name in ('slots', 'ring_dst', 'final_src', 'final_dst', 'required'):
                self.assertTrue(torch.equal(getattr(before, name), getattr(after, name)))

    def test_collector_snapshots_directly_into_graph_inputs_and_publishes_once(self):
        pool = fake_pool(3)
        ws = FullNWorkspace(pool, 4, 32)
        graph = NS(workspace=ws, warmed=True, run=Mock())
        plan = make_plan(pool, 3)
        track_slots = torch.tensor([20])
        with bg.BatchCollector(pool, plan, graph=graph, token_count=8) as collector:
            for i in pool.layer_ids:
                dense = torch.full((3, 2, 16, 16), float(i))
                tracked = dense[:1].bfloat16()
                native.FactoredGDNPool.commit_extend_batched(
                    pool, i, plan, dense, tracked, track_slots)
                self.assertTrue(plan.pending[-1][0].is_set_to(ws.slabs['normal'][i, :3]))
                graph.run.assert_not_called()
        self.assertTrue(collector.published)
        graph.run.assert_called_once()
        self.assertEqual(plan.pending, [])

    def test_bounded_prewarm_captures_once_and_reuses_the_same_buffers(self):
        pool = fake_pool(2)
        ws = FullNWorkspace(pool, 4, 8)
        graph = bg.PrefillBatchGraph(include_tail=False, shared=ws.shared, workspace=ws)
        stream, cuda_graph = Mock(), Mock()
        with ExitStack() as stack:
            stack.enter_context(patch.object(bg.BatchBuffers, 'evaluate'))
            for name, value in (('current_stream', stream), ('Stream', stream),
                    ('CUDAGraph', cuda_graph), ('memory_allocated', 0), ('memory_reserved', 0),
                    ('graph_pool_handle', object()), ('synchronize', None)):
                stack.enter_context(patch.object(torch.cuda, name, return_value=value))
            for name in ('stream', 'graph'):
                stack.enter_context(patch.object(torch.cuda, name, side_effect=lambda *a, **k: nullcontext()))
            graph.prewarm(pool, eager=native.factorize_layers, policy=())
            self.assertEqual({k[:2] for k in graph.entries},
                             {x for x in bg.prewarm_shapes() if max(x[0], x[1] or 0) <= 4})
            before = graph.stats['captured']
            states = [(torch.randn(3, 2, 16, 16), None)] * 2
            graph.run(pool, make_plan(pool, 3), states, None, None, None,
                      eager=native.factorize_layers, policy=())
            self.assertEqual(graph.stats['captured'], before)
            self.assertEqual(unique_state_bytes(ws.shared), state_memory_bytes(2, 2, 16, 16, 4))

    def test_real_install_bounds_and_summaries_include_fallbacks(self):
        with worker() as w, patch.dict(os.environ, {FLAG: '1'}), patch(
                'sglang.srt.model_executor.runner.get_is_capture_mode',
                return_value=False) as capture_mode:
            w.runner.server_args.chunked_prefill_size = 128
            w.runner.server_args.max_running_requests = 2
            init_graphs(w.runner, w.capture)
            self.assertEqual(w.pool._agg_prefill_graph.workspace.capacity, 2)
            for rows, tokens, selected in ((1, 64, True), (3, 64, False), (1, 129, False)):
                batch = schedule(w, rows=rows, tokens=tokens)
                fb = w.Forward.init_new(batch, w.runner)
                self.assertEqual(fb._pfactor_agg_contract, selected)
                w.owner.prepare_forward_batch(fb)
                if selected: w.backend.init_forward_metadata(fb)
                w.owner.forward(fb.input_ids, fb.input_ids, fb)
            stats = w.owner._agg_fulln_summary
            self.assertEqual((stats.forwards, stats.trunk_replays, stats.batch_publications, stats.fallbacks),
                             (3, 1, 1, 2))
            fb.forward_mode = ForwardMode.DECODE
            w.owner.forward(fb.input_ids, fb.input_ids, fb)
            self.assertEqual(stats.forwards, 3)
            # Install captures this function in its closure. Mutate the same
            # callable, rather than replacing the module name after install.
            capture_mode.return_value = True
            fb.forward_mode = ForwardMode.EXTEND
            w.owner.forward(fb.input_ids, fb.input_ids, fb)
            self.assertEqual(stats.forwards, 3)

    def test_periodic_and_final_summary_and_invalid_interval(self):
        with patch.object(agg.logger, 'info') as log:
            stats = agg.FullNSummary('null', 3)
            stats.record(1, trunk=True, published=True)
            stats.record(1, fallback=True)
            self.assertEqual(log.call_count, 1)
            stats.record(8, trunk=True, published=True)
            self.assertEqual(log.call_count, 2)
            stats.log()
            self.assertEqual(log.call_count, 3)
            args = log.call_args.args
            self.assertIn('fallbacks=%d', args[0])
            self.assertEqual(args[1:], ('null', 3, 2, 2, 1, 8, 'shutdown'))
        with self.assertRaises(ValueError): agg.FullNSummary('null', 0)

    def test_default_off_retains_unbounded_prewarm_and_overlap_rejection(self):
        with worker() as w:
            init_graphs(w.runner, w.capture)
            self.assertIsNone(w.pool._agg_prefill_graph.workspace)
            self.assertFalse(hasattr(w.pool, '_agg_fulln_workspace_limits'))
        with worker() as w, patch('sglang.srt.runtime_context.get_schedule',
                                 return_value=NS(disable_overlap_schedule=False)):
            with self.assertRaisesRegex(ValueError, 'isolated scheduling'):
                init_graphs(w.runner, w.capture)


if __name__ == '__main__':
    unittest.main()
