"""P3 checkpoint normalization and P/D ownership gates (CPU, real tensors)."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'registered/unit/mem_cache'))

from sglang.srt.disaggregation.state_handoff import FactorStateHandoff, FactorTransferFence
from sglang.srt.mem_cache.gdn_pd_fulln_collector import ShallowFullNCollector
from sglang.srt.mem_cache import gdn_prefill_agg_contract as contract
from test_gdn_prefill_batch_graph import fake_pool, make_plan
from twinstar_sgl.pd_shallow import SplitBoundaryPhase
from twinstar_sgl.pd_emitter_graph import DenseEmitterCapture
from twinstar_sgl import pd_emitter_graph as emitter_graph


class ShallowFullNTest(unittest.TestCase):
    def test_nonfinal_shared_extent_retains_one_factorization_and_final_copy(self):
        pool, captured = fake_pool(2), []
        plan = make_plan(pool, 1)
        graph = NS(warmed=True, run=lambda p, plan, states, *args, **kw:
                   captured.append((list(states), args)))
        values = [torch.randn(1, 2, 16, 16) for _ in range(2)]
        with ShallowFullNCollector(pool, plan, graph=graph, same_extent=True) as c:
            for layer in range(2):
                # Deep metadata is rebuilt: equal controls have distinct
                # storage, but name the same source and checkpoint slot.
                c.add(layer, values[layer], None, None, plan.slots.clone(), torch.tensor([20]))
        states, controls = captured[0]
        self.assertTrue(all(tracked is None for _, tracked in states))
        self.assertTrue(all(normal is value for (normal, _), value in zip(states, values)))
        self.assertIsNone(controls[0])
        self.assertTrue(torch.equal(controls[1], plan.slots))
        self.assertEqual(controls[2].tolist(), [20])

    def test_same_extent_rejects_changed_checkpoint_before_publication(self):
        pool = fake_pool(2)
        plan = make_plan(pool, 1)
        graph = NS(warmed=True, run=Mock())
        dense = torch.randn(1, 2, 16, 16)
        with self.assertRaisesRegex(RuntimeError, 'destinations changed'):
            with ShallowFullNCollector(pool, plan, graph=graph, same_extent=True) as c:
                c.add(0, dense, None, None, plan.slots.clone(), torch.tensor([20]))
                c.add(1, dense, None, None, plan.slots.clone(), torch.tensor([21]))
        graph.run.assert_not_called()

    def test_fulln_and_0n_keep_common_checkpoint_but_encode_shallow_h(self):
        for route, encoded in (('0n', 64), ('0', 64), ('0', 65), ('1', 64), ('1', 65)):
            class Schedule:
                def _mamba_radix_cache_v2_req_prepare_for_extend(self, req):
                    # The real scheduler's prompt_p_extent policy.
                    extent = req.extend_range.length - int(not req._pfactor_agg_contract)
                    req.kv.mamba_last_track_seqlen = extent
                    return extent
            class Forward:
                @classmethod
                def init_new(cls, batch, runner):
                    return NS(forward_mode=batch.forward_mode, batch_size=1,
                              input_ids=torch.arange(65), extend_seq_lens_cpu=[65],
                              mamba_track_seqlens=torch.tensor([encoded]),
                              mamba_track_mask=torch.tensor([True]), seq_lens=torch.tensor([65]),
                              extend_prefix_lens=torch.tensor([0]))
            class Backend:
                def init_forward_metadata(self, batch): pass
            class Handoff(FactorStateHandoff): pass
            req = NS(extend_range=NS(length=65, end=65), origin_input_ids=list(range(65)),
                     kv=NS(mamba_last_track_idx=None,
                     mamba_next_track_idx=0, mamba_last_track_seqlen=None))
            batch = Schedule()
            batch.reqs = [req]
            batch.forward_mode = NS(is_extend=lambda: True, is_mixed=lambda: False)
            runner = NS(model=NS(config=NS(mamba_chunk_size=64)))
            with patch.dict(os.environ, {contract.FLAG: '1'}), \
                 patch.object(contract, '_PD_MICRO_ROUTE', route):
                contract.install_contracts(Forward, Schedule, Backend, Handoff, shallow=True)
                self.assertEqual(batch._mamba_radix_cache_v2_req_prepare_for_extend(req), 64)
                result = Forward.init_new(batch, runner)
            self.assertEqual(result._pfactor_agg_contract, route != '0n')
            if route != '0n':
                self.assertEqual(result._pd_fulln_emitter_track_seqlens.item(), 64)
                self.assertEqual(result.mamba_track_seqlens.item(), 65)

    def test_one_publication_normalizes_intermediate_and_final_checkpoints(self):
        pool = fake_pool(2)
        plan = make_plan(pool, 2)
        shallow, deep = torch.randn(2, 2, 16, 16), torch.randn(2, 2, 16, 16)
        checkpoint = torch.randn(2, 2, 16, 16)
        slots = torch.tensor([50, 51])
        graph = NS(warmed=True, run=Mock())
        with ShallowFullNCollector(pool, plan, graph=graph) as c:
            c.add(0, shallow, checkpoint, slots, None, None)
            # Deep row 0 uses its final state; row 1 uses intermediate h.
            c.add(1, deep, checkpoint[1:], slots[1:], plan.slots[:1], slots[:1])
        self.assertEqual(graph.run.call_count, 1)
        args = graph.run.call_args.args
        # Collector clears its pending list after publication; mock stores it
        # by reference, so inspect inside the call in the next regression.
        self.assertTrue(c.published)
        self.assertTrue(torch.equal(c.checkpoint_slots, slots))

    def test_normalized_values_and_destinations_survive_permuted_sources(self):
        pool, captured = fake_pool(2), []
        plan = make_plan(pool, 2)
        graph = NS(warmed=True, run=lambda p, plan, states, *args, **kw:
                   captured.append((list(states), args)))
        shallow = torch.full((2, 2, 16, 16), 2.)
        deep = torch.stack((torch.full((2, 16, 16), 7.), torch.full((2, 16, 16), 9.)))
        h = torch.full_like(shallow, 3.)
        slots = torch.tensor([50, 51])
        with ShallowFullNCollector(pool, plan, graph=graph) as c:
            c.add(0, shallow, h, slots, None, None)
            c.add(1, deep, None, None, plan.slots.flip(0), slots.flip(0))
        states, controls = captured[0]
        self.assertTrue(torch.equal(states[0][0], shallow))
        self.assertTrue(torch.equal(states[1][0], deep))
        self.assertTrue(torch.equal(states[0][1], h))
        self.assertTrue(torch.equal(states[1][1], deep))
        self.assertTrue(torch.equal(controls[0], slots))
        self.assertEqual(controls[1:], (None, None))

    def test_missing_checkpoint_source_fails_before_publication(self):
        pool = fake_pool(1)
        plan = make_plan(pool, 1)
        graph = NS(warmed=True, run=Mock())
        with self.assertRaises(RuntimeError):
            with ShallowFullNCollector(pool, plan, graph=graph) as c:
                c.add(0, torch.ones(1, 2, 16, 16), None, None,
                      torch.tensor([999]), torch.tensor([50]))
        graph.run.assert_not_called()

    def test_fulln_phase_delegates_to_native_fence_and_legacy_keeps_mixed_phase(self):
        pool = NS(cfg=NS(r=16, strict_chunk=True), layer_ids=[0, 32],
                  count=torch.tensor([[16], [16]]))
        state = NS(valid=torch.tensor([0]))
        original = NS(pool=pool, before_send=Mock())
        req = NS(kv=NS(mamba_pool_idx=torch.tensor([0])), _pfactor_agg_contract=True)
        SplitBoundaryPhase(original, state).before_send(req)
        original.before_send.assert_called_once_with(req)
        self.assertIs(req._pd_fulln_boundary_valid, state.valid)
        self.assertEqual(req.factored_prefill_boundary_steps, 0)
        req._pfactor_agg_contract = False
        with self.assertRaisesRegex(RuntimeError, 'not published'):
            SplitBoundaryPhase(original, state).before_send(req)
        state.valid.fill_(1)
        with self.assertRaisesRegex(RuntimeError, 'r\\+1/r'):
            SplitBoundaryPhase(original, state).before_send(req)
        pool.count[0] += 1
        SplitBoundaryPhase(original, state).before_send(req)

    def test_transfer_worker_waits_before_boundary_check_and_rejects_early_reader(self):
        count, valid = torch.tensor([[0]]), torch.tensor([0])
        events = []
        def publication():
            events.append('publication')
            count.fill_(16)
        pub = NS(synchronize=publication, query=lambda: True)
        producer = NS(synchronize=lambda: (events.append('producer'), valid.fill_(1)), query=lambda: True)
        fence = FactorTransferFence(pub, count, torch.tensor([0]), 16, boundary_valid=valid)
        fence.wait(producer)
        self.assertTrue(fence.validated)
        self.assertEqual(events, ['publication', 'producer'])
        valid.zero_()
        early = FactorTransferFence(None, count, torch.tensor([0]), 16, boundary_valid=valid)
        with self.assertRaisesRegex(RuntimeError, 'not published'):
            early.wait(NS(synchronize=lambda: None, query=lambda: True))
        self.assertFalse(early.validated)

    def test_graph_recording_does_not_publish_factors(self):
        plan = NS(next_layer=24)
        recorder = DenseEmitterCapture(plan)
        data = torch.randn(1, 2, 16, 16)
        for layer in range(32, 48):
            if layer % 4 != 3:
                recorder.add(layer, data, None, None, None, None)
        self.assertEqual(plan.next_layer, 36)
        self.assertEqual(len(recorder.rows), 12)
        self.assertTrue(all(row[1] is data for row in recorder.rows))

    def test_explicit_0n_keeps_exact_tail_emitters_eager(self):
        owner = NS(pd_shallow_role='prefill', fullstack_v3_latent=False,
                   emitter_ids=list(range(31, 48)))
        backend = NS(linear_attn_backend=NS(factored=NS(_exact_tail_transaction=object())))
        with patch.dict(os.environ, {'TWINSTAR_PD_EMITTER_GRAPH': '1'}):
            self.assertFalse(emitter_graph.try_emit(owner, None, None, backend=backend,
                                                   prefix_metadata=None, split=True))

    def test_graph_replay_donates_current_outputs_to_current_live_collector(self):
        manager = object.__new__(emitter_graph.CollectingEmitterGraph)
        manager.entries, manager.limit = {}, 8
        manager.stats = dict(captured=0, replayed=0)
        manager.invalidate_moved_ring = Mock()
        manager.progress = Mock()
        manager.check_dense_rebind = lambda entry, fb: entry['graph'].replay()
        static = torch.zeros(1)
        bound = {}
        def capture(source, fb, backend, metadata):
            bound.update(source)
            static.copy_(bound['streams'])
            return dict(inputs=NS(bind=lambda source: bound.update(source)),
                        graph=NS(replay=lambda: static.copy_(bound['streams'])),
                        rows=[(32, static, None, None, None, None)], rebound_checked=True)
        manager.capture_dense = capture
        factor = NS(cfg=NS(init_method='k31'))
        backend = NS(linear_attn_backend=NS(factored=factor))
        with patch.object(emitter_graph, 'source_inputs', side_effect=lambda owner, streams, *args: {'streams': streams}), \
             patch.object(emitter_graph, 'structure', side_effect=lambda source: ('cuda-layout', tuple(source['streams'].shape))), \
             patch.object(emitter_graph, 'factor_policy', return_value='same-policy'):
            manager.owner = NS()
            for value in (2., 5.):
                live = NS(next_layer=24, last_layer=35, pending=[], batch_collector=NS(add=Mock()))
                self.assertTrue(manager.run(torch.tensor([value]), NS(batch_size=1), backend,
                                            NS(factored_extend=live), True))
                donated = live.batch_collector.add.call_args.args
                self.assertEqual(donated[1].item(), value)
        self.assertEqual(manager.stats, dict(captured=1, replayed=1))


if __name__ == '__main__':
    unittest.main()
