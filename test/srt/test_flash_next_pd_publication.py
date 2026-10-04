"""PD publication lifetime/order and host packing; GPU admission is separate."""
import copy
import inspect
import json
import os
from contextlib import nullcontext
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from test_flash_next_p48_contract import worker, init_graphs, IDS, ForwardMode, torch
from test_flash_next_agg_fulln import worker as agg_worker
from test_gdn_prefill_batch_graph import fake_pool, make_plan
from test_gdn_factored_host_sync import _pool, POOL_FIELDS
from sglang.srt.mem_cache import gdn_factored_pool as native
from sglang.srt.mem_cache.gdn_prefill_batch_graph import BatchBuffers, BatchCollector
from sglang.srt.mem_cache.gdn_pd_publication import (
    PDBatchPublication, CudaPublicationRuntime, install_forward_join,
)
from sglang.srt.disaggregation.state_handoff import FactorStateHandoff
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

FLAG = 'SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED'


class Runtime:
    """CPU stream model: device-visible writes complete at the reader join."""
    def __init__(self):
        self.events = []

    def launch_after_forward(self, operation, inputs):
        self.events.extend(('forward_done', 'side_wait_forward', 'launch'))
        return NS(operation=operation)

    def join(self, ticket):
        self.events.append('complete_publication')
        ticket.operation()
        ticket.operation = None
        self.events.append('reader_join')


class ArithmeticGraph:
    """Real input binding/factorization; CPU indexing replaces CUDA stores."""
    include_tail = False
    warmed = True

    def __init__(self):
        self.calls = 0

    def run(self, pool, plan, states, track_slots, final_src, final_dst, *, eager, policy):
        b = plan.slots.numel()
        bt = None if track_slots is None else track_slots.numel()
        inputs = BatchBuffers(pool, b, bt, include_tail=False)
        inputs.bind(plan, states, track_slots, final_src, final_dst)
        for data, omega, slots, stale in (
                (inputs.normal, inputs.omega, plan.slots, 0),
                (inputs.tracked, inputs.track_omega, track_slots, 1)):
            if data is None:
                continue
            values = eager(data, pool.vbar, pool.cfg, omega=omega)
            for i, (a, u, w) in enumerate(values):
                pool.a[i, slots], pool.U[i, slots], pool.W[i, slots] = a, u, w
                pool.count[i, slots] = pool.cfg.r
            pool.stale[slots] = stale
            pool.prefix_valid[slots] = 1
        pool.dense_of[plan.slots] = plan.ring_dst.int()
        pool.dense_required[plan.slots] = plan.dense_required_after_commit
        self.calls += 1


def with_reader(pool):
    pool.pside_join = MethodType(native.FactoredGDNPool.pside_join, pool)
    pool.cfg.strict_chunk = True
    runtime = Runtime()
    pool._pd_batch_publication = PDBatchPublication(pool, runtime)
    return runtime, pool._pd_batch_publication


class PDPublicationTest(unittest.TestCase):
    def test_cuda_event_order_and_lowest_priority(self):
        events = []
        main = NS(wait_event=lambda e: events.append('main_wait'))
        side = NS(wait_event=lambda e: events.append('side_wait'))
        event = lambda: NS(record=lambda stream: events.append(
            'forward_done' if stream is main else 'publication_done'))
        tensor = NS(is_cuda=True, record_stream=lambda stream: events.append('retain'))
        with patch.object(torch.cuda, 'Stream', return_value=side) as stream, \
             patch.object(torch.cuda, 'current_stream', return_value=main), \
             patch.object(torch.cuda, 'Event', side_effect=event), \
             patch.object(torch.cuda, 'stream', return_value=nullcontext()):
            rt = CudaPublicationRuntime('cuda:0')
            ticket = rt.launch_after_forward(lambda: events.append('graph'), [tensor])
            rt.join(ticket)
        stream.assert_called_once_with(device='cuda:0', priority=0)
        self.assertEqual(events, ['forward_done', 'side_wait', 'graph', 'retain',
                                  'publication_done', 'main_wait'])

    def test_real_factorization_and_fields_equal_after_send_join_for_1_2_4_8_rows(self):
        for rows in (1, 2, 4, 8):
            with self.subTest(rows=rows):
                torch.manual_seed(100 + rows)
                original = fake_pool(2, width=32)
                eager_pool, side_pool = copy.deepcopy(original), copy.deepcopy(original)
                rt, side = with_reader(side_pool)
                states = [(torch.randn(rows, 2, 32, 32), torch.randn(1, 2, 32, 32))
                          for _ in original.layer_ids]
                track_slots = torch.tensor([30])
                eager_plan, side_plan = make_plan(eager_pool, rows), make_plan(side_pool, rows)
                eager_graph, side_graph = ArithmeticGraph(), ArithmeticGraph()
                eager_graph.run(eager_pool, eager_plan, states, track_slots, None, None,
                                eager=native.factorize_layers, policy=())
                before = side_pool.count.clone()
                with BatchCollector(side_pool, side_plan, graph=side_graph, publication=side) as c:
                    for lid, (normal, tracked) in enumerate(states):
                        c.add(lid, normal, tracked, track_slots, None, None)
                side.start_after_forward()
                self.assertTrue(torch.equal(before, side_pool.count))
                self.assertEqual(side_plan.pending, [])
                req = NS(kv=NS(mamba_pool_idx=torch.tensor(1)), factored_prefill_boundary_steps=0)
                FactorStateHandoff(side_pool).before_send(req)
                for name in POOL_FIELDS:
                    a, b = getattr(eager_pool, name), getattr(side_pool, name)
                    self.assertTrue(torch.equal(a, b), (rows, name))
                    if a.is_floating_point(): self.assertTrue(torch.isfinite(b).all())
                self.assertEqual(side_graph.calls, 1)
                self.assertEqual(rt.events[-1], 'reader_join')
                self.assertIsNone(side.pending)
                FactorStateHandoff(side_pool).before_send(req)
                self.assertEqual(side_graph.calls, 1)

    def test_early_reader_drains_without_waiting_for_next_request(self):
        pool = fake_pool(2)
        rt, side = with_reader(pool)
        plan = make_plan(pool)
        graph = NS(include_tail=False, warmed=True, run=Mock(
            side_effect=lambda *a, **k: pool.count[:, 1].fill_(pool.cfg.r)))
        side.submit(graph, plan, [(torch.ones(1), None)] * 2, (None, None, None),
                    eager=None, policy=())
        req = NS(kv=NS(mamba_pool_idx=torch.tensor(1)))
        FactorStateHandoff(pool).before_send(req)
        self.assertEqual(side.stats['early_reader'], 1)
        self.assertEqual(rt.events, ['forward_done', 'side_wait_forward', 'launch',
                                     'complete_publication', 'reader_join'])

    def test_reader_table_joins_before_payload_access(self):
        class Joined(Exception): pass
        pool = native.FactoredGDNPool.__new__(native.FactoredGDNPool)
        pool._exact_tail_transaction = None
        pool.pside_join = Mock(side_effect=Joined)
        readers = dict(reset_slots=(None,), copy_slots=(None, None), get_cpu_slots=(None,),
                       load_cpu_slots=(None, None), iter_transfer_state_entries=(),
                       plan_extend=(None, []), initial_dense=(0, None),
                       mark_transferred_slots=(None,), layer_tensors=(0,))
        for method, args in readers.items():
            with self.subTest(reader=method), self.assertRaises(Joined):
                result = getattr(pool, method)(*args)
                if inspect.isgenerator(result):
                    list(result)
        self.assertEqual(pool.pside_join.call_count, len(readers))

    def test_runner_join_precedes_decode_graph_replay_and_next_prefill(self):
        for mode in ('decode_graph_replay', 'next_prefill'):
            pool = fake_pool(2)
            rt, side = with_reader(pool)
            graph = NS(include_tail=False, warmed=True, run=Mock())
            side.submit(graph, make_plan(pool), [], (None, None, None),
                        eager=None, policy=())
            side.start_after_forward()
            runner = NS(forward=Mock(side_effect=lambda *a, **k: rt.events.append(mode)))
            original = runner.forward
            install_forward_join(runner, pool)
            runner.forward('batch', reinit_attn_backend=True)
            original.assert_called_once_with('batch', reinit_attn_backend=True)
            self.assertEqual(rt.events[-2:], ['reader_join', mode])
            graph.run.assert_called_once()
            self.assertIsNone(side.pending)

    def test_incomplete_tail_unwarmed_and_rebind_fail_closed(self):
        pool = fake_pool(2)
        rt, side = with_reader(pool)
        plan = make_plan(pool)
        states = [(torch.ones(1), None)] * 2
        for tail, warmed in ((True, True), (False, False)):
            with self.assertRaises(RuntimeError):
                side.submit(NS(include_tail=tail, warmed=warmed), plan, states,
                            (None, None, None), eager=None, policy=())
        graph = NS(include_tail=False, warmed=True, run=Mock())
        side.submit(graph, plan, states, (None, None, None), eager=None, policy=())
        with self.assertRaisesRegex(RuntimeError, 'reused'):
            side.submit(graph, plan, states, (None, None, None), eager=None, policy=())
        side.join()
        self.assertEqual(graph.run.call_count, 1)

    def test_failed_enqueue_never_allows_publication(self):
        pool = fake_pool(2)
        side = PDBatchPublication(pool, NS(launch_after_forward=Mock(side_effect=ValueError('launch'))))
        side.submit(NS(include_tail=False, warmed=True), make_plan(pool), [],
                    (None, None, None), eager=None, policy=())
        with self.assertRaisesRegex(ValueError, 'launch'): side.start_after_forward()
        with self.assertRaisesRegex(RuntimeError, 'not publishable'): side.join()
        self.assertIsNotNone(side.pending)

    def test_PD_native_install_flag_off_and_on_preserves_first_token_and_count(self):
        outputs = []
        for enabled in (None, '0', '1'):
            with worker() as w:
                w.runner.forward = Mock()
                if enabled is not None: w.stack.enter_context(patch.dict(os.environ, {FLAG: enabled}))
                w.pool.pside_join = MethodType(native.FactoredGDNPool.pside_join, w.pool)
                backend = NS(forward_metadata=NS(factored_extend=w.plan))
                w.stack.enter_context(forward_context(ForwardContext(
                    attn_backend=NS(linear_attn_backend=backend))))
                ids = torch.arange(8)
                def core(ids, positions, fb, **kwargs):
                    for lid in IDS:
                        dense = torch.full((1, 2, 16, 16), float(lid))
                        native.FactoredGDNPool.commit_extend_batched(w.pool, lid, w.plan, dense)
                    return NS(next_token_logits=torch.stack((ids.float(), -ids.float()), -1))
                w.owner.model.forward = core
                init_graphs(w.runner, w.capture)
                side = getattr(w.pool, '_pd_batch_publication', None)
                self.assertEqual(side is not None, enabled == '1')
                if side is not None: side.runtime = Runtime()
                w.pool._agg_prefill_graph.run = Mock(side_effect=lambda *a, **k:
                    w.pool.count[:, 1].fill_(w.pool.cfg.r))
                fb = NS(_pfactor_agg_contract=True, batch_size=1, forward_mode=ForwardMode.EXTEND,
                        input_ids=ids, extend_seq_lens_cpu=[8], spec_info=None)
                output = w.owner.forward(ids, ids, fb)
                if side is not None:
                    self.assertEqual(w.pool._agg_prefill_graph.run.call_count, 0)
                req = NS(_pfactor_agg_contract=True, kv=NS(mamba_pool_idx=torch.tensor(1)))
                FactorStateHandoff(w.pool).before_send(req)
                self.assertEqual(w.pool._agg_prefill_graph.run.call_count, 1)
                self.assertEqual(req.factored_prefill_boundary_steps, 0)
                outputs.append((output.next_token_logits.clone(), w.pool.count.clone()))
        for values in outputs[1:]:
            for a, b in zip(outputs[0], values): self.assertTrue(torch.equal(a, b))

    def test_AGG_ignores_PD_only_flag(self):
        with agg_worker() as w, patch.dict(os.environ, {FLAG: '1'}):
            init_graphs(w.runner, w.capture)
            self.assertFalse(hasattr(w.pool, '_pd_batch_publication'))

    def test_host_transfer_counts_constant_with_rows_and_full_ring(self):
        ledger = []
        original_tolist = torch.Tensor.tolist
        for full in (False, True):
            for rows in (1, 2, 4, 8):
                pools = [_pool(native, flag) for flag in (False, True)]
                for pool in pools:
                    pool.prefix_valid.fill_(1); pool.dense_required.zero_()
                    pool.stale.fill_(1); pool.dense_of.fill_(-1)
                    pool.ring_owner = list(range(len(pool.ring_owner))) if full else [-1] * len(pool.ring_owner)
                transfers = []
                with patch.object(torch.Tensor, 'tolist', lambda x: (
                        transfers.append(x.numel()), original_tolist(x))[1]), \
                     patch.object(pools[1], '_upload_plan_lists', wraps=pools[1]._upload_plan_lists) as upload:
                    got = pools[1].plan_extend(torch.arange(rows), [64] * rows,
                            prefix_lens=[0] * rows, prompt_final=[True] * rows)
                expected = pools[0].plan_extend(torch.arange(rows), [64] * rows,
                            prefix_lens=[0] * rows, prompt_final=[True] * rows)
                self.assertEqual(len(transfers), 1, (full, rows, transfers))
                upload.assert_called_once()
                for name in ('slots', 'use_ring', 'ring_src', 'ring_dst', 'ring_dst_rows',
                             'dense_required_after_commit', 'use_prefix'):
                    self.assertTrue(torch.equal(getattr(expected, name), getattr(got, name)), (full, rows, name))
                for name in POOL_FIELDS:
                    self.assertTrue(torch.equal(getattr(pools[0], name), getattr(pools[1], name)))
                self.assertEqual(pools[0].ring_owner, pools[1].ring_owner)
                ledger.append(dict(rows=rows, full_ring=full, D2H_calls=1, H2D_plan_calls=1,
                    metadata_bytes=transfers[0] * 8, calls_added_per_row=0, milliseconds='GPU trace required'))
        print('PD054_COST_MODEL', json.dumps(ledger))


if __name__ == '__main__':
    unittest.main()
