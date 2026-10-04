"""CPU admission for #054b. Native Mooncake queue/worker + byte-copy engine.

CUDA events are controllable CPU events; CUDA scheduling/performance and model
NLL remain GPU gates. Run with the same twinstar_sgl helper as the P48 tests.
"""
import copy
import ctypes
import os
import queue
import threading
import unittest
from collections import defaultdict
from contextlib import contextmanager
from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
from test_flash_next_pd_publication import (
    ArithmeticGraph, BatchCollector, FactorStateHandoff, ForwardMode,
    PDBatchPublication, fake_pool, make_plan, native, torch, worker, init_graphs,
)
from test_gdn_factored_host_sync import _pool
from sglang.srt.disaggregation.state_handoff import FactorTransferFence
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.observability.trace import TraceNullContext
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.mem_cache.gdn_pd_publication import install_forward_join, forward_slot_ids
from sglang.srt.mem_cache.gdn_prefill_batch_graph import PrefillBatchGraph

FLAG = 'SGLANG_GDN_PD_PUBLISH_JOIN_OFFLOAD'


class Event:
    def __init__(self, operation=None, ready=False):
        self.operation = operation
        self.ready = threading.Event()
        self.entered = threading.Event()
        self.wait_threads = []
        if ready:
            self.ready.set()

    def complete(self):
        if not self.ready.is_set():
            if self.operation is not None:
                self.operation()
            self.ready.set()

    def record(self):
        self.complete()

    def synchronize(self):
        self.wait_threads.append(threading.get_ident())
        self.entered.set()
        if not self.ready.wait(5):
            raise RuntimeError('CPU test event timed out')

    def query(self):
        return self.ready.is_set()


class Runtime:
    def __init__(self):
        self.events = []
        self.tickets = []

    def launch_after_forward(self, operation, inputs):
        self.events.append('launch')
        ticket = Event(operation)
        self.tickets.append(ticket)
        return ticket

    def join(self, ticket):
        # Simulates device wait ordering, not a real host event wait.
        self.events.append('join')
        ticket.complete()


def publication(pool, enabled=True):
    pool.pside_join = MethodType(native.FactoredGDNPool.pside_join, pool)
    pool.cfg.strict_chunk = True
    rt = Runtime()
    pub = pool._pd_batch_publication = PDBatchPublication(pool, rt, offload_join=enabled)
    return rt, pub


def submit(pub, slots, operation):
    plan = NS(slots=torch.tensor(sorted(slots)), ring_dst=None,
              dense_required_after_commit=None)
    graph = NS(include_tail=False, warmed=True,
               run=lambda *a, launch_replay=None, **kw: operation() if launch_replay is None else launch_replay(operation))
    with pub.forward_scope(frozenset(slots)):
        pub.submit(graph, plan, [], (None, None, None), eager=None, policy=())
        pub.start_after_forward()
    return pub.ticket


class StopWorker(BaseException):
    pass


class WorkerQueue(queue.Queue):
    def get(self):
        item = super().get()
        if item is None:
            raise StopWorker()
        return item


class Transport:
    """Real sender -> add_transfer_request -> worker -> native MAMBA offsets.

    Only the network engine is replaced by memmove over registered CPU tensors.
    No socket, CUDA device, Mooncake engine, or service startup is needed.
    """
    def __init__(self, pool, slot=1):
        self.sent = []
        self.errors = []
        self.finished = threading.Event()
        self.queue = WorkerQueue()
        self.source = [t[li] for li in range(len(pool.layer_ids))
                       for t in (pool.a, pool.U, pool.W, pool.count)]
        self.dest = [torch.full_like(t, -7) for t in self.source]
        self.slot = slot
        m = self.manager = NS(
            disaggregation_mode=DisaggregationMode.PREFILL,
            request_status={7: KVPoll.Transferring},
            _staging_outstanding=defaultdict(int),
            transfer_queues=[self.queue], enable_trace=False, enable_staging=False,
            enable_deferred_decode_kv_release=False, flashnext_staging=None,
            enable_all_cp_ranks_for_transfer=False, is_dummy_cp_rank=False,
            session_lock=threading.Lock(), failed_sessions=set(), session_failures=defaultdict(int),
            pp_size=1, attn_tp_size=1, is_mla_backend=False, is_hybrid_mla_backend=False,
            bootstrap_port=99, req_to_decode_prefix_len={},
            kv_args=NS(kv_data_ptrs=[1]),
        )
        req = NS(room=7, mooncake_session_id='peer:5', is_dummy=False,
                 dst_kv_indices=np.array([1]), dst_device_kv_indices=None,
                 required_dst_info_num=1, endpoint='host', dst_port=5)
        m.transfer_infos = {7: {'peer:5': req}}
        m.decode_kv_args_table = {'peer:5': NS(
            flashnext_staging=False, requires_dcp_relayout=False,
            dst_attn_tp_size=1, dst_kv_ptrs=[1], dst_tp_rank=0,
            dst_kv_item_len=1, dst_kv_layer_ids=[], dst_aux_ptrs=[1])}
        m.check_status = lambda room: m.request_status[room]
        m._prefill_unique_rank = lambda: 0
        m._get_dsa_cache_transfer_skip_flags = lambda target: (False, False)
        m.send_kvcache_slice = lambda *a, **kw: self.mark('kv')
        m.send_kvcache = m.send_kvcache_slice
        m.send_aux = lambda *a: self.mark('aux')

        def conclude_transfer(*, bootstrap_room, status, **kwargs):
            m.request_status[bootstrap_room] = status
            self.finished.set()
        m.conclude_transfer = conclude_transfer
        m.conclude_failure = lambda **kw: conclude_transfer(status=KVPoll.Failed, **kw)

        def transfer(session, sources, destinations, lengths):
            self.mark('state')
            for src, dst, length in zip(sources, destinations, lengths):
                ctypes.memmove(dst, src, length)
            return 0
        m.engine = NS(batch_transfer_sync=transfer)
        m._transfer_data = MethodType(MooncakeKVManager._transfer_data, m)
        m.add_transfer_request = MethodType(MooncakeKVManager.add_transfer_request, m)
        m.maybe_send_extra = lambda req, indices, executor, target: MooncakeKVManager._send_mamba_state(
            m, req, [slot], [t.data_ptr() for t in self.source],
            [t[0].numel() * t.element_size() for t in self.source],
            [t.data_ptr() for t in self.dest], [slot])
        s = self.sender = NS(kv_mgr=m, bootstrap_room=7, aux_index=0,
                            trace_ctx=TraceNullContext())
        s._prepare_send_indices = lambda indices, state: (indices, slice(None), True, False)
        s._record_transfer_indices = Mock()
        s.set_state_handoff_fence = MethodType(MooncakeKVSender.set_state_handoff_fence, s)
        self.thread = None

    def mark(self, what):
        self.sent.append(what)
        return 0

    def enqueue(self):
        with patch.object(torch.cuda, 'Event', Event):
            MooncakeKVSender.send(self.sender, np.array([1]), state_indices=[[self.slot]])

    def start(self):
        def run():
            try:
                MooncakeKVManager.transfer_worker(self.manager, self.queue, None)
            except StopWorker:
                pass
            except BaseException as error:
                self.errors.append(error)
                self.finished.set()
        self.thread = threading.Thread(target=run)
        self.thread.start()

    def stop(self):
        self.queue.put(None)
        self.thread.join(6)
        assert not self.thread.is_alive(), 'worker did not stop'
        assert not self.errors, self.errors

    def received(self):
        return [t[self.slot].contiguous().view(torch.uint8).numpy().tobytes() for t in self.dest]


def request(sender, slot=1, steps=0):
    return NS(disagg_kv_sender=sender, kv=NS(mamba_pool_idx=torch.tensor(slot)),
              factored_prefill_boundary_steps=steps, _pfactor_agg_contract=True)


class PublishJoinTest(unittest.TestCase):
    def test_native_graph_binds_before_later_forward_reuses_source_outputs(self):
        pool = fake_pool(2)
        _, pub = publication(pool)
        graph = PrefillBatchGraph(include_tail=False)
        graph.warmed = True
        states = [(torch.ones(1), None)] * 2
        snapshots, observed = [], []
        def bind(plan, values, *controls):
            snapshots[:] = [value.clone() for value, _ in values]
        buffers = NS(bind=bind)
        graph.entries[graph.key(1, None, None, (), False)] = (
            buffers, NS(replay=lambda: observed.extend(t.clone() for t in snapshots)))
        with pub.forward_scope(frozenset({1})):
            pub.submit(graph, make_plan(pool), states, (None, None, None), eager=None, policy=())
            with patch('sglang.srt.mem_cache.gdn_prefill_joint.enabled', return_value=False):
                pub.start_after_forward()
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(observed, [])
        states[0][0].fill_(99)  # Next runner reuses its static output.
        pub.ticket.complete()
        self.assertTrue(all(torch.equal(value, torch.ones(1)) for value in observed))

    def test_worker_count_read_uses_its_own_device_stream_after_both_events(self):
        order = []
        active = []
        class Count:
            is_cuda = True
            device = 'cuda:2'
            def __getitem__(self, key):
                self.assert_ready = len(order) == 4 and active == ['worker']
                order.append('count')
                return self
            def __eq__(self, other): return self
            def all(self): return self
            def item(self): return self.assert_ready
        @contextmanager
        def stream_context(stream):
            active.append(stream)
            try:
                yield
            finally:
                active.pop()
        def event(name):
            return NS(synchronize=lambda: order.append(name + '_wait'),
                      query=lambda: (order.append(name + '_query') or True))
        with patch.object(torch.cuda, 'Stream', return_value='worker') as create, \
             patch.object(torch.cuda, 'stream', side_effect=stream_context):
            FactorTransferFence(event('publication'), Count(), 1, 8).wait(event('producer'))
        create.assert_called_once_with(device='cuda:2')
        self.assertEqual(order, ['publication_wait', 'publication_query',
                                 'producer_wait', 'producer_query', 'count'])

    def test_scheduler_handoff_does_not_wait_or_read_count_worker_blocks_before_any_send(self):
        pool = fake_pool(2)
        rt, pub = publication(pool)
        done = submit(pub, {1}, lambda: pool.count[:, 1].fill_(pool.cfg.r))
        wire = Transport(pool)
        with patch.object(torch.Tensor, 'item', side_effect=AssertionError('scheduler count read')):
            FactorStateHandoff(pool).before_send(request(wire.sender))
            wire.enqueue()
        self.assertEqual(rt.events, ['launch'])
        wire.start()
        try:
            self.assertTrue(done.entered.wait(2))
            self.assertEqual(wire.sent, [])
            self.assertFalse(wire.finished.is_set())
            done.complete()
            self.assertTrue(wire.finished.wait(2))
            self.assertEqual(wire.sent, ['kv', 'state', 'aux'])
            self.assertEqual(wire.manager.request_status[7], KVPoll.Success)
            self.assertEqual(done.wait_threads, [wire.thread.ident])
        finally:
            done.complete()
            wire.stop()

    def test_incomplete_event_failed_wait_missing_producer_and_bad_count_never_send(self):
        for failure in ('incomplete', 'wait_error', 'missing_producer', 'count'):
            with self.subTest(failure=failure):
                pool = fake_pool(2)
                _, pub = publication(pool)
                done = submit(pub, {1}, lambda: pool.count[:, 1].fill_(pool.cfg.r))
                done.complete()
                if failure == 'incomplete':
                    done.query = lambda: False
                elif failure == 'wait_error':
                    done.synchronize = Mock(side_effect=RuntimeError('bad CUDA event'))
                elif failure == 'count':
                    pool.count[1, 1, 0] = pool.cfg.r - 1
                wire = Transport(pool)
                FactorStateHandoff(pool).before_send(request(wire.sender))
                wire.enqueue()
                if failure == 'missing_producer':
                    wire.queue.queue[0].wait_event = None
                wire.start()
                try:
                    self.assertTrue(wire.finished.wait(2))
                finally:
                    wire.stop()
                self.assertEqual(wire.sent, [])
                self.assertEqual(wire.manager.request_status[7], KVPoll.Failed)
                self.assertEqual(dict(wire.manager._staging_outstanding), {})

    def test_wire_bytes_equal_for_real_factorization_default_off_and_offload(self):
        for rows in (1, 2, 4, 8):
            torch.manual_seed(100 + rows)
            source = fake_pool(2, width=32)
            states = [(torch.randn(rows, 2, 32, 32), torch.randn(1, 2, 32, 32))
                      for _ in source.layer_ids]
            payloads = []
            for enabled in (False, True):
                pool = copy.deepcopy(source)
                rt, pub = publication(pool, enabled)
                plan = make_plan(pool, rows)
                track_slots = torch.tensor([30])
                with pub.forward_scope(frozenset(range(1, rows + 1)) | {30}):
                    with BatchCollector(pool, plan, graph=ArithmeticGraph(), publication=pub) as collect:
                        for lid, (normal, tracked) in enumerate(states):
                            collect.add(lid, normal, tracked, track_slots, None, None)
                    pub.start_after_forward()
                done = pub.ticket
                received = []
                for slot in range(1, rows + 1):
                    wire = Transport(pool, slot)
                    FactorStateHandoff(pool).before_send(request(wire.sender, slot))
                    wire.enqueue()
                    if enabled:
                        done.complete()
                    wire.start()
                    try:
                        self.assertTrue(wire.finished.wait(2))
                    finally:
                        wire.stop()
                    self.assertEqual(wire.manager.request_status[7], KVPoll.Success)
                    received.append(wire.received())
                payloads.append(received)
            self.assertEqual(payloads[0], payloads[1], rows)

    def test_old_fence_survives_static_graph_reuse_and_later_publication(self):
        pool = fake_pool(2)
        rt, pub = publication(pool)
        first = submit(pub, {1}, lambda: pool.count[:, 1].fill_(pool.cfg.r))
        wire = Transport(pool)
        FactorStateHandoff(pool).before_send(request(wire.sender))
        fence = wire.sender._state_handoff_fence
        second = submit(pub, {2}, lambda: pool.count[:, 2].fill_(pool.cfg.r))
        self.assertIsNot(first, second)
        self.assertIs(fence.publication_done, first)
        self.assertTrue(first.query())
        self.assertFalse(second.query())
        wire.enqueue()
        wire.start()
        try:
            self.assertTrue(wire.finished.wait(2))
        finally:
            wire.stop()
        self.assertEqual(wire.sent, ['kv', 'state', 'aux'])
        self.assertFalse(second.query())

    def test_same_slot_decode_joins_but_disjoint_prefill_and_decode_do_not(self):
        pool = fake_pool(2)
        rt, pub = publication(pool)
        mapping = torch.arange(40)
        runner = NS(req_to_token_pool=NS(req_index_to_mamba_index_mapping=mapping))
        def forward(forward_batch, **kwargs):
            # The eager reader and the graph dispatch must both be ordered.
            pool.pside_join(forward_local=True)
            rt.events.append('forward')
        runner.forward = forward
        install_forward_join(runner, pool)
        for mode, slots, joins in ((ForwardMode.EXTEND, [2], False),
                                  (ForwardMode.DECODE, [2], False),
                                  (ForwardMode.DECODE, [1], True)) * 4:
            if pub.pending is None:
                submit(pub, {1, 30}, lambda: None)
            before = rt.events.count('join')
            batch = NS(forward_mode=mode, req_pool_indices=torch.tensor(slots),
                       _pfactor_agg_contract=True)
            runner.forward(batch, reinit_attn_backend=True)
            self.assertEqual(rt.events.count('join') - before, int(joins))
            self.assertIsNone(pub.forward_slots)
        # A tracked/COW destination is as much a hazard as the live state.
        for field in ('mamba_track_indices', 'mamba_cow_src_indices',
                      'mamba_cow_dst_indices', 'mamba_clear_indices'):
            submit(pub, {30}, lambda: None)
            batch = NS(forward_mode=ForwardMode.EXTEND, req_pool_indices=torch.tensor([2]),
                       _pfactor_agg_contract=True, **{field: torch.tensor([30])})
            runner.forward(forward_batch=batch)
            self.assertIsNone(pub.pending)

    def test_unknown_modes_and_virtual_slot_mapping_keep_global_barrier(self):
        pool = fake_pool(2)
        rt, pub = publication(pool)
        runner = NS(req_to_token_pool=NS(req_index_to_mamba_index_mapping=torch.arange(40)),
                    forward=Mock())
        install_forward_join(runner, pool)
        for changes in ({'_pfactor_agg_contract': False}, {'spec_info': object()},
                        {'can_run_tbo': True}, {'tbo_split_seq_index': 1}):
            batch = NS(forward_mode=ForwardMode.EXTEND, req_pool_indices=torch.tensor([2]),
                       _pfactor_agg_contract=True)
            vars(batch).update(changes)
            submit(pub, {1}, lambda: None)
            runner.forward(batch)
            self.assertIsNone(pub.pending)
        runner.req_to_token_pool.mamba_v2p_table = torch.arange(40)
        self.assertIsNone(forward_slot_ids(runner, batch))

    def test_external_cache_reader_and_recycle_join_only_conflicting_slots(self):
        pool = fake_pool(2)
        rt, pub = publication(pool)
        submit(pub, {1, 30}, lambda: None)
        pool.pside_join(torch.tensor([2, 3]))
        self.assertEqual(rt.events, ['launch'])
        pool.pside_join((torch.tensor([3]), torch.tensor([30])))
        self.assertEqual(rt.events, ['launch', 'join'])
        submit(pub, {1}, lambda: None)
        pool.pside_join()  # Unknown whole-pool reader is always conservative.
        self.assertIsNone(pub.pending)

    def test_pending_metadata_is_not_read_and_dense_ring_rows_cannot_be_evicted(self):
        pool = _pool(native, True)
        _, pub = publication(pool)
        pool.stale.fill_(1)
        pool.dense_of.fill_(-1)
        pool.dense_required.zero_()
        pool.prefix_valid.fill_(1)
        pool.ring_owner = [1, 3, 4, 5]
        pool.ring_lru = [0, 1, 2, 3]
        submit(pub, {1}, lambda: None)
        selected = []
        original = torch.Tensor.index_select
        def select(tensor, dim, index):
            if any(tensor is t for t in (pool.stale, pool.dense_of, pool.dense_required, pool.prefix_valid)):
                selected.append(index.tolist())
            return original(tensor, dim, index)
        with pub.forward_scope(frozenset({2})), patch.object(torch.Tensor, 'index_select', select):
            plan = pool.plan_extend(torch.tensor([2]), [64], prefix_lens=[0], prompt_final=[True])
        self.assertTrue(selected)
        self.assertTrue(all(1 not in indices for indices in selected))
        self.assertNotEqual(plan.ring_dst.item(), 0)
        self.assertEqual(pool.ring_owner[0], 1)
        self.assertIsNotNone(pub.pending)

    def test_context_failure_restores_scope_and_publication_failure_stays_closed(self):
        pool = fake_pool(2)
        _, pub = publication(pool)
        submit(pub, {1}, lambda: None)
        with self.assertRaisesRegex(ValueError, 'forward failed'):
            with pub.forward_scope(frozenset({2})):
                raise ValueError('forward failed')
        self.assertIsNone(pub.forward_slots)
        pub.failed = ValueError('launch failed')
        with self.assertRaisesRegex(RuntimeError, 'not publishable'):
            FactorStateHandoff(pool).before_send(request(Transport(pool).sender))

    def test_fallback_request_keeps_original_join_and_count_validation(self):
        pool = fake_pool(2)
        rt, pub = publication(pool)
        submit(pub, {1}, lambda: pool.count[:, 1].fill_(pool.cfg.r))
        req = request(NS())
        req._pfactor_agg_contract = False
        FactorStateHandoff(pool).before_send(req)
        self.assertEqual(rt.events, ['launch', 'join'])
        pool.count[:, 1].zero_()
        with self.assertRaisesRegex(RuntimeError, 'before final prefill'):
            FactorStateHandoff(pool).before_send(req)

    def test_invalid_boundary_missing_slot_and_unsupported_transport_fail_before_queue(self):
        pool = fake_pool(2)
        _, pub = publication(pool)
        submit(pub, {1}, lambda: None)
        for req in (request(NS()), request(NS(), steps=2)):
            with self.assertRaises(RuntimeError):
                FactorStateHandoff(pool).before_send(req)
        req = request(NS())
        req.kv.mamba_pool_idx = None
        with self.assertRaisesRegex(RuntimeError, 'without a mamba slot'):
            FactorStateHandoff(pool).before_send(req)

    def test_native_install_new_flag_defaults_off_and_prerequisites_fail_closed(self):
        for value in (None, '0', '1'):
            with self.subTest(value=value), worker() as w:
                w.runner.forward = Mock()
                w.pool.host_sync_free = True
                env = {'SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED': '1'}
                if value is not None:
                    env[FLAG] = value
                with patch.dict(os.environ, env):
                    init_graphs(w.runner, w.capture)
                self.assertEqual(w.pool._pd_batch_publication.offload_join, value == '1')
        for host_sync, deferred in ((False, '1'), (True, '0')):
            with worker() as w:
                w.runner.forward = Mock()
                w.pool.host_sync_free = host_sync
                with patch.dict(os.environ, {FLAG: '1', 'SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED': deferred}):
                    with self.assertRaises(ValueError):
                        init_graphs(w.runner, w.capture)


if __name__ == '__main__':
    unittest.main()
