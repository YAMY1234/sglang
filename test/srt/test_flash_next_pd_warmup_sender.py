"""#054c: 14 CPU contracts for default PD warmup and sender compatibility.

The complete production warmup/create_sender/send_kv_chunk functions execute
with CPU model outputs and in-process HTTP responses. Readiness here proves
startup control flow, not a loaded GPU model, CUDA scheduling, or RDMA. Fake
has no transfer worker; its send is also exercised on a controlled CPU thread.
Mooncake uses the original queue/worker with the existing CPU memmove engine.
"""
import __future__
import ast
import asyncio
import copy
import logging
import os
from pathlib import Path
import subprocess
import threading
import traceback
import unittest
from functools import wraps
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
from test_flash_next_pd_publish_join import (
    Transport, fake_pool, init_graphs, publication, request, submit, torch,
    worker,
)
from test_gdn_factored_host_sync import POOL_FIELDS
from sglang.srt.disaggregation import state_handoff as handoff
from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.fake.conn import FakeKVSender
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVSender
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST, KVClassType, TransferBackend, _is_fake_transfer,
    get_kv_class,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.common import kv_to_page_indices
from sglang.srt.mem_cache.gdn_prefill_agg_contract import install_contracts

ROOT = Path(__file__).resolve().parents[2]
KEYS = (
    'SGLANG_GDN_FACTORED_HOST_SYNC_FREE',
    'SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED',
    'SGLANG_GDN_PD_PUBLISH_JOIN_OFFLOAD',
    'SGLANG_FLASHNEXT_PD_TRUNK_PREFILL_GRAPH',
)
E = dict.fromkeys(KEYS, '0')
DDOUBLE = dict(zip(KEYS, ('1', '1', '1', '0')))


def production_function(relative, name, scope, owner=None):
    """Execute unmodified source functions, retaining production stack lines."""
    path = ROOT / 'python/sglang/srt' / relative
    tree = ast.parse(path.read_text())
    body = tree.body
    if owner is not None:
        body = next(n for n in body if isinstance(n, ast.ClassDef)
                    and n.name == owner).body
    fn = copy.deepcopy(next(n for n in body if getattr(n, 'name', None) == name))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), scope)
    return scope[name]


def fake_sender():
    return FakeKVSender(NS(), FAKE_BOOTSTRAP_HOST, 7, [0], 0)


def native_agg_handoff(pool):
    """Install the real full-N wrapper; the scheduler still sets P31 r+1."""
    def native_track(self, req):
        pass

    @wraps(native_track)
    def legacy_track(self, req):
        return native_track(self, req)

    class Forward:
        @classmethod
        def init_new(cls, *args, **kwargs):
            pass

    class Schedule:
        _mamba_radix_cache_v2_req_prepare_for_extend = legacy_track

    class Backend:
        def init_forward_metadata(self, *args):
            pass

    class Handoff(handoff.FactorStateHandoff):
        @wraps(handoff.FactorStateHandoff.before_send)
        def before_send(self, req):
            return handoff.FactorStateHandoff.before_send(self, req)

    install_contracts(Forward, Schedule, Backend, Handoff)
    return Handoff(pool)


def warmup_case(environment, *, bad_count=False):
    """Default HTTP PD warmup through native scheduler handoff and ready hook."""
    pool = fake_pool(2)
    pool.cfg.strict_chunk = True
    rt = pub = None
    if environment[KEYS[1]] == '1':
        rt, pub = publication(pool, environment[KEYS[2]] == '1')
    state = NS(pool=pool, publication=pub, runtime=rt, requests=[], senders=[],
               payloads=[], killed=Mock(), ready=Mock(), metadata=Mock())
    manager = NS(kv_args=NS(state_types=[StateType.MAMBA]))
    bootstrap = NS(transfer_backend=TransferBackend.MOONCAKE,
                   _check_if_req_exceed_kv_capacity=lambda req: False,
                   _process_req=Mock(), tp_rank=0, pp_rank=0, bootstrap_port=99,
                   kv_manager=manager)
    request_pool = NS(pd_state_handoffs={'factor': native_agg_handoff(pool)},
                      req_to_token=torch.arange(4).reshape(1, 4),
                      req_index_to_mamba_index_mapping=torch.tensor([1]),
                      translate_mamba_indices=lambda slot: slot.reshape(-1))
    scheduler = NS(
        token_to_kv_pool_allocator=NS(
            page_size=1, get_kvcache=lambda: NS(),
            translate_kv_indices_for_transfer=lambda indices: indices),
        req_to_token_pool=request_pool, enable_overlap=False, enable_staging=False,
        disagg_prefill_bootstrap_queue=bootstrap, model_config=NS(),
        disagg_metadata_buffers=NS(set_buf=state.metadata),
        disagg_prefill_pending_chunk_rids=set())
    scope = dict(np=np, torch=torch, TransferBackend=TransferBackend,
                 KVClassType=KVClassType, StateType=StateType,
                 FAKE_BOOTSTRAP_HOST=FAKE_BOOTSTRAP_HOST, get_kv_class=get_kv_class,
                 _is_fake_transfer=_is_fake_transfer, kv_to_page_indices=kv_to_page_indices,
                 get_disagg=lambda: NS(flashnext_pd_staging=False), _is_npu=False)
    create_sender = production_function('disaggregation/prefill.py', 'create_sender',
                                        scope, 'PrefillBootstrapQueue')
    send_chunk = production_function('disaggregation/prefill.py', 'send_kv_chunk',
                                     scope, 'SchedulerDisaggregationPrefillMixin')

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def read(self):
            return b'{}'

    class Session:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def post(self, url, *, json, ssl):
            state.payloads.append(json)
            req = request(None)
            vars(req).update(bootstrap_host=json['bootstrap_host'],
                             bootstrap_room=json['bootstrap_room'],
                             disagg_prefill_dp_rank=None, start_send_idx=0,
                             origin_input_ids=json['input_ids'], rid='cpu-warmup',
                             extend_range=NS(end=4), disagg_decode_prefix_len=0)
            req.kv.req_pool_idx = 0
            state.requests.append(req)
            assert create_sender(bootstrap, req, 1)
            sender = req.disagg_kv_sender
            state.senders.append(sender)
            assert isinstance(sender, FakeKVSender)
            sender.init(4, 0)
            count = pool.cfg.r - int(bad_count)
            if pub is not None:
                done = submit(pub, {1}, lambda: pool.count[:, 1].fill_(count))
                done.complete()  # CPU model has finished its publication.
            else:
                pool.count[:, 1].fill_(count)
            with patch('sglang.srt.model_executor.fullstack_policy.fullstack_enabled',
                       return_value=True):
                send_chunk(scheduler, req, last_chunk=True)
            assert req.factored_prefill_boundary_steps == 0
            assert sender.poll() == KVPoll.Success
            return Response()

    status = NS(Starting='Starting', Up='Up', UnHealthy='UnHealthy')
    tokenizer = NS(server_status=status.Starting)
    serving = NS(api_key=None, skip_tokenizer_init=True, skip_server_warmup=False)
    execution = NS(moe=NS(is_ep_scale_joiner=False))
    model = NS(checkpoint_engine_wait_weights_before_ready=False,
               delete_ckpt_after_loading=False)
    scope = dict(
        asyncio=asyncio, os=os, np=np, envs=envs,
        aiohttp=NS(ClientSession=Session, ClientTimeout=lambda **kw: NS(**kw)),
        FAKE_BOOTSTRAP_HOST=FAKE_BOOTSTRAP_HOST,
        get_parallel=lambda: NS(dp_size=1), get_serving=lambda: serving,
        get_disagg=lambda: NS(disaggregation_mode='prefill', language_only=False,
                              language_model_only=False),
        get_exec=lambda: execution, get_model=lambda: model,
        get_observability=lambda: NS(debug_tensor_dump_input_file=None),
        ssl_verify_of=lambda args: False, is_mps=lambda: False,
        time=NS(sleep=lambda seconds: None),
        requests=NS(get=lambda *a, **kw: NS(status_code=200,
                                           json=lambda: {'is_generation': True})),
        logger=logging.getLogger('pubjoin.cpu.warmup'),
        get_exception_traceback=traceback.format_exc, kill_process_tree=state.killed,
        _global_state=NS(tokenizer_manager=tokenizer), ServerStatus=status,
        _freeze_gc_after_server_warmup=Mock())
    for name in ('_send_disaggregation_warmup_requests', '_execute_server_warmup',
                 '_wait_and_warmup'):
        production_function('entrypoints/http_server.py', name, scope)
    with patch.dict(os.environ, dict(environment, SGLANG_RUST_SERVER='0')):
        scope['_wait_and_warmup'](NS(url=lambda: 'http://cpu-fixture'),
                                  launch_callback=state.ready)
    state.status = tokenizer.server_status
    return state


def field_bytes(pool):
    return {name: (None if getattr(pool, name) is None else
                   getattr(pool, name).contiguous().view(torch.uint8).numpy().tobytes())
            for name in POOL_FIELDS}


class PDWarmupSenderTest(unittest.TestCase):
    def test_default_pd_warmup_ddouble_reaches_server_up_and_ready(self):
        with self.assertLogs('pubjoin.cpu.warmup', level='INFO') as logs:
            state = warmup_case(DDOUBLE)
        self.assertEqual(state.status, 'Up')
        state.ready.assert_called_once_with()
        state.killed.assert_not_called()
        self.assertTrue(any('ready to roll' in line for line in logs.output))
        self.assertEqual(state.payloads[0]['bootstrap_host'], FAKE_BOOTSTRAP_HOST)
        self.assertEqual(state.payloads[0]['sampling_params']['max_new_tokens'], 8)
        self.assertEqual(state.requests[0].factored_prefill_boundary_steps, 0)
        self.assertTrue(state.senders[0].has_sent)
        self.assertEqual(state.runtime.events, ['launch'])
        state.metadata.assert_called_once_with(state.requests[0])

    def test_default_pd_warmup_invalid_count_never_advertises_ready(self):
        with self.assertLogs('pubjoin.cpu.warmup', level='ERROR') as logs:
            state = warmup_case(DDOUBLE, bad_count=True)
        self.assertEqual(state.status, 'Starting')
        state.ready.assert_not_called()
        state.killed.assert_called_once()
        self.assertIn('before final prefill r truncation', '\n'.join(logs.output))
        self.assertEqual(state.senders[0].poll(), KVPoll.Failed)
        self.assertFalse(state.senders[0].has_sent)

    def test_default_pd_warmup_e_reaches_ready_without_a_fence(self):
        with patch.object(FakeKVSender, 'set_state_handoff_fence',
                          side_effect=AssertionError('E attached a fence')):
            state = warmup_case(E)
        self.assertEqual(state.status, 'Up')
        state.ready.assert_called_once_with()
        self.assertIsNone(state.publication)
        self.assertEqual(state.senders[0].poll(), KVPoll.Success)

    def test_native_installer_obeys_all_four_e_and_ddouble_switches(self):
        for values in (E, DDOUBLE):
            with self.subTest(values=values), worker() as w:
                w.runner.forward = Mock()
                w.pool.host_sync_free = values[KEYS[0]] == '1'
                with patch.dict(os.environ, values):
                    init_graphs(w.runner, w.capture)
                pub = getattr(w.pool, '_pd_batch_publication', None)
                if values is E:
                    self.assertIsNone(pub)
                else:
                    self.assertTrue(pub.offload_join)
                self.assertEqual(values[KEYS[3]], '0')

    def test_fake_cpu_done_event_is_complete_and_fence_is_consumed(self):
        sender = fake_sender()
        fence = NS(count=NS(is_cuda=False), wait=Mock())
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
        sender.set_state_handoff_fence(fence)
        sender.send(np.array([1]), [[1]])
        producer = fence.wait.call_args.args[0]
        producer.synchronize()
        self.assertTrue(producer.query())
        fence.wait.assert_called_once()
        self.assertIsNone(sender._state_handoff_fence)
        self.assertEqual(sender.poll(), KVPoll.Success)
        sender.send(np.array([1]), [[1]])
        fence.wait.assert_called_once()

    def test_fake_rejects_duplicate_fence_without_losing_the_first(self):
        sender = fake_sender()
        first = NS(count=NS(is_cuda=False), wait=Mock())
        sender.set_state_handoff_fence(first)
        with self.assertRaisesRegex(RuntimeError, 'attached twice'):
            sender.set_state_handoff_fence(object())
        self.assertIs(sender._state_handoff_fence, first)
        sender.send(np.array([1]), [[1]])
        first.wait.assert_called_once()

    def test_fake_before_send_and_thread_wait_for_publication_and_check_count(self):
        pool = fake_pool(2)
        rt, pub = publication(pool)
        done = submit(pub, {1}, lambda: pool.count[:, 1].fill_(pool.cfg.r))
        sender = fake_sender()
        with patch.object(torch.Tensor, 'item', side_effect=AssertionError('scheduler count')):
            handoff.FactorStateHandoff(pool).before_send(request(sender))
        errors = []
        def send():
            try:
                sender.send(np.array([1]), [[1]])
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=send)
        thread.start()
        try:
            self.assertTrue(done.entered.wait(2))
            self.assertFalse(sender.has_sent)
            self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
        finally:
            done.complete()
            thread.join(6)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(done.wait_threads, [thread.ident])
        self.assertEqual(rt.events, ['launch'])
        self.assertEqual(sender.poll(), KVPoll.Success)

    def test_fake_cuda_producer_record_precedes_fence_wait(self):
        order = []
        producer = NS(record=lambda: order.append('producer_record'))
        fence = NS(count=NS(is_cuda=True), wait=lambda event: order.append(
            'fence_wait' if event is producer else 'wrong_event'))
        sender = fake_sender()
        sender.set_state_handoff_fence(fence)
        with patch.object(torch.cuda, 'Event', return_value=producer) as create:
            sender.send(np.array([1]), [[1]])
        create.assert_called_once_with()
        self.assertEqual(order, ['producer_record', 'fence_wait'])
        self.assertEqual(sender.poll(), KVPoll.Success)

    def test_fake_bad_count_or_event_failure_blocks_success(self):
        for failure in ('count', 'event'):
            with self.subTest(failure=failure):
                pool = fake_pool(2)
                _, pub = publication(pool)
                done = submit(pub, {1}, lambda: pool.count[:, 1].fill_(pool.cfg.r))
                done.complete()
                if failure == 'count':
                    pool.count[1, 1, 0] -= 1
                else:
                    done.synchronize = Mock(side_effect=RuntimeError('bad event'))
                sender = fake_sender()
                handoff.FactorStateHandoff(pool).before_send(request(sender))
                with self.assertRaises(RuntimeError):
                    sender.send(np.array([1]), [[1]])
                self.assertFalse(sender.has_sent)
                self.assertEqual(sender.poll(), KVPoll.Failed)
                self.assertIsNone(sender._state_handoff_fence)

    def test_mooncake_before_send_queue_worker_checks_count_before_any_bytes(self):
        for valid in (True, False):
            with self.subTest(valid=valid):
                pool = fake_pool(2)
                _, pub = publication(pool)
                expected = pool.cfg.r if valid else pool.cfg.r - 1
                done = submit(pub, {1}, lambda: pool.count[:, 1].fill_(expected))
                wire = Transport(pool)
                with patch.object(torch.Tensor, 'item', side_effect=AssertionError('scheduler count')):
                    handoff.FactorStateHandoff(pool).before_send(request(wire.sender))
                    wire.enqueue()
                wire.start()
                try:
                    self.assertTrue(done.entered.wait(2))
                    self.assertEqual(wire.sent, [])
                    done.complete()
                    self.assertTrue(wire.finished.wait(2))
                finally:
                    done.complete()
                    wire.stop()
                self.assertEqual(wire.manager.request_status[7],
                                 KVPoll.Success if valid else KVPoll.Failed)
                self.assertEqual(wire.sent, ['kv', 'state', 'aux'] if valid else [])
                self.assertEqual(done.wait_threads, [wire.thread.ident])

    def test_startup_probe_and_repeated_fallback_warn_once_per_sender_type(self):
        class MissingSender:
            pass
        class NoncallableSender:
            set_state_handoff_fence = None
        for sender_type in (MissingSender, NoncallableSender):
            with self.subTest(sender_type=sender_type):
                scope = dict(envs=envs, is_mla_backend=lambda pool: False,
                             get_kv_class=lambda *args: sender_type,
                             KVClassType=KVClassType, TransferBackend=TransferBackend)
                init = production_function('disaggregation/prefill.py', '__init__',
                                            scope, 'PrefillBootstrapQueue')
                scheduler = NS(tp_worker=NS(model_runner=NS(effective_max_total_num_tokens=4)))
                bootstrap = NS(_init_kv_manager=Mock())
                with self.assertLogs(handoff.logger, level='WARNING') as logs, \
                     patch.dict(os.environ, dict(DDOUBLE, SGLANG_DISAGG_STAGING_BUFFER='0')):
                    init(bootstrap, None, None, None, None, 0, 1, 0, 99,
                         None, 4, scheduler, None, 0, 1, TransferBackend.MOONCAKE)
                    for _ in range(3):
                        pool = fake_pool(2)
                        rt, pub = publication(pool)
                        submit(pub, {1}, lambda: pool.count[:, 1].fill_(pool.cfg.r))
                        handoff.FactorStateHandoff(pool).before_send(request(sender_type()))
                        self.assertEqual(rt.events, ['launch', 'join'])
                self.assertEqual(len(logs.output), 1)
                self.assertIn(sender_type.__name__, logs.output[0])
                self.assertIn('set_state_handoff_fence', logs.output[0])
                self.assertIn('synchronous join', logs.output[0])

    def test_fallback_retains_count_validation_and_rejects_corrupt_layers(self):
        class LegacySender:
            pass
        pool = fake_pool(2)
        rt, pub = publication(pool)
        submit(pub, {1}, lambda: pool.count[:, 1].fill_(pool.cfg.r - 1))
        with patch.object(handoff.logger, 'warning'), \
             self.assertRaisesRegex(RuntimeError, 'before final prefill'):
            handoff.FactorStateHandoff(pool).before_send(request(LegacySender()))
        self.assertEqual(rt.events, ['launch', 'join'])

    def test_e_startup_never_probes_sender_and_supported_types_do_not_warn(self):
        for sender_type in (FakeKVSender, MooncakeKVSender):
            with patch.object(handoff.logger, 'warning') as warning:
                self.assertTrue(handoff.supports_state_handoff_fence(sender_type))
                warning.assert_not_called()
        get_class = Mock(side_effect=AssertionError('E probed sender'))
        scope = dict(envs=envs, is_mla_backend=lambda pool: False,
                     get_kv_class=get_class, KVClassType=KVClassType,
                     TransferBackend=TransferBackend)
        init = production_function('disaggregation/prefill.py', '__init__',
                                    scope, 'PrefillBootstrapQueue')
        scheduler = NS(tp_worker=NS(model_runner=NS(effective_max_total_num_tokens=4)))
        with patch.dict(os.environ, dict(E, SGLANG_DISAGG_STAGING_BUFFER='0')):
            init(NS(_init_kv_manager=Mock()), None, None, None, None, 0, 1, 0, 99,
                 None, 4, scheduler, None, 0, 1, TransferBackend.MOONCAKE)
        get_class.assert_not_called()

    def test_e_four_zero_fields_and_wire_bytes_match_frozen_054b(self):
        relative = 'python/sglang/srt/disaggregation/state_handoff.py'
        baseline = os.getenv('PUBJOIN_BASELINE_ROOT')
        source = ((Path(baseline) / relative).read_text() if baseline else
                  subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                           '4ef95092e1a7bec3ea434111b15616552f82517c:' + relative],
                                          text=True))
        namespace = {}
        exec(compile(source, 'frozen-054b/' + relative, 'exec'), namespace)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                source_pool = fake_pool(2)
                source_pool.cfg.strict_chunk = True
                for name in ('a', 'U', 'W'):
                    setattr(source_pool, name, getattr(source_pool, name).to(dtype))
                source_pool.count[:, 1].fill_(source_pool.cfg.r)
                observed = []
                for handler_cls in (namespace['FactorStateHandoff'], handoff.FactorStateHandoff):
                    with patch.dict(os.environ, E):
                        pool = copy.deepcopy(source_pool)
                        before = field_bytes(pool)
                        wire = Transport(pool)
                        wire.sender.set_state_handoff_fence = Mock(
                            side_effect=AssertionError('E attached a fence'))
                        handler_cls(pool).before_send(request(wire.sender))
                        wire.enqueue()
                        wire.start()
                        try:
                            self.assertTrue(wire.finished.wait(2))
                        finally:
                            wire.stop()
                        self.assertEqual(wire.manager.request_status[7], KVPoll.Success)
                        self.assertEqual(field_bytes(pool), before)
                        observed.append((field_bytes(pool), wire.received()))
                        pool.count[1, 1, 0] -= 1
                        with self.assertRaisesRegex(RuntimeError, 'before final prefill'):
                            handler_cls(pool).before_send(request(wire.sender))
                self.assertEqual(observed[0], observed[1])


if __name__ == '__main__':
    unittest.main()
