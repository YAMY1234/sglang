"""Actual Endpoint and native transfer worker; model only device scheduling/DMA."""
import ast
from collections import defaultdict, deque
import contextlib
import ctypes
import logging
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('TRITON_INTERPRET', '1')
import numpy as np
import torch

from test_flashnext_staging import StagingTest
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.flashnext_staging import _RESERVES, Storage
from sglang.srt.disaggregation.flashnext_staging_manifest import LeasePool
from sglang.srt.disaggregation.flashnext_staging_transport import Endpoint, HEADER, StagingCancelled


def native(name, filename, cls):
    path = Path(__file__).parents[2] / 'python/sglang/srt/disaggregation' / filename
    tree = ast.parse(path.read_text())
    body = next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(time=time, KVPoll=KVPoll, logger=logging.getLogger(__name__))
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    return scope[name], scope


class Event:
    def __init__(self, **kw): self.t = 0
    def record(self): self.t = time.perf_counter()
    def synchronize(self): pass
    def elapsed_time(self, end): return (end.t-self.t)*1000


class Stream:
    def __init__(self, **kw): self.drained = False
    def wait_event(self, event): pass
    def synchronize(self): self.drained = True


class CancellationTest(unittest.TestCase):
    def fixture(self, stack):
        for name, value in dict(Event=Event, Stream=Stream, set_device=lambda _: None,
                                stream=lambda _: contextlib.nullcontext()).items():
            stack.enter_context(patch.object(torch.cuda, name, value))
        endpoints=[]
        for seed in (101, 102):
            catalog=StagingTest().make_catalog(seed)
            storage=Storage([torch.empty(32768,dtype=torch.uint8)], LeasePool(slots=1,slot_bytes=32768),None,32768)
            manager=SimpleNamespace(kv_args=SimpleNamespace(gpu_id=0,flashnext_staging_catalog=catalog),
                enable_staging=False,dcp_size=1,pp_size=1,enable_deferred_decode_kv_release=True,
                engine=SimpleNamespace(batch_register=lambda *args:0),attn_tp_rank=0,attn_tp_size=2,
                local_ip='127.0.0.1',rank_port=19,get_session_id=lambda:'source',flashnext_staging_metrics=deque())
            stack.enter_context(patch.dict(_RESERVES,{0:storage}))
            endpoints.append(Endpoint(manager))
        p,d=endpoints
        p._send=lambda ip,port,parts:d.on_message([HEADER,*parts])
        d._send=lambda ip,port,parts:p.on_message([HEADER,*parts])
        return p,d

    def run_cancel(self, when):
        with contextlib.ExitStack() as stack:
            p,d=self.fixture(stack);pm=p.manager
            pages=np.asarray([1,2]);states=[[[2]],[3],pages]
            def register(room):d.register_room(room=room,kv_indices=pages,state_indices=states,prefix=0)
            register(19)
            base_send=p._send
            def send(ip,port,parts):
                if parts[0]==b'RESERVE' and b':19:' in parts[1] and when in ('before_reserve','after_clear'):
                    d.mark_aborted(19)
                    if when=='after_clear':d.clear_room(19)
                base_send(ip,port,parts)
            p._send=send
            bulk=[]
            def transfer(session,blocks):
                room=next(iter(d.active.values()))[1].room
                if room==19 and when=='during_bulk':
                    d.mark_aborted(19)
                    # Cancellation notification must not free the DMA target.
                    lease=next(iter(d.active.values()))[1]
                    with self.assertRaises(RuntimeError):d.storage.leases.release(lease)
                    with self.assertRaises(RuntimeError):d.clear_room(19)
                for a,b,n in blocks:ctypes.memmove(b,a,n)
                bulk.append(room)
                return 0
            pm._transfer_data=transfer
            target=SimpleNamespace(flashnext_staging=True,dst_attn_tp_size=2,requires_dcp_relayout=False,dst_aux_ptrs=[])
            pm.enable_trace=False;pm._staging_outstanding=defaultdict(int);pm.request_status={}
            pm.check_status=pm.request_status.__getitem__;pm.transfer_infos={};pm.session_lock=threading.Lock()
            pm.failed_sessions=set();pm.session_failures=defaultdict(int);pm._prefill_unique_rank=lambda:0
            pm.decode_kv_args_table={'destination':target};pm.flashnext_staging=p
            pm._get_dsa_cache_transfer_skip_flags=lambda target:(False,False)
            pm.send_aux=lambda *args:0;pm.req_to_decode_prefix_len={};pm.bootstrap_port=19
            pm.conclude_failure=lambda *,bootstrap_room,failure_reason:pm.request_status.update({bootstrap_room:KVPoll.Failed})
            pm.conclude_transfer=lambda *,bootstrap_room,status,**kw:pm.request_status.update({bootstrap_room:status})
            acknowledgements=[]
            def drained(room):
                self.assertEqual(pm._staging_outstanding.get(room,0),0)
                self.assertTrue(p.stream.drained)
                if room==19:
                    self.assertTrue(d.abort_drained(room));d.clear_room(room)
                acknowledgements.append(room)
            pm._maybe_ack_drained_abort=drained
            def chunk(room):
                pm.request_status[room]=KVPoll.Transferring
                pm.transfer_infos[room]={'destination':SimpleNamespace(room=room,is_dummy=False,
                    mooncake_session_id='destination',endpoint='127.0.0.1',dst_port=23,dst_kv_indices=pages,
                    dst_device_kv_indices=None,decode_prefix_len=0,required_dst_info_num=1)}
                return TransferKVChunk(room=room,prefill_kv_indices=pages,index_slice=slice(0,2),
                    is_last_chunk=True,prefill_aux_index=0,state_indices=states,num_kv_tokens=65,wait_event=Event())
            class Queue:
                step=0
                def get(self):
                    self.step+=1
                    if self.step==1:return chunk(19)
                    if self.step==2:
                        register(20);return chunk(20)
                    raise SystemExit
            worker,_=native('transfer_worker','mooncake/conn.py','MooncakeKVManager')
            with self.assertRaises(SystemExit):worker(pm,queue=Queue(),executor=None)
            self.assertEqual(pm.request_status,{19:KVPoll.Failed,20:KVPoll.Success})
            self.assertEqual(acknowledgements,[19,20])
            self.assertFalse(pm.failed_sessions);self.assertFalse(pm.session_failures)
            self.assertFalse(d.active);self.assertFalse(pm._staging_outstanding)
            self.assertIn(20,bulk)
            for endpoint in (p,d):
                lease=endpoint.storage.leases.acquire(room=21,nbytes=32768)
                self.assertIsNotNone(lease);endpoint.storage.leases.release(lease)

    def test_cancel_before_reserve_then_next_request(self):self.run_cancel('before_reserve')
    def test_cancel_clear_late_reserve_then_next_request(self):self.run_cancel('after_clear')
    def test_cancel_during_dma_then_next_request(self):self.run_cancel('during_bulk')

    def test_unknown_room_and_wrong_generation_remain_errors(self):
        with contextlib.ExitStack() as stack:
            p,d=self.fixture(stack)
            request=dict(key='unknown',manifest=SimpleNamespace(room=99,nbytes=64),ip='127.0.0.1',port=19)
            d.waiting.append(request);d._grant_waiting()
            with self.assertRaisesRegex(RuntimeError,'room not live'):p._wait('unknown',b'SLOT')
            d.register_room(room=99,kv_indices=np.array([1]),state_indices=[],prefix=0)
            d.waiting.append(request);d._grant_waiting();slot=p._wait('unknown',b'SLOT')
            with self.assertRaisesRegex(ValueError,'generation'):d._scatter('unknown',slot['generation']+1)

    def test_tombstone_expiry_preserves_live_destinations(self):
        with contextlib.ExitStack() as stack:
            p,d=self.fixture(stack)
            with patch('time.monotonic',return_value=0):d.mark_aborted(1);d.mark_aborted(2)
            d.rooms[2]={}
            with patch('time.monotonic',return_value=121):d.mark_aborted(3)
            self.assertNotIn(1,d.aborted);self.assertIn(2,d.aborted);self.assertIn(3,d.aborted)

    def test_cancel_queued_reservation_behind_full_pool(self):
        with contextlib.ExitStack() as stack:
            p,d=self.fixture(stack)
            for room in (1,2,3):
                d.register_room(room=room,kv_indices=np.array([1]),state_indices=[],prefix=0)
                d.waiting.append(dict(key=str(room),manifest=SimpleNamespace(room=room,nbytes=64),ip='127.0.0.1',port=19))
            d._grant_waiting()
            self.assertEqual([r['manifest'].room for r in d.waiting],[2,3])
            d.mark_aborted(3)
            with self.assertRaises(StagingCancelled):p._wait('3',b'SLOT')
            self.assertEqual([r['manifest'].room for r in d.waiting],[2])
            self.assertEqual({lease.room for _,lease in d.active.values()},{1})

    def test_real_receiver_abort_marks_room_before_native_notification(self):
        path=Path(__file__).parents[2]/'python/sglang/srt/disaggregation/mooncake/conn.py'
        tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MooncakeKVReceiver')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='abort')
        order=[]
        class Base:
            def abort(self):order.append('native');return 'native-result'
        cls.bases=[ast.Name(id='Base',ctx=ast.Load())];cls.body=[method];cls.decorator_list=[]
        scope=dict(Base=Base)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(path),'exec'),scope)
        receiver=scope['MooncakeKVReceiver']()
        receiver.bootstrap_room=19
        receiver.kv_mgr=SimpleNamespace(flashnext_staging=SimpleNamespace(mark_aborted=lambda room:order.append(('staging',room))))
        self.assertEqual(receiver.abort(),'native-result')
        self.assertEqual(order,[('staging',19),'native'])
        receiver.kv_mgr.flashnext_staging=None;order.clear()
        self.assertEqual(receiver.abort(),'native-result');self.assertEqual(order,['native'])

    def test_dense_staging_timeout_keeps_destination_until_ack(self):
        resolve,scope=native('resolve_deferred_releases','decode.py','DecodeTransferQueue')
        scope['agree_deferred_releases']=lambda values,group:values
        releases=[];ready=[False]
        manager=SimpleNamespace(flashnext_staging=object(),is_abort_release_safe=lambda *args:ready[0])
        req=SimpleNamespace(req=SimpleNamespace(bootstrap_room=9),kv_receiver=SimpleNamespace(kv_mgr=manager))
        queue=SimpleNamespace(scheduler=SimpleNamespace(req_to_token_pool=SimpleNamespace()),
            _deferred_releases=[(req,0,7,2)],gloo_group=None,deferred_kv_release_timeout=1,
            _do_release=lambda *args:releases.append(args))
        resolve(queue);self.assertFalse(releases);self.assertEqual(len(queue._deferred_releases),1)
        ready[0]=True;resolve(queue);self.assertEqual(len(releases),1);self.assertFalse(queue._deferred_releases)


if __name__=='__main__':unittest.main()
