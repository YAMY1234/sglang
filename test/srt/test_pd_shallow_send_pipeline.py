"""P12 CPU protocol gate. Real worker/Endpoint/catalog/copy; modeled CUDA/DMA.

The live D/backend gate and timing gates remain integration-owned. Frozen source
bytes avoid attributing upstream QSA variability to this transport schedule.
"""
import os
os.environ.setdefault('TRITON_INTERPRET', '1')

import contextlib
import ctypes
from collections import defaultdict, deque
from pathlib import Path
import threading
import time
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch

import test_flashnext_staging as staging_fixture
from test_flash_next_pd_publish_join import Event, StopWorker, WorkerQueue, Transport
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.flashnext_staging import Entry, Storage, _RESERVES
from sglang.srt.disaggregation.flashnext_staging_kernels import copy_payload
from sglang.srt.disaggregation.flashnext_staging_manifest import LeasePool
from sglang.srt.disaggregation.flashnext_staging_transport import Endpoint, HEADER
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
from sglang.srt.disaggregation.pd_shallow_send import staged_kv_phases
from sglang.srt.disaggregation.state_handoff import FactorTransferFence, FactorStateHandoff


class CudaEvent(Event):
    def __init__(self, **kwargs):
        super().__init__()
        self.t = 0

    def record(self):
        self.t = time.perf_counter()
        self.complete()

    def elapsed_time(self, end):
        return (end.t - self.t) * 1000


class Stream:
    def __init__(self, **kwargs): pass
    def wait_event(self, event): event.synchronize()
    def synchronize(self): pass


class P12Test(unittest.TestCase):
    def catalog(self, seed):
        catalog = staging_fixture.StagingTest().make_catalog(seed)
        generator = torch.Generator().manual_seed(seed)
        for i, (name, shape, dtype) in enumerate((
            ('gdn_factored_a', (8, 2, 16), torch.float32),
            ('gdn_factored_U', (8, 2, 4, 16), torch.float16),
            ('gdn_factored_W', (8, 2, 4, 16), torch.float16),
            ('pd_boundary_inputs', (8, 19), torch.float32),
            ('pd_boundary_valid', (8, 1), torch.int32))):
            tensor = torch.randint(0, 90, shape, generator=generator).to(dtype)
            catalog._add(Entry(0, 'mamba.' + name + '.0', tensor, 0, i + 2, 0))
        return catalog

    def run_staged(self, kv_first, failure=None):
        p, d = self.catalog(71), self.catalog(72)
        src, dst = np.array([11, 4, 8, 2, 9]), np.array([3, 10, 1, 7, 5])
        maps = np.stack((src, src[::-1], src, src[::-1]))
        states, dst_states = [[[2]], [3], maps[:2]], [[[5]], [6], dst]
        count = p.by_key[0, 'mamba.gdn_factored_count.0'].tensor
        valid = p.by_key[0, 'mamba.pd_boundary_valid.0'].tensor
        count[2].fill_(-5); valid[2].zero_()
        publication = Event(lambda: (count[2].fill_(15 if failure == 'count' else 16), valid[2].fill_(1)))
        producer = CudaEvent(); producer.record()
        if failure == 'producer': producer.query = lambda: False
        fence = FactorTransferFence(publication, count.unsqueeze(0), torch.tensor(2), 16,
                                    boundary_valid=valid, kv_first=kv_first)
        sent, errors, received, early = [], [], [], threading.Event()
        finished = threading.Event()

        def manager(catalog):
            return NS(kv_args=NS(gpu_id=0, flashnext_staging_catalog=catalog, kv_data_ptrs=[1]),
                enable_staging=False, dcp_size=1, pp_size=1, enable_deferred_decode_kv_release=True,
                engine=NS(batch_register=lambda *a: 0), attn_tp_rank=0, attn_tp_size=2,
                local_ip='127.0.0.1', rank_port=19, get_session_id=lambda: 'source',
                flashnext_staging_metrics=deque())
        pm, dm = manager(p), manager(d)
        def storage():
            return Storage([torch.full((16384,), 213, dtype=torch.uint8) for _ in range(2)],
                           LeasePool(slots=2, slot_bytes=16384), None, 32768)
        ps, ds = storage(), storage()
        initial = {key: entry.tensor.clone() for key, entry in d.by_key.items()}
        with contextlib.ExitStack() as stack:
            for name, value in dict(Event=CudaEvent, Stream=Stream, set_device=lambda *a: None,
                                    stream=lambda *a: contextlib.nullcontext()).items():
                stack.enter_context(patch.object(torch.cuda, name, value))
            stack.enter_context(patch.dict(_RESERVES, {0: ps}))
            pe = Endpoint(pm); _RESERVES[0] = ds; de = Endpoint(dm)
            pe._send = lambda ip, port, parts: de.on_message([HEADER, *parts])
            de._send = lambda ip, port, parts: pe.on_message([HEADER, *parts])
            de.register_room(room=19, kv_indices=dst, state_indices=dst_states, prefix=0)

            def transfer(session, blocks):
                request, lease = next(iter(de.active.values()))
                manifest = request['manifest']
                early_keys, _, early_ranges, _ = staged_kv_phases(manifest)
                if not publication.query():
                    # The oracle catches a deliberate attempt to write any
                    # factor/h31/boundary byte before the original fence.
                    base = ds.buffers[lease.slot].data_ptr()
                    self.assertTrue(kv_first)
                    self.assertEqual([(b - base, n) for _, b, n in blocks], list(early_ranges))
                    self.assertFalse(fence.validated)
                    self.assertTrue(all(torch.equal(entry.tensor, initial[key]) for key, entry in d.by_key.items()))
                for a, b, n in blocks: ctypes.memmove(b, a, n)
                sent.append(('kv' if not publication.query() else 'final', sum(n for _, _, n in blocks)))
                if not publication.query(): early.set()
                return 0
            pm._transfer_data = transfer
            target = NS(flashnext_staging=True, dst_attn_tp_size=2, requires_dcp_relayout=False,
                        dst_aux_ptrs=[])
            request = NS(room=19, is_dummy=False, mooncake_session_id='destination',
                endpoint='127.0.0.1', dst_port=23, dst_kv_indices=dst, dst_device_kv_indices=None,
                decode_prefix_len=0, required_dst_info_num=1)
            chunk = TransferKVChunk(room=19, prefill_kv_indices=src, index_slice=slice(0, 5),
                is_last_chunk=True, prefill_aux_index=0, state_indices=states, num_kv_tokens=257,
                prefill_kv_indices_by_entry=maps, wait_event=producer, state_handoff_fence=fence)
            pm.enable_trace=False; pm._staging_outstanding=defaultdict(int)
            pm.request_status={19:KVPoll.Transferring}; pm.check_status=pm.request_status.__getitem__
            pm.transfer_infos={19:{'destination':request}}; pm.session_lock=threading.Lock()
            pm.failed_sessions=set(); pm._prefill_unique_rank=lambda:0
            pm.decode_kv_args_table={'destination':target}; pm.flashnext_staging=pe
            pm._get_dsa_cache_transfer_skip_flags=lambda target:(False,False)
            pm.send_aux=lambda *a: (sent.append(('aux', 0)) or 0)
            pm.req_to_decode_prefix_len={}; pm.bootstrap_port=19
            def drained(room):
                if pm.request_status.get(room) == KVPoll.Failed and not pm._staging_outstanding.get(room, 0):
                    self.assertTrue(de.abort_drained(room))
            pm._maybe_ack_drained_abort=drained
            def conclude(*,bootstrap_room,status,**kwargs):
                if status == KVPoll.Success:
                    self.assertFalse(de.active)
                    self.assertTrue(fence.validated)
                pm.request_status[bootstrap_room]=status
                finished.set()
            pm.conclude_transfer=conclude
            pm.conclude_failure=lambda **kw: conclude(status=KVPoll.Failed, **kw)
            work = WorkerQueue(); work.put(chunk); work.put(None)
            def run():
                try: MooncakeKVManager.transfer_worker(pm, work, None)
                except StopWorker: pass
                except BaseException as error: errors.append(error); finished.set()
            thread=threading.Thread(target=run); thread.start()
            try:
                if failure == 'producer':
                    self.assertTrue(finished.wait(2))
                else:
                    self.assertTrue(publication.entered.wait(2))
                    if kv_first:
                        self.assertTrue(early.is_set())
                        self.assertTrue(de.active, 'D lease must remain held before READY')
                    else:
                        self.assertFalse(sent)
                    if failure == 'abort': pm.request_status[19]=KVPoll.Failed
                    self.assertFalse(finished.is_set())
                    publication.complete()
            finally:
                publication.complete(); thread.join(5)
            self.assertFalse(thread.is_alive()); self.assertFalse(errors, errors)
            expected_status = KVPoll.Failed if failure else KVPoll.Success
            self.assertEqual(pm.request_status[19], expected_status)
            self.assertFalse(de.active); self.assertFalse(pm._staging_outstanding)
            if failure:
                self.assertNotIn(('aux', 0), sent)
                self.assertTrue(all(torch.equal(entry.tensor, initial[key]) for key, entry in d.by_key.items()))
                if failure == 'producer': self.assertFalse(sent)
            else:
                self.assertEqual(sent[-1], ('aux', 0))
                manifest, source = p.source_payload(room=19,generation=71,source_rank=0,source_tp=2,
                    prompt_tokens=257,token_start=0,token_end=257,kv_indices=src,kv_by_entry=maps,
                    state_indices=states,chunk_index=0,last_chunk=True,shallow_boundary=True)
                destination = d.destination_payload(manifest=manifest,kv_indices=dst,state_indices=dst_states,
                    decode_prefix_tokens=0,destination_rank=0,destination_tp=2)
                for field in manifest.fields:
                    a,b=source[field.key],destination[field.key]
                    self.assertTrue(torch.equal(a.tensor[a.rows],b.tensor[b.rows]),field.key)
                    received.append((field.key,b.tensor[b.rows].clone()))
                metrics=list(pm.flashnext_staging_metrics)
                if kv_first:
                    self.assertTrue(all(m['kv_bytes']+m['final_bytes']==m['bytes'] for m in metrics))
                total = sum(size for kind,size in sent)
            for storage in (ps,ds):
                leases=[storage.leases.acquire(room=100+i,nbytes=1024) for i in range(2)]
                self.assertTrue(all(lease is not None for lease in leases))
                for lease in leases: storage.leases.release(lease)
        return (received, total) if not failure else None

    def test_same_frozen_source_worker_staging_all_categories_equal(self):
        before, before_bytes = self.run_staged(False)
        after, after_bytes = self.run_staged(True)
        self.assertEqual(before_bytes, after_bytes)
        self.assertEqual([key for key,_ in before], [key for key,_ in after])
        self.assertTrue(all(torch.equal(a,b) for (_,a),(_,b) in zip(before,after)))

    def test_failure_and_abort_never_scatter_state_or_send_aux(self):
        for failure in ('producer','count','abort'):
            with self.subTest(failure=failure): self.run_staged(True, failure)

    def test_fence_kv_phase_has_no_factor_read_and_does_not_validate(self):
        count = MagicMock()
        count.__getitem__.side_effect = AssertionError('factor read before publication')
        count.is_cuda = False
        producer = CudaEvent(); producer.record()
        fence = FactorTransferFence(Event(),count,1,16,kv_first=True)
        fence.wait_kv(producer)
        self.assertFalse(fence.validated)
        count.__getitem__.assert_not_called()

    def test_partial_gather_preserves_full_manifest_and_all_wire_bytes(self):
        p = self.catalog(19)
        pages = np.array([2, 4])
        manifest, source = p.source_payload(room=19,generation=1,source_rank=0,source_tp=2,
            prompt_tokens=65,token_start=0,token_end=65,kv_indices=pages,kv_by_entry=None,
            state_indices=[[[2]],[3],pages],chunk_index=0,last_chunk=True,shallow_boundary=True)
        identity = manifest.to_bytes()
        early, late, early_ranges, late_ranges = staged_kv_phases(manifest)
        self.assertEqual(sum(n for _,n in early_ranges+late_ranges), manifest.nbytes)
        complete = torch.empty(manifest.nbytes,dtype=torch.uint8)
        partial = torch.full_like(complete,213)
        copy_payload(manifest=manifest,local=source,staging=complete,gather=True)
        copy_payload(manifest=manifest,local=source,staging=partial,gather=True,field_keys=early)
        for offset,n in late_ranges:
            self.assertTrue(bool((partial[offset:offset+n] == 213).all()))
        # Deliberate early state copy must be detected by that byte oracle.
        count_key = (0,'mamba.gdn_factored_count.0')
        corrupt = partial.clone()
        copy_payload(manifest=manifest,local=source,staging=corrupt,gather=True,
                     field_keys=early | {count_key})
        self.assertFalse(all(bool((corrupt[o:o+n] == 213).all()) for o,n in late_ranges))
        copy_payload(manifest=manifest,local=source,staging=partial,gather=True,field_keys=late)
        self.assertTrue(torch.equal(partial,complete))
        self.assertEqual(manifest.to_bytes(),identity)
        with self.assertRaises(ValueError):
            copy_payload(manifest=manifest,local=source,staging=partial,gather=True,
                         field_keys={(999,'not-a-field')})

    def test_profile_guard_and_fake_warmup_keep_full_fence(self):
        from sglang.srt.disaggregation.fake.conn import FakeKVSender
        pool = NS(cfg=NS(r=16,strict_chunk=True),count=torch.full((1,3),16))
        done = CudaEvent(); done.record()
        pool._pd_batch_publication = NS(offload_join=True, transfer_event=lambda:done)
        sender = FakeKVSender(NS(), 'unused', 1, [0], 0)
        req = NS(disagg_kv_sender=sender,kv=NS(mamba_pool_idx=torch.tensor(1)),
            _pfactor_agg_contract=True,factored_prefill_boundary_steps=0)
        with patch.dict(os.environ,{'SGLANG_GDN_PD_SHALLOW_SEND_PIPELINE':'1'}):
            with self.assertRaisesRegex(RuntimeError,'requires shallow'):
                FactorStateHandoff(pool).before_send(req)
            req._pd_fulln_boundary_valid = torch.ones(3,dtype=torch.int32)
            FactorStateHandoff(pool).before_send(req)
            fence = sender._state_handoff_fence
            self.assertTrue(fence.kv_first)
            sender.send(np.array([1]), [[1]])
            self.assertTrue(fence.validated)
            self.assertEqual(sender.poll(),KVPoll.Success)

    def test_sender_scope_rejects_unknown_deep_or_staged_mismatch(self):
        for ids, staging, pp, accepted in (([3],False,1,True), ([],False,1,False),
            ([3,31],False,1,False), ([3],True,1,False), ([3],False,2,False)):
            with self.subTest(ids=ids,staging=staging,pp=pp):
                sender = object.__new__(MooncakeKVSender)
                sender.kv_mgr = NS(enable_staging=staging,pp_size=pp,dcp_size=1,
                    flashnext_staging=None,kv_args=NS(kv_layer_ids=ids,kv_data_ptrs=[1]*len(ids)))
                fence = FactorTransferFence(None,None,1,16,kv_first=True)
                if accepted:
                    sender.set_state_handoff_fence(fence)
                    self.assertIs(sender._state_handoff_fence,fence)
                else:
                    with self.assertRaises(ValueError): sender.set_state_handoff_fence(fence)

    def test_unstaged_worker_still_gates_state_after_kv(self):
        from test_flash_next_pd_publish_join import fake_pool, publication, submit, request
        for abort in (False,True):
            pool = fake_pool(2)
            _, pub = publication(pool)
            done = submit(pub,{1},lambda:pool.count[:,1].fill_(pool.cfg.r))
            wire = Transport(pool)
            wire.manager.kv_args.kv_layer_ids=[3]
            wire.sender.supports_kv_first_state_handoff=True
            req = request(wire.sender)
            req._pd_fulln_boundary_valid=torch.ones(40,dtype=torch.int32)
            with patch.dict(os.environ,{'SGLANG_GDN_PD_SHALLOW_SEND_PIPELINE':'1'}):
                FactorStateHandoff(pool).before_send(req)
            wire.enqueue(); wire.start()
            try:
                self.assertTrue(done.entered.wait(2))
                self.assertEqual(wire.sent,['kv'])
                if abort: wire.manager.request_status[7]=KVPoll.Failed
                done.complete()
                if not abort: self.assertTrue(wire.finished.wait(2))
            finally:
                done.complete(); wire.stop()
            self.assertEqual(wire.sent,['kv'] if abort else ['kv','state','aux'])
            self.assertEqual(wire.manager.request_status[7],KVPoll.Failed if abort else KVPoll.Success)


if __name__ == '__main__': unittest.main()
