"""PD publication record/lifetime CPU gate; CUDA/RDMA are not admitted here."""
import unittest
from contextlib import contextmanager
from dataclasses import replace
from types import MethodType, SimpleNamespace as NS
from unittest.mock import patch

import pd_shallow_publication_cpu as cpu
from sglang.srt.mem_cache.gdn_pd_overlap import bind_result_record
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

FLAG = 'SGLANG_GDN_PD_PUBLISH_OVERLAP_OK'
torch = cpu.torch


class EnqueueRuntime(cpu.Runtime):
    def join(self, event):
        # A stream wait does not imply host completion or cross-stream waits.
        self.events.append(('device-wait', event))


@contextmanager
def resident():
    with cpu.worker('1', runtime=EnqueueRuntime(), recipe={FLAG: '1'}) as w:
        assert cpu.warmup(w).ready.call_count == 1
        yield w


def forward(w, batch_id=1):
    req = Req('record-'+str(batch_id), 'cpu', [1]*8, SamplingParams(max_new_tokens=1))
    w.rp.alloc([req]); req.kv.kv_allocated_len = 8
    slot = w.rp.mamba_allocator.alloc(1)
    track = w.rp.mamba_allocator.alloc(1)
    req.kv.mamba_pool_idx = slot[0]
    w.rp.req_index_to_mamba_index_mapping[req.kv.req_pool_idx] = slot[0]
    fb = cpu.batch(offset=req.kv.req_pool_idx)
    fb.pd_publication_batch_id = batch_id
    fb.mamba_track_indices = track
    w.runner.forward(fb)
    batch = NS(forward_iter=batch_id, reqs=[req], req_to_token_pool=w.rp)
    result = NS(pd_publication_record=fb.pd_publication_record)
    return req, batch, result, slot, track


def versions(w):
    return {name: getattr(w.pool, name)._version for name in
            ('a','U','W','count','stale','dense_of','prefix_valid','dense_required')}


class PublicationRecords(unittest.TestCase):
    def test_independent_forward_schedule_transfer_edges(self):
        with resident() as w:
            req, batch, result, slot, _ = forward(w)
            record = result.pd_publication_record
            manager = w.pub.records
            before = manager.stats.copy()
            with patch.object(torch.cuda, 'current_stream', return_value=NS(cuda_stream=101)):
                with w.pub.forward_scope(frozenset({90})):
                    pass
                self.assertEqual(manager.stats['forward_waits'], before['forward_waits'])
                with w.pub.forward_scope(frozenset(slot.tolist())):
                    pass
            self.assertFalse(record.publication_done.query())
            # Schedule must register its OWN wait despite forward's wait.
            with patch.object(torch.cuda, 'current_stream', return_value=NS(cuda_stream=102)):
                bind_result_record(batch, result)
            self.assertEqual(manager.stats['schedule_waits'], before['schedule_waits']+1)
            self.assertEqual(manager.stats['forward_waits'], before['forward_waits']+1)
            sender = cpu.fake_sender(); req.disagg_kv_sender = sender
            w.handler().before_send(req)
            fence = sender._state_handoff_fence
            self.assertIs(fence.record, record)
            record.publication_done.complete()
            producer = cpu.Event(ready=True)
            fence.wait(producer)
            self.assertEqual(manager.stats['transfer_waits'], before['transfer_waits']+1)
            self.assertGreater(manager.stats['publication_edges'], 0)
            self.assertEqual(manager.stats['implicit_d2h_added'], 0)

    def test_raw_mamba_free_cannot_reenter_before_retirement(self):
        with resident() as w:
            req, batch, result, slot, _ = forward(w)
            manager = w.pub.records
            w.rp.mamba_allocator.free(slot)
            self.assertNotIn(int(slot[0]), w.rp.mamba_allocator.free_slots.tolist())
            # Device completion alone is not consumer retirement.
            result.pd_publication_record.publication_done.complete()
            manager.reap()
            self.assertNotIn(int(slot[0]), w.rp.mamba_allocator.free_slots.tolist())
            bind_result_record(batch, result)
            self.assertNotIn(int(slot[0]), w.rp.mamba_allocator.free_slots.tolist())
            manager.release_request(req)
            self.assertIn(int(slot[0]), w.rp.mamba_allocator.free_slots.tolist())
            self.assertEqual(manager.stats['publications'], manager.stats['retired'])
            self.assertFalse(manager.deferred_frees)

    def test_abort_before_result_keeps_request_and_mamba_leases(self):
        with resident() as w:
            req, batch, result, slot, _ = forward(w)
            index = req.kv.req_pool_idx
            manager = w.pub.records
            w.rp.mamba_allocator.free(slot)
            w.rp.free(req)  # real request free, wrapped only on this P instance
            self.assertIsNone(req.kv.req_pool_idx)
            self.assertNotIn(index, w.rp.free_slots)
            self.assertNotIn(int(slot[0]), w.rp.mamba_allocator.free_slots.tolist())
            result.pd_publication_record.publication_done.complete()
            manager.reap()
            self.assertNotIn(index, w.rp.free_slots)
            bind_result_record(batch, result)  # cancelled FIFO result: no sender
            manager.reap()
            self.assertIn(index, w.rp.free_slots)
            self.assertIn(int(slot[0]), w.rp.mamba_allocator.free_slots.tolist())
            self.assertFalse(manager.states)

    def test_raw_request_row_lease_and_generation(self):
        with resident() as w:
            req, batch, result, _, _ = forward(w)
            index = req.kv.req_pool_idx
            generation = int(w.rp.req_generation[index])
            w.rp.free_rows([index])
            self.assertNotIn(index, w.rp.free_slots)
            self.assertEqual(int(w.rp.req_generation[index]), generation)
            bind_result_record(batch, result)
            result.pd_publication_record.publication_done.complete()
            w.pub.records.release_request(req)
            self.assertIn(index, w.rp.free_slots)
            self.assertEqual(w.rp.alloc_rows(1), [index])
            self.assertEqual(int(w.rp.req_generation[index]), generation+1)

    def test_donation_hot_prefix_reader_and_schedule_versions(self):
        with resident() as w:
            req, batch, result, _, track = forward(w)
            manager = w.pub.records
            # Production extra-buffer donation only changes index ownership.
            new = w.rp.mamba_allocator.alloc(1)
            req.kv.mamba_ping_pong_track_buffer = torch.cat((track, new))
            req.kv.mamba_last_track_idx = 0
            w.rp.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(32,2,dtype=torch.long)
            for name in ('get_mamba_ping_pong_keep_idx','set_mamba_ping_pong_slot'):
                setattr(w.rp,name,MethodType(getattr(HybridReqToTokenPool,name),w.rp))
            before = versions(w)
            donated = HybridReqToTokenPool.donate_mamba_ping_pong_slot(w.rp,req,new)
            self.assertTrue(torch.equal(donated,track))
            self.assertEqual(versions(w),before)
            self.assertFalse(result.pd_publication_record.publication_done.query())
            # Eviction cannot recycle that donated slot until the lease retires.
            w.rp.mamba_allocator.free(donated)
            self.assertNotIn(int(track[0]), w.rp.mamba_allocator.free_slots.tolist())
            count = manager.stats['forward_waits']
            with w.pub.forward_scope(frozenset(track.tolist())):
                w.pool.pside_join(forward_local=True)
            self.assertEqual(manager.stats['forward_waits'],count+1)
            bind_result_record(batch,result)
            result.pd_publication_record.publication_done.complete()
            manager.release_request(req)
            self.assertIn(int(track[0]),w.rp.mamba_allocator.free_slots.tolist())

    def test_bad_batch_identity_and_live_clear_fail_closed(self):
        with resident() as w:
            req,batch,result,slot,_=forward(w)
            wrong=NS(pd_publication_record=replace(result.pd_publication_record,batch_id=99))
            with self.assertRaisesRegex(RuntimeError,'FIFO batch/publication'):
                bind_result_record(batch,wrong)
            for allocator in (w.rp,w.rp.mamba_allocator):
                with self.assertRaisesRegex(RuntimeError,'live PD publication'):
                    allocator.clear()
            with w.pub.forward_scope(frozenset({90})):
                with self.assertRaisesRegex(RuntimeError,'release allocator slots'):
                    w.rp.mamba_allocator.free(slot)

    def test_default_off_does_not_wrap_allocators_or_add_events(self):
        with cpu.worker('1',recipe={FLAG:'0'}) as w:
            self.assertIsNone(w.pub.records)
            self.assertNotIn('free',vars(w.rp.mamba_allocator))
            self.assertNotIn('free_rows',vars(w.rp))
            self.assertEqual(cpu.warmup(w).ready.call_count,1)


if __name__=='__main__':
    unittest.main()
