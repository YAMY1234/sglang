"""Actual contract install/track/plan/collector replay with CPU events and graph stand-ins."""
from contextlib import contextmanager
import copy
import os
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest

from test_flash_next_agg_fulln import worker, schedule, init_graphs, torch
from test_gdn_fulln_overlap import Runtime
from sglang.srt.mem_cache import gdn_fulln_overlap as ov
from sglang.srt.mem_cache import gdn_prefill_agg_contract as agg

FLAG='SGLANG_GDN_AGG_FULLN_OVERLAP_OK'


@contextmanager
def overlapped(flag=True):
    with worker() as w, patch.dict(os.environ,{FLAG:str(int(flag))}), patch(
        'sglang.srt.runtime_context.get_schedule',return_value=NS(disable_overlap_schedule=False)
    ), patch.object(ov,'CudaBoundaryRuntime',lambda device:Runtime()):
        init_graphs(w.runner,w.capture)
        yield w


def forward(w,sb):
    fb=w.Forward.init_new(sb,w.runner)
    w.owner.prepare_forward_batch(fb)
    w.backend.init_forward_metadata(fb)
    output=w.owner.forward(fb.input_ids,fb.input_ids,fb)
    return fb,output


def finish(w,sb):
    c=w.pool._agg_fulln_overlap; r=sb.fulln_overlap_record
    from collections import deque
    queued=NS(result_queue=deque([(copy.copy(sb),NS())]))
    def process():
        batch,_=queued.result_queue.popleft()
        c.runtime.copy_sync()
        batch.fulln_overlap_record.before_result()
    assert c.drain_before_planning(queued,process)
    return r


class FullNOverlapContractTest(unittest.TestCase):
    def test_optin_install_on_and_off_and_no_PD_side_effect(self):
        for flag in (False,True):
            if not flag:
                with self.assertRaisesRegex(ValueError,'isolated scheduling'):
                    with overlapped(flag):pass
            else:
                with overlapped(flag) as w:
                    self.assertIsNotNone(w.pool._agg_fulln_overlap)
                    self.assertFalse(hasattr(w.owner,'_exact_tail_installed'))
        with worker() as w, patch.dict(os.environ,{FLAG:'1'}):
            # Schedule is already isolated: opting in does not install a new path.
            init_graphs(w.runner,w.capture)
            self.assertFalse(hasattr(w.pool,'_agg_fulln_overlap'))

    def test_actual_trunk_collector_plan_and_logits_match_isolated_bitwise(self):
        for rows in (1,2,4,8):
            for prefix in (0,64):
                with self.subTest(rows=rows,prefix=prefix):
                    with worker() as old:
                        init_graphs(old.runner,old.capture)
                        sb=schedule(old,rows=rows,prefix=prefix)
                        fb,output=forward(old,sb)
                        logits=output.next_token_logits.clone()
                        fields={n:getattr(old.pool,n).clone() for n in ('a','U','W','count','stale','dense_of','prefix_valid')}
                        tracks=[tuple(getattr(r.kv,n) for n in ov.TRACK_FIELDS) for r in sb.reqs]
                    with overlapped() as w:
                        sb=schedule(w,rows=rows,prefix=prefix)
                        fb,output=forward(w,sb)
                        self.assertTrue(torch.equal(logits,output.next_token_logits))
                        for n,value in fields.items():self.assertTrue(torch.equal(value,getattr(w.pool,n)),n)
                        self.assertEqual(tracks,[tuple(getattr(r.kv,n) for n in ov.TRACK_FIELDS) for r in sb.reqs])
                        self.assertEqual(w.backend.plan_calls,1)
                        self.assertEqual(w.pool._agg_prefill_graph.run.call_count,1)
                        self.assertIs(fb.fulln_overlap_record.plan,w.backend.forward_metadata.factored_extend)
                        self.assertTrue(all(not hasattr(r,'_pfactor_track_before') for r in sb.reqs))
                        self.assertTrue(all(not hasattr(r,'_pfactor_agg_contract') for r in sb.reqs))
                        finish(w,sb)

    def test_batch_snapshots_ignore_poisoned_shared_request_markers(self):
        with overlapped() as w:
            sb=schedule(w,rows=2); r=sb.fulln_overlap_record
            for req in sb.reqs:
                req._pfactor_track_before=(None,(99,99,99),False)
                req._pfactor_agg_contract=False
            fb,_=forward(w,sb)
            self.assertIs(fb.fulln_overlap_record,r)
            self.assertTrue(fb._pfactor_agg_contract)
            self.assertEqual([x.kv.mamba_last_track_seqlen for x in sb.reqs],[64,64])
            finish(w,sb)

    def test_graph_reject_and_late_TBO_restore_N_minus_one_without_publication(self):
        for cause in ('graph','tbo'):
            with self.subTest(cause=cause), overlapped() as w:
                sb=schedule(w,rows=2)
                if cause=='graph':w.owner._prefill_runners['trunk'].can_run.return_value=False
                else:sb.can_run_tbo=True;sb.tbo_split_seq_index=0
                fb=w.Forward.init_new(sb,w.runner)
                self.assertFalse(fb._pfactor_agg_contract)
                self.assertIsNone(sb.fulln_overlap_record)
                self.assertEqual([r.kv.mamba_last_track_seqlen for r in sb.reqs],[None,None])
                self.assertEqual(w.pool._agg_fulln_overlap.stats['publication_events'],0)

    def test_mixed_TBO_single_and_large_batches_keep_fallback(self):
        for kwargs in ({'mixed':True},{'tbo':True},{'tokens':1},{'rows':17}):
            with self.subTest(kwargs=kwargs), overlapped() as w:
                sb=schedule(w,**kwargs);fb=w.Forward.init_new(sb,w.runner)
                self.assertFalse(fb._pfactor_agg_contract);self.assertIsNone(sb.fulln_overlap_record)
                self.assertEqual(sum(w.pool._agg_fulln_overlap.stats.values()),0)

    def test_plan_pending_and_workspace_cannot_be_reused_before_result(self):
        with overlapped() as w:
            sb=schedule(w,rows=2);fb,_=forward(w,sb);r=sb.fulln_overlap_record
            with self.assertRaisesRegex(RuntimeError,'prior result drain'):schedule(w)
            self.assertEqual(r.plan.pending,[])
            with self.assertRaisesRegex(RuntimeError,'before publication'):r.before_result()
            finish(w,sb)
            next_sb=schedule(w,rows=2,prefix=64);next_fb,_=forward(w,next_sb)
            self.assertIsNot(next_fb.fulln_overlap_record,r)
            self.assertTrue(torch.all(w.pool.count[:,1:3]==w.pool.cfg.r));finish(w,next_sb)

    def test_backend_metadata_replacement_and_duplicate_plan_fail_closed(self):
        with overlapped() as w:
            sb=schedule(w);fb=w.Forward.init_new(sb,w.runner)
            w.owner.prepare_forward_batch(fb);w.backend.init_forward_metadata(fb)
            with self.assertRaisesRegex(RuntimeError,'more than once'):w.backend.init_forward_metadata(fb)
            with self.assertRaisesRegex(RuntimeError,'does not match'):w.owner.forward(fb.input_ids,fb.input_ids,fb)

    def test_decode_init_has_no_new_snapshot_or_event(self):
        from test_flash_next_agg_fulln import ForwardMode
        with overlapped() as w:
            sb=schedule(w);forward(w,sb);finish(w,sb)
            sb.forward_mode=ForwardMode.DECODE
            c=w.pool._agg_fulln_overlap
            stats=dict(c.stats)
            with patch.object(c,'begin',side_effect=AssertionError('decode snapshot')):
                fb=w.Forward.init_new(sb,w.runner)
            self.assertFalse(getattr(fb,'_pfactor_agg_contract',False))
            self.assertEqual(stats,c.stats)

    def test_no_extra_tensor_D2H_during_plan_or_forward(self):
        def run(enabled):
            counts={n:0 for n in ('item','cpu','tolist','numpy')}
            from contextlib import ExitStack
            with ExitStack() as stack:
                for n in counts:
                    original=getattr(torch.Tensor,n)
                    def wrapper(t,*args,_n=n,_original=original,**kw):
                        counts[_n]+=1;return _original(t,*args,**kw)
                    stack.enter_context(patch.object(torch.Tensor,n,wrapper))
                w=stack.enter_context(overlapped() if enabled else worker())
                if not enabled:init_graphs(w.runner,w.capture)
                sb=schedule(w,rows=4,prefix=64);forward(w,sb)
                if enabled:finish(w,sb)
            return counts
        self.assertEqual(run(False),run(True))


if __name__=='__main__':unittest.main()
