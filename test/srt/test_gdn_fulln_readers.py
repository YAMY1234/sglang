"""Real overlap-loop ordering and the actual CPU ownership-reader hooks."""
import ast
from collections import deque
import copy
from types import SimpleNamespace as NS
from unittest.mock import Mock
import unittest

from test_gdn_fulln_overlap import (
    ov, Runtime, request, batch, prepared, ROOT,
)
import test_gdn_fulln_overlap as original


def method(name, **scope):
    path = ROOT / 'python/sglang/srt/managers/scheduler.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    body = copy.deepcopy(next(n for n in cls.body if getattr(n, 'name', '') == name))
    body.decorator_list = []
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, body], type_ignores=[])),
                 str(path), 'exec'), scope)
    return scope[name]


def pending():
    rt = Runtime()
    c = ov.FullNOverlap('cpu', rt)
    sb = batch(request(7))
    r = prepared(c, sb)
    c.publish(r)
    scheduler = NS(last_batch=sb, result_queue=deque([(copy.copy(sb), None)]))
    def pop():
        saved, _ = scheduler.result_queue.popleft()
        rt.copy_sync()
        saved.fulln_overlap_record.before_result()
        c.result_consumed(saved)
    c.start_iteration(scheduler, pop)
    return rt, c, sb, r


class ReaderBoundaryTest(unittest.TestCase):
    def test_next_disjoint_decode_launches_before_prefill_result_without_wait(self):
        runner = original.OverlapEventTest()
        for enabled in (False, True):
            trace, c = runner.run_sequence(('prefill', 'decode', 'decode'), enabled=enabled)
            self.assertLess(trace.index(('forward', 1)), trace.index(('result', 0)))
            self.assertEqual(c.stats['result_waits'], 0)
            self.assertEqual(c.stats['plan_events'], 0)

    def test_same_owner_decode_uses_forward_stream_order_not_cpu_result_drain(self):
        trace, c = original.OverlapEventTest().run_sequence(
            ('prefill', 'decode', 'decode'), decode_same_owner=True)
        self.assertLess(trace.index(('forward', 1)), trace.index(('result', 0)))
        self.assertEqual(c.stats['result_waits'], 0)
        self.assertEqual(c.stats['drained'], 1)

    def test_batch_view_detaches_but_fifo_retains_immutable_record(self):
        rt, c, sb, r = pending()
        self.assertIsNone(sb.fulln_overlap_record)
        saved = c.scheduler.result_queue[0][0]
        self.assertIs(saved.fulln_overlap_record, r)
        self.assertIsNot(saved.reqs, None)
        self.assertFalse(r.consumed)
        self.assertFalse(c.read_owners((request(91),)))
        self.assertNotIn('schedule-wait-event', rt.events)
        self.assertTrue(c.read_owners((sb.reqs[0],)))
        self.assertTrue(r.consumed)
        self.assertEqual(c.stats['result_waits'], 1)

    def test_checkpoint_slot_reader_waits_before_mutation_and_reuse(self):
        for selected in (False, True):
            rt, c, sb, r = pending()
            next_batch = batch(sb.reqs[0])
            new = c.begin(next_batch, selected)
            self.assertTrue(r.consumed)
            self.assertTrue(c.drained_this_iteration)
            self.assertEqual(c.stats['plan_events'], 1)
            self.assertEqual(rt.events.count('schedule-wait-event'), 1)
            self.assertGreater(new.serial, r.serial)
            sb.reqs[0].kv.mamba_last_track_seqlen = 128
            self.assertEqual(len(c.scheduler.result_queue), 0)

    def test_actual_hot_prefix_stash_reads_only_its_slot(self):
        for intersect in (False, True):
            rt, c, sb, r = pending()
            req = sb.reqs[0] if intersect else request(91)
            seen = []
            def cache(req, tree, chunked):
                seen.append(req)
                self.assertEqual(r.consumed, intersect)
            scheduler = NS(fulln_overlap_controller=c, tree_cache=object())
            method('stash_chunked_request', maybe_cache_unfinished_req=cache)(scheduler, req)
            self.assertEqual(seen, [req])
            self.assertEqual(rt.events.count('schedule-wait-event'), int(intersect))

    def test_chunk_cancel_waits_before_release_and_keeps_native_order(self):
        rt, c, sb, r = pending()
        req = sb.reqs[0]
        calls = []
        req.rid = 'cancelled'
        req.time_stats = NS(trace_ctx=NS(abort=lambda **kw: calls.append('abort-trace')))
        req.to_finish = object()
        scheduler = NS(_pending_chunked_abort_req=req, chunked_req=req,
            fulln_overlap_controller=c, disaggregation_mode='null', tree_cache=object(),
            _release_aborted_request=lambda rid: calls.append('release-id'),
            ipc_channels=NS(send_to_tokenizer=NS(send_output=lambda *x: calls.append('send'))))
        def release(*args, **kw):
            self.assertTrue(r.consumed)
            calls.append('release-slots')
        method('process_pending_chunked_abort',
            prepare_abort=lambda *args: calls.append('prepare-abort'),
            DisaggregationMode=NS(PREFILL='prefill'), release_kv_cache=release,
            _make_abort_req=lambda req: req, logger=NS(debug=lambda *args: None))(scheduler)
        self.assertIsNone(scheduler.chunked_req)
        self.assertIsNone(scheduler._pending_chunked_abort_req)
        self.assertEqual(calls, ['prepare-abort','abort-trace','release-id','release-slots','send'])
        self.assertEqual(rt.events.count('schedule-wait-event'), 1)

    def test_mem_retract_consumes_owner_then_rechecks_live_set(self):
        rt, c, sb, r = pending()
        sb.batch_size=lambda: len(sb.reqs)
        sb.filter_batch=lambda: None
        sb.is_empty=lambda: False
        sb.check_decode_mem=lambda: False
        scheduler=NS(fulln_overlap_controller=c, update_running_batch=Mock(return_value='replanned'))
        out=method('update_running_batch', TEST_RETRACT=False)(scheduler,sb)
        self.assertEqual(out,'replanned')
        self.assertTrue(r.consumed)
        scheduler.update_running_batch.assert_called_once_with(sb)

    def test_priority_preemption_drains_owner_before_victim_selection(self):
        rt,c,sb,r=pending()
        req=sb.reqs[0]
        req.finished=lambda:r.consumed
        adder=NS(tree_cache=NS(req_to_token_pool=NS(factored_gdn_pool=NS(_agg_fulln_overlap=c))),
                 running_batch=sb,preempt_list=[],rem_total_tokens=0)
        incoming=NS(full_untruncated_fill_ids=[1,2],prefix_indices=[],sampling_params=NS(max_new_tokens=1))
        path=ROOT/'python/sglang/srt/managers/schedule_policy.py'
        cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='PrefillAdder')
        fn=copy.deepcopy(next(n for n in cls.body if getattr(n,'name','')=='preempt_to_schedule'))
        fn.decorator_list=[]
        future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        scope=dict(get_schedule=lambda:NS(schedule_low_priority_values_first=True),CLIP_MAX_NEW_TOKENS=100)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future,fn],type_ignores=[])),str(path),'exec'),scope)
        self.assertFalse(scope['preempt_to_schedule'](adder,incoming))
        self.assertTrue(r.consumed)
        self.assertEqual(rt.events.count('schedule-wait-event'),1)

    def test_empty_or_disjoint_reader_does_not_walk_result_queue(self):
        rt,c,sb,r=pending()
        c.scheduler.result_queue=None
        self.assertFalse(c.read_owners(()))
        self.assertFalse(c.read_owners((request(99),)))
        self.assertFalse(r.consumed)
        self.assertNotIn('schedule-wait-event',rt.events)


if __name__ == '__main__':
    unittest.main()
