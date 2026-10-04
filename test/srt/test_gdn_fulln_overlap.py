"""CPU event/ownership tests; runs the actual overlap-loop body, no CUDA mocks of math."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from collections import deque
import unittest

ROOT = Path(__file__).resolve().parents[2]
NAME = 'sglang.srt.mem_cache.gdn_fulln_overlap'
if NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(NAME, ROOT/'python/sglang/srt/mem_cache/gdn_fulln_overlap.py')
    module = importlib.util.module_from_spec(spec); sys.modules[NAME] = module
    spec.loader.exec_module(module)
ov = sys.modules[NAME]


class Runtime:
    def __init__(self):
        self.events = []; self.done = []; self.host_syncs = 0
    def attach(self, scheduler): self.events.append('attach')
    def fence_before_plan(self):
        self.events.append('prior-forward-event'); return object()
    def published(self):
        event = NS(ready=False); self.done.append(event)
        self.events.append('publish-event'); return event
    def wait_for_result(self, event): self.events.append('schedule-wait-event')
    def complete(self, event): return event.ready
    def copy_sync(self):
        self.host_syncs += 1; self.events.append('existing-copy-sync')
        for event in self.done: event.ready = True


def request(slot=1):
    return NS(kv=NS(req_pool_idx=slot, mamba_last_track_idx=0,
                   mamba_next_track_idx=1, mamba_last_track_seqlen=64))


def batch(req=None):
    return NS(reqs=[request() if req is None else req], fulln_overlap_record=None)


def prepared(controller, source):
    record = controller.begin(source, True)
    req = source.reqs[0]; fields = tuple(getattr(req.kv, n) for n in ov.TRACK_FIELDS)
    record.tracks[id(req)] = ov.TrackSnapshot(req, fields, fields, True)
    record.seal(True); record.plan = NS(pending=[], slots=(req.kv.req_pool_idx,))
    return record


def actual_loop():
    path = ROOT/'python/sglang/srt/managers/scheduler.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    method = copy.deepcopy(next(n for n in cls.body if getattr(n, 'name', '') == 'event_loop_overlap'))
    method.decorator_list = []
    scope = dict(deque=deque, envs=NS(SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY=NS(get=lambda: False)))
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), str(path), 'exec'), scope)
    return scope['event_loop_overlap']


class OverlapEventTest(unittest.TestCase):
    def test_req_flag_scope_and_failure_restore(self):
        req = request()
        with ov.track_selection(req, True): self.assertTrue(req._pfactor_agg_contract)
        self.assertFalse(hasattr(req, '_pfactor_agg_contract'))
        req._pfactor_agg_contract = False
        with self.assertRaises(ValueError):
            with ov.track_selection(req, True): raise ValueError('track failed')
        self.assertFalse(req._pfactor_agg_contract)

    def test_incomplete_publication_blocks_result_and_copy_completion_allows(self):
        rt = Runtime(); c = ov.FullNOverlap('cpu', rt); sb = batch(); r = prepared(c, sb); c.publish(r)
        with self.assertRaisesRegex(RuntimeError, 'before publication'): r.before_result()
        rt.copy_sync(); r.before_result(); self.assertTrue(r.result_validated)
        self.assertEqual(rt.host_syncs, 1)

    def test_slot_reuse_and_track_mutation_are_rejected(self):
        for field, value in [('req_pool_idx', 9), ('mamba_next_track_idx', 0), ('mamba_last_track_seqlen', 128)]:
            with self.subTest(field=field):
                rt=Runtime(); c=ov.FullNOverlap('cpu',rt); sb=batch(); r=prepared(c,sb); c.publish(r); rt.copy_sync()
                setattr(sb.reqs[0].kv,field,value)
                with self.assertRaises(RuntimeError): r.before_result()

    def test_hot_prefix_slot_can_be_evicted_and_reused_only_after_drain(self):
        rt=Runtime(); c=ov.FullNOverlap('cpu',rt); sb=batch(); r=prepared(c,sb); c.publish(r)
        with self.assertRaisesRegex(RuntimeError,'prior result drain'): prepared(c,batch())
        queue=deque([(copy.copy(sb),NS())]); scheduler=NS(result_queue=queue)
        def consume():
            saved,_=queue.popleft(); rt.copy_sync(); saved.fulln_overlap_record.before_result()
        self.assertTrue(c.drain_before_planning(scheduler,consume))
        self.assertIsNone(sb.fulln_overlap_record); self.assertIsNone(c.pending)
        # New request generation reuses the same slot; old record cannot consume it.
        next_batch=batch(request(1)); new=prepared(c,next_batch)
        self.assertGreater(new.serial,r.serial)
        self.assertEqual(new.identities[0][1],r.identities[0][1])
        with self.assertRaises(RuntimeError): r.before_result()

    def test_missing_queue_ownership_and_unvalidated_reader_fail_closed(self):
        rt=Runtime(); c=ov.FullNOverlap('cpu',rt); sb=batch(); r=prepared(c,sb); c.publish(r)
        with self.assertRaisesRegex(RuntimeError,'queue ownership'):
            c.drain_before_planning(NS(result_queue=deque()),lambda:None)
        queue=deque([(copy.copy(sb),NS())]); rt.copy_sync()
        with self.assertRaisesRegex(RuntimeError,'did not complete'):
            c.drain_before_planning(NS(result_queue=queue),lambda:queue.popleft())

    def test_batch_copy_retains_record_and_forward_batch_declares_it(self):
        path=ROOT/'python/sglang/srt/managers/schedule_batch.py'
        tree=ast.parse(path.read_text()); cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ScheduleBatch')
        method=copy.deepcopy(next(n for n in cls.body if getattr(n,'name','')=='copy')); method.decorator_list=[]
        scope={'ScheduleBatch':lambda **kw:NS(**kw)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(path),'exec'),scope)
        attrs={n.attr:None for n in ast.walk(method) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self'}
        attrs.update(reqs=[request()],fulln_overlap_record=object()); src=NS(**attrs)
        out=scope['copy'](src); self.assertIs(out.fulln_overlap_record,src.fulln_overlap_record)
        self.assertIsNot(out.reqs,src.reqs)
        fb=ast.parse((ROOT/'python/sglang/srt/model_executor/forward_batch_info.py').read_text())
        fbcls=next(n for n in fb.body if isinstance(n,ast.ClassDef) and n.name=='ForwardBatch')
        self.assertIn('fulln_overlap_record',[n.target.id for n in fbcls.body if isinstance(n,ast.AnnAssign)])

    def run_sequence(self, kinds, enabled=True, consecutive_disable=False):
        rt=Runtime(); c=ov.FullNOverlap('cpu',rt); trace=[]; i=0; processed=[]; last_req=request()
        s=NS(gracefully_exit=False,_engine_paused=False,running_batch=NS(),last_batch=None,
             is_generation=True,enable_unified_memory=False,req_to_token_pool=NS(factored_gdn_pool=NS()),
             forward_stream=object(),schedule_stream=object())
        if enabled: s.req_to_token_pool.factored_gdn_pool._agg_fulln_overlap=c
        s.ingest_requests=lambda:trace.append(('ingest',len(processed)))
        def get_next(**kw):
            nonlocal i
            if i==len(kinds):
                i+=1;return NS(running_batch=s.running_batch,batch_to_run=None)
            if i>len(kinds):
                s.gracefully_exit=True;return NS(running_batch=s.running_batch,batch_to_run=None)
            kind=kinds[i]; n=i; i+=1; trace.append(('plan',n))
            sb=batch(last_req if kind=='prefill' else request(n+10)); sb.number=n; sb.kind=kind
            sb.copy=lambda:copy.copy(sb)
            if enabled and kind=='prefill':
                r=prepared(c,sb); r.plan.pending[:]=[n]*36
            return NS(running_batch=s.running_batch,batch_to_run=sb)
        s.get_next_batch_to_run=get_next
        s.is_disable_overlap_for_batch=lambda b,last_batch:bool(consecutive_disable and b and last_batch and b.kind==last_batch.kind=='prefill')
        def run(sb):
            trace.append(('forward',sb.number))
            if sb.fulln_overlap_record is not None:
                # Same-stream publication consumes exactly this batch's pending slab.
                self.assertEqual(sb.fulln_overlap_record.plan.pending,[sb.number]*36)
                c.publish(sb.fulln_overlap_record)
            return NS()
        s.run_batch=run; s._apply_war_barrier=lambda:None
        def process(sb,result):
            rt.copy_sync()
            if sb.fulln_overlap_record is not None:sb.fulln_overlap_record.before_result()
            trace.append(('result',sb.number)); processed.append(sb.number)
        s.process_batch_result=process; s.launch_batch_sample_if_needed=lambda *a:None
        s.on_idle=lambda:None
        actual_loop()(s)
        self.assertEqual(processed,list(range(len(kinds))))
        self.assertEqual(rt.host_syncs,len(kinds))
        return trace,c

    def test_actual_loop_alternating_prefill_decode_and_consecutive_chunks(self):
        for kinds in [('decode','prefill','decode','prefill','decode'),('prefill','prefill','decode'),('prefill',)]:
            for disable in (False,True):
                with self.subTest(kinds=kinds,disable=disable):
                    trace,c=self.run_sequence(kinds,consecutive_disable=disable)
                    for i,kind in enumerate(kinds[:-1]):
                        if kind=='prefill':self.assertLess(trace.index(('result',i)),trace.index(('plan',i+1)))
                    self.assertEqual(c.stats['drained'],kinds.count('prefill'))
                    self.assertEqual(c.stats['plan_events'],c.stats['publication_events'])

    def test_default_off_and_decode_only_preserve_existing_overlap(self):
        for enabled in (False,True):
            trace,c=self.run_sequence(('decode','decode','decode'),enabled=enabled)
            self.assertLess(trace.index(('forward',1)),trace.index(('result',0)))
            self.assertEqual(sum(c.stats.values()),0)
        trace,_=self.run_sequence(('prefill','decode'),enabled=False)
        self.assertLess(trace.index(('forward',1)),trace.index(('result',0)))

    def test_no_added_host_synchronization_or_tensor_reads_in_boundary(self):
        tree=ast.parse((ROOT/'python/sglang/srt/mem_cache/gdn_fulln_overlap.py').read_text())
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for forbidden in ('synchronize','cpu','item','tolist','numpy'):
            self.assertNotIn(forbidden,calls)
        # Result hook is after the existing copy_done host wait, not ahead of it.
        path=ROOT/'python/sglang/srt/managers/scheduler_components/batch_result_processor.py'
        tree=ast.parse(path.read_text()); method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='process_batch_result_prefill')
        text=ast.unparse(method);self.assertLess(text.index('result.copy_done.synchronize()'),text.index('fulln_overlap_record.before_result()'))


if __name__=='__main__':unittest.main()
