"""Execute the actual PD loop bodies with bounded CPU doubles, no torch import."""
import ast
import dataclasses
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest

ROOT = Path(sys.argv.pop(1)) if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir() else Path.cwd()
REL = 'python/sglang/srt/managers/scheduler_pp_mixin.py'
NEW = (ROOT / REL).read_text()
OLD = subprocess.check_output(['git', '-C', str(ROOT), 'show', os.environ.get('PP_B2_BASELINE', '800b637') + ':' + REL], text=True)
PARALLEL = types.SimpleNamespace(pp_size=4, pp_async_batch_depth=0)

def load(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SchedulerPPMixin')
    wanted = {'event_loop_pp_disagg_prefill', 'event_loop_pp_disagg_prefill_overlap_b2', '_pp_b2_commit_result'}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    for n in methods:
        n.decorator_list = []
    owner = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PPDeferredPrefillResult']
    ns = {'dataclass': dataclasses.dataclass, 'os': os, 'get_parallel': lambda: PARALLEL,
          'logger': types.SimpleNamespace(info=lambda *a: None),
          'CheckPhase': types.SimpleNamespace(BUSY='busy')}
    unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                           *owner, ast.ClassDef(name='Loop', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), 'actual_pd_loop', 'exec'), ns)
    return ns

NEW_NS, OLD_NS = load(NEW), load(OLD)

class Done(Exception): pass

class Batch:
    def __init__(self, frame, chunk=False):
        self.frame = frame
        self.reqs = [types.SimpleNamespace(rid=f'r{frame}')]
        self.chunked_req = self.reqs[0] if chunk else None
        self.batch_is_full = False
        self.spec_info = object()
        self.next_draft_input = self.spec_info

class Fake:
    def __init__(self, rank, empty=False, chunks=False, limit=100):
        self.rank, self.empty, self.chunks, self.limit = rank, empty, chunks, limit
        self.trace, self.frame = [], -1
        self.pp_group = types.SimpleNamespace(is_last_rank=rank == 3)
        self.dp_attn_adapter = types.SimpleNamespace(maybe_prepare_mlp_sync_batch=lambda b: b)
        self.invariant_checker = types.SimpleNamespace(run=lambda *a: None)
        self.server_args = types.SimpleNamespace(disable_cuda_graph=True)
        self.enable_staging = False
        self.enable_lora = False
        self.chunked_req = None
        self.disagg_prefill_transfer_queue = []
        self._pp_spec_relay = False
    def init_pp_loop_state(self):
        self.pp_loop_size=4
        self.mbs=[None]*4; self.last_mbs=[None]*4; self.mb_metadata=[None]*4
        self.running_mbs=[types.SimpleNamespace(batch_is_full=False) for _ in range(4)]
        self.last_rank_comm_queue=[]; self.send_req_work=[]; self.pp_outputs=None
    def increment_scheduler_iterations(self):
        self.frame += 1
        if self.frame == self.limit: raise Done()
    def _pp_ingest_requests(self): return [('ingress',self.frame)]
    def _pp_commit_comm_work(self, work): pass
    def process_bootstrapped_queue(self): pass
    def _pp_pd_get_prefill_transferred_ids(self): return {'transfer': self.frame}
    def process_prefill_chunk(self, **kw): pass
    def _process_hicache_events(self): pass
    def _pp_may_launch(self): return True
    def get_new_batch_prefill(self, running):
        b = None if self.empty and self.frame % 3 == 0 else Batch(self.frame, self.chunks and self.frame%5==0)
        return types.SimpleNamespace(batch_to_run=b, running_batch=running)
    def _pp_recv_proxy_tensors(self): return None
    def _pp_commit_proxy_send_work(self): pass
    def _pp_launch_batch(self, slot, batch, proxy, metadata, queue):
        self.trace.append(('launch',batch.frame,slot))
        metadata[slot]=types.SimpleNamespace(frame=batch.frame)
        return batch.frame, self.event(batch.frame)
    def event(self, frame):
        return types.SimpleNamespace(synchronize=lambda: self.trace.append(('wait',frame)))
    def _pp_commit_send_output_work_and_preprocess_output_tensors(self, a, b):
        self.trace.append(('output-frame',a,b))
        target=self.mbs[b]
        return None, target, self.event(target.frame if target else -1)
    def _pp_pd_send_consensus_release_ids(self, t, i, r, ids):
        self.trace.append(('release-frame',i,ids['transfer']))
        return [], r
    def _pp_recv_pyobj_from_prev_stage(self): return {'release':self.frame}
    def _pp_process_batch_result(self, batch, result):
        assert result is batch
        assert batch.spec_info is result.next_draft_input
        self.trace.append(('process',batch.frame))
    def process_disagg_prefill_transfer_queue(self, ids): self.trace.append(('transfer-process',ids['release']))
    def _pp_send_reqs_to_next_stage(self, reqs):
        self.trace.append(('request-frame',reqs[0][1])); return []
    def _pp_send_pyobj_to_next_stage(self, ids, async_send=False):
        self.trace.append(('transfer-frame',ids['transfer'])); return []
    def _pp_send_proxy_to_next_stage(self, result): self.trace.append(('proxy-frame',result))
    def on_idle(self): self.trace.append(('idle',self.frame))

def run(ns, rank, on=False, **kw):
    engine=type('Engine',(Fake,ns['Loop']),{})(rank,**kw)
    previous=os.environ.get('SGLANG_PP_PD_OVERLAP_B2')
    os.environ['SGLANG_PP_PD_OVERLAP_B2']='1' if on else '0'
    try:
        with unittest.TestCase().assertRaises(Done): engine.event_loop_pp_disagg_prefill()
    finally:
        if previous is None: os.environ.pop('SGLANG_PP_PD_OVERLAP_B2',None)
        else: os.environ['SGLANG_PP_PD_OVERLAP_B2']=previous
    return engine

class Contracts(unittest.TestCase):
    def test_off_original_body_bytes(self):
        def body(s):
            at=s.index('    def event_loop_pp_disagg_prefill(self: Scheduler):')
            start=s.index('        self.init_pp_loop_state()',at)
            end=s.index('    @DynamicGradMode()',start)
            return s[start:end]
        # The added B2 methods follow the original body: delimit by original
        # end marker to compare its preserved bytes exactly.
        end=NEW.index('    def _pp_b2_commit_result')
        self.assertEqual(body(OLD),body(NEW[:end]+'    @DynamicGradMode()'))
    def test_off_actual_execution_equal(self):
        for rank in range(4):
            for empty,chunks in [(False,False),(True,False),(True,True)]:
                self.assertEqual(run(OLD_NS,rank,empty=empty,chunks=chunks).trace,
                                 run(NEW_NS,rank,empty=empty,chunks=chunks).trace)
    def test_on_frames_and_exactly_once_results(self):
        for rank in range(4):
            for empty,chunks in [(False,False),(True,False),(True,True)]:
                off=run(OLD_NS,rank,empty=empty,chunks=chunks)
                on=run(NEW_NS,rank,on=True,empty=empty,chunks=chunks)
                frames=lambda e:[x for x in e.trace if x[0].endswith('-frame')]
                self.assertEqual(frames(off),frames(on))
                results=lambda e:[x for x in e.trace if x[0]=='process' and x[1]<94]
                self.assertEqual(results(off),results(on))
                self.assertEqual(len(results(on)),len(set(results(on))))
    def test_next_launch_precedes_old_result(self):
        e=run(NEW_NS,0,on=True)
        pos={x:i for i,x in enumerate(e.trace)}
        self.assertLess(pos[('launch',4,0)],pos[('wait',0)])
        self.assertLess(pos[('wait',0)],pos[('process',0)])
    def test_chunk_retains_old_boundary(self):
        e=run(NEW_NS,3,on=True,chunks=True)
        pos={x:i for i,x in enumerate(e.trace)}
        self.assertLess(pos[('process',0)],pos[('launch',4,0)])
    def test_generation_and_owner_guard(self):
        e=type('Engine',(Fake,NEW_NS['Loop']),{})(0)
        e.init_pp_loop_state()
        b=Batch(0); e.mbs[0]=b; md=object(); e.mb_metadata[0]=md
        owner=NEW_NS['PPDeferredPrefillResult'](0,1,b,md,b,e.event(0),False,None)
        with self.assertRaisesRegex(RuntimeError,'generation'):
            e._pp_b2_commit_result(owner,[2,0,0,0])
        e.mb_metadata[0]=object()
        with self.assertRaisesRegex(RuntimeError,'generation'):
            e._pp_b2_commit_result(owner,[1,0,0,0])
        self.assertEqual(e.trace,[])
    def test_empty_next_launch_drains(self):
        e=run(NEW_NS,0,on=True,empty=True)
        # Batch2 is pending when empty frame6 reuses its logical slot.
        self.assertIn(('process',2),e.trace)

    def test_pending_abort_and_draft_identity(self):
        class AbortFake(Fake):
            def _pp_ingest_requests(self):
                if self.frame == 4:
                    self.mbs[0].reqs[0].aborted = True
                return super()._pp_ingest_requests()
            def _pp_process_batch_result(self, batch, result):
                super()._pp_process_batch_result(batch,result)
                if batch.frame == 0:
                    assert batch.reqs[0].aborted
        e=type('Engine',(AbortFake,NEW_NS['Loop']),{})(0)
        with self.assertRaises(Done): e.event_loop_pp_disagg_prefill_overlap_b2()
        self.assertIn(('process',0),e.trace)
    def test_shared_req_rejected_before_forward(self):
        class SharedFake(Fake):
            def get_new_batch_prefill(self,running):
                plan=super().get_new_batch_prefill(running)
                if self.frame == 4: plan.batch_to_run.reqs=self.mbs[0].reqs
                return plan
        e=type('Engine',(SharedFake,NEW_NS['Loop']),{})(0)
        with self.assertRaisesRegex(RuntimeError,'shares a pending Req'):
            e.event_loop_pp_disagg_prefill_overlap_b2()
        self.assertNotIn(('launch',4,0),e.trace)
    def test_pending_metadata_retained_during_next_launch(self):
        class OwnerFake(Fake):
            def _pp_launch_batch(self,slot,batch,proxy,metadata,queue):
                old=self.mbs[slot]; md=self.mb_metadata[slot]
                result=super()._pp_launch_batch(slot,batch,proxy,metadata,queue)
                if batch.frame == 4:
                    assert old.frame == 0
                    assert self.mbs[slot] is old
                    assert self.mb_metadata[slot] is md
                    assert metadata[slot] is not md
                return result
        e=type('Engine',(OwnerFake,NEW_NS['Loop']),{})(0)
        with self.assertRaises(Done): e.event_loop_pp_disagg_prefill_overlap_b2()
    def test_lora_rejected(self):
        e=type('Engine',(Fake,NEW_NS['Loop']),{})(0)
        e.enable_lora=True
        with self.assertRaisesRegex(ValueError,'no LoRA'):
            e.event_loop_pp_disagg_prefill_overlap_b2()

if __name__=='__main__': unittest.main(verbosity=2)
