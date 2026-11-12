"""Ownership and actual-PD-loop order tests; no CUDA numerical claims."""
import ast
from array import array
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[4]
MODULE = ROOT / "python/sglang/srt/managers/pp_prefill_lookahead.py"
spec = importlib.util.spec_from_file_location("pp_b1_contract", MODULE)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
PPPreparedInput = m.PPPreparedInput

class Tensor:
    def __init__(self, ids): self.ids = list(ids)
    def narrow(self, dim, start, count): return Tensor(self.ids[start:start+count])

def req(rid="r"):
    return NS(rid=rid, origin_input_ids=array("q", range(20)), output_ids=[],
        inflight_steps=0, finished=lambda: False, session=None,
        multimodal_inputs=None, input_embeds=None, positional_embed_overrides=None,
        beam_group=None, is_dllm=lambda: False, prefix_len=0, extend_end=20,
        get_fill_ids=lambda: array("q", range(20)))

def prepare(q, busy=(), slot=1, gen=1):
    return PPPreparedInput.prepare(slot,gen,q,set(busy),Tensor)

class Ownership(unittest.TestCase):
    def test_independent_backing_and_committed_prefix(self):
        r=req(); p=prepare([r]);r.prefix_len=4;r.extend_end=12
        self.assertEqual(p.for_batch([r]).ids,list(range(4,12)))
        r.origin_input_ids[0]=99
        self.assertEqual(p.input_ids_cpu.ids,list(range(20)))
        self.assertIsNone(p.for_batch([r]))
    def test_skip_shared_chunk_or_inflight(self):
        a,b,c=req("a"),req("b"),req("c");b.inflight_steps=1
        self.assertIs(prepare([a,b,c],[id(a)]).req,c)
    def test_no_early_generation_commit(self):
        p=prepare([req()],slot=2,gen=8)
        self.assertFalse(p.can_commit(2,7,6))
        self.assertTrue(p.can_commit(2,7,7))
        self.assertFalse(p.can_commit(2,8,8))
        self.assertFalse(p.can_commit(1,7,7))
    def test_replaced_or_aborted_request_rejected(self):
        r=req();p=prepare([r]);self.assertIsNone(p.for_batch([req()]))
        r.finished=lambda: True;self.assertIsNone(p.for_batch([r]))
    def test_mutated_fill_and_output_rejected(self):
        r=req();p=prepare([r]);r.get_fill_ids=lambda: array("q",[9]*20)
        self.assertIsNone(p.for_batch([r]));r.output_ids=[3]
        self.assertIsNone(p.for_batch([r]))
    def test_batched_fallback(self):
        a,b=req(),req();self.assertIsNone(prepare([a]).for_batch([a,b]))

class Done(Exception): pass

def loop_function(source):
    tree=ast.parse(source)
    fn=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef)
            and x.name=="event_loop_pp_disagg_prefill")
    fn.decorator_list=[]
    return fn

class Fake:
    def __init__(self,last):
        self.frames=[]; self.tick=0; self.pp_group=NS(is_last_rank=last)
        self.send_req_work=[];self._pp_spec_relay=False;self.device="cpu"
        self.chunked_req=None;self.waiting_queue=[req(str(i)) for i in range(14)]
        self.disagg_prefill_transfer_queue=[];self.enable_staging=False
        self.invariant_checker=NS(run=lambda *a: None)
    def init_pp_loop_state(self):
        self.pp_loop_size=4;self.mbs=[None]*4;self.last_mbs=[None]*4
        self.mb_metadata=[None]*4;self.last_rank_comm_queue=[]
        self.running_mbs=[NS(batch_is_full=False) for _ in range(4)]
    def increment_scheduler_iterations(self):
        self.tick+=1
        if self.tick>14: raise Done()
    def emit(self,x): self.frames.append((self.tick,x))
    def _pp_ingest_requests(self): self.emit("ingress");return [self.tick]
    def _pp_commit_comm_work(self,w): self.emit("retire")
    def process_bootstrapped_queue(self): self.emit("bootstrap")
    def _pp_pd_get_prefill_transferred_ids(self): self.emit("transferred");return {}
    def process_prefill_chunk(self,**kw): self.emit("chunk")
    def _process_hicache_events(self): self.emit("hicache")
    def _pp_may_launch(self): return True
    def get_new_batch_prefill(self,running):
        self.emit("admit");r=self.waiting_queue.pop(0)
        p=getattr(self,"_pp_pd_input_plan",None)
        if p is not None:
            assert p.for_batch([r]) is not None
            self.used.append((self.tick,p.generation,p.slot))
        return NS(batch_to_run=NS(reqs=[r]),running_batch=running)
    dp_attn_adapter=NS(maybe_prepare_mlp_sync_batch=lambda b:b)
    def _pp_recv_proxy_tensors(self):self.emit("proxy-recv");return None
    def _pp_commit_proxy_send_work(self):self.emit("proxy-retire")
    def _pp_launch_batch(self,slot,batch,proxy,metadata,q):
        self.emit("forward");metadata[slot]=NS();return NS(),None
    def _pp_commit_send_output_work_and_preprocess_output_tensors(self,*a):
        self.emit("output-ring")
        return None,None,NS(synchronize=lambda:self.emit("wait-result"))
    def _pp_pd_send_consensus_release_ids(self,*a):self.emit("release-send");return [],{}
    def _pp_recv_pyobj_from_prev_stage(self):self.emit("release-recv");return {}
    def _pp_process_batch_result(self,b,r):self.emit("process-result")
    def process_disagg_prefill_transfer_queue(self,r):self.emit("transfer")
    def _pp_send_reqs_to_next_stage(self,r):self.emit("req-send");return []
    def _pp_send_pyobj_to_next_stage(self,r,**kw):self.emit("transferred-send");return []
    def _pp_send_proxy_to_next_stage(self,r):self.emit("proxy-send")
    def on_idle(self): pass
    def maybe_sleep_on_idle(self): pass

def execute(source,on,last):
    fake=Fake(last);fake.used=[]
    class Prepared:
        @staticmethod
        def prepare(*args):
            fake.emit("prepare");return PPPreparedInput.prepare(*args)
    ns=dict(Scheduler=Fake, ScheduleBatch=object, Optional=__import__('typing').Optional,
        List=list, Dict=dict, get_parallel=lambda:NS(pp_size=4,pp_async_batch_depth=0),
        envs=NS(SGLANG_PP_PD_OVERLAP_B1=NS(get=lambda:on), SGLANG_PP_PD_EARLY_PUBLISH=NS(get=lambda:False)),
        PPPreparedInput=Prepared,flatten_arrays_to_pinned_cpu=lambda parts,pin:Tensor(parts[0]),
        is_pin_memory_available=lambda dev:False,CheckPhase=NS(BUSY=1))
    module=ast.Module(body=[loop_function(source)],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),"actual_pd_loop","exec"),ns)
    try:ns['event_loop_pp_disagg_prefill'](fake)
    except Done:pass
    return fake

class LoopOrder(unittest.TestCase):
    def test_off_matches_unmodified_loop_last_and_nonlast(self):
        path='python/sglang/srt/managers/scheduler_pp_mixin.py'
        original=subprocess.check_output(['git','show','a09f958:'+path],cwd=ROOT,text=True)
        patched=(ROOT/path).read_text()
        for last in [False,True]:
            self.assertEqual(execute(original,False,last).frames,
                             execute(patched,False,last).frames)
    def test_on_preserves_frames_and_hides_prepare_before_old_wait(self):
        source=(ROOT/'python/sglang/srt/managers/scheduler_pp_mixin.py').read_text()
        for last in [False,True]:
            off,on=execute(source,False,last),execute(source,True,last)
            self.assertEqual(off.frames,[(t,x) for t,x in on.frames if x!='prepare'])
            self.assertGreater(len(on.used),8)
            for tick in range(5,15):
                order=[x for t,x in on.frames if t==tick]
                self.assertLess(order.index('forward'),order.index('prepare'))
                self.assertLess(order.index('prepare'),order.index('wait-result'))
                self.assertLess(order.index('wait-result'),order.index('process-result'))

if __name__=='__main__':unittest.main()
