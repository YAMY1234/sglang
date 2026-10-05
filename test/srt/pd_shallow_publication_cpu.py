"""CPU backend for the complete imported P warmup/transaction/transport path.

Only tensor kernels/CUDA execution and HTTP/RDMA transport are CPU backends.
Production installers, prewarm/evaluate/run, transactions, sender/worker and
server warmup functions execute normally. No AST extraction.
"""
import copy
import os
import sys
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace as NS
from unittest.mock import Mock, patch

from sglang.test.test_utils import maybe_stub_sgl_kernel
maybe_stub_sgl_kernel()
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'test/registered/unit/mem_cache'))
from test_gdn_prefill_batch_graph import fake_pool, make_plan
from test_gdn_prefill_exact_tail import layer
from test_flash_next_pd_publish_join import Event, Runtime, Transport
from sglang.srt.mem_cache import gdn_pd_shallow_publication as candidate
from sglang.srt.mem_cache import gdn_prefill_exact_tail as exact
from sglang.srt.mem_cache import gdn_prefill_batch_graph as graph_module
from sglang.srt.mem_cache import gdn_factored_pool as native
from sglang.srt.model_executor import model_runner
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.entrypoints import http_server
from sglang.srt.disaggregation.state_handoff import HandoffKind, FactorStateHandoff
from sglang.srt.disaggregation.fake.conn import FakeKVSender
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.attention.linear.kernels import gdn_factored_io, gdn_factored, gdn_triton
from sglang.srt.models.flash_next_duet import pd_shallow
from twinstar_sgl import pd_shallow_gdn

IDS = list(candidate.GDN_IDS)
POLICY = (native.ORTH_METHOD,native.ORTH_WARPS_OVERRIDE,native.factorize_dense)
RECIPE = dict(SGLANG_GDN_PREFILL_EXACT_TAIL_BATCH='1',SGLANG_GDN_PREFILL_COMMIT_GRAPH='1',
    SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED='1',SGLANG_GDN_PD_PUBLISH_JOIN_OFFLOAD='1',
    SGLANG_GDN_FACTORED_HOST_SYNC_FREE='1',TWINSTAR_PD_FACTOR_ONLY_TAIL='0',
    SGLANG_GDN_PREFILL_JOIN_BRANCHES='0',SGLANG_GDN_PREFILL_FACTOR_GRAPH_K31='0',
    SGLANG_RUST_SERVER='0',CUDA_VISIBLE_DEVICES='',TRITON_INTERPRET='1')


class CpuGraph:
    def __init__(self): self.body = None
    def replay(self):
        if self.body is not None: return self.body()


class Valid:
    def __getitem__(self, grid):
        def run(valid, slots, *args):
            for slot in slots.tolist():
                if slot >= 0: valid[slot]=1
        return run


def scatter(source,target,slots):
    for i,slot in enumerate(slots.tolist()):
        if slot>=0: target[:,slot].copy_(source[:,i])


def store(a,u,w,fa,fu,fw,count,stale,dense_of,slots,r,*,stale_value,**kwargs):
    for i,slot in enumerate(slots.tolist()):
        if slot<0: continue
        fa[slot].copy_(a[i]);fu[slot].copy_(u[i]);fw[slot].copy_(w[i])
        count[slot].fill_(r);stale[slot]=stale_value;dense_of[slot]=-1
        # A real CPU copy into the indirect ring used by the production graph.
        if 'dense' in kwargs and int(kwargs['ring_dst'][i])>=0:
            ptr=int(kwargs['ring'][0]); target=RINGS[ptr]
            row=int(kwargs['ring_dst'][i]); target[row].copy_(kwargs['dense'][i])
            dense_of[slot]=row

RINGS={}


def packed_factor(mixed,a,b,*,fa,fu,fw,fcount,ssm_state_indices,**kwargs):
    # CPU arithmetic backend: identical inputs/order for both paths, including
    # actual writes to each shallow live factor and no tracked-slot append.
    for row,slot in enumerate(ssm_state_indices.tolist()):
        if slot<0: continue
        rank=int(fcount[slot,0]); width=fu.shape[-1]
        fu[slot,:,rank].copy_(mixed[row,:width].expand(fu.shape[1],-1))
        fw[slot,:,rank].copy_(a[row,:,None].expand(-1,width))
        fa[slot].add_(b[row,:,None])
        fcount[slot]+=1
    return torch.zeros(1,len(mixed),fa.shape[1],fa.shape[2])


def packed_dense(mixed,a,b,*,ssm_states,**kwargs):
    out=ssm_states.mean(-1).unsqueeze(0)
    ssm_states.add_(0.125)  # prove the owned N-1 snapshot survives tail writes
    return out


def cuda_backend(stack):
    stream=Mock()
    for name,value in (('current_stream',stream),('Stream',stream),('memory_allocated',0),
        ('memory_reserved',0),('graph_pool_handle',object()),('synchronize',None),
        ('is_current_stream_capturing',False)):
        stack.enter_context(patch.object(torch.cuda,name,return_value=value))
    stack.enter_context(patch.object(torch.cuda,'CUDAGraph',CpuGraph))
    stack.enter_context(patch.object(torch.cuda,'Event',Event))
    for name in ('stream','graph'):
        stack.enter_context(patch.object(torch.cuda,name,side_effect=lambda *a,**kw:nullcontext()))
    stack.enter_context(patch.object(gdn_factored_io,'store_factored',store))
    stack.enter_context(patch.object(gdn_factored,'factored_packed_decode',packed_factor))
    stack.enter_context(patch.object(gdn_triton.TritonGDNKernel,'packed_decode',staticmethod(packed_dense)))
    stack.enter_context(patch.object(graph_module,'scatter_rows',scatter))
    stack.enter_context(patch.object(graph_module,'_publish_valid',Valid()))


def batch(rows=1,offset=0,final=True):
    ids=torch.arange(offset,offset+rows)
    return NS(batch_size=rows,forward_mode=ForwardMode.EXTEND,req_pool_indices=ids,
        req_pool_indices_cpu=ids.clone(),extend_seq_lens_cpu=[8]*rows,
        extend_prefix_lens_cpu=[0]*rows,twinstar_prompt_final=[final]*rows,
        mamba_track_indices=ids+50,mamba_track_mask=torch.ones(rows,dtype=torch.bool),
        mamba_cow_src_indices=None,mamba_cow_dst_indices=None,mamba_clear_indices=None,
        spec_info=None,can_run_tbo=False,tbo_split_seq_index=None,
        input_ids=torch.arange(8*rows),positions=torch.arange(8*rows))


@contextmanager
def worker(flag='1',*,runtime=None,arm='PC',role='prefill',recipe=None):
    torch.manual_seed(7204); torch.set_num_threads(1)
    with ExitStack() as stack:
        env=dict(RECIPE,**{candidate.FLAG:flag})
        if recipe: env.update(recipe)
        stack.enter_context(patch.dict(os.environ,env))
        stack.enter_context(patch('sglang.srt.runtime_context.get_schedule',return_value=NS(disable_overlap_schedule=True)))
        capture_mode=stack.enter_context(patch('sglang.srt.model_executor.runner.get_is_capture_mode',return_value=False))
        stack.enter_context(patch.object(pd_shallow_gdn,'split_boundary',pd_shallow_gdn.split_boundary))
        cuda_backend(stack)
        pool=fake_pool(layers=36,width=16,heads=2,capacity=100)
        pool.layer_ids=IDS;pool.layer_map={lid:i for i,lid in enumerate(IDS)}
        pool.cfg.strict_chunk=True;pool.batch_prefill=True;pool.batch_prefill_final_copy=True
        pool.host_sync_free=True;pool.size=99;pool.device=torch.device('cpu')
        pool.pside_join=MethodType(native.FactoredGDNPool.pside_join,pool)
        pool.layer_index=lambda lid:pool.layer_map[lid]
        pool.ring_owner=[-1]*16;pool.ring_lru=list(range(16))
        for value in pool.dense_ring: RINGS[value.data_ptr()]=value
        rp=NS(factored_gdn_pool=pool,req_generation=torch.zeros(32,dtype=torch.long),
            req_index_to_mamba_index_mapping=torch.arange(1,33),mamba_v2p_table=None,
            pd_state_handoffs={HandoffKind.STATE_FACTOR:FactorStateHandoff(pool)},
            mamba_pool=NS(size=99,register_slot_state=Mock()))
        if arm in ('S','P'):
            rp.factored_gdn_pool=None;rp.pd_state_handoffs={}
            rp.mamba_pool.mamba_cache=NS(temporal=torch.zeros(1,dtype=torch.bfloat16))
        layers=[]
        for lid in IDS:
            if lid>=31: continue
            obj=RadixLinearAttention.__new__(RadixLinearAttention)
            torch.nn.Module.__init__(obj);vars(obj).update(vars(layer(lid)))
            layers.append(obj)
        owner=NS(n_layers=48,p_layer_ids=list(range(31)),emitter_ids=list(range(31,48)),
            pd_shallow_role='prefill' if arm in ('PC','P') else None,
            fullstack=dict(prefill_layer_trim=arm in ('PC','P'),gdn_rank=8 if arm in ('PC','C') else 0,
                           gdn_every=8 if arm in ('PC','C') else 0,duet_spec={}),
            model=NS(model=NS(modules=lambda:iter(layers))))
        args=NS(disaggregation_mode=role,pp_size=1,speculative_algorithm=None,is_embedding=False)
        w=NS(pool=pool,rp=rp,owner=owner,stack=stack,layers=layers,plans=[],tails=[],outputs=[],calls=[])
        w.capture_mode=capture_mode
        def core(input_ids,positions,fb):
            tx=pool._exact_tail_transaction
            rows=fb.batch_size
            plan=make_plan(pool,rows);plan.slots=rp.req_index_to_mamba_index_mapping[fb.req_pool_indices]
            plan.ring_dst=torch.arange(rows);plan.dense_required_after_commit=torch.full((rows,),int(not all(fb.twinstar_prompt_final)),dtype=torch.int32)
            prefix=NS(req_pool_indices_cpu=fb.req_pool_indices_cpu,batch_size=rows)
            tail_rows=[i for i,final in enumerate(fb.twinstar_prompt_final) if final]
            boundary=NS(req_pool_indices_cpu=fb.req_pool_indices_cpu[tail_rows],batch_size=len(tail_rows),
                        mamba_track_mask=torch.zeros(len(tail_rows),dtype=torch.bool))
            metadata=NS(mamba_cache_indices=plan.slots,factored_extend=plan)
            tail_metadata=NS(mamba_cache_indices=plan.slots[tail_rows],factored_extend=plan)
            backend=NS(factored=pool,forward_metadata=metadata,_track_mamba_state_decode=Mock())
            def prefix_forward(layer,sub,mixed,a,b,**kwargs):
                dense=torch.full((rows,2,16,16),(layer.layer_id+1)*0.015625)
                dense+=torch.eye(16)[None,None]*0.5
                tracked=dense.to(torch.bfloat16)
                track_slots=fb.mamba_track_indices;final_src=final_dst=None
                if getattr(fb,'cpu_aligned_checkpoint',False):
                    tracked=tracked[:1];track_slots=track_slots[:1]
                    final_src=plan.slots[1:];final_dst=fb.mamba_track_indices[1:]
                tx.add(layer.layer_id,plan,dense,tracked,track_slots,final_src,final_dst)
                return torch.zeros(1,mixed.shape[0],2,16)
            backend.forward_extend=prefix_forward
            def decode(layer,sub,mixed,a,b,**kwargs):
                return tx.decode(backend,layer,sub,mixed,a,b,torch.empty(0),torch.empty(0),tail_metadata.mamba_cache_indices)
            backend.forward_decode=decode
            pre=torch.arange(2*rows);last=torch.arange(2*rows,2*rows+len(tail_rows))
            context=(pd_shallow_gdn.split_boundary(backend,prefix,pre,boundary,last,metadata,tail_metadata)
                     if tail_rows else nullcontext())
            with context:
                for lid in IDS:
                    value=layer(lid)
                    size=2*rows+len(tail_rows) if lid<31 else 2*rows
                    backend.forward_extend(value,fb,torch.full((size,96),0.125),
                        torch.full((size,2),0.25),torch.full((size,2),0.5))
            w.calls.append('model-return')
            w.plans.append(plan);w.tails.append(tuple(tx.tails))
            state=rp.pd_boundary_state
            state.hidden[plan.slots]=torch.arange(10240,dtype=torch.float32).bfloat16()
            state.position[plan.slots]=7;state.valid[plan.slots]=1
            result=input_ids.float()*2+positions.float()
            w.outputs.append(result.clone())
            return result
        owner.forward=core
        runner=NS(model=owner,req_to_token_pool=rp,server_args=args,device='cpu',
                  forward=lambda fb,**kw:owner.forward(fb.input_ids,fb.positions,fb))
        w.runner=runner
        # Real attach imports the unchanged helper and installs its validator.
        if owner.pd_shallow_role=='prefill': pd_shallow.attach(owner,runner)
        def prewarm():
            g=pool._prefill_batch_graph=graph_module.PrefillBatchGraph()
            g.prewarm(pool,eager=native.factorize_layers,policy=POLICY)
            for buffers,captured in g.entries.values():
                captured.body=lambda buffers=buffers:buffers.evaluate(native.factorize_layers)
        pool.prewarm_commit_graph=prewarm
        pool.prewarm_k31_batch_graph=MethodType(native.FactoredGDNPool.prewarm_k31_batch_graph,pool)
        capture=NS(eager_runner=object(),prefill=NS(runner=None),decode=NS(runner=None),memory_usage=0,time_usage=0)
        stack.enter_context(patch.object(model_runner,'capture_cuda_graphs',return_value=capture))
        w.init=lambda:model_runner.ModelRunner.init_cuda_graphs(runner)
        if arm=='PC' and role=='prefill':
            w.init()
            w.pub=getattr(pool,'_pd_shallow_publication',None)
            w.runtime=runtime or Runtime()
            if w.pub is not None:w.pub.runtime=w.runtime
        else:
            w.pub=None;w.runtime=runtime or Runtime()
        w.handler=lambda:rp.pd_state_handoffs[HandoffKind.STATE_FACTOR]
        yield w


def request(sender,slot=1,index=0):
    return NS(rid='cpu-shallow',disagg_kv_sender=sender,
              kv=NS(mamba_pool_idx=torch.tensor(slot),req_pool_idx=index))


def field_bytes(w):
    fields=[getattr(w.pool,name) for name in ('a','U','W','count','stale','dense_of','dense_required','prefix_valid','dense_ring')]
    fields += [w.rp.pd_boundary_state.hidden,w.rp.pd_boundary_state.position,w.rp.pd_boundary_state.valid]
    return [t.contiguous().view(torch.uint8).numpy().tobytes() for t in fields]


def fake_sender():
    return FakeKVSender(NS(),http_server.FAKE_BOOTSTRAP_HOST,7,[0],0)


def warmup(w):
    ready,killed,payloads=Mock(),Mock(),[]
    class Response:
        status=200
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def read(self):return b'{}'
    class Session:
        def __init__(self,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        def post(self,url,*,json,ssl):
            payloads.append(json);sender=fake_sender()
            fb=batch();w.runner.forward(fb)
            req=request(sender)
            record=getattr(fb,'pd_publication_record',None)
            if record is not None:
                from sglang.srt.mem_cache.gdn_pd_overlap import bind_result_record
                bind_result_record(NS(forward_iter=record.batch_id,reqs=[req],req_to_token_pool=w.rp),
                                   NS(pd_publication_record=record))
            w.handler().before_send(req)
            if w.pub is not None:w.pub.ticket.complete()
            sender.send(np.array([1]),state_indices=[[1]])
            assert sender.poll()==KVPoll.Success
            return Response()
    tokenizer=NS(server_status=http_server.ServerStatus.Starting)
    options=dict(get_serving=lambda:NS(api_key=None,skip_tokenizer_init=True,skip_server_warmup=False),
        get_parallel=lambda:NS(dp_size=1),get_disagg=lambda:NS(disaggregation_mode='prefill',language_only=False,language_model_only=False),
        get_exec=lambda:NS(moe=NS(is_ep_scale_joiner=False)),
        get_model=lambda:NS(checkpoint_engine_wait_weights_before_ready=False,delete_ckpt_after_loading=False),
        get_observability=lambda:NS(debug_tensor_dump_input_file=None),ssl_verify_of=lambda _:False,is_mps=lambda:False,
        _global_state=NS(tokenizer_manager=tokenizer),_freeze_gc_after_server_warmup=Mock(),kill_process_tree=killed,
        aiohttp=NS(ClientSession=Session,ClientTimeout=lambda **kw:NS(**kw)),time=NS(sleep=lambda _:None))
    with ExitStack() as stack:
        for k,v in options.items():stack.enter_context(patch.object(http_server,k,v))
        stack.enter_context(patch.object(http_server.requests,'get',return_value=NS(status_code=200,json=lambda:{'is_generation':True})))
        http_server._wait_and_warmup(NS(url=lambda:'http://cpu-fixture'),launch_callback=ready)
    return NS(ready=ready,killed=killed,payloads=payloads,status=tokenizer.server_status)
