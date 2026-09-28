"""Shared P layer graphs with native arithmetic and task-owned input storage.

Each entry is independent of model-layer addresses. A single binding kernel
copies the current layer parameters, true token rows and selected state slot.
Only local buffers are changed during capture/replay; the caller publishes
states in the original pool order after replay. Production dispatch is opt-in.
"""
from collections import OrderedDict
import gc
import json
import os
import torch

from sglang.srt.utils.graph_capture import graph_capture_lock
import triton
import triton.language as tl

from .gdn_pside_prefill import bucket


def eligible(backend, layer, batch, raw, metadata, conv):
    if (os.environ.get('SGLANG_GDN_PSIDE_COMPOSITE') != '1'
            or os.environ.get('SGLANG_GDN_PSIDE_GRAPH') != '1'):
        return False
    if backend._model_runner.server_args.disaggregation_mode != 'prefill':
        raise RuntimeError('composite P graph cannot run on D or AGG')
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
    pool=backend.factored;plan=getattr(metadata,'factored_extend',None)
    return bool(pool is not None and pool.batch_prefill and plan is not None
        and plan.slots.numel()==1 and plan.n_ring_src in (0,1)
        and metadata.query_start_loc.numel()==2
        and isinstance(backend.kernel_dispatcher.extend_kernel,TritonGDNKernel)
        and raw.is_cuda and not torch.cuda.is_current_stream_capturing()
        and bucket(raw.shape[0]) is not None and conv.is_contiguous()
        and pool.prefix_dense is None and layer.bias is None
        and layer.num_q_heads==layer.num_k_heads and layer.head_q_dim==layer.head_k_dim
        and batch.extend_seq_lens_cpu is not None
        and list(batch.extend_seq_lens_cpu)==[raw.shape[0]]
        and not batch.forward_mode.is_target_verify()
        and not backend._stepwise_active(batch)
        and getattr(backend,'mis_metadata',None) is None
        and backend._factored_batch_trunc and backend._factored_side_stream is None
        and getattr(metadata,'state_checkpoint_cu_starts',None) is None)


def run_layer(backend, layer, conv, plan, raw, a, b, *, finish, tail=None):
    graph=getattr(backend,'_pside_layer_graph',None)
    if graph is None:graph=backend._pside_layer_graph=PsideLayerGraph()
    value=graph.run(backend.factored,layer,conv,plan,raw,a,b,finish=finish,tail=tail)
    seen=backend.__dict__.setdefault('_pside_layer_receipts',set())
    key=(layer.layer_id,raw.shape[0],plan.all_fresh,plan.n_ring_src,finish)
    if key not in seen:
        seen.add(key)
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        print('PSIDE_LAYER_GRAPH '+json.dumps(dict(rank=get_tensor_model_parallel_rank(),
            layer=layer.layer_id,tokens=raw.shape[0],bucket=bucket(raw.shape[0]),
            finish=finish,all_fresh=plan.all_fresh,ring=plan.n_ring_src,stats=graph.stats)),flush=True)
    return value


@triton.jit
def _bind(SOURCES, DESTS, SLOT, RING_SLOT, REAL_END, CONV_CU,
          N, COUNTS:tl.constexpr, WIDTHS:tl.constexpr, STRIDES:tl.constexpr,
          KINDS:tl.constexpr, PADS:tl.constexpr, RING_ACTIVE:tl.constexpr,
          BLOCK:tl.constexpr):
    pid=tl.program_id(0)
    if pid==0:
        tl.store(REAL_END,N)
        tl.store(CONV_CU,0)
        tl.store(CONV_CU+1,N)
    begin=0
    for i in tl.static_range(len(COUNTS)):
        blocks=tl.cdiv(COUNTS[i],BLOCK)
        if pid>=begin and pid<begin+blocks:
            x=(pid-begin)*BLOCK+tl.arange(0,BLOCK)
            mask=x<COUNTS[i]
            if KINDS[i]==1:
                row=x//WIDTHS[i]
                valid=mask & (row<N)
                value=tl.load(SOURCES[i]+row*STRIDES[i]+x%WIDTHS[i],valid,other=0)
                value=tl.where(valid,value.to(tl.float32),PADS[i]).to(value.dtype)
            elif KINDS[i]==2:
                slot=tl.load(SLOT).to(tl.int64)
                value=tl.load(SOURCES[i]+slot*STRIDES[i]+x,mask & (slot>=0),other=0)
            elif KINDS[i]==3:
                slot=tl.load(RING_SLOT).to(tl.int64)
                value=tl.load(SOURCES[i]+slot*STRIDES[i]+x,mask & RING_ACTIVE & (slot>=0),other=0)
            else:
                value=tl.load(SOURCES[i]+x,mask,other=0)
            tl.store(DESTS[i]+x,value,mask)
        begin+=blocks


def input_layout(pool, layer, conv, qkv, a, b, tail, *, finish):
    """(tensor, kind, owned shape, padding), where kind 2/3 gathers one slot."""
    li=pool.layer_index(layer.layer_id)
    n=qkv.shape[0];capacity=bucket(n)
    if capacity is None or pool.prefix_dense is not None or layer.bias is not None:
        raise ValueError('unsupported composite pool/conv configuration')
    if not conv.is_contiguous():raise ValueError('composite conv pool must be contiguous')
    inputs={
        'raw':(qkv,1,(capacity,qkv.shape[-1]),0.),
        'a':(a,1,(capacity,a.shape[-1]),float('-inf')),
        'b':(b,1,(capacity,b.shape[-1]),float('-inf')),
        'conv':(conv,2,(1,)+tuple(conv.shape[1:]),0.),
        'fa':(pool.a[li],2,(1,)+tuple(pool.a.shape[2:]),0.),
        'fu':(pool.U[li],2,(1,)+tuple(pool.U.shape[2:]),0.),
        'fw':(pool.W[li],2,(1,)+tuple(pool.W.shape[2:]),0.),
        'count':(pool.count[li],2,(1,)+tuple(pool.count.shape[2:]),0.),
        'ring':(pool.dense_ring[li],3,(1,pool.hv,pool.v,pool.k),0.),
        'vbar':(pool.vbar[li],0,tuple(pool.vbar[li].shape),0.),
        'log':(layer.A_log,0,tuple(layer.A_log.shape),0.),
        'bias':(layer.dt_bias,0,tuple(layer.dt_bias.shape),0.),
        'weight':(layer.conv_weights,0,tuple(layer.conv_weights.shape),0.)}
    if finish:
        if tail is None:tail=(qkv[:1],a[:1],b[:1])  # unused tail for prefix-only emitters
        for name,x in zip(('tail_raw','tail_a','tail_b'),tail):
            inputs[name]=(x,0,tuple(x.shape),0.)
    for name,(x,kind,shape,pad) in inputs.items():
        if kind==1:
            if x.ndim!=2 or x.stride(-1)!=1:raise ValueError('unsupported token strides: '+name)
        elif not x.is_contiguous():raise ValueError('non-contiguous non-token input: '+name)
    return inputs


class LayerEntry:
    def __init__(self, inputs, capacity, mode, finish, layer, cfg):
        if mode not in ('fresh','cached','ring'):raise ValueError('unknown P initial-state mode')
        self.capacity,self.mode,self.finish=capacity,mode,finish
        self.cfg=cfg
        self.h,self.hv,self.k,self.v=layer.num_q_heads,layer.num_v_heads,layer.head_k_dim,layer.head_v_dim
        self.activation=layer.activation
        self.buffers={name:torch.empty(shape,device=x.device,dtype=x.dtype)
                      for name,(x,kind,shape,pad) in inputs.items()}
        device=inputs['raw'][0].device
        self.cu=torch.tensor([0,capacity],dtype=torch.int32,device=device)
        self.conv_cu=torch.tensor([0,capacity],dtype=torch.int32,device=device)
        self.real_end=torch.tensor([capacity],dtype=torch.int32,device=device)
        self.rows=torch.zeros(1,dtype=torch.int32,device=device)
        self.slots=torch.zeros(1,dtype=torch.int64,device=device)
        self.row_numbers=torch.arange(capacity,device=device)
        self.has_initial=torch.tensor([mode!='fresh'],dtype=torch.bool,device=device)
        self.omega=(torch.randn(1,self.hv,self.v,cfg.r+cfg.init_oversample,
            generator=torch.Generator(device=device).manual_seed(0),device=device) if finish else None)
        self.graph=None;self.outputs=None;self.pinned=()

    def bind(self, inputs, plan, n):
        sources=tuple(item[0] for item in inputs.values())
        dests=tuple(self.buffers.values())
        kinds=tuple(item[1] for item in inputs.values())
        counts=tuple(t.numel() for t in dests)
        widths=tuple(x.shape[-1] if kind==1 else count for x,kind,count in zip(sources,kinds,counts))
        strides=tuple(x.stride(0) if kind in (1,2,3) else 0 for x,kind in zip(sources,kinds))
        _bind[(sum(triton.cdiv(n,1024) for n in counts),)](sources,dests,plan.slots,plan.ring_src,
            self.real_end,self.conv_cu,n,counts,widths,strides,kinds,
            tuple(item[3] for item in inputs.values()),self.mode=='ring',1024,num_warps=4)

    def evaluate(self):
        from sglang.srt.mem_cache.gdn_factored_pool import densify,factorize_layers
        from sglang.srt.layers.attention.linear.gdn_backend import causal_conv1d_fn
        from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
        from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
        from sglang.srt.layers.attention.linear.kernels.gdn_factored import factored_packed_decode
        from .gdn_prefill_block_pad import chunk_padded
        t=self.buffers
        if self.mode=='fresh':
            state=torch.zeros((1,self.hv,self.v,self.k),dtype=torch.float32,device=t['raw'].device)
        elif self.mode=='ring':state=t['ring'].clone()
        else:state=densify(t['fa'],t['fu'],t['fw'],t['count'],t['vbar']).contiguous()
        mixed=causal_conv1d_fn(t['raw'].clone().transpose(0,1),t['weight'],bias=None,
            activation=self.activation,conv_states=t['conv'],has_initial_state=self.has_initial,
            cache_indices=self.slots,query_start_loc=self.conv_cu,seq_lens_cpu=[self.capacity])
        mixed=torch.where(self.row_numbers[None,:]<self.real_end,mixed,0).transpose(0,1)
        q,k,v=torch.split(mixed,(self.h*self.k,self.h*self.k,self.hv*self.v),dim=-1)
        q=q.reshape(1,self.capacity,self.h,self.k).contiguous()
        k=k.reshape(1,self.capacity,self.h,self.k).contiguous()
        v=v.reshape(1,self.capacity,self.hv,self.v).contiguous()
        g,beta=fused_gdn_gating(t['log'],t['a'],t['b'],t['bias'])
        output,_,h=chunk_padded(q,k,v,g,beta,state,self.rows,self.cu,self.real_end)
        result=dict(output=output,dense=state,h=h,conv=t['conv'])
        if self.finish:
            factors=factorize_layers([state],t['vbar'].unsqueeze(0),self.cfg,omega=self.omega)[0]
            # The published N-1 prefix survives the private recurrent append.
            result.update(prefix=tuple(x.clone() for x in factors),prefix_conv=t['conv'].clone())
            fa,fu,fw=(x.contiguous() for x in factors)
            count=torch.full((1,self.hv),self.cfg.r,dtype=torch.int32,device=fa.device)
            stale=torch.zeros(1,dtype=torch.int32,device=fa.device)
            tail=causal_conv1d_update(t['tail_raw'].clone(),t['conv'],t['weight'],None,
                self.activation,conv_state_indices=self.slots)
            tail_out=factored_packed_decode(tail,t['tail_a'],t['tail_b'],A_log=t['log'],dt_bias=t['bias'],
                scale=self.k**-0.5,vbar=t['vbar'],fa=fa,fu=fu,fw=fw,fcount=count,stale=stale,
                ssm_state_indices=self.slots,num_q_heads=self.h,num_v_heads=self.hv,
                head_k_dim=self.k,head_v_dim=self.v,r=self.cfg.r,rfull=self.cfg.rfull,
                truncate=False,async_stream=None,**self.cfg.kernel_kwargs())
            result.update(tail=tail_out.transpose(0,1),live=(fa,fu,fw),live_count=count)
        return result

    def capture(self, inputs, plan, n):
        from .gdn_prefill_block_graph import pin_chunk_metadata
        self.bind(inputs,plan,n)
        self.pinned=pin_chunk_metadata(self.cu,self.capacity)
        stream=torch.cuda.Stream();current=torch.cuda.current_stream();stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):self.bind(inputs,plan,n);self.evaluate()
        current.wait_stream(stream)
        self.bind(inputs,plan,n)
        self.graph=torch.cuda.CUDAGraph()
        enabled=gc.isenabled();gc.disable()
        try:
            with graph_capture_lock, torch.cuda.graph(self.graph,stream=stream,capture_error_mode='thread_local'):
                self.outputs=self.evaluate()
        finally:
            if enabled:gc.enable()
        self.stream=stream


class PsideLayerGraph:
    def __init__(self,max_entries=24):
        self.entries=OrderedDict();self.max_entries=max_entries
        self.stats=dict(captured=0,replayed=0,evicted=0)

    def run(self,pool,layer,conv,plan,qkv,a,b,*,finish,tail=None):
        if not qkv.is_cuda or torch.cuda.is_current_stream_capturing():
            raise ValueError('P layer graph requires a non-capturing CUDA caller')
        if plan.slots.numel()!=1 or plan.n_ring_src not in (0,1):raise ValueError('singleton P plan required')
        inputs=input_layout(pool,layer,conv,qkv,a,b,tail,finish=finish)
        n=qkv.shape[0];capacity=bucket(n)
        mode='fresh' if plan.all_fresh else 'ring' if plan.n_ring_src else 'cached'
        key=(capacity,mode,finish,repr(pool.cfg),layer.activation,
             tuple((name,shape,x.dtype,x.device) for name,(x,kind,shape,pad) in inputs.items()),
             torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        if key not in self.entries:
            entry=LayerEntry(inputs,capacity,mode,finish,layer,pool.cfg)
            entry.capture(inputs,plan,n);self.entries[key]=entry;self.stats['captured']+=1
            while len(self.entries)>self.max_entries:
                self.entries.popitem(last=False);self.stats['evicted']+=1
        entry=self.entries[key];self.entries.move_to_end(key)
        entry.bind(inputs,plan,n);entry.graph.replay();self.stats['replayed']+=1
        # Output/dense can outlive this call (cross-layer batched commits).
        # Small publication tensors are consumed before the next graph call.
        result=dict(entry.outputs)
        result['output']=result['output'][:,:n].clone()
        result['dense']=result['dense'].clone()
        if finish:result['tail']=result['tail'].clone()
        return result


def tracked_factors(pool, li, track_dense):
    """Capture the separate radix-checkpoint factor call, preserving its seed/shape."""
    from . import gdn_factored_pool as native
    from .gdn_prefill_factor_graph import PrefillFactorGraph
    graph = getattr(pool, '_pside_track_factor_graph', None)
    if graph is None:
        graph = pool._pside_track_factor_graph = PrefillFactorGraph()
    return graph.run([track_dense], pool.vbar[li:li+1], pool.cfg,
        eager=native.factorize_layers,
        policy=(native.ORTH_METHOD, native.ORTH_WARPS_OVERRIDE, native.factorize_dense))[0]


def publish_split(pool, layer_id, plan, result, *, final_src, final_dst, has_tail,
                  track_dense=None, track_slots=None):
    """Native publication order for one split layer; private tail never leaks.

    Conv tracking remains the caller's original metadata operation. A tail
    marks the live slot stale without discarding its ring ownership: this
    differs from a radix destination, which is deliberately factored-only.
    """
    from sglang.srt.layers.attention.linear.kernels.gdn_factored_io import store_factored
    from sglang.srt.layers.attention.linear.kernels.gdn_factored import factored_expiry_truncate_layers
    li=pool.layer_index(layer_id)
    if plan.pending or plan.next_layer!=li or plan.last_layer!=li:
        raise ValueError('split publication requires an empty one-layer plan')
    if li==0 and pool.cfg.factored_prefix:
        pool.invalidate_prefix_dense(plan.slots)
        if track_slots is not None:pool.invalidate_prefix_dense(track_slots)
    banks=tuple(getattr(pool,name)[li] for name in ('a','U','W','count'))
    store_factored(*result['prefix'],*banks,pool.stale,pool.dense_of,plan.slots,pool.cfg.r,
        stale_value=0,dense=result['dense'],ring=pool.dense_ring[li],ring_dst=plan.ring_dst)
    pool.save_prefix_dense(layer_id,plan.slots,result['dense'])
    if track_dense is not None:
        # Native tracked checkpoints form their own factorize_layers call;
        # never concatenate them with the final dense state (GEMM shape/seed).
        tracked=tracked_factors(pool,li,track_dense)
        store_factored(*tracked,*banks,pool.stale,pool.dense_of,track_slots,pool.cfg.r,stale_value=1)
        pool.save_prefix_dense(layer_id,track_slots,track_dense)
    if pool.dense_required is not None:
        pool.dense_required[plan.slots.clamp_min(0)]=plan.dense_required_after_commit
    plan.next_layer+=1
    if final_src is not None and final_src.numel():pool.copy_slots_layer(layer_id,final_src,final_dst)
    if has_tail:
        if li==0:pool.invalidate_prefix_dense(plan.slots)
        # stale_value=1 would also clear dense_of, unlike native packed decode.
        store_factored(*result['live'],*banks,pool.stale,pool.dense_of,plan.slots,pool.cfg.r+1,stale_value=0)
        pool.stale.index_fill_(0,plan.slots,1)
        if pool.is_last_layer(layer_id):
            factored_expiry_truncate_layers(pool.U,pool.W,pool.count,plan.slots,pool.cfg.r,pool.cfg.rfull)
