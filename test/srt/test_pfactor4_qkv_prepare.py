"""M6 real split/L2 and backend/pool byte gate, CPU replay then CUDA math.

CPU executes the Triton split/L2 interpreter and the native backend/pool;
convolution/recurrence are explicitly labelled ownership stubs on CPU only.
CUDA executes the native convolution and recurrence, including M5 interaction.
"""
import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.kernels.ops.attention.triton_gdn_fused_proj import fused_qkv_split_gdn_prefill
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNConfig, FactoredGDNPool, FactoredExtendPlan
from sglang.srt.mem_cache.gdn_prefill_qkv_prepare import prepare, eligible

FLAGS=dict(SGLANG_GDN_PREFILL_QKV_PREPARE='1',SGLANG_GDN_PREFILL_QKV_PREPARE_CHECK='0',
    SGLANG_GDN_PREFILL_BLOCK_GRAPH='0',SGLANG_GDN_PREFILL_BLOCK_BUCKETS='1',
    SGLANG_GDN_PREFILL_BLOCK_SHORT_ONLY='1',SGLANG_GDN_PREFILL_DYNAMIC_BIND='1',
    SGLANG_GDN_PREFILL_BLOCK_PRECAPTURE='0',SGLANG_GDN_PREFILL_BLOCK_GRAPH_CHECK='0',
    SGLANG_GDN_PSIDE_GRAPH='0',SGLANG_GDN_PSIDE_COMPOSITE='0',SGLANG_GDN_PREFILL_DENSE_GRAPH='0',
    SGLANG_GDN_FACTORED_DUMP='',SGLANG_GDN_AGG_COMMIT_GRAPH='0')


def same(x,y):
    return x.shape==y.shape and x.dtype==y.dtype and torch.equal(
        x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))


def baseline(mixed,h,hv,d):
    q,k,v=fused_qkv_split_gdn_prefill(mixed,h,h,hv,d,d,d)
    return l2norm_fwd(q),l2norm_fwd(k),v


def component(n,layout,dtype,device,*,special=False):
    h,hv,d=(8,24,128) if device=='cuda' else (2,6,16)
    width=(2*h+hv)*d
    if layout=='token':mixed=torch.randn(n,width,device=device,dtype=dtype)
    elif layout=='channel':mixed=torch.randn(width,n,device=device,dtype=dtype).T
    else:mixed=torch.randn(n,width*2,device=device,dtype=dtype)[:,::2]
    if special:mixed.zero_();mixed[:,::3]=-0.;mixed[:,::7]=1e-12
    saved=mixed.clone();expected=baseline(mixed,h,hv,d);got=prepare(mixed,h,hv,d)
    assert all(same(x,y) for x,y in zip(expected,got)),(n,layout,dtype)
    assert same(saved,mixed)
    retained=tuple(x.clone() for x in got);mixed.fill_(3);prepare(mixed,h,hv,d)
    assert all(same(x,y) for x,y in zip(retained,got))
    return dict(passed=True,n=n,layout=layout,dtype=str(dtype),stride=list(mixed.stride()),
                native_split_l2=True,bitwise=True,outputs_owned=True,special=special)


def guards():
    layer=NS(num_q_heads=2,num_k_heads=2,num_v_heads=6,head_q_dim=16,head_k_dim=16,head_v_dim=16)
    backend=NS(factored=object(),_stepwise_active=lambda fb:False,
        kernel_dispatcher=NS(extend_kernel=TritonGDNKernel(),extend_uses_state_checkpoints=False))
    mixed=torch.empty(17,160,dtype=torch.bfloat16);batch=NS()
    def admits(**kw):return eligible(backend,layer,batch,mixed,target_verify=kw.get('verify',False),cuda=kw.get('cuda',True))
    assert admits() and not admits(verify=True) and not admits(cuda=False)
    for flag in ('SGLANG_GDN_PSIDE_GRAPH','SGLANG_GDN_PREFILL_DENSE_GRAPH','SGLANG_GDN_FACTORED_DUMP'):
        with patch.dict(os.environ,{flag:'1'}):assert not admits()
    with patch.dict(os.environ,SGLANG_GDN_PREFILL_QKV_PREPARE='0'):assert not admits()
    backend.factored=None;assert not admits();backend.factored=object()
    backend._stepwise_active=lambda fb:True;assert not admits();backend._stepwise_active=lambda fb:False
    backend.kernel_dispatcher.extend_kernel=object();assert not admits()
    backend.kernel_dispatcher.extend_kernel=TritonGDNKernel()
    backend.kernel_dispatcher.extend_uses_state_checkpoints=True;assert not admits()
    backend.kernel_dispatcher.extend_uses_state_checkpoints=False
    layer.num_k_heads=1;assert not admits();layer.num_k_heads=2
    layer.head_v_dim=8;assert not admits();layer.head_v_dim=16
    mixed=torch.empty(0,160);assert not admits()
    return dict(passed=True,checks=13,normalization_ownership=True,unsupported_paths_native=True)


def kernel_contract(device):
    # Exercise the actual dispatcher method; numerical recurrence has its own GPU gate below.
    kernel=TritonGDNKernel();tensor=torch.empty(1,device=device);calls=[]
    def record(**kw):calls.append(kw['use_qk_l2norm_in_kernel']);return None,None,None
    module='sglang.srt.layers.attention.linear.kernels.gdn_triton'
    with patch(module+'.chunk_gated_delta_rule',record),patch(module+'.is_cpu',return_value=False), \
            patch(module+'.is_npu',return_value=False),patch(module+'.is_xpu',return_value=False):
        for ready in (False,True):
            kernel.extend(tensor,tensor,tensor,tensor,tensor,ssm_states=tensor,
                cache_indices=tensor,query_start_loc=tensor,factored_qk_ready=ready)
        try:
            kernel.extend(tensor,tensor,tensor,tensor,tensor,ssm_states=tensor,
                cache_indices=tensor,query_start_loc=tensor,factored_qk_ready=tensor)
        except TypeError:pass
        else:raise AssertionError('Device tensor used as host normalization metadata')
    assert calls==[True,False]
    return dict(passed=True,raw_normalized=True,prepared_not_normalized_twice=True)


def backend_pool(n,B,block,device,constants):
    h,hv,d=(8,24,128) if device=='cuda' else (1,2,128)
    width=(2*h+hv)*d;torch.manual_seed(0x6046+n+B)
    raw=torch.randn(n,width,device=device,dtype=torch.bfloat16)*.1
    log=torch.randn(hv,device=device);a=torch.randn(n,hv,device=device,dtype=torch.bfloat16)
    b=torch.randn_like(a);bias=torch.randn(width,device=device,dtype=torch.bfloat16)*.1
    weights=torch.randn(width,4,device=device,dtype=torch.bfloat16)*.1
    cfg=FactoredGDNConfig(r=8,m=8,dtype=torch.float16,ring=B,strict_chunk=1,
        factored_prefix=1,init_method='k31',vbar_path=str(constants))
    lengths=[n//B]*(B-1)+[n-(n//B)*(B-1)]
    results=[];published=[];activations=[]
    for enabled in ('0','1'):
        pool=FactoredGDNPool(size=8,cache_params=NS(shape=NS(temporal=(hv,d,d))),
            mamba_layer_ids=[0],device=device,cfg=copy.copy(cfg))
        slots=torch.arange(1,B+1,device=device);conv=torch.zeros(8,width,3,device=device,dtype=torch.bfloat16)
        cu=torch.tensor([0]+[sum(lengths[:i]) for i in range(1,B+1)],device=device,dtype=torch.int32)
        plan=FactoredExtendPlan(slots=slots,use_ring=torch.zeros(B,device=device,dtype=torch.bool),
            ring_src=torch.zeros(B,device=device,dtype=torch.long),ring_dst=torch.arange(B,device=device),
            ring_dst_rows=torch.arange(B,device=device),all_fresh=True,last_layer=0,
            dense_required_after_commit=torch.zeros(B,device=device,dtype=torch.int32))
        m=ForwardMetadata(query_start_loc=cu,mamba_cache_indices=slots,factored_extend=plan,
            has_mamba_track_mask=True,track_conv_indices=torch.tensor([[0,1,2]],device=device),
            conv_states_mask_indices=torch.tensor([4],device=device),
            track_ssm_h_src=torch.tensor([0],device=device),track_ssm_h_dst=torch.tensor([4],device=device),
            track_ssm_final_src=slots,track_ssm_final_dst=torch.arange(5,5+B,device=device))
        layer=NS(layer_id=0,A_log=log,dt_bias=torch.zeros(hv,device=device,dtype=torch.bfloat16),
            num_q_heads=h,num_k_heads=h,num_v_heads=hv,head_q_dim=d,head_k_dim=d,head_v_dim=d,
            q_dim=h*d,k_dim=h*d,v_dim=hv*d,conv_weights=weights,bias=bias,activation='silu')
        batch=NS(batch_size=B,forward_mode=NS(is_target_verify=lambda:False),
            extend_prefix_lens=torch.zeros(B,device=device,dtype=torch.int32),extend_seq_lens_cpu=lengths)
        kernel=TritonGDNKernel();cpu_calls=[]
        if device=='cpu':
            def cpu_extend(q,k,v,g,beta,ssm_states,cache_indices,query_start_loc,**kw):
                ready=kw.get('factored_qk_ready',False);cpu_calls.append(ready)
                normalized=q if ready else l2norm_fwd(q.contiguous())
                ssm_states.add_(normalized.float().sum())
                return v.clone(),None,ssm_states.unsqueeze(0).clone()
            kernel.extend=cpu_extend
        backend=object.__new__(GDNAttnBackend);backend.factored=pool;backend.mis_metadata=None
        backend.kernel_dispatcher=NS(extend_kernel=kernel,extend=kernel.extend,extend_uses_state_checkpoints=False)
        backend._model_runner=NS(server_args=NS(disaggregation_mode='null'))
        backend._stepwise_active=lambda fb:False;backend.forward_metadata=m
        backend.req_to_token_pool=NS(mamba2_layer_cache=lambda li:NS(conv=[conv],temporal=torch.empty(0,device=device)))
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ,SGLANG_GDN_PREFILL_QKV_PREPARE=enabled,
                SGLANG_GDN_PREFILL_BLOCK_GRAPH=str(int(block))))
            stack.enter_context(patch('sglang.srt.distributed.get_tensor_model_parallel_rank',return_value=0))
            if device=='cpu':
                # CPU replay owns routing; CUDA below owns numerical conv/recurrence admission.
                module='sglang.srt.layers.attention.linear.gdn_backend'
                stack.enter_context(patch(module+'.is_cuda',return_value=True))
                stack.enter_context(patch(module+'.fused_qkv_split_gdn_prefill',fused_qkv_split_gdn_prefill,create=True))
                stack.enter_context(patch(module+'.causal_conv1d_fn',lambda x,*args,**kw:x.clone()))
            out=backend.forward_extend(layer,batch,raw.clone(),a,b)
            if block and B==1:
                assert backend._agg_prefill_block_graph.qk_prepared==(enabled=='1')
                from sglang.srt.mem_cache.gdn_agg_prefill import run
                q,k,v=prepare(raw,h,hv,d)
                try:
                    run(backend,layer,q,k,v,a,b,torch.zeros(1,hv,d,d,device=device),
                        torch.tensor([0],device=device),cu,m,qk_prepared=enabled!='1')
                except RuntimeError as error:
                    assert 'normalization ownership' in str(error)
                else:raise AssertionError('Captured graph reused with a different normalization owner')
        results.append(out)
        fields=('a','U','W','count','stale','dense_of','dense_required','dense_ring','prefix_valid')
        published.append({name:getattr(pool,name).clone() for name in fields}|{'conv':conv.clone()})
        activations.append(dict(prepared_layers=sorted(getattr(backend,'_qkv_prepare_receipts',set())),
            graph=hasattr(backend,'_agg_prefill_block_graph'),cpu_normalization_owner=cpu_calls))
    checks={name:same(published[0][name],published[1][name]) for name in published[0]}
    checks['output']=same(*results);assert all(checks.values()),(n,B,block,checks)
    assert activations[0]['prepared_layers']==[] and activations[1]['prepared_layers']==[0]
    assert all(r['graph']==(block and B==1) for r in activations)
    if device=='cpu':assert all(not r for r in activations[0]['cpu_normalization_owner']) and all(activations[1]['cpu_normalization_owner'])
    return dict(passed=True,n=n,B=B,block=block,checks=checks,activations=activations,
        native_forward=True,native_pool=True,conv_and_recurrence_math=device=='cuda',bias=True,
        tracked_checkpoint=True,final_prefix=True)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    path=Path(a.output);device='cpu' if os.environ.get('TRITON_INTERPRET')=='1' else 'cuda'
    torch.manual_seed(0x6046)
    with patch.dict(os.environ,FLAGS):
        rows=[component(n,layout,dtype,device,special=n==17) for n in (1,17,65,263)
              for layout in ('token','channel','strided') for dtype in (torch.bfloat16,torch.float16,torch.float32)]
        if device=='cuda':rows += [component(n,layout,torch.bfloat16,device) for n in (8192,32768) for layout in ('token','channel')]
        guard=guards();contract=kernel_contract(device)
        constants=path.with_name('qkv-vbar-'+device+'.pt')
        torch.save({'vbar':torch.zeros(1,24 if device=='cuda' else 2,128)},constants)
        pools=[backend_pool(n,B,block,device,constants) for n,B,block in
               ((17,1,False),(263 if device=='cuda' else 65,2,False),
                (263 if device=='cuda' else 17,1,True),(263 if device=='cuda' else 65,2,True))]
    out=dict(passed=True,complete=True,device=device,model_dtype='bfloat16',factor_dtype='float16',
        rows=rows,guards=guard,kernel_contract=contract,pools=pools,production_default=False,
        scope='Native split/L2 bytes; real backend and pool; CPU conv/recurrence routing stubs labelled; GPU native math.')
    path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)


if __name__=='__main__':main()
