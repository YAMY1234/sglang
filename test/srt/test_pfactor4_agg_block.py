"""M5 CPU binding/real-backend contract replay, then GPU bytewise admission.

CPU ownership tests use a labelled evaluator; they do not claim CUDA math.
GPU uses the served Triton dispatcher and compares output/state/checkpoints,
then all published bytes in a real FP16 factor pool.
"""
import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNConfig, FactoredGDNPool, FactoredExtendPlan
from sglang.srt.mem_cache.gdn_agg_prefill import eligible
from sglang.srt.mem_cache.gdn_agg_prefill_block_graph import PrefillBlockGraph, check_result

FLAGS=dict(SGLANG_GDN_PREFILL_BLOCK_GRAPH='1',SGLANG_GDN_PREFILL_BLOCK_BUCKETS='1',
    SGLANG_GDN_PREFILL_BLOCK_SHORT_ONLY='1',SGLANG_GDN_PREFILL_DYNAMIC_BIND='1',
    SGLANG_GDN_PREFILL_BLOCK_PRECAPTURE='0',SGLANG_GDN_PREFILL_BLOCK_GRAPH_CHECK='0',
    SGLANG_GDN_PSIDE_GRAPH='0',SGLANG_GDN_PREFILL_DENSE_GRAPH='0')


def tensors(n,device,*,full=False):
    hq,hv,width=(8,24,128) if full else (1,2,4)
    mixed=torch.randn(1,n,(2*hq+hv)*width,device=device,dtype=torch.bfloat16)*.05
    # The production projection supplies noncontiguous Q/K/V slices.
    return dict(q=mixed[:,:,:hq*width].view(1,n,hq,width),
        k=mixed[:,:,hq*width:2*hq*width].view(1,n,hq,width),
        v=mixed[:,:,2*hq*width:].view(1,n,hv,width),
        a=torch.randn(n,hv,device=device,dtype=torch.bfloat16),
        b=torch.randn(n,hv,device=device,dtype=torch.bfloat16),
        log=torch.randn(hv,device=device),bias=torch.randn(hv,device=device,dtype=torch.bfloat16),
        state=torch.randn(1,hv,width,width,device=device)*.01,
        rows=torch.tensor([0],device=device,dtype=torch.int32),
        cu=torch.tensor([0,n],device=device,dtype=torch.int32))


def ownership_only(t):
    n=int(t['cu'][-1])
    t['state'].add_(t['q'][:,:n].float().sum())
    out=t['v']*t['log'][0]
    h=t['state'].unsqueeze(1).repeat(1,(n+63)//64,1,1,1)
    return out,None,h


def real(t):
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    gate,beta=fused_gdn_gating(t['log'],t['a'],t['b'],t['bias'])
    return TritonGDNKernel().extend(q=t['q'],k=t['k'],v=t['v'],g=gate,beta=beta,
        ssm_states=t['state'],cache_indices=t['rows'],query_start_loc=t['cu'])


def guards(device):
    t=tensors(263,device)
    kernel=TritonGDNKernel()
    backend=NS(_model_runner=NS(server_args=NS(disaggregation_mode='null')),
        kernel_dispatcher=NS(extend_kernel=kernel,extend_uses_state_checkpoints=False))
    m=ForwardMetadata(query_start_loc=t['cu'],mamba_cache_indices=t['rows'])
    def admits(q=t['q'],rows=t['rows'],cu=t['cu'],capturing=False):
        return eligible(backend,q,rows,cu,m,capturing=capturing)
    assert admits()
    for role in ('prefill','decode'):
        backend._model_runner.server_args.disaggregation_mode=role
        assert not admits(),role
    backend._model_runner.server_args.disaggregation_mode='null'
    assert not admits(capturing=True)
    assert not admits(rows=torch.tensor([0,1],device=device))
    assert not admits(cu=torch.tensor([0,100,263],device=device))
    assert not admits(q=torch.empty(1,8193,1,4,device=device))
    m.track_ssm_recompute_dst=torch.tensor([1],device=device);assert not admits()
    m.track_ssm_recompute_dst=None
    backend.kernel_dispatcher.extend_uses_state_checkpoints=True;assert not admits()
    backend.kernel_dispatcher.extend_uses_state_checkpoints=False
    backend.kernel_dispatcher.extend_kernel=object();assert not admits()
    return dict(passed=True,checks=9,native_metadata=True,native_kernel_type=True)


def pool_case(device,n,constants):
    heads=24 if device=='cuda' else 2
    width=128
    cfg=FactoredGDNConfig(r=8,m=8,dtype=torch.float16,ring=1,strict_chunk=1,
        factored_prefix=1,init_method='k31',vbar_path=str(constants))
    t=tensors(n,device,full=device=='cuda')
    if device=='cpu':
        # The pool keeps real K/V=128; only its recurrence is an ownership stub.
        t=tensors(n,device,full=True)
        t.update(q=t['q'][:,:,:1],k=t['k'][:,:,:1],v=t['v'][:,:,:2],
            a=t['a'][:,:2],b=t['b'][:,:2],log=t['log'][:2],bias=t['bias'][:2])
    layer=NS(layer_id=0,A_log=t['log'],dt_bias=t['bias'])
    outputs=[];pools=[];stats=[]
    for enabled in ('0','1'):
        pool=FactoredGDNPool(size=4,cache_params=NS(shape=NS(temporal=(heads,width,width))),
            mamba_layer_ids=[0],device=device,cfg=copy.copy(cfg))
        slots=torch.tensor([1],device=device)
        plan=FactoredExtendPlan(slots=slots,use_ring=torch.zeros(1,device=device,dtype=torch.bool),
            ring_src=torch.zeros(1,device=device,dtype=torch.long),ring_dst=torch.tensor([0],device=device),
            ring_dst_rows=torch.tensor([0],device=device),all_fresh=True,last_layer=0,
            dense_required_after_commit=torch.zeros(1,device=device,dtype=torch.int32))
        m=ForwardMetadata(query_start_loc=t['cu'],mamba_cache_indices=slots,factored_extend=plan,
            has_mamba_track_mask=True,track_ssm_h_src=torch.tensor([0],device=device),
            track_ssm_h_dst=torch.tensor([2],device=device),
            track_ssm_final_src=slots,track_ssm_final_dst=torch.tensor([3],device=device))
        kernel=TritonGDNKernel()
        if device=='cpu':
            def cpu_extend(q,k,v,g,beta,ssm_states,cache_indices,query_start_loc,**kw):
                n=int(query_start_loc[-1]);ssm_states.add_(q[:,:n].float().sum())
                return v.clone(),None,ssm_states.unsqueeze(1).repeat(1,(n+63)//64,1,1,1)
            kernel.extend=cpu_extend
        dispatcher=NS(extend_kernel=kernel,extend=kernel.extend,extend_uses_state_checkpoints=False)
        backend=object.__new__(GDNAttnBackend)
        backend.factored=pool;backend.kernel_dispatcher=dispatcher
        backend._model_runner=NS(server_args=NS(disaggregation_mode='null'))
        backend._stepwise_active=lambda fb:False
        with patch.dict(os.environ,SGLANG_GDN_PREFILL_BLOCK_GRAPH=enabled), \
                patch('sglang.srt.distributed.get_tensor_model_parallel_rank',return_value=0):
            out=backend._forward_extend_factored(layer=layer,forward_batch=NS(batch_size=1),
                query=t['q'],key=t['k'],value=t['v'],a=t['a'],b=t['b'],
                query_start_loc=t['cu'],forward_metadata=m,output=None)
        outputs.append(out);pools.append(pool)
        stats.append(dict(backend._agg_prefill_block_graph.stats) if enabled=='1' else None)
    fields=('a','U','W','count','stale','dense_of','dense_required','dense_ring','prefix_valid')
    checks={name:torch.equal(getattr(pools[0],name).contiguous().view(torch.uint8),
                            getattr(pools[1],name).contiguous().view(torch.uint8)) for name in fields}
    checks['output']=torch.equal(outputs[0].contiguous().view(torch.uint8),outputs[1].contiguous().view(torch.uint8))
    assert all(checks.values()),checks
    return dict(passed=True,tokens=n,checks=checks,graph=stats[1],native_backend=True,native_pool=True,
        native_recurrence=device=='cuda',tracked_checkpoint=True,final_prefix=True)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    output=Path(args.output);device='cpu' if os.environ.get('TRITON_INTERPRET')=='1' else 'cuda'
    torch.manual_seed(0x504634)
    graph=PrefillBlockGraph(bucketed=True);rows=[];retained=[]
    lengths=(1,16,17,31,32,33,63,64,65,127,128,129,255,256,257,263,
             511,512,527,1023,1024,2047,2048,4095,4096,6687,8191,8192,263,17)
    with patch.dict(os.environ,FLAGS):
        guard_result=guards(device)
        evaluate=real if device=='cuda' else ownership_only
        for n in lengths:
            for layer in range(2):
                t=tensors(n,device,full=device=='cuda')
                reference={k:v.clone() for k,v in t.items()}
                expected=evaluate(reference)
                if device=='cuda':
                    from sglang.srt.mem_cache.gdn_prefill_block_graph import pin_chunk_metadata
                    for extra in range(6):
                        pin_chunk_metadata(torch.tensor([0,65+extra],device=device,dtype=torch.int32),65+extra)
                got=graph.run(t,evaluate)
                checks=check_result(got,expected,reference['state'])
                for value,saved in retained:
                    assert torch.equal(value.contiguous().view(torch.uint8),saved.contiguous().view(torch.uint8))
                retained=[(got[0],got[0].clone()),(got[1],got[1].clone())]
                rows.append(dict(tokens=n,layer=layer,checks=checks))
        assert len(graph.entries)==10,len(graph.entries)
        if device=='cuda':assert graph.stats['captured']==10 and graph.stats['replayed']==len(rows)
        constants=output.with_name('agg-block-vbar-'+device+'.pt')
        torch.save({'vbar':{0:torch.randn(24 if device=='cuda' else 2,128)}},constants)
        pools=[pool_case(device,n,constants) for n in (65,263)]
        # Exercise the startup precapture route independently from real data.
        pre=PrefillBlockGraph(bucketed=True)
        pre.precapture(tensors(17,device,full=device=='cuda'),evaluate,8192)
        assert pre.prewarmed and len(pre.entries)==10
    result=dict(passed=True,complete=True,device=device,model_dtype='bfloat16',factor_dtype='float16',
        rows=rows,guards=guard_result,pools=pools,stats=dict(graph.stats),precapture=dict(pre.stats),
        real_gpu_recurrence=device=='cuda',cpu_scope='dynamic binder bytes, ownership, guards, real backend/pool routing',
        source_port='56cdc14515d',production_default=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print('PFACTOR4_AGG_BLOCK_GATE',json.dumps(result),flush=True)


if __name__=='__main__':main()
