"""Real factor pool/commit/store replay through the P31 split adapter.

Compare all stored bytes, including shallow count-eight snapshots, exact ring,
tracked checkpoints and slot authority. No tolerance is relaxed for batching.
"""
import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNConfig, FactoredGDNPool, FactoredExtendPlan
from sglang.srt.mem_cache.gdn_prefill_deep_batch import DeepPrefixBatch
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from twinstar_sgl.pd_shallow_gdn import split_boundary

LAYERS = tuple(lid for lid in range(48) if lid % 4 != 3)
DEEP = tuple(lid for lid in LAYERS if lid >= 31)
FIELDS = ('a', 'U', 'W', 'count', 'stale', 'dense_of', 'dense_required', 'dense_ring', 'prefix_valid')


def run_case(device, heads, batch, kind, tracked, final, group, constants, *, factor_graph=False):
    cfg = FactoredGDNConfig(r=8, m=8, dtype=torch.float16, ring=batch,
        strict_chunk=1, factored_prefix=1, init_method='k31', vbar_path=str(constants))
    def make():
        p = FactoredGDNPool(size=3*batch+1, cache_params=NS(shape=NS(temporal=(heads,128,128))),
            mamba_layer_ids=list(LAYERS), device=device, cfg=copy.copy(cfg))
        # Shallow N-1 is already saved at destination; live shallow has appended
        # exactly one token. Deep publication must not recopy those layers.
        p.a[:24,1:batch+1].fill_(9);p.a[:24,batch+1:2*batch+1].fill_(8)
        p.U[:24,1:batch+1].fill_(9);p.U[:24,batch+1:2*batch+1].fill_(8)
        p.W[:24,1:batch+1].fill_(9);p.W[:24,batch+1:2*batch+1].fill_(8)
        p.count[:24,1:batch+1].fill_(9);p.count[:24,batch+1:2*batch+1].fill_(8)
        p.dense_of[1:batch+1]=torch.arange(batch,device=device,dtype=torch.int32)
        p.ring_owner=list(range(1,batch+1))
        row_bytes=batch*heads*128*128*4*(2 if tracked else 1)
        p.batch_prefill_max_bytes=group*row_bytes
        return p
    def metadata():
        slots=torch.arange(1,batch+1,device=device)
        plan=FactoredExtendPlan(slots=slots,use_ring=torch.zeros(batch,dtype=torch.bool,device=device),
            ring_src=torch.zeros(batch,dtype=torch.long,device=device),ring_dst=torch.arange(batch,device=device),
            ring_dst_rows=torch.arange(batch,device=device),all_fresh=True,last_layer=len(LAYERS)-1,
            dense_required_after_commit=torch.zeros(batch,dtype=torch.int32,device=device))
        return ForwardMetadata(query_start_loc=torch.arange(batch+1,device=device),mamba_cache_indices=slots,
            factored_extend=plan,track_ssm_final_src=slots if final else None,
            track_ssm_final_dst=slots+batch if final else None)
    torch.manual_seed(0x504634)
    states={lid:torch.randn(batch,heads,128,128,device=device) for lid in DEEP}
    if kind=='rank4':states={lid:(x[...,:4]@x[...,:4,:]).contiguous() for lid,x in states.items()}
    if kind=='zero':states={lid:torch.zeros_like(x) for lid,x in states.items()}
    snapshots={lid:(x*.75).contiguous() for lid,x in states.items()} if tracked else {}
    pools=[];group_sizes=[];observer_counts=[]
    for enabled in ('0','1'):
        p=make();m=metadata();observed=[];sizes=[]
        backend=NS(factored=p,forward_metadata=m)
        native=p._commit_extend_group
        def commit(*args,**kwargs):
            sizes.append(len(args[1].pending));return native(*args,**kwargs)
        p._commit_extend_group=commit
        def extend(layer,forward_batch,mixed_qkv,a,b,**kwargs):
            current=backend.forward_metadata
            p.commit_extend_batched(layer.layer_id,current.factored_extend,states[layer.layer_id],
                snapshots.get(layer.layer_id),torch.arange(2*batch+1,3*batch+1,device=device) if tracked else None,
                current.track_ssm_final_src,current.track_ssm_final_dst)
            return states[layer.layer_id]
        backend.forward_extend=extend
        with patch.dict(os.environ,SGLANG_PFACTOR4_DEEP_BATCH=enabled,
                SGLANG_PFACTOR4_DEEP_FACTOR_GRAPH='1' if factor_graph and enabled=='1' else '0',
                SGLANG_GDN_PSIDE_COMPOSITE='0',TWINSTAR_PD_EMITTER_GRAPH='0',
                SGLANG_GDN_PSIDE_GRAPH='0',SGLANG_GDN_PREFILL_COMMIT_GRAPH='0'):
            with split_boundary(backend,None,None,None,None,m,None,publication_observer=observed.append):
                for lid in DEEP:
                    backend.forward_extend(NS(layer_id=lid),NS(batch_size=batch),None,None,None)
                    if enabled=='1' and lid!=DEEP[-1]:assert observed==[], 'premature prefix publication'
        assert backend.forward_metadata is m and backend.forward_extend is extend
        assert m.factored_extend.pending==[] and m.factored_extend.next_layer==0
        assert observed==[final]*len(DEEP)
        assert torch.all(p.count[:24,1:batch+1]==9) and torch.all(p.count[:24,batch+1:2*batch+1]==8)
        pools.append(p);group_sizes.append(sizes);observer_counts.append(len(observed))
    rows={name:dict(exact=torch.equal(getattr(pools[0],name),getattr(pools[1],name)),
        max_abs=float((getattr(pools[0],name).float()-getattr(pools[1],name).float()).abs().max())) for name in FIELDS}
    assert group_sizes[0]==[1]*12 and group_sizes[1]==[group]*(12//group),group_sizes
    graph_stats = None
    if factor_graph:
        graph = pools[1]._pfactor4_deep_factor_graph
        graph_stats = dict(graph.stats)
        assert graph.stats['replayed' if device=='cuda' else 'fallback'] > 0
        if device=='cuda':
            assert graph.stats['captured'] > 0 and graph.stats['retained_bytes'] <= 1024 << 20
    # Failed or incomplete emitter execution must restore dispatch and discard
    # its private pending group without copying any destination.
    p=make();m=metadata();backend=NS(factored=p,forward_metadata=m);observed=[]
    def incomplete(layer,forward_batch,*args,**kwargs):
        plan=backend.forward_metadata.factored_extend
        plan.next_layer+=1;plan.pending.append((states[layer.layer_id],None))
        return None
    backend.forward_extend=incomplete
    with patch.dict(os.environ,SGLANG_PFACTOR4_DEEP_BATCH='1',SGLANG_GDN_PSIDE_COMPOSITE='0',TWINSTAR_PD_EMITTER_GRAPH='0'):
        for deliberate in (False,True):
            try:
                with split_boundary(backend,None,None,None,None,m,None,publication_observer=observed.append):
                    backend.forward_extend(NS(layer_id=DEEP[0]),NS(batch_size=batch),None,None,None)
                    private=backend.forward_metadata.factored_extend
                    if deliberate:raise ValueError('deliberate failure')
            except (ValueError,RuntimeError) as exc:
                assert ('deliberate' if deliberate else 'incomplete') in str(exc)
            else:raise AssertionError('incomplete batch accepted')
            assert not private.pending and observed==[] and backend.forward_metadata is m
    return dict(B=batch,heads=heads,kind=kind,tracked=tracked,final_copy=final,group_limit=group,
        passed=all(x['exact'] for x in rows.values()),fields=rows,commit_groups=group_sizes,
        native_pool=True,adapter=True,shallow_live_count=9,shallow_prefix_count=8,abort_passed=True,
        factor_graph=factor_graph, graph_stats=graph_stats)


def graph_buffer_replay(device, heads, constants):
    """TP-sliced, batch-broadcast omega and normal/track output ownership."""
    from sglang.srt.mem_cache.gdn_prefill_factor_graph import FactorizeBuffers, PrefillFactorGraph
    from sglang.srt.mem_cache.gdn_factored_pool import _factorize_deep_preserving_sink

    cfg=FactoredGDNConfig(r=8,m=8,dtype=torch.float16,ring=1,init_method='k31',
                         vbar_path=str(constants))
    pool=FactoredGDNPool(size=2,cache_params=NS(shape=NS(temporal=(heads,128,128))),
        mamba_layer_ids=list(LAYERS),device=device,cfg=cfg)
    graph=PrefillFactorGraph(max_input_bytes=192 << 20)
    rows=[]
    # Two layers keep the dynamic replay gate bounded; the main gate above
    # exercises all twelve deep layers and the real pool publication path.
    for batch in (1,8):
        states=[torch.randn(batch,heads,128,128,device=device) for _ in range(2)]
        vbar=pool.vbar[:2].clone();omega=pool.init_omega(batch)
        buffers=FactorizeBuffers(states,vbar,cfg,omega=omega)
        assert buffers.omega.data_ptr()!=omega.data_ptr()
        assert torch.equal(buffers.omega,omega)
        if batch>1:assert buffers.omega.stride(0)==omega.stride(0)==0
        for repeat in range(3):
            changed=[s*(repeat+1)*.25 for s in states]
            changed_vbar=vbar*(1+repeat*.1)
            owned=omega[:1].clone()*(1+repeat*.05)
            changed_omega=owned.expand_as(omega) if batch>1 else owned
            buffers.bind(changed,changed_vbar,omega=changed_omega)
            assert torch.equal(buffers.omega,changed_omega)
            expected=_factorize_deep_preserving_sink(changed,changed_vbar,cfg,omega=changed_omega)
            actual=graph.run(changed,changed_vbar,cfg,omega=changed_omega,
                eager=_factorize_deep_preserving_sink,policy=('exact-probe-replay',))
            assert all(torch.equal(a,b) for lhs,rhs in zip(actual,expected) for a,b in zip(lhs,rhs))
            saved=[[v.clone() for v in row] for row in actual]
            graph.run([s*.75 for s in changed],changed_vbar,cfg,omega=changed_omega,
                eager=_factorize_deep_preserving_sink,policy=('exact-probe-replay',))
            assert all(torch.equal(a,b) for lhs,rhs in zip(actual,saved) for a,b in zip(lhs,rhs)), 'tracked replay overwrote normal output'
        rows.append(dict(B=batch,replays=3,omega_exact=True,independent_outputs=True))
    # Unsupported shapes/budgets still consume the exact supplied probe.
    tiny=PrefillFactorGraph(max_input_bytes=1)
    result=tiny.run(states,vbar,cfg,omega=omega,eager=_factorize_deep_preserving_sink,policy=())
    expected=_factorize_deep_preserving_sink(states,vbar,cfg,omega=omega)
    assert all(torch.equal(a,b) for lhs,rhs in zip(result,expected) for a,b in zip(lhs,rhs))
    assert tiny.stats['captured']==0 and tiny.stats['fallback']==1
    return dict(passed=True,rows=rows,stats=dict(graph.stats),explicit_fallback=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--factor-graph',action='store_true');a=parser.parse_args()
    output=Path(a.output);device='cpu' if os.environ.get('TRITON_INTERPRET')=='1' else 'cuda'
    heads=2 if device=='cpu' else 24
    constants=output.with_name('deep-batch-vbar-'+device+'.pt')
    torch.manual_seed(0x504634)
    torch.save({'vbar':{lid:torch.randn(heads,128) for lid in LAYERS}},constants)
    cases=[(1,'full',False,True,12),(8,'full',True,True,2),(1,'rank4',True,False,12),(1,'zero',False,True,12)]
    if device=='cuda':cases+=[(8,'rank4',True,True,12),(8,'full',False,True,12)]
    rows=[]
    graph_gate=graph_buffer_replay(device,heads,constants) if a.factor_graph else None
    for case in cases:
        rows.append(run_case(device,heads,*case,constants,factor_graph=a.factor_graph))
        result=dict(passed=all(r['passed'] for r in rows),complete=len(rows)==len(cases),device=device,factor_dtype='float16',
            gate='all a/U/W/count/authority/ring bytes exact; shallow live9 and saved8 unchanged',rows=rows,
            factor_graph=a.factor_graph,graph_gate=graph_gate)
        output.write_text(json.dumps(result,indent=2)+'\n')
    print('PFACTOR4_DEEP_BATCH_GATE',json.dumps(result),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
