"""M10 native pool publication/ownership gate, CPU replay then CUDA bytes.

The model fixture supplies final dense states to the real commit API. It does
not simulate attention or claim a whole-model numerical gate. GPU exercises
36 layers/HV24/FP16 factors; CPU uses two layers/HV1 with native algebra/store.
"""
import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.gdn_agg_commit import SHAPES, eligible, install_forward
from sglang.srt.mem_cache.gdn_agg_commit_budget import reservation_bytes, unused_reservation_bytes
from sglang.srt.mem_cache.gdn_factored_pool import (
    FactoredGDNConfig, FactoredGDNPool, FactoredExtendPlan, factorize_layers,
    factorize_dense, ORTH_METHOD, ORTH_WARPS_OVERRIDE,
)
from sglang.srt.mem_cache.gdn_prefill_batch_graph import BatchBuffers, PrefillBatchGraph
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode

FIELDS = ('a','U','W','count','stale','dense_of','dense_required','dense_ring','prefix_valid')
POLICY = (ORTH_METHOD, ORTH_WARPS_OVERRIDE, factorize_dense)


class CPUReplay:
    """Explicitly not a CUDA graph: execute native bind/algebra/store on CPU."""
    warmed = True
    def __init__(self):
        self.buffers = {}; self.shared = {}; self.replayed = 0
    def run(self, pool, plan, states, track, src, dst, *, eager, policy):
        key = (plan.slots.numel(),None if track is None else track.numel())
        assert key in SHAPES
        if key not in self.buffers:
            self.buffers[key] = BatchBuffers(pool,*key,self.shared,include_tail=False)
        b = self.buffers[key]
        b.bind(plan,states,track,src,dst); b.evaluate(eager)
        self.replayed += 1


def budgets():
    with patch.dict(os.environ,SGLANG_GDN_AGG_COMMIT_GRAPH='0',SGLANG_GDN_AGG_COMMIT_RESERVE_MB='1024'):
        assert reservation_bytes('null')==1<<30
        assert unused_reservation_bytes('null',128<<20)==896<<20
        for role in ('prefill','decode'):
            try:reservation_bytes(role)
            except ValueError:pass
            else:raise AssertionError('PD reservation accepted')
        try:unused_reservation_bytes('null',1025<<20)
        except RuntimeError:pass
        else:raise AssertionError('over-budget graph accepted')
    with patch.dict(os.environ,SGLANG_GDN_AGG_COMMIT_GRAPH='1',SGLANG_GDN_AGG_COMMIT_RESERVE_MB='0'):
        try:reservation_bytes('null')
        except ValueError:pass
        else:raise AssertionError('implicit unbudgeted graph accepted')
    return dict(passed=True,checks=6,common_reference_reservation=True)


def byte_equal(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(
        a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def run_case(device,constants,tracked,final):
    heads = 24 if device=='cuda' else 1
    layers = [i for i in range(48) if i%4!=3] if device=='cuda' else [0,1]
    cfg = FactoredGDNConfig(r=8,m=8,dtype=torch.float16,ring=1,strict_chunk=1,
        factored_prefix=1,init_method='k31',vbar_path=str(constants))
    pools=[];graphs=[];owners=[];backends=[];states=[]
    for enabled in (False,True):
        p=FactoredGDNPool(size=5,cache_params=NS(shape=NS(temporal=(heads,128,128))),
            mamba_layer_ids=layers,device=device,cfg=copy.copy(cfg))
        p._agg_commit_stats=dict(committed=0,fallback=0)
        backend=NS(forward_metadata=None)
        def model(input_ids,positions,forward_batch,*,p=p,backend=backend):
            m=backend.forward_metadata
            for i,lid in enumerate(layers):
                p.commit_extend_batched(lid,m.factored_extend,states[i],
                    states[i]*.75 if tracked else None,m.track_ssm_h_dst,
                    m.track_ssm_final_src,m.track_ssm_final_dst)
            return input_ids
        owner=NS(forward=model)
        if enabled:
            graph=PrefillBatchGraph(include_tail=False,shapes=SHAPES) if device=='cuda' else CPUReplay()
            if device=='cuda':graph.prewarm(p,eager=factorize_layers,policy=POLICY)
            install_forward(owner,p,lambda backend=backend:backend,graph)
            graphs.append(graph)
        owners.append(owner);pools.append(p);backends.append(backend)
    checks=[]
    for repeat in range(2):
        states[:]=[torch.randn(1,heads,128,128,device=device)*(.1+repeat*.05) for _ in layers]
        batch=NS(forward_mode=ForwardMode.EXTEND,batch_size=1)
        for p,backend,owner in zip(pools,backends,owners):
            slots=torch.tensor([1+repeat],device=device)
            plan=FactoredExtendPlan(slots=slots,use_ring=torch.zeros(1,device=device,dtype=torch.bool),
                ring_src=torch.zeros(1,device=device,dtype=torch.long),ring_dst=torch.zeros(1,device=device,dtype=torch.long),
                ring_dst_rows=torch.zeros(1,device=device,dtype=torch.long),all_fresh=repeat==0,last_layer=len(layers)-1,
                dense_required_after_commit=torch.full((1,),repeat,device=device,dtype=torch.int32))
            # Rebind real slot IDs between replays. Final copy deliberately
            # overwrites the tracked checkpoint, preserving native ordering.
            m=ForwardMetadata(query_start_loc=torch.tensor([0,64],device=device),mamba_cache_indices=slots,
                factored_extend=plan,track_ssm_h_dst=torch.tensor([3],device=device) if tracked else None,
                track_ssm_final_src=slots if final else None,
                track_ssm_final_dst=torch.tensor([3],device=device) if final else None)
            backend.forward_metadata=m
            assert eligible(batch,m,p)
            for changes in (dict(batch_size=2),dict(forward_mode=ForwardMode.DECODE),dict(can_run_tbo=True),
                            dict(spec_info=object()),dict(_pfactor_legacy_mixed=True)):
                bad=copy.copy(batch);bad.__dict__.update(changes);assert not eligible(bad,m,p)
            x=torch.tensor([repeat],device=device);result=owner.forward(x,None,batch)
            assert result is x and not plan.pending and not getattr(plan,'batch_collector',None)
        row={name:byte_equal(getattr(pools[0],name),getattr(pools[1],name)) for name in FIELDS}
        assert all(row.values()),row
        checks.append(dict(repeat=repeat,fields=row))
    # Incomplete/failed model calls must release collector and private tensors.
    p=pools[1];m=backends[1].forward_metadata
    for fail in (False,True):
        m.factored_extend.next_layer=0
        def interrupted(*args,**kwargs):
            p.commit_extend_batched(layers[0],m.factored_extend,states[0])
            if fail:raise ValueError('deliberate model failure')
        owner=NS(forward=interrupted)
        install_forward(owner,p,lambda:backends[1],graphs[0])
        try:owner.forward(None,None,batch)
        except (RuntimeError,ValueError) as exc:
            assert ('deliberate' if fail else 'before every layer') in str(exc)
        else:raise AssertionError('partial model was published')
        assert not m.factored_extend.pending and not m.factored_extend.batch_collector and not p._agg_commit_active
    return dict(passed=True,tracked=tracked,final=final,layers=len(layers),heads=heads,
        replays=checks,abort_cleanup=True,native_pool=True,native_factorization=True,
        cuda_graph=device=='cuda',cpu_cuda_graph_stub=device=='cpu',
        captured=graphs[0].stats['captured'] if device=='cuda' else 0)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=Path(args.output);device='cpu' if os.environ.get('TRITON_INTERPRET')=='1' else 'cuda'
    torch.manual_seed(0x4d3130)
    constants=output.with_name('agg-commit-vbar-'+device+'.pt')
    lids=[i for i in range(48) if i%4!=3] if device=='cuda' else [0,1]
    torch.save({'vbar':{i:torch.randn(24 if device=='cuda' else 1,128) for i in lids}},constants)
    with patch.dict(os.environ,SGLANG_GDN_PREFILL_COMMIT_GRAPH='0',SGLANG_GDN_PSIDE_GRAPH='0',
                    SGLANG_GDN_PREFILL_JOIN_BRANCHES='0'):
        budget=budgets()
        rows=[run_case(device,constants,*case) for case in ((False,False),(True,False),(True,True))]
    result=dict(passed=True,complete=True,device=device,model_dtype='bfloat16',factor_dtype='float16',
        budget=budget,rows=rows,production_default=False,
        boundary='real grouped commit vs model-entry collection; GPU bytes before end-to-end service admission')
    output.write_text(json.dumps(result,indent=2)+'\n');print('PFACTOR4_AGG_COMMIT_GATE',json.dumps(result),flush=True)


if __name__=='__main__':main()
