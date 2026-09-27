"""Exact whole-layer ring reads, followed by unchanged live commit graphs."""
import argparse,copy,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
os.environ['REPLAY_TEST_DEVICE']=a.device
import torch
import test_factored_prefill_graph as tx
# This test selects the owned-ring branch; unrelated startup densify warmup is excluded.
tx.module._PREFILL_INITIAL_GRAPH=False
os.environ['SGLANG_GDN_PREFILL_STORE_LAYERS']='1'
os.environ['SGLANG_GDN_PREFILL_FACTOR_PAIR']='0'


def make_plan(pool,batch,repeat):
    slots=torch.tensor([2,5][:batch]);src=torch.tensor(([2,0] if repeat%2 else [0,0])[:batch])
    dst=torch.tensor(([1,2] if repeat%2 else [0,1])[:batch])
    return tx.module.FactoredExtendPlan(slots=slots,use_ring=torch.ones(batch,dtype=torch.bool),
        ring_src=src,ring_dst=dst,ring_dst_rows=torch.arange(batch),n_ring_src=batch,
        last_layer=len(pool.layer_ids)-1,dense_required_after_commit=torch.full((batch,),repeat%2,dtype=torch.int32))


def case(layers,heads,key,batch,tracked,returned_copy):
    old=tx.pool(layers,heads,key);new=copy.deepcopy(old)
    for owner in (old,new):owner.prefill_commit_graph=tx.graphs.PrefillCommitGraph() if tx.GPU else tx.InterpretedGraph()
    for repeat in range(3):
        delta=torch.randn(layers,batch,heads,key,key)*.003
        extra=torch.randn_like(delta) if tracked else None
        for flag,owner in ((0,old),(1,new)):
            os.environ['SGLANG_GDN_PREFILL_RING_STAGE']=str(flag)
            plan=make_plan(owner,batch,repeat);before=owner.dense_ring.clone()
            if flag and tracked:plan.track_stage=extra
            for li in range(layers):
                dense=owner.initial_dense(li,plan)
                tx.same(dense,before[li][plan.ring_src],'initial ring copy')
                assert dense.is_contiguous() and dense.data_ptr()!=owner.dense_ring.data_ptr()
                if returned_copy:dense=dense.clone()
                dense.add_(delta[li])
                if li==0:tx.same(before,owner.dense_ring,'gather and recurrence cannot mutate ring')
                owner.commit_extend_batched(li,plan,dense,extra[li] if tracked else None,
                    torch.tensor([6,8][:batch]) if tracked else None,
                    final_src=plan.slots[:1],final_dst=torch.tensor([9]))
            assert not plan.pending
            assert (plan.stage is not None)==bool(flag)
            assert plan.staged==bool(flag and not returned_copy)
        if tx.GPU:torch.cuda.synchronize()
        for field in tx.FIELDS:tx.same(getattr(old,field),getattr(new,field),'ring-stage publication '+field)
    return dict(layers=layers,heads=heads,key=key,batch=batch,tracked=tracked,returned_copy=returned_copy,replays=3,bitwise=True,
        graph_stats=[x.prefill_commit_graph.stats for x in (old,new)] if tx.GPU else None)


def fallbacks():
    owner=tx.pool(2,2,16);rows=[]
    for reason in ('off','budget','partial','mixed','prefix_dense'):
        plan=make_plan(owner,2,0)
        os.environ['SGLANG_GDN_PREFILL_RING_STAGE']='0' if reason=='off' else '1'
        owner._STAGE_MAX_BYTES=1 if reason=='budget' else 128<<20
        owner.prefix_dense=torch.randn(2,10,2,16,16) if reason=='prefix_dense' else None
        if reason=='partial':plan.last_layer=0
        if reason=='mixed':plan.n_ring_src=1;plan.use_ring[1]=False
        if reason=='prefix_dense':plan.use_prefix=torch.ones(2,dtype=torch.bool)
        for li in range(2):tx.same(owner.initial_dense(li,plan),owner._initial_dense_eager(li,plan),'unchanged '+reason)
        assert plan.stage is None
        rows.append(dict(fallback=reason,bitwise=True))
    return rows


torch.manual_seed(298832);start=time.time();rows=[]
for dims in ([(2,2,16),(36,24,128)] if tx.GPU else [(2,2,16)]):
    cases=((1,False,False),(1,True,False),(1,True,True),(2,True,False))
    for batch,tracked,returned_copy in cases:
        if dims[0]==36 and batch>1:continue
        rows.append(case(*dims,batch,tracked,returned_copy));print(rows[-1],flush=True)
rows+=fallbacks()
r=dict(complete=True,passed=True,device=a.device,rows=rows,seconds=time.time()-start,
    scope='Owned-ring copies and unchanged factorization/publication only; no model or TTFT admission.')
if tx.GPU:
    source=torch.randn(36,3,24,128,128);ids=torch.tensor([2]);samples={}
    for name in ('per_layer','all_layers'):
        def read():return [source[li][ids].contiguous() for li in range(36)] if name=='per_layer' else torch.index_select(source,1,ids)
        for _ in range(5):read()
        times=[]
        for _ in range(3):
            torch.cuda.synchronize();begin=time.perf_counter()
            for _ in range(30):read()
            torch.cuda.synchronize();times.append((time.perf_counter()-begin)*1000/30)
        samples[name]=times
    r['ring_gather_wall_ms']=samples
    r['timing_scope']='Three 30-read batches, launch/allocation plus GPU completion; excludes commit and model.'
a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
