"""Compose prefix restoration and factor publication with live graph rebinding."""
import argparse,copy,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
os.environ['REPLAY_TEST_DEVICE']=args.device
import torch
import test_factored_prefill_graph as tx


def case(layers,heads,key,batch,tracked):
    old=tx.pool(layers,heads,key);new=copy.deepcopy(old)
    for pool in (old,new):pool.prefill_commit_graph=tx.graphs.PrefillCommitGraph() if tx.GPU else tx.InterpretedGraph()
    restores=[]
    for index in range(2):
        os.environ['SGLANG_GDN_PREFILL_DENSIFY_BATCH']=str(index)
        restores.append(tx.initial.PrefillDensifyAllGraph())
        assert restores[-1].batched == bool(index)
    for repeat in range(3):
        slots=torch.tensor(([2,5] if repeat%2==0 else [5,2])[:batch])
        ring=torch.tensor(([0,1] if repeat%2==0 else [2,-1])[:batch])
        track=torch.tensor([6,8][:batch]) if tracked else None
        delta=torch.randn(layers,batch,heads,key,key)*.003
        extra=torch.randn_like(delta) if tracked else None
        restored=[]
        for index,pool in enumerate((old,new)):
            os.environ['SGLANG_GDN_PREFILL_DENSIFY_BATCH']=str(index)
            os.environ['SGLANG_GDN_PREFILL_STORE_LAYERS']=str(index)
            os.environ['SGLANG_GDN_PREFILL_FACTOR_PAIR']='0'
            # Singleton optimized path; multi-request shape uses the registered fallback.
            if batch==1:
                dense=(restores[index].run(pool,slots,tx.module.densify) if tx.GPU else
                       tx.initial.densify_all_layers(pool,slots,tx.module.densify,batched=bool(index)))
            else:
                dense=torch.stack([tx.module.densify(pool.a[i][slots],pool.U[i][slots],pool.W[i][slots],pool.count[i][slots],pool.vbar[i]) for i in range(layers)])
            restored.append(dense.clone())
            dense.add_(delta)  # model recurrence overwrites the initial output in place
            plan=tx.module.FactoredExtendPlan(slots=slots,use_ring=torch.zeros(batch,dtype=torch.bool),
                ring_src=torch.zeros(batch,dtype=torch.long),ring_dst=ring,ring_dst_rows=torch.where(ring>=0)[0],
                last_layer=layers-1,dense_required_after_commit=torch.full((batch,),repeat%2,dtype=torch.int32),
                stage=dense,track_stage=extra)
            for li in range(layers):pool.commit_extend_batched(li,plan,dense[li],extra[li] if tracked else None,track,
                final_src=slots[:1],final_dst=torch.tensor([9]))
            assert not plan.pending
        if tx.GPU:torch.cuda.synchronize()
        tx.same(restored[0],restored[1],'composed restore')
        for field in tx.FIELDS:tx.same(getattr(old,field),getattr(new,field),'composed publication '+field)
    return dict(layers=layers,heads=heads,key=key,batch=batch,tracked=tracked,replays=3,bitwise=True,
        graph_stats=[pool.prefill_commit_graph.stats for pool in (old,new)] if tx.GPU else None)


torch.manual_seed(298832);start=time.time();rows=[]
for dims in ([(2,2,16),(36,24,128)] if tx.GPU else [(2,2,16)]):
    for batch,tracked in ((1,False),(1,True),(2,True)):
        if dims[0]==36 and batch>1:continue
        rows.append(case(*dims,batch,tracked));print(rows[-1],flush=True)
result=dict(complete=True,passed=True,device=args.device,rows=rows,seconds=time.time()-start)
args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
