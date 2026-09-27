"""Prefill-only metadata stores preserve all slot values and avoid CPU scalar sync."""
import argparse,copy,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
os.environ['REPLAY_TEST_DEVICE']=a.device
import torch
import test_factored_prefill_graph as tx


def case(layers,heads,key,limit,required):
    old=tx.pool(layers,heads,key);new=copy.deepcopy(old)
    for owner in (old,new):
        owner.prefix_layer_limit=limit
        if not required:owner.dense_required=None
    for repeat,(src,dst) in enumerate((([2],[5]),([5,2],[2,5]),([],[]),([2,2],[5,6]),([0],[-1]))):
        source=torch.tensor(src,dtype=torch.long);dest=torch.tensor(dst,dtype=torch.long)
        os.environ['SGLANG_GDN_PREFILL_COPY_METADATA_DEVICE']='0';old._copy_prefill_slots(source,dest)
        os.environ['SGLANG_GDN_PREFILL_COPY_METADATA_DEVICE']='1';new._copy_prefill_slots(source,dest)
        for field in tx.FIELDS:
            if getattr(old,field) is not None:tx.same(getattr(old,field),getattr(new,field),'metadata '+field)
    # Public copy entry never activates the prefill optimization implicitly.
    check=copy.deepcopy(new);new.copy_slots(torch.tensor([2]),torch.tensor([4]));check.copy_slots(torch.tensor([2]),torch.tensor([4]),_prefill_device_metadata=False)
    for field in tx.FIELDS:
        if getattr(new,field) is not None:tx.same(getattr(new,field),getattr(check,field),'default '+field)
    return dict(layers=layers,heads=heads,key=key,limit=limit,required=required,bitwise=True)


def graph_case():
    p=tx.pool(2,2,16);expected=copy.deepcopy(p);src=torch.tensor([2]);dst=torch.tensor([5])
    os.environ['SGLANG_GDN_PREFILL_COPY_METADATA_DEVICE']='1'
    # Warm on a separate stream; then capture and change both live index buffers.
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):p._copy_prefill_slots(src,dst)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):p._copy_prefill_slots(src,dst)
    expected._copy_prefill_slots(src,dst)
    for x,y in ((2,5),(5,6),(6,2)):
        src.fill_(x);dst.fill_(y);graph.replay();expected._copy_prefill_slots(src,dst)
        torch.cuda.synchronize()
        for field in tx.FIELDS:tx.same(getattr(p,field),getattr(expected,field),'replayed '+field)
    return dict(graph=True,live_rebinds=3,bitwise=True)


def sync_counts():
    owner=tx.pool(36,24,128);src=torch.tensor([2]);dst=torch.tensor([5]);rows={}
    for flag in (0,1):
        os.environ['SGLANG_GDN_PREFILL_COPY_METADATA_DEVICE']=str(flag)
        for _ in range(3):owner._copy_prefill_slots(src,dst)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function('prefill-copy-body'):owner._copy_prefill_slots(src,dst)
        counts={}
        for event in prof.events():
            parent=event.cpu_parent
            while parent is not None and parent.name!='prefill-copy-body':parent=parent.cpu_parent
            if parent is not None and event.name in ('cudaStreamSynchronize','cudaMemcpyAsync'):
                counts[event.name]=counts.get(event.name,0)+1
        rows[str(flag)]=counts
    assert rows['0'].get('cudaStreamSynchronize',0)==3,rows
    assert rows['1'].get('cudaStreamSynchronize',0)==0,rows
    return rows


torch.manual_seed(298832);start=time.time();rows=[]
for dims in ([(2,2,16),(36,24,128)] if tx.GPU else [(2,2,16)]):
    for limit in (dims[0],1):
        for required in (True,False):rows.append(case(*dims,limit,required))
r=dict(complete=True,passed=True,device=a.device,rows=rows,scope='Prefill slot-copy values, graph live indices and host synchronization only; no model admission.')
if tx.GPU:r['graph']=graph_case();r['sync_calls']=sync_counts()
r['seconds']=time.time()-start;a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
