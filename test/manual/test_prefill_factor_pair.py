"""Paired final/tracked factorization must retain the frozen factors exactly."""
import argparse,copy,json,os,time
from pathlib import Path
from types import SimpleNamespace
import torch
import test_factored_prefill_graph as tx


def same(a,b,label):
    assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)), label


def case(layers,heads,key,batch,track_batch):
    p=tx.pool(layers,heads,key)
    dense=torch.randn(layers,batch,heads,key,key)
    tracked=torch.randn(layers,track_batch,heads,key,key) if track_batch else None
    plan=tx.module.FactoredExtendPlan(slots=torch.arange(batch),
        use_ring=torch.zeros(batch,dtype=torch.bool),ring_src=torch.zeros(batch,dtype=torch.long),
        ring_dst=torch.arange(batch),ring_dst_rows=torch.arange(batch),last_layer=layers-1,
        dense_required_after_commit=torch.zeros(batch,dtype=torch.int32),
        pending=[(dense[i],tracked[i] if track_batch else None) for i in range(layers)])
    slots=torch.arange(track_batch)+6 if track_batch else None
    pools=[copy.deepcopy(p),copy.deepcopy(p)];buffers=[]
    for flag,pool in zip(('0','1'),pools):
        os.environ['SGLANG_GDN_PREFILL_FACTOR_PAIR']=flag
        buffers.append(tx.graphs.CommitBuffers(pool,plan,slots))
    assert buffers[1].factor_pair == bool(track_batch and track_batch==batch)
    for repeat in range(3):
        dense.normal_()
        if tracked is not None:tracked.normal_()
        for b in buffers:b.bind(plan,slots)
        results=[b.evaluate(tx.module.factorize_layers) for b in buffers]
        for kind in (0,1):
            if results[0][kind] is None:
                assert results[1][kind] is None
                continue
            for layer,(left,right) in enumerate(zip(results[0][kind],results[1][kind])):
                for name,a,b in zip(('a','U','W'),left,right):same(a,b,(repeat,kind,layer,name))
        for field in tx.FIELDS:same(getattr(pools[0],field),getattr(pools[1],field),field)
    return dict(layers=layers,heads=heads,key=key,batch=batch,track_batch=track_batch,
                paired=buffers[1].factor_pair,replays=3,bitwise=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    torch.manual_seed(298832);started=time.time()
    dims=[(2,2,16)] + ([(36,24,128)] if tx.GPU else [])
    rows=[case(*shape,b,t) for shape in dims for b,t in ((1,1),(1,0),(2,1),(2,2))
          if shape[0]!=36 or b==1]
    out=dict(complete=True,passed=True,device='cuda' if tx.GPU else 'cpu',rows=rows,seconds=time.time()-started)
    args.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
