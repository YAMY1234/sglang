"""CPU-interpreted stores, then the same live-buffer transaction on CUDA."""
import copy
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch

GPU = os.environ.get('REPLAY_TEST_DEVICE') == 'cuda'
if not GPU and (os.environ.get('TRITON_INTERPRET') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
    raise RuntimeError('CPU interpreter and CUDA hidden required')
if GPU:
    torch.set_default_device('cuda')
ROOT = Path(__file__).resolve().parents[2]


def package(name, path):
    m=ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m


package('prefill_owners', ROOT/'python/sglang/srt/mem_cache')
package('prefill_kernels', ROOT/'python/sglang/srt/layers/attention/linear/kernels')
for name in ('gdn_factored', 'gdn_factored_io', 'gdn_prefill_reference'):
    sys.modules['sglang.srt.layers.attention.linear.kernels.'+name] = importlib.import_module('prefill_kernels.'+name)
module=importlib.import_module('prefill_owners.gdn_factored_pool')
graphs=importlib.import_module('prefill_owners.gdn_prefill_commit_graph')
initial=importlib.import_module('prefill_owners.gdn_prefill_initial_graph')
# Existing relative graph imports resolve through the same isolated modules.
sys.modules['prefill_owners.gdn_prefill_initial_graph']=initial


def same(a,b,label):
    if not torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)):
        raise AssertionError(label)


def pool(layers,heads,key):
    p=module.FactoredGDNPool.__new__(module.FactoredGDNPool)
    p.cfg=module.FactoredGDNConfig(dtype=torch.float16,strict_chunk=1,factored_prefix=1)
    p.layer_ids=list(range(layers));p.layer_map={i:i for i in p.layer_ids}
    p.hv=p.h=heads;p.k=p.v=key;p.device='cuda' if GPU else 'cpu'
    p.a=torch.randn(layers,10,heads,key)*.01
    p.U=torch.randn(layers,10,heads,16,key,dtype=torch.float16)*.01;p.W=torch.randn_like(p.U)
    p.count=torch.full((layers,10,heads),8,dtype=torch.int32)
    p.stale=torch.ones(10,dtype=torch.int32);p.dense_of=torch.full((10,),-1,dtype=torch.int32)
    p.dense_required=torch.ones(10,dtype=torch.int32)
    p.prefix_dense=None;p.prefix_dense_valid=None;p.prefix_factored_valid=torch.zeros(10,dtype=torch.int32)
    p.dense_ring=torch.randn(layers,3,heads,key,key);p.vbar=torch.randn(layers,heads,key)*.01
    p.ring_owner=[2,5,-1];p.ring_lru=[0,1,2]
    p.batch_prefill=True;p.batch_prefill_final_copy=True;p.batch_prefill_max_bytes=512<<20
    p.prefill_commit_graph=None;p.spec_state=None;p.stats=dict(densified=0)
    return p


class InterpretedGraph:
    def __init__(self):self.buffers=None
    def run(self,p,plan,track_slots,*,factorize,policy):
        if self.buffers is None:
            self.buffers=graphs.CommitBuffers(p,plan,track_slots)
            before={n:getattr(p,n).clone() for n in FIELDS}
            self.buffers.evaluate(factorize)
            for n,t in before.items():same(t,getattr(p,n),'disabled warmup '+n)
        self.buffers.bind(plan,track_slots)
        self.buffers.evaluate(factorize)
        return True


FIELDS=('a','U','W','count','stale','dense_of','dense_required','prefix_factored_valid','dense_ring')

def case(layers,heads,key,batch,tracked):
    base=pool(layers,heads,key);old=copy.deepcopy(base);new=copy.deepcopy(base)
    new.prefill_commit_graph=graphs.PrefillCommitGraph() if GPU else InterpretedGraph()
    for repeat in range(3):
        slots=torch.tensor(([2,5] if repeat%2==0 else [5,2])[:batch])
        # Change payload, slot/ring bindings and continuation metadata on replay.
        ring=torch.tensor(([0,1] if repeat%2==0 else [2,-1])[:batch])
        track=torch.tensor([6,8][:batch]) if tracked else None
        dense=torch.randn(layers,batch,heads,key,key)
        extra=torch.randn_like(dense) if tracked else None
        for p in (old,new):
            plan=module.FactoredExtendPlan(slots=slots,use_ring=torch.zeros(batch,dtype=torch.bool),
                ring_src=torch.zeros(batch,dtype=torch.long),ring_dst=ring,
                ring_dst_rows=torch.where(ring>=0)[0],last_layer=layers-1,
                dense_required_after_commit=torch.full((batch,),repeat%2,dtype=torch.int32))
            for li in range(layers):
                p.commit_extend_batched(li,plan,dense[li],extra[li] if tracked else None,track,
                    final_src=slots[:1],final_dst=torch.tensor([9]))
            assert not plan.pending and plan.next_layer==layers
        for n in FIELDS:same(getattr(old,n),getattr(new,n),'published '+n)
        assert old.ring_owner==new.ring_owner and old.ring_lru==new.ring_lru
        # Restored dense values and live slot rebinding use original algebra.
        plan=module.FactoredExtendPlan(slots=slots[:1],use_ring=torch.zeros(1,dtype=torch.bool),
            ring_src=torch.zeros(1,dtype=torch.long),ring_dst=torch.tensor([-1]),ring_dst_rows=torch.empty(0,dtype=torch.long))
        graph=initial.PrefillInitialGraph()
        for li in range(layers):
            expected=new._initial_dense_eager(li,plan)
            output=graph.run(new,li,plan)
            same(expected,output,'initial dense graph')
            output.fill_(99)
            same(expected,graph.run(new,li,plan),'initial output must not alias')
    if GPU:
        assert new.prefill_commit_graph.stats['captured']==1
        assert new.prefill_commit_graph.stats['replayed']==3
    return dict(layers=layers,heads=heads,key=key,batch=batch,tracked=tracked,replays=3,bitwise=True,
                stats=new.prefill_commit_graph.stats if GPU else None)


if __name__=='__main__':
    torch.manual_seed(688)
    rows=[]
    shapes=[(2,2,16),(36,24,128)] if GPU else [(2,2,16)]
    for dims in shapes:
        for batch,tracked in ((1,False),(1,True),(2,False)):
            if dims[0]==36 and batch==2:continue
            rows.append(case(*dims,batch,tracked));print(rows[-1],file=sys.stderr,flush=True)
    print(json.dumps(dict(passed=True,device='CUDA' if GPU else 'CPU',cases=rows)))
