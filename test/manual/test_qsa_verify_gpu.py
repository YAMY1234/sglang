"""Causal compression and accepted-ring publication, using production kernels."""
import json
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.qsa_indexer import qsa_index_k_compress_store
from sglang.srt.mem_cache.qsa_verify_state import QSAVerifyState, causal_groups


def run():
    torch.manual_seed(427)
    device = "cuda"
    cos_sin = torch.randn(1024,128,device=device)
    axis = torch.zeros(64,dtype=torch.int32,device=device)
    weight = torch.randn(128,dtype=torch.bfloat16,device=device)
    req = torch.tensor([1,3],device=device).repeat_interleave(4)
    outcomes=[]
    for offset in range(4):
        for graphed in (False,True):
            old = torch.randn(20,1,128,dtype=torch.bfloat16,device=device)
            oldrope = torch.zeros(20,3,dtype=torch.int64,device=device)
            start = 100+offset
            for request in (1,3):
                for p in range(start-4,start):
                    oldrope[request*4+p%4] = p
            original=old.clone(); original_rope=oldrope.clone()
            positions=torch.arange(start,start+4,device=device).repeat(2)
            keys=torch.randn(8,1,128,dtype=torch.bfloat16,device=device)
            rope=positions[:,None].expand(-1,3).contiguous()
            pool=SimpleNamespace(qsa_compress_ratio=4,qsa_key_state_buffer_pool=[old],
                 qsa_rope_position_buffer=oldrope,_transfer_full_attention_id=lambda _:0,
                 get_qsa_key_state_buffer=lambda _:old)
            tx=QSAVerifyState(pool,3,4)
            expected=torch.zeros(16,128,dtype=torch.bfloat16,device=device)
            actual=torch.zeros_like(expected)
            reference=old.clone(); reference_rope=oldrope.clone()
            for step in range(4):
                rows=torch.tensor([step,4+step],device=device)
                pp=positions[rows]; rr=req[rows]
                reference[rr*4+pp%4]=keys[rows]
                reference_rope[rr*4+pp%4]=rope[rows]
                group=rr[:,None]*4+(pp[:,None]-3+torch.arange(4,device=device))%4
                writes=torch.where((pp+1)%4==0,rr,0)
                qsa_index_k_compress_store(reference.flatten(1),group.int(),reference_rope,
                    cos_sin,axis,weight,writes.int(),expected,4,128,1e-6,True)
            def body():
                groups,coords=causal_groups(old,oldrope,keys,rope,req,positions,4,4)
                locs=torch.arange(32,device=device).reshape(8,4)
                writes=torch.where((positions+1)%4==0,req,0)
                qsa_index_k_compress_store(groups.flatten(0,1).flatten(1),locs.int(),coords.flatten(0,1),
                    cos_sin,axis,weight,writes.int(),actual,4,128,1e-6,True)
                tx.record(0,keys,rope,positions)
            body()
            if graphed:
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): body()
                graph.replay()
            assert torch.equal(actual[[1,3]],expected[[1,3]]), (offset,graphed,"compressed keys")
            assert torch.equal(old,original) and torch.equal(oldrope,original_rope)
            tx.begin(torch.tensor([1,3],device=device))
            tx.commit(torch.tensor([0,2],device=device))
            for row in (0,4,5,6):
                original[req[row]*4+positions[row]%4]=keys[row]
                original_rope[req[row]*4+positions[row]%4]=rope[row]
            assert torch.equal(old,original) and torch.equal(oldrope,original_rope)
            outcomes.append(dict(offset=offset,graph=graphed,compressed_exact=True,ring_exact=True))
    print(json.dumps(dict(passed=True,gpu=torch.cuda.get_device_name(),cases=outcomes)),flush=True)


if __name__=="__main__": run()
