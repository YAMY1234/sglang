"""Byte-level scatter ownership tests and graph-replay timing, no algebra change."""
import copy
import json
import os
from types import SimpleNamespace

import torch
from sglang.srt.layers.attention.linear.kernels.gdn_factored_io import store_factored
from sglang.srt.layers.attention.linear.kernels.gdn_prefill_store_layers import store_layers

GPU = os.environ.get('REPLAY_TEST_DEVICE') == 'cuda'
DEV = 'cuda' if GPU else 'cpu'
assert GPU or os.environ.get('TRITON_INTERPRET') == '1'
FIELDS = ('a', 'U', 'W', 'count', 'stale', 'dense_of', 'dense_ring')


def make(l, h, k, v, dtype):
    def rand(*shape, dt=torch.float32):
        return torch.randn(shape, dtype=dt, device=DEV)
    return SimpleNamespace(cfg=SimpleNamespace(r=8),
        a=rand(l, 12, h, k), U=rand(l, 12, h, 16, k, dt=dtype),
        W=rand(l, 12, h, 16, v, dt=dtype),
        count=torch.full((l, 12, h), 15, dtype=torch.int32, device=DEV),
        stale=torch.ones(12, dtype=torch.int32, device=DEV),
        dense_of=torch.full((12,), 7, dtype=torch.int32, device=DEV),
        dense_ring=rand(l, 5, h, v, k))


def reference(factors, pool, slots, *, stale_value, dense=None, ring_dst=None):
    a, u, w = factors
    l, _, h, _ = pool.a.shape
    for i in range(l):
        sl = slice(i*h, (i+1)*h)
        store_factored(a[:, sl], u[:, sl].contiguous(), w[:, sl].contiguous(),
            pool.a[i], pool.U[i], pool.W[i], pool.count[i], pool.stale,
            pool.dense_of, slots, pool.cfg.r, stale_value=stale_value,
            dense=dense[i] if dense is not None else None,
            ring=pool.dense_ring[i] if dense is not None else None, ring_dst=ring_dst)


def same(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def case(l, h, k, v, batch, dtype, active, tracked):
    base=make(l,h,k,v,dtype);old=copy.deepcopy(base);new=copy.deepcopy(base)
    for repeat in range(3):
        # Non-contiguous a and slots, padded negative rows, partial ring writes.
        a=torch.randn(batch,l*h,k*2,device=DEV)[...,::2]
        u=torch.randn(batch,l*h,16,k,device=DEV,dtype=dtype)
        w=torch.randn(batch,l*h,16,v,device=DEV,dtype=dtype)
        slots=torch.tensor([2,99,5,99,8,99],device=DEV)[::2][:batch]
        if not active:slots.fill_(-1)
        elif repeat==1:slots[-1]=-1
        elif repeat==2:slots=slots.flip(0)
        ring=torch.tensor([1,-1,3],device=DEV)[:batch]
        dense=torch.randn(l,batch,h,v,k,device=DEV)
        args=dict(stale_value=0,dense=dense,ring_dst=ring)
        reference((a,u,w),old,slots,**args);store_layers((a,u,w),new,slots,**args)
        if tracked:
            # Overlapping tracked destination must win after the main scatter.
            dst=torch.tensor([2,7,9],device=DEV)[:batch]
            if not active:dst.fill_(-1)
            factors=(a+1,u+1,w+1)
            reference(factors,old,dst,stale_value=1)
            store_layers(factors,new,dst,stale_value=1)
        for field in FIELDS:
            assert same(getattr(old,field),getattr(new,field)),(field,repeat)
            if not active:assert same(getattr(new,field),getattr(base,field)),field
    return dict(layers=l,heads=h,key=k,value=v,batch=batch,dtype=str(dtype),
                active=active,tracked=tracked,replays=3,bitwise=True)


def timing():
    l,h,k,v,b=36,24,128,128,1
    pool=make(l,h,k,v,torch.float16)
    factors=(torch.randn(b,l*h,k,device=DEV),
             torch.randn(b,l*h,16,k,device=DEV,dtype=torch.float16),
             torch.randn(b,l*h,16,v,device=DEV,dtype=torch.float16))
    slots=torch.tensor([2],device=DEV);ring=torch.tensor([1],device=DEV)
    dense=torch.randn(l,b,h,v,k,device=DEV)
    def bench(fn):
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):fn(factors,pool,slots,stale_value=0,dense=dense,ring_dst=ring)
        torch.cuda.current_stream().wait_stream(stream)
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):fn(factors,pool,slots,stale_value=0,dense=dense,ring_dst=ring)
        for _ in range(10):g.replay()
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(200):g.replay()
        end.record();end.synchronize()
        return start.elapsed_time(end)*1000/200
    old,new=bench(reference),bench(store_layers)
    return dict(reference_us=old,candidate_us=new,delta_us=new-old,
                scope='36-layer B1 store graph only, cached buffers; excludes factorization and serving')


def main():
    torch.manual_seed(297)
    rows=[]
    shapes=[(1,2,16,8),(3,2,16,16)] if not GPU else [(1,2,16,8),(3,2,16,16),(36,24,128,128)]
    for dims in shapes:
        for b in (1,2,3):
            for dtype in (torch.float16,torch.bfloat16):
                for active,tracked in ((False,False),(False,True),(True,False),(True,True)):
                    rows.append(case(*dims,b,dtype,active,tracked))
    print(json.dumps(dict(complete=True,passed=True,device=DEV,cases=rows,
                         timing=timing() if GPU else None)))


if __name__=='__main__':main()
