"""Integer-only prefill metadata gather; preserves the host ring policy."""
import os
import torch
import triton
import triton.language as tl


@triton.jit
def _gather(slots, stale, dense_of, required, valid, out,
            N:tl.constexpr, SLOT_STRIDE:tl.constexpr, STALE_STRIDE:tl.constexpr,
            DENSE_STRIDE:tl.constexpr, REQUIRED_STRIDE:tl.constexpr, VALID_STRIDE:tl.constexpr,
            HAS_REQUIRED:tl.constexpr,HAS_VALID:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    s=tl.load(slots+i*SLOT_STRIDE,i<N,other=0).to(tl.int64)
    safe=tl.maximum(s,0)
    a=tl.load(stale+safe*STALE_STRIDE,i<N,other=0).to(tl.int64)
    b=tl.load(dense_of+safe*DENSE_STRIDE,i<N,other=0).to(tl.int64)
    tl.store(out+i,s,i<N);tl.store(out+N+i,a,i<N);tl.store(out+2*N+i,b,i<N)
    if HAS_REQUIRED:
        c=tl.load(required+safe*REQUIRED_STRIDE,i<N,other=0).to(tl.int64)
        tl.store(out+3*N+i,c,i<N)
    if HAS_VALID:
        d=tl.load(valid+safe*VALID_STRIDE,i<N,other=0).to(tl.int64)
        tl.store(out+(3+HAS_REQUIRED)*N+i,d,i<N)


def gather_metadata(slots,stale,dense_of,required=None,valid=None):
    """Return the same int64 (metadata, batch) tensor as independent gathers/stack."""
    if slots.ndim!=1 or slots.dtype!=torch.int64:raise ValueError('expected int64 slot vector')
    if any(t is not None and (t.ndim!=1 or t.device!=slots.device) for t in (stale,dense_of,required,valid)):
        raise ValueError('metadata must be co-located vectors')
    if slots.device.type!='cuda' and os.environ.get('TRITON_INTERPRET')!='1':
        raise ValueError('CPU primitive requires the explicit Triton interpreter')
    n=slots.numel();out=torch.empty((3+int(required is not None)+int(valid is not None),n),dtype=torch.int64,device=slots.device)
    if n:
        _gather[(triton.cdiv(n,128),)](slots,stale,dense_of,required if required is not None else stale,
            valid if valid is not None else stale,out,n,slots.stride(0),stale.stride(0),dense_of.stride(0),
            required.stride(0) if required is not None else 0,valid.stride(0) if valid is not None else 0,
            required is not None,valid is not None,128,num_warps=4)
    return out
