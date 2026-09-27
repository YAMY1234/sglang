"""B1 post-step expiry and radix publication in one graph node.

Each program owns one layer/head, including both its truncation and its copy.
The singleton restriction excludes cross-request source/destination aliasing.
"""
import triton
import triton.language as tl

from .gdn_factored import (
    _factored_expiry_truncate_kernel, MGS_REL_TOL, TRUNC_ITERS,
    TRUNC_WARPS_BY_RMAX,
)


@triton.jit
def _expiry_track(a, u, w, count, stale, prefix, indices, mask, destinations,
                  AL:tl.constexpr, UL:tl.constexpr, WL:tl.constexpr, CL:tl.constexpr,
                  HV:tl.constexpr, K:tl.constexpr, V:tl.constexpr,
                  RMAX:tl.constexpr, R:tl.constexpr, RFULL:tl.constexpr,
                  ITERS:tl.constexpr, REL_TOL:tl.constexpr):
    _factored_expiry_truncate_kernel(
        u,w,count,indices,1,HV,K,V,RMAX,R,RFULL,ITERS,REL_TOL,
        STRIDE_LAYER_U=UL,STRIDE_LAYER_W=WL,STRIDE_LAYER_COUNT=CL)
    # Only this program writes the source layer/head during expiry.
    tl.debug_barrier()
    if tl.load(mask)==0:
        return
    head=tl.program_id(0)
    layer=tl.program_id(1).to(tl.int64)
    src=tl.load(indices).to(tl.int64)
    dst=tl.load(destinations).to(tl.int64)
    if layer==0 and head==0:
        tl.store(prefix+tl.maximum(dst,0),0)
    if src<0 or dst<0 or src==dst:
        return
    k=tl.arange(0,K)
    v=tl.arange(0,V)
    r=tl.arange(0,RMAX)
    source=src*HV+head
    destination=dst*HV+head
    av=tl.load(a+layer*AL+source*K+k)
    uv=tl.load(u+layer*UL+source*RMAX*K+r[:,None]*K+k[None,:])
    wv=tl.load(w+layer*WL+source*RMAX*V+r[:,None]*V+v[None,:])
    cv=tl.load(count+layer*CL+source)
    tl.store(a+layer*AL+destination*K+k,av)
    tl.store(u+layer*UL+destination*RMAX*K+r[:,None]*K+k[None,:],uv)
    tl.store(w+layer*WL+destination*RMAX*V+r[:,None]*V+v[None,:],wv)
    tl.store(count+layer*CL+destination,cv)
    if layer==0 and head==0:
        tl.store(stale+dst,1)


def expiry_track(pool, indices, mask, destinations):
    assert indices.numel()==mask.numel()==destinations.numel()==1
    assert pool.cfg.r==8 and pool.cfg.rfull==16 and pool.U.shape[-2]==16
    assert pool.prefix_valid is not None
    a,u,w,c=pool.a,pool.U,pool.W,pool.count
    _expiry_track[(a.shape[2],a.shape[0])](
        a,u,w,c,pool.stale,pool.prefix_valid,indices,mask,destinations,
        a.stride(0),u.stride(0),w.stride(0),c.stride(0),a.shape[2],a.shape[-1],w.shape[-1],
        u.shape[-2],pool.cfg.r,pool.cfg.rfull,TRUNC_ITERS,MGS_REL_TOL,
        num_warps=TRUNC_WARPS_BY_RMAX[u.shape[-2]])
