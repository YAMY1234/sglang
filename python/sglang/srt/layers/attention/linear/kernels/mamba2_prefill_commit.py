"""Experimental all-layer prompt-end factor publication, no resident dense state.

One CTA owns (layer, request, head). Producers stay alive in the extend plan;
all producers and this consumer use the forward stream; finish publishes valid
only after launch, and the next reader is ordered on that stream. Slot reuse
cannot race an open plan. The pointer table is transient, not a dense pool.

The full kernel ports GDN's iterative cold factorizer (MGS2 range finder and
Rayleigh-Ritz subspace iteration). It is NOT bitwise equivalent to Lightning's
released k31 Cholesky/eigh factorizer; qualification must test that difference.
The QR-only kernel keeps k31's projected state and changes only its factor gauge.
"""
import torch
import triton
import triton.language as tl
from .gdn_factored import _mgs_rolled, MGS_REL_TOL


@triton.jit
def _cold(PTRS, LAYERS, SLOTS, DIR, OMEGA, A, U, V, COUNT,
          H: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
          SIZE: tl.constexpr, RM: tl.constexpr, R: tl.constexpr,
          WIDTH: tl.constexpr, K: tl.constexpr, ITERS: tl.constexpr, REL_TOL: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    layer = tl.load(LAYERS + row).to(tl.int64)
    slot = tl.load(SLOTS + row).to(tl.int64)
    ptr = tl.load(PTRS + row).to(tl.pointer_type(tl.float32))
    p, n, k = tl.arange(0,P), tl.arange(0,N), tl.arange(0,K)
    s = tl.load(ptr + head*P*N + p[:,None]*N+n[None,:])
    d = tl.load(DIR + (layer*H+head)*P+p)
    a = tl.sum(s*d[:,None],0)/tl.maximum(tl.sum(d*d,0),1.e-12)
    c = s-d[:,None]*a[None,:]
    omega = tl.load(OMEGA+head*N*WIDTH+n[:,None]*WIDTH+k[None,:],k[None,:]<WIDTH,0.)
    y = tl.dot(c,omega,input_precision='ieee')
    for _ in range(ITERS):
        y = _mgs_rolled(y,k,WIDTH,2,REL_TOL)
        y = tl.dot(c,tl.dot(tl.trans(c),y,input_precision='ieee'),input_precision='ieee')
    q = _mgs_rolled(y,k,WIDTH,2,REL_TOL)
    t = tl.dot(tl.trans(c),q,input_precision='ieee')
    gram = tl.dot(tl.trans(t),t,input_precision='ieee')
    diag = tl.sum(tl.where(k[:,None]==k[None,:],gram,0.),1)
    order = tl.sum(((diag[None,:]>diag[:,None]) | ((diag[None,:]==diag[:,None]) & (k[None,:]<k[:,None]))).to(tl.int32),1)
    z = ((order[:,None]==k[None,:]) & (k[None,:]<R) & (k[:,None]<WIDTH)).to(tl.float32)
    for _ in range(ITERS):
        z = tl.dot(gram,tl.dot(gram,z,input_precision='ieee'),input_precision='ieee')
        z = _mgs_rolled(z,k,R,2,REL_TOL)
    left = tl.dot(q,z,input_precision='ieee')
    # Decode appends assume orthonormal U. Roundoff in q@z is reorthogonalized.
    left = _mgs_rolled(left,k,R,2,REL_TOL)
    right = tl.dot(tl.trans(left),c,input_precision='ieee')
    dst=(layer*SIZE+slot)*H+head
    tl.store(A+dst*N+n,a)
    tl.store(U+dst*RM*P+k[:,None]*P+p[None,:],tl.trans(left),k[:,None]<R)
    tl.store(V+dst*RM*N+k[:,None]*N+n[None,:],right,k[:,None]<R)
    tl.store(COUNT+dst,R)


@triton.jit
def _qr_store(LEFT, RIGHT, COEFF, LAYERS, SLOTS, A, U, V, COUNT,
              H: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
              SIZE: tl.constexpr, RM: tl.constexpr, R: tl.constexpr, REL_TOL: tl.constexpr):
    row,head=tl.program_id(0),tl.program_id(1)
    layer=tl.load(LAYERS+row).to(tl.int64);slot=tl.load(SLOTS+row).to(tl.int64)
    p,n,r=tl.arange(0,P),tl.arange(0,N),tl.arange(0,R)
    src=row*H+head;dst=(layer*SIZE+slot)*H+head
    left=tl.load(LEFT+src*P*R+p[:,None]*R+r[None,:])
    right=tl.load(RIGHT+src*R*N+r[:,None]*N+n[None,:])
    q=_mgs_rolled(left,r,R,2,REL_TOL)
    transform=tl.dot(tl.trans(q),left,input_precision='ieee')
    v=tl.dot(transform,right,input_precision='ieee')
    a=tl.load(COEFF+src*N+n)
    tl.store(A+dst*N+n,a)
    tl.store(U+dst*RM*P+r[:,None]*P+p[None,:],tl.trans(q))
    tl.store(V+dst*RM*N+r[:,None]*N+n[None,:],v)
    tl.store(COUNT+dst,R)


def commit_device(pool, plan):
    pointers, layers, rows = [], [], []
    for layer, row_ids, state in plan.pending:
        assert state.is_contiguous() and state.dtype == torch.float32
        stride=state.stride(0)*state.element_size()
        for i,row in enumerate(row_ids):
            pointers.append(state.data_ptr()+i*stride);layers.append(layer);rows.append(row)
    ptrs=torch.tensor(pointers,device=pool.device,dtype=torch.int64)
    layer_ids=torch.tensor(layers,device=pool.device,dtype=torch.int64)
    slots=plan.slots[torch.tensor(rows,device=pool.device,dtype=torch.int64)]
    _,size,heads,capacity,p=pool.u.shape
    width=pool._probes[False].shape[-1]
    assert pool.rank==16 and p==64 and pool.n==128 and capacity==32
    _cold[(len(rows),heads)](ptrs,layer_ids,slots,pool.directions,pool._probes[False],
        pool.coeff,pool.u,pool.v,pool.fcount,heads,p,pool.n,size,capacity,pool.rank,
        width,triton.next_power_of_2(width),4,MGS_REL_TOL,num_warps=8,enable_fp_fusion=False)


def qr_store(pool,layers,slots,coeff,left,right):
    # Contiguity is part of the kernel contract, not assumed from linalg views.
    coeff,left,right=(t.contiguous() for t in (coeff,left,right))
    _,size,heads,capacity,p=pool.u.shape
    assert pool.rank==16 and p==64 and pool.n==128
    _qr_store[(layers.numel(),heads)](left,right,coeff,layers,slots,pool.coeff,pool.u,
        pool.v,pool.fcount,heads,p,pool.n,size,capacity,pool.rank,MGS_REL_TOL,num_warps=8,enable_fp_fusion=False)
