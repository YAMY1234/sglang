"""K3-A experimental graph-safe small Gram eigensolver and split projection.

Rows store factor columns. Round-robin Jacobi rotates N/2 disjoint pairs
concurrently; all N-1 rounds visit every pair once. No host-side expiry read.
The split variant publishes an active flag separately from count so projection
programs cannot race with the program that resets count.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _jacobi_vectors(G, N: tl.constexpr, R: tl.constexpr, SWEEPS: tl.constexpr):
    x = tl.arange(0, N)
    eye = x[:, None] == x[None, :]
    Z = eye.to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(G)), 1.e-30)
    G = G / scale
    for sweep in range(SWEEPS):
        for turn in range(N - 1):
            partner = tl.where(x == N - 1, turn,
                              tl.where(x == turn, N - 1, (2 * turn - x + 2 * (N - 1)) % (N - 1)))
            diag = tl.sum(tl.where(eye, G, 0.), 1)
            other = tl.gather(diag, partner, 0)
            cross = tl.sum(tl.where(x[None, :] == partner[:, None], G, 0.), 1)
            # Use the same rounded off-diagonal for both rotations: applying
            # the left/right products can introduce a tiny asymmetry. Without
            # this, nearly equal diagonals can yield non-orthogonal pair angles.
            cross = 0.5 * (cross + tl.gather(cross, partner, 0))
            # Each column i uses c*col_i + s_i*col_partner; signs are opposite
            # in a pair. tau_i=(a_ii-a_jj)/(2*a_ij) gives that convention.
            tau = (diag - other) / tl.where(tl.abs(cross) > 1.e-30, 2. * cross, 1.)
            t = tl.where(tau >= 0., 1., -1.) / (tl.abs(tau) + tl.sqrt(1. + tau * tau))
            # Equal diagonals require opposite signs explicitly.
            t = tl.where(diag == other, tl.where(x < partner, 1., -1.), t)
            t = tl.where(tl.abs(cross) > 1.e-12 * tl.sqrt(tl.abs(diag * other)), t, 0.)
            c = tl.rsqrt(1. + t * t)
            s = t * c
            pc = tl.broadcast_to(partner[None, :], (N, N))
            pr = tl.broadcast_to(partner[:, None], (N, N))
            G = G * c[None, :] + tl.gather(G, pc, 1) * s[None, :]
            G = G * c[:, None] + tl.gather(G, pr, 0) * s[:, None]
            Z = Z * c[None, :] + tl.gather(Z, pc, 1) * s[None, :]
    d = tl.sum(tl.where(eye, G, 0.), 1)
    rank = tl.sum(((d[None, :] > d[:, None]) |
                   ((d[None, :] == d[:, None]) & (x[None, :] < x[:, None]))).to(tl.int32), 1)
    order = tl.sum(tl.where(rank[:, None] == x[None, :], x[:, None], 0), 0)
    Z = tl.gather(Z, tl.broadcast_to(order[None, :], (N, N)), 1)
    return tl.where(x[None, :] < R, Z, 0.)


@triton.jit
def _jacobi_vectors_unrolled(G, N: tl.constexpr, R: tl.constexpr, SWEEPS: tl.constexpr):
    x = tl.arange(0, N)
    eye = x[:, None] == x[None, :]
    Z = eye.to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(G)), 1.e-30)
    G = G / scale
    for sweep in range(SWEEPS):
        for turn in tl.static_range(N - 1):
            partner = tl.where(x == N - 1, turn,
                              tl.where(x == turn, N - 1, (2 * turn - x + 2 * (N - 1)) % (N - 1)))
            diag = tl.sum(tl.where(eye, G, 0.), 1)
            other = tl.gather(diag, partner, 0)
            cross = tl.sum(tl.where(x[None, :] == partner[:, None], G, 0.), 1)
            # Use the same rounded off-diagonal for both rotations: applying
            # the left/right products can introduce a tiny asymmetry. Without
            # this, nearly equal diagonals can yield non-orthogonal pair angles.
            cross = 0.5 * (cross + tl.gather(cross, partner, 0))
            # Each column i uses c*col_i + s_i*col_partner; signs are opposite
            # in a pair. tau_i=(a_ii-a_jj)/(2*a_ij) gives that convention.
            tau = (diag - other) / tl.where(tl.abs(cross) > 1.e-30, 2. * cross, 1.)
            t = tl.where(tau >= 0., 1., -1.) / (tl.abs(tau) + tl.sqrt(1. + tau * tau))
            # Equal diagonals require opposite signs explicitly.
            t = tl.where(diag == other, tl.where(x < partner, 1., -1.), t)
            t = tl.where(tl.abs(cross) > 1.e-12 * tl.sqrt(tl.abs(diag * other)), t, 0.)
            c = tl.rsqrt(1. + t * t)
            s = t * c
            pc = tl.broadcast_to(partner[None, :], (N, N))
            pr = tl.broadcast_to(partner[:, None], (N, N))
            G = G * c[None, :] + tl.gather(G, pc, 1) * s[None, :]
            G = G * c[:, None] + tl.gather(G, pr, 0) * s[:, None]
            Z = Z * c[None, :] + tl.gather(Z, pc, 1) * s[None, :]
    d = tl.sum(tl.where(eye, G, 0.), 1)
    rank = tl.sum(((d[None, :] > d[:, None]) |
                   ((d[None, :] == d[:, None]) & (x[None, :] < x[:, None]))).to(tl.int32), 1)
    order = tl.sum(tl.where(rank[:, None] == x[None, :], x[:, None], 0), 0)
    Z = tl.gather(Z, tl.broadcast_to(order[None, :], (N, N)), 1)
    return tl.where(x[None, :] < R, Z, 0.)


@triton.jit
def _mgs_gather(Y, N: tl.constexpr, R: tl.constexpr):
    x = tl.arange(0, N)
    Q = Y
    n0 = tl.sqrt(tl.sum(Y * Y, 0))
    for p in tl.static_range(2):
        for j in tl.static_range(R):
            y = tl.gather(Q, tl.full((N, 1), j, tl.int32), 1).reshape((N,))
            proj = tl.where(x < j, tl.sum(Q * y[:, None], 0), 0.)
            y = y - tl.sum(Q * proj[None, :], 1)
            n = tl.sqrt(tl.sum(y*y, 0))
            n0j = tl.gather(n0, tl.full((1,), j, tl.int32), 0).sum(0)
            ok = n > 1.e-12
            if p == 0:
                ok = ok & (n > 1.e-4*n0j)
            y = tl.where(ok, y/tl.maximum(n, 1.e-30), 0.)
            Q = tl.where(x[None, :] == j, y[:, None], Q)
    return Q


@triton.jit
def _subspace_vectors(G, N: tl.constexpr, R: tl.constexpr, ITERS: tl.constexpr):
    x = tl.arange(0, N)
    d = tl.sum(tl.where(x[:,None] == x[None,:], G, 0.), 1)
    rank = tl.sum(((d[None,:] > d[:,None]) | ((d[None,:] == d[:,None]) & (x[None,:] < x[:,None]))).to(tl.int32), 1)
    Z = ((rank[:,None] == x[None,:]) & (x[None,:] < R)).to(tl.float32)
    for i in range(ITERS):
        Z = tl.dot(G, Z, input_precision="ieee")
        Z = _mgs_gather(Z, N, R)
    return Z


@triton.jit
def _jacobi_expiry(U, W, Count, Indices, Zbuf, Active,
                   STRIDE: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                   RMAX: tl.constexpr, FULL: tl.constexpr, R: tl.constexpr,
                   SWEEPS: tl.constexpr, SPLIT: tl.constexpr, PRECISION: tl.constexpr, UNROLL: tl.constexpr, SUBSPACE: tl.constexpr):
    pid = tl.program_id(0)
    slot = tl.load(Indices + (pid // H) * STRIDE).to(tl.int64)
    head = slot * H + pid % H
    active = False
    if slot >= 0:
        active = tl.load(Count + head) >= FULL
    if SPLIT:
        tl.store(Active + pid, active)
    if not active:
        return
    x = tl.arange(0, RMAX)
    d = tl.arange(0, D)
    ptr = head * RMAX * D + x[:, None] * D + d[None, :]
    w = tl.load(W + ptr, x[:, None] < FULL, 0).to(tl.float32)
    G = tl.dot(w, tl.trans(w), input_precision=PRECISION)
    if SUBSPACE:
        Z = _subspace_vectors(G, RMAX, R, SWEEPS)
    elif UNROLL:
        Z = _jacobi_vectors_unrolled(G, RMAX, R, SWEEPS)
    else:
        Z = _jacobi_vectors(G, RMAX, R, SWEEPS)
    if SPLIT:
        tl.store(Zbuf + pid * RMAX * RMAX + x[:, None] * RMAX + x[None, :], Z)
    else:
        u = tl.load(U + ptr, x[:, None] < FULL, 0).to(tl.float32)
        un = tl.dot(tl.trans(Z), u, input_precision=PRECISION)
        wn = tl.dot(tl.trans(Z), w, input_precision=PRECISION)
        tl.store(U + ptr, un, x[:, None] < R)
        tl.store(W + ptr, wn, x[:, None] < R)
        tl.store(Count + head, R)


@triton.jit
def _gram_stage(W, Count, Indices, Gram, Active, STRIDE: tl.constexpr,
                H: tl.constexpr, D: tl.constexpr, RMAX: tl.constexpr, FULL: tl.constexpr, LIB: tl.constexpr = False, PRECISION: tl.constexpr = "ieee"):
    pid = tl.program_id(0)
    slot = tl.load(Indices+(pid//H)*STRIDE).to(tl.int64)
    head = slot*H + pid%H
    active = False
    if slot >= 0:
        active = tl.load(Count+head)>=FULL
    tl.store(Active+pid, active)
    x=tl.arange(0,RMAX); d=tl.arange(0,D)
    if not active:
        if LIB:
            tl.store(Gram+pid*RMAX*RMAX+x[:,None]*RMAX+x[None,:],(x[:,None]==x[None,:]).to(tl.float32))
        return
    w=tl.load(W+head*RMAX*D+x[:,None]*D+d[None,:],x[:,None]<FULL,0)
    if PRECISION == "ieee":
        w=w.to(tl.float32)
    g=tl.dot(w,tl.trans(w),input_precision=PRECISION)
    tl.store(Gram+pid*RMAX*RMAX+x[:,None]*RMAX+x[None,:],g)


@triton.jit
def _project_split(U, W, Count, Indices, Zbuf, Active,
                   STRIDE: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                   RMAX: tl.constexpr, FULL: tl.constexpr, R: tl.constexpr,
                   BLOCK: tl.constexpr, PRECISION: tl.constexpr):
    pid = tl.program_id(0)
    tile = tl.program_id(1)
    if tl.load(Active + pid) == 0:
        return
    slot = tl.load(Indices + (pid // H) * STRIDE).to(tl.int64)
    head = slot * H + pid % H
    x = tl.arange(0, RMAX)
    d = (tile % (D // BLOCK)) * BLOCK + tl.arange(0, BLOCK)
    z = tl.load(Zbuf + pid * RMAX * RMAX + x[:, None] * RMAX + x[None, :])
    ptr = head * RMAX * D + x[:, None] * D + d[None, :]
    if tile < D // BLOCK:
        v = tl.load(U + ptr, x[:, None] < FULL, 0).to(tl.float32)
        out = tl.dot(tl.trans(z), v, input_precision=PRECISION)
        tl.store(U + ptr, out, x[:, None] < R)
    else:
        v = tl.load(W + ptr, x[:, None] < FULL, 0).to(tl.float32)
        out = tl.dot(tl.trans(z), v, input_precision=PRECISION)
        tl.store(W + ptr, out, x[:, None] < R)
    if tile == 0:
        tl.store(Count + head, R)


@triton.jit
def _project_one(U, W, Count, Indices, Zbuf, Active,
                 STRIDE: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                 RMAX: tl.constexpr, FULL: tl.constexpr, R: tl.constexpr, LIB: tl.constexpr = False, PRECISION: tl.constexpr = "ieee"):
    pid=tl.program_id(0)
    if tl.load(Active+pid)==0: return
    slot=tl.load(Indices+(pid//H)*STRIDE).to(tl.int64)
    head=slot*H+pid%H
    x=tl.arange(0,RMAX); d=tl.arange(0,D)
    if LIB:
        z=tl.load(Zbuf+pid*RMAX*RMAX+x[:,None]+(RMAX-1-x[None,:])*RMAX,x[None,:]<R,0)
    else:
        z=tl.load(Zbuf+pid*RMAX*RMAX+x[:,None]*RMAX+x[None,:])
    ptr=head*RMAX*D+x[:,None]*D+d[None,:]
    u=tl.load(U+ptr,x[:,None]<FULL,0).to(tl.float32)
    w=tl.load(W+ptr,x[:,None]<FULL,0).to(tl.float32)
    un=tl.dot(tl.trans(z),u,input_precision=PRECISION)
    wn=tl.dot(tl.trans(z),w,input_precision=PRECISION)
    tl.store(U+ptr,un,x[:,None]<R);tl.store(W+ptr,wn,x[:,None]<R)
    tl.store(Count+head,R)


def truncate(U, W, count, indices, r, full, *, sweeps=5, split=False,
             warps=4, precision="ieee", scratch=None, unroll=False, subspace=False):
    """In-place expiry only. scratch=(float32 [B*H,N,N], int32 [B*H])."""
    _, h, n, d = U.shape
    assert W.shape == U.shape and n in (16, 32) and d == 128
    b = indices.numel()
    if split:
        if scratch is None:
            scratch = (torch.empty((b*h, n, n), device=U.device, dtype=torch.float32),
                       torch.empty((b*h,), device=U.device, dtype=torch.int32))
        z, active = scratch
    else:
        z, active = U, count
    _jacobi_expiry[(b*h,)](U, W, count, indices, z, active, STRIDE=indices.stride(0),
                          H=h, D=d, RMAX=n, FULL=full, R=r, SWEEPS=sweeps,
                          SPLIT=split, PRECISION=precision, UNROLL=unroll, SUBSPACE=subspace, num_warps=warps)
    if split:
        _project_split[(b*h, 8)](U, W, count, indices, z, active, STRIDE=indices.stride(0),
                                H=h, D=d, RMAX=n, FULL=full, R=r, BLOCK=32,
                                PRECISION=precision, num_warps=4)


# Scratch is stream-local and shared by sequential layers. This avoids allocating
# one large eigenspace buffer for every layer in every captured graph shape.
_TENSOR_SCRATCH = {}


def truncate_tensor(U, W, count, indices, r, full, *, iters=3, passes=2, extension=None, split=False, precision="ieee", rows=False, lanes=0, whole=False, parallel=False, parallel_lanes=32, chol=0, lu=0, lu_chol=0):
    if extension is None:
        from .gdn_jacobi_cuda import load_extension
        extension = load_extension()
    if whole:
        ((extension.parallel if parallel_lanes==32 else getattr(extension,f"parallel{parallel_lanes}")) if parallel else ((extension.wholecholp2 if chol==2 else extension.wholechol) if chol else ((extension.wholelup2 if lu==2 else extension.wholelu) if lu else ((extension.wholelucholp2 if lu_chol==2 else extension.wholeluchol) if lu_chol else extension.whole))))(U,W,count,indices,r,full,iters,passes)
        return
    _, h, n, d = U.shape
    assert n == 32 and d == 128 and r <= 16 and full <= n
    assert U.dtype == W.dtype and U.dtype in (torch.float32, torch.bfloat16)
    b = indices.numel()
    key = (U.device.index, torch.cuda.current_stream().cuda_stream, b, h, n)
    if key not in _TENSOR_SCRATCH:
        _TENSOR_SCRATCH[key] = (torch.empty((b*h,n,n),device=U.device,dtype=torch.float32),
                              torch.empty((b*h,n,n),device=U.device,dtype=torch.float32),
                              torch.empty((b*h,),device=U.device,dtype=torch.int32))
    gram,z,active = _TENSOR_SCRATCH[key]
    _gram_stage[(b*h,)](W,count,indices,gram,active,STRIDE=indices.stride(0),H=h,D=d,RMAX=n,FULL=full,PRECISION=precision,num_warps=4)
    if split:
        (getattr(extension,f"tensorlanes{lanes}") if lanes else (extension.tensorrows if rows else extension.tensorvectors))(gram,z,active,r,iters,passes,U,W,count,indices,full)
        _project_one[(b*h,)](U,W,count,indices,z,active,STRIDE=indices.stride(0),H=h,D=d,RMAX=n,FULL=full,R=r,PRECISION=precision,num_warps=4)
        return
    extension.tensorproject1(gram,z,active,r,iters,passes,U,W,count,indices,full)
