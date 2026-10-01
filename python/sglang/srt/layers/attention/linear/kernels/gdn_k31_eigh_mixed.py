"""Experimental R3-a: same-threshold fp32 Jacobi and one fp64 Newton correction.

Only the explicitly enabled 36-layer/864-matrix prefill caller uses this.
The admitted fp64 kernel and all decode/expiry paths remain independent.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _fp32_jacobi(G_ptr, D_ptr, Z_ptr, COUNT_ptr, N: tl.constexpr, SWEEPS: tl.constexpr,
                     EARLY_EXIT: tl.constexpr = False):
    m = tl.program_id(0).to(tl.int64)
    x = tl.arange(0, N)
    eye = x[:, None] == x[None, :]
    G = tl.load(G_ptr + m * N * N + x[:, None] * N + x[None, :])
    Z = eye.to(tl.float32)
    scale = tl.max(tl.max(tl.abs(G), 1), 0)
    scale = tl.where(scale > 0., scale, 1.)
    G = G / scale
    sweep = 0
    active = True
    completed = 0
    while (sweep < SWEEPS) & active:
        if EARLY_EXIT:
            # A complete sweep is the identity when every pair already takes
            # the existing t=0 branch. Check that *same* per-pair threshold,
            # including its symmetrisation, rather than relaxing convergence
            # or lowering the fixed twelve-sweep upper bound. Nothing leaves
            # the device and the branch is uniform within this matrix's CTA.
            diagonal = tl.sum(tl.where(eye, G, 0.), 1)
            cross_all = 0.5 * (G + tl.trans(G))
            threshold = 1.e-20 * tl.sqrt(tl.abs(
                diagonal[:, None] * diagonal[None, :]))
            rotating = tl.where(eye, False, tl.abs(cross_all) > threshold)
            active = tl.max(tl.max(rotating.to(tl.int32), 1), 0) != 0
        if active:
            completed += 1
            for turn in range(N - 1):
                partner = tl.where(x == N - 1, turn,
                                   tl.where(x == turn, N - 1, (2 * turn - x + 2 * (N - 1)) % (N - 1)))
                diag = tl.sum(tl.where(eye, G, 0.), 1)
                other = tl.gather(diag, partner, 0)
                cross = tl.sum(tl.where(x[None, :] == partner[:, None], G, 0.), 1)
                # one shared (symmetrised) off-diagonal per pair, as in the fp32 solver
                cross = 0.5 * (cross + tl.gather(cross, partner, 0))
                tau = (diag - other) / tl.where(tl.abs(cross) > 1.e-300, 2. * cross, 1.)
                # Algebraically identical tangent, without fp32 tau**2 overflow.
                inv = 1. / tl.maximum(tl.abs(tau), 1.)
                scaled_tau = tau * inv
                t = (tl.where(tau >= 0., 1., -1.) * inv /
                     (tl.abs(scaled_tau) + tl.sqrt(inv * inv + scaled_tau * scaled_tau)))
                t = tl.where(diag == other, tl.where(x < partner, 1., -1.), t)
                # rotate unless the pair is already diagonal to well below fp64 resolution
                t = tl.where(tl.abs(cross) > 1.e-20 * tl.sqrt(tl.abs(diag * other)), t, 0.)
                c = 1. / tl.sqrt(1. + t * t)
                s = t * c
                pc = tl.broadcast_to(partner[None, :], (N, N))
                pr = tl.broadcast_to(partner[:, None], (N, N))
                G = G * c[None, :] + tl.gather(G, pc, 1) * s[None, :]
                G = G * c[:, None] + tl.gather(G, pr, 0) * s[:, None]
                Z = Z * c[None, :] + tl.gather(Z, pc, 1) * s[None, :]
        sweep += 1
    d = tl.sum(tl.where(eye, G, 0.), 1)
    # ascending order (ties by index), as torch.linalg.eigh
    rank = tl.sum(((d[None, :] < d[:, None]) | ((d[None, :] == d[:, None]) & (x[None, :] < x[:, None]))).to(tl.int32), 1)
    order = tl.sum(tl.where(rank[:, None] == x[None, :], x[:, None], 0), 0)
    Z = tl.gather(Z, tl.broadcast_to(order[None, :], (N, N)), 1)
    d = tl.gather(d, order, 0) * scale
    tl.store(D_ptr + m * N + x, d)
    tl.store(Z_ptr + m * N * N + x[:, None] * N + x[None, :], Z)

    tl.store(COUNT_ptr + m, completed)


def fp32_vectors(g):
    """Input is the public dispatcher's padded fp64 Gram (N=32)."""
    n=g.shape[-1]
    assert n==32 and g.dtype==torch.float64
    flat=g.reshape(-1,n,n).contiguous()
    # Normalize before the fp32 cast so zero/tiny and large Grams stay finite.
    scale=flat.abs().amax(dim=(-2,-1),keepdim=True).clamp_min(1e-300)
    normalized=(flat/scale).float()
    d=torch.empty(flat.shape[:2],device=g.device,dtype=torch.float32)
    z=torch.empty_like(normalized)
    count=torch.empty(flat.shape[0],device=g.device,dtype=torch.int32)
    _fp32_jacobi[(flat.shape[0],)](normalized,d,z,count,N=32,SWEEPS=12,EARLY_EXIT=True,num_warps=1)
    return z.double(),count


def correct_once(g,q):
    """One Newton update for both GQ=Q Lambda and Q^T Q=I.

    In close eigenspaces the divided spectral gap is unsafe: retain the
    fp32-selected basis and correct its metric, with Rayleigh quotients.
    One Newton polar normalization removes the update's second-order metric
    error. There is no extra eigensolve or hidden fp64 fallback.
    """
    n=g.shape[-1];eye=torch.eye(n,device=g.device,dtype=torch.float64)
    metric=q.transpose(-1,-2)@q
    projected=q.transpose(-1,-2)@(g@q)
    values=projected.diagonal(dim1=-2,dim2=-1)/metric.diagonal(dim1=-2,dim2=-1)
    error=metric-eye
    gaps=values[...,None,:]-values[...,:,None]
    scale=values.abs().amax(-1,keepdim=True).clamp_min(1e-300)[...,None]
    safe=gaps.abs()>torch.finfo(torch.float32).eps**.5*scale
    residual=projected-error*values[...,None,:]
    update=torch.where(safe,residual/torch.where(safe,gaps,torch.ones_like(gaps)),-.5*error)
    corrected=q+q@update
    corrected=corrected@(.5*(3*eye-corrected.transpose(-1,-2)@corrected))
    corrected=corrected/corrected.square().sum(-2,keepdim=True).sqrt().clamp_min(1e-300)
    values=(corrected*(g@corrected)).sum(-2)
    order=values.argsort(dim=-1,stable=True)
    return values.gather(-1,order),corrected.gather(-1,order[...,None,:].expand_as(corrected))


def eigh(g):
    n=g.shape[-1];flat=g.reshape(-1,n,n).contiguous()
    q,_=fp32_vectors(flat)
    d,z=correct_once(flat,q)
    return d.reshape(g.shape[:-1]),z.reshape(g.shape)
