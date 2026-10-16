"""Admitted k31 prefill R3-a: same-threshold fp32 Jacobi and one fp64 Newton correction.

Only the service-gated 36-layer/864-matrix prefill caller defaults to this.
SGLANG_GDN_K31_MIXED_EIGH=0 preserves the FP64 control.
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


# ---------------------------------------------------------------------------- B' (docs/223 §4.3, lead 20:2x)
# The mixed solver keeps its fp32-selected basis where an eigengap is below
# sqrt(eps32) * scale (correct_once). At the rank-r cut that can keep a
# different r-th direction than the fp64 Jacobi, i.e. a different kept span.
# Matrices with a near-degenerate cut that still carries energy are routed to
# the fp64 Jacobi (the pre-B b >= 2 solver) in a fixed-capacity sub-batch;
# everything is static-shaped (stable argsort + index_select/index_copy).
GAP_ROUTE_GAP = 3.45e-4      # (l_r - l_{r+1}) / l_1 below this: near-degenerate cut
GAP_ROUTE_FLOOR = 6e-8       # l_r / l_1 above this: the cut carries resolvable energy
_GAP_ROUTE_STATS = {}


def gap_route_flags(d, r):
    """d: ascending eigenvalues (..., N) of the padded Gram; padded pairs are the smallest."""
    top, cut, nxt = d[..., -1], d[..., -r], d[..., -r - 1]
    scale = top.abs().clamp_min(1e-300)
    return ((cut - nxt) < GAP_ROUTE_GAP * scale) & (cut > GAP_ROUTE_FLOOR * scale)


def gap_route_stats(device):
    """(routed, overflow) device counters since start (host sync; call rarely)."""
    entry = _GAP_ROUTE_STATS.get(torch.device(device))
    return (0, 0) if entry is None else tuple(int(t.item()) for t in entry)


def eigh_gap_routed(g, *, r=16, capacity=0.4, fp64_solver=None, mixed_solver=None):
    """Mixed eigh, with near-degenerate-cut matrices replaced by the fp64 Jacobi result."""
    if mixed_solver is None:
        mixed_solver = eigh
    if fp64_solver is None:
        def fp64_solver(m):
            import os
            from .gdn_k31_eigh import eigh as fp64_eigh
            return fp64_eigh(m, early_exit=os.environ.get("SGLANG_GDN_K31_EIGH_EARLY_EXIT", "1") == "1")
    n = g.shape[-1]
    flat = g.reshape(-1, n, n).contiguous()
    d, z = mixed_solver(flat)
    d, z = d.reshape(-1, n).clone(), z.reshape(-1, n, n).clone()
    flags = gap_route_flags(d, r)
    count = flat.shape[0]
    cap = max(1, min(count, int(-(-capacity * count // 1))))
    # flagged first (stable), fixed size: graph-safe gather of up to `cap` matrices
    order = torch.argsort((~flags).to(torch.int32), stable=True)[:cap]
    picked = flags.index_select(0, order)
    d64, z64 = fp64_solver(flat.index_select(0, order))
    d_new = torch.where(picked[:, None], d64.reshape(cap, n), d.index_select(0, order))
    z_new = torch.where(picked[:, None, None], z64.reshape(cap, n, n), z.index_select(0, order))
    d.index_copy_(0, order, d_new)
    z.index_copy_(0, order, z_new)
    stats = _GAP_ROUTE_STATS.get(g.device)
    if stats is None:
        stats = _GAP_ROUTE_STATS[g.device] = (torch.zeros((), dtype=torch.int64, device=g.device),
                                              torch.zeros((), dtype=torch.int64, device=g.device))
    routed = flags.sum()
    stats[0].add_(routed)
    stats[1].add_((routed - cap).clamp_min(0))
    return d.reshape(g.shape[:-1]), z.reshape(g.shape)

