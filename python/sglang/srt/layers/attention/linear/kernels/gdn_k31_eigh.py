"""#1019: graph-capturable replacement for the k31 prompt-end ``torch.linalg.eigh`` (fp64, 16 x 16 Gram matrices).

``torch.linalg.eigh`` checks cuSOLVER's info on the host, which synchronises and cannot run inside a CUDA graph; the
whole-layer prefill commit graph was therefore off for init_method=k31.  This is the same decomposition done by a
round-robin (parallel-ordering) cyclic Jacobi in fp64 -- the algorithm of the fork's graph-safe fp32 decode solver
``gdn_truncate._jacobi_vectors``, widened to fp64 and run a fixed number of sweeps past machine-precision convergence
(16 x 16 converges quadratically in ~6 sweeps).  One program per matrix, no host reads.  Output = eigenvalues ascending
and the matching eigenvector columns, as ``torch.linalg.eigh``; eigenvectors equal the reference's up to sign (and up to
a rotation inside exactly degenerate eigenspaces), so the rank-r projector and the stored state match to ~1e-15.
"""
import torch
import triton
import triton.language as tl

SWEEPS = 12


@triton.jit
def _k31_eigh_kernel(G_ptr, D_ptr, Z_ptr, N: tl.constexpr, SWEEPS: tl.constexpr):
    m = tl.program_id(0).to(tl.int64)
    x = tl.arange(0, N)
    eye = x[:, None] == x[None, :]
    G = tl.load(G_ptr + m * N * N + x[:, None] * N + x[None, :])
    Z = eye.to(tl.float64)
    scale = tl.maximum(tl.max(tl.max(tl.abs(G), 1), 0), 1.e-300)
    G = G / scale
    for sweep in range(SWEEPS):
        for turn in range(N - 1):
            partner = tl.where(x == N - 1, turn,
                               tl.where(x == turn, N - 1, (2 * turn - x + 2 * (N - 1)) % (N - 1)))
            diag = tl.sum(tl.where(eye, G, 0.), 1)
            other = tl.gather(diag, partner, 0)
            cross = tl.sum(tl.where(x[None, :] == partner[:, None], G, 0.), 1)
            # one shared (symmetrised) off-diagonal per pair, as in the fp32 solver
            cross = 0.5 * (cross + tl.gather(cross, partner, 0))
            tau = (diag - other) / tl.where(tl.abs(cross) > 1.e-300, 2. * cross, 1.)
            t = tl.where(tau >= 0., 1., -1.) / (tl.abs(tau) + tl.sqrt(1. + tau * tau))
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
    d = tl.sum(tl.where(eye, G, 0.), 1)
    # ascending order (ties by index), as torch.linalg.eigh
    rank = tl.sum(((d[None, :] < d[:, None]) | ((d[None, :] == d[:, None]) & (x[None, :] < x[:, None]))).to(tl.int32), 1)
    order = tl.sum(tl.where(rank[:, None] == x[None, :], x[:, None], 0), 0)
    Z = tl.gather(Z, tl.broadcast_to(order[None, :], (N, N)), 1)
    d = tl.gather(d, order, 0) * scale
    tl.store(D_ptr + m * N + x, d)
    tl.store(Z_ptr + m * N * N + x[:, None] * N + x[None, :], Z)


def eigh(g: torch.Tensor):
    """Batched symmetric fp64 eigendecomposition, including the r16 24x24 Gram.

    Pad non-power-of-two matrices below their Gershgorin spectral bound, then
    discard those extra eigenpairs. All operations stay on device and capture;
    the existing r8 16x16 path executes exactly the original kernel.
    """
    n = g.shape[-1]
    if g.dtype != torch.float64 or n < 1 or g.shape[-2] != n:
        raise ValueError("k31 Jacobi eigh takes square nonempty fp64 matrices")
    if n & (n - 1):
        padded_n = 1 << (n - 1).bit_length()
        padded = torch.zeros(*g.shape[:-2], padded_n, padded_n, dtype=g.dtype, device=g.device)
        padded[..., :n, :n] = g
        bound = g.abs().sum(-1).amax(-1)
        diagonal = -(2 * bound + 1)
        padded[..., n:, n:] = diagonal[..., None, None] * torch.eye(
            padded_n - n, dtype=g.dtype, device=g.device)
        d, z = eigh(padded)
        return d[..., -n:].contiguous(), z[..., :n, -n:].contiguous()
    flat = g.reshape(-1, n, n).contiguous()
    d = torch.empty(flat.shape[:2], dtype=torch.float64, device=g.device)
    z = torch.empty_like(flat)
    if flat.shape[0]:
        _k31_eigh_kernel[(flat.shape[0],)](flat, d, z, N=n, SWEEPS=SWEEPS, num_warps=1)
    return d.reshape(g.shape[:-1]), z.reshape(g.shape)
