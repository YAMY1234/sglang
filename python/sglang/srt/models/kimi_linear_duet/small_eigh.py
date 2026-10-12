"""Kimi's unpadded batched small-Gram Jacobi adapter and CPU math mirror.

The rotation/order rules follow Flash-Next's FP64 round-robin Jacobi. Keeping
this specialization in Kimi leaves every shared/Flash-Next byte unchanged.
"""

import torch

REL_TOL = 1e-14
MAX_SWEEPS = 12


def small_eigh(gram, *, cpu_jacobi=False):
    if gram.dtype != torch.float64 or gram.shape[-2] != gram.shape[-1]:
        raise ValueError("small Gram must be symmetric square FP64")
    n = gram.shape[-1]
    if n < 2 or n > 64 or n % 2:
        from sglang.srt.duet.state_factor import small_eigh as shared_small_eigh

        # Preserve arbitrary CLI ranks outside the even small-Gram fast path.
        return shared_small_eigh(gram)
    if gram.is_cuda:
        from .jacobi import eigh

        return eigh(gram, rel_tol=REL_TOL, max_sweeps=MAX_SWEEPS)
    if cpu_jacobi:
        return jacobi_cpu(gram)
    return torch.linalg.eigh(gram)


def jacobi_cpu(gram, *, rel_tol=REL_TOL, max_sweeps=MAX_SWEEPS, diagnostics=None):
    """CPU mirror of the GPU rotations, masking and off-diagonal norm gate.

    CPU host decisions here are validation only; CUDA calls the device kernel.
    """
    n = gram.shape[-1]
    if n < 2 or n % 2:
        raise ValueError("round-robin Jacobi requires even dimension >= 2")
    scale = gram.abs().amax(dim=(-1, -2)).clamp_min(1e-300)
    g = gram / scale[..., None, None]
    z = torch.eye(n, dtype=gram.dtype, device=gram.device).expand_as(g).clone()
    eye = torch.eye(n, dtype=torch.bool, device=gram.device)
    x = torch.arange(n, device=gram.device)
    counts = torch.zeros(gram.shape[:-2], dtype=torch.int32)
    for sweep in range(max_sweeps):
        active = g.masked_fill(eye, 0).square().sum((-1, -2)) > (
            rel_tol**2 * g.square().sum((-1, -2))
        )
        if not active.any():
            break
        counts += active.to(counts.dtype)
        for turn in range(n - 1):
            partner = torch.where(
                x == n - 1,
                turn,
                torch.where(x == turn, n - 1, (2 * turn - x + 2 * (n - 1)) % (n - 1)),
            )
            diag = g.diagonal(dim1=-2, dim2=-1)
            other = diag[..., partner]
            cross = g[..., x, partner]
            cross = 0.5 * (cross + cross[..., partner])
            tau = (diag - other) / torch.where(cross.abs() > 1e-300, 2 * cross, 1)
            t = torch.where(tau >= 0, 1.0, -1.0) / (
                tau.abs() + (1 + tau.square()).sqrt()
            )
            t = torch.where(diag == other, torch.where(x < partner, 1.0, -1.0), t)
            t = torch.where(
                active[..., None] & (cross.abs() > 1e-30 * (diag * other).abs().sqrt()),
                t,
                0,
            )
            c = (1 + t.square()).rsqrt()
            sine = t * c
            g = g * c[..., None, :] + g[..., :, partner] * sine[..., None, :]
            g = g * c[..., :, None] + g[..., partner, :] * sine[..., :, None]
            z = z * c[..., None, :] + z[..., :, partner] * sine[..., None, :]
    d = g.diagonal(dim1=-2, dim2=-1)
    order = d.argsort(dim=-1, stable=True)
    d = d.gather(-1, order) * scale[..., None]
    z = z.gather(-1, order[..., None, :].expand_as(z))
    if diagnostics is not None:
        diagnostics["sweeps"] = counts.tolist()
        diagnostics["relative_off_diagonal"] = (
            g.masked_fill(eye, 0).norm(dim=(-1, -2))
            / g.norm(dim=(-1, -2)).clamp_min(1e-300)
        ).tolist()
    return d, z
