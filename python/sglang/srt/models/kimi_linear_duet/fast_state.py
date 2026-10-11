"""Batched FP32 KDA range finder with a reference-precision small Gram solver.

Matrices are independent across the leading (layer, slot) dimension. Fixed
batch-one probes preserve the reference RNG contract for every request.
"""

import torch

from sglang.srt.duet.state_factor import OMEGA_SEED, OVERSAMPLE

from .small_eigh import small_eigh

MGS_REL_TOL = 0.0
MGS_PASSES = 4


def probes(heads, values, keys, rank, device):
    width = min(rank + OVERSAMPLE, values, keys)
    generator = torch.Generator(device=device).manual_seed(OMEGA_SEED)
    cold = torch.randn(
        1, heads, values, width, device=device, generator=generator, dtype=torch.float32
    )
    generator.manual_seed(OMEGA_SEED)
    extra = torch.randn(
        1,
        heads,
        values,
        width - rank,
        device=device,
        generator=generator,
        dtype=torch.float32,
    )
    return cold, extra


def factorize(
    state,
    direction,
    rank,
    previous,
    use_previous,
    omega,
    *,
    explicit=True,
    diagnostics=None,
):
    """(M,H,K,V) -> sink coefficient, U, V, next right basis, all fp32.

    Range finder and factors are FP32; only the small Gram eigensolve is FP64.

    Reconstruct as coeff[..., :, None] * direction[..., None, :] + U @ V.T.
    Reuse Flash-Next MGS with four passes; dependent columns are zeroed in FP32.
    Four passes retain near-dependent columns without the default relative
    drop tolerance; real warm states require this to meet the FP64 error gate.
    Eigenvector signs/rotations are intentionally unspecified.
    """
    from sglang.srt.layers.attention.linear.kernels.gdn_prefill_reference import (
        _ieee_fp32_matmul,
    )

    with _ieee_fp32_matmul():
        return _factorize(
            state,
            direction,
            rank,
            previous,
            use_previous,
            omega,
            explicit=explicit,
            diagnostics=diagnostics,
        )


def _factorize(
    state,
    direction,
    rank,
    previous,
    use_previous,
    omega,
    *,
    explicit=True,
    diagnostics=None,
):
    sf = state.float()
    direction = direction.to(device=sf.device, dtype=torch.float32)
    coeff = torch.einsum("mhkv,mhv->mhk", sf, direction)
    coeff = coeff / direction.square().sum(-1).clamp_min(1e-12)[..., None]
    if not explicit:
        coeff = torch.zeros_like(coeff)
    content = sf - coeff[..., :, None] * direction[..., None, :]
    cold, extra = omega
    warm_probe = torch.cat(
        (previous.float(), extra.expand(*previous.shape[:-1], -1)), -1
    )
    probe = torch.where(use_previous[:, None, None, None], warm_probe, cold)

    def orth(y):
        if y.is_cuda:
            from .mgs import orthonormalize

            # The shared Triton helper writes its contiguous input in place.
            # Clone so factor V remains unchanged when computing the warm basis.
            return orthonormalize(y.clone(), passes=MGS_PASSES, rel_tol=MGS_REL_TOL)
        from sglang.srt.mem_cache.gdn_factored_pool import gram_schmidt

        return gram_schmidt(y, passes=MGS_PASSES, rel_tol=MGS_REL_TOL)

    y = content @ probe
    y = content @ orth(content.transpose(-1, -2) @ orth(y))
    q = orth(y)
    reduced = q.transpose(-1, -2) @ content
    gram = (reduced @ reduced.transpose(-1, -2)).double()
    jitter = 1e-7 * gram.diagonal(dim1=-2, dim2=-1).mean(-1) + 1e-30
    gram = gram + jitter[..., None, None] * torch.eye(
        gram.shape[-1], device=sf.device, dtype=torch.float64
    )

    # KDA states can be nearly low-rank: an FP32 eigensolver loses the low
    # energy subspace around the rank boundary (real-dump CPU gate). Only this
    # small Gram uses FP64, through Kimi's unpadded specialization of Flash-Next's Jacobi.
    solved = (
        small_eigh(gram)
        if diagnostics is None
        else small_eigh(gram, diagnostics=diagnostics)
    )
    rotation = solved[1][..., -rank:].float()
    u = q @ rotation
    v = (u.transpose(-1, -2) @ content).transpose(-1, -2)
    return coeff, u, v, orth(v)


def project(
    state,
    direction,
    rank,
    previous,
    use_previous,
    omega,
    *,
    explicit=True,
    diagnostics=None,
):
    if rank >= min(state.shape[-2:]):
        return state, None
    from sglang.srt.layers.attention.linear.kernels.gdn_prefill_reference import (
        _ieee_fp32_matmul,
    )

    with _ieee_fp32_matmul():
        coeff, u, v, basis = factorize(
            state,
            direction,
            rank,
            previous,
            use_previous,
            omega,
            explicit=explicit,
            diagnostics=diagnostics,
        )
        dense = coeff[..., :, None] * direction[..., None, :] + u @ v.transpose(-1, -2)
    return dense.to(state.dtype), basis
