"""Fused prompt-end k31 in CUDA (docs/223 §9.4; round-2 R1 after the Triton version hit the micro kill line).

Same interface and algorithm as gdn_k31_fused.factorize_k31_fused (the Triton version, kept for the CPU-interpreter
cross-check): `a` with the reference torch ops, kernels/jit/csrc/attention/gdn_k31_fused.cu phase 1, the admitted
Triton _fp32_jacobi launched as fp32_vectors does, then the CUDA finalize (Newton + factors).

The extension is JIT-compiled with torch.utils.cpp_extension on first use and cached under TORCH_EXTENSIONS_DIR.
"""
import os
from functools import lru_cache

import torch

from .gdn_k31_eigh_mixed import _fp32_jacobi
from .gdn_k31_fused import DIM, M_DIRECTIONS, M_PAD, _fallback_counter, fused_supported

_EXT_NAME = "gdn_k31_fused_cuda"


@lru_cache(maxsize=1)
def load_ext():
    from torch.utils import cpp_extension

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "../../../../../kernels/jit/csrc/attention/gdn_k31_fused.cu")
    arch = os.environ.get("SGLANG_GDN_K31_FUSED_ARCH", "103a")
    return cpp_extension.load(
        name=_EXT_NAME,
        sources=[os.path.normpath(src)],
        extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", "--ptxas-options=-v",
                           "-gencode", f"arch=compute_{arch},code=sm_{arch}"],
        extra_cflags=["-O3"],
        verbose=os.environ.get("SGLANG_GDN_K31_FUSED_VERBOSE", "0") == "1",
    )


def _timed(profile, name, fn):
    if profile is None:
        fn()
        return
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    profile.append((name, start, end))


def factorize_k31_cuda(s, vbar, omega, heads, r, rmax, dtype, *, debug=None, force_full_newton=False,
                       profile=None):
    """s (B, LH, V, K) with LH = layers * heads, vbar (LH, V), omega (B or 1, heads, V, 24) -> a, U, W as
    factorize_prefill_k31(mixed_eigh=True, mixed_cholqr=True). debug: dict receiving the per-matrix stage tensors
    (same keys as the Triton version). profile: list receiving (kernel, start event, end event)."""
    if not fused_supported(s, vbar, omega, r, rmax, dtype):
        raise ValueError("k31 fused: unsupported geometry")
    ext = load_ext()
    s = s.float().contiguous()
    vb = vbar.float()
    a = torch.einsum("bhvk,hv->bhk", s, vb) / vb.square().sum(-1).clamp_min(1e-12)[None, :, None]
    b, lh = s.shape[:2]
    n = b * lh
    vb = vb.contiguous()
    a = a.contiguous()
    om = omega.float()
    if om.stride(-1) != 1 or om.stride(-2) != M_DIRECTIONS:
        om = om.contiguous()
    om_sb = 0 if om.shape[0] == 1 else om.stride(0)
    dev = s.device
    q = torch.empty(n, DIM, M_PAD, device=dev, dtype=torch.float32)
    g64 = torch.empty(n, M_PAD, M_PAD, device=dev, dtype=torch.float64)
    gn = torch.empty(n, M_PAD, M_PAD, device=dev, dtype=torch.float32)
    d = torch.empty(n, M_PAD, device=dev, dtype=torch.float32)
    z = torch.empty(n, M_PAD, M_PAD, device=dev, dtype=torch.float32)
    sweeps = torch.empty(n, device=dev, dtype=torch.int32)
    u = torch.empty(b, lh, rmax, DIM, device=dev, dtype=dtype)
    w = torch.empty(b, lh, rmax, DIM, device=dev, dtype=dtype)
    dbg = debug is not None
    dbg1 = torch.empty(n, 5, DIM, M_PAD, device=dev, dtype=torch.float32) if dbg else q
    dbg2 = torch.empty(n, M_PAD + 1, M_PAD, device=dev, dtype=torch.float64) if dbg else g64
    count = _fallback_counter(dev)
    _timed(profile, "phase1", lambda: ext.phase1(s, vb, a, om, q, g64, gn, count, dbg1, lh, heads, om_sb,
                                                 om.stride(1), dbg))
    _timed(profile, "jacobi", lambda: _fp32_jacobi[(n,)](gn, d, z, sweeps, N=32, SWEEPS=12, EARLY_EXIT=True,
                                                          num_warps=1))
    _timed(profile, "finalize", lambda: ext.finalize(s, vb, a, q, g64, z, u, w, dbg2, lh, rmax,
                                                     force_full_newton, dbg))
    if dbg:
        debug.update(q=q, g64=g64, gn=gn, z=z, stages=dbg1, eig=dbg2)
    return a, u, w
