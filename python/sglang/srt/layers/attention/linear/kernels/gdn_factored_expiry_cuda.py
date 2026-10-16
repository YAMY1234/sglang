"""docs/222 §11 R3-a: lean CUDA expiry truncation (opt-in SGLANG_GDN_FACTORED_TRUNC_CUDA=1); build before decode
graph capture (the backend preloads it when the switch is on)."""
import hashlib
import os
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
_extension = None
_attempted = False
_logged = False
# 0 = R3-a (§11.1); 1..8 = the Triton-order MGS (§11.3, 1 + ROWORD + 2 * FUSE + 4 * XW); 9..12 = variant 0 with another
# not-due CTA footprint / carveout (§11.6)
VARIANT = int(os.environ.get("SGLANG_GDN_FACTORED_TRUNC_CUDA_VARIANT", "0"))


def load_extension():
    global _extension, _attempted
    if _attempted:
        return _extension
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("GDN CUDA expiry must be preloaded before decode graph capture")
    _attempted = True
    src = Path(__file__).resolve().parents[5] / "jit/csrc/attention/gdn_factored_expiry.cu"
    hdr = src.with_suffix(".cuh")
    identity = hashlib.sha256(src.read_bytes() + hdr.read_bytes() + b"sm_103a-O3-v2").hexdigest()[:12]
    try:
        from torch.utils.cpp_extension import load

        _extension = load(
            name="gdn_factored_expiry_sm103a_" + identity,
            sources=[str(src)],
            extra_include_paths=[str(src.parent)],
            extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", "--ptxas-options=-v",
                               "-gencode", "arch=compute_103a,code=sm_103a"],
            extra_cflags=["-O3"],
            verbose=True,
        )
        _extension.prepare()  # attributes of the §11.6 variants 9..12 only (variant 0 untouched)
    except Exception:
        _extension = None
        logger.exception("FACTORED_TRUNC_CUDA_BUILD_FAILED: using the Triton expiry")
    return _extension


def supported(fu, fw, r, rfull, iters) -> bool:
    return (fu.dtype == torch.float16 and fw.dtype == torch.float16 and fu.shape[-2:] == (32, 128)
            and fw.shape[-2:] == (32, 128) and r == 16 and iters == 3 and r < rfull <= 32)


def truncate(fu, fw, fcount, indices, r, rfull, iters, rel_tol) -> bool:
    """Run the CUDA expiry if built and the shape is supported; False -> caller uses the Triton route."""
    global _logged
    ext = load_extension()
    if ext is None or not supported(fu, fw, r, rfull, iters):
        return False
    if not _logged:
        _logged = True
        logger.info("FACTORED_TRUNC_CUDA_ACTIVE r=%d rfull=%d variant=%d layers=%s batch=%d", r, rfull, VARIANT,
                    fu.shape[0] if fu.dim() == 5 else 1, indices.numel())
    ext.expiry_truncate(fu, fw, fcount, indices, r, rfull, iters, rel_tol, VARIANT)
    return True


class CondExpiry:
    """docs/222 §12 R8 (SGLANG_GDN_FACTORED_TRUNC_COND=1): the per-layer side-stream expiry inside a CUDA-graph IF
    conditional node whose condition one setter kernel (per decode forward, after layer 0's step) sets on the device.
    Build with `prepare()` before any capture (extension + the IF-body stream); per captured forward call `begin_forward`
    on the side stream at layer 0, then `run(li, launch)` per layer on the side stream: `launch` is the unchanged expiry
    launch, captured as the IF body.  Not-due layers then launch nothing; a due layer runs the same launch as today."""

    def __init__(self):
        ext = load_extension()
        if ext is None:
            raise RuntimeError("SGLANG_GDN_FACTORED_TRUNC_COND=1 but the R3-a/R8 extension did not build")
        self.ext = ext
        self.body = torch.cuda.ExternalStream(ext.cond_body_stream())
        self.handles = None

    def begin_forward(self, count, indices, rfull, setter=True):
        """Handles of the graph being captured (one per layer, default 0 at every launch) + the setter kernel on the
        current (side) stream.  setter=False is the parity's negative control (no body ever runs)."""
        self.handles = self.ext.cond_handles(count.shape[0])
        if setter:
            self.ext.cond_set(count, indices, rfull, self.handles)

    def run(self, li, launch):
        assert self.handles is not None, "R8: begin_forward must run at layer 0 of the captured forward"
        self.ext.cond_begin(self.handles[li])
        try:
            with torch.cuda.stream(self.body):
                launch()
        finally:
            self.ext.cond_end()


_cond = None


def cond_expiry():
    """The process's CondExpiry (created on first call; call before decode-graph capture)."""
    global _cond
    if _cond is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SGLANG_GDN_FACTORED_TRUNC_COND: prepare the R8 extension before decode graph capture")
        _cond = CondExpiry()
    return _cond
