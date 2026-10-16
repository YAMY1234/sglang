"""docs/222 §11 R3-a: lean CUDA expiry truncation (opt-in SGLANG_GDN_FACTORED_TRUNC_CUDA=1); build before decode
graph capture (the backend preloads it when the switch is on)."""
import hashlib
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
_extension = None
_attempted = False
_logged = False


def load_extension():
    global _extension, _attempted
    if _attempted:
        return _extension
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("GDN CUDA expiry must be preloaded before decode graph capture")
    _attempted = True
    src = Path(__file__).resolve().parents[5] / "jit/csrc/attention/gdn_factored_expiry.cu"
    hdr = src.with_suffix(".cuh")
    identity = hashlib.sha256(src.read_bytes() + hdr.read_bytes() + b"sm_103a-O3-v1").hexdigest()[:12]
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
    except Exception:
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
        logger.info("FACTORED_TRUNC_CUDA_ACTIVE r=%d rfull=%d layers=%s batch=%d", r, rfull,
                    fu.shape[0] if fu.dim() == 5 else 1, indices.numel())
    ext.expiry_truncate(fu, fw, fcount, indices, r, rfull, iters, rel_tol)
    return True
