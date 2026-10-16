"""Opt-in WS-1 step extension; build before decode CUDA-graph capture."""
import hashlib
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
_extension = None
_attempted = False


def load_extension():
    global _extension, _attempted
    if _attempted:
        return _extension
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("GDN CUDA step must be preloaded before decode graph capture")
    _attempted = True
    src = Path(__file__).resolve().parents[5] / "jit/csrc/attention/gdn_factored_step.cu"
    identity = hashlib.sha256(src.read_bytes() + b"sm_103a-O3-vectorize-v2").hexdigest()[:12]
    try:
        from torch.utils.cpp_extension import load

        _extension = load(
            name="gdn_factored_step_sm103a_" + identity,
            sources=[str(src)],
            extra_cuda_cflags=[
                "-O3", "-std=c++17", "-lineinfo", "--ptxas-options=-v", "--extra-device-vectorization",
                "-gencode", "arch=compute_103a,code=sm_103a",
            ],
            extra_cflags=["-O3"],
            verbose=True,
        )
    except Exception:
        logger.exception("FACTORED_STEP_CUDA_BUILD_FAILED: using Triton step")
    return _extension
