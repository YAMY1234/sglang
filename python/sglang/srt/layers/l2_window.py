"""Persisting-L2 access-policy window on the lm_head weight (SGLANG_Q35MF_LL16_XLG_MB, a measurement switch).

While the switch is set, every lm_head launch on the current stream (also when captured into a CUDA graph) carries a
window over the first SGLANG_Q35MF_LL16_XLG_MB MiB of the weight (hit ratio 1, hits persisting, misses streaming), and
the context's persisting-L2 limit is raised to the device maximum. Only the cache policy changes, never a value.
"""

import ctypes
import logging
from contextlib import contextmanager

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_LIMIT_PERSISTING = 6  # CU_LIMIT_PERSISTING_L2_CACHE_SIZE
_ATTR_WINDOW = 1  # CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW == CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW
_PROP_NORMAL, _PROP_STREAMING, _PROP_PERSISTING = 0, 1, 2
_ATTR_MAX_PERSISTING = 108  # CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE
_ATTR_MAX_WINDOW = 109  # CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE


class _Window(ctypes.Structure):
    _fields_ = [
        ("base_ptr", ctypes.c_void_p),
        ("num_bytes", ctypes.c_size_t),
        ("hitRatio", ctypes.c_float),
        ("hitProp", ctypes.c_int),
        ("missProp", ctypes.c_int),
    ]


class _AttrValue(ctypes.Union):
    _fields_ = [("window", _Window), ("pad", ctypes.c_char * 64)]


_cu = None
_window_bytes_max = 0


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} -> CUresult {rc}")


def _init() -> None:
    """Load the driver API once and raise the persisting-L2 limit to the device maximum."""
    global _cu, _window_bytes_max
    if _cu is not None:
        return
    cu = ctypes.CDLL("libcuda.so.1")
    dev, v = ctypes.c_int(0), ctypes.c_int(0)
    _check(cu.cuCtxGetDevice(ctypes.byref(dev)), "cuCtxGetDevice")
    _check(cu.cuDeviceGetAttribute(ctypes.byref(v), _ATTR_MAX_PERSISTING, dev), "cuDeviceGetAttribute")
    max_persisting = v.value
    _check(cu.cuDeviceGetAttribute(ctypes.byref(v), _ATTR_MAX_WINDOW, dev), "cuDeviceGetAttribute")
    _check(cu.cuCtxSetLimit(_LIMIT_PERSISTING, ctypes.c_size_t(max_persisting)), "cuCtxSetLimit")
    _window_bytes_max = min(max_persisting, v.value)
    _cu = cu
    logger.info(
        "lm_head L2 window: persisting-L2 limit set to %d B (window at most %d B).",
        max_persisting,
        _window_bytes_max,
    )


def _set_stream_window(stream: torch.cuda.Stream, base: int, nbytes: int) -> None:
    v = _AttrValue()
    v.window.base_ptr = base
    v.window.num_bytes = nbytes
    v.window.hitRatio = 1.0 if nbytes else 0.0
    v.window.hitProp = _PROP_PERSISTING if nbytes else _PROP_NORMAL
    v.window.missProp = _PROP_STREAMING if nbytes else _PROP_NORMAL
    _check(
        _cu.cuStreamSetAttribute(ctypes.c_void_p(stream.cuda_stream), _ATTR_WINDOW, ctypes.byref(v)),
        "cuStreamSetAttribute",
    )


def reset_persisting() -> None:
    """Demote every persisting line to normal (a block of the A/B harness that runs without the window)."""
    if _cu is not None:
        _check(_cu.cuCtxResetPersistingL2Cache(), "cuCtxResetPersistingL2Cache")


@contextmanager
def lm_head_window(weight: torch.Tensor):
    """Launches inside carry the window over the start of `weight` when SGLANG_Q35MF_LL16_XLG_MB > 0."""
    mb = envs.SGLANG_Q35MF_LL16_XLG_MB.get()
    if mb <= 0 or not weight.is_cuda:
        yield
        return
    _init()
    stream = torch.cuda.current_stream()
    nbytes = min(mb * 2**20, _window_bytes_max, weight.numel() * weight.element_size())
    _set_stream_window(stream, weight.data_ptr(), nbytes)
    try:
        yield
    finally:
        _set_stream_window(stream, 0, 0)
