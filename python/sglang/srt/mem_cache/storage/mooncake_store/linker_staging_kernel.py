"""Scatter copies for SGLANG_LINKER_EARLY_READS: ranges given as absolute addresses
(device destinations, pinned-host sources). ``memcpy_batch``: one
cudaMemcpyBatchAsync on the copy engines (SGLANG_LINKER_EARLY_COPY=ce); the Triton
kernel reads the sources through UVA on the SMs (=kernel)."""

import ctypes

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_kernel(dst_ptr, src_ptr, size_ptr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    dst = tl.load(dst_ptr + i).to(tl.pointer_type(tl.int8))
    src = tl.load(src_ptr + i).to(tl.pointer_type(tl.int8))
    n = tl.load(size_ptr + i)
    for off in range(0, n, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < n
        tl.store(dst + idx, tl.load(src + idx, mask=mask), mask=mask)


def scatter_copy(dst: np.ndarray, src: np.ndarray, sizes: np.ndarray, block: int = 4096) -> None:
    """One program per range, on the current stream."""
    args = [torch.from_numpy(np.ascontiguousarray(a, dtype=np.int64)).pin_memory().to("cuda", non_blocking=True)
            for a in (dst, src, sizes)]
    _scatter_kernel[(len(dst),)](*args, BLOCK=block)



class _MemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class MemcpyAttributes(ctypes.Structure):
    """struct cudaMemcpyAttributes (CUDA 13 driver_types.h)."""

    _fields_ = [
        ("srcAccessOrder", ctypes.c_int),
        ("srcLocHint", _MemLocation),
        ("dstLocHint", _MemLocation),
        ("flags", ctypes.c_uint),
    ]


_SRC_ACCESS_ORDER_STREAM = 0x1
_FLAG_PREFER_OVERLAP_WITH_COMPUTE = 0x1
_cudart = None


def _runtime():
    global _cudart
    if _cudart is None:
        try:
            _cudart = ctypes.CDLL("libcudart.so.13")  # torch has loaded it already
        except OSError:
            import glob
            import os

            import nvidia

            _cudart = ctypes.CDLL(glob.glob(os.path.join(nvidia.__path__[0], "cu13", "lib", "libcudart.so.13*"))[0])
        fn = _cudart.cudaMemcpyBatchAsync
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.POINTER(MemcpyAttributes), ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_void_p,
        ]
    return _cudart


def memcpy_batch(dst: np.ndarray, src: np.ndarray, sizes: np.ndarray, stream: int) -> None:
    """cudaMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, stream):
    every entry with one attribute (stream-ordered source access, prefer overlap
    with compute); the copies may complete in any order."""
    dst = np.ascontiguousarray(dst, dtype=np.uint64)
    src = np.ascontiguousarray(src, dtype=np.uint64)
    sizes = np.ascontiguousarray(sizes, dtype=np.uint64)
    attrs = MemcpyAttributes(_SRC_ACCESS_ORDER_STREAM, _MemLocation(), _MemLocation(), _FLAG_PREFER_OVERLAP_WITH_COMPUTE)
    idx = (ctypes.c_size_t * 1)(0)
    def call(n):
        return _runtime().cudaMemcpyBatchAsync(
            dst.ctypes.data, src.ctypes.data, sizes.ctypes.data, n, ctypes.byref(attrs), idx, 1, stream
        )

    rc = call(len(dst))
    if rc != 0:
        raise RuntimeError(f"cudaMemcpyBatchAsync({len(dst)} entries) failed: {_describe(rc, call, dst, src, sizes)}")


def _describe(rc, call, dst, src, sizes) -> str:
    """The error, the calling thread's device and the first failing entry (CUDA 13
    reports no index: shortest failing prefix, re-copying the same bytes)."""
    rt = _runtime()
    rt.cudaGetErrorString.restype = ctypes.c_char_p
    device = ctypes.c_int(-1)
    rt.cudaGetDevice(ctypes.byref(device))
    lo, hi = 1, len(dst)
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if call(mid) != 0 else (mid + 1, hi)
    i = lo - 1

    def where(ptr):
        attrs = (ctypes.c_int * 64)()  # cudaPointerAttributes starts {type, device}
        err = rt.cudaPointerGetAttributes(attrs, ctypes.c_void_p(int(ptr)))
        return f"type {attrs[0]} device {attrs[1]}" if err == 0 else f"attributes error {err}"

    return (f"{rc} {rt.cudaGetErrorString(rc).decode()}; thread device {device.value}; entry {i} of {len(dst)}: "
            f"dst {int(dst[i]):#x} ({where(dst[i])}), src {int(src[i]):#x} ({where(src[i])}), {int(sizes[i])} B")
