"""Device memory for the linker helper: FABRIC export of the VMM allocations behind
the linker's registered buffers, and their import in the helper (CUDA driver API
through ctypes, as validated on VR200 by the 1-GPU smoke, 8.29.39). Plain
cudaMalloc memory cannot be exported this way: the KV pool must come from a VMM
allocator (SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK)."""

import ctypes

_FABRIC = 0x8
_cu = None


def _lib():
    global _cu
    if _cu is None:
        _cu = ctypes.CDLL("libcuda.so.1")
    return _cu


def _ck(name, *args):
    rc = getattr(_lib(), name)(*args)
    if rc != 0:
        msg = ctypes.c_char_p()
        _lib().cuGetErrorString(rc, ctypes.byref(msg))
        raise RuntimeError(f"{name} -> {rc} {msg.value}")


class _Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AccessDesc(ctypes.Structure):
    _fields_ = [("location", _Loc), ("flags", ctypes.c_int)]


def export_fabric(ranges: list[tuple[int, int]]) -> list[tuple]:
    """For each (ptr, nbytes), the VMM allocation holding it, exported once:
    [("fabric", allocation_base, allocation_size, handle_bytes)]."""
    exports, seen = [], set()
    for ptr, _ in ranges:
        base, size = ctypes.c_uint64(), ctypes.c_size_t()
        _ck("cuMemGetAddressRange_v2", ctypes.byref(base), ctypes.byref(size), ctypes.c_uint64(ptr))
        if base.value in seen:
            continue
        seen.add(base.value)
        handle = ctypes.c_uint64()
        _ck("cuMemRetainAllocationHandle", ctypes.byref(handle), ctypes.c_void_p(base.value))
        buf = (ctypes.c_ubyte * 64)()
        try:
            _ck("cuMemExportToShareableHandle", buf, handle, ctypes.c_int(_FABRIC), ctypes.c_uint64(0))
        finally:
            _ck("cuMemRelease", handle)
        exports.append(("fabric", base.value, size.value, bytes(buf)))
    return exports


def import_fabric(payload: bytes, size: int, device: int | None = None) -> tuple[int, int]:
    """Map one exported allocation here; returns (local address, handle)."""
    if device is None:
        import torch

        device = torch.cuda.current_device()
    handle = ctypes.c_uint64()
    buf = (ctypes.c_ubyte * 64).from_buffer_copy(payload)
    _ck("cuMemImportFromShareableHandle", ctypes.byref(handle), buf, ctypes.c_int(_FABRIC))
    ptr = ctypes.c_uint64()
    _ck("cuMemAddressReserve", ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_size_t(0),
        ctypes.c_uint64(0), ctypes.c_uint64(0))
    _ck("cuMemMap", ptr, ctypes.c_size_t(size), ctypes.c_size_t(0), handle, ctypes.c_uint64(0))
    access = _AccessDesc(_Loc(1, device), 3)  # device location, read-write
    _ck("cuMemSetAccess", ptr, ctypes.c_size_t(size), ctypes.byref(access), ctypes.c_size_t(1))
    return ptr.value, handle.value
