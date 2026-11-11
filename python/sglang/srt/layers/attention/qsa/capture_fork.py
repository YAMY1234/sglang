"""Fork a captured side stream from a completed projection, without a main-stream event."""

import ctypes

_driver = None


def _api():
    global _driver
    if _driver is None:
        lib = ctypes.CDLL("libcuda.so.1")
        info = lib.cuStreamGetCaptureInfo_v3
        info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        info.restype = ctypes.c_int
        update = lib.cuStreamUpdateCaptureDependencies_v2
        update.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        update.restype = ctypes.c_int
        _driver = (info, update)
    return _driver


def projection_frontier(stream):
    """The capture nodes the stream's next work would depend on; None outside capture."""
    info, _ = _api()
    status, capture_id, graph = ctypes.c_int(), ctypes.c_uint64(), ctypes.c_void_p()
    nodes = ctypes.POINTER(ctypes.c_void_p)()
    edge_data, count = ctypes.c_void_p(), ctypes.c_size_t()
    error = info(
        stream.cuda_stream,
        ctypes.byref(status),
        ctypes.byref(capture_id),
        ctypes.byref(graph),
        ctypes.byref(nodes),
        ctypes.byref(edge_data),
        ctypes.byref(count),
    )
    if error:
        raise RuntimeError(f"cuStreamGetCaptureInfo_v3: {error}")
    if status.value == 0:
        return None
    if status.value != 1 or not count.value:
        raise RuntimeError("Active projection capture has no dependency frontier")
    # The driver-owned array expires at the next stream API call; the nodes live with the graph.
    return tuple(nodes[i] for i in range(count.value))


def fork_after_projection(stream, nodes):
    """Make the stream's captured dependencies exactly ``nodes`` (full completion edges)."""
    _, update = _api()
    array = (ctypes.c_void_p * len(nodes))(*nodes)
    # NULL edge data: full completion, so the side stream reads the projection only
    # after it is written. Flag 1 (SET) replaces the temporary QKV-event dependency.
    error = update(stream.cuda_stream, array, None, len(nodes), 1)
    if error:
        raise RuntimeError(f"cuStreamUpdateCaptureDependencies_v2: {error}")
