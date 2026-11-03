"""Reusable events for the NVIDIA megakernel overlap scheduler."""

import torch

from sglang.srt.environ import envs

_ENABLED = envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get()
_JOIN_EVENTS = {}
_COPY_DONE_RING = []
_COPY_DONE_POS = 0


def wait_stream(dst, src):
    if not _ENABLED:
        return dst.wait_stream(src)
    key = (dst.cuda_stream, src.cuda_stream)
    event = _JOIN_EVENTS.get(key)
    if event is None:
        event = _JOIN_EVENTS[key] = torch.cuda.Event()
    # Re-recording does not change a wait that has already been enqueued.
    event.record(src)
    dst.wait_event(event)


def copy_done_event(device_module):
    """Results are consumed one overlap pass later, before this ring wraps."""
    global _COPY_DONE_POS
    if not _ENABLED:
        return device_module.Event()
    if not _COPY_DONE_RING:
        _COPY_DONE_RING.extend(device_module.Event() for _ in range(4))
    event = _COPY_DONE_RING[_COPY_DONE_POS]
    _COPY_DONE_POS = (_COPY_DONE_POS + 1) % 4
    return event
