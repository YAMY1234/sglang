"""Reusable transient resources for NVIDIA EAGLE decode rounds."""

from __future__ import annotations

import torch

from sglang.srt.speculative.prepared_next_draft import (
    prepared_next_draft_enabled,
)
from sglang.srt.utils.common import async_d2h, is_pin_memory_available

_ROUND_RING_DEPTH = 8
_COPY_EVENT_RING_DEPTH = 4

_join_events: dict[tuple[int, int], torch.cuda.Event] = {}
_copy_event_ring: list[torch.cuda.Event] = []
_copy_event_pos = 0
_round_tensor_rings: dict[tuple, list] = {}
_host_tensor_rings: dict[tuple, list] = {}


def enabled() -> bool:
    return prepared_next_draft_enabled()


def wait_stream(waiting_stream, source_stream) -> None:
    """Order two streams, reusing one event for each stream pair when enabled."""
    if not enabled():
        waiting_stream.wait_stream(source_stream)
        return
    key = (waiting_stream.cuda_stream, source_stream.cuda_stream)
    event = _join_events.get(key)
    if event is None:
        event = _join_events[key] = torch.cuda.Event()
    event.record(source_stream)
    waiting_stream.wait_event(event)


def empty(shape: tuple, *, dtype: torch.dtype, device) -> torch.Tensor:
    """Allocate or reuse a fully overwritten, round-local device tensor."""
    if not enabled():
        return torch.empty(shape, dtype=dtype, device=device)
    key = (shape, dtype, torch.device(device))
    ring = _round_tensor_rings.get(key)
    if ring is None:
        ring = _round_tensor_rings[key] = [
            [
                torch.empty(shape, dtype=dtype, device=device)
                for _ in range(_ROUND_RING_DEPTH)
            ],
            0,
        ]
    position = ring[1]
    ring[1] = (position + 1) % _ROUND_RING_DEPTH
    return ring[0][position]


def copy_done_event(device_module):
    """Return an event that remains unused for at least four result copies."""
    global _copy_event_pos
    if not enabled():
        return device_module.Event()
    if not _copy_event_ring:
        _copy_event_ring.extend(
            device_module.Event() for _ in range(_COPY_EVENT_RING_DEPTH)
        )
    event = _copy_event_ring[_copy_event_pos]
    _copy_event_pos = (_copy_event_pos + 1) % _COPY_EVENT_RING_DEPTH
    return event


def copy_to_host(tensor: torch.Tensor) -> torch.Tensor:
    """Copy into a reusable pinned buffer when speculative decode is enabled."""
    if (
        not enabled()
        or tensor.device.type == "cpu"
        or not is_pin_memory_available(tensor.device)
    ):
        return async_d2h(tensor)
    key = (tensor.device, tuple(tensor.shape), tensor.dtype)
    ring = _host_tensor_rings.get(key)
    if ring is None:
        ring = _host_tensor_rings[key] = [
            [
                torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
                for _ in range(_ROUND_RING_DEPTH)
            ],
            0,
        ]
    host = ring[0][ring[1]]
    ring[1] = (ring[1] + 1) % _ROUND_RING_DEPTH
    host.copy_(tensor, non_blocking=True)
    tensor.record_stream(torch.get_device_module(tensor.device).current_stream())
    return host
