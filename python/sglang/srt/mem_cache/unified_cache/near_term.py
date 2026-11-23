"""Near-term prefetch policy for the external linker (pure Python, no torch).

Loads start only for queued requests the next few prefill batches will admit,
and only while the device keeps a reserve for forced chunk continuations.
"""

from __future__ import annotations

from collections.abc import Iterable, Hashable


def plan_window(
    queue: Iterable[tuple[Hashable, int, int]],
    chunk_tokens: int,
    batches: int,
    restore_ratio: float = 0.0,
    ahead0: int = 0,
) -> dict:
    """Handles whose load must start now, each mapped to the work ahead of it.

    ``queue`` is (handle, work_tokens, restorable_tokens) in admission order, where
    work is what the request prefills after its restore. A request is in the window
    when the work ahead of it fits ``batches`` chunks plus its own restore lead
    (restore_ratio prefill tokens per restored token). The head is always included.
    A load must leave the work ahead of its request free, or it could take the
    memory an earlier request needs to be admitted. ``ahead0`` is work already
    ahead of the whole queue (an in-flight chunked request's tail); with it the
    head is no longer always included.
    """
    base = max(batches, 0) * max(chunk_tokens, 0)
    window, ahead = {}, max(ahead0, 0)
    for i, (handle, work, restorable) in enumerate(queue):
        if (i == 0 and not ahead0) or ahead <= base + restore_ratio * max(restorable, 0):
            window[handle] = ahead
        ahead += max(work, 0)
    return window


def plan_from_queue(
    reqs, chunk_tokens: int, batches: int, restorable=None, restore_ratio: float = 0.0,
    ahead0: int = 0, info: dict | None = None,
) -> tuple[dict, Hashable]:
    """Window and head from waiting requests in admission order; ``restorable`` maps
    a handle to the pool tokens its finished lookup found (0 while unknown). With
    ``info``, info["head_work"] is the head's own work (tokens it still computes)."""
    queue = []
    for r in reqs:
        handle = r.cache_request_handle
        restore = restorable(handle) if restorable is not None else 0
        queue.append((handle, max(len(r.origin_input_ids) - r.prefix_len - restore, 0), restore))
    if info is not None:
        info["head_work"] = queue[0][1] if queue else 0
    return (plan_window(queue, chunk_tokens, batches, restore_ratio, ahead0),
            queue[0][0] if queue else None)


def load_fits(free_tokens: int, load_tokens: int, reserve_tokens: int) -> bool:
    """A load may take its slots only if free + evictable stays >= the reserve."""
    return free_tokens - load_tokens >= reserve_tokens
