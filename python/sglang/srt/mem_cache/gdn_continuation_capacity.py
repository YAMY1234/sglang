"""Capacity-only growth for exact, unfinished prefill continuation states.

The initial ring is a reservation, not a limit on valid prefill batches.
Only mandatory destinations which the planner cannot place trigger growth;
completed rows keep their existing optional retention policy.
"""
import logging

import torch

logger = logging.getLogger(__name__)


def grow_continuation_ring(pool, missing: int) -> None:
    """Append exactly the missing positions, preserving every old state byte.

Called before the planner publishes ownership, outside CUDA capture. Readers
of captured ring addresses must key their cache by ``ring_generation``.
"""
    if missing <= 0:
        return
    old = pool.dense_ring
    capacity = len(pool.ring_owner)
    target = capacity + missing
    limit = pool.ring_capacity_limit
    if target > limit:
        raise RuntimeError(
            "x256 dense ring exhausted by unfinished prompts: "
            f"required={target}, request capacity={limit}, current={capacity}"
        )
    if old.is_cuda:
        with torch.cuda.device(old.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("cannot grow GDN continuation ring during CUDA capture")
            # Growth is rare and occurs in host planning. Drain all consumers
            # before moving a tensor whose slices may have crossed streams.
            torch.cuda.synchronize(old.device)
    shape = list(old.shape)
    shape[1] = target
    replacement = old.new_zeros(shape)
    replacement[:, :capacity].copy_(old)
    if old.is_cuda:
        torch.cuda.synchronize(old.device)
    # Allocation/copy failures leave both host and device authority unchanged.
    pool.dense_ring = replacement
    pool.ring_owner.extend([-1] * missing)
    pool.ring_lru.extend(range(capacity, target))
    pool.ring_generation = getattr(pool, "ring_generation", 0) + 1
    logger.info(
        "GDN continuation ring grew %d -> %d (request limit %d): "
        "additional %.3f MiB, actual %.3f MiB, generation %d",
        capacity, target, limit, (replacement.nbytes - old.nbytes) / (1 << 20),
        replacement.nbytes / (1 << 20), pool.ring_generation,
    )
