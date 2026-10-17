"""TwinStar docs/234 (A'): radix-COW ``FactoredGDNPool.copy_slots`` without host synchronizations.

The default ``copy_slots`` writes the per-slot metadata with scalar advanced assignment
(``stale[dst] = 1``, ``dense_of[dst] = -1``, ``dense_required[dst] = 0``, and ``tensor[n:, dst] = c`` under a
prefix-layer limit).  Each stages a CPU scalar and synchronizes the stream, so an equal-prefix hit extend waits for the
other requests' queued decode at its COW, before any of its own work is enqueued (served r2 layertrace: the arm's first
blocking call, 3 syncs).  ``index_fill_`` writes the same constants with a device-only kernel; tensor-valued copies are
unchanged.  Same bytes, same order of writes.

Opt-in per call: ``SGLANG_GDN_FACTORED_COPY_SYNC_FREE=1`` (read at every COW, so an in-process A/B can switch it per
extend forward).  ``SGLANG_GDN_FACTORED_COPY_SYNC_FREE_CHECK=1`` (gate only, synchronizes): every sync-free COW is also
replayed with the default assignments on clones of the metadata and compared bitwise, and three deliberately wrong
variants (stale / dense_required / prefix_valid) are compared the same way and must be rejected.
"""
from __future__ import annotations

import logging
import os
import time

import torch

logger = logging.getLogger(__name__)

STATS = dict(calls=0, checks=0, mismatch=0, neg_stale=0, neg_required=0, neg_valid=0, neg_na=0)
_LOG_EVERY = 50


def _on(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true", "yes", "y")


def _meta(pool):
    return [pool.stale, pool.dense_of, pool.dense_required, pool.prefix_valid]


def _reference(before, src_index, dst_index):
    """The default copy_slots metadata writes, applied to clones (advanced assignment, as in the pool)."""
    stale, dense_of, required, valid = [None if t is None else t.clone() for t in before]
    stale[dst_index] = 1
    dense_of[dst_index] = -1
    if required is not None:
        required[dst_index] = 0
    if valid is not None:
        valid[dst_index] = valid[src_index]
    return [stale, dense_of, required, valid]


def _equal(xs, ys):
    return all((x is None and y is None) or (x is not None and y is not None and torch.equal(x, y))
               for x, y in zip(xs, ys))


def _check(pool, before, src_index, dst_index, idx):
    ref = _reference(before, src_index, dst_index)
    STATS["checks"] += 1
    if not _equal(_meta(pool), ref):
        STATS["mismatch"] += 1
        if STATS["mismatch"] <= 20:
            logger.error("GDN_COPY_SYNC_FREE_MISMATCH call=%d fields=%s", STATS["calls"],
                         [i for i, (x, y) in enumerate(zip(_meta(pool), ref))
                          if not ((x is None and y is None) or torch.equal(x, y))])
    # Negative controls: the same comparison must reject a wrong stale, dense_required or prefix_valid write.
    wrong = [t.clone() if t is not None else None for t in ref]
    wrong[0].index_fill_(0, idx, 0)
    STATS["neg_stale"] += int(not _equal(wrong, ref))
    for k, make in ((2, lambda t: t.index_fill_(0, idx, 1)),
                    (3, lambda t: t.index_copy_(0, idx, 1 - t.index_select(0, idx)))):
        if ref[k] is None:
            STATS["neg_na"] += 1
            continue
        wrong = [t.clone() if t is not None else None for t in ref]
        make(wrong[k])
        STATS["neg_required" if k == 2 else "neg_valid"] += int(not _equal(wrong, ref))


def copy_slots_sync_free(pool, src_index: torch.Tensor, dst_index: torch.Tensor) -> None:
    """``FactoredGDNPool.copy_slots`` after its publication join and empty check; same writes, no host sync."""
    check = _on("SGLANG_GDN_FACTORED_COPY_SYNC_FREE_CHECK")
    before = [None if t is None else t.clone() for t in _meta(pool)] if check else None
    idx = dst_index.reshape(-1).to(device=pool.stale.device, dtype=torch.long, non_blocking=True)
    n = pool.prefix_layer_count()
    for tensor in (
        pool.a,
        pool.U,
        pool.W,
        pool.count,
        *([pool.warm_v] if pool.warm_v is not None else []),
    ):
        tensor[:n, dst_index] = tensor[:n, src_index]
        if n < len(pool.layer_ids):
            tensor[n:].index_fill_(1, idx, pool.cfg.r if tensor is pool.count else 0)
    pool.stale.index_fill_(0, idx, 1)  # a copied slot is factored-only (compact prefix cache)
    pool.dense_of.index_fill_(0, idx, -1)
    if pool.dense_required is not None:
        pool.dense_required.index_fill_(0, idx, 0)
    if pool.prefix_dense is not None:
        pool.prefix_dense[:, dst_index] = pool.prefix_dense[:, src_index]
    if pool.prefix_valid is not None:
        pool.prefix_valid[dst_index] = pool.prefix_valid[src_index]
    STATS["calls"] += 1
    if STATS["calls"] == 1:
        logger.info("GDN_FACTORED_COPY_SYNC_FREE_ACTIVE (radix COW metadata via index_fill_; docs/234 (A'); "
                    "prefix layers %d/%d, check=%d)", n, len(pool.layer_ids), int(check))
    if check:
        _check(pool, before, src_index, dst_index, idx)
    if STATS["calls"] % (10 if check else _LOG_EVERY) == 0 or (check and 0 < STATS["mismatch"] <= 20):
        logger.info("GDN_COPY_SYNC_FREE_STATS %s t=%d", " ".join(f"{k}={v}" for k, v in STATS.items()), time.time_ns())
