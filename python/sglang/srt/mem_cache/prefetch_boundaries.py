"""Opt-in hybrid endpoints and background agreement for private reservations.

No collective runs in the PP scheduler. Reservations are not tree state and
cannot be used for IO until every rank grants the same legal endpoint.
"""

from bisect import bisect_right
import hashlib
from queue import Queue, Empty
import threading
import time

import torch


def bounded_prefix(boundaries, capacity_pages):
    index = bisect_right(boundaries, max(0, capacity_pages))
    return boundaries[index - 1] if index else 0


def identity_word(operation):
    return int.from_bytes(
        hashlib.sha256(str(operation.request_id).encode()).digest()[:7], "big"
    )


def checked_min(reduce, operation, sequence, value):
    key = identity_word(operation)
    frame = torch.tensor([sequence, -sequence, key, -key, value], dtype=torch.int64)
    reduce(frame)
    if frame[0] != -frame[1] or frame[2] != -frame[3]:
        raise RuntimeError("PP prefetch allocation sequence/request mismatch")
    return int(frame[-1])


def intersect_boundaries(operation, local, max_pages, reduce):
    # Shape is derived from the broadcast request, checked before vector IO.
    size = checked_min(reduce, operation, max_pages, max_pages)
    mask = torch.zeros(size + 1, dtype=torch.int32)
    for page in local:
        if not 0 < page <= size:
            raise ValueError("Invalid restorable prefetch boundary")
        mask[page] = 1
    reduce(mask)
    return torch.nonzero(mask).flatten().tolist()


class PrefetchAllocConsensus:
    """One FIFO ticket per queried request, including misses and local aborts."""

    def __init__(self, reduce, stop_event, page_size):
        self.reduce, self.stop_event, self.page_size = reduce, stop_event, page_size
        self.requests, self.ready = Queue(), Queue()
        self.pending = {}  # only the scheduler accesses private ownership
        self.error = None
        self.sequence = 0
        self.thread = threading.Thread(
            target=self.run, name="prefetch-allocation", daemon=True
        )
        self.thread.start()

    def submit(self, operation, indices):
        key = id(operation)
        if key in self.pending:
            raise RuntimeError("Duplicate prefetch allocation ticket")
        self.pending[key] = (indices, time.monotonic())
        self.requests.put(
            (operation, 0 if indices is None else len(indices) // self.page_size)
        )

    def take(self, operation):
        return self.pending.pop(id(operation))[0]

    def check(self):
        if self.error is not None:
            raise RuntimeError("PP prefetch allocation worker failed") from self.error
        if any(time.monotonic() - entry[1] > 120 for entry in self.pending.values()):
            raise RuntimeError("PP prefetch allocation pending beyond 120 seconds")

    def assert_idle(self):
        self.check()
        if self.pending:
            raise RuntimeError("Cannot reset with private prefetch reservations")

    def run(self):
        try:
            while not self.stop_event.is_set() or not self.requests.empty():
                try:
                    operation, pages = self.requests.get(timeout=0.2)
                except Empty:
                    continue
                self.sequence += 1
                pages = checked_min(self.reduce, operation, self.sequence, pages)
                self.ready.put((operation, pages))
        except Exception as error:
            self.error = error


def record_boundary_proof(cache, operation):
    """Bounded smoke-only record after *real* KV+sidecar GET, never a fake hit.

    Only a root prefix is self-contained for replay after the local-cache flush.
    Log files are compressed on write; dataset payload never goes to plain scratch.
    """
    import gzip
    import json
    import os
    from pathlib import Path

    if (
        operation.storage_start != 0
        or not operation.completed_tokens
        or not any(str(t.name) == "mamba" for t in operation.pool_transfers or ())
    ):
        return
    tokens = getattr(operation, "diagnostic_input_ids", [])[
        : operation.completed_tokens
    ]
    if len(tokens) != operation.completed_tokens:
        return
    key = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
    seen = getattr(cache, "_prefetch_proofs", set())
    if len(seen) >= 8 or key in seen:
        return
    seen.add(key)
    cache._prefetch_proofs = seen
    cc = cache.cache_controller
    root = Path(os.environ.get("Q35_HIGHX_LOG_DIR", "/logs"))
    name = root / f"boundary156-pp{cc.pp_rank}-tp{cc.tp_rank}-{key[:16]}.json.gz"
    payload = dict(
        version=1,
        verified_by="successful_root_hybrid_get",
        pp=cc.pp_rank,
        tp=cc.tp_rank,
        request_id=operation.request_id,
        prefix_sha256=key,
        prefix_tokens=tokens,
        state_hash=operation.hash_value[-1],
        restorable_prefix_pages=getattr(operation, "restorable_prefix_pages", None),
    )
    name.write_bytes(
        gzip.compress(json.dumps(payload).encode(), compresslevel=1, mtime=0)
    )
