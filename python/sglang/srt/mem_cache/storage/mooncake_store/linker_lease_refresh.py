"""Renew the store leases of prefixes the device serves, for the direct linker.

Mooncake evicts the objects with the oldest lease first, and only a lookup or a
read renews a lease (an existence query grants one). A prefix that keeps hitting
on the device is never looked up in the store, so its lease dates from its write
and it is evicted before cold objects; once it leaves the device it misses. The
scheduler hands such a hit's page hashes to a background thread that re-queries
their objects under a key-rate cap, each anchor node at most once per interval.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Optional

logger = logging.getLogger(__name__)

# One master RPC per this many object keys.
_KEYS_PER_QUERY = 4096
_STATS_LOG_INTERVAL_S = 60.0


class LinkerLeaseRefresher:
    def __init__(
        self,
        *,
        build_keys: Callable[[list[str]], list[str]],
        renew: Callable[[list[str]], list[int]],
        interval_s: float,
        max_keys_per_s: int,
        max_pending_pages: int = 1 << 20,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.interval_s = interval_s
        self._build_keys = build_keys
        self._renew = renew
        self._max_keys_per_s = max_keys_per_s
        self._max_pending_pages = max_pending_pages
        self._clock = clock
        # Scheduler thread only: anchor node -> last submission.
        self._stamps: dict[object, float] = {}
        self._next_prune = clock() + interval_s
        # Shared with the worker under _cond.
        self._cond = threading.Condition()
        self._pending: deque[list[str]] = deque()
        self._pending_pages = 0
        self._stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Worker only.
        self._tokens = float(max_keys_per_s)
        self._tokens_at = clock()
        self.stats = {
            "prefixes": 0,
            "pages": 0,
            "keys": 0,
            "present": 0,
            "dropped_pages": 0,
            "errors": 0,
        }

    # ---- scheduler thread ----

    def is_due(self, anchor: object) -> bool:
        """Whether the prefix ending at ``anchor`` was not renewed this interval."""
        now = self._clock()
        if now >= self._next_prune:
            fresh_after = now - self.interval_s
            self._stamps = {a: t for a, t in self._stamps.items() if t > fresh_after}
            self._next_prune = now + self.interval_s
        stamp = self._stamps.get(anchor)
        return stamp is None or stamp <= now - self.interval_s

    def submit(self, anchor: object, page_hashes: list[str]) -> None:
        self._stamps[anchor] = self._clock()
        with self._cond:
            self._pending.append(page_hashes)
            self._pending_pages += len(page_hashes)
            # Over the cap the oldest submissions are dropped until their
            # anchors come due again (counted in dropped_pages).
            while self._pending_pages > self._max_pending_pages:
                dropped = self._pending.popleft()
                self._pending_pages -= len(dropped)
                self.stats["dropped_pages"] += len(dropped)
            self._cond.notify()

    # ---- worker thread ----

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="linker-lease-refresh", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        with self._cond:
            self._cond.notify()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        next_log = self._clock() + _STATS_LOG_INTERVAL_S
        logged = dict(self.stats)
        while not self._stopping.is_set():
            batch = self._take_pending(timeout=_STATS_LOG_INTERVAL_S)
            if batch:
                self.renew_batch(batch)
            now = self._clock()
            if now >= next_log:
                self._log_stats(logged, _STATS_LOG_INTERVAL_S + now - next_log)
                logged = dict(self.stats)
                next_log = now + _STATS_LOG_INTERVAL_S

    def _take_pending(self, timeout: float) -> list[list[str]]:
        with self._cond:
            if not self._pending and not self._stopping.is_set():
                self._cond.wait(timeout=timeout)
            batch = list(self._pending)
            self._pending.clear()
            self._pending_pages = 0
        return batch

    def renew_batch(self, batch: list[list[str]]) -> None:
        keys = []
        for page_hashes in batch:
            keys.extend(self._build_keys(page_hashes))
            self.stats["prefixes"] += 1
            self.stats["pages"] += len(page_hashes)
        for start in range(0, len(keys), _KEYS_PER_QUERY):
            if self._stopping.is_set():
                return
            chunk = keys[start : start + _KEYS_PER_QUERY]
            self._acquire(len(chunk))
            try:
                results = self._renew(chunk)
            except Exception:
                if self.stats["errors"] == 0:
                    logger.exception("Linker store lease refresh query failed")
                self.stats["errors"] += 1
                continue
            self.stats["keys"] += len(chunk)
            self.stats["present"] += sum(1 for r in results if r == 1)

    def _acquire(self, num_keys: int) -> None:
        """Token bucket over keys, one second of burst."""
        if self._max_keys_per_s <= 0:
            return
        now = self._clock()
        self._tokens = min(
            float(self._max_keys_per_s),
            self._tokens + (now - self._tokens_at) * self._max_keys_per_s,
        )
        self._tokens_at = now
        if self._tokens < num_keys:
            self._stopping.wait((num_keys - self._tokens) / self._max_keys_per_s)
            self._tokens_at = self._clock()
            self._tokens = float(num_keys)
        self._tokens -= num_keys

    def _log_stats(self, last: dict[str, int], elapsed_s: float) -> None:
        delta = {name: self.stats[name] - last[name] for name in self.stats}
        logger.info(
            "Linker store lease refresh: %d prefixes, %d pages, %d keys (%.1f%% "
            "present, %.0f keys/s), %d pages dropped, %d errors in %.0f s; totals %s",
            delta["prefixes"],
            delta["pages"],
            delta["keys"],
            100.0 * delta["present"] / max(1, delta["keys"]),
            delta["keys"] / max(elapsed_s, 1e-9),
            delta["dropped_pages"],
            delta["errors"],
            elapsed_s,
            self.stats,
        )
