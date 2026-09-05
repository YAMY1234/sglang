"""Low-overhead wall-time/byte accounting for Mooncake Store operations.

Every storage thread records into one process-wide table; the table is dumped
as a single ``MCPROF`` log line at a fixed cadence so the per-op cost of the
L3 path (put, get, exists, layer-wise range get, forward-thread stalls) can be
read off worker logs without a profiler attached.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

_DUMP_INTERVAL_S = 30.0


def sizes_total(buffer_sizes: Sequence[Any]) -> int:
    total = 0
    for item in buffer_sizes:
        if isinstance(item, (list, tuple)):
            total += sum(int(v) for v in item)
        else:
            total += int(item)
    return total


class _OpStat:
    __slots__ = ("count", "seconds", "nbytes", "max_seconds", "items")

    def __init__(self) -> None:
        self.count = 0
        self.seconds = 0.0
        self.nbytes = 0
        self.max_seconds = 0.0
        self.items = 0


class OpProfiler:
    def __init__(self, tag: str = "") -> None:
        self.tag = tag
        self._lock = threading.Lock()
        self._stats: dict[str, _OpStat] = {}
        self._cum: dict[str, _OpStat] = {}
        self._last_dump = time.monotonic()

    def record(
        self, name: str, seconds: float, *, nbytes: int = 0, items: int = 0
    ) -> None:
        with self._lock:
            for table in (self._stats, self._cum):
                stat = table.get(name)
                if stat is None:
                    stat = table[name] = _OpStat()
                stat.count += 1
                stat.seconds += seconds
                stat.nbytes += nbytes
                stat.items += items
                if seconds > stat.max_seconds:
                    stat.max_seconds = seconds
            now = time.monotonic()
            if now - self._last_dump < _DUMP_INTERVAL_S:
                return
            window = now - self._last_dump
            self._last_dump = now
            snapshot = self._stats
            self._stats = {}
        self._dump(snapshot, window)

    def _dump(self, snapshot: dict[str, _OpStat], window: float) -> None:
        parts = []
        for name in sorted(snapshot):
            s = snapshot[name]
            avg_ms = 1000.0 * s.seconds / s.count if s.count else 0.0
            gbps = s.nbytes / s.seconds / 1e9 if s.seconds > 0 else 0.0
            parts.append(
                f"{name}[n={s.count} items={s.items} sec={s.seconds:.3f} "
                f"avg_ms={avg_ms:.2f} max_ms={1000.0 * s.max_seconds:.1f} "
                f"bytes={s.nbytes} GBps={gbps:.2f}]"
            )
        logger.info("MCPROF %s window=%.1fs %s", self.tag, window, " ".join(parts))


_PROFILER = OpProfiler()


def get_profiler() -> OpProfiler:
    return _PROFILER


def set_tag(tag: str) -> None:
    _PROFILER.tag = tag


class timed:
    """``with timed("mc.put", nbytes=...)`` records the block's wall time."""

    __slots__ = ("name", "nbytes", "items", "_start")

    def __init__(self, name: str, *, nbytes: int = 0, items: int = 0) -> None:
        self.name = name
        self.nbytes = nbytes
        self.items = items
        self._start = 0.0

    def __enter__(self) -> "timed":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        _PROFILER.record(
            self.name,
            time.perf_counter() - self._start,
            nbytes=self.nbytes,
            items=self.items,
        )
