"""When stage 1 runs a forward, and over how many tails (padded to a graph bucket)."""

from __future__ import annotations

from typing import Tuple

import msgspec


class TailBatchPolicy(msgspec.Struct, frozen=True):
    min_tails: int
    max_tails: int
    wait_ms: float
    # Tail counts with a captured CUDA graph, ascending.
    buckets: Tuple[int, ...]

    def __post_init__(self) -> None:
        assert 0 < self.min_tails <= self.max_tails, (self.min_tails, self.max_tails)
        assert list(self.buckets) == sorted(set(self.buckets)), self.buckets
        assert self.buckets and self.buckets[-1] >= self.max_tails, (
            f"no graph bucket holds {self.max_tails} tails: {self.buckets}"
        )

    def take_count(self, *, ready: int, oldest_wait_ms: float) -> int:
        """Tails to run now: 0 to wait for more."""
        if ready == 0:
            return 0
        if ready >= self.min_tails or oldest_wait_ms >= self.wait_ms:
            return min(ready, self.max_tails)
        return 0

    def bucket(self, tails: int) -> int:
        for size in self.buckets:
            if size >= tails:
                return size
        raise ValueError(
            f"{tails} tails exceed the largest graph bucket {self.buckets[-1]}"
        )


def parse_buckets(text: str) -> Tuple[int, ...]:
    return tuple(sorted({int(x) for x in text.split(",") if x.strip()}))
