"""Opt-in timing events for the L3 prefetch path (SGLANG_PREFETCH_TIMING).

Callers record around existing operations only; nothing here adds an RPC, a
collective, a device sync or a poll. Events are buffered and written as JSONL
by a daemon thread, one file per process.
"""

import json
import os
import threading
import time
from collections import deque
from typing import Callable, Optional, Union

from sglang.srt.environ import envs


class PrefetchTiming:
    def __init__(self, directory: str, pp_rank: int, flush_interval_s: float = 2.0):
        self._events: deque = deque()
        self._path = os.path.join(
            directory, f"prefetch-timing-pp{pp_rank}-pid{os.getpid()}.jsonl"
        )
        self._pp_rank = pp_rank
        self._interval = flush_interval_s
        thread = threading.Thread(target=self._flush_loop, daemon=True)
        thread.start()

    def record(self, kind: str, rid: Optional[str], **fields) -> None:
        # deque.append is atomic; callers run on scheduler, prefetch and sync threads.
        self._events.append((time.monotonic_ns(), time.time_ns(), kind, rid, fields))

    def _flush_loop(self) -> None:
        while True:
            time.sleep(self._interval)
            self.flush()

    def flush(self) -> None:
        lines = []
        while self._events:
            mono, wall, kind, rid, fields = self._events.popleft()
            lines.append(json.dumps(dict(mono_ns=mono, wall_ns=wall, pp=self._pp_rank,
                                         kind=kind, rid=rid, **fields)))
        if lines:
            with open(self._path, "a") as stream:
                stream.write("\n".join(lines) + "\n")


_timing: Optional[PrefetchTiming] = None


def prefetch_timing(pp_rank: Union[int, Callable[[], int]]) -> Optional[PrefetchTiming]:
    """The process recorder when SGLANG_PREFETCH_TIMING is set, else None.
    pp_rank may be a callable, evaluated only when the recorder is created."""
    global _timing
    if _timing is None and envs.SGLANG_PREFETCH_TIMING.get():
        rank = pp_rank() if callable(pp_rank) else pp_rank
        _timing = PrefetchTiming(envs.SGLANG_PREFETCH_TIMING_DIR.get(), rank)
    return _timing
