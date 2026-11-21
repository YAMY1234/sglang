"""Opt-in, keyed L3 lifetime evidence. Disabled calls do no tree or file work."""

from __future__ import annotations

import atexit
import json
import os
import socket
import threading
import time
from pathlib import Path


class L3LifetimeLedger:
    def __init__(self, directory: str, pp_rank: int, tp_rank: int = 0):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.pp_rank, self.tp_rank = pp_rank, tp_rank
        self.host, self.pid = socket.gethostname(), os.getpid()
        self._lock = threading.Lock()
        self._seq = 0
        self._file = (directory / f"source-{self.host}-pp{pp_rank}-tp{tp_rank}-{self.pid}.jsonl").open(
            "a", buffering=1024 * 1024
        )
        self._last_flush = time.monotonic()
        atexit.register(self.close)
        self.emit("ledger_open", schema=1)

    @classmethod
    def from_env(cls, pp_rank: int, tp_rank: int = 0):
        if pp_rank != 0 or os.environ.get("SGLANG_HICACHE_L3_LIFETIME_LEDGER", "0") != "1":
            return None
        directory = os.environ.get("SGLANG_HICACHE_L3_LIFETIME_LEDGER_DIR")
        if not directory:
            raise ValueError("L3 lifetime ledger requires SGLANG_HICACHE_L3_LIFETIME_LEDGER_DIR")
        return cls(directory, pp_rank, tp_rank)

    def emit(self, event: str, **fields):
        with self._lock:
            if self._file is None:
                return
            self._seq += 1
            row = dict(schema=1, seq=self._seq, wall_ns=time.time_ns(), mono_ns=time.monotonic_ns(),
                       event=event, host=self.host, pid=self.pid,
                       pp_rank=self.pp_rank, tp_rank=self.tp_rank)
            row.update(fields)
            self._file.write(json.dumps(row, separators=(",", ":")) + "\n")
            now = time.monotonic()
            if now - self._last_flush >= 1.0:
                self._file.flush()
                self._last_flush = now

    def drain_tree(self, tree, **context):
        for wall_ns, event, node_id, component, tier, keys, depth, reason in tree.take_l3_lifetime_events():
            self.emit(event, wall_ns=wall_ns, node_id=node_id, component=component,
                      tier=tier, keys=keys, depth=depth, reason=reason, **context)

    def close(self):
        with self._lock:
            if self._file is not None:
                self._file.flush()
                self._file.close()
                self._file = None
