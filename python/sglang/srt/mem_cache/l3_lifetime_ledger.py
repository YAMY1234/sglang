"""Opt-in, keyed L3 lifetime evidence. Disabled calls do no tree or file work.

The scheduler thread only stamps each event and hands it to a writer thread, which builds and
serializes the rows (JSON on the scheduler thread cost the PP0 beat about 5-10 ms). With
SGLANG_HICACHE_L3_LIFETIME_LEDGER_KEYS=ends, key lists keep their first and last key and a count."""

from __future__ import annotations

import atexit
import json
import os
import queue
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
        self._seq = 0
        self._file = (directory / f"source-{self.host}-pp{pp_rank}-tp{tp_rank}-{self.pid}.jsonl").open(
            "a", buffering=1024 * 1024
        )
        self._key_ends = os.environ.get("SGLANG_HICACHE_L3_LIFETIME_LEDGER_KEYS", "all") == "ends"
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._writer = threading.Thread(target=self._write_loop, name="l3-ledger-writer", daemon=True)
        self._writer.start()
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
        # Callers pass fresh objects (copied key lists, scalars): the writer reads them later.
        self._queue.put((time.time_ns(), time.monotonic_ns(), event, fields))

    def drain_tree(self, tree, **context):
        events = tree.take_l3_lifetime_events()
        if events:
            self._queue.put((time.time_ns(), time.monotonic_ns(), None, (events, context)))

    def backlog(self) -> int:
        """Queued items the writer has not taken yet (a tree drain counts as one)."""
        return self._queue.qsize()

    def _write_loop(self):
        last_flush = time.monotonic()
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                item = ()
            if item is None:
                break
            if item:
                wall, mono, event, fields = item
                if event is not None:
                    self._write(wall, mono, event, fields)
                else:
                    events, context = fields
                    for wall_ns, ev, node_id, component, tier, keys, depth, reason in events:
                        self._write(wall, mono, ev, dict(wall_ns=wall_ns, node_id=node_id, component=component,
                                                         tier=tier, keys=keys, depth=depth, reason=reason, **context))
            now = time.monotonic()
            if now - last_flush >= 1.0:
                self._file.flush()
                last_flush = now
        self._file.flush()
        self._file.close()

    def _write(self, wall, mono, event, fields):
        self._seq += 1
        row = dict(schema=1, seq=self._seq, wall_ns=wall, mono_ns=mono, event=event, host=self.host, pid=self.pid,
                   pp_rank=self.pp_rank, tp_rank=self.tp_rank)
        row.update(fields)
        if self._key_ends:
            for k, v in list(row.items()):
                if k.endswith("keys") and isinstance(v, list) and len(v) > 2:
                    row[k] = [v[0], v[-1]]
                    row[k[:-4] + "key_count"] = len(v)
        self._file.write(json.dumps(row, separators=(",", ":")) + "\n")

    def close(self):
        if self._writer.is_alive():
            self._queue.put(None)
            self._writer.join(timeout=60)
