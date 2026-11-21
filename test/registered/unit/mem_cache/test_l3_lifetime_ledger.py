"""L3 lifetime ledger: rows are built and written on the writer thread, in emit order."""

import glob
import json
import os
import tempfile
import unittest
from unittest import mock

from sglang.srt.mem_cache.l3_lifetime_ledger import L3LifetimeLedger
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Tree:
    def __init__(self, events):
        self.events = events

    def take_l3_lifetime_events(self):
        events, self.events = self.events, []
        return events


class TestL3LifetimeLedger(unittest.TestCase):
    def _rows(self, keys_mode):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(
            os.environ, {"SGLANG_HICACHE_L3_LIFETIME_LEDGER_KEYS": keys_mode}
        ):
            ledger = L3LifetimeLedger(d, pp_rank=0)
            ledger.drain_tree(
                _Tree([(11, "evict", 5, 2, "host", ["a", "b", "c"], 192, "pathcap")]),
                window=1,
            )
            ledger.emit("storage_write_issue", kv_keys=["x", "y", "z", "w"], op_id=7)
            ledger.close()
            return [json.loads(line) for f in glob.glob(f"{d}/*.jsonl") for line in open(f)]

    def test_rows_in_emit_order_with_event_fields(self):
        rows = self._rows("all")
        self.assertEqual([r["event"] for r in rows], ["ledger_open", "evict", "storage_write_issue"])
        self.assertEqual([r["seq"] for r in rows], [1, 2, 3])
        evict = rows[1]
        self.assertEqual((evict["wall_ns"], evict["node_id"], evict["keys"], evict["window"]), (11, 5, ["a", "b", "c"], 1))
        self.assertEqual(rows[2]["kv_keys"], ["x", "y", "z", "w"])

    def test_key_ends_keeps_first_last_and_count(self):
        rows = self._rows("ends")
        self.assertEqual((rows[1]["keys"], rows[1]["key_count"]), (["a", "c"], 3))
        self.assertEqual((rows[2]["kv_keys"], rows[2]["kv_key_count"]), (["x", "w"], 4))


if __name__ == "__main__":
    unittest.main()
