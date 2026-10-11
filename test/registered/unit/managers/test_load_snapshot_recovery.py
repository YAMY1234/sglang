"""CPU regression for live load-slot unlink and replacement (TP/PD transport)."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.srt.managers.load_snapshot import (
    LoadSnapshot,
    ShmLoadSnapshotReader,
    ShmLoadSnapshotWriter,
)


class TestLoadSnapshotRecovery(unittest.TestCase):
    def test_independent_reader_exit_does_not_unlink_or_register_slot(self):
        # Independent interpreter: a SharedMemory(create=False) attachment
        # would register its name and risk unlink on tracker shutdown in 3.12.
        module = sys.modules[ShmLoadSnapshotReader.__module__]
        source = getattr(module, "__cpu_source__", None)
        if source is None:
            source = Path(module.__file__).read_text()
        child = """
import ast, json, sys, types
from multiprocessing import resource_tracker
payload = json.load(sys.stdin)
def reject_registration(name, kind):
    if kind == "shared_memory":
        raise AssertionError("load reader must not register SHM with resource_tracker")
resource_tracker.register = reject_registration
tree = ast.parse(payload["source"])
tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and (n.module or "").startswith("sglang."))]
module = types.ModuleType("isolated_snapshot")
sys.modules[module.__name__] = module
exec(compile(tree, "load_snapshot.py", "exec"), module.__dict__)
reader = module.ShmLoadSnapshotReader(payload["path"], 1)
print(json.dumps(reader.read(0).to_dict({"core"})))
# Exit normally without explicitly closing the attachment.
"""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "loads.shm")
            writer = ShmLoadSnapshotWriter(path, dp_size=1, dp_rank=0)
            try:
                writer.write(
                    LoadSnapshot(dp_rank=0, num_total_tokens=123, num_running_reqs=7)
                )
                completed = subprocess.run(
                    [sys.executable, "-c", child],
                    input=json.dumps({"source": source, "path": path}),
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=20,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertNotIn("resource_tracker", completed.stderr)
                self.assertEqual(json.loads(completed.stdout)["num_total_tokens"], 123)
                self.assertTrue(os.path.exists(path))
                self.assertEqual(os.stat(path).st_nlink, 1)
                reader = ShmLoadSnapshotReader(path, dp_size=1)
                try:
                    self.assertEqual(reader.read(0).num_running_reqs, 7)
                finally:
                    reader.close()
            finally:
                writer.close()

    def test_reader_attaching_after_unlink_recovers_actual_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "loads.shm")
            writer = ShmLoadSnapshotWriter(path, dp_size=1, dp_rank=0)
            reader = ShmLoadSnapshotReader(path, dp_size=1)
            reader.close()  # tokenizer has not yet attached during model loading
            try:
                os.unlink(path)
                self.assertEqual(reader.read_all(), [])
                writer.write(
                    LoadSnapshot(dp_rank=0, num_total_tokens=123, num_running_reqs=7)
                )
                loads = reader.read_all()
                self.assertEqual(len(loads), 1)
                self.assertEqual(loads[0].num_total_tokens, 123)
                self.assertEqual(loads[0].num_running_reqs, 7)
            finally:
                reader.close()
                writer.close()

    def test_attached_reader_follows_replacement_and_idle_is_nonempty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "loads.shm")
            writer = ShmLoadSnapshotWriter(path, dp_size=1, dp_rank=0)
            reader = ShmLoadSnapshotReader(path, dp_size=1)
            try:
                writer.write(
                    LoadSnapshot(dp_rank=0, num_total_tokens=99, num_running_reqs=3)
                )
                self.assertEqual(reader.read(0).num_total_tokens, 99)
                for tokens, running in [(24, 2), (0, 0), (18, 1)]:
                    os.unlink(path)
                    writer.write(
                        LoadSnapshot(
                            dp_rank=0, num_total_tokens=tokens, num_running_reqs=running
                        )
                    )
                    self.assertTrue(
                        os.path.exists(path), "writer must restore the published path"
                    )
                    self.assertEqual(reader.read(0).num_total_tokens, tokens)
                    self.assertEqual(reader.read(0).num_running_reqs, running)
                    self.assertEqual(len(reader.read_all()), 1)
            finally:
                reader.close()
                writer.close()

    def test_dp_writers_repair_without_erasing_other_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "loads.shm")
            writers = [
                ShmLoadSnapshotWriter(path, dp_size=2, dp_rank=i) for i in range(2)
            ]
            reader = ShmLoadSnapshotReader(path, dp_size=2)
            try:
                os.unlink(path)
                for i, writer in enumerate(writers):
                    writer.write(
                        LoadSnapshot(dp_rank=i, num_total_tokens=(i + 1) * 100)
                    )
                fresh = ShmLoadSnapshotReader(path, dp_size=2)
                try:
                    self.assertEqual(
                        [s.num_total_tokens for s in fresh.read_all()], [100, 200]
                    )
                    self.assertEqual(
                        [s.num_total_tokens for s in reader.read_all()], [100, 200]
                    )
                finally:
                    fresh.close()
            finally:
                reader.close()
                for writer in writers:
                    writer.close()


if __name__ == "__main__":
    unittest.main()
