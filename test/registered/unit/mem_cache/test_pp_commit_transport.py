"""Actual three-process CPU/Gloo coverage; no GPU or SGLang imports required."""

import importlib.util
import multiprocessing
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt/mem_cache"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def worker(rank, path, result):
    core = load("pp_commit")
    transport = load("pp_commit_transport")
    from datetime import timedelta

    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=15),
    )
    control = dist.new_group([0, 1, 2], backend="gloo", timeout=timedelta(seconds=15))
    mailbox = transport.PreviousRoundReports(control)
    state = core.CommitBoundary(rank, 3)
    applied = []
    try:
        for step in range(40):
            if step == (5 if rank == 2 else 0):
                state.stage(
                    core.OperationId(0, "backup", "common-key", 0),
                    ["common-key"],
                    lambda: applied.append(state.round),
                )
            if rank == 0:
                frame = state.leader_frame(mailbox.poll())
                data = mailbox.encode(frame)
            else:
                data = torch.empty(mailbox.FRAME_BYTES, dtype=torch.uint8)
                dist.recv(data, src=rank - 1, tag=310)
                frame = mailbox.decode(data)
            if rank < 2:
                dist.send(data, dst=rank + 1, tag=310)
            state.accept_frame(frame)
            start = time.monotonic()
            # Publishing 100 snapshots exercises bounded coalescing while the
            # network worker owns a separate buffer. No scheduler Work.wait.
            for _ in range(100):
                mailbox.publish(state.ready())
            elapsed = time.monotonic() - start
            if elapsed > 1:
                raise AssertionError("scheduler publish blocked on peer")
            time.sleep(0.01)
        result.put((rank, state.committed, applied, mailbox.stats))
        mailbox.close(timeout=2)
        dist.barrier()
        if not mailbox.close(timeout=2):
            raise AssertionError("control threads did not close")
    finally:
        dist.destroy_process_group(control)
        dist.destroy_process_group()


class TransportTest(unittest.TestCase):
    def test_three_rank_gloo_late_ack_without_scheduler_reverse_wait(self):
        context = multiprocessing.get_context("spawn")
        result = context.Queue()
        with tempfile.TemporaryDirectory() as directory:
            processes = [
                context.Process(target=worker, args=(rank, directory + "/init", result))
                for rank in range(3)
            ]
            for process in processes:
                process.start()
            try:
                rows = [result.get(timeout=30) for _ in processes]
                for process in processes:
                    process.join(timeout=10)
                    self.assertEqual(process.exitcode, 0)
                self.assertEqual([row[1] for row in rows], [1, 1, 1])
                applied = [row[2] for row in rows]
                self.assertTrue(all(len(value) == 1 for value in applied))
                self.assertEqual(len({value[0] for value in applied}), 1)
                self.assertGreaterEqual(applied[0][0], 7)
                self.assertTrue(any(row[3]["coalesced"] > 0 for row in rows))
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(5)

    def test_report_frame_bounds_and_round_trip(self):
        cls = load("pp_commit_transport").PreviousRoundReports
        report = {"wire_seq": 1, "report": {"round": 10, "confirmed": 23}}
        self.assertEqual(cls.decode(cls.encode(report)), report)
        with self.assertRaisesRegex(RuntimeError, "bounded report"):
            cls.encode({"large": "x" * cls.FRAME_BYTES})
        with self.assertRaisesRegex(RuntimeError, "malformed report"):
            cls.decode(torch.zeros(cls.FRAME_BYTES, dtype=torch.uint8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
