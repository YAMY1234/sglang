"""py-spy crash dumps must end within two bounded attempts, whatever the number of ranks."""

import subprocess
import threading
import time
import unittest
from unittest import mock

from sglang.srt.utils import cudacore_pyspy_dump_utils as dump_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _procs(*pids):
    return [mock.Mock(pid=pid) for pid in pids]


class TestPyspyDumpBounded(CustomTestCase):
    def _dump(self, run, pids=(11,)):
        with (
            mock.patch.object(
                dump_utils, "collect_scheduler_processes", return_value=_procs(*pids)
            ),
            mock.patch.object(dump_utils.subprocess, "run", side_effect=run) as m,
        ):
            dump_utils.pyspy_dump_schedulers(scheduler_only=True)
        return m

    def test_every_attempt_has_a_timeout(self):
        m = self._dump(lambda cmd, **kw: mock.Mock(stdout="trace"))
        self.assertEqual(m.call_count, 1)
        self.assertEqual(m.call_args.kwargs["timeout"], dump_utils.PYSPY_DUMP_TIMEOUT_S)
        self.assertEqual(
            m.call_args.args[0], ["py-spy", "dump", "--native", "--pid", "11"]
        )

    def test_native_timeout_falls_back_to_plain_dump(self):
        def run(cmd, **kw):
            if "--native" in cmd:
                raise subprocess.TimeoutExpired(cmd, kw["timeout"])
            return mock.Mock(stdout="trace")

        m = self._dump(run)
        self.assertEqual(
            [c.args[0] for c in m.call_args_list],
            [
                ["py-spy", "dump", "--native", "--pid", "11"],
                ["py-spy", "dump", "--pid", "11"],
            ],
        )

    def test_missing_py_spy_is_not_retried(self):
        def run(cmd, **kw):
            raise FileNotFoundError("py-spy")

        self.assertEqual(self._dump(run).call_count, 1)

    def test_ranks_are_dumped_concurrently(self):
        in_flight, peak, lock = [0], [0], threading.Lock()

        def run(cmd, **kw):
            with lock:
                in_flight[0] += 1
                peak[0] = max(peak[0], in_flight[0])
            time.sleep(0.2)
            with lock:
                in_flight[0] -= 1
            raise subprocess.TimeoutExpired(cmd, kw["timeout"])

        start = time.monotonic()
        m = self._dump(run, pids=(11, 12, 13, 14))
        self.assertEqual(m.call_count, 8)
        self.assertEqual(peak[0], 4)
        # Two attempts in sequence per rank, ranks in parallel: ~0.4 s, not ~1.6 s.
        self.assertLess(time.monotonic() - start, 1.2)


if __name__ == "__main__":
    unittest.main()
