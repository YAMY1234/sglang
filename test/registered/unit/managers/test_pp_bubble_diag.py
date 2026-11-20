"""PP bubble diag: the wall partition is exhaustive and exclusive, pass and
forward counters bin correctly, and the module is inert when the flag is off."""

import io
import json
import time
import unittest
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.managers import pp_bubble_diag
from sglang.srt.managers.pp_bubble_diag import PPBubbleDiag


class _FakeEvent:
    """Timing event on a virtual clock: record() stamps the clock's value."""

    def __init__(self, clock, enable_timing=True):
        self.clock = clock
        self.t = None

    def record(self, stream=None):
        self.t = self.clock["now"]

    def query(self):
        return self.t is not None and self.t <= self.clock["done_upto"]

    def elapsed_time(self, other):
        return other.t - self.t


def _diag(chunk_budget=16384, fwd_file=None):
    clock = {"now": 0.0, "done_upto": 1e18}
    d = PPBubbleDiag(
        pp_rank=3,
        event_factory=lambda enable_timing: _FakeEvent(clock, enable_timing),
        interval_s=60.0,
        chunk_budget=chunk_budget,
        fwd_file=fwd_file,
    )
    return d, clock


def _forward(d, clock, pre, start, end, new=16384, cached=0, has_load=False):
    clock["now"] = pre
    e_pre = d.forward_event()
    clock["now"] = start
    e_start = d.forward_event()
    clock["now"] = end
    d.on_forward(
        pre=e_pre,
        start=e_start,
        end=d.forward_event(),
        new_tokens=new,
        cached_tokens=cached,
        has_load=has_load,
    )


class TestWallPartition(unittest.TestCase):
    def test_partition_sums_to_wall_and_nesting_is_exclusive(self):
        d, _ = _diag()
        t0 = time.perf_counter_ns()
        d.phase("build")
        time.sleep(0.01)
        with d.span("relay._can_terminate_prefetch"):
            time.sleep(0.02)
            with d.span("drain"):
                time.sleep(0.01)
        d.phase("launch")
        time.sleep(0.005)
        stats = d.snapshot(1.0)
        ms = stats["ms"]
        self.assertAlmostEqual(
            stats["wall_ms"], (time.perf_counter_ns() - t0) / 1e6, delta=2.0
        )
        self.assertAlmostEqual(sum(ms.values()), stats["wall_ms"], delta=0.05)
        # Inner wait pauses the outer one: 20 ms outer, 10 ms inner.
        self.assertAlmostEqual(ms["wait.relay._can_terminate_prefetch"], 20, delta=5)
        self.assertAlmostEqual(ms["wait.drain"], 10, delta=5)
        self.assertGreaterEqual(ms["host.build"], 9)
        self.assertGreaterEqual(ms["host.launch"], 4)
        self.assertEqual(stats["calls"]["wait.drain"], 1)

    def test_phase_inside_wait_does_not_steal_wait_time(self):
        d, _ = _diag()
        with d.span("d2h"):
            d.phase("release")
            time.sleep(0.005)
        ms = d.snapshot(1.0)["ms"]
        self.assertGreaterEqual(ms["wait.d2h"], 4)
        self.assertNotIn("host.release", ms)

    def test_span_in_phase_labels_by_current_phase(self):
        d, _ = _diag()
        d.phase("ingest")
        with d.span_in_phase("send_commit"):
            pass
        self.assertIn("wait.send_commit.ingest", d.snapshot(1.0)["calls"])


class TestPasses(unittest.TestCase):
    def test_vote_relay_is_binned_by_queue_length(self):
        d, _ = _diag()
        d.begin_pass(70)
        for _ in range(3):
            with d.span("relay." + pp_bubble_diag.VOTE_CALLER):
                time.sleep(0.002)
        d.note_skip("prefetch_running")
        d.note_skip("prefetch_running")
        d.end_pass(admitted_tokens=9000)
        d.begin_pass(0)
        d.end_pass(admitted_tokens=0)
        s = d.snapshot(1.0)
        self.assertEqual(s["bin_passes"], [1, 0, 0, 0, 1])
        self.assertEqual(s["bin_vote_calls"], [0, 0, 0, 0, 3])
        self.assertGreater(s["bin_vote_ms"][4], 5)
        self.assertEqual(s["bin_vote_ms"][0], 0)
        self.assertEqual(s["skips"], {"prefetch_running": 2})
        self.assertEqual(s["skip_passes"], 1)
        self.assertEqual(s["queued_passes"], 1)
        self.assertEqual(s["under_budget_queued"], 1)
        self.assertEqual(s["lost_tokens_skip_passes"], 16384 - 9000)
        self.assertEqual(s["queue_max"], 70)

    def test_full_pass_without_skips_loses_nothing(self):
        d, _ = _diag()
        d.begin_pass(5)
        d.end_pass(admitted_tokens=16384)
        s = d.snapshot(1.0)
        self.assertEqual(s["under_budget_queued"], 0)
        self.assertEqual(s["lost_tokens_skip_passes"], 0)
        self.assertEqual(s["bin_passes"], [0, 1, 0, 0, 0])


class TestForwardGaps(unittest.TestCase):
    def test_gap_split_and_load_buckets(self):
        buf = io.StringIO()
        d, clock = _diag(fwd_file=buf)
        d.begin_pass(4)
        _forward(d, clock, pre=0, start=0, end=200)
        # CPU enqueued 30 ms after the last forward ended, then waited 10 ms on upstream.
        _forward(d, clock, pre=230, start=240, end=440, cached=50000, has_load=True)
        d.begin_pass(0)
        _forward(d, clock, pre=440, start=445, end=545, new=8000)
        s = d.snapshot(1.0)
        self.assertEqual(s["fwd_n"], 3)
        self.assertAlmostEqual(s["fwd_ms"], 500)
        self.assertEqual(s["gap_n"], 2)
        self.assertAlmostEqual(s["gap_cpu_ms"], 30)
        self.assertAlmostEqual(s["gap_dep_ms"], 15)
        self.assertEqual(s["qgap_n"], 1)
        self.assertAlmostEqual(s["qgap_cpu_ms"], 30)
        self.assertEqual((s["full_load_n"], s["full_noload_n"]), (1, 1))
        self.assertAlmostEqual(s["full_load_ms"], 200)
        self.assertEqual(s["full_load_cached"], 50000)
        rows = buf.getvalue().strip().splitlines()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1].split(",")[1:], ["200.000", "30.000", "10.000", "16384", "50000", "1", "4"])
        self.assertEqual(rows[0].split(",")[2], "-1.000")

    def test_unfinished_forward_is_read_later(self):
        d, clock = _diag()
        clock["done_upto"] = 100
        _forward(d, clock, pre=0, start=0, end=200)
        self.assertEqual(d.fwd_n, 0)
        clock["done_upto"] = 1e18
        d._drain_events()
        self.assertEqual(d.fwd_n, 1)


class TestIntervalLog(unittest.TestCase):
    def test_logs_once_per_interval_and_resets(self):
        d, _ = _diag()
        d.begin_pass(3)
        d.end_pass(admitted_tokens=16384)
        with self.assertLogs(pp_bubble_diag.logger, level="INFO") as cm:
            self.assertIsNone(d.maybe_log(now=d._logged_at + 10))
            stats = d.maybe_log(now=d._logged_at + 61)
        self.assertEqual(stats["passes"], 1)
        line = cm.output[-1].split("PP bubble diag: ", 1)[1]
        self.assertEqual(json.loads(line)["pp_rank"], 3)
        self.assertEqual(d.passes, 0)


class _FakeCache:
    """Mimics the URC call shape: caller -> _all_reduce -> _pp_sync -> relay_wait."""

    def _pp_sync(self):
        with pp_bubble_diag.relay_wait():
            pass

    def _all_reduce(self):
        self._pp_sync()

    def agree_on_all_ranks(self):
        self._all_reduce()

    def _can_terminate_prefetch(self):
        self._all_reduce()

    def _maybe_flush_due(self):
        self.agree_on_all_ranks()


class TestModuleHooks(unittest.TestCase):
    def tearDown(self):
        pp_bubble_diag._diag = None

    def test_off_by_default_is_inert(self):
        with envs.SGLANG_PP_BUBBLE_DIAG.override(False):
            self.assertIsNone(pp_bubble_diag.init(0, None, 16384))
        self.assertIsNone(pp_bubble_diag.get())
        self.assertIs(pp_bubble_diag.wait("x"), pp_bubble_diag._NULL)
        self.assertIs(pp_bubble_diag.relay_wait(), pp_bubble_diag._NULL)
        _FakeCache()._can_terminate_prefetch()

    def test_relay_wait_labels_the_all_reduce_caller(self):
        with envs.SGLANG_PP_BUBBLE_DIAG.override(True):
            d = pp_bubble_diag.init(2, None, 16384)
        cache = _FakeCache()
        cache._can_terminate_prefetch()
        cache._maybe_flush_due()
        calls = d.snapshot(1.0)["calls"]
        self.assertEqual(calls["wait.relay._can_terminate_prefetch"], 1)
        self.assertEqual(calls["wait.relay._maybe_flush_due"], 1)

    def test_fwd_dir_writes_per_rank_csv(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp, envs.SGLANG_PP_BUBBLE_DIAG.override(
            True
        ), envs.SGLANG_PP_BUBBLE_DIAG_FWD_DIR.override(tmp):
            d = pp_bubble_diag.init(5, None, 16384)
            d._fwd_file.flush()
            import os

            (name,) = os.listdir(tmp)
            self.assertTrue(name.startswith("fwd-pp5-"))
            with open(os.path.join(tmp, name)) as f:
                self.assertEqual(f.read(), pp_bubble_diag.FWD_CSV_HEADER)
            d._fwd_file.close()


if __name__ == "__main__":
    unittest.main()
