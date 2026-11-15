"""L3 prefetch admission without per-candidate PP sync (cache mode, timeout policy).

SGLANG_PREFETCH_PP0_MISS_SHORTCUT: PP0's verdict also ends a prefetch whose own
storage query found nothing (the stage MIN can only agree on zero), retiring it
exactly like a drained zero hit. SGLANG_PREFETCH_VERDICT_RELAY: PP0's verdicts
reach later stages on the same loop iteration's request relay instead of the
PP chain. SGLANG_PREFETCH_TIMING: buffered timing events.
"""

import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.cache_controller import (  # noqa: E402
    HiCacheController,
    PrefetchOperation,
)
from sglang.srt.managers.scheduler_pp_mixin import (  # noqa: E402
    PPPrefetchVerdicts,
    SchedulerPPMixin,
)
from sglang.srt.mem_cache import unified_radix_cache as urc  # noqa: E402
from sglang.srt.mem_cache.prefetch_timing import PrefetchTiming  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _cache(pp_rank=0, pp_size=4, relay=False, shortcut=True, deadline=False):
    cache = object.__new__(UnifiedRadixCache)
    cache.pp_rank, cache.pp_size = pp_rank, pp_size
    cache.prefetch_verdict_relay = relay
    cache.pp_verdicts_in, cache.pp_verdicts_out = deque(), []
    cache.prefetch_pp0_miss_shortcut = shortcut
    cache.prefetch_stop_policy = "timeout"
    cache._prefetch_timing = None
    cache.chain = []
    cache._all_reduce = lambda tensor, op: cache.chain.append(int(tensor.item()))
    cache._all_reduce_attn_groups = lambda tensor, op: None
    cache._prefetch_deadline_expired = lambda operation: deadline
    return cache


class _Req:
    """Hashable request handle stand-in (ongoing_prefetch is keyed by handle)."""

    def __init__(self, rid):
        self.rid = rid


def _op(local=None):
    return SimpleNamespace(local_query_tokens=local, local_miss_terminated=False,
                           host_indices=None, deadline_expired=False,
                           is_terminated=lambda: False)


class TestStopVerdict(CustomTestCase):
    def test_pp0_verdict_from_local_query(self):
        for local, deadline, expected in (
            (0, False, urc._STOP_LOCAL_MISS),
            (0, True, urc._STOP_LOCAL_MISS),
            (None, False, urc._STOP_WAIT),  # not queried yet: the default 0 is not evidence
            (None, True, urc._STOP_DEADLINE),
            (512, False, urc._STOP_WAIT),
            (512, True, urc._STOP_DEADLINE),
        ):
            with self.subTest(local=local, deadline=deadline):
                cache = _cache(deadline=deadline)
                self.assertEqual(cache._prefetch_stop_verdict(_op(local)), expected)
                self.assertEqual(cache.chain, [expected])

    def test_later_stage_takes_pp0_verdict_not_its_own(self):
        cache = _cache(pp_rank=2, deadline=True)
        cache._all_reduce = lambda tensor, op: tensor.fill_(urc._STOP_LOCAL_MISS)
        self.assertEqual(cache._prefetch_stop_verdict(_op(512)), urc._STOP_LOCAL_MISS)


class TestVerdictRelay(CustomTestCase):
    def test_verdicts_follow_pp0_in_order_without_chain(self):
        caches = [_cache(pp_rank=r, relay=True) for r in range(4)]
        verdicts = [urc._STOP_WAIT, urc._STOP_LOCAL_MISS, urc._STOP_WAIT]
        for value in verdicts:
            t = torch.tensor(value, dtype=torch.int)
            caches[0]._exchange_pp0_verdict(t)
        for rank in range(1, 4):
            caches[rank].pp_verdicts_in.extend(caches[rank - 1].pp_verdicts_out)
            got = []
            for _ in verdicts:
                t = torch.tensor(-1, dtype=torch.int)
                caches[rank]._exchange_pp0_verdict(t)
                got.append(int(t.item()))
            self.assertEqual(got, verdicts)
        self.assertEqual(caches[3].pp_verdicts_out, [])  # last stage forwards nothing
        self.assertTrue(all(not c.chain for c in caches))

    def test_missing_relay_verdict_fails_loudly(self):
        with self.assertRaises(AssertionError):
            _cache(pp_rank=1, relay=True)._exchange_pp0_verdict(torch.tensor(0))

    def test_unconsumed_verdict_at_next_ingest_fails_loudly(self):
        scheduler = SchedulerPPMixin()
        scheduler.tree_cache = _cache(pp_rank=1, relay=True)
        kept = scheduler._pp_take_relay_extras(["req", PPPrefetchVerdicts([urc._STOP_WAIT])])
        self.assertEqual(kept, ["req"])
        self.assertEqual(list(scheduler.tree_cache.pp_verdicts_in), [urc._STOP_WAIT])
        with self.assertRaises(AssertionError):
            scheduler._pp_take_relay_extras([PPPrefetchVerdicts([urc._STOP_WAIT])])


class TestAdmissionCheck(CustomTestCase):
    def _cache_with(self, verdict, shortcut=True):
        cache = _cache(shortcut=shortcut)
        operation = _op(0)
        request = _Req("r1")
        cache.linker = None
        cache.cache_controller = SimpleNamespace(pp_prefetch_decisions={},
                                                 terminate_prefetch=Mock())
        cache.ongoing_prefetch = {request: SimpleNamespace(operation=operation)}
        cache._prefetch_stop_verdict = Mock(return_value=verdict)
        cache._can_terminate_prefetch = Mock(return_value=False)
        cache._start_storage_prefetch_wait = Mock()
        cache._mark_prefetch_deadline_expired = Mock()
        cache._finish_storage_prefetch = Mock()
        cache.storage_prefetch_retries = Mock()
        cache.revoke_pending_prefetch = Mock()
        return cache, request, operation

    def test_local_miss_retires_like_a_drained_miss(self):
        cache, request, operation = self._cache_with(urc._STOP_LOCAL_MISS)
        self.assertTrue(cache.check_prefetch_progress(request))
        self.assertTrue(operation.local_miss_terminated)
        cache._mark_prefetch_deadline_expired.assert_not_called()
        cache.cache_controller.terminate_prefetch.assert_called_once_with(operation)
        cache.storage_prefetch_retries.poll_miss.assert_called_once_with("r1")
        cache.revoke_pending_prefetch.assert_called_once_with(request)
        cache._finish_storage_prefetch.assert_not_called()

    def test_wait_verdict_keeps_request_waiting(self):
        cache, request, operation = self._cache_with(urc._STOP_WAIT)
        self.assertFalse(cache.check_prefetch_progress(request))
        cache._start_storage_prefetch_wait.assert_called_once_with(operation)
        cache.revoke_pending_prefetch.assert_not_called()

    def test_deadline_verdict_keeps_deadline_semantics(self):
        cache, request, operation = self._cache_with(urc._STOP_DEADLINE)
        self.assertTrue(cache.check_prefetch_progress(request))
        self.assertFalse(operation.local_miss_terminated)
        cache._mark_prefetch_deadline_expired.assert_called_once_with(operation)

    def test_shortcut_off_is_the_stock_path(self):
        cache, request, _ = self._cache_with(urc._STOP_LOCAL_MISS, shortcut=False)
        self.assertFalse(cache.check_prefetch_progress(request))
        cache._prefetch_stop_verdict.assert_not_called()
        cache._can_terminate_prefetch.assert_called_once()


class TestPrefetchThreadLocalResult(CustomTestCase):
    def _run(self, operation, query):
        controller = object.__new__(HiCacheController)
        controller.storage_stop_event = Event()
        controller.storage_stop_event.set()  # drain the queue once, then exit
        controller.prefetch_queue = Queue()
        controller.prefetch_queue.put(operation)
        controller.prefetch_hit_queue = Queue()
        controller.page_size = 64
        controller._on_prefetch_dequeue = lambda op: None
        controller._storage_hit_query = query
        controller._sync_prefetch_hit_query = lambda op, count: count
        controller.prefetch_thread_func()
        return controller.prefetch_hit_queue.get_nowait()

    def test_only_a_completed_query_sets_local_result(self):
        op = PrefetchOperation("r1", list(range(256)))
        self._run(op, lambda o: ([], 0))
        self.assertEqual(op.local_query_tokens, 0)

        failed = PrefetchOperation("r2", list(range(256)))
        def boom(o):
            raise RuntimeError("storage down")
        self._run(failed, boom)
        self.assertIsNone(failed.local_query_tokens)

        cancelled = PrefetchOperation("r3", list(range(256)))
        cancelled.mark_terminate()
        self._run(cancelled, lambda o: (["h"], 64))
        self.assertIsNone(cancelled.local_query_tokens)


class TestPrefetchTiming(CustomTestCase):
    def test_events_flush_as_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            timing = PrefetchTiming(directory, pp_rank=1, flush_interval_s=3600)
            timing.record("check", "r1", dur_s=0.5, terminate=False)
            timing.flush()
            rows = [json.loads(line) for path in Path(directory).glob("*.jsonl")
                    for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["kind"], rows[0]["rid"], rows[0]["pp"]), ("check", "r1", 1))
        self.assertEqual(rows[0]["dur_s"], 0.5)


if __name__ == "__main__":
    unittest.main()
