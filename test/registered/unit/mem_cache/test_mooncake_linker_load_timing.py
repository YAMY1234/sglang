"""The Mooncake linker logs each request's load time and bytes on the timing rank."""

import os
import queue
from concurrent.futures import Future, ThreadPoolExecutor
from collections import Counter, deque
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolGroup
from sglang.srt.mem_cache.unified_cache.components.base import LinkerTransferPhase
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.load_slot_reserve import LoadSlotReserve
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.storage.mooncake_store import mooncake_direct_linker
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LOGGER = "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker"
# Bytes one key moves per layer, by pool.
KEY_BYTES = {PoolName.KV: 100, PoolName.SWA: 10}


class _Pool:
    def __init__(self, name):
        self.name = name

    def prepare_locations(self, host_indices):
        return list(host_indices)

    def get_prepared_layer_range_meta(self, locations, layer):
        size = KEY_BYTES[self.name]
        return [[0]] * len(locations), [[size]] * len(locations), [[0]] * len(locations)


def _transfer(name, pages):
    return SimpleNamespace(
        name=name,
        keys=[f"{name}-{i}" for i in range(pages)],
        host_indices=range(pages),
        hit_policy=PoolHitPolicy.ALL_PAGES,
    )


def _linker(log_load_timing):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.log_load_timing = log_load_timing
    linker.num_layers = 2
    linker.pools = {name: _Pool(name) for name in KEY_BYTES}
    linker.layer_done_counter = MagicMock()
    linker.pinned, linker.load_pins = {}, {}
    linker.pinned_others, linker.pin_expiry = {}, deque()
    linker.read_retries = Counter()
    store = MagicMock()
    store.batch_get_session_start.side_effect = lambda keys: [0] * len(keys)
    store.batch_get_into_multi_buffer_ranges.side_effect = (
        lambda keys, ptrs, sizes, offsets: [sum(item) for item in sizes]
    )
    linker.storage = SimpleNamespace(
        store=store,
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, 1),
        _tag_keys=lambda keys: keys,
        recently_loaded=mooncake_direct_linker._RecentKeys(16),
    )
    return linker


class TestMooncakeLinkerLoadTiming(CustomTestCase):
    def test_each_request_logs_its_own_bytes_after_all_layers_land(self):
        linker = _linker(log_load_timing=True)
        requests = [
            [_transfer(PoolName.KV, 2), _transfer(PoolName.SWA, 1)],
            [_transfer(PoolName.KV, 3)],
        ]
        with self.assertLogs(LOGGER, level="INFO") as logs:
            linker.load_layer_wise(7, requests, [("a", 0.0), ("b", None)])
        lines = [line for line in logs.output if "linker load" in line]
        self.assertEqual(len(lines), 2)
        # Two layers: (2 x 100 + 1 x 10) x 2 and 3 x 100 x 2.
        self.assertIn("rid=a pages=2 bytes=420 ", lines[0])
        self.assertIn("rid=b pages=3 bytes=600 ", lines[1])
        self.assertIn("batch_reqs=2", lines[0])
        self.assertIn("wait=0ms", lines[1])  # Unknown queue time counts from start.
        linker.layer_done_counter.complete.assert_any_call(7, 1)
        linker.layer_done_counter.fail.assert_not_called()

    def test_disabled_rank_and_failed_loads_log_nothing(self):
        for log_load_timing, fail in [(False, False), (True, True)]:
            with self.subTest(log_load_timing=log_load_timing, fail=fail):
                linker = _linker(log_load_timing)
                if fail:
                    linker.storage.store.batch_get_into_multi_buffer_ranges.side_effect = (
                        lambda keys, ptrs, sizes, offsets: [0] * len(keys)
                    )
                with patch.object(mooncake_direct_linker.logger, "info") as info:
                    with patch.object(mooncake_direct_linker.logger, "exception"):
                        linker.load_layer_wise(
                            1, [[_transfer(PoolName.KV, 1)]], [("a", 0.0)]
                        )
                info.assert_not_called()
                self.assertEqual(linker.layer_done_counter.fail.called, fail)

    def test_sessions_pinned_at_admission_are_not_started_again(self):
        linker = _linker(log_load_timing=False)
        a, b = _transfer(PoolName.KV, 2), _transfer(PoolName.SWA, 1)
        linker.load_layer_wise(
            7, [[a], [b]], [("a", 0.0), ("b", 0.0)], pinned=frozenset({"a"})
        )
        store = linker.storage.store
        # Only b's object is started; a's sessions (pinned) end with b's.
        store.batch_get_session_start.assert_called_once_with(b.keys)
        ended = [c.args[0] for c in store.batch_get_session_end.call_args_list]
        self.assertEqual(sorted(sum(ended, [])), sorted(a.keys + b.keys))
        linker.layer_done_counter.fail.assert_not_called()

    def test_later_stages_pins_outlive_the_launch_until_the_grace_ends(self):
        linker = _linker(log_load_timing=False)
        linker.pool_group = SimpleNamespace(
            resolve_transfers=lambda transfers, **kwargs: transfers
        )
        linker.pending_loads, linker.load_queued_at = {}, {}
        linker.mla_broadcaster, linker.gc_frozen = None, True
        linker.load_queue, linker.broadcast_loads = queue.Queue(), {}
        linker.stats = {"load": 0}
        linker.layer_done_counter.update_producer.return_value = 3
        linker.pinned_others = {"r": ["later-stage-key"]}
        self.assertTrue(linker.load("r", [_transfer(PoolName.KV, 1)]))
        with patch.object(mooncake_direct_linker, "device_module", MagicMock()):
            linker.start_layer_wise_loading()
        store = linker.storage.store
        # Every later stage starts its own load after PP0's launch: held on.
        store.batch_get_session_end.assert_not_called()
        self.assertEqual(linker.pinned_others, {})
        linker._end_expired_pins(time.monotonic() + linker._PIN_GRACE_S + 1)
        store.batch_get_session_end.assert_called_once_with(["later-stage-key"])
        self.assertEqual(len(linker.pin_expiry), 0)

    def test_launched_loads_keep_the_later_stages_pins_renewed_until_the_grace_ends(self):
        # Rhea pm-lk1: a later stage started its own load after the store lease
        # (about 10 s) and found PP0's pinned objects evicted (-704).
        linker = _linker(log_load_timing=False)
        linker.pool_group = SimpleNamespace(
            resolve_transfers=lambda transfers, **kwargs: transfers
        )
        linker.pending_loads, linker.load_queued_at = {}, {}
        linker.mla_broadcaster, linker.gc_frozen = None, True
        linker.load_queue, linker.broadcast_loads = queue.Queue(), {}
        linker.stats = {"load": 0}
        linker.layer_done_counter.update_producer.return_value = 3
        linker.pinned_others = {"r": ["later-stage-key"]}
        linker.prepin_renew_s = 2.0
        linker.prepin_pool = ThreadPoolExecutor(1)
        self.assertTrue(linker.load("r", [_transfer(PoolName.KV, 1)]))
        with patch.object(mooncake_direct_linker, "device_module", MagicMock()):
            linker.start_layer_wise_loading()
        store = linker.storage.store
        t0 = linker.pin_expiry[0][2]
        renewals = lambda: [c.args[0] for c in store.batch_get_session_start.call_args_list
                            if c.args[0] == ["later-stage-key"]]
        linker._renew_held_pins(t0 + 1.0)  # not due yet
        linker.prepin_pool.submit(lambda: None).result(timeout=5)
        self.assertEqual(renewals(), [])
        linker._renew_held_pins(t0 + 2.5)
        linker._renew_held_pins(t0 + 3.0)  # renewed at 2.5: not due again
        linker.prepin_pool.submit(lambda: None).result(timeout=5)
        self.assertEqual(renewals(), [["later-stage-key"]])
        self.assertEqual(linker.pin_renewals, 1)
        linker._end_expired_pins(t0 + linker._PIN_GRACE_S + 1)
        linker._renew_held_pins(t0 + linker._PIN_GRACE_S + 5)  # ended: nothing left
        linker.prepin_pool.submit(lambda: None).result(timeout=5)
        self.assertEqual(len(renewals()), 1)
        store.batch_get_session_end.assert_called_with(["later-stage-key"])

    def _read_once_with(self, first_codes):
        """A two-object KV load whose first range read returns ``first_codes``
        (per object; None = read fine) and whose retry reads fine."""
        linker = _linker(log_load_timing=False)
        store = linker.storage.store
        calls = []

        def ranges(keys, ptrs, sizes, offsets):
            good = [sum(item) for item in sizes]
            calls.append(list(keys))
            if len(calls) == 1:
                return [good[i] if c is None else c for i, c in enumerate(first_codes)]
            return good

        store.batch_get_into_multi_buffer_ranges.side_effect = ranges
        linker.num_layers = 1
        with patch.object(mooncake_direct_linker.logger, "warning") as warning:
            with patch.object(mooncake_direct_linker.logger, "exception") as exception:
                linker.load_layer_wise(7, [[_transfer(PoolName.KV, 2)]], [("a", 0.0)])
        return linker, calls, warning, exception

    def test_a_failed_transfer_is_read_again_once_before_the_load_fails(self):
        # Hecate H112: one stage's failed read aborted its requests there only,
        # split the PP stages' queues and crashed the worker.
        linker, calls, warning, _ = self._read_once_with([None, -800])
        self.assertEqual(calls[1], ["kv-1"])
        linker.layer_done_counter.fail.assert_not_called()
        warning.assert_called_once()
        self.assertIn("retried %d objects", warning.call_args.args[0])
        linker.storage.store.batch_get_session_start.assert_called_once()  # no renewal
        self.assertEqual(linker.read_retries, {"retried": 1, "recovered": 1})

    def test_an_expired_lease_is_renewed_before_the_read_again(self):
        linker, calls, _, _ = self._read_once_with([-707, None])
        self.assertEqual(calls[1], ["kv-0"])
        store = linker.storage.store
        self.assertEqual(store.batch_get_session_start.call_args.args[0], ["kv-0"])
        linker.layer_done_counter.fail.assert_not_called()

    def test_a_lapsed_or_cut_session_is_renewed_before_the_read_again(self):
        # Rhea pm-lk1: a per-key read failed on -707 and -600 together and was
        # not retried, which aborted its requests on every stage.
        linker, calls, _, _ = self._read_once_with([-600, -707])
        self.assertEqual(calls[1], ["kv-0", "kv-1"])
        store = linker.storage.store
        self.assertEqual(store.batch_get_session_start.call_args.args[0], ["kv-0", "kv-1"])
        linker.layer_done_counter.fail.assert_not_called()
        self.assertEqual(linker.read_retries, {"retried": 2, "recovered": 2})

    def test_a_retry_that_fails_again_fails_the_load_and_is_not_recovered(self):
        linker = _linker(log_load_timing=False)
        linker.num_layers = 1
        linker.storage.store.batch_get_into_multi_buffer_ranges.side_effect = (
            lambda keys, ptrs, sizes, offsets: [-800] * len(keys)
        )
        with patch.object(mooncake_direct_linker.logger, "exception"):
            linker.load_layer_wise(7, [[_transfer(PoolName.KV, 2)]], [("a", 0.0)])
        linker.layer_done_counter.fail.assert_called()
        self.assertEqual(linker.read_retries, {"retried": 2, "recovered": 0})

    def test_a_missing_object_still_fails_and_names_its_codes(self):
        linker, calls, _, exception = self._read_once_with([-704, -800])
        self.assertEqual(len(calls), 1)  # a permanent failure is not retried
        linker.layer_done_counter.fail.assert_called()
        error = linker.layer_done_counter.fail.call_args.args[1]
        self.assertIn("failed by code {-704: 1, -800: 1}", str(error))

    def test_queue_time_follows_the_request_into_the_load_thread(self):
        linker = _linker(log_load_timing=True)
        linker.pool_group = SimpleNamespace(
            resolve_transfers=lambda transfers, **kwargs: transfers
        )
        linker.pending_loads = {}
        linker.load_queued_at = {}
        linker.mla_broadcaster = None
        linker.gc_frozen = True
        linker.load_queue = queue.Queue()
        linker.broadcast_loads = {}
        linker.stats = {"load": 0}
        linker.layer_done_counter.update_producer.return_value = 3
        self.assertTrue(linker.load("kept", [_transfer(PoolName.KV, 1)]))
        self.assertTrue(linker.load("dropped", [_transfer(PoolName.KV, 1)]))
        self.assertTrue(linker.cancel_queued_load("dropped"))
        with patch.object(mooncake_direct_linker, "device_module", MagicMock()):
            self.assertEqual(linker.start_layer_wise_loading(), 3)
        index, pending, _, queued_at, late = linker.load_queue.get_nowait()
        self.assertEqual(
            (index, list(pending), list(queued_at)), (3, ["kept"], ["kept"])
        )
        self.assertIsNone(late)  # not overlapped: no pool waits again
        self.assertEqual(linker.load_queued_at, {})


class TestAsyncAdmissionPrepins(CustomTestCase):
    """SGLANG_LINKER_ASYNC_ADMISSION: pre-pins are renewed while their request
    stays in the next-admitted window and end once it leaves it."""

    def _linker_with(self, entries, renew_s=2.0):
        linker = _linker(log_load_timing=False)
        linker.pin_samples, linker.pin_holes = [], 0
        linker.prepin_pool = ThreadPoolExecutor(1)
        linker.prepin_renew_s, linker.prepin_renewals = renew_s, 0
        linker.prepins = {}
        for rid, (pages, age) in entries.items():
            done = Future()
            done.set_result((pages, 0.0))
            own, others = [f"{rid}-own"], [f"{rid}-other"]
            linker.prepins[rid] = (
                ("tag", rid), done, 2, own, others, time.monotonic() - age,
                [(False, 0, 2)],
            )
        return linker

    def _flush(self, linker):
        linker.prepin_pool.submit(lambda: None).result(timeout=5)

    def test_a_kept_prepin_is_renewed_after_the_interval_and_not_before(self):
        linker = self._linker_with({"old": (2, 5.0), "new": (2, 0.0)})
        store = linker.storage.store
        linker.keep_prepins(["old", "new"])
        self._flush(linker)
        store.batch_get_session_start.assert_called_once_with(["old-own", "old-other"])
        store.batch_get_session_end.assert_not_called()
        self.assertEqual(linker.prepin_renewals, 1)
        self.assertLess(time.monotonic() - linker.prepins["old"][5], 1.0)
        # The renewed pin is what admission takes.
        with patch.object(linker, "_prepin_tag", return_value=("tag", "old")):
            self.assertEqual(linker.take_prepin("old", []), 2)
        self.assertEqual(linker.pinned["old"], ["old-own"])

    def test_a_prepin_leaving_the_window_ends_its_sessions_and_a_hole_holds_none(self):
        linker = self._linker_with({"gone": (2, 0.0), "hole": (1, 0.0), "kept": (2, 0.0)})
        linker.keep_prepins(["kept"])
        self._flush(linker)
        store = linker.storage.store
        store.batch_get_session_end.assert_called_once_with(["gone-own", "gone-other"])
        self.assertEqual(list(linker.prepins), ["kept"])

    def test_an_urgent_prepin_is_held_unrenewed_and_jumps_the_window_queue(self):
        # A hit admission keeps skipping must not lose its pin to a renewal in
        # flight (it reads as not ready) nor wait behind the window's pins.
        linker = self._linker_with({"r": (2, 5.0)})
        store = linker.storage.store
        linker.keep_prepins(["r"], urgent=["r"])
        self._flush(linker)
        store.batch_get_session_start.assert_not_called()
        self.assertTrue(linker.prepins["r"][1].done())

        linker.urgent_pool = None
        gate = threading.Event()
        linker.prepin_pool.submit(lambda: gate.wait(5))  # the window's thread is busy
        queued = Future()  # a window pin of "q" still queued behind it
        linker.prepins["q"] = (("tag", "q"), queued, 2, ["q-own"], [], time.monotonic(), [(False, 0, 2)])
        store.batch_get_session_start.side_effect = lambda keys: [0] * len(keys)
        with (
            patch.object(linker, "_prepin_tag", return_value=("tag", "q")),
            patch.object(linker, "_pin_keys", return_value=(2, ["q-own"], [], [(False, 0, 2)])),
        ):
            linker.start_prepin("q", [], urgent=True)
            self.assertEqual(linker.prepins["q"][1].result(timeout=5)[0], 2)
        self.assertTrue(queued.cancelled())
        gate.set()

    def test_prepins_start_on_the_configured_number_of_threads(self):
        # Rhea pm-lk3: one pre-pin thread at 44-61 ms p50 per session start left
        # most hits unready when admission reached them.
        linker = self._linker_with({})
        linker.prepin_pool = None
        linker.storage.store.batch_get_session_start.side_effect = lambda keys: [0] * len(keys)
        with (
            envs.SGLANG_LINKER_PREPIN_THREADS.override(4),
            patch.object(linker, "_prepin_tag", return_value=("tag", "r")),
            patch.object(linker, "_pin_keys", return_value=(1, ["k"], [], [(False, 0, 1)])),
        ):
            linker.start_prepin("r", [])
        self.assertEqual(linker.prepin_pool._max_workers, 4)
        self.assertEqual(linker.prepins["r"][1].result(timeout=5)[0], 1)

    def test_a_drop_ends_the_sessions_only_after_a_queued_renewal(self):
        linker = self._linker_with({"r": (2, 5.0)})
        store = linker.storage.store
        gate, order = threading.Event(), []

        def start(keys):
            gate.wait(5)
            order.append("start")
            return [0] * len(keys)

        store.batch_get_session_start.side_effect = start
        store.batch_get_session_end.side_effect = lambda keys: order.append("end")
        linker.keep_prepins(["r"])  # renewal queued, blocked on the gate
        linker.keep_prepins([])  # the request left the window
        gate.set()
        self._flush(linker)
        self.assertEqual(order, ["start", "end"])
        self.assertEqual(linker.prepins, {})


class TestLoadOverlap(CustomTestCase):
    """SGLANG_LINKER_LOAD_OVERLAP: a batch's load waits for the GPU work queued
    before the previous batch's launch, not for the previous forward."""

    def _start_batches(self, overlap, batches):
        linker = _linker(log_load_timing=False)
        linker.pool_group = SimpleNamespace(
            resolve_transfers=lambda transfers, **kwargs: transfers
        )
        linker.pending_loads, linker.load_queued_at = {}, {}
        linker.mla_broadcaster, linker.gc_frozen = None, True
        linker.load_queue, linker.broadcast_loads = queue.Queue(), {}
        linker.stats = {"load": 0}
        linker.load_overlap = overlap
        events = []

        def event():
            events.append(MagicMock(name=f"event{len(events)}"))
            return events[-1]

        waits = []
        with patch.object(mooncake_direct_linker, "device_module") as device:
            device.Event.side_effect = event
            for rid in batches:
                if rid is not None:
                    linker.load(rid, [_transfer(PoolName.KV, 1)])
                linker.start_layer_wise_loading()
                while not linker.load_queue.empty():
                    waits.append(events.index(linker.load_queue.get_nowait()[2]))
        return waits, events, linker

    def test_a_load_waits_for_the_launch_before_the_previous_batch(self):
        # Batch 0 loads (no earlier launch: its own ready event), batch 1 has no
        # load, batch 2 loads and waits only for batch 1's launch event.
        waits, events, linker = self._start_batches(True, ["a", None, "b"])
        # Events: 0 launch(a), 1 ready(a), 2 launch(none), 3 launch(b), 4 ready(b).
        self.assertEqual(waits, [1, 2])
        self.assertEqual(linker.stats["load_overlapped"], 1)
        for index in (0, 2, 3):
            events[index].record.assert_called_once()

    def test_without_the_flag_a_load_waits_for_all_prior_work(self):
        waits, events, _ = self._start_batches(False, ["a", None, "b"])
        self.assertEqual(waits, [0, 1])  # each batch's own ready event
        self.assertEqual(len(events), 2)


class _FakeEvent:
    def __init__(self):
        self.done = False
        self.recorded = False

    def record(self):
        self.recorded = True

    def query(self):
        return self.done


class _PageAllocator:
    """Hands out free pages in order; freed pages go back to the end."""

    def __init__(self, num_pages, page_size):
        self.page_size = page_size
        self.free_pages = list(range(num_pages))

    def alloc(self, num_tokens):
        n = num_tokens // self.page_size
        if n > len(self.free_pages):
            return None
        pages, self.free_pages = self.free_pages[:n], self.free_pages[n:]
        return (
            torch.tensor(pages)[:, None] * self.page_size + torch.arange(self.page_size)
        ).reshape(-1)

    def free(self, indices):
        self.free_pages.extend((indices.view(-1, self.page_size)[:, 0] // self.page_size).tolist())


def _pages_of(indices, page_size):
    return set((indices.view(-1, page_size)[:, 0] // page_size).tolist())


class TestLoadSlotReserve(CustomTestCase):
    """SGLANG_LINKER_LOAD_RESERVE_PAGES: load slots whose ids are on the CPU, so
    an overlapped load never reads an index the GPU has not written (Hecate D4b)."""

    def _reserve(self, target=4, page_size=2, pages=16):
        allocator = _PageAllocator(pages, page_size)
        events = []

        def event():
            events.append(_FakeEvent())
            return events[-1]

        reserve = LoadSlotReserve(
            target, page_size, torch.device("cpu"), allocator.alloc, event, pin_memory=False
        )
        return reserve, allocator, events

    def test_slots_come_only_from_held_pages_never_from_live_or_evicted_ones(self):
        reserve, allocator, events = self._reserve()
        held = set(reserve.free)
        in_flight = allocator.alloc(4)  # the running batch's pages
        evicted = allocator.alloc(2)
        allocator.free(evicted)  # just evicted: back in the allocator
        cpu, device = reserve.take(2)
        taken = _pages_of(cpu, 2)
        self.assertTrue(taken <= held)
        self.assertFalse(taken & _pages_of(in_flight, 2))
        self.assertFalse(taken & _pages_of(evicted, 2))
        self.assertEqual(cpu.device.type, "cpu")
        self.assertTrue(torch.equal(cpu, device))

    def test_a_replacement_joins_only_once_its_ids_are_on_the_cpu(self):
        reserve, allocator, events = self._reserve(target=4)
        self.assertIsNotNone(reserve.take(3))
        self.assertEqual((len(reserve.free), reserve.held_pages()), (1, 4))
        self.assertTrue(events[-1].recorded)
        # The replacement's copy has not landed: too few ids, fall back.
        self.assertIsNone(reserve.take(2))
        self.assertEqual(reserve.counts["fallback"], 1)
        events[-1].done = True
        cpu, _ = reserve.take(2)
        self.assertEqual(len(_pages_of(cpu, 2)), 2)

    def test_held_pages_count_as_protected_in_the_tree_accounting(self):
        reserve, _, _ = self._reserve(target=4)
        reserve.take(1)
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(protected_size=lambda: 10, full_protected_size=lambda: 10)
        self.assertEqual(cache.full_protected_size(), 10)
        cache.linker_load_reserve = reserve
        self.assertEqual(cache.full_protected_size(), 10 + reserve.held_tokens())
        self.assertEqual(cache.protected_size(), 10 + 4 * 2)

    def test_a_cache_reset_drops_the_reserve_from_the_accounting(self):
        reserve, _, _ = self._reserve(target=4)
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(protected_size=lambda: 0, full_protected_size=lambda: 0)
        cache.linker, cache._reset_full = None, lambda: None
        cache.linker_load_reserve = reserve
        self.assertGreater(cache.full_protected_size(), 0)
        cache.reset()
        self.assertEqual((cache.full_protected_size(), cache.protected_size()), (0, 0))


class TestReservedLoadSlotsInTheFullComponent(CustomTestCase):
    def _component(self, pages=16):
        component = FullComponent.__new__(FullComponent)
        allocator = _PageAllocator(pages, 2)
        allocator.device = "cpu"
        allocator.available_size = lambda: len(allocator.free_pages) * 2
        component.cache = SimpleNamespace(page_size=2, evict=MagicMock())
        component._full_allocator = lambda: allocator
        return component, allocator

    def test_without_the_reserve_a_load_allocates_as_before(self):
        component, _ = self._component()
        with envs.SGLANG_LINKER_LOAD_RESERVE_PAGES.override(0):
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOAD, None, ["a", "b"]
            )
        self.assertIsNone(transfer.cpu_indices)
        self.assertEqual(transfer.device_indices.tolist(), [0, 1, 2, 3])
        self.assertIsNone(getattr(component.cache, "linker_load_reserve", None))

    def test_with_the_reserve_a_load_lands_in_cpu_known_slots(self):
        component, _ = self._component()
        with envs.SGLANG_LINKER_LOAD_RESERVE_PAGES.override(3):
            first = component.build_external_linker_transfer(
                LinkerTransferPhase.LOAD, None, ["a", "b"]
            )
            reserve = component.cache.linker_load_reserve
            self.assertEqual(_pages_of(first.cpu_indices, 2), {0, 1})
            self.assertTrue(torch.equal(first.cpu_indices, first.device_indices))
            self.assertEqual(reserve.held_pages(), 3)  # replaced from the allocator

    def test_a_load_larger_than_the_reserve_allocates_as_before(self):
        component, _ = self._component()
        with envs.SGLANG_LINKER_LOAD_RESERVE_PAGES.override(1):
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOAD, None, ["a", "b"]
            )
        self.assertIsNone(transfer.cpu_indices)
        self.assertEqual(component.cache.linker_load_reserve.counts["fallback"], 1)


class TestResolveCpuSlots(CustomTestCase):
    def _group(self, mapper):
        entry = SimpleNamespace(
            translate_indices=lambda x: x if mapper is None else mapper(x),
            _index_mapper=mapper,
        )
        return SimpleNamespace(sources={PoolName.KV: PoolName.KV}, entry_map={PoolName.KV: entry})

    def test_cpu_slots_resolve_to_cpu_locations_unless_a_mapper_needs_the_device(self):
        cpu = torch.arange(4)
        device = torch.arange(4) + 100  # stands in for the device copy
        transfer = PoolTransfer(name=PoolName.KV, keys=["a"], device_indices=device, cpu_indices=cpu)
        (plain,) = DevicePoolGroup.resolve_transfers(self._group(None), [transfer])
        self.assertIs(plain.host_indices, cpu)
        (mapped,) = DevicePoolGroup.resolve_transfers(self._group(lambda x: x + 1), [transfer])
        self.assertEqual(mapped.host_indices.tolist(), [101, 102, 103, 104])


class TestOverlappedLoadReadsCpuSlotsFirst(CustomTestCase):
    """With overlap, a pool whose slot indices are on the CPU is read before
    this batch's GPU work finishes; a pool with GPU-written indices (KDA state)
    only after it (late_wait)."""

    def _run(self, late_wait):
        order = []
        linker = _linker(log_load_timing=False)
        linker.load_per_key, linker.stats = True, {}

        class _Recording(_Pool):
            def prepare_locations(self, host_indices):
                order.append(("locate", self.name))
                return list(range(len(list(host_indices))))

        linker.pools = {name: _Recording(name) for name in KEY_BYTES}
        store = linker.storage.store
        store.batch_get_into_multi_buffer_ranges.side_effect = lambda keys, ptrs, sizes, offsets: (
            order.append(("read", tuple(k.split("-")[0] for k in keys))) or [sum(s) for s in sizes]
        )
        state = _transfer(PoolName.SWA, 1)
        state.host_indices = MagicMock(device=SimpleNamespace(type="cuda"), __iter__=lambda self: iter([0]))
        kv = _transfer(PoolName.KV, 2)
        kv.host_indices = torch.arange(2)
        wait = (lambda: order.append(("wait",))) if late_wait else None
        linker.load_layer_wise(7, [[kv, state]], [("a", 0.0)], late_wait=wait)
        linker.layer_done_counter.fail.assert_not_called()
        return order

    def test_cpu_slot_pools_are_read_before_the_wait_and_the_rest_after(self):
        kv, swa = PoolName.KV.value, PoolName.SWA.value
        order = self._run(late_wait=True)
        self.assertEqual(
            order,
            [("locate", PoolName.KV), ("read", (kv, kv)), ("wait",),
             ("locate", PoolName.SWA), ("read", (swa,))],
        )

    def test_without_overlap_one_read_covers_every_pool(self):
        order = self._run(late_wait=False)
        self.assertEqual([o[0] for o in order], ["locate", "locate", "read"])

    def test_overlap_needs_the_reserve(self):
        self.assertFalse(MooncakeDirectLinker._overlap_enabled(True, 0))
        self.assertTrue(MooncakeDirectLinker._overlap_enabled(True, 8))
        self.assertFalse(MooncakeDirectLinker._overlap_enabled(False, 8))


class TestLoadPerKey(CustomTestCase):
    """SGLANG_LINKER_LOAD_PER_KEY: one range get per batch, every layer of a key
    in its entry, the same bytes as the per-layer reads."""

    def test_one_call_carries_every_layer_and_logs_the_same_bytes(self):
        requests = [
            [_transfer(PoolName.KV, 2), _transfer(PoolName.SWA, 1)],
            [_transfer(PoolName.KV, 3)],
        ]
        logged = {}
        for per_key in (False, True):
            linker = _linker(log_load_timing=True)
            linker.load_per_key = per_key
            with self.assertLogs(LOGGER, level="INFO") as logs:
                linker.load_layer_wise(7, requests, [("a", 0.0), ("b", 0.0)])
            logged[per_key] = [
                line.split("bytes=")[1].split()[0] for line in logs.output if "bytes=" in line
            ]
            calls = linker.storage.store.batch_get_into_multi_buffer_ranges.call_args_list
            completed = [c.args for c in linker.layer_done_counter.complete.call_args_list]
            self.assertEqual(completed, [(7, 0), (7, 1)])
            linker.layer_done_counter.fail.assert_not_called()
        self.assertEqual(logged[True], logged[False])
        self.assertEqual(logged[True], ["420", "600"])
        self.assertEqual(len(calls), 1)
        keys, ptrs, sizes, offsets = calls[0].args
        self.assertEqual(keys, ["kv-0", "kv-1", "kv-0", "kv-1", "kv-2", "swa-0"])
        self.assertEqual(sizes[0], [100, 100])  # both layers of the key
        self.assertEqual(sizes[-1], [10, 10])

    def test_a_failed_object_fails_the_whole_batch(self):
        linker = _linker(log_load_timing=False)
        linker.load_per_key = True
        linker.storage.store.batch_get_into_multi_buffer_ranges.side_effect = (
            lambda keys, ptrs, sizes, offsets: [-704] + [sum(i) for i in sizes[1:]]
        )
        with patch.object(mooncake_direct_linker.logger, "exception"):
            linker.load_layer_wise(7, [[_transfer(PoolName.KV, 2)]], [("a", 0.0)])
        linker.layer_done_counter.fail.assert_called()
        linker.layer_done_counter.complete.assert_not_called()


def _array_sums(sizes, indptr):
    bounds = indptr.tolist()
    return [int(sizes[a:b].sum()) for a, b in zip(bounds, bounds[1:])]


class TestStoreArrays(CustomTestCase):
    """SGLANG_LINKER_STORE_ARRAYS: range gets pass flat uint64 arrays and a CSR
    indptr to the store's *_arrays method when it has one, else the list call."""

    def _linker(self, with_method=True, per_key=False, first_codes=None):
        linker = _linker(log_load_timing=True)
        linker.store_arrays = True
        linker.load_per_key = per_key
        linker.vector_meta = False
        store = linker.storage.store
        calls = []

        def arrays(keys, ptrs, sizes, offsets, indptr):
            for field in (ptrs, sizes, offsets, indptr):
                self.assertEqual(field.dtype, np.uint64)
            calls.append((list(keys), ptrs, sizes, offsets, indptr))
            good = _array_sums(sizes, indptr)
            if first_codes is not None and len(calls) == 1:
                return [good[i] if c is None else c for i, c in enumerate(first_codes)]
            return good

        if with_method:
            store.batch_get_into_multi_buffer_ranges_arrays.side_effect = arrays
        else:
            del store.batch_get_into_multi_buffer_ranges_arrays
        return linker, calls

    def _bytes(self, linker, requests):
        with self.assertLogs(LOGGER, level="INFO") as logs:
            linker.load_layer_wise(7, requests, [("a", 0.0), ("b", 0.0)][: len(requests)])
        return [line.split("bytes=")[1].split()[0] for line in logs.output if "bytes=" in line]

    def test_the_array_call_replaces_the_list_call_with_the_same_bytes(self):
        requests = [
            [_transfer(PoolName.KV, 2), _transfer(PoolName.SWA, 1)],
            [_transfer(PoolName.KV, 3)],
        ]
        for per_key in (False, True):
            with self.subTest(per_key=per_key):
                linker, calls = self._linker(per_key=per_key)
                self.assertEqual(self._bytes(linker, requests), ["420", "600"])
                linker.storage.store.batch_get_into_multi_buffer_ranges.assert_not_called()
                linker.layer_done_counter.fail.assert_not_called()
                self.assertEqual(len(calls), 1 if per_key else 4)
                keys, ptrs, sizes, offsets, indptr = calls[0]
                if per_key:
                    self.assertEqual(keys, ["kv-0", "kv-1", "kv-0", "kv-1", "kv-2", "swa-0"])
                    self.assertEqual(indptr.tolist(), [0, 2, 4, 6, 8, 10, 12])
                    self.assertEqual(sizes.tolist(), [100] * 10 + [10, 10])
                else:
                    self.assertEqual(indptr.tolist(), list(range(len(keys) + 1)))

    def test_vector_meta_hands_the_numpy_ranges_to_the_store_as_built(self):
        # VECTOR_META + COALESCE_RANGES: the per-key arrays reach the store without
        # a round trip through Python lists, and equal what the list call gets.
        table = np.array([[0, 1000, 64, 8, 0], [0, 1008, 64, 8, 8]], dtype=np.int64)

        class _VectorPool(_Pool):
            def key_range_table(self, num_layers):
                return table

            def key_range_rows(self, locations):
                return np.asarray(locations, dtype=np.int64).reshape(-1, 1)

        got = {}
        for arrays in (False, True):
            linker, calls = self._linker(per_key=True)
            linker.store_arrays = arrays
            linker.vector_meta = linker.coalesce_ranges = True
            linker.flag_hits = Counter()
            linker.pools = {PoolName.KV: _VectorPool(PoolName.KV)}
            linker.log_load_timing = False
            linker.load_layer_wise(7, [[_transfer(PoolName.KV, 3)]], [("a", 0.0)])
            linker.layer_done_counter.fail.assert_not_called()
            store = linker.storage.store
            if arrays:
                store.batch_get_into_multi_buffer_ranges.assert_not_called()
                _, ptrs, sizes, offsets, indptr = calls[0]
                got[arrays] = (ptrs.tolist(), sizes.tolist(), offsets.tolist(), indptr.tolist())
            else:
                _, ptrs, sizes, offsets = store.batch_get_into_multi_buffer_ranges.call_args.args
                got[arrays] = (ptrs, sizes, offsets)
        self.assertEqual(got[True][0], [1000, 1064, 1128])  # coalesced: one range a key
        self.assertEqual(got[True][3], [0, 1, 2, 3])
        bounds = got[True][3]
        self.assertEqual(
            tuple(
                [field[a:b] for a, b in zip(bounds, bounds[1:])] for field in got[True][:3]
            ),
            got[False],
        )

    def test_retries_read_only_the_failed_keys_through_the_array_call(self):
        linker, calls = self._linker(first_codes=[-707, None])
        linker.num_layers = 1
        with patch.object(mooncake_direct_linker.logger, "warning"):
            linker.load_layer_wise(7, [[_transfer(PoolName.KV, 2)]], [("a", 0.0)])
        self.assertEqual([c[0] for c in calls], [["kv-0", "kv-1"], ["kv-0"]])
        self.assertEqual(calls[1][4].tolist(), [0, 1])
        store = linker.storage.store
        self.assertEqual(store.batch_get_session_start.call_args.args[0], ["kv-0"])
        linker.layer_done_counter.fail.assert_not_called()
        self.assertEqual(linker.read_retries, {"retried": 1, "recovered": 1})

    def test_without_the_method_the_list_call_runs_and_logs_once(self):
        requests = [[_transfer(PoolName.KV, 2)], [_transfer(PoolName.KV, 3)]]
        for per_key in (False, True):
            with self.subTest(per_key=per_key):
                linker, calls = self._linker(with_method=False, per_key=per_key)
                with patch.object(mooncake_direct_linker.logger, "warning") as warning:
                    self.assertEqual(self._bytes(linker, requests), ["400", "600"])
                    linker.load_layer_wise(7, requests, [("a", 0.0), ("b", 0.0)])
                warning.assert_called_once()
                self.assertIn("SGLANG_LINKER_STORE_ARRAYS", warning.call_args.args[0])
                self.assertTrue(linker.storage.store.batch_get_into_multi_buffer_ranges.called)
                linker.layer_done_counter.fail.assert_not_called()


class TestSessionRefcount(CustomTestCase):
    """Mooncake ends a key's read session for every holder in the client, so
    with refcounting a session starts on the first holder and ends with the
    last (Hecate r5: two loads on shared keys failed with -600)."""

    def _refs(self, codes=None):
        store = MagicMock()
        store.batch_get_session_start.side_effect = lambda keys: [
            (codes or {}).get(key, 0) for key in keys
        ]
        return mooncake_direct_linker._SessionRefs(store, counted=True), store

    def test_shared_keys_start_on_the_first_holder_and_end_with_the_last(self):
        refs, store = self._refs()
        self.assertEqual(refs.start(["a", "b"]), [0, 0])
        self.assertEqual(refs.start(["b", "c", "c"]), [0, 0, 0])
        starts = [c.args[0] for c in store.batch_get_session_start.call_args_list]
        self.assertEqual(starts, [["a", "b"], ["c"]])
        refs.end(["a", "b"])
        store.batch_get_session_end.assert_called_once_with(["a"])
        refs.end(["b", "c"])
        refs.end(["c"])
        ends = [c.args[0] for c in store.batch_get_session_end.call_args_list]
        self.assertEqual(ends, [["a"], ["b"], ["c"]])
        self.assertEqual(refs.counts, {})

    def test_a_failed_start_keeps_no_session_and_counts_nothing(self):
        refs, store = self._refs(codes={"gone": -704})
        refs.start(["held"])
        self.assertEqual(refs.start(["held", "new", "gone"]), [0, 0, -704])
        store.batch_get_session_end.assert_called_once_with(["new"])
        self.assertEqual(refs.counts, {"held": 1})
        refs.end(["held", "gone"])  # an end of a key never held is a no-op
        self.assertEqual(store.batch_get_session_end.call_args.args[0], ["held"])

    def test_a_load_sharing_a_prepinned_prefix_leaves_the_pin_held(self):
        linker = _linker(log_load_timing=False)
        linker.count_sessions = True
        store = linker.storage.store
        transfer = _transfer(PoolName.KV, 2)
        linker.sessions.start(["kv-0"])  # another request's pre-pin
        linker.load_layer_wise(7, [[transfer]], [("b", 0.0)])
        starts = [c.args[0] for c in store.batch_get_session_start.call_args_list]
        self.assertEqual(starts, [["kv-0"], ["kv-1"]])
        store.batch_get_session_end.assert_called_once_with(["kv-1"])
        linker.layer_done_counter.fail.assert_not_called()
        linker.sessions.end(["kv-0"])
        self.assertEqual(store.batch_get_session_end.call_args.args[0], ["kv-0"])

    def test_an_async_end_leaves_the_rpc_to_its_thread_and_a_restart_cancels_it(self):
        # Rhea pm-lk3: ending expired pins took 24% of PP0's scheduler thread.
        refs, store = self._refs()
        gate = threading.Event()
        refs.ender = ThreadPoolExecutor(1)
        refs.ender.submit(gate.wait, 5)  # the end thread is busy
        refs.start(["a", "b"])
        refs.end_async(["a", "b"])
        store.batch_get_session_end.assert_not_called()  # not on this thread
        self.assertEqual(refs.counts, {})
        refs.start(["b"])  # b is held again before its end runs
        gate.set()
        refs.flush()
        store.batch_get_session_end.assert_called_once_with(["a"])
        self.assertEqual(refs.counts, {"b": 1})

    def test_no_store_rpc_runs_under_the_count_lock(self):
        refs, store = self._refs()
        entered, release = threading.Event(), threading.Event()

        def slow(keys):
            if keys == ["a"]:
                entered.set()
                release.wait(5)
            return [0] * len(keys)

        store.batch_get_session_start.side_effect = slow
        first = threading.Thread(target=refs.start, args=(["a"],))
        first.start()
        self.assertTrue(entered.wait(5))
        done = threading.Event()
        threading.Thread(target=lambda: (refs.start(["x"]), done.set())).start()
        self.assertTrue(done.wait(5))  # another key's start is not held up
        release.set()
        first.join(5)
        self.assertEqual(refs.counts, {"a": 1, "x": 1})

    def test_a_start_waits_for_an_end_of_the_same_key_already_on_the_wire(self):
        refs, store = self._refs()
        order, entered, release = [], threading.Event(), threading.Event()

        def end(keys):
            entered.set()
            release.wait(5)
            order.append(("end", list(keys)))

        store.batch_get_session_end.side_effect = end
        store.batch_get_session_start.side_effect = lambda keys: (
            order.append(("start", list(keys))) or [0] * len(keys)
        )
        refs.start(["k"])
        refs.end_async(["k"])
        self.assertTrue(entered.wait(5))
        starter = threading.Thread(target=refs.start, args=(["k"],))
        starter.start()
        release.set()
        starter.join(5)
        refs.flush()
        self.assertEqual(order, [("start", ["k"]), ("end", ["k"]), ("start", ["k"])])
        self.assertEqual(refs.counts, {"k": 1})

    def test_expired_pins_end_off_the_scheduler_thread(self):
        linker = _linker(log_load_timing=False)
        linker.count_sessions = True
        linker.sessions.start(["later-stage-key"])
        gate = threading.Event()
        linker.sessions.ender = ThreadPoolExecutor(1)
        linker.sessions.ender.submit(gate.wait, 5)
        linker.pin_expiry.append([0.0, ["later-stage-key"], 0.0])
        linker._end_expired_pins(1.0)
        store = linker.storage.store
        store.batch_get_session_end.assert_not_called()
        gate.set()
        linker.sessions.flush()
        store.batch_get_session_end.assert_called_once_with(["later-stage-key"])

    def test_without_refcounting_every_end_reaches_the_store(self):
        linker = _linker(log_load_timing=False)
        linker.sessions.start(["kv-0"])
        linker.load_layer_wise(7, [[_transfer(PoolName.KV, 2)]], [("b", 0.0)])
        store = linker.storage.store
        store.batch_get_session_end.assert_called_once_with(["kv-0", "kv-1"])


def _meminfo(node, total_kb, free_kb, file_kb, dirty_kb, reclaimable_kb):
    return "".join(
        f"Node {node} {key}: {value} kB\n"
        for key, value in (
            ("MemTotal", total_kb),
            ("MemFree", free_kb),
            ("Active(file)", file_kb),
            ("Inactive(file)", 0),
            ("Dirty", dirty_kb),
            ("Writeback", 0),
            ("SReclaimable", reclaimable_kb),
        )
    ) + f"Node {node} HugePages_Total:     0\n"


class TestRecentKeys(CustomTestCase):
    def test_full_set_drops_its_oldest_keys_in_constant_time(self):
        # Hecate H106: dropping a full plain dict's first key scanned the slots
        # freed before it; every load then held its scheduler for seconds.
        recent = mooncake_direct_linker._RecentKeys(500_000)
        loads = [[f"k{load}-{i}" for i in range(4_800)] for load in range(125)]
        for keys in loads[:105]:  # one load's objects at a time, past capacity
            recent.add(keys)
        began = time.perf_counter()
        for keys in loads[105:]:
            recent.add(keys)
        self.assertLess(time.perf_counter() - began, 0.5)  # a plain dict took ~2 s
        self.assertEqual(len(recent.keys), 500_000)
        self.assertNotIn("k20-3999", recent)  # the oldest 100,000 keys are gone
        self.assertIn("k20-4000", recent)
        recent.add(["k20-4000"])  # a key read again moves to the newest end
        recent.add(["newest"])
        self.assertIn("k20-4000", recent)
        self.assertNotIn("k20-4001", recent)


class TestMooncakeLinkerNumaHeadroom(CustomTestCase):
    def _sysfs(self, tmp, nodes):
        paths = []
        for node, info in nodes.items():
            path = f"{tmp}/node{node}/meminfo"
            os.makedirs(os.path.dirname(path))
            with open(path, "w") as f:
                f.write(info)
            paths.append(path)
        return patch.object(mooncake_direct_linker.glob, "glob", return_value=paths)

    def test_available_counts_free_clean_file_and_reclaimable(self):
        with tempfile.TemporaryDirectory() as tmp, self._sysfs(
            tmp,
            {1: _meminfo(1, 1000, 50, 30, 10, 5), 0: _meminfo(0, 1000, 400, 0, 0, 0)},
        ):
            nodes = mooncake_direct_linker._numa_available()
        self.assertEqual(nodes, [(0, 400 * 1024, 1000 * 1024), (1, 75 * 1024, 1000 * 1024)])

    def test_first_request_logs_once_and_flags_nodes_under_ten_percent(self):
        linker = _linker(log_load_timing=False)
        linker._numa_first_request_logged = False
        with tempfile.TemporaryDirectory() as tmp, self._sysfs(
            tmp,
            {0: _meminfo(0, 1000, 400, 0, 0, 0), 1: _meminfo(1, 1000, 50, 30, 10, 5)},
        ), self.assertLogs(LOGGER, level="INFO") as logs:
            linker._log_numa_first_request()
            linker._log_numa_first_request()
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(logs.records[0].levelname, "WARNING")
        self.assertIn("at first request", logs.output[0])
        self.assertIn("UNDER 10% on nodes [1]", logs.output[0])

    def test_linkers_built_without_init_never_log_headroom(self):
        linker = _linker(log_load_timing=False)
        with patch.object(mooncake_direct_linker, "_numa_available") as available:
            linker._log_numa_first_request()
        available.assert_not_called()


class TestMooncakeLinkerNumaResidency(CustomTestCase):
    def test_resident_bytes_sum_each_node_at_its_page_size(self):
        maps = (
            "aaaa default anon=3 dirty=3 N0=2 N1=1 kernelpagesize_kB=64\n"
            "bbbb prefer:1 file=/x mapped=4 N1=4 kernelpagesize_kB=4\n"
            "cccc default\n"
        )
        with tempfile.NamedTemporaryFile("w") as f:
            f.write(maps)
            f.flush()
            held = mooncake_direct_linker._resident_by_numa(f.name)
        self.assertEqual(held, {0: 2 * 65536, 1: 65536 + 4 * 4096})


class TestMooncakeLinkerDebugWaitAllLayers(CustomTestCase):
    def _start(self, flag):
        linker = _linker(log_load_timing=False)
        linker.mla_broadcaster = None
        linker.pending_loads = {"r": [_transfer(PoolName.KV, 1)]}
        linker.load_queued_at = {}
        linker.gc_frozen = True
        linker.broadcast_loads = {}
        linker.stats = {"load": 0}
        linker.load_queue = queue.Queue()
        linker.layer_done_counter = mooncake_direct_linker.LayerWiseLoadCounter(2)
        landed = []

        def land():
            index, *_ = linker.load_queue.get(timeout=5)
            time.sleep(0.05)
            for layer in range(2):
                landed.append(layer)
                linker.layer_done_counter.complete(index, layer)

        loader = threading.Thread(target=land)
        loader.start()
        with patch.object(
            mooncake_direct_linker.envs.SGLANG_LINKER_DEBUG_WAIT_ALL_LAYERS_FILE,
            "get",
            return_value=flag,
        ), patch.object(mooncake_direct_linker, "device_module") as device:
            linker.start_layer_wise_loading()
            done_at_return = list(landed)
        loader.join()
        return done_at_return, device

    def test_flag_file_makes_the_batch_wait_for_every_layer(self):
        with tempfile.NamedTemporaryFile() as flag:
            done, device = self._start(flag.name)
        self.assertEqual(done, [0, 1])
        device.synchronize.assert_called_once()

    def test_without_the_flag_file_loading_stays_layer_wise(self):
        done, device = self._start("/nonexistent/flag")
        self.assertEqual(done, [])
        device.synchronize.assert_not_called()


class _LeaseStore:
    """remove(): -706 for keys in ``leased`` (once each), else 0."""

    def __init__(self, leased=()):
        self.leased, self.removed = set(leased), []

    def remove(self, key):
        if key in self.leased:
            self.leased.discard(key)
            return -706
        self.removed.append(key)
        return 0

    def batch_remove(self, keys):
        self.batches = getattr(self, "batches", []) + [len(keys)]
        return [self.remove(key) for key in keys]


def _budget_linker(budget, stage, state_units=4, store=None, protect=0, recall=0):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.pools = {PoolName.KV: _Pool(PoolName.KV), PoolName.MAMBA: _Pool(PoolName.MAMBA)}
    linker.pool_group = SimpleNamespace(resolve_transfers=lambda transfers, **kwargs: transfers)
    linker.offload_pools = {PoolName.KV, PoolName.MAMBA}
    linker.gc_frozen, linker.state_path_cap, linker.page_size = True, 0, 64
    linker.record_offload_ready = lambda: []
    linker.offload_queue = queue.Queue()
    linker.pending_loads, linker.load_queued_at, linker.pinned = {}, {}, {}
    linker.log_load_timing = False
    linker.storage = SimpleNamespace(
        store=store or _LeaseStore(),
        _tag_keys=lambda keys: keys,
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, 1),
        pool_suffixes={},
        mla_suffix=f"pp{stage}",
        recently_loaded=mooncake_direct_linker._RecentKeys(16),
    )
    linker.store_tree = mooncake_direct_linker._StoreTree(
        budget, state_units, protect=protect, recall=recall
    )
    linker.page_remove_retries, linker.page_remove_counts = deque(), Counter()
    return linker


def _pages(*names):
    return SimpleNamespace(name=PoolName.KV, keys=list(names), host_indices=range(len(names)),
                           hit_policy=PoolHitPolicy.ALL_PAGES)


def _state(page):
    return SimpleNamespace(name=PoolName.MAMBA, keys=[page], host_indices=range(1),
                           hit_policy=PoolHitPolicy.TRAILING_PAGES)


def _queued_victim_pages(linker, drain=False):
    """Pages whose removal is queued (each stage names its own objects)."""
    pages = []
    while not linker.offload_queue.empty():
        task = linker.offload_queue.get_nowait()
        if task[0] == "remove":
            pages += sorted({key.split("_")[0] for key in task[1]})
            if drain:
                linker._remove_pages(*task[1:])
    return pages


class TestStoreBudgetTailFirst(CustomTestCase):
    """SGLANG_LINKER_STORE_BUDGET_PAGES: each stage evicts its stored prefixes
    leaf first, least recently used, and every stage picks the same pages
    (Rhea pm-lk2: Mooncake's own eviction left holes in 13% of PP0 lookups)."""

    EVENTS = [
        ("offload", ["a1", "a2"], None, (), ["a1", "a2"]),
        ("offload", ["a3", "a4"], None, ("a4",), ["a1", "a2", "a3", "a4"]),
        ("offload", ["b1", "b2", "b3"], None, ("b3",), ["b1", "b2", "b3"]),
        ("load", ["a1", "a2", "a3", "a4"]),
        ("offload", ["c1", "c2", "c3"], None, (), ["c1", "c2", "c3"]),
        ("offload", ["a5"], None, (), ["a1", "a2", "a3", "a4", "a5"]),
        ("offload", ["d1", "d2"], None, ("d2",), ["d1", "d2"]),
    ]

    def _play(self, linker, drain):
        victims = []
        for event in self.EVENTS:
            if event[0] == "offload":
                _, pages, _, states, path = event
                transfers = [_pages(*pages)] + [_state(page) for page in states]
                linker.offload(transfers, path_hashes=lambda path=path: path)
                # The put task goes first; removals queue after it.
                linker.offload_queue.get_nowait()
            else:
                linker.load("r-" + event[1][0], [_pages(*event[1])])
            victims.append(_queued_victim_pages(linker, drain))
        return victims

    def test_every_stage_evicts_the_same_pages_whatever_its_removals_did(self):
        # Stage 0 removes at once and a lease refuses one of its removals;
        # stage 1 has not run any removal yet. Their keys differ (own suffix).
        stage0 = _budget_linker(16, 0, store=_LeaseStore(leased={"b3_pp0_k"}))
        stage1 = _budget_linker(16, 1)
        victims0, victims1 = self._play(stage0, drain=True), self._play(stage1, drain=False)
        self.assertEqual(victims0, victims1)
        self.assertTrue(any(victims0))
        self.assertEqual(stage0.page_remove_counts["lease_retries"], 1)
        self.assertEqual(len(stage0.page_remove_retries), 1)

    def test_low_fraction_sets_the_pages_per_eviction_pass(self):
        # SGLANG_LINKER_STORE_LOW_FRACTION: closer to 1, fewer pages per pass.
        for low_fraction, per_pass in ((0.9, 11), (0.99, 2)):
            tree = mooncake_direct_linker._StoreTree(100, 4, low_fraction=low_fraction)
            for i in range(101):
                tree.add([f"p{i}"], None, ())
            self.assertEqual(len(tree.evict()), per_pass)

    def test_batch_removal_matches_one_call_per_key(self):
        # SGLANG_LINKER_BATCH_REMOVE: the same removed keys, counts and lease
        # retries as one call per key, in one store call per pass.
        keys = [f"p{i}_pp0_k" for i in range(10)]
        pages = [f"p{i}" for i in range(10)]
        runs = []
        for batch in (False, True):
            linker = _budget_linker(16, 0, store=_LeaseStore(leased={"p3_pp0_k"}))
            linker.batch_remove = batch
            linker.store_tree.evicted["p5"] = (0, None)
            linker.store_tree.recalled["p5"] = 7  # stored again after this eviction
            linker._remove_pages(keys, pages, decided=5)
            runs.append(linker)
        per_key, batched = runs
        self.assertEqual(per_key.storage.store.removed, batched.storage.store.removed)
        self.assertNotIn("p5_pp0_k", batched.storage.store.removed)
        self.assertEqual(per_key.page_remove_counts, batched.page_remove_counts)
        self.assertEqual(batched.page_remove_counts["lease_retries"], 1)
        self.assertEqual(len(batched.page_remove_retries), 1)
        self.assertEqual(batched.storage.store.batches, [9])

    def test_tail_first_and_a_loaded_prefix_outlives_an_untouched_one(self):
        linker = _budget_linker(16, 0)
        evicted = [page for batch in self._play(linker, drain=False) for page in batch]
        tree = linker.store_tree
        # Never a page whose stored child was kept: every survivor's parent survives.
        for page, node in tree.nodes.items():
            self.assertTrue(node[0] is None or node[0] in tree.nodes, page)
        # b (written before the load of a, never used since) goes before a.
        self.assertTrue({"b1", "b2", "b3"} <= set(evicted))
        self.assertTrue({"a1", "a2", "a3", "a4", "a5"} <= set(tree.nodes))
        self.assertEqual(evicted.index("b3"), 0)  # its tail first

    def test_budget_hysteresis_and_state_pages(self):
        linker = _budget_linker(10, 0, state_units=4)
        tree = linker.store_tree
        linker.offload([_pages("a1", "a2", "a3"), _state("a3")], path_hashes=lambda: ["a1", "a2", "a3"])
        self.assertEqual(tree.units, 3 + 4)
        self.assertEqual(tree.evict(), [])  # within the budget: nothing
        linker.offload([_pages("b1", "b2"), _state("b2")], path_hashes=lambda: ["b1", "b2"])
        # 13 units > 10: evict down to 95% (9), oldest leaf first (a3 with its state).
        self.assertLessEqual(tree.units, 9)
        linker.offload_queue.get_nowait(), linker.offload_queue.get_nowait()
        removal = linker.offload_queue.get_nowait()
        self.assertEqual(removal[:3], ("remove", ["a3_pp0_k", "a3_pp0_mamba"], ["a3", "a3"]))

    def test_a_page_stored_before_its_parent_keeps_the_parent(self):
        tree = mooncake_direct_linker._StoreTree(3, 4)
        tree.add(["c"], "b", ())  # b not stored yet
        tree.add(["b"], "a", ())
        tree.touch(["c"])  # c is now newer than b
        tree.add(["x", "y"], None, ())  # 4 units > 3
        # b still has a stored child: c goes first, then b, never b before c.
        self.assertEqual([page for page, _ in tree.evict()], ["c", "b"])

    def test_victims_unchanged_while_the_heap_holds_only_chain_ends(self):
        """earlyab8: one heap entry per added page let the lazy heap reach 4x
        the nodes and the O(nodes) rebuild stalled every stage (283 ms at 1.1M
        pages). Pushing only the pages that stay leaves picks the same victims."""
        import heapq
        import random

        class _EveryPage(mooncake_direct_linker._StoreTree):
            def add(self, pages, parent, states):  # the previous push: every page as it goes in
                self.tick += 1
                for page in pages:
                    node = self.nodes.get(page)
                    if node is None and self.recall:
                        node = self._recall(page)
                    if node is None:
                        node = self._insert(page, parent)
                    else:
                        node[2] = self.tick
                    self._push_if_leaf(page, node)
                    parent = page
                for page in states:
                    node = self.nodes.get(page)
                    if node is not None and not node[3]:
                        node[3] = True
                        self.units += self.state_units
                        self.states += 1

        for seed in range(5):
            rng, trees = random.Random(seed), []
            for cls in (mooncake_direct_linker._StoreTree, _EveryPage):
                trees.append(cls(400, 7, protect=3, recall=20))
            chains, victims, n = [], [[], []], 0
            for step in range(300):
                if chains and rng.random() < 0.4:
                    chain = rng.choice(chains)
                    used = chain[: rng.randint(1, len(chain))]
                    for tree in trees:
                        tree.touch(used)
                    continue
                length = rng.randint(1, 30)
                pages = [f"p{n + i}" for i in range(length)]
                n += length
                parent = rng.choice(chains)[-1] if chains and rng.random() < 0.5 else None
                states = [pages[-1]] if rng.random() < 0.5 else []
                for i, tree in enumerate(trees):
                    tree.add(pages, parent, states)
                    victims[i].append(tree.evict())
                chains.append(pages)
            self.assertEqual(victims[0], victims[1], seed)
            self.assertLess(len(trees[0].heap), len(trees[1].heap), seed)

    def test_off_by_default(self):
        linker = _budget_linker(16, 0)
        linker.store_tree = None
        linker.offload([_pages("a1")], path_hashes=lambda: ["a1"])
        self.assertEqual(linker.offload_queue.qsize(), 1)  # the put only


def _run_due_retries(linker):
    linker.page_remove_retries = deque(
        (0.0, *entry[1:]) for entry in linker.page_remove_retries
    )
    linker._remove_pages([])


class TestStoreBudgetKeepsReusedPages(CustomTestCase):
    """Rhea pm-lk3: the budget removed pages PP0 had admitted (pinned, not yet
    read on every stage); the removal waited out the pin's lease and the stage
    loads then failed with -704. A page used again after its eviction keeps its
    objects, and the tree takes it back alike on every stage."""

    def _evict_b(self, linker, drain):
        """Store a (4 pages) then b (3 pages) over a budget of 6: a3, a4 go."""
        linker.offload([_pages("a1", "a2", "a3", "a4")], path_hashes=lambda: ["a1", "a2", "a3", "a4"])
        linker.load("r-a", [_pages("a1", "a2")])  # a1, a2 used; a3, a4 not
        linker.offload([_pages("b1", "b2", "b3")], path_hashes=lambda: ["b1", "b2", "b3"])
        while not linker.offload_queue.empty():
            task = linker.offload_queue.get_nowait()
            if task[0] == "remove":
                self.assertEqual(sorted(set(task[2])), ["a3", "a4"])
                if drain:
                    linker._remove_pages(*task[1:])
                else:
                    linker.offload_queue.put(task)
                    break

    def test_a_load_after_eviction_drops_pending_and_retried_removals(self):
        # Stage 0 ran its removals and a lease refused a4 (pinned on PP0);
        # stage 1 has not run any yet. Then the request using a3/a4 is admitted.
        stage0 = _budget_linker(6, 0, store=_LeaseStore(leased={"a4_pp0_k"}), recall=64)
        stage1 = _budget_linker(6, 1, recall=64)
        self._evict_b(stage0, drain=True)
        self._evict_b(stage1, drain=False)
        for stage in (stage0, stage1):
            stage.load("r-a2", [_pages("a1", "a2", "a3", "a4")])
        self.assertEqual(stage0.store_tree.nodes, stage1.store_tree.nodes)
        self.assertEqual(stage0.store_tree.units, stage1.store_tree.units)
        self.assertIn("a4", stage0.store_tree.nodes)
        # Stage 0: a3 was gone already, the lease retry of a4 is dropped.
        _run_due_retries(stage0)
        self.assertEqual(stage0.storage.store.removed, ["a3_pp0_k"])
        self.assertEqual(stage0.page_remove_counts["recalled"], 1)
        # Stage 1: both queued removals are dropped before the call.
        task = stage1.offload_queue.get_nowait()
        stage1._remove_pages(*task[1:])
        self.assertEqual(stage1.storage.store.removed, [])
        self.assertEqual(stage1.page_remove_counts["recalled"], 2)

    def test_a_store_again_after_eviction_keeps_the_new_objects(self):
        linker = _budget_linker(6, 0, store=_LeaseStore(leased={"a4_pp0_k"}), recall=64)
        self._evict_b(linker, drain=True)
        # a4 is written again under a3 before its lease retry runs.
        linker.offload([_pages("a3", "a4")], path_hashes=lambda: ["a1", "a2", "a3", "a4"])
        linker.offload_queue.get_nowait()  # the put
        # Back over the budget: the re-stored page is the newest, b goes.
        self.assertEqual(linker.offload_queue.get_nowait()[2], ["b3", "b2"])
        _run_due_retries(linker)
        self.assertEqual(linker.storage.store.removed, ["a3_pp0_k"])
        tree = linker.store_tree
        self.assertEqual(tree.nodes["a4"][0], "a3")
        self.assertEqual(tree.nodes["a3"][1], 1)  # a4 hangs from a3 again
        # A later eviction of the page is a new decision and does run.
        tree.touch(["b1", "a1", "a2", "a3"])
        tree.add(["c1", "c2"], None, ())
        linker._queue_tail_eviction(tree.evict())
        task = linker.offload_queue.get_nowait()
        self.assertEqual(task[2][0], "a4")  # used least recently
        linker.storage.store.leased.add("a4_pp0_k")  # and its lease retry too
        linker._remove_pages(*task[1:])
        self.assertNotIn("a4_pp0_k", linker.storage.store.removed)
        _run_due_retries(linker)
        self.assertIn("a4_pp0_k", linker.storage.store.removed)

    def test_no_recall_past_the_window_or_when_off(self):
        for recall in (0, 2):
            linker = _budget_linker(6, 0, recall=recall)
            self._evict_b(linker, drain=False)
            for _ in range(3):
                linker.store_tree.touch(["b1"])
            linker.load("r-a2", [_pages("a1", "a2", "a3", "a4")])
            self.assertNotIn("a4", linker.store_tree.nodes)
            task = linker.offload_queue.get_nowait()
            linker._remove_pages(*task[1:])
            self.assertEqual(sorted(linker.storage.store.removed), ["a3_pp0_k", "a4_pp0_k"])

    def test_recent_pages_are_never_evicted(self):
        tree = mooncake_direct_linker._StoreTree(5, 4, protect=2)
        tree.add(["a1", "a2", "a3", "a4"], None, ())  # tick 1
        tree.touch(["a1", "a2"])  # tick 2
        tree.add(["b1", "b2", "b3"], None, ())  # tick 3: 7 units > 5
        # a3, a4 (tick 1) are past the window and go; a1, a2 (tick 2) stay
        # although the tree is still over 95% of the budget.
        self.assertEqual([page for page, _ in tree.evict()], ["a4", "a3"])
        self.assertEqual(tree.units, 5)
        self.assertEqual(tree.evict(), [])
        self.assertEqual(tree.counts["protected_stops"], 1)


class _RemoveAllStore(_LeaseStore):
    def __init__(self, passes=(7, 2, 0)):
        super().__init__()
        self.remove_all_calls = 0
        self.passes = list(passes)

    def remove_all(self):
        self.remove_all_calls += 1
        return self.passes.pop(0) if self.passes else 0


def _toggle_linker(stage=0):
    linker = _budget_linker(16, stage, store=_RemoveAllStore())
    linker.reset = lambda: None
    linker.pp_rank, linker.offload_owner = stage, True
    linker.lease_refresher = None
    linker.store_epoch = 0
    linker.storage.config_prefix = "tag_model"
    linker.state_parent, linker.state_children = {"s": None}, Counter({"s": 1})
    linker.state_remove_counts, linker.read_retries = Counter(a=1), Counter(b=1)
    linker.write_bytes, linker.lookup_counts = Counter(c=1), Counter(d=1)
    linker.flag_hits = Counter(vector_meta_calls=3)
    linker.state_remove_retries = deque([1])
    linker.lookup_samples, linker.pin_samples = [(1, 0.1)], [(1, 0.1)]
    linker.pin_holes = linker.state_removed = linker.state_remove_failed = 2
    linker.prepin_renewals = linker.pin_renewals = 2
    linker.stats = {"lookup": 5, "load": 5}
    return linker


class TestRuntimeLinkerFlags(CustomTestCase):
    """POST /linker_flags: per-call switches re-read without a restart, a cold
    reset of the linker's store state, and per-flag engagement counters."""

    def test_refresh_flags_switches_the_code_path(self):
        linker = _toggle_linker()
        with envs.SGLANG_LINKER_VECTOR_META.override(False), envs.SGLANG_LINKER_CPU_INDICES.override(False):
            linker.refresh_flags()
            self.assertFalse(linker.vector_meta)
            self.assertFalse(linker.cpu_indices)
        with envs.SGLANG_LINKER_VECTOR_META.override(True), envs.SGLANG_LINKER_CPU_INDICES.override(True), envs.SGLANG_LINKER_COALESCE_RANGES.override(True):
            linker.storage._vector_put_flag = False
            linker.refresh_flags()
            self.assertTrue(linker.vector_meta and linker.cpu_indices and linker.coalesce_ranges)
            # The store re-reads SGLANG_LINKER_VECTOR_PUT on its next put.
            self.assertIsNone(linker.storage._vector_put_flag)
            self.assertEqual(linker.flag_values()["SGLANG_LINKER_VECTOR_META"], "True")

    def test_refresh_flags_rereads_the_page_checksum_rate(self):
        linker = _toggle_linker()
        with envs.SGLANG_LINKER_PAGE_CHECKSUM.override(0.01):
            linker.refresh_flags()
            self.assertEqual(linker.page_checksum_rate, 0.01)
        with envs.SGLANG_LINKER_PAGE_CHECKSUM.override(0.0):
            linker.refresh_flags()
            self.assertEqual(linker.page_checksum_rate, 0.0)
        self.assertIn("SGLANG_LINKER_PAGE_CHECKSUM", linker.TOGGLE_FLAGS)

    def test_gil_tax_duty_cycle(self):
        from sglang.srt.utils.gil_tax import _calibrate, tax_sleep_s

        # Bursts take ``duty`` of each burst + sleep cycle.
        for duty in (0.03, 0.06, 0.5):
            sleep = tax_sleep_s(duty, 0.002)
            self.assertAlmostEqual(0.002 / (0.002 + sleep), duty)
        n = _calibrate(0.002)
        t = time.perf_counter()
        sum(range(n))
        self.assertLess(time.perf_counter() - t, 0.02)

    def test_reset_cold_clears_store_state_and_moves_the_namespace(self):
        linker = _toggle_linker(stage=0)
        linker.store_tree.add(["a1", "a2"], None, ())
        with patch.object(mooncake_direct_linker.time, "sleep"):
            linker.reset_cold()
        self.assertEqual(linker.store_tree.units, 0)
        self.assertEqual(linker.store_tree.budget, 16)
        for table in ("state_parent", "state_children", "state_remove_counts",
                      "read_retries", "write_bytes", "lookup_counts", "flag_hits"):
            self.assertFalse(getattr(linker, table), table)
        self.assertFalse(linker.state_remove_retries or linker.lookup_samples)
        self.assertEqual(linker.stats, {"lookup": 0, "load": 0})
        self.assertEqual(linker.storage.config_prefix, "tag_model-e1")
        # Passes repeat until one removes nothing (leased / in-flight objects survive a pass).
        self.assertEqual(linker.storage.store.remove_all_calls, 3)
        with patch.object(mooncake_direct_linker.time, "sleep"):
            linker.reset_cold()
        self.assertEqual(linker.storage.config_prefix, "tag_model-e2")
        self.assertEqual(linker.storage.store.remove_all_calls, 4)

    def test_reset_cold_remove_passes_stop_at_the_cap_or_an_error(self):
        linker = _toggle_linker(stage=0)
        linker.storage.store.passes = [5] * 40
        with patch.object(mooncake_direct_linker.time, "sleep") as sleep:
            linker.reset_cold()
        self.assertEqual(linker.storage.store.remove_all_calls, linker.RESET_REMOVE_PASSES)
        self.assertEqual(sleep.call_count, linker.RESET_REMOVE_PASSES)
        linker.storage.store.passes, linker.storage.store.remove_all_calls = [9, -1, 9], 0
        with patch.object(mooncake_direct_linker.time, "sleep"):
            linker.reset_cold()
        self.assertEqual(linker.storage.store.remove_all_calls, 2)

    def test_reset_cold_removes_store_objects_on_pp0_only(self):
        linker = _toggle_linker(stage=3)
        linker.reset_cold()
        self.assertEqual(linker.storage.store.remove_all_calls, 0)
        self.assertEqual(linker.storage.config_prefix, "tag_model-e1")

    def test_engagement_counters_are_logged(self):
        linker = _toggle_linker()
        linker.flag_hits = Counter(cpu_indices_attached=4)
        linker.storage.vector_put_batches = 2
        linker.stats_logged_at = 0.0
        with patch.object(mooncake_direct_linker.logger, "info") as info:
            try:
                linker._maybe_log_stats(every_s=0.0)
            except Exception:
                pass
        line = info.call_args_list[0]
        self.assertIn("flag engagement", line.args[0])
        self.assertEqual(line.args[1], {"cpu_indices_attached": 4, "vector_put_batches": 2})

    def test_scheduler_rejects_flags_outside_the_allowlist_and_when_busy(self):
        from sglang.srt.managers.io_struct import LinkerFlagsReqInput
        from sglang.srt.managers.scheduler import Scheduler

        calls = []
        linker = SimpleNamespace(
            reset_cold=lambda: calls.append("cold"),
            refresh_flags=lambda: calls.append("refresh"),
        )
        sched = SimpleNamespace(
            LINKER_TOGGLE_FLAGS=Scheduler.LINKER_TOGGLE_FLAGS,
            DEBUG_TOGGLE_FLAGS=Scheduler.DEBUG_TOGGLE_FLAGS,
            tree_cache=SimpleNamespace(linker=linker),
            is_fully_idle=lambda: True,
            flush_cache=lambda: calls.append("flush") or True,
        )
        bad = Scheduler.linker_flags_wrapped(
            sched, LinkerFlagsReqInput(flags={"SGLANG_LINKER_LOAD_THREADS": "4"})
        )
        self.assertFalse(bad.success)
        self.assertIn("SGLANG_LINKER_LOAD_THREADS", bad.message)
        self.assertEqual(calls, [])
        sched.is_fully_idle = lambda: False
        busy = Scheduler.linker_flags_wrapped(
            sched, LinkerFlagsReqInput(flags={"SGLANG_LINKER_VECTOR_META": "1"})
        )
        self.assertEqual((busy.success, busy.message), (False, "busy"))
        sched.is_fully_idle = lambda: True
        with envs.SGLANG_LINKER_VECTOR_META.override(False):
            with patch("sglang.srt.managers.scheduler.get_parallel",
                       return_value=SimpleNamespace(pp_rank=0)):
                ok = Scheduler.linker_flags_wrapped(
                    sched, LinkerFlagsReqInput(flags={"SGLANG_LINKER_VECTOR_META": "1"})
                )
            self.assertTrue(ok.success)
            self.assertEqual(ok.effective["SGLANG_LINKER_VECTOR_META"], "True")
        self.assertEqual(calls, ["flush", "cold", "refresh"])

    def test_scheduler_switches_debug_flags_without_a_linker(self):
        from sglang.srt.managers.io_struct import LinkerFlagsReqInput
        from sglang.srt.managers.scheduler import Scheduler

        calls = []
        sched = SimpleNamespace(
            LINKER_TOGGLE_FLAGS=Scheduler.LINKER_TOGGLE_FLAGS,
            DEBUG_TOGGLE_FLAGS=Scheduler.DEBUG_TOGGLE_FLAGS,
            tree_cache=SimpleNamespace(),
            is_fully_idle=lambda: True,
            flush_cache=lambda: calls.append("flush") or True,
        )
        refused = Scheduler.linker_flags_wrapped(
            sched, LinkerFlagsReqInput(flags={"SGLANG_LINKER_VECTOR_META": "1"})
        )
        self.assertEqual((refused.success, refused.message), (False, "no external linker"))
        with envs.SGLANG_DEBUG_GIL_TAX.override(-1.0):
            with patch("sglang.srt.managers.scheduler.get_parallel",
                       return_value=SimpleNamespace(pp_rank=0)):
                ok = Scheduler.linker_flags_wrapped(
                    sched, LinkerFlagsReqInput(flags={"SGLANG_DEBUG_GIL_TAX": "0.06"}, reset=True)
                )
            self.assertTrue(ok.success)
            self.assertEqual(ok.effective["SGLANG_DEBUG_GIL_TAX"], "0.06")
        self.assertEqual(calls, ["flush"])

    def test_wrapper_reset_cold_zeroes_stats_after_reset(self):
        from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
            UnifiedCacheLinkerWrapper,
        )

        calls = []
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.reset = lambda: calls.append("reset")
        wrapper.cache_linker = SimpleNamespace(reset_cold=lambda: calls.append("cold"))
        wrapper._lookup_stats = {"issued": 9, "requeried": 2}
        wrapper._hold_stats = {"held": 4}
        wrapper._phase_s = {"enq": {"n": 3.0, "rpc": 1.5}}
        wrapper._occupied_peak = 7
        wrapper._requery_waits = [1.0]
        wrapper.reset_cold()
        self.assertEqual(calls, ["reset", "cold"])
        self.assertEqual(wrapper._lookup_stats, {"issued": 0, "requeried": 0})
        self.assertEqual(wrapper._hold_stats, {"held": 0})
        self.assertEqual(wrapper._phase_s, {"enq": {"n": 0.0, "rpc": 0.0}})
        self.assertEqual((wrapper._occupied_peak, wrapper._requery_waits), (0, []))

if __name__ == "__main__":
    unittest.main()
