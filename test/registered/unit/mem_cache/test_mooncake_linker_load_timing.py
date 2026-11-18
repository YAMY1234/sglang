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


if __name__ == "__main__":
    unittest.main()
