from collections import deque
import sys
from concurrent.futures import Future
from typing import NamedTuple
import threading
import time
import unittest
from array import array
from collections import Counter, defaultdict
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.cache_action import (
    UpdatePendingBackupOnNodeSplit,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import (
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.mem_cache.unified_cache.pp_linker_relay import PPLinkerRelay
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LinkerAdmissionVerdicts,
    LINKER_ADMISSION_SKIPPED,
    ExternalCacheHitMarker,
    LayerWiseLoadCounter,
    UnifiedCacheLinker,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_disagg, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.kits.unified_radix_cache import fixtures as shared_cache_suite
from sglang.test.kits.unified_radix_cache.config import (
    CacheConfig,
)
from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
    node_children,
    node_device_lock_ref,
    node_device_value,
)
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=18, suite="base-a-test-cpu")
register_cuda_ci(est_time=18, stage="base-b", runner_config="1-gpu-small")


class _FakeLinker(UnifiedCacheLinker):
    def __init__(self):
        self.layer_done_counter = object()
        self.restorable = []
        self.queued_loads = {}
        self.queued_offloads = []
        self.offload_sessions = []
        self.load_landed = True
        self.completed_offloads = []
        self.reset_count = 0
        self.closed = False
        self.pools = {}

    def lookup(self, rid, transfers):
        return list(self.restorable)

    def load(self, rid, transfers):
        self.queued_loads[rid] = list(transfers)
        return True

    def start_layer_wise_loading(self):
        return 3

    def cancel_queued_load(self, rid):
        if rid not in self.queued_loads:
            return False
        del self.queued_loads[rid]
        return True

    def finish_layer_wise_loading(self, counter_index):
        return self.load_landed

    def offload(self, transfers, session_id=None):
        self.queued_offloads.append(list(transfers))
        self.offload_sessions.append(session_id)
        return True

    def num_completed_offloads(self):
        return len(self.completed_offloads)

    def pop_completed_offload(self):
        return self.completed_offloads.pop(0)

    def reset(self):
        self.reset_count += 1

    def close(self):
        self.closed = True


class _MappingRecorder:
    def __init__(self):
        self.mapping = []

    def set_full_to_swa_mapping(self, full, swa):
        self.mapping.append((full.clone(), swa.clone()))


class _FakeExternalTreeCore:
    def __init__(self, nodes=None, offload_transfers=None):
        self.enable_external_cache_linker = False
        self.nodes = nodes or {}
        self.offload_transfers = offload_transfers or [
            PoolTransfer(name=PoolName.KV, keys=["page"])
        ]

    def build_external_linker_offload_transfers(self, node_id):
        node = self.nodes[node_id]
        if node.external_cache_stored or node.write_through_pending_id is not None:
            return None
        return list(self.offload_transfers)

    def mark_external_linker_offload_pending(self, node_id):
        node = self.nodes[node_id]
        assert (
            not node.external_cache_stored and node.write_through_pending_id is None
        ), "invalid external offload state"
        node.write_through_pending_id = node_id

    def finish_external_linker_offload(self, node_ids, ack_id, success):
        nodes = [self.nodes[node_id] for node_id in node_ids]
        assert all(node.write_through_pending_id == ack_id for node in nodes), (
            "invalid external offload state"
        )
        for node in nodes:
            node.write_through_pending_id = None
            node.external_cache_stored |= success


def _cache_for_wrapper(**kwargs):
    defaults = {
        "_components_tuple": (),
        "components": {},
        "tree_core": SimpleNamespace(enable_external_cache_linker=False),
        "tree_components": (ComponentType.FULL,),
        "write_through_threshold": 256,
        "pp_size": 1,
        "pp_group": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _swa_allocator(swa_req_ring):
    if swa_req_ring is None:
        return SimpleNamespace()
    allocator = SWATokenToKVPoolAllocator.__new__(SWATokenToKVPoolAllocator)
    allocator._swa_req_ring = swa_req_ring
    return allocator


def test_cache_linker_attachment_is_backend_independent():
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.tree_core = SimpleNamespace(
        enable_external_cache_linker=False,
        write_through_threshold=256,
    )
    cache.tree_components = (ComponentType.FULL,)
    cache.linker = None
    cache._components_tuple = ()
    cache.components = {}
    linker = _FakeLinker()

    cache.init_cache_linker(linker)

    assert cache.linker.cache_linker is linker
    assert cache.tree_core.enable_external_cache_linker
    assert cache.write_through_threshold == 1
    assert cache.linker.layer_done_counter is linker.layer_done_counter


@pytest.mark.parametrize("component_type", [ComponentType.C128])
def test_cache_linker_rejects_unsupported_tree_components(component_type):
    cache = _cache_for_wrapper(tree_components=(ComponentType.FULL, component_type))

    with pytest.raises(ValueError, match=component_type.name):
        UnifiedCacheLinkerWrapper(cache, _FakeLinker())

    assert not cache.tree_core.enable_external_cache_linker


class _InMemoryUnifiedCacheLinker(UnifiedCacheLinker):
    """Controllable transport for shared Python/Rust TreeCore tests."""

    def __init__(self, stored_keys=None):
        self.layer_done_counter = object()
        self.pools = dict.fromkeys(PoolName)
        self.stored_keys = defaultdict(set) if stored_keys is None else stored_keys
        self.lookup_calls = []
        self.offload_calls = []
        self.pending_offloads = []
        self.queued_loads = {}
        self.started_loads = []
        self.load_landed = True
        self.completed_offloads = []

    @staticmethod
    def _clone_transfer(transfer):
        return replace(
            transfer,
            host_indices=(
                None if transfer.host_indices is None else transfer.host_indices.clone()
            ),
            device_indices=(
                None
                if transfer.device_indices is None
                else transfer.device_indices.clone()
            ),
            keys=None if transfer.keys is None else list(transfer.keys),
            nodes_to_load=(
                None if transfer.nodes_to_load is None else list(transfer.nodes_to_load)
            ),
        )

    @classmethod
    def _clone_transfers(cls, transfers):
        return [cls._clone_transfer(transfer) for transfer in transfers]

    def lookup(self, rid, transfers):
        transfers = self._clone_transfers(transfers)
        self.lookup_calls.append((rid, transfers))
        by_pool = {transfer.name: transfer for transfer in transfers}
        kv = by_pool.get(PoolName.KV)
        if kv is None or not kv.keys:
            return []

        restorable = []
        for prefix_pages in range(1, len(kv.keys) + 1):
            if not set(kv.keys[:prefix_pages]) <= self.stored_keys[PoolName.KV]:
                continue
            for transfer in transfers:
                if transfer.name == PoolName.KV:
                    continue
                if transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                    window_pages = max(1, len(transfer.keys or ()))
                    required = kv.keys[
                        max(0, prefix_pages - window_pages) : prefix_pages
                    ]
                else:
                    required = kv.keys[:prefix_pages]
                if not set(required) <= self.stored_keys[transfer.name]:
                    break
            else:
                restorable.append(prefix_pages)
        return restorable

    def load(self, rid, transfers):
        self.queued_loads[rid] = self._clone_transfers(transfers)
        return True

    def start_layer_wise_loading(self):
        if not self.queued_loads:
            return -1
        rids = list(self.queued_loads)
        self.started_loads.append(rids)
        return len(self.started_loads) - 1

    def cancel_queued_load(self, rid):
        return self.queued_loads.pop(rid, None) is not None

    def finish_layer_wise_loading(self, counter_index):
        for rid in self.started_loads[counter_index]:
            self.queued_loads.pop(rid, None)
        return self.load_landed

    def offload(self, transfers, session_id=None):
        transfers = self._clone_transfers(transfers)
        self.offload_calls.append(transfers)
        self.pending_offloads.append(transfers)
        return True

    def num_completed_offloads(self):
        return len(self.completed_offloads)

    def pop_completed_offload(self):
        return self.completed_offloads.pop(0)

    def complete_next_offload(self, success):
        transfers = self.pending_offloads.pop(0)
        if success:
            for transfer in transfers:
                self.stored_keys[transfer.name].update(transfer.keys or ())
        self.completed_offloads.append(success)

    def reset(self):
        self.queued_loads.clear()
        self.pending_offloads.clear()
        self.completed_offloads.clear()

    def close(self):
        self.reset()


class _TreeCoreBackendTestMixin:
    tree_core_backend = "python"

    def setUp(self):
        previous = shared_cache_suite.TREE_CORE_TEST_BACKEND
        self.addCleanup(
            setattr,
            shared_cache_suite,
            "TREE_CORE_TEST_BACKEND",
            previous,
        )
        shared_cache_suite.TREE_CORE_TEST_BACKEND = self.tree_core_backend
        super().setUp()


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestUnifiedCacheLinkerPythonBackend(
    _TreeCoreBackendTestMixin, UnifiedRadixCacheFixtures, CustomTestCase
):
    def _bind_loaded_request(self, cache, req_to_token_pool, req, tokens, loaded):
        """Leave ``req`` as admission does: loaded tail written, owned by ``req``."""
        req.kv.cache_protected_len = req.prefix_len
        req.prefix_len += len(loaded)
        prefix = cache.prefix_device_indices(req)
        self.assertTrue(torch.equal(prefix[-len(loaded) :], loaded))
        req_to_token_pool.write((req.kv.req_pool_idx, slice(0, len(prefix))), prefix)
        req.origin_input_ids, req.output_ids = array("q", tokens), array("q")
        req.full_untruncated_fill_ids = array("q", tokens)
        req.extend_end = len(tokens)
        req.kv.kv_committed_len = len(prefix)
        req.kv.kv_allocated_len = len(prefix)
        req.extra_key = None
        req.lock = cache.lock(req.last_node)

    def _pp_stages_loading(self, tokens, landed, insert_at_release):
        """One PP stage per ``landed`` entry, each admitting the same store hit
        of ``tokens`` and finishing its load as given."""
        cfg = CacheConfig(page_size=2, kv_size=64, max_context_len=64)
        self.cfg = cfg
        stored_keys = defaultdict(set)
        producer, allocator, req_pool = build_fixture(cfg)
        producer_linker = _InMemoryUnifiedCacheLinker(stored_keys)
        producer.init_cache_linker(producer_linker)
        self._insert(producer, allocator, req_pool, tokens)
        producer_linker.complete_next_offload(True)
        producer.check_hicache_events()
        stages = []
        for stage_landed in landed:
            cache, _, req_pool = build_fixture(cfg)
            linker = _InMemoryUnifiedCacheLinker(stored_keys)
            cache.init_cache_linker(linker)
            cache.linker.insert_at_release = insert_at_release
            req = self._make_req(req_pool)
            match = cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", tokens)), req=req)
            )
            self._apply_match_to_req(req, match)
            cache.init_load_back(
                InitLoadBackParams(
                    best_match_node=match.best_match_node,
                    host_hit_length=match.host_hit_length,
                    req=req,
                )
            )
            self.assertGreaterEqual(cache.ready_to_load_host_cache(), 0)
            loaded = cache.linker.private_load_indices[req.rid]
            self._bind_loaded_request(cache, req_pool, req, tokens, loaded)
            linker.load_landed = stage_landed
            cache.pp_size = 2
            req.disagg_kv_sender = MagicMock()
            stages.append((cache, req))
        return stages

    def _device_prefix_len(self, cache, tokens):
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        return match.device_prefix_len

    def test_pp_one_stage_load_failure_splits_the_stage_verdicts(self):
        # Hecate H112: only the failed stage aborted the request, so the PP
        # stages' inflight queues differed and the worker crashed.
        stages = self._pp_stages_loading(list(range(1, 9)), (True, False), False)
        verdicts = [cache.finish_external_linker_loads([req]) for cache, req in stages]
        self.assertEqual(verdicts, [[], [stages[1][1]]])

    def test_pp_load_consensus_fails_a_one_stage_failure_on_every_stage(self):
        from sglang.srt.disaggregation.prefill import (
            SchedulerDisaggregationPrefillMixin as Prefill,
        )

        tokens = list(range(1, 9))
        stages = self._pp_stages_loading(tokens, (True, False), True)
        with envs.SGLANG_LINKER_PP_LOAD_CONSENSUS.override(True):
            verdicts = [
                cache.finish_external_linker_loads([req]) for cache, req in stages
            ]
        self.assertEqual(verdicts, [[], []])  # no stage aborts alone
        # The failed stage fails its send: PP0's release poll takes the minimum.
        stages[0][1].disagg_kv_sender.abort.assert_not_called()
        stages[1][1].disagg_kv_sender.abort.assert_called_once()
        for cache, req in stages:  # every stage concludes the Failed release
            req.disagg_kv_sender.failure_exception.side_effect = RuntimeError()
            scheduler = SimpleNamespace(
                tree_cache=cache,
                _release_aborted_request=MagicMock(),
                metrics_reporter=SimpleNamespace(enable_metrics=False),
            )
            with get_parallel().override(tp_rank=0):
                Prefill.handle_inflight_transfer_failure(scheduler, req)
            self.assertFalse(req.kv.holds_kv)
            self.assertEqual(self._device_prefix_len(cache, tokens), 0)
            cache.sanity_check()

    def test_pp_load_consensus_inserts_a_loading_chunk_only_at_release(self):
        from sglang.srt.disaggregation.prefill import (
            SchedulerDisaggregationPrefillMixin as Prefill,
        )

        tokens = list(range(1, 9))
        ((cache, req),) = self._pp_stages_loading(tokens, (True,), True)
        self.assertTrue(req.linker_insert_at_release)
        self.assertEqual(cache.finish_external_linker_loads([req]), [])
        Prefill.checkpoint_disagg_prefill(SimpleNamespace(tree_cache=cache), req)
        self.assertEqual(req.prefix_len, len(tokens))
        self.assertEqual(self._device_prefix_len(cache, tokens), 0)
        release_kv_cache(req, cache, checkpoint=True)  # release: Success
        self.assertEqual(self._device_prefix_len(cache, tokens), len(tokens))
        cache.sanity_check()

    def test_full_offload_load_round_trip_and_dedup(self):
        cfg = CacheConfig(page_size=2, kv_size=64, max_context_len=64)
        self.cfg = cfg
        stored_keys = defaultdict(set)

        producer, producer_allocator, producer_req_pool = build_fixture(cfg)
        producer_linker = _InMemoryUnifiedCacheLinker(stored_keys)
        producer.init_cache_linker(producer_linker)
        tokens = list(range(1, 9))

        inserted = self._insert(producer, producer_allocator, producer_req_pool, tokens)
        self.assertEqual(len(producer_linker.offload_calls), 1)
        (kv_offload,) = producer_linker.offload_calls[0]
        self.assertEqual(kv_offload.name, PoolName.KV)
        self.assertEqual(
            kv_offload.keys,
            producer.tree_core.get_hash_values(inserted.last_device_node),
        )
        self.assertTrue(
            torch.equal(
                kv_offload.device_indices,
                node_device_value(
                    producer, inserted.last_device_node, ComponentType.FULL
                ),
            )
        )
        self.assertEqual(
            node_device_lock_ref(
                producer, inserted.last_device_node, ComponentType.FULL
            ),
            1,
        )
        self.assertFalse(
            producer.tree_core.is_external_cache_stored(inserted.last_device_node)
        )
        self.assertEqual(
            producer.tree_core.get_write_through_pending_id(inserted.last_device_node),
            inserted.last_device_node,
        )

        self._insert(producer, producer_allocator, producer_req_pool, tokens)
        self.assertEqual(len(producer_linker.offload_calls), 1)

        producer_linker.complete_next_offload(True)
        producer.check_hicache_events()
        self.assertEqual(
            node_device_lock_ref(
                producer, inserted.last_device_node, ComponentType.FULL
            ),
            0,
        )
        self.assertTrue(
            producer.tree_core.is_external_cache_stored(inserted.last_device_node)
        )
        self._insert(producer, producer_allocator, producer_req_pool, tokens)
        self.assertEqual(len(producer_linker.offload_calls), 1)

        consumer, _, consumer_req_pool = build_fixture(cfg)
        consumer_linker = _InMemoryUnifiedCacheLinker(stored_keys)
        consumer.init_cache_linker(consumer_linker)
        req = self._make_req(consumer_req_pool)
        match = consumer.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)), req=req)
        )
        self.assertEqual(match.device_prefix_len, 0)
        self.assertEqual(match.host_hit_length, len(tokens))
        self._apply_match_to_req(req, match)

        loaded_len, loaded_node = consumer.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=match.host_hit_length,
                req=req,
            )
        )
        self.assertEqual(loaded_len, len(tokens))
        loaded = consumer.linker.private_load_indices[req.rid]
        # The tail is request-private: nothing is published until it lands.
        self.assertEqual(loaded_node, req.last_node)
        (kv_load,) = consumer_linker.queued_loads[req.rid]
        self.assertEqual(kv_load.name, PoolName.KV)
        self.assertEqual(kv_load.keys, kv_offload.keys)
        self.assertTrue(torch.equal(kv_load.device_indices, loaded))

        self.assertGreaterEqual(consumer.ready_to_load_host_cache(), 0)
        self._bind_loaded_request(consumer, consumer_req_pool, req, tokens, loaded)
        self.assertEqual(consumer.finish_external_linker_loads([req]), [])
        consumer.checkpoint(req, up_to=req.extend_end)
        consumer.unlock(req.lock)
        req.lock = None
        final_match = consumer.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        )
        self.assertTrue(
            torch.equal(
                consumer.path_device_indices(final_match.last_device_node), loaded
            )
        )
        consumer_linker.complete_next_offload(True)
        consumer.check_hicache_events()
        consumer.sanity_check()

    def test_eagle_lookup_uses_bigram_tail_hashes(self):
        cfg = CacheConfig(
            page_size=2,
            is_eagle=True,
            kv_size=64,
            max_context_len=64,
        )
        self.cfg = cfg
        stored_keys = defaultdict(set)
        tokens = list(range(1, 10))

        producer, producer_allocator, producer_req_pool = build_fixture(cfg)
        producer_linker = _InMemoryUnifiedCacheLinker(stored_keys)
        producer.init_cache_linker(producer_linker)
        self._insert(producer, producer_allocator, producer_req_pool, tokens)
        self.assertEqual(len(producer_linker.offload_calls), 1)
        (kv_offload,) = producer_linker.offload_calls[0]
        self.assertEqual(len(kv_offload.keys), 4)
        producer_linker.complete_next_offload(True)
        producer.check_hicache_events()

        consumer, consumer_allocator, consumer_req_pool = build_fixture(cfg)
        consumer_linker = _InMemoryUnifiedCacheLinker(stored_keys)
        consumer.init_cache_linker(consumer_linker)
        consumer.write_through_threshold = sys.maxsize
        self._insert(
            consumer,
            consumer_allocator,
            consumer_req_pool,
            tokens[:5],
        )
        req = self._make_req(consumer_req_pool)
        lookup_key = RadixKey(array("q", tokens))

        match = consumer.match_prefix(MatchPrefixParams(key=lookup_key, req=req))

        self.assertTrue(lookup_key.is_bigram)
        self.assertEqual(match.device_prefix_len, 4)
        self.assertEqual(match.host_hit_length, 4)
        self.assertEqual(len(consumer_linker.lookup_calls), 1)
        _, transfers = consumer_linker.lookup_calls[0]
        (kv_lookup,) = transfers
        self.assertEqual(kv_lookup.keys, kv_offload.keys[2:])

    def test_decode_match_skips_linker_lookup(self):
        cfg = CacheConfig(page_size=2, kv_size=64, max_context_len=64)
        self.cfg = cfg
        stored_keys = defaultdict(set)
        producer, producer_allocator, producer_req_pool = build_fixture(cfg)
        producer_linker = _InMemoryUnifiedCacheLinker(stored_keys)
        producer.init_cache_linker(producer_linker)
        tokens = list(range(1, 9))
        self._insert(producer, producer_allocator, producer_req_pool, tokens)
        producer_linker.complete_next_offload(True)
        producer.check_hicache_events()

        consumer, _, consumer_req_pool = build_fixture(cfg)
        consumer_linker = _InMemoryUnifiedCacheLinker(stored_keys)
        consumer.init_cache_linker(consumer_linker)
        key = RadixKey(array("q", tokens))
        with get_disagg().override(disaggregation_mode="decode"):
            req = self._make_req(consumer_req_pool)
            match = consumer.match_prefix(MatchPrefixParams(key=key, req=req))
        self.assertEqual(match.host_hit_length, 0)
        self.assertEqual(consumer_linker.lookup_calls, [])
        self.assertFalse(consumer.linker.has_hit(req.rid))

        req = self._make_req(consumer_req_pool)
        match = consumer.match_prefix(MatchPrefixParams(key=key, req=req))
        self.assertEqual(match.host_hit_length, len(tokens))
        consumer.linker.release_request(req.rid)

    def test_failed_split_offload_retries_and_reset_releases_locks(self):
        cfg = CacheConfig(page_size=1, kv_size=64, max_context_len=64)
        self.cfg = cfg
        cache, allocator, req_to_token_pool = build_fixture(cfg)
        linker = _InMemoryUnifiedCacheLinker()
        cache.init_cache_linker(linker)
        tokens = [1, 2, 3, 4]

        inserted = self._insert(cache, allocator, req_to_token_pool, tokens)
        original_node = inserted.last_device_node
        self.assertEqual(len(linker.offload_calls), 1)
        original_keys = linker.offload_calls[0][0].keys

        self._insert(cache, allocator, req_to_token_pool, tokens[:2])
        (parent,) = node_children(cache, cache.root_node_handle())
        (child,) = node_children(cache, parent)
        self.assertEqual(child, original_node)
        self.assertEqual(
            cache.linker.pending_offloads[0].publish_node_ids, [parent, child]
        )
        self.assertEqual(cache.tree_core.get_write_through_pending_id(parent), child)
        self.assertEqual(cache.tree_core.get_write_through_pending_id(child), child)
        self.assertFalse(cache.tree_core.is_external_cache_stored(parent))
        self.assertFalse(cache.tree_core.is_external_cache_stored(child))

        linker.complete_next_offload(False)
        cache.check_hicache_events()
        for node_id in (parent, child):
            self.assertIsNone(cache.tree_core.get_write_through_pending_id(node_id))
            self.assertFalse(cache.tree_core.is_external_cache_stored(node_id))
            self.assertEqual(
                node_device_lock_ref(cache, node_id, ComponentType.FULL), 0
            )

        self._insert(cache, allocator, req_to_token_pool, tokens)
        retry_calls = linker.offload_calls[1:]
        self.assertEqual(len(retry_calls), 2)
        self.assertEqual(
            [key for transfers in retry_calls for key in transfers[0].keys],
            original_keys,
        )
        for _ in retry_calls:
            linker.complete_next_offload(True)
        cache.check_hicache_events()
        for node_id in (parent, child):
            self.assertIsNone(cache.tree_core.get_write_through_pending_id(node_id))
            self.assertTrue(cache.tree_core.is_external_cache_stored(node_id))
            self.assertEqual(
                node_device_lock_ref(cache, node_id, ComponentType.FULL), 0
            )

        self._insert(cache, allocator, req_to_token_pool, tokens)
        self.assertEqual(len(linker.offload_calls), 3)

        extended = self._insert(cache, allocator, req_to_token_pool, tokens + [5, 6])
        pending_node = extended.last_device_node
        self.assertEqual(len(cache.linker.pending_offloads), 1)
        self.assertEqual(
            cache.tree_core.get_write_through_pending_id(pending_node), pending_node
        )
        self.assertEqual(
            node_device_lock_ref(cache, pending_node, ComponentType.FULL), 1
        )

        cache.linker.reset()
        self.assertEqual(cache.linker.pending_offloads, [])
        self.assertIsNone(cache.tree_core.get_write_through_pending_id(pending_node))
        self.assertEqual(
            node_device_lock_ref(cache, pending_node, ComponentType.FULL), 0
        )
        cache.sanity_check()
        cache.reset()
        cache.sanity_check()

    def test_swa_hit_loads_private_tail_and_dedups_on_insert(self):
        self._run_swa_hit(load_landed=True)

    def test_swa_failed_load_frees_private_slots_unpublished(self):
        self._run_swa_hit(load_landed=False)

    def _run_swa_hit(self, load_landed):
        cfg = CacheConfig(
            page_size=1,
            components=(ComponentType.FULL, ComponentType.SWA),
            sliding_window_size=2,
            kv_size=64,
            max_context_len=64,
        )
        self.cfg = cfg
        stored_keys = defaultdict(set)
        tokens = list(range(1, 7))

        producer, producer_allocator, producer_req_pool = build_fixture(cfg)
        producer_linker = _InMemoryUnifiedCacheLinker(stored_keys)
        producer.init_cache_linker(producer_linker)
        self._insert(producer, producer_allocator, producer_req_pool, tokens)
        self.assertGreaterEqual(len(producer_linker.offload_calls), 1)
        all_keys = []
        for transfers in producer_linker.offload_calls:
            offloads = {transfer.name: transfer for transfer in transfers}
            self.assertEqual(set(offloads), {PoolName.KV, PoolName.SWA})
            self.assertEqual(offloads[PoolName.KV].keys, offloads[PoolName.SWA].keys)
            all_keys.extend(offloads[PoolName.KV].keys)
            producer_linker.complete_next_offload(True)
        producer.check_hicache_events()

        stored_keys[PoolName.KV].difference_update(all_keys[-2:])

        consumer, consumer_allocator, consumer_req_pool = build_fixture(cfg)
        consumer_linker = _InMemoryUnifiedCacheLinker(stored_keys)
        consumer.init_cache_linker(consumer_linker)
        consumer.write_through_threshold = sys.maxsize
        self._insert(consumer, consumer_allocator, consumer_req_pool, tokens[:2])

        req = self._make_req(consumer_req_pool)
        match = consumer.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)), req=req)
        )
        self.assertEqual(match.device_prefix_len, 2)
        self.assertEqual(match.host_hit_length, 2)
        self.assertEqual(match.swa_host_hit_length, 2)
        self._apply_match_to_req(req, match)

        self._insert(consumer, consumer_allocator, consumer_req_pool, tokens[:3])
        raced_match = consumer.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens[:3])))
        )
        raced_full = consumer.path_device_indices(raced_match.last_device_node)[
            -1:
        ].clone()

        loaded_len, loaded_node = consumer.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=match.host_hit_length,
                req=req,
            )
        )
        self.assertEqual(loaded_len, 2)
        loaded = consumer.linker.private_load_indices[req.rid]
        load_by_pool = {
            transfer.name: transfer
            for transfer in consumer_linker.queued_loads[req.rid]
        }
        self.assertEqual(set(load_by_pool), {PoolName.KV, PoolName.SWA})
        for transfer in load_by_pool.values():
            self.assertEqual(transfer.keys, all_keys[2:4])
        translated = consumer_allocator.translate_loc_from_full_to_swa(loaded)
        self.assertTrue(
            torch.equal(translated, load_by_pool[PoolName.SWA].device_indices)
        )

        full_free = consumer_allocator.full_attn_allocator.available_size
        swa_free = consumer_allocator.swa_attn_allocator.available_size
        before = (full_free(), swa_free())
        self.assertGreaterEqual(consumer.ready_to_load_host_cache(), 0)
        self._bind_loaded_request(consumer, consumer_req_pool, req, tokens[:4], loaded)
        if load_landed:
            self.assertEqual(consumer.finish_external_linker_loads([req]), [])
            consumer.checkpoint(req, up_to=req.extend_end)
            consumer.unlock(req.lock)
            req.lock = None
            final = consumer.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", tokens[:4])))
            )
            # The raced page stays; the request's duplicate of it is freed.
            self.assertTrue(
                torch.equal(
                    consumer.path_device_indices(final.last_device_node)[2:3],
                    raced_full,
                )
            )
            self.assertTrue(
                torch.equal(
                    consumer.path_device_indices(final.last_device_node)[3:], loaded[1:]
                )
            )
        else:
            consumer_linker.load_landed = False
            self.assertEqual(consumer.finish_external_linker_loads([req]), [req])
            self.assertEqual(req.discard_output_reason.status_code, 503)
            self.assertTrue(req.skip_radix_cache_insert)
            release_kv_cache(req, consumer, checkpoint=False)
            self.assertFalse(req.kv.holds_kv)
            self.assertTrue(req.kv.is_kv_released)
            self.assertIsNone(req.lock)
            for component in (ComponentType.FULL, ComponentType.SWA):
                self.assertEqual(
                    node_device_lock_ref(consumer, loaded_node, component), 0
                )
            self.assertEqual((full_free(), swa_free()), (before[0] + 2, before[1] + 2))
            unpublished = consumer.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", tokens[:4])))
            )
            self.assertEqual(unpublished.device_prefix_len, 3)
        consumer.sanity_check()

    def _mamba_linker_consumer(self, tokens):
        self.cfg = CacheConfig(
            page_size=1,
            components=(ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA),
            sliding_window_size=2,
            kv_size=64,
            max_context_len=64,
        )
        stored_keys = defaultdict(set)

        producer, producer_allocator, producer_req_pool = build_fixture(self.cfg)
        producer_linker = _InMemoryUnifiedCacheLinker(stored_keys)
        producer.init_cache_linker(producer_linker)
        inserted = self._insert(producer, producer_allocator, producer_req_pool, tokens)
        # Ancestors of the chain carry no state; only the checkpoint node does.
        *ancestors, offload = producer_linker.offload_calls
        for transfers in ancestors:
            self.assertEqual(
                {transfer.name for transfer in transfers}, {PoolName.KV, PoolName.SWA}
            )
        offloads = {transfer.name: transfer for transfer in offload}
        self.assertEqual(set(offloads), {PoolName.KV, PoolName.SWA, PoolName.MAMBA})
        mamba_offload = offloads[PoolName.MAMBA]
        self.assertEqual(mamba_offload.keys, offloads[PoolName.KV].keys[-1:])
        self.assertEqual(mamba_offload.hit_policy, PoolHitPolicy.TRAILING_PAGES)
        self.assertTrue(
            torch.equal(
                mamba_offload.device_indices,
                node_device_value(
                    producer, inserted.last_device_node, ComponentType.MAMBA
                ),
            )
        )
        for _ in producer_linker.offload_calls:
            producer_linker.complete_next_offload(True)
        producer.check_hicache_events()

        consumer, _, consumer_req_pool = build_fixture(self.cfg)
        consumer_linker = _InMemoryUnifiedCacheLinker(stored_keys)
        consumer.init_cache_linker(consumer_linker)

        return consumer, consumer_req_pool, consumer_linker

    def test_mamba_state_round_trips_only_at_its_checkpoint(self):
        tokens = list(range(1, 7))
        consumer, consumer_req_pool, consumer_linker = self._mamba_linker_consumer(
            tokens
        )

        # Every KV and SWA page is stored, but only the full prefix has a state.
        short = consumer.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", tokens[:5])),
                req=self._make_req(consumer_req_pool),
            )
        )
        self.assertEqual(short.host_hit_length, 0)

        req = self._make_req(consumer_req_pool)
        match = consumer.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)), req=req)
        )
        self.assertEqual(match.host_hit_length, len(tokens))
        self.assertEqual(match.mamba_host_hit_length, 1)
        self._apply_match_to_req(req, match)
        loaded, _ = consumer.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=match.host_hit_length,
                req=req,
            )
        )
        self.assertEqual(loaded, len(tokens))
        loads = {
            transfer.name: transfer
            for transfer in consumer_linker.queued_loads[req.rid]
        }
        self.assertEqual(set(loads), {PoolName.KV, PoolName.SWA, PoolName.MAMBA})
        self.assertEqual(loads[PoolName.MAMBA].keys, loads[PoolName.KV].keys[-1:])
        loaded_state = loads[PoolName.MAMBA].device_indices
        self.assertEqual(int(req.kv.mamba_pool_idx), int(loaded_state))
        self.assertFalse(req.kv.mamba_needs_clear)

        loaded_indices = consumer.linker.private_load_indices[req.rid]
        self.assertGreaterEqual(consumer.ready_to_load_host_cache(), 0)
        self._bind_loaded_request(
            consumer, consumer_req_pool, req, tokens, loaded_indices
        )
        self.assertEqual(consumer.finish_external_linker_loads([req]), [])
        consumer.checkpoint(req, up_to=req.extend_end)
        consumer.unlock(req.lock)
        req.lock = None
        final_match = consumer.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        )
        self.assertEqual(final_match.device_prefix_len, len(tokens))
        for _ in consumer_linker.offload_calls:
            consumer_linker.complete_next_offload(True)
        consumer.check_hicache_events()
        consumer.sanity_check()

    def test_mamba_load_supersedes_a_shorter_device_cow(self):
        tokens = list(range(1, 7))
        consumer, consumer_req_pool, consumer_linker = self._mamba_linker_consumer(
            tokens
        )
        self._insert(
            consumer,
            consumer.token_to_kv_pool_allocator,
            consumer_req_pool,
            tokens[:3],
        )
        req = self._make_req(consumer_req_pool)
        match = consumer.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)), req=req, cow_mamba=True)
        )
        self.assertEqual(match.device_prefix_len, 3)
        self.assertEqual(match.host_hit_length, 3)
        self.assertIsNotNone(req.kv.mamba_cow_src_index)
        self._apply_match_to_req(req, match)
        allocator = consumer_req_pool.mamba_allocator
        before = allocator.available_size()

        consumer.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=match.host_hit_length,
                req=req,
            )
        )
        # The CoW from the shorter checkpoint would overwrite the loaded state.
        loaded_state = next(
            transfer.device_indices
            for transfer in consumer_linker.queued_loads[req.rid]
            if transfer.name == PoolName.MAMBA
        )
        self.assertEqual(int(req.kv.mamba_pool_idx), int(loaded_state))
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertEqual(allocator.available_size(), before)

        self.assertGreaterEqual(consumer.ready_to_load_host_cache(), 0)
        self.assertEqual(consumer.finish_external_linker_loads([req]), [])
        consumer.sanity_check()


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestUnifiedCacheLinkerTreeCorePythonBackend(
    _TreeCoreBackendTestMixin,
    UnifiedRadixCacheFixtures,
    CustomTestCase,
):
    """TreeCore linker contracts shared by the Python and Rust inspectors."""

    def test_builds_opaque_external_offload_transfers(self):
        cfg = CacheConfig(page_size=2, kv_size=64, max_context_len=64)
        self.cfg = cfg
        cache, allocator, req_to_token_pool = build_fixture(cfg)
        core = cache.tree_core
        core.enable_external_cache_linker = True

        inserted = self._insert(cache, allocator, req_to_token_pool, list(range(1, 9)))
        node_id = inserted.last_device_node
        transfers = core.build_external_linker_offload_transfers(node_id)

        self.assertIsNotNone(transfers)
        (transfer,) = transfers
        self.assertEqual(transfer.name, PoolName.KV)
        self.assertIsNone(transfer.host_indices)
        self.assertIsNotNone(transfer.device_indices)
        self.assertEqual(transfer.keys, core.get_hash_values(node_id))
        self.assertTrue(
            torch.equal(
                transfer.device_indices,
                node_device_value(cache, node_id, ComponentType.FULL),
            )
        )

        core.mark_external_linker_offload_pending(node_id)
        self.assertFalse(core.is_external_cache_stored(node_id))
        self.assertIsNone(core.build_external_linker_offload_transfers(node_id))
        with self.assertRaisesRegex(AssertionError, "invalid external offload state"):
            core.mark_external_linker_offload_pending(node_id)
        self.assertEqual(core.get_write_through_pending_id(node_id), node_id)
        self.assertFalse(core.is_external_cache_stored(node_id))

        core.finish_external_linker_offload([node_id], node_id, success=True)
        self.assertIsNone(core.get_write_through_pending_id(node_id))
        self.assertTrue(core.is_external_cache_stored(node_id))

    def test_external_state_updates_are_atomic_and_path_scoped(self):
        cfg = CacheConfig(page_size=1, kv_size=64, max_context_len=64)
        self.cfg = cfg
        cache, allocator, req_to_token_pool = build_fixture(cfg)
        core = cache.tree_core
        core.enable_external_cache_linker = True

        anchor = self._insert(cache, allocator, req_to_token_pool, [1]).last_device_node
        middle = self._insert(
            cache, allocator, req_to_token_pool, [1, 2]
        ).last_device_node
        tail = self._insert(
            cache, allocator, req_to_token_pool, [1, 2, 3, 4]
        ).last_device_node
        unrelated = self._insert(
            cache, allocator, req_to_token_pool, [9]
        ).last_device_node
        self.assertEqual(core.get_parent_node_id(middle), anchor)
        self.assertEqual(core.get_parent_node_id(tail), middle)

        with self.assertRaisesRegex(RuntimeError, "not an ancestor"):
            core.mark_external_cache_stored_path(tail, unrelated)
        self.assertFalse(core.is_external_cache_stored(middle))
        self.assertFalse(core.is_external_cache_stored(tail))

        core.mark_external_cache_stored_path(tail, anchor)
        self.assertTrue(core.is_external_cache_stored(tail))
        self.assertTrue(core.is_external_cache_stored(middle))
        self.assertFalse(core.is_external_cache_stored(anchor))
        self.assertFalse(core.is_external_cache_stored(unrelated))
        with self.assertRaisesRegex(AssertionError, "invalid external offload state"):
            core.mark_external_linker_offload_pending(tail)

        split_tail = self._insert(
            cache, allocator, req_to_token_pool, [9, 10, 11]
        ).last_device_node
        self.assertEqual(core.get_parent_node_id(split_tail), unrelated)
        core.mark_external_linker_offload_pending(split_tail)
        self._insert(cache, allocator, req_to_token_pool, [9, 10])
        split_parent = core.get_parent_node_id(split_tail)
        self.assertIsNotNone(split_parent)
        self.assertNotEqual(split_parent, unrelated)
        self.assertEqual(core.get_parent_node_id(split_parent), unrelated)
        for node_id in (split_parent, split_tail):
            self.assertEqual(core.get_write_through_pending_id(node_id), split_tail)
            self.assertFalse(core.is_external_cache_stored(node_id))

        independent = self._insert(
            cache, allocator, req_to_token_pool, [20]
        ).last_device_node
        core.mark_external_linker_offload_pending(independent)
        with self.assertRaisesRegex(AssertionError, "invalid external offload state"):
            core.finish_external_linker_offload(
                [independent, split_parent], independent, success=False
            )
        self.assertEqual(core.get_write_through_pending_id(independent), independent)
        for node_id in (split_parent, split_tail):
            self.assertEqual(core.get_write_through_pending_id(node_id), split_tail)
            self.assertFalse(core.is_external_cache_stored(node_id))

        core.finish_external_linker_offload([independent], independent, success=False)
        core.finish_external_linker_offload(
            [split_parent, split_tail], split_tail, success=False
        )
        for node_id in (split_parent, split_tail):
            self.assertIsNone(core.get_write_through_pending_id(node_id))
            self.assertFalse(core.is_external_cache_stored(node_id))
        self.assertTrue(core.is_external_cache_stored(middle))
        self.assertTrue(core.is_external_cache_stored(tail))

    def test_failed_offload_preserves_independently_confirmed_state(self):
        cfg = CacheConfig(page_size=1, kv_size=64, max_context_len=64)
        self.cfg = cfg
        cache, allocator, req_to_token_pool = build_fixture(cfg)
        core = cache.tree_core
        core.enable_external_cache_linker = True

        anchor = self._insert(cache, allocator, req_to_token_pool, [1]).last_device_node
        node_id = self._insert(
            cache, allocator, req_to_token_pool, [1, 2]
        ).last_device_node
        core.mark_external_linker_offload_pending(node_id)

        core.mark_external_cache_stored_path(node_id, anchor)
        self.assertEqual(core.get_write_through_pending_id(node_id), node_id)
        self.assertTrue(core.is_external_cache_stored(node_id))

        core.finish_external_linker_offload([node_id], node_id, success=False)
        self.assertIsNone(core.get_write_through_pending_id(node_id))
        self.assertTrue(core.is_external_cache_stored(node_id))


class TestUnifiedCacheLinkerRustBackend(TestUnifiedCacheLinkerPythonBackend):
    tree_core_backend = "rust"


class TestUnifiedCacheLinkerTreeCoreRustBackend(
    TestUnifiedCacheLinkerTreeCorePythonBackend
):
    tree_core_backend = "rust"


def test_restorable_prefix_intersects_sparse_rank_results():
    remote_mask = torch.tensor([0, 0, 1, 0, 0], dtype=torch.int)

    def intersect_remote_mask(mask, op):
        assert op == torch.distributed.ReduceOp.MIN
        mask.copy_(torch.minimum(mask, remote_mask))

    cache = _cache_for_wrapper(_all_reduce_attn_groups=intersect_remote_mask)
    wrapper = UnifiedCacheLinkerWrapper(cache, _FakeLinker())

    hit_pages = wrapper._sync_restorable_prefix([2, 4], num_pages=4, device_hit_pages=0)

    assert hit_pages == 2


def test_async_offload_pins_node_until_completion():
    linker = _FakeLinker()
    lock_params = object()
    locks = []
    unlocks = []

    def inc_lock_ref(node):
        locks.append(node)
        return SimpleNamespace(to_dec_params=lambda: lock_params)

    node_id = 7
    node = SimpleNamespace(
        id=node_id,
        external_cache_stored=False,
        write_through_pending_id=None,
    )
    cache = _cache_for_wrapper(
        tree_core=_FakeExternalTreeCore({node_id: node}),
        inc_lock_ref=inc_lock_ref,
        dec_lock_ref=lambda node, params: unlocks.append((node, params)),
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)

    wrapper.offload_nodes([node_id])

    assert locks == [node_id]
    assert not node.external_cache_stored
    assert node.write_through_pending_id == node_id
    assert not unlocks

    linker.completed_offloads.append(False)
    completed = wrapper.take_completed_offloads(finish_count=1)
    wrapper.commit_completed_offloads(completed)

    assert not node.external_cache_stored
    assert node.write_through_pending_id is None
    assert unlocks == [(node_id, lock_params)]


def test_offload_carries_the_inserting_session():
    linker = _FakeLinker()
    nodes = {
        node_id: SimpleNamespace(
            id=node_id, external_cache_stored=False, write_through_pending_id=None
        )
        for node_id in (7, 8)
    }
    cache = _cache_for_wrapper(
        tree_core=_FakeExternalTreeCore(nodes),
        inc_lock_ref=lambda node_id: SimpleNamespace(to_dec_params=lambda: None),
        insert_session_id="session-a",
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)

    wrapper.offload_nodes([7])
    cache.insert_session_id = None
    wrapper.offload_nodes([8])

    assert linker.offload_sessions == ["session-a", None]


def test_offload_skips_node_already_stored_by_tree_core():
    linker = _FakeLinker()
    node = SimpleNamespace(
        id=7,
        external_cache_stored=True,
        write_through_pending_id=None,
    )
    cache = _cache_for_wrapper(
        tree_core=_FakeExternalTreeCore({node.id: node}),
        inc_lock_ref=lambda node_id: pytest.fail("stored node must not be locked"),
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)

    wrapper.offload_nodes([node.id])

    assert linker.queued_offloads == []


def test_layer_counter_reports_failure_after_the_forward():
    ready = []
    counter = LayerWiseLoadCounter(2, lambda index, layer: ready.append(layer))
    index = counter.update_producer()
    counter.set_consumer(index)
    counter.complete(index, 0)
    counter.fail(index, RuntimeError("storage read failed"))

    counter.wait_until(0)
    counter.wait_until(1)  # does not raise into the forward

    assert ready == [0, 1]  # an MLA dedup source still broadcasts layer 1
    assert counter.finish(index) is False


def test_layer_counter_finish_waits_for_layers_still_loading():
    counter = LayerWiseLoadCounter(1)
    index = counter.update_producer()
    loader = threading.Timer(0.05, counter.complete, (index, 0))
    loader.start()

    assert counter.finish(index) is True  # the forward never waited on layer 0
    loader.join()


def test_finish_loads_adopts_a_peer_rank_failure():
    reduce = MagicMock(side_effect=lambda value, op: value.fill_(0))
    wrapper = UnifiedCacheLinkerWrapper(
        _cache_for_wrapper(_all_reduce_attn_groups=reduce), _FakeLinker()
    )
    wrapper.inflight_load_rids = ["a", "b"]
    wrapper.start_layer_wise_loading()

    assert wrapper.finish_loads() == ["a", "b"]  # this rank landed, a peer did not
    assert wrapper.finish_loads() == []  # nothing loaded: no collective
    reduce.assert_called_once()


@pytest.mark.parametrize("device_len", [0, 2])
@pytest.mark.parametrize("tail_len", [0, 4])
def test_prefix_device_indices_includes_private_linker_tail(device_len, tail_len):
    path = torch.arange(device_len, dtype=torch.int64)
    tail = torch.arange(10, 10 + tail_len, dtype=torch.int64)
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.tree_core = SimpleNamespace(
        root_node_handle=lambda extra_key: 0,
        collect_full_device_indices=lambda node, root: path,
    )
    cache.linker = SimpleNamespace(
        private_load_indices={"rid": tail} if tail_len else {}
    )
    req = SimpleNamespace(
        rid="rid", extra_key=None, last_node=1, prefix_len=device_len + tail_len
    )

    expected = torch.cat([path, tail])
    assert torch.equal(cache.prefix_device_indices(req), expected)
    assert torch.equal(cache.prefix_device_indices(req), expected)
    assert len(path) == device_len


@pytest.mark.parametrize("landed", [True, False])
def test_finish_loads_releases_private_index_references(landed, caplog):
    backend = _FakeLinker()
    backend.load_landed = landed
    wrapper = UnifiedCacheLinkerWrapper(
        _cache_for_wrapper(_all_reduce_attn_groups=lambda value, op: None), backend
    )
    wrapper.inflight_load_rids = ["rid"]
    wrapper.private_load_indices["rid"] = torch.arange(4)
    wrapper.start_layer_wise_loading()

    assert wrapper.finish_loads() == ([] if landed else ["rid"])
    assert wrapper.private_load_indices == {}
    assert ("External linker load failed" in caplog.text) is (not landed)


@pytest.mark.parametrize("operation", ["reset", "close"])
def test_linker_lifecycle_releases_private_index_references(operation):
    wrapper = UnifiedCacheLinkerWrapper(_cache_for_wrapper(), _FakeLinker())
    wrapper.private_load_indices["rid"] = torch.arange(4)

    getattr(wrapper, operation)()

    assert wrapper.private_load_indices == {}


def test_release_request_cancels_queued_load():
    linker = _FakeLinker()
    wrapper = UnifiedCacheLinkerWrapper(_cache_for_wrapper(), linker)
    wrapper.hit_markers["rid"] = object()
    wrapper.inflight_load_rids = ["rid"]
    linker.queued_loads["rid"] = [object()]
    wrapper.private_load_indices["rid"] = torch.arange(4)

    wrapper.release_request("rid")

    assert wrapper.hit_markers == {}
    assert wrapper.inflight_load_rids == []
    assert "rid" not in linker.queued_loads
    assert wrapper.private_load_indices == {}


def test_failed_offload_rolls_back_split_fragments():
    linker = _FakeLinker()
    lock_params = object()
    unlocks = []
    child = SimpleNamespace(
        id=7,
        external_cache_stored=False,
        write_through_pending_id=None,
    )
    parent = SimpleNamespace(
        id=8,
        external_cache_stored=False,
        write_through_pending_id=None,
    )
    nodes = {child.id: child, parent.id: parent}

    cache = _cache_for_wrapper(
        tree_core=_FakeExternalTreeCore(nodes),
        inc_lock_ref=lambda node_id: SimpleNamespace(to_dec_params=lambda: lock_params),
        dec_lock_ref=lambda node_id, params: unlocks.append((node_id, params)),
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.offload_nodes([child.id])

    parent.external_cache_stored = child.external_cache_stored
    parent.write_through_pending_id = child.write_through_pending_id
    wrapper.replace_pending_offload_node(child.id, child.id, [parent.id, child.id])
    linker.completed_offloads.append(False)
    wrapper.commit_completed_offloads(wrapper.take_completed_offloads(finish_count=1))

    assert not parent.external_cache_stored
    assert not child.external_cache_stored
    assert parent.write_through_pending_id is None
    assert child.write_through_pending_id is None
    assert unlocks == [(child.id, lock_params)]


def test_split_action_retargets_pending_external_offload():
    calls = []
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.linker = SimpleNamespace(
        replace_pending_offload_node=lambda *args: calls.append(("linker", *args))
    )
    cache._replace_pending_write_through_node = lambda *args: calls.append(
        ("hicache", *args)
    )
    cache.components = {
        ComponentType.FULL: SimpleNamespace(
            apply_component_action=lambda action: calls.append(("full", action))
        )
    }
    action = UpdatePendingBackupOnNodeSplit(
        ack_id=7,
        old_node_id=7,
        new_node_id=8,
        new_child_node_id=7,
    )

    cache._apply_cache_action(action)

    assert calls == [
        ("hicache", 7, 7, [8, 7]),
        ("linker", 7, 7, [8, 7]),
        ("full", action),
    ]


def test_reset_quiesces_backend_before_releasing_pending_locks():
    events = []

    class _QuiescentFakeLinker(_FakeLinker):
        def reset(self):
            events.append("backend")
            super().reset()

    linker = _QuiescentFakeLinker()
    node = SimpleNamespace(
        id=7,
        external_cache_stored=False,
        write_through_pending_id=None,
    )
    cache = _cache_for_wrapper(
        tree_core=_FakeExternalTreeCore({node.id: node}),
        inc_lock_ref=lambda node_id: SimpleNamespace(to_dec_params=object),
        dec_lock_ref=lambda node_id, params: events.append(("unlock", node_id)),
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.inflight_load_rids = ["rid"]
    wrapper.offload_nodes([node.id])

    wrapper.reset()

    assert events == ["backend", ("unlock", node.id)]
    assert wrapper.inflight_load_rids == []
    assert wrapper.pending_offloads == []
    assert not node.external_cache_stored
    assert node.write_through_pending_id is None


def test_close_quiesces_backend_before_releasing_pending_loads():
    events = []

    class _ClosingFakeLinker(_FakeLinker):
        def close(self):
            events.append("backend")
            super().close()

    linker = _ClosingFakeLinker()
    wrapper = UnifiedCacheLinkerWrapper(_cache_for_wrapper(), linker)
    wrapper.inflight_load_rids = ["rid"]

    wrapper.close()

    assert events == ["backend"]
    assert linker.closed
    assert wrapper.inflight_load_rids == []


def test_check_hicache_events_commits_common_rank_results():
    committed = []
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.pp_size = 1
    cache.linker = SimpleNamespace(
        num_completed_offloads=lambda: 3,
        take_completed_offloads=lambda count: [True] * count,
        commit_completed_offloads=committed.append,
    )

    reduce_calls = 0

    def reduce_to_common_state(value, op):
        nonlocal reduce_calls
        assert op == torch.distributed.ReduceOp.MIN
        reduce_calls += 1
        if reduce_calls == 1:
            value.copy_(torch.tensor([1]))
        else:
            value.fill_(0)

    cache._all_reduce_attn_groups = reduce_to_common_state

    cache.check_hicache_events()

    assert committed == [[False]]


@pytest.mark.parametrize(
    "swa_req_ring",
    [None, False, True],
    ids=["other-allocator", "paged", "request-ring"],
)
@pytest.mark.parametrize("enable_hicache", [False, True])
def test_swa_reuse_policy_tracks_layout_without_a_tier_condition(
    monkeypatch, swa_req_ring, enable_hicache
):
    from sglang.srt.layers.attention.dsv4 import unified_kv_gate

    unified_kv = swa_req_ring is True
    monkeypatch.setattr(unified_kv_gate, "is_unified_kv_triton", lambda: unified_kv)
    component = SWAComponent.__new__(SWAComponent)
    component.sliding_window_size = 128
    component.gates_prefix_match = not unified_kv
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.token_to_kv_pool_allocator = _swa_allocator(swa_req_ring)
    cache.components = {ComponentType.SWA: component}
    cache.cache_controller = object() if enable_hicache else None
    cache.decoder_swa_bounded_replay = False
    cache.tree_core = SimpleNamespace(
        enable_hicache=enable_hicache,
        has_swa_host_pool=enable_hicache and not unified_kv,
    )
    component.cache = cache
    component.tree_core = cache.tree_core
    # #32759: request-relative SWA needs tail re-prefill even without HiCache.
    assert cache.swa_reprefill_tail_tokens() == (128 if unified_kv else 0)
    node = SimpleNamespace(
        component_data={
            ComponentType.SWA: SimpleNamespace(value=None, host_value=None)
        },
        backuped=False,
        evicted=False,
    )
    assert component.create_match_validator(match_device_only=True)(node) is unified_kv


def test_cache_without_swa_needs_no_reprefill():
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.components = {}
    assert cache.swa_reprefill_tail_tokens() == 0


@pytest.mark.parametrize("swa_stored", [True, False])
def test_skip_swa_when_linker_store_lacks_swa(swa_stored):
    """A store without SWA pages must tombstone restored tails, not mark them hit."""
    linker = _FakeLinker()
    if swa_stored:
        linker.pools[PoolName.SWA] = object()
    swa = object()
    cache = _cache_for_wrapper(
        components={ComponentType.SWA: swa},
        _components_tuple=(swa,),
        tree_components=(ComponentType.FULL, ComponentType.SWA),
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    assert wrapper._skip_swa is (not swa_stored)
    assert (swa in wrapper._components) is swa_stored


@pytest.fixture
def full_linker_component():
    def build_transfer(phase, node, keys):
        keys = ["offload"] if phase == LinkerTransferPhase.OFFLOAD else list(keys)
        return PoolTransfer(
            name=PoolName.KV,
            keys=keys,
            device_indices=(
                None
                if phase == LinkerTransferPhase.LOOKUP
                else torch.arange(len(keys) * 2)
            ),
        )

    return SimpleNamespace(
        component_type=ComponentType.FULL,
        build_external_linker_transfer=MagicMock(side_effect=build_transfer),
        update_external_linker_load=lambda phase, req, full_transfer, transfer, prefix_len, **kwargs: (
            transfer
        ),
    )


def test_linker_filters_unstored_swa_from_lookup(full_linker_component):
    full = full_linker_component
    swa = SWAComponent.__new__(SWAComponent)
    swa.build_external_linker_transfer = MagicMock(
        side_effect=AssertionError("excluded SWA reached linker")
    )
    node = SimpleNamespace(id=1, external_cache_stored=False)
    cache = _cache_for_wrapper(
        _components_tuple=(full, swa),
        components={ComponentType.FULL: full, ComponentType.SWA: swa},
        tree_core=SimpleNamespace(enable_external_cache_linker=False, is_eagle=False),
        page_size=2,
        _all_reduce_attn_groups=lambda value, op: None,
        get_last_hash_value=lambda node: None,
        resolve_node_handle=lambda node_id: node,
        inc_lock_ref=lambda node_id: SimpleNamespace(to_dec_params=lambda: object()),
        dec_lock_ref=MagicMock(),
    )
    backend = _FakeLinker()
    backend.restorable = [2]
    wrapper = UnifiedCacheLinkerWrapper(cache, backend)
    assert wrapper._components == (full,)
    result = MatchResult(
        device_prefix_len=0,
        last_device_node=0,
        last_host_node=0,
        best_match_node=0,
    )
    matched = wrapper.match(
        RadixKey(array("q", [1, 2, 3, 4])), SimpleNamespace(rid="match"), result
    )
    assert matched.host_hit_length == 4
    assert [c.args[0] for c in full.build_external_linker_transfer.call_args_list] == [
        LinkerTransferPhase.LOOKUP,
    ]
    swa.build_external_linker_transfer.assert_not_called()


def _page_hashes(tokens, prior_hash, page_size):
    """A pure-Python stand-in for the chained page hash (no native extension)."""
    hashes, last = [], prior_hash or "root"
    for start in range(0, len(tokens), page_size):
        last = f"{last}/{list(tokens[start : start + page_size])}"
        hashes.append(last)
    return hashes


class _CountingLinker(_FakeLinker):
    def __init__(self, restorable=(), stage_restorable=()):
        super().__init__()
        self.restorable = list(restorable)
        self.stage_restorable = list(stage_restorable)
        self.lookups = 0
        self.stage_lookups = 0
        self.refreshed = []

    def lookup(self, rid, transfers):
        self.lookups += 1
        return list(self.restorable)

    def lookup_stages(self, rid, transfers):
        self.stage_lookups += 1
        return list(self.stage_restorable)

    def refresh_device_prefix(self, anchor, hashes):
        self.refreshed.append((anchor, hashes()))


def _match_wrapper(component, backend, **cache_kwargs):
    cache = _cache_for_wrapper(
        _components_tuple=(component,),
        components={ComponentType.FULL: component},
        tree_core=SimpleNamespace(
            enable_external_cache_linker=False,
            is_eagle=False,
            get_prefix_hash_values=lambda node: [f"device-{node}"],
        ),
        page_size=2,
        _all_reduce_attn_groups=lambda value, op: None,
        get_last_hash_value=lambda node: f"anchor-{node}",
        **cache_kwargs,
    )
    return UnifiedCacheLinkerWrapper(cache, backend)


def _device_result(device_len):
    return MatchResult(
        device_prefix_len=device_len,
        last_device_node=device_len,
        last_host_node=device_len,
        best_match_node=device_len,
    )


_KEY = RadixKey(array("q", [1, 2, 3, 4, 5, 6]))


def test_lookup_once_reuses_a_request_lookup_across_passes(full_linker_component):
    # SPF matches every waiting request on every scheduling pass; with
    # SGLANG_LINKER_LOOKUP_ONCE only the first match queries the store.
    backend = _CountingLinker(restorable=[1, 3])
    with (
        envs.SGLANG_LINKER_LOOKUP_ONCE.override(True),
        patch(
            "sglang.srt.mem_cache.unified_cache.unified_cache_linker.get_storage_hash_str",
            side_effect=_page_hashes,
        ) as hashed,
    ):
        wrapper = _match_wrapper(full_linker_component, backend)
        req = SimpleNamespace(rid="r")
        for _ in range(3):
            assert wrapper.match(_KEY, req, _device_result(0)).host_hit_length == 6
        assert backend.lookups == 1
        assert hashed.call_count == 1
        # A deeper device prefix reuses the lookup: the stop at 6 tokens remains.
        assert wrapper.match(_KEY, req, _device_result(2)).host_hit_length == 4
        assert backend.lookups == 1
        # A shallower one, or a new lifetime, looks the store up again.
        wrapper.release_request("r")
        assert wrapper.match(_KEY, req, _device_result(0)).host_hit_length == 6
        assert backend.lookups == 2


def test_pp_stages_admit_pp0s_relayed_lookup(full_linker_component):
    with patch(
        "sglang.srt.mem_cache.unified_cache.unified_cache_linker.get_storage_hash_str",
        side_effect=_page_hashes,
    ):
        pp0_backend = _CountingLinker(stage_restorable=[1, 3])
        pp0 = _match_wrapper(full_linker_component, pp0_backend, pp_size=8, pp_rank=0)
        req0 = SimpleNamespace(rid="r", linker_pp_hit=None, linker_pp_lookup_pending=True)
        assert pp0.match(_KEY, req0, _device_result(0)).host_hit_length == 6
        assert req0.linker_pp_hit == (0, [2, 6])
        assert (pp0_backend.lookups, pp0_backend.stage_lookups) == (0, 1)

        # A later stage admits from the relayed result with no store query.
        later_backend = _CountingLinker()
        later = _match_wrapper(full_linker_component, later_backend, pp_size=8, pp_rank=5)
        req5 = SimpleNamespace(
            rid="r", linker_pp_hit=req0.linker_pp_hit, linker_pp_lookup_pending=False
        )
        for device_len, hit in ((0, 6), (2, 4), (6, 0)):
            matched = later.match(_KEY, req5, _device_result(device_len))
            assert matched.host_hit_length == hit, device_len
        assert (later_backend.lookups, later_backend.stage_lookups) == (0, 0)

        # Pages between a shrunk device prefix and the looked-up start were
        # never checked: the relayed stop is admitted and PP0 looks the whole
        # tail up again before the load, on every stage alike.
        req5.linker_pp_hit = (2, [6])
        assert later.match(_KEY, req5, _device_result(0)).host_hit_length == 6
        assert "r" in later.pp_forced_relookup
        req5.linker_pp_hit = None
        assert later.match(_KEY, req5, _device_result(0)).host_hit_length == 0
        assert (later_backend.lookups, later_backend.stage_lookups) == (0, 0)


def test_device_hit_renews_its_store_leases(full_linker_component):
    backend = _CountingLinker()
    with patch(
        "sglang.srt.mem_cache.unified_cache.unified_cache_linker.get_storage_hash_str",
        side_effect=_page_hashes,
    ):
        wrapper = _match_wrapper(full_linker_component, backend)
        wrapper.match(_KEY, SimpleNamespace(rid="r"), _device_result(0))
        wrapper.match(_KEY, SimpleNamespace(rid="r"), _device_result(6))
    assert backend.refreshed == [(6, ["device-6"])]


def _relayed(value):
    """_all_reduce for a later PP stage: PP0's value arrives over _pp_sync."""

    def all_reduce(data, op):
        data[0] = value

    return all_reduce


def test_pp_stale_hit_is_looked_up_again_and_shrinks_before_its_load(
    full_linker_component,
):
    # Rhea 57360: a hit relayed at enqueue waited 354 s, its objects were
    # evicted, and the load failed. A hit older than the lease is looked up
    # again first; what is gone is recomputed, not a failed read.
    with patch(
        "sglang.srt.mem_cache.unified_cache.unified_cache_linker.get_storage_hash_str",
        side_effect=_page_hashes,
    ):
        backend = _CountingLinker(stage_restorable=[1, 3])
        pp0 = _match_wrapper(
            full_linker_component,
            backend,
            pp_size=8,
            pp_rank=0,
            _all_reduce=lambda data, op: None,
        )
        req = SimpleNamespace(rid="r", linker_pp_hit=None, linker_pp_lookup_pending=True)
        assert pp0.match(_KEY, req, _device_result(0)).host_hit_length == 6
        req.linker_pp_lookup_pending = False
        pp0.pp_lookup_time["r"] -= pp0.stale_after_s + 1
        backend.stage_restorable = [1]  # pages 2-3 evicted while queued

        assert pp0.load_back(req) is None
        assert backend.stage_lookups == 2
        assert req.linker_pp_hit == (0, [2])
        assert (pp0.pp_relookups, pp0.pp_relookups_shrunk) == (1, 1)
        # The next pass admits what is left, without another store query.
        assert pp0.match(_KEY, req, _device_result(0)).host_hit_length == 2
        assert backend.stage_lookups == 2


def test_pp_hit_outliving_its_device_prefix_is_looked_up_again_not_dropped(
    full_linker_component,
):
    # Rhea 57360: while a request queued, its device prefix shrank below the
    # point its store hit was looked up from, and the relayed hit was dropped.
    with patch(
        "sglang.srt.mem_cache.unified_cache.unified_cache_linker.get_storage_hash_str",
        side_effect=_page_hashes,
    ):
        backend = _CountingLinker(stage_restorable=[1, 2])
        pp0 = _match_wrapper(
            full_linker_component,
            backend,
            pp_size=8,
            pp_rank=0,
            _all_reduce=lambda data, op: None,
        )
        req = SimpleNamespace(rid="r", linker_pp_hit=None, linker_pp_lookup_pending=True)
        # Enqueue: device prefix 2, store stops at 4 and 6 tokens.
        pp0.match(_KEY, req, _device_result(2))
        req.linker_pp_lookup_pending = False
        assert req.linker_pp_hit == (2, [4, 6])
        # The device prefix is gone by admission: the hit is kept, unchecked.
        assert pp0.match(_KEY, req, _device_result(0)).host_hit_length == 6
        assert pp0.pp_counts["anchor_shrunk"] == 1
        # PP0 looks the tail from the device prefix up again though the hit is
        # fresh; only the first page survived, and the hit re-anchors at 0.
        backend.stage_restorable = [1]
        assert pp0.load_back(req) is None
        assert backend.stage_lookups == 2
        assert req.linker_pp_hit == (0, [2])
        assert "r" not in pp0.pp_forced_relookup
        assert (pp0.pp_counts["recovered_pages"], pp0.pp_counts["lost_pages"]) == (1, 2)
        assert pp0.match(_KEY, req, _device_result(0)).host_hit_length == 2


def test_pp_hit_relayed_with_no_stops_is_looked_up_again_when_its_device_prefix_shrinks(
    full_linker_component,
):
    # Rhea 59439: a follow-up turn enqueued with its prompt on device, so PP0
    # looked up only the new tail and relayed no stops. The prompt was evicted
    # while the request queued and recomputed, though the store held it.
    with patch(
        "sglang.srt.mem_cache.unified_cache.unified_cache_linker.get_storage_hash_str",
        side_effect=_page_hashes,
    ):
        backend = _CountingLinker(stage_restorable=[])
        pp0 = _match_wrapper(
            full_linker_component,
            backend,
            pp_size=8,
            pp_rank=0,
            _all_reduce=lambda data, op: None,
        )
        req = SimpleNamespace(rid="r", linker_pp_hit=None, linker_pp_lookup_pending=True)
        # Enqueue: device prefix 4, nothing stored past it.
        assert pp0.match(_KEY, req, _device_result(4)).host_hit_length == 0
        req.linker_pp_lookup_pending = False
        assert req.linker_pp_hit == (4, [])
        # The device prefix is gone by admission: its end is admitted, unchecked.
        assert pp0.match(_KEY, req, _device_result(0)).host_hit_length == 4
        assert pp0.pp_counts["anchor_shrunk"] == 1
        # PP0 looks it up again before the load; only the first page is left.
        backend.stage_restorable = [1]
        assert pp0.load_back(req) is None
        assert backend.stage_lookups == 2
        assert req.linker_pp_hit == (0, [2])
        assert (pp0.pp_counts["recovered_pages"], pp0.pp_counts["lost_pages"]) == (1, 1)
        assert pp0.match(_KEY, req, _device_result(0)).host_hit_length == 2
        assert (pp0.pp_counts["shrunk_relooked"], pp0.pp_counts["shrunk_unlooked"]) == (1, 0)

        # A shrunk request that still leaves match with no hit is counted.
        req2 = SimpleNamespace(rid="r2", linker_pp_hit=(4, []), linker_pp_lookup_pending=False)
        pp0.cache._all_reduce_attn_groups = lambda value, op: value.zero_()
        assert pp0.match(_KEY, req2, _device_result(0)).host_hit_length == 0
        assert pp0.pp_counts["shrunk_unlooked"] == 1
        assert "r2" not in pp0.pp_forced_relookup


class _PinningLinker(_CountingLinker):
    """pin_hit returns pages[rid] (default: the whole hit) and records it."""

    pins_hits = True

    def __init__(self, stage_restorable=(), pages=None):
        super().__init__(stage_restorable=stage_restorable)
        self.pages, self.pins, self.unpins = dict(pages or {}), [], []
        self.prepin_starts, self.urgent_starts = [], []
        self.last_pin_rpc_s = 0.004

    def pin_hit(self, rid, transfers):
        kv = next(t for t in transfers if t.name == PoolName.KV)
        self.pins.append(rid)
        return self.pages.get(rid, len(kv.keys))

    def unpin(self, rid):
        self.unpins.append(rid)

    prepinned = None

    def start_prepin(self, rid, transfers, urgent=False):
        self.prepin_starts.append(rid)
        if urgent:
            self.urgent_starts.append(rid)


    def take_prepin(self, rid, transfers):
        return self.prepinned

    def peek_prepin(self, rid, transfers):
        return self.prepinned


def test_pp_admission_pins_the_hit_and_a_hole_shrinks_it_before_the_load(
    full_linker_component,
):
    # Rhea 60076: every stale or shrunk hit was looked up again on PP0 at
    # admission. The stages pin the hit instead; an object evicted since the
    # lookup shrinks it to a stop PP0 finds below the hole, and the request
    # waits a pass instead of failing its load.
    with patch(
        "sglang.srt.mem_cache.unified_cache.unified_cache_linker.get_storage_hash_str",
        side_effect=_page_hashes,
    ):
        backend = _PinningLinker(stage_restorable=[1, 3], pages={"hole": 2})
        pp0 = _match_wrapper(
            full_linker_component,
            backend,
            pp_size=8,
            pp_rank=0,
            _all_reduce=lambda data, op: None,
        )
        req = SimpleNamespace(rid="r", linker_pp_hit=None, linker_pp_lookup_pending=True)
        assert pp0.match(_KEY, req, _device_result(0)).host_hit_length == 6
        pp0.pp_lookup_time["r"] -= pp0.stale_after_s + 1  # stale: pinned, not re-looked up
        assert pp0._pp_fresh_hit_pages(req, pp0.hit_markers["r"]) == 3
        assert (backend.pins, backend.unpins, backend.stage_lookups) == (["r"], [], 1)
        assert pp0.admission_rpc_s == backend.last_pin_rpc_s

        hole = SimpleNamespace(
            rid="hole", linker_pp_hit=None, linker_pp_lookup_pending=True
        )
        pp0.match(_KEY, hole, _device_result(0))
        hole.linker_pp_lookup_pending = False
        backend.stage_restorable = [1]  # the deepest stop below the hole
        assert pp0.load_back(hole) is None
        assert backend.unpins == ["hole"]
        assert (pp0.pp_load_holes, backend.stage_lookups) == (1, 3)
        assert hole.linker_pp_hit == (0, [2])
        assert len(pp0.admission_samples) == 1
        assert pp0.match(_KEY, hole, _device_result(0)).host_hit_length == 2


def test_only_pp0_pins_and_later_stages_take_its_relayed_verdict(
    full_linker_component,
):
    # Hecate H106: the keep relay carries PP0's value only, so a later
    # stage's own pin was overwritten; PP0 pins every stage instead.
    with patch(
        "sglang.srt.mem_cache.unified_cache.unified_cache_linker.get_storage_hash_str",
        side_effect=_page_hashes,
    ):
        backend = _PinningLinker()
        later = _match_wrapper(
            full_linker_component,
            backend,
            pp_size=8,
            pp_rank=5,
            _all_reduce=_relayed(1),
        )
        req = SimpleNamespace(
            rid="r", linker_pp_hit=(0, [2, 6]), linker_pp_lookup_pending=False
        )
        assert later.match(_KEY, req, _device_result(0)).host_hit_length == 6
        assert later._pp_fresh_hit_pages(req, later.hit_markers["r"]) == 1
        assert backend.pins == []
        assert backend.unpins == ["r"]  # PP0 shrank the hit: nothing is held here


def test_pp0_takes_a_finished_prepin_and_pins_itself_only_when_none_is_ready(
    full_linker_component,
):
    # Hecate H112: the all-stage pin at admission held PP0's scheduler about
    # 70 ms per hit; with SGLANG_LINKER_PREPIN_DEPTH the next waiting hits are
    # pinned after each pass and admission takes the finished pin.
    with (
        envs.SGLANG_LINKER_PREPIN_DEPTH.override(2),
        patch(
            "sglang.srt.mem_cache.unified_cache.unified_cache_linker.get_storage_hash_str",
            side_effect=_page_hashes,
        ),
    ):
        backend = _PinningLinker(stage_restorable=[1, 3])
        pp0 = _match_wrapper(
            full_linker_component,
            backend,
            pp_size=8,
            pp_rank=0,
            _all_reduce=lambda data, op: None,
        )
        reqs = [
            SimpleNamespace(rid=r, linker_pp_hit=None, linker_pp_lookup_pending=True)
            for r in ("a", "b", "c")
        ]
        for req in reqs[:2] + reqs[2:]:
            pp0.match(_KEY, req, _device_result(0))
        miss = SimpleNamespace(rid="none", linker_pp_hit=None, linker_pp_lookup_pending=False)
        pp0.prepin([miss] + reqs)  # the first two waiting requests with a hit
        assert backend.prepin_starts == ["a", "b"]

        backend.prepinned = 3
        assert pp0._pp_fresh_hit_pages(reqs[0], pp0.hit_markers["a"]) == 3
        assert backend.pins == []
        backend.prepinned = None
        assert pp0._pp_fresh_hit_pages(reqs[1], pp0.hit_markers["b"]) == 3
        assert backend.pins == ["b"]
        assert pp0.prepin_counts == {"ready": 1, "missed": 1}


def _pp_pair(component, pp0_backend, later_backend):
    """PP0 and a later stage; the later stage's relay delivers PP0's last value."""
    relayed = []

    def pp0_reduce(data, op):
        relayed.append(int(data[0]))

    def later_reduce(data, op):
        data[0] = relayed.pop(0)

    pp0 = _match_wrapper(
        component, pp0_backend, pp_size=8, pp_rank=0, _all_reduce=pp0_reduce
    )
    later = _match_wrapper(
        component, later_backend, pp_size=8, pp_rank=5, _all_reduce=later_reduce
    )
    return pp0, later


def test_async_admission_skips_an_unready_hit_on_every_stage_until_its_prepin_lands(
    full_linker_component,
):
    # Hecate H117: PP0 pinned every unready hit on its scheduler thread, about
    # 0.2 s per batch. With async admission an unready hit is skipped for the
    # pass on every stage alike (the keep relay carries the skip); its pre-pin
    # keeps running and a later pass admits it with no store call.
    with (
        envs.SGLANG_LINKER_PREPIN_DEPTH.override(4),
        envs.SGLANG_LINKER_ASYNC_ADMISSION.override(True),
        patch(
            "sglang.srt.mem_cache.unified_cache.unified_cache_linker.get_storage_hash_str",
            side_effect=_page_hashes,
        ),
    ):
        pp0_backend, later_backend = _PinningLinker(stage_restorable=[1, 3]), _PinningLinker()
        pp0, later = _pp_pair(full_linker_component, pp0_backend, later_backend)
        req0 = SimpleNamespace(rid="r", linker_pp_hit=None, linker_pp_lookup_pending=True)
        pp0.match(_KEY, req0, _device_result(0))
        req5 = SimpleNamespace(
            rid="r", linker_pp_hit=req0.linker_pp_hit, linker_pp_lookup_pending=False
        )
        req0.linker_pp_lookup_pending = False
        later.match(_KEY, req5, _device_result(0))

        assert pp0.load_back(req0) is LINKER_ADMISSION_SKIPPED
        assert later.load_back(req5) is LINKER_ADMISSION_SKIPPED
        assert pp0_backend.pins == [] and later_backend.pins == []
        assert pp0_backend.prepin_starts == ["r"]
        assert pp0_backend.queued_loads == {} and later_backend.queued_loads == {}
        assert pp0.skip_counts == later.skip_counts == {"r": 1}
        assert pp0.admission_samples == []  # a skip is not an admission

        # The next pass matches again; the pre-pin has landed: both stages
        # admit the whole hit, and PP0 made no store call at admission.
        pp0_backend.prepinned = 3
        pp0.match(_KEY, req0, _device_result(0))
        later.match(_KEY, req5, _device_result(0))
        assert pp0._pp_fresh_hit_pages(req0, pp0.hit_markers["r"]) == 3
        assert later._pp_fresh_hit_pages(req5, later.hit_markers["r"]) == 3
        assert pp0_backend.pins == []
        assert pp0.skip_counts == later.skip_counts == {}
        assert pp0.async_counts["skipped"] == 1 and pp0.async_counts["urgent"] == 0


def test_async_admission_never_pins_at_admission_and_holds_a_starving_hits_pin(
    full_linker_component,
):
    # A hit skipped _SKIP_LIMIT passes in a row used to be pinned synchronously
    # on PP0's scheduler thread (the H117 cost again). Now it is still skipped,
    # its pre-pin turns urgent and stays in the window beyond the depth, and the
    # first pass after the pin lands admits it with no store call.
    with (
        envs.SGLANG_LINKER_PREPIN_DEPTH.override(2),
        envs.SGLANG_LINKER_ASYNC_ADMISSION.override(True),
        # The progress bound off: this test holds a starving hit's pin.
        envs.SGLANG_LINKER_SKIP_FALLBACK_PASSES.override(0),
        envs.SGLANG_LINKER_SKIP_FALLBACK_MS.override(0),
        patch(
            "sglang.srt.mem_cache.unified_cache.unified_cache_linker.get_storage_hash_str",
            side_effect=_page_hashes,
        ),
    ):
        backend = _PinningLinker(stage_restorable=[1, 3])
        kept = []
        backend.keep_prepins = lambda rids, urgent=(): kept.append((list(rids), list(urgent)))
        pp0 = _match_wrapper(
            full_linker_component,
            backend,
            pp_size=8,
            pp_rank=0,
            _all_reduce=lambda data, op: None,
        )
        req = SimpleNamespace(rid="r", linker_pp_hit=None, linker_pp_lookup_pending=True)
        pp0.match(_KEY, req, _device_result(0))
        limit = UnifiedCacheLinkerWrapper._SKIP_LIMIT
        for _ in range(limit + 2):
            hit = pp0.hit_markers["r"]
            assert pp0._pp_fresh_hit_pages(req, hit) == UnifiedCacheLinkerWrapper._SKIPPED
        assert backend.pins == []  # never pinned on the scheduler thread
        assert backend.urgent_starts == ["r"] * 3  # from the limit-th skip on
        assert pp0.async_counts["skipped"] == limit + 2 and pp0.async_counts["urgent"] == 3

        # A deep queue of hits ahead of it in policy order cannot push its pin
        # out of the window: it is kept and held, the others fill the depth.
        others = []
        for rid in ("a", "b", "c"):
            other = SimpleNamespace(rid=rid, linker_pp_hit=None, linker_pp_lookup_pending=True)
            pp0.match(_KEY, other, _device_result(0))
            others.append(other)
        pp0.prepin(others + [req])
        rids, urgent = kept[-1]
        assert urgent == ["r"] and rids[0] == "r" and len(rids) == 1 + 2

        backend.prepinned = 3
        assert pp0._pp_fresh_hit_pages(req, pp0.hit_markers["r"]) == 3
        assert backend.pins == [] and pp0.skip_counts == {}


def _verdict_pair(component, pp0_backend, later_backend):
    """PP0 and a later stage in batched-relay mode; counts each per-hit relay."""
    relays = []

    def pp0_reduce(data, op):
        relays.append(("pp0", int(data[0])))

    def later_reduce(data, op):
        data[0] = next(v for who, v in reversed(relays) if who == "pp0")
        relays.append(("later", int(data[0])))

    pp0 = _match_wrapper(component, pp0_backend, pp_size=8, pp_rank=0, _all_reduce=pp0_reduce)
    later = _match_wrapper(component, later_backend, pp_size=8, pp_rank=5, _all_reduce=later_reduce)
    return pp0, later, relays


def test_batched_relay_admits_and_skips_alike_without_a_relay_per_hit(
    full_linker_component,
):
    # Rhea pm-lk3: one PP relay per hit (640-930 skips a minute) kept PP0 in
    # _drain_async_work 29% of its scheduler thread. The verdicts ride the
    # request relay once per round; every stage applies the same map.
    with (
        envs.SGLANG_LINKER_PREPIN_DEPTH.override(4),
        envs.SGLANG_LINKER_ASYNC_ADMISSION.override(True),
        envs.SGLANG_LINKER_ASYNC_BATCHED_RELAY.override(True),
        patch(
            "sglang.srt.mem_cache.unified_cache.unified_cache_linker.get_storage_hash_str",
            side_effect=_page_hashes,
        ),
    ):
        pp0_backend, later_backend = _PinningLinker(stage_restorable=[1, 3]), _PinningLinker()
        pp0, later, relays = _verdict_pair(full_linker_component, pp0_backend, later_backend)
        reqs = {}
        for rid in ("ready", "unready"):
            r0 = SimpleNamespace(rid=rid, linker_pp_hit=None, linker_pp_lookup_pending=True)
            pp0.match(_KEY, r0, _device_result(0))
            r5 = SimpleNamespace(rid=rid, linker_pp_hit=r0.linker_pp_hit, linker_pp_lookup_pending=False)
            r0.linker_pp_lookup_pending = False
            later.match(_KEY, r5, _device_result(0))
            reqs[rid] = (r0, r5)
        verdicts = LinkerAdmissionVerdicts(ready={"ready": (3, 3)})
        pp0.apply_verdicts(verdicts)
        later.apply_verdicts(verdicts)
        pp0_backend.prepinned = 3
        for rid, want in (("ready", 3), ("unready", UnifiedCacheLinkerWrapper._SKIPPED)):
            r0, r5 = reqs[rid]
            assert pp0._pp_fresh_hit_pages(r0, pp0.hit_markers[rid]) == want
            assert later._pp_fresh_hit_pages(r5, later.hit_markers[rid]) == want
        assert relays == []  # no per-hit relay at all
        assert pp0.skip_counts == later.skip_counts == {"unready": 1}
        assert pp0_backend.prepin_starts == ["unready"] and later_backend.prepin_starts == []


def test_batched_relay_skips_a_stale_verdict_and_relays_only_a_hole(
    full_linker_component,
):
    with (
        envs.SGLANG_LINKER_PREPIN_DEPTH.override(4),
        envs.SGLANG_LINKER_ASYNC_ADMISSION.override(True),
        envs.SGLANG_LINKER_ASYNC_BATCHED_RELAY.override(True),
        patch(
            "sglang.srt.mem_cache.unified_cache.unified_cache_linker.get_storage_hash_str",
            side_effect=_page_hashes,
        ),
    ):
        pp0_backend, later_backend = _PinningLinker(stage_restorable=[1, 3]), _PinningLinker()
        pp0, later, relays = _verdict_pair(full_linker_component, pp0_backend, later_backend)
        r0 = SimpleNamespace(rid="r", linker_pp_hit=None, linker_pp_lookup_pending=True)
        pp0.match(_KEY, r0, _device_result(0))
        r5 = SimpleNamespace(rid="r", linker_pp_hit=r0.linker_pp_hit, linker_pp_lookup_pending=False)
        r0.linker_pp_lookup_pending = False
        later.match(_KEY, r5, _device_result(0))
        # A verdict for a different hit length is stale: skipped everywhere.
        for w in (pp0, later):
            w.apply_verdicts(LinkerAdmissionVerdicts(ready={"r": (2, 2)}))
        assert pp0._pp_fresh_hit_pages(r0, pp0.hit_markers["r"]) == UnifiedCacheLinkerWrapper._SKIPPED
        assert later._pp_fresh_hit_pages(r5, later.hit_markers["r"]) == UnifiedCacheLinkerWrapper._SKIPPED
        # A verdict whose pre-pin found a hole relays PP0's re-lookup, once.
        for w in (pp0, later):
            w.apply_verdicts(LinkerAdmissionVerdicts(ready={"r": (3, 2)}))
        pp0.match(_KEY, r0, _device_result(0))
        later.match(_KEY, r5, _device_result(0))
        got = pp0._pp_fresh_hit_pages(r0, pp0.hit_markers["r"])
        assert later._pp_fresh_hit_pages(r5, later.hit_markers["r"]) == got == 1
        assert [who for who, _ in relays] == ["pp0", "later"]


def test_batched_relay_verdicts_are_the_landed_prepins_of_the_window(
    full_linker_component,
):
    with (
        envs.SGLANG_LINKER_PREPIN_DEPTH.override(8),
        envs.SGLANG_LINKER_ASYNC_ADMISSION.override(True),
        envs.SGLANG_LINKER_ASYNC_BATCHED_RELAY.override(True),
        envs.SGLANG_LINKER_PREPIN_WINDOW_MIN.override(3),
        patch(
            "sglang.srt.mem_cache.unified_cache.unified_cache_linker.get_storage_hash_str",
            side_effect=_page_hashes,
        ),
    ):
        backend = _PinningLinker(stage_restorable=[1, 3])
        landed = {"a": 3, "c": 3, "d": 3}  # d landed but is outside the window
        backend.peek_prepin = lambda rid, transfers: landed.get(rid)
        kept = []
        backend.keep_prepins = lambda rids, urgent=(): kept.append(list(rids))
        pp0 = _match_wrapper(full_linker_component, backend, pp_size=8, pp_rank=0, _all_reduce=lambda d, op: None)
        reqs = []
        for rid in ("a", "b", "c", "d"):
            r = SimpleNamespace(rid=rid, linker_pp_hit=None, linker_pp_lookup_pending=True)
            pp0.match(_KEY, r, _device_result(0))
            reqs.append(r)
        pp0.prepin(reqs)
        # The window floor (3) holds although no hit was admitted yet; the
        # verdicts are the window's landed pre-pins only (d is outside it).
        assert kept[-1] == ["a", "b", "c"]
        sent = pp0.take_verdicts()
        assert sent.ready == {"a": (3, 3), "c": (3, 3)}
        assert pp0.take_verdicts().ready == {}  # sent once


class _LandingPinLinker(_PinningLinker):
    """Pre-pins land when the test says: landed[rid] = pages once it has."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.landed = {}

    def take_prepin(self, rid, transfers):
        return self.landed.pop(rid, None)

    def peek_prepin(self, rid, transfers):
        return self.landed.get(rid)


def _hit_pair_reqs(pp0, later, rids):
    reqs = {}
    for rid in rids:
        fields = dict(rid=rid, last_node=f"{rid}-node", kv=None, prefix_len=0)
        fields.update(priority=0, swa_branching_seqlen=None)
        r0 = SimpleNamespace(linker_pp_hit=None, linker_pp_lookup_pending=True, **fields)
        pp0.match(_KEY, r0, _device_result(0))
        r5 = SimpleNamespace(linker_pp_hit=r0.linker_pp_hit, linker_pp_lookup_pending=False, **fields)
        r0.linker_pp_lookup_pending = False
        reqs[rid] = (r0, r5)
    return reqs


def _admission_pass(pp0, later, reqs):
    """One scheduling round on PP0 and a later stage, as the scheduler runs it:
    PP0's verdicts ride the request relay to every stage (batched relay), each
    stage matches and admits its waiting hits in the same order, then PP0
    pre-pins the next window. Admitted requests leave ``reqs``. Returns
    rid -> (PP0's load_back result, the later stage's)."""
    verdicts = pp0.take_verdicts()
    if verdicts is not None:
        pp0.apply_verdicts(verdicts)
        later.apply_verdicts(verdicts)
    out = {}
    for rid, (r0, r5) in list(reqs.items()):
        pp0.match(_KEY, r0, _device_result(0))
        later.match(_KEY, r5, _device_result(0))
        out[rid] = (pp0.load_back(r0), later.load_back(r5))
        if out[rid][0] is not LINKER_ADMISSION_SKIPPED:
            del reqs[rid]
    pp0.prepin([r0 for r0, _ in reqs.values()])
    return out


def _async_env(batched, passes=0, ms=0):
    return (
        envs.SGLANG_LINKER_PREPIN_DEPTH.override(8),
        envs.SGLANG_LINKER_ASYNC_ADMISSION.override(True),
        envs.SGLANG_LINKER_ASYNC_BATCHED_RELAY.override(batched),
        envs.SGLANG_LINKER_SKIP_FALLBACK_PASSES.override(passes),
        envs.SGLANG_LINKER_SKIP_FALLBACK_MS.override(ms),
        patch(
            "sglang.srt.mem_cache.unified_cache.unified_cache_linker.get_storage_hash_str",
            side_effect=_page_hashes,
        ),
    )


def _stage_pair(component, backend, batched):
    if batched:
        return _verdict_pair(component, backend, _PinningLinker())
    pp0, later = _pp_pair(component, backend, _PinningLinker())
    return pp0, later, None


_SKIP = LINKER_ADMISSION_SKIPPED


def test_batched_relay_admits_a_skipped_hit_once_its_late_prepin_lands(
    full_linker_component,
):
    # Rhea pm-lk4: a hit skipped once never got a verdict again (68,800 skips a
    # minute, every one urgent, no batch): load_back popped its hit marker
    # before skipping, and the window gives verdicts only to hits holding one.
    a, b, c, d, e, f = _async_env(batched=True)
    with a, b, c, d, e, f:
        backend = _LandingPinLinker(stage_restorable=[1, 3])
        pp0, later, relays = _verdict_pair(full_linker_component, backend, _PinningLinker())
        reqs = _hit_pair_reqs(pp0, later, ["r"])
        assert _admission_pass(pp0, later, reqs) == {"r": (_SKIP, _SKIP)}
        backend.landed["r"] = 3  # lands after the first window looked
        assert _admission_pass(pp0, later, reqs) == {"r": (_SKIP, _SKIP)}
        assert pp0.next_verdicts == {"r": (3, 3)}  # this window saw it land
        r0, r5 = reqs["r"]
        out = _admission_pass(pp0, later, reqs)
        assert out["r"][0] is not _SKIP and out["r"][1] is not _SKIP
        assert reqs == {} and backend.pins == [] and relays == []
        assert pp0.skip_counts == later.skip_counts == {}
        assert pp0.async_counts["fallback"] == 0


def test_async_admission_a_skipped_hits_marker_lasts_until_its_next_match(
    full_linker_component,
):
    # The marker a skip keeps feeds the next window only; a match that finds
    # no linker hit (here: the whole key is on the device now) drops it.
    a, b, c, d, e, f = _async_env(batched=True)
    with a, b, c, d, e, f:
        backend = _PinningLinker(stage_restorable=[1, 3])
        pp0, later, _ = _verdict_pair(full_linker_component, backend, _PinningLinker())
        reqs = _hit_pair_reqs(pp0, later, ["r"])
        assert _admission_pass(pp0, later, reqs) == {"r": (_SKIP, _SKIP)}
        assert pp0.has_hit("r") and later.has_hit("r")
        r0, _ = reqs["r"]
        pp0.match(_KEY, r0, _device_result(len(_KEY)))
        assert not pp0.has_hit("r")


@pytest.mark.parametrize("batched", [True, False])
def test_async_admission_recomputes_a_hit_whose_prepin_never_lands(
    full_linker_component, batched
):
    # The progress guarantee: a hit skipped SKIP_FALLBACK_PASSES passes in a
    # row is admitted on every stage alike as a miss of its unready pages.
    a, b, c, d, e, f = _async_env(batched, passes=4)
    with a, b, c, d, e, f:
        backend = _PinningLinker(stage_restorable=[1, 3])  # never lands
        pp0, later, relays = _stage_pair(full_linker_component, backend, batched)
        reqs = _hit_pair_reqs(pp0, later, ["r"])
        r0, r5 = reqs["r"]
        trace = [_admission_pass(pp0, later, reqs)["r"] for _ in range(5)]
        assert trace[:4] == [(_SKIP, _SKIP)] * 4
        assert trace[4] == ((0, "r-node"), (0, "r-node"))
        assert reqs == {} and backend.pins == []
        assert r0.linker_pp_hit == r5.linker_pp_hit == (0, [])
        assert pp0.skip_counts == later.skip_counts == {}
        assert pp0.async_counts["fallback"] == later.async_counts["fallback"] == 1
        assert pp0.async_counts["fallback_pages"] == 3
        assert "r" in backend.unpins
        if batched:
            assert relays == []


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("hole_fallback", [True, False])
def test_hole_fallback_admits_a_holed_hit_as_a_miss_on_every_stage(
    full_linker_component, batched, hole_fallback
):
    # Rhea pm-lk5: a hole returned None, which ended the admission pass for
    # every request behind it (>= 87 of 268 non-full batches). With
    # SGLANG_LINKER_HOLE_FALLBACK the holed hit is recomputed on every stage
    # alike and the hit behind it loads in the same pass.
    a, b, c, d, e, f = _async_env(batched)
    with a, b, c, d, e, f, envs.SGLANG_LINKER_HOLE_FALLBACK.override(hole_fallback):
        backend = _LandingPinLinker(stage_restorable=[1, 3])
        backend.landed.update(h=2, r=3)  # h's pin found page 2 of 3 gone
        traced = []
        backend.trace_hole = lambda rid, transfers, keep, looked_up: traced.append(
            (rid, keep, looked_up is not None)
        )
        pp0, later, relays = _stage_pair(full_linker_component, backend, batched)
        reqs = _hit_pair_reqs(pp0, later, ["h", "r"])
        h0, h5 = reqs["h"]
        out = _admission_pass(pp0, later, reqs)
        if batched:
            assert out == {"h": (_SKIP, _SKIP), "r": (_SKIP, _SKIP)}
            backend.landed.update(h=2, r=3)  # peeked, not yet taken
            out = _admission_pass(pp0, later, reqs)
        if hole_fallback:
            assert out["h"] == ((0, "h-node"), (0, "h-node"))
            assert h0.linker_pp_hit == h5.linker_pp_hit == (0, [])
            assert pp0.async_counts["hole_fallback"] == 1
            assert later.async_counts["hole_fallback"] == 1
            assert pp0.async_counts["hole_fallback_pages"] == 3
            assert backend.stage_lookups == 2  # the matches: no re-lookup
        else:
            assert out["h"] == (None, None)
            assert pp0.async_counts["hole_fallback"] == 0
            assert h0.linker_pp_hit == h5.linker_pp_hit == (0, [2])
            assert backend.stage_lookups == 3
        assert out["r"][0] == out["r"][1] and out["r"][0][0] > 0
        assert "h" in backend.unpins and pp0.pp_load_holes == later.pp_load_holes == 1
        assert traced == [("h", 2, True)]  # SGLANG_LINKER_HOLE_TRACE's hook


def test_async_admission_recomputes_a_hit_skipped_longer_than_the_time_bound(
    full_linker_component,
):
    a, b, c, d, e, f = _async_env(batched=True, passes=0, ms=5)
    with a, b, c, d, e, f:
        backend = _PinningLinker(stage_restorable=[1, 3])
        pp0, later, _ = _verdict_pair(full_linker_component, backend, _PinningLinker())
        reqs = _hit_pair_reqs(pp0, later, ["r"])
        assert _admission_pass(pp0, later, reqs) == {"r": (_SKIP, _SKIP)}
        assert _admission_pass(pp0, later, reqs) == {"r": (_SKIP, _SKIP)}
        time.sleep(0.01)
        assert _admission_pass(pp0, later, reqs) == {"r": (_SKIP, _SKIP)}
        assert _admission_pass(pp0, later, reqs) == {"r": ((0, "r-node"), (0, "r-node"))}


def test_async_admission_fallback_leaves_a_landed_prepin_to_load(
    full_linker_component,
):
    # A pin that lands inside the bound loads its pages; nothing falls back.
    a, b, c, d, e, f = _async_env(batched=True, passes=4, ms=200)
    with a, b, c, d, e, f:
        backend = _LandingPinLinker(stage_restorable=[1, 3])
        pp0, later, _ = _verdict_pair(full_linker_component, backend, _PinningLinker())
        reqs = _hit_pair_reqs(pp0, later, ["r"])
        for _ in range(3):
            assert _admission_pass(pp0, later, reqs) == {"r": (_SKIP, _SKIP)}
        backend.landed["r"] = 3
        assert _admission_pass(pp0, later, reqs) == {"r": (_SKIP, _SKIP)}
        assert pp0.skip_counts["r"] == 4  # at the bound, but its pin is ready
        out = _admission_pass(pp0, later, reqs)
        assert out["r"][0] not in (_SKIP, (0, "r-node"))
        assert out["r"][0] == out["r"][1]
        assert pp0.async_counts["fallback"] == 0


@pytest.mark.parametrize("batched", [True, False])
def test_async_admission_keeps_admitting_when_every_hit_is_unready(
    full_linker_component, batched
):
    # pm-lk4's shape: a deep queue of hits, no pre-pin ever ready. Every hit
    # is admitted within SKIP_FALLBACK_PASSES + 1 passes, the same on both
    # stages, and from then on no pass with waiting hits admits none.
    a, b, c, d, e, f = _async_env(batched, passes=4)
    with a, b, c, d, e, f:
        backend = _PinningLinker(stage_restorable=[1, 3])
        pp0, later, _ = _stage_pair(full_linker_component, backend, batched)
        rids = [f"r{i}" for i in range(12)]
        reqs = _hit_pair_reqs(pp0, later, rids)
        admitted_per_pass = []
        for _ in range(5):
            out = _admission_pass(pp0, later, reqs)
            assert all(p0 == p5 for p0, p5 in out.values())
            admitted_per_pass.append(sum(p0 is not _SKIP for p0, _ in out.values()))
        assert admitted_per_pass == [0, 0, 0, 0, 12]
        assert reqs == {}
        assert pp0.async_counts["fallback"] == 12
        assert pp0.async_counts["fallback_pages"] == 36


def test_async_admission_prepins_the_admission_window_and_ends_the_rest(
    full_linker_component,
):
    with (
        envs.SGLANG_LINKER_PREPIN_DEPTH.override(8),
        envs.SGLANG_LINKER_ASYNC_ADMISSION.override(True),
        patch(
            "sglang.srt.mem_cache.unified_cache.unified_cache_linker.get_storage_hash_str",
            side_effect=_page_hashes,
        ),
    ):
        backend = _PinningLinker(stage_restorable=[1, 3])
        kept = []
        backend.keep_prepins = lambda rids, urgent=(): kept.append(list(rids))
        pp0 = _match_wrapper(
            full_linker_component,
            backend,
            pp_size=8,
            pp_rank=0,
            _all_reduce=lambda data, op: None,
        )
        reqs = [
            SimpleNamespace(rid=r, linker_pp_hit=None, linker_pp_lookup_pending=True)
            for r in "abcdef"
        ]
        for req in reqs:
            pp0.match(_KEY, req, _device_result(0))
        miss = SimpleNamespace(rid="none", linker_pp_hit=None, linker_pp_lookup_pending=False)
        # No admissions seen yet: the window is the floor of two hits.
        pp0.prepin([miss] + reqs)
        assert kept[-1] == ["a", "b"]
        # Three hits a pass for a while: the window follows (2 x 3 + 1 = 7).
        for _ in range(20):
            pp0.pass_hits = 3
            pp0.prepin(reqs)
        assert kept[-1] == list("abcdef")
        assert pp0.hits_per_pass > 2.9
        # The policy reorders the queue: the window is the new head.
        pp0.hits_per_pass, pp0.pass_hits = 0.0, 0
        pp0.prepin(list(reversed(reqs)))
        assert kept[-1] == ["f", "e"]


def test_pp_fresh_hit_loads_without_a_second_lookup(full_linker_component):
    with patch(
        "sglang.srt.mem_cache.unified_cache.unified_cache_linker.get_storage_hash_str",
        side_effect=_page_hashes,
    ):
        backend = _CountingLinker(stage_restorable=[1, 3])
        pp0 = _match_wrapper(
            full_linker_component,
            backend,
            pp_size=8,
            pp_rank=0,
            _all_reduce=lambda data, op: None,
        )
        req = SimpleNamespace(rid="r", linker_pp_hit=None, linker_pp_lookup_pending=True)
        pp0.match(_KEY, req, _device_result(0))
        assert pp0._pp_fresh_hit_pages(req, pp0.hit_markers["r"]) == 3
        assert backend.stage_lookups == 1
        assert pp0.pp_relookups == 0


def test_later_pp_stage_takes_pp0s_relookup_without_a_store_query(
    full_linker_component,
):
    with patch(
        "sglang.srt.mem_cache.unified_cache.unified_cache_linker.get_storage_hash_str",
        side_effect=_page_hashes,
    ):
        backend = _CountingLinker()
        later = _match_wrapper(
            full_linker_component,
            backend,
            pp_size=8,
            pp_rank=5,
            _all_reduce=_relayed(1),
        )
        req = SimpleNamespace(
            rid="r", linker_pp_hit=(0, [2, 6]), linker_pp_lookup_pending=False
        )
        assert later.match(_KEY, req, _device_result(0)).host_hit_length == 6
        assert later.load_back(req) is None
        assert req.linker_pp_hit == (0, [2])
        assert (backend.lookups, backend.stage_lookups) == (0, 0)


@pytest.mark.parametrize("pp_rank", [0, 3])
def test_linker_pp_enqueue_matches_once_and_only_pp0_looks_up(pp_rank):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.pp_rank = pp_rank
    pending_at_match = []
    req = SimpleNamespace(linker_pp_lookup_pending=False)
    req.init_next_round_input = lambda tree_cache, cow_mamba: pending_at_match.append(
        (tree_cache is cache, cow_mamba, req.linker_pp_lookup_pending)
    )
    cache.linker_pp_enqueue(req)
    assert pending_at_match == [(True, False, pp_rank == 0)]
    assert req.linker_pp_lookup_pending is False


def test_pp_offload_commits_follow_pp0():
    # Offloads lock their nodes until committed, so every stage commits PP0's
    # count and verdicts; a later stage first waits for its own puts.
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.pp_size, cache.pp_rank = 8, 3
    waited, committed = [], []
    cache.linker = SimpleNamespace(
        num_completed_offloads=lambda: 1,
        wait_completed_offloads=waited.append,
        take_completed_offloads=lambda count: [True] * count,
        commit_completed_offloads=committed.append,
    )
    cache._drain_async_work = lambda: None
    from_pp0 = [torch.tensor([3]), torch.tensor([1, 0, 1])]

    def receive_from_pp0(value, op):
        assert op == torch.distributed.ReduceOp.MIN
        value.copy_(from_pp0.pop(0))

    cache._all_reduce = receive_from_pp0

    cache.check_hicache_events()

    assert waited == [3]
    assert committed == [[True, False, True]]
    assert not from_pp0


def _piggyback_stage(pp_rank, completed, local_results=None):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.pp_size, cache.pp_rank = 8, pp_rank
    cache.pp_counts_piggyback = True
    cache.pp_counts_in, cache.pp_counts_out = deque(), []
    log = {"waited": [], "committed": []}
    cache.linker = SimpleNamespace(
        num_completed_offloads=lambda: completed,
        wait_completed_offloads=log["waited"].append,
        take_completed_offloads=lambda count: (local_results or [True] * count)[:count],
        commit_completed_offloads=log["committed"].append,
    )

    def no_pp_comm(*args):
        raise AssertionError("piggyback mode must not touch the PP chain")

    cache._drain_async_work = no_pp_comm
    cache._all_reduce = no_pp_comm
    cache._all_reduce_attn_groups = lambda data, op: None
    return cache, log


def test_pp_offload_counts_piggyback_chain_follows_pp0():
    # SGLANG_LINKER_PP_COUNTS_PIGGYBACK: PP0's commits ride the request relay;
    # every stage commits PP0's count and verdicts, with no per-step PP comm.
    pp0, log0 = _piggyback_stage(0, completed=2, local_results=[True, False])
    pp0.check_hicache_events()
    assert log0 == {"waited": [], "committed": [[True, False]]}
    assert pp0.pp_counts_out == [[2, 1, 0]]

    mid, log_mid = _piggyback_stage(3, completed=5, local_results=[True, True])
    mid.pp_counts_in.extend(pp0.pp_counts_out)
    mid.check_hicache_events()
    assert log_mid == {"waited": [2], "committed": [[True, False]]}
    assert mid.pp_counts_out == [[2, 1, 0]] and not mid.pp_counts_in

    last, log_last = _piggyback_stage(7, completed=0)
    last.pp_counts_in.extend(mid.pp_counts_out)
    last.check_hicache_events()
    assert log_last == {"waited": [2], "committed": [[True, False]]}
    assert last.pp_counts_out == []


def test_pp_offload_counts_piggyback_relays_a_row_every_step():
    # A zero-count step still relays a row, so later stages consume one per step.
    pp0, log0 = _piggyback_stage(0, completed=0)
    pp0.check_hicache_events()
    assert pp0.pp_counts_out == [[0]] and log0["committed"] == []
    later, log = _piggyback_stage(2, completed=4)
    later.pp_counts_in.extend(pp0.pp_counts_out)
    later.check_hicache_events()
    assert log == {"waited": [], "committed": []}
    assert later.pp_counts_out == [[0]]


def test_pp_offload_counts_piggyback_requires_the_relayed_row():
    later, _ = _piggyback_stage(4, completed=1)
    with pytest.raises(AssertionError, match="missing from the PP relay"):
        later.check_hicache_events()


class _FakeConsensusStore:
    """PPConsensusStore stand-in: every rank's map, PP0 collects them all."""

    def __init__(self, maps, rank):
        self.maps, self.rank = maps, rank

    def __setitem__(self, key, value):
        self.maps[self.rank][key] = value

    def pop(self, key, default=None):
        return self.maps[self.rank].pop(key, default)

    def collect(self, key):
        return [m.get(key) for m in self.maps]


def _consensus_stages(size, finished):
    """Relays of ``size`` stages over one fake store; ``finished[r]`` is the list
    of offload outcomes rank r's backend completes (handed out on demand)."""
    from sglang.srt.mem_cache.unified_cache.pp_linker_relay import PPLinkerRelay

    maps = [dict() for _ in range(size)]
    stages = []
    for rank in range(size):
        log = {"waited": [], "committed": []}
        queue_ = list(finished[rank])
        w = SimpleNamespace(pp_rank=rank, pp_size=size, log_lookup_stats=True)
        w.num_completed_offloads = lambda q=queue_: len(q)
        w.take_completed_offloads = lambda n, q=queue_: [q.pop(0) for _ in range(n)]
        w.wait_completed_offloads = log["waited"].append
        w.commit_completed_offloads = lambda s, log=log: log["committed"].append(list(s))
        relay = PPLinkerRelay(w)
        relay.attach_store(_FakeConsensusStore(maps, rank))
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.pp_size, cache.pp_rank = size, rank
        cache.pp_counts_consensus = True
        cache.linker = SimpleNamespace(pp_relay=relay)

        def no_pp_comm(*args):
            raise AssertionError("consensus mode must not touch the PP chain")

        cache._drain_async_work = no_pp_comm
        cache._all_reduce = no_pp_comm
        stages.append((cache, relay, log, queue_))
    return stages, maps


def _consensus_step(stages):
    """One loop iteration: every stage reports (PP0 also decides), then PP0's
    actions are delivered to every stage in order. Later stages run first so PP0
    sees this iteration's posts (in a server they land a step later)."""
    for cache, *_ in reversed(stages):
        cache._check_linker_events_pp()
    actions = stages[0][1].take_actions()
    if actions is not None:
        for _, relay, _, _ in stages:
            relay.receive(actions.actions)


def test_pp_offload_consensus_commits_the_min_without_waiting():
    # SGLANG_LINKER_PP_COUNTS_CONSENSUS: PP0 commits only what every stage has
    # finished; a lagging stage holds the commit back instead of blocking.
    stages, _ = _consensus_stages(3, [[True] * 3, [True] * 3, [True]])
    _consensus_step(stages)
    assert [log["committed"] for _, _, log, _ in stages] == [[[True]]] * 3
    stages[2][3].extend([True, True])  # the lagging stage's puts land
    _consensus_step(stages)
    assert [log["committed"] for _, _, log, _ in stages] == [[[True], [True, True]]] * 3
    assert all(not log["waited"] for _, _, log, _ in stages)
    _consensus_step(stages)  # nothing new: no action, nothing committed
    assert [len(log["committed"]) for _, _, log, _ in stages] == [2, 2, 2]


def test_pp_offload_consensus_verdict_is_the_and_across_stages():
    stages, maps = _consensus_stages(3, [[True, True, True], [True, False, True], [False, True, True]])
    _consensus_step(stages)
    assert [log["committed"] for _, _, log, _ in stages] == [[[False, False, True]]] * 3
    # Committed failure votes are cleared from the store.
    assert all(not any(k.startswith("lko0:") for k in m) for m in maps)


def test_pp_offload_consensus_commit_order_follows_the_absolute_head():
    stages, _ = _consensus_stages(2, [[True, False], [True]])
    _consensus_step(stages)
    stages[1][3].append(True)
    _consensus_step(stages)
    for _, relay, log, _ in stages:
        assert log["committed"] == [[True], [False]]
        assert relay.offload_head == 2 and relay.offload_results == []


def test_pp_offload_consensus_flush_drops_stale_votes():
    # A flush (relay reset) between the stages' reports and PP0's decision:
    # nothing from the old epoch is committed, and its store votes are cleared.
    stages, maps = _consensus_stages(2, [[False], [True]])
    for cache, *_ in stages:
        cache.linker.pp_relay.offload_report()
    assert maps[0] == {"lko0:0": 0, "lkoc0": 1, "lkoe": 0, "lkos": 0}
    for _, relay, _, _ in stages:
        relay.reset()
    assert not any(k.startswith(("lko0:", "lkoc0")) for m in maps for k in m)
    _consensus_step(stages)
    assert all(log["committed"] == [] for _, _, log, _ in stages)
    assert stages[0][1].take_actions() is None


def test_pp_offload_consensus_waits_for_every_stage_to_ack():
    # GLM's acknowledgement: PP0 decides no offload while a stage has not
    # applied every action issued so far.
    stages, _ = _consensus_stages(2, [[True, True], [True]])
    _consensus_step(stages)
    pp0, later = stages[0][1], stages[1][1]
    assert later.offload_serial == 1
    stages[1][3].append(True)
    later.offload_serial = 0  # pretend stage 1 has not applied offload action 1 yet
    later._ack_posted = None
    stages[1][0]._check_linker_events_pp()
    stages[0][0]._check_linker_events_pp()
    assert pp0.take_actions() is None


def test_pp_offload_consensus_advances_under_continuous_prefetch_actions():
    # Prefetch actions issued every step, delivered to stage 1 two steps late,
    # never hold offload commits back: they are gated on offload actions only.
    from sglang.srt.mem_cache.unified_cache.pp_linker_relay import _Request

    stages, _ = _consensus_stages(2, [[True] * 40, [True] * 40])
    for _, relay, _, _ in stages:
        relay.entries["h"] = _Request(req=None, cycle=1, phase="ready")
    pp0, later = stages[0][1], stages[1][1]
    in_flight = []
    committed = []
    for step in range(12):
        stages[0][3].append(True)
        stages[1][3].append(True)
        pp0._issue("h", pp0.entries["h"], "stop", None)
        for cache, *_ in reversed(stages):
            cache._check_linker_events_pp()
        actions = pp0.take_actions().actions
        pp0.receive(actions)
        in_flight.append(actions)
        if len(in_flight) > 2:
            later.receive(in_flight.pop(0))
        committed.append(pp0.offload_head)
    # Commits keep advancing (once per offload round trip), never starve.
    assert committed[-1] >= 40
    assert all(b >= a for a, b in zip(committed, committed[1:]))
    assert sum(b > a for a, b in zip(committed, committed[1:])) >= 3


def _window_step(stages, deliver):
    """One iteration with a commit window: every stage reports (PP0 also
    decides), PP0 applies its own actions at once, and each later stage r gets
    them queued in ``deliver[r]``; a stage applies its queue only when listed."""
    pp0 = stages[0][1]
    for cache, *_ in reversed(stages):
        cache._check_linker_events_pp()
    actions = pp0.take_actions()
    if actions is not None:
        pp0.receive(actions.actions)
        for queued in deliver.values():
            queued.append(actions.actions)


def _deliver(stages, deliver, ranks):
    for r in ranks:
        while deliver[r]:
            stages[r][1].receive(deliver[r].pop(0))


def test_pp_offload_consensus_window_issues_ahead_of_acks():
    # SGLANG_LINKER_PP_OFFLOAD_COMMIT_WINDOW: PP0 issues up to N commits a stage
    # has not applied yet, never more; delivered late, they apply in order.
    stages, _ = _consensus_stages(2, [[True], [True]])
    pp0, later = stages[0][1], stages[1][1]
    pp0.commit_window = 2
    deliver = {1: []}
    _window_step(stages, deliver)
    for *_, q in stages:
        q.append(True)
    _window_step(stages, deliver)  # stage 1 acked nothing yet: commit 2 still goes out
    assert pp0.offload_issued == 2 and len(deliver[1]) == 2
    for *_, q in stages:
        q.append(True)
    _window_step(stages, deliver)  # two unacknowledged: the window holds commit 3
    assert pp0.offload_issued == 2 and later.offload_head == 0
    _deliver(stages, deliver, [1])
    assert stages[1][2]["committed"] == [[True], [True]] and later.offload_head == 2
    _window_step(stages, deliver)  # stage 1's ack arrives with its next report
    assert pp0.offload_issued == 3
    _deliver(stages, deliver, [1])
    assert [log["committed"] for _, _, log, _ in stages] == [[[True], [True], [True]]] * 2


def test_pp_offload_consensus_window_gates_on_the_slowest_ack():
    # Out-of-order acknowledgements: stage 2 applies every commit at once, stage
    # 1 lags; the window counts from the slowest stage.
    stages, _ = _consensus_stages(3, [[True]] * 3)
    pp0 = stages[0][1]
    pp0.commit_window = 2
    deliver = {1: [], 2: []}
    _window_step(stages, deliver)
    _deliver(stages, deliver, [2])
    for _ in range(2):
        for *_, q in stages:
            q.append(True)
        _window_step(stages, deliver)
        _deliver(stages, deliver, [2])
    assert pp0.offload_issued == 2
    assert stages[2][1].offload_head == 2 and stages[1][1].offload_head == 0
    _deliver(stages, deliver, [1])
    _window_step(stages, deliver)
    assert pp0.offload_issued == 3
    _deliver(stages, deliver, [1, 2])
    assert [log["committed"] for _, _, log, _ in stages] == [[[True], [True], [True]]] * 3


def test_pp_offload_consensus_window_reset_drops_commits_in_flight():
    # A reset between PP0's issue and a stage's delivery: the old epoch's
    # commits are dropped there, and the new epoch commits from head 0.
    stages, maps = _consensus_stages(2, [[True], [True]])
    pp0, later = stages[0][1], stages[1][1]
    pp0.commit_window = 4
    deliver = {1: []}
    _window_step(stages, deliver)
    for *_, q in stages:
        q.append(True)
    _window_step(stages, deliver)
    assert pp0.offload_issued == 2 and len(deliver[1]) == 2
    for _, relay, _, _ in stages:
        relay.reset()
    _deliver(stages, deliver, [1])
    assert stages[1][2]["committed"] == [] and later.offload_head == 0
    assert not any(k.startswith(("lko0:", "lkoc0")) for m in maps for k in m)
    for *_, q in stages:
        q.append(True)
    _window_step(stages, deliver)
    _deliver(stages, deliver, [1])
    assert stages[1][2]["committed"] == [[True]]
    assert pp0.offload_head == later.offload_head == 1 and pp0.epoch == later.epoch == 1


def test_pp_offload_consensus_window_of_one_waits_like_before():
    stages, _ = _consensus_stages(2, [[True], [True]])
    pp0 = stages[0][1]
    assert pp0.commit_window == 1
    deliver = {1: []}
    _window_step(stages, deliver)
    for *_, q in stages:
        q.append(True)
    _window_step(stages, deliver)
    assert pp0.offload_issued == 1


def test_pp_offload_consensus_logs_locked_tokens_by_wait(caplog):
    # PP0's locked tokens split by what each uncommitted offload waits for:
    # index 0 finished everywhere (commit), 1 on PP0 only (other stages' puts),
    # 2 not on PP0 (its own put).
    import logging

    stages, _ = _consensus_stages(2, [[True, True], [True]])
    pp0 = stages[0][1]
    pp0.w.pending_offloads = [SimpleNamespace(lock_node_id=k) for k in (1, 2, 3)]
    pp0.w.pending_offload_tokens = {1: 10, 2: 20, 3: 40}
    pp0._stats_next = 0
    with caplog.at_level(logging.INFO):
        _consensus_step(stages)
    assert "at peak 40 / 20 / 10, mean 40 / 20 / 10" in caplog.text


def test_pp_offload_consensus_logs_which_stage_holds_the_minimum(caplog):
    # PP0 records, per stage, each offload's finish lag behind PP0 and the steps
    # that stage held "finished on all" back; a reset starts the record over.
    import logging

    stages, _ = _consensus_stages(3, [[True, True], [True, True], [True]])
    pp0 = stages[0][1]
    _consensus_step(stages)
    assert [len(x) for x in pp0._finish_lag] == [2, 2, 1]
    assert pp0._min_steps == [0, 0, 1]
    stages[2][3].append(True)
    pp0._stats_next = 0
    with caplog.at_level(logging.INFO):
        _consensus_step(stages)
    assert "finish lag behind PP0" in caplog.text and "PP2 " in caplog.text
    assert pp0._finish_lag == [[], [], []] and pp0._min_steps == [0, 0, 0]
    for _, relay, _, _ in stages:
        relay.reset()
    assert pp0._own_base == 0 and pp0._stage_seen == [0, 0, 0]


@pytest.mark.parametrize("swa_stored", [True, False], ids=["stored", "unstored"])
@pytest.mark.parametrize(
    "completion", [True, False, None], ids=["success", "failure", "reset"]
)
def test_offload_filters_tree_core_swa_transfers_and_preserves_lifecycle(
    swa_stored, completion
):
    node = SimpleNamespace(
        id=7, external_cache_stored=False, write_through_pending_id=None
    )
    transfers = [
        PoolTransfer(name=PoolName.KV, keys=["page"]),
        PoolTransfer(name=PoolName.SWA, keys=["page"]),
    ]
    core = _FakeExternalTreeCore({node.id: node}, transfers)
    swa = SWAComponent.__new__(SWAComponent)
    lock_params = object()
    cache = _cache_for_wrapper(
        components={ComponentType.SWA: swa},
        _components_tuple=(swa,),
        tree_core=core,
        inc_lock_ref=MagicMock(
            return_value=SimpleNamespace(to_dec_params=lambda: lock_params)
        ),
        dec_lock_ref=MagicMock(),
    )
    backend = _FakeLinker()
    if swa_stored:
        backend.pools[PoolName.SWA] = object()
    wrapper = UnifiedCacheLinkerWrapper(cache, backend)

    wrapper.offload_nodes([node.id, node.id])
    expected = transfers if swa_stored else transfers[:1]
    assert backend.queued_offloads == [expected]
    assert core.offload_transfers == transfers
    assert node.write_through_pending_id == node.id
    assert not node.external_cache_stored
    cache.inc_lock_ref.assert_called_once_with(node.id)
    cache.dec_lock_ref.assert_not_called()

    if completion is None:
        wrapper.reset()
        assert backend.reset_count == 1
    else:
        backend.completed_offloads.append(completion)
        wrapper.commit_completed_offloads(wrapper.take_completed_offloads(1))
    assert not wrapper.pending_offloads
    assert node.write_through_pending_id is None
    assert node.external_cache_stored is (completion is True)
    cache.dec_lock_ref.assert_called_once_with(node.id, lock_params)
    if completion is True:
        wrapper.offload_nodes([node.id])
        assert backend.queued_offloads == [expected]
    else:
        wrapper.offload_nodes([node.id])
        assert backend.queued_offloads == [expected, expected]
        wrapper.reset()


@pytest.mark.parametrize(
    "swa_stored,previous_boundary,expected_boundary",
    [
        pytest.param(False, None, 4, id="unstored-tombstones"),
        pytest.param(False, 8, 8, id="preserve-existing-boundary"),
        pytest.param(True, None, 2, id="stored-prepare-boundary"),
    ],
)
@pytest.mark.parametrize("load_refused", [False, True], ids=["queued", "refused"])
def test_linker_load_preserves_swa_boundaries(
    full_linker_component,
    swa_stored,
    previous_boundary,
    expected_boundary,
    load_refused,
):
    full = full_linker_component
    swa = SWAComponent.__new__(SWAComponent)
    participates = swa_stored

    def prepare(phase, req, full_transfer, transfer, prefix_len, **kwargs):
        if phase == ExternalLinkerLoadPhase.PREPARE:
            req.kv = ReqKvInfo(
                kv_allocated_len=prefix_len,
                component_evicted_seqlens={ComponentType.SWA: prefix_len - 2},
            )
        return transfer

    swa.build_external_linker_transfer = MagicMock(
        return_value=PoolTransfer(
            name=PoolName.SWA, keys=["a", "b"], device_indices=torch.arange(20, 24)
        )
    )
    swa.update_external_linker_load = MagicMock(side_effect=prepare)
    cache = _cache_for_wrapper(
        _components_tuple=(full, swa),
        page_size=2,
        components={ComponentType.FULL: full, ComponentType.SWA: swa},
        insert=MagicMock(side_effect=AssertionError("the tail must stay private")),
    )
    backend = _FakeLinker()
    backend.load = MagicMock(return_value=not load_refused)
    if swa_stored:
        backend.pools[PoolName.SWA] = object()
    wrapper = UnifiedCacheLinkerWrapper(cache, backend)
    wrapper.hit_markers["rid"] = ExternalCacheHitMarker(
        tail_hashes=["a", "b"],
        device_hit_len=0,
    )
    kv = (
        None
        if previous_boundary is None
        else ReqKvInfo(
            kv_allocated_len=previous_boundary,
            component_evicted_seqlens={ComponentType.SWA: previous_boundary},
        )
    )
    req = SimpleNamespace(
        rid="rid",
        kv=kv,
        prefix_len=0,
        last_node=0,
        priority=0,
        swa_branching_seqlen=2,  # inside the loaded tail
    )
    restored_len, last_node = wrapper.load_back(req)

    assert last_node == 0
    if load_refused:
        # Refused before PREPARE: allocations undone, request untouched, recompute.
        assert restored_len == 0 and req.kv is kv
        assert wrapper.inflight_load_rids == []
        assert wrapper.private_load_indices == {}
        phases = [c.args[0] for c in swa.update_external_linker_load.call_args_list]
        assert phases == ([ExternalLinkerLoadPhase.ABORT] if participates else [])
        return
    assert restored_len == 4
    restored = wrapper.private_load_indices["rid"]
    assert restored.tolist() == list(range(4))
    assert wrapper.inflight_load_rids == ["rid"]
    assert torch.equal(wrapper.private_load_indices["rid"], restored)
    assert req.swa_branching_seqlen is None
    assert req.kv.get_evicted_seqlen(ComponentType.SWA) == expected_boundary
    assert req.kv.kv_allocated_len == (previous_boundary or 4)
    assert [c.args[0] for c in full.build_external_linker_transfer.call_args_list] == [
        LinkerTransferPhase.LOAD
    ]
    if not participates:
        swa.build_external_linker_transfer.assert_not_called()
        swa.update_external_linker_load.assert_not_called()


class TestOffloadReadyEvents(CustomTestCase):
    def _record(self, forward_stream):
        device = MagicMock(name="device_module")
        device.Event.side_effect = lambda: MagicMock(name="event")
        linker = _FakeLinker()
        linker.forward_stream = forward_stream
        with patch(
            "sglang.srt.mem_cache.unified_cache.unified_cache_linker.device_module",
            device,
        ):
            return linker.record_offload_ready()

    def test_offload_waits_on_the_forward_writing_the_pages(self):
        forward_stream = MagicMock(name="forward_stream")
        indices_ready, kv_written = self._record(forward_stream)
        indices_ready.record.assert_called_once_with()
        kv_written.record.assert_called_once_with(forward_stream)
        # The offload thread waits; the forward stream's own order is untouched.
        self.assertEqual(forward_stream.mock_calls, [])

    def test_offload_without_a_forward_stream_waits_on_the_current_one(self):
        (indices_ready,) = self._record(None)
        indices_ready.record.assert_called_once_with()

    def test_mooncake_offload_thread_waits_on_every_event(self):
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        order = MagicMock()
        linker.storage = order.storage
        order.storage.batch_set_v2.return_value = {}
        order.storage.last_put_objects = {}
        order.storage.last_exist_skipped = 0
        linker.stats = {"offload": 1}
        linker._init_offload_queue()
        linker.object_bytes, linker.write_bytes = {PoolName.KV: 1}, Counter()
        linker.stats_logged_at = time.monotonic()
        transfer = PoolTransfer(name=PoolName.KV, keys=[])
        linker.offload_queue.put(
            ([transfer], 1, [order.indices_ready, order.kv_written], "session-a", [])
        )
        linker.offload_queue.put(None)
        linker.offload_thread_func()
        self.assertEqual(
            order.mock_calls,
            [
                call.indices_ready.synchronize(),
                call.kv_written.synchronize(),
                call.storage.batch_set_v2(
                    [transfer],
                    HiCacheStorageExtraInfo(extra_info={"session_id": "session-a"}),
                ),
            ],
        )
        self.assertTrue(linker.offload_results.get_nowait())

    def _threaded_linker(self, delays, log):
        """Two offload threads over a store whose put of key k sleeps delays[k]."""
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)

        def batch_set_v2(transfers, extra_info):
            key = transfers[0].keys[0]
            log.append(("put", key))
            time.sleep(delays[key])
            log.append(("landed", key))
            return {t.name: [key != "bad"] * len(t.keys) for t in transfers}

        def remove(key):
            log.append(("remove", key))
            return 0

        linker.storage = SimpleNamespace(
            batch_set_v2=batch_set_v2, last_put_objects={}, last_exist_skipped=0,
            store=SimpleNamespace(remove=remove),
        )
        linker.stats, linker.stats_logged_at = {"offload": 1}, time.monotonic()
        linker.object_bytes, linker.write_bytes = {PoolName.KV: 1}, Counter()
        linker.store_tree = SimpleNamespace(is_recalled=lambda page, decided: False)
        linker.page_remove_retries, linker.page_remove_counts = deque(), Counter()
        linker._init_offload_queue()
        return linker

    def _run_threads(self, linker, tasks, threads=2):
        for task in tasks:
            linker.offload_queue.put(task)
        for _ in range(threads):
            linker.offload_queue.put(None)
        workers = [threading.Thread(target=linker.offload_thread_func) for _ in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=10)
        return [linker.offload_results.get_nowait() for _ in range(linker.offload_results.qsize())]

    def test_offload_threads_hand_results_on_in_queue_order(self):
        # SGLANG_LINKER_OFFLOAD_THREADS=2: a slow first put lands after a fast
        # second one, and the results still leave in queue order.
        log = []
        linker = self._threaded_linker({"slow": 0.2, "bad": 0.0}, log)
        put = lambda key: ([PoolTransfer(name=PoolName.KV, keys=[key])], 1, [], None, [])
        results = self._run_threads(linker, [put("slow"), put("bad")])
        self.assertLess(log.index(("landed", "bad")), log.index(("landed", "slow")))
        self.assertEqual(results, [True, False])

    def test_offload_threads_remove_only_after_the_puts_queued_before(self):
        # A removal waits for every task queued before it, here a slow put that
        # may write the same objects, even though another thread is free.
        log = []
        linker = self._threaded_linker({"slow": 0.2}, log)
        put = ([PoolTransfer(name=PoolName.KV, keys=["slow"])], 1, [], None, [])
        results = self._run_threads(linker, [put, ("remove", ["slow_k"], ["slow"], 0)])
        self.assertLess(log.index(("landed", "slow")), log.index(("remove", "slow_k")))
        self.assertEqual(results, [True])
        self.assertEqual(linker.page_remove_counts["removed"], 1)


# ---- prefetch mode (SGLANG_LINKER_LOAD_MODE=prefetch, GLM's model) ----


def _done(value):
    future = Future()
    future.set_result(value)
    return future


def _prefetch_wrapper(components, backend, **cache_kwargs):
    cache = _cache_for_wrapper(
        _components_tuple=tuple(components),
        components={c.component_type: c for c in components},
        tree_core=SimpleNamespace(enable_external_cache_linker=False, is_eagle=False),
        page_size=2,
        _all_reduce_attn_groups=lambda value, op: None,
        get_last_hash_value=lambda node: f"anchor-{node}",
        token_to_kv_pool_allocator=SimpleNamespace(size_full=1000),
        req_to_token_pool=SimpleNamespace(),
        tp_world_size=1,
        inc_lock_ref=lambda node_id: SimpleNamespace(to_dec_params=lambda: "lock"),
        dec_lock_ref=MagicMock(),
        **cache_kwargs,
    )
    with envs.SGLANG_LINKER_LOAD_MODE.override("prefetch"):
        return UnifiedCacheLinkerWrapper(cache, backend)


def _state_component(freed):
    def build_transfer(phase, node, keys):
        transfer = PoolTransfer(name=PoolName.MAMBA, keys=[keys[-1]])
        if phase == LinkerTransferPhase.LOAD:
            transfer.device_indices = torch.tensor([7])
        return transfer

    def update(phase, req, full_transfer, transfer, prefix_len):
        freed.append((PoolName.MAMBA, phase))

    return SimpleNamespace(
        component_type=ComponentType.MAMBA,
        build_external_linker_transfer=build_transfer,
        update_external_linker_load=update,
    )


class _PrefetchLinker(_CountingLinker):
    def __init__(self, landed=True):
        super().__init__(restorable=[2])
        self.landed = landed
        self.prefetched = []

    def prefetch(self, transfers):
        self.prefetched.append(list(transfers))
        return _done(self.landed)


def test_prefetch_mode_is_off_by_default(full_linker_component):
    backend = _CountingLinker(restorable=[2])
    wrapper = _match_wrapper(full_linker_component, backend)
    assert wrapper.load_mode == "layer_wise"
    # Off, prefetch() touches neither the request nor the store.
    assert wrapper.prefetch(SimpleNamespace()) is None
    wrapper.check_prefetch_events()
    assert not wrapper.has_pending_prefetch()
    wrapper.match(_KEY, SimpleNamespace(rid="r"), _device_result(0))
    assert backend.lookups == 1


def test_prefetch_mode_matches_only_published_device_kv(full_linker_component):
    backend = _PrefetchLinker()
    wrapper = _prefetch_wrapper([full_linker_component], backend)
    result = _device_result(2)
    assert wrapper.match(_KEY, SimpleNamespace(rid="r"), result) is result
    assert backend.lookups == 0


def test_prefetch_state_cap_bounds_reads_holding_a_state_slot(full_linker_component):
    backend = _PrefetchLinker()
    with envs.SGLANG_LINKER_PREFETCH_MAX_STATES.override(1):
        wrapper = _prefetch_wrapper(
            [full_linker_component, _state_component([])], backend
        )
    handles = [_Handle("a"), _Handle("b")]
    for handle in handles:
        wrapper._start_prefetch_load(
            handle, _KEY, ["h1", "h2", "h3"], 0, 0, "lock", [2], 1.0
        )
    assert list(wrapper.prefetches) == handles[:1]
    assert wrapper._lookup_stats["state_capped"] == 1
    wrapper.cache.dec_lock_ref.assert_called_once_with(0, "lock")
    assert len(backend.prefetched) == 1


@pytest.mark.parametrize("landed", [True, False])
def test_prefetch_publishes_a_landed_read_and_frees_a_failed_one(
    full_linker_component, landed
):
    freed = []
    backend = _PrefetchLinker(landed=landed)
    inserted = []
    wrapper = _prefetch_wrapper(
        [full_linker_component, _state_component(freed)],
        backend,
        path_device_indices=lambda node: torch.tensor([], dtype=torch.int64),
        insert=lambda params: inserted.append(params) or SimpleNamespace(mamba_exist=False),
        prefetch_loaded_tokens_by_reqid={},
        prefetch_loaded_storage_start_by_reqid={},
    )
    wrapper.cache.req_to_token_pool = SimpleNamespace(
        mamba_pool=SimpleNamespace(replayssm_write_pos=None)
    )
    handle = _Handle("a")
    wrapper._start_prefetch_load(handle, _KEY, ["h1", "h2", "h3"], 0, 0, "lock", [2], 1.0)
    wrapper.check_prefetch_events()
    assert not wrapper.prefetches
    if landed:
        (params,) = inserted
        assert len(params.key) == 4 and params.prev_prefix_len == 0
        assert params.inserted_len == 4 and torch.equal(params.mamba_value, torch.tensor([7]))
        assert wrapper.cache.prefetch_loaded_tokens_by_reqid[handle] == 4
        assert freed == []
    else:
        assert inserted == []
        assert freed == [(PoolName.MAMBA, ExternalLinkerLoadPhase.ABORT)]
    # The anchor lock taken at lookup is released either way.
    wrapper.cache.dec_lock_ref.assert_called_once_with(0, "lock")


class _Handle(NamedTuple):
    rid: str


def test_prefetch_requery_is_capped_per_request(full_linker_component):
    wrapper = _prefetch_wrapper([full_linker_component], _PrefetchLinker())
    wrapper.requery_on_device_loss = True
    handle = _Handle("r")
    req = SimpleNamespace(cache_request_handle=handle, prefix_len=0, rid="r")
    wrapper.match_targets[handle] = 4
    with envs.SGLANG_LINKER_MAX_REQUERIES.override(2):
        wrapper.max_requeries = envs.SGLANG_LINKER_MAX_REQUERIES.get()
    outcomes = []
    for _ in range(4):
        outcomes.append(wrapper._local_requery_after_device_loss(req))
        wrapper._pending_requeries.clear()  # the boundary started its lookup
    assert outcomes == [True, True, False, False]
    assert wrapper._lookup_stats["requeried"] == 2
    assert wrapper._lookup_stats["requery_capped"] == 2
    wrapper._release_prefetch_request("r")
    assert wrapper._requery_counts["r"] == 0


@pytest.mark.parametrize(
    "result, reason", [(False, "read_failed"), ("session_start:-704x2", "session_start:-704x2")]
)
def test_prefetch_abort_counted_by_reason(full_linker_component, result, reason):
    backend = _PrefetchLinker(landed=result)
    wrapper = _prefetch_wrapper(
        [full_linker_component, _state_component([])],
        backend,
        path_device_indices=lambda node: torch.tensor([], dtype=torch.int64),
        insert=lambda params: SimpleNamespace(mamba_exist=False),
        prefetch_loaded_tokens_by_reqid={},
        prefetch_loaded_storage_start_by_reqid={},
    )
    handle = _Handle("a")
    wrapper._start_prefetch_load(handle, _KEY, ["h1", "h2", "h3"], 0, 0, "lock", [2], 1.0)
    wrapper.check_prefetch_events()
    assert wrapper._abort_reasons == {reason: 1}


def test_prefetch_cancel_records_its_reason(full_linker_component):
    backend = _PrefetchLinker(landed=True)
    wrapper = _prefetch_wrapper([full_linker_component, _state_component([])], backend)
    handle = _Handle("a")
    wrapper._start_prefetch_load(handle, _KEY, ["h1", "h2", "h3"], 0, 0, "lock", [2], 1.0)
    wrapper._cancel_prefetch(handle, "timeout")
    wrapper._finish_prefetch(handle, publish=False)
    assert wrapper._abort_reasons == {"timeout": 1}


def _failed_staged_load(fallback_results):
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    future = Future()
    future.staged = True
    future.set_exception(RuntimeError("cudaMemcpyBatchAsync failed"))
    reads = []

    def prefetch(transfers):
        read = Future()
        read.set_result(fallback_results[len(reads)])
        reads.append(transfers)
        return read

    wrapper.cache_linker = SimpleNamespace(prefetch=prefetch)
    wrapper._staged_fallbacks = Counter()
    state = SimpleNamespace(future=future, component_transfers=[(None, "kv"), (None, "mamba")], abort_reason=None)
    return wrapper, state, reads


def test_finish_of_a_failed_staged_load_raises_under_strict():
    # The early read voted for it, so every stage publishes; this stage's slots were never filled.
    wrapper, state, reads = _failed_staged_load([True])
    with patch.object(envs.SGLANG_LINKER_EARLY_STRICT, "get", return_value=True):
        with pytest.raises(RuntimeError, match="failed after its vote"):
            wrapper._staged_load_failed(_Handle("a"), state, True)
    assert reads == []


def test_failed_staged_load_is_read_from_the_store_or_not_published():
    wrapper, state, reads = _failed_staged_load([True])
    assert wrapper._staged_load_failed(_Handle("a"), state, True) is True
    assert reads == [["kv", "mamba"]] and state.future.result() is True
    assert not hasattr(state.future, "copy_event")  # the forward does not wait on the failed copy
    # Both store reads fail: never publish the unfilled slots; counted.
    wrapper, state, reads = _failed_staged_load([False, "session_start:-704x2"])
    assert wrapper._staged_load_failed(_Handle("b"), state, True) is False
    assert len(reads) == 2 and state.abort_reason == "staged_fallback_failed"
    assert wrapper._staged_fallbacks == {"fallback": 1, "fallback_failed": 1}


class _VoteStore:
    """PPConsensusStore stand-in: every rank's map, collected on PP0."""

    def __init__(self, maps, rank):
        self.maps, self.rank = maps, rank

    def __setitem__(self, key, value):
        self.maps[self.rank][key] = value

    def pop(self, key, default=None):
        return self.maps[self.rank].pop(key, default)

    def collect(self, key):
        return [m.get(key) for m in self.maps]


class _RelayWrapper:
    timeline = None
    def __init__(self, rank, size):
        self.pp_rank, self.pp_size = rank, size
        self.lookups, self.prefetches = {}, {}
        self.cache = SimpleNamespace(page_size=2, dec_lock_ref=MagicMock())
        self._draining_lookups = False
        self.started, self.finished, self.cancelled = [], [], []

    def _start_prefetch_load(self, handle, key, hashes, start, anchor, lock, restorable, timeout):
        self.started.append((handle, restorable))
        self.prefetches[handle] = SimpleNamespace(
            future=Future(), discard=False, abort_reason=None
        )

    def _finish_prefetch(self, handle, publish, hold=True):
        self.prefetches.pop(handle, None)
        self.finished.append((handle, publish))

    local_abort_reason = staticmethod(UnifiedCacheLinkerWrapper.local_abort_reason)

    def _cancel_prefetch(self, handle, reason="cancelled"):
        self.cancelled.append(handle)
        self.lookups.pop(handle, None)
        state = self.prefetches.get(handle)
        if state is not None:
            state.discard = True

    def _start_prefetch_from_staging(self, handle, key, hashes, start, anchor, lock, restorable, timeout, staged):
        self.started.append((handle, restorable, "staged", staged))
        future = Future()
        future.set_result(True)  # the H2D copy is queued on the stream: landed for the vote
        self.prefetches[handle] = SimpleNamespace(future=future, discard=False, abort_reason=None)


class _EarlyStub:
    """SGLANG_LINKER_EARLY_READS stand-in: what PP0 published, which staged reads
    landed (the test lands them), staging taken / dropped by (epoch, cycle)."""

    def __init__(self):
        self.published, self.landed, self.dropped = [], [], []
        self.staged = {}
        self.epoch = 0

    def publish(self, epoch, cycle, handle, lookup, pages):
        self.published.append((epoch, cycle, handle, pages))

    def land(self, epoch, cycle, pages):
        self.staged[(epoch, cycle)] = (pages, f"slab-{cycle}")
        self.landed.append((epoch, cycle))

    def take_landed(self):
        out, self.landed = self.landed, []
        return out

    def take(self, epoch, cycle, pages, rid=None):
        staged = self.staged.get((epoch, cycle))
        if staged is None or staged[0] != pages:
            return None
        return self.staged.pop((epoch, cycle))[1]

    def drop(self, epoch, cycle):
        self.staged.pop((epoch, cycle), None)
        self.dropped.append((epoch, cycle))

    def reset(self, epoch):
        self.staged.clear()
        self.epoch = epoch


def _relay_stages(size=2):
    maps = [{} for _ in range(size)]
    stages = []
    for rank in range(size):
        wrapper = _RelayWrapper(rank, size)
        relay = PPLinkerRelay(wrapper)
        relay.attach_store(_VoteStore(maps, rank))
        stages.append((relay, wrapper))
    return stages


def _queue_lookup(stages, handle, restorable):
    for relay, wrapper in stages:
        relay.begin(SimpleNamespace(cache_request_handle=handle))
        wrapper.lookups[handle] = SimpleNamespace(
            key=_KEY,
            hashes=["h1", "h2", "h3"],
            start=0,
            anchor_node_id=0,
            anchor_lock="lock",
            timeout_s=1.0,
            wait_started=None,
            future=_done(restorable) if wrapper.pp_rank == 0 else None,
        )


def _relay_round(stages):
    """Every stage votes and PP0 decides; PP0's actions reach every stage."""
    for relay, _ in reversed(stages):
        relay.poll()
    actions = stages[0][0].take_actions()
    for relay, _ in stages:
        if actions is not None:
            relay.receive(actions.actions)


def _progress(stages, handle, stop_due=lambda h: False):
    return [relay.progress(handle, stop_due) for relay, _ in stages]


@pytest.mark.parametrize("later_landed", [True, False])
def test_pp_prefetch_relay_publishes_only_when_every_stage_landed(later_landed):
    stages = _relay_stages()
    handle = _Handle("a")
    _queue_lookup(stages, handle, [1, 3])
    _relay_round(stages)
    # PP0's lookup becomes the same load on every stage.
    assert [w.started for _, w in stages] == [[(handle, [3])]] * 2
    assert _progress(stages, handle) == [False, False]
    stages[0][1].prefetches[handle].future.set_result(True)
    _relay_round(stages)
    # A stage still reading holds the finish back everywhere.
    assert [w.finished for _, w in stages] == [[], []]
    stages[1][1].prefetches[handle].future.set_result(later_landed)
    _relay_round(stages)
    assert [w.finished for _, w in stages] == [[(handle, later_landed)]] * 2
    assert _progress(stages, handle) == [True, True]
    # The votes of a finished read leave the store.
    assert all(not relay.store.maps[rank] for rank, (relay, _) in enumerate(stages))


def test_pp_prefetch_relay_progress_changes_only_as_actions_apply():
    stages = _relay_stages()
    handle = _Handle("a")
    _queue_lookup(stages, handle, [3])
    _relay_round(stages)
    for _, wrapper in stages:
        wrapper.prefetches[handle].future.set_result(True)
    for relay, _ in reversed(stages):
        relay.poll()
    # PP0 decided, but nothing applies until the relay delivers the finish.
    assert _progress(stages, handle) == [False, False]
    actions = stages[0][0].take_actions()
    stages[0][0].receive(actions.actions)
    assert _progress(stages, handle)[0] is True
    stages[1][0].receive(actions.actions)
    assert _progress(stages, handle) == [True, True]


def test_pp_prefetch_relay_stop_ends_the_wait_on_every_stage_in_order():
    stages = _relay_stages()
    handle = _Handle("a")
    _queue_lookup(stages, handle, [3])
    _relay_round(stages)
    # PP0 alone runs the stop policy; the stop applies everywhere.
    assert _progress(stages, handle, stop_due=lambda h: True) == [False, False]
    _relay_round(stages)
    assert _progress(stages, handle) == [True, True]
    assert [w.cancelled for _, w in stages] == [[handle]] * 2
    # The read still finishes before its slots are freed, unpublished.
    for _, wrapper in stages:
        wrapper.prefetches[handle].future.set_result(True)
    _relay_round(stages)
    _relay_round(stages)
    assert [w.finished for _, w in stages] == [[(handle, False)]] * 2


def _early_stages(size=2):
    stages = _relay_stages(size)
    for _, wrapper in stages:
        wrapper.early = _EarlyStub()
    return stages


def test_pp_early_votes_finish_before_the_request_reaches_the_last_stage():
    stages = _early_stages()
    (pp0, w0), (pp1, w1) = stages
    handle = _Handle("a")
    _queue_lookup(stages[:1], handle, [1, 3])  # stage 1 has not received the request yet
    _relay_round(stages)
    assert w0.early.published == [(0, 1, handle, 3)]
    assert w0.started == [(handle, [3])] and w1.started == [] and pp1.inbox
    # Both stages land their staged reads; stage 1 still has no request.
    w0.early.land(0, 1, 3)
    w1.early.land(0, 1, 3)
    w0.prefetches[handle].future.set_result(True)
    _relay_round(stages)
    assert w0.finished == [(handle, True)] and w1.finished == []
    # The request reaches stage 1: its load (from staging) and the finish apply in order.
    _queue_lookup(stages[1:], handle, None)
    pp1.receive([])
    assert w1.started == [(handle, [3], "staged", "slab-1")]
    assert w1.finished == [(handle, True)] and not pp1.inbox
    assert _progress(stages, handle) == [True, True]
    _relay_round(stages)
    # Every early vote and normal vote of the finished read left the store.
    assert all(not relay.store.maps[rank] for rank, (relay, _) in enumerate(stages))


def test_pp_early_read_failure_falls_back_to_the_stage_read_and_its_vote():
    stages = _early_stages()
    (pp0, w0), (pp1, w1) = stages
    handle = _Handle("a")
    _queue_lookup(stages, handle, [2])
    _relay_round(stages)
    w0.early.land(0, 1, 2)  # stage 1's early read failed: it posts nothing
    _relay_round(stages)
    assert w1.started == [(handle, [2])]  # no staging: today's read
    assert [w.finished for _, w in stages] == [[], []]
    w1.prefetches[handle].future.set_result(True)
    _relay_round(stages)
    assert [w.finished for _, w in stages] == [[(handle, True)]] * 2


def test_pp_early_staging_is_used_only_for_the_issued_pages():
    stages = _early_stages()
    (pp0, w0), (pp1, w1) = stages
    handle = _Handle("a")
    w1.early.staged[(0, 1)] = (2, "slab-stale")  # staged for another page count
    _queue_lookup(stages, handle, [3])
    _relay_round(stages)
    assert w1.started == [(handle, [3])] and (0, 1) in w1.early.staged


def test_pp_early_stop_and_reset_drop_staging_and_votes():
    stages = _early_stages()
    (pp0, w0), (pp1, w1) = stages
    handle = _Handle("a")
    _queue_lookup(stages, handle, [3])
    _relay_round(stages)
    w1.early.land(0, 1, 3)
    pp0.progress(handle, lambda h: True)  # PP0's stop policy fires
    _relay_round(stages)
    assert all((0, 1) in w.early.dropped for _, w in stages)
    assert all(not relay.store.maps[rank] for rank, (relay, _) in enumerate(stages))
    for relay, w in stages:
        relay.reset()
        assert w.early.epoch == 1


def _early_read_stages(size):
    """``size`` relay stages with real EarlyReads; staged and direct reads stay
    pending until the test completes them (``pending_early`` / ``pending_direct``).
    A finish on a staged load still being queued completes it, as the real wait
    does; a finish on any other unfinished read fails."""
    import numpy as np

    from sglang.srt.mem_cache.storage.mooncake_store.linker_staging import EarlyReads, StagingSlab

    stages = _relay_stages(size)
    pending_early, pending_direct = [], []

    class _Checked(_RelayWrapper):
        def _finish_prefetch(self, handle, publish, hold=True):
            state = self.prefetches.pop(handle, None)
            if state is not None and not state.future.done() and getattr(state.future, "staged", False):
                pending_direct.remove(state.future)
                state.future.set_result(True)
            # A direct read is never awaited here: its vote gated the finish.
            assert state is None or state.future.done(), (handle, self.pp_rank)
            self.finished.append((handle, publish))

        def _start_prefetch_load(self, handle, *args):
            super()._start_prefetch_load(handle, *args)
            pending_direct.append(self.prefetches[handle].future)

        def _start_prefetch_from_staging(self, handle, *args, staged):
            # The staging thread queues the scatter later: a pending, staged future.
            super()._start_prefetch_load(handle, *args)
            future = self.prefetches[handle].future
            future.staged = True
            pending_direct.append(future)
            self.started[-1] = (handle, args[-2], "staged", staged)
            self.early.release(staged)

    for rank, (relay, wrapper) in enumerate(stages):
        checked = _Checked(rank, size)
        relay.w = checked
        stages[rank] = (relay, checked)
        buf = np.zeros(StagingSlab.size_of({"kv": (8, 6)}), dtype=np.uint8)

        def read(keys, ptrs):
            future = Future()
            pending_early.append(future)
            return future

        checked.early = EarlyReads(
            StagingSlab({"kv": (8, 6)}, buf, buf.ctypes.data),
            lambda hashes: {"kv": ([f"k{h}" for h in hashes], 8)},
            read,
        )
    others = [w.early for _, w in stages[1:]]
    stages[0][1].early.send = lambda msg: [e.start(msg) for e in others]
    return stages, pending_early, pending_direct


def _complete_some(rng, *pending):
    for futures in pending:
        for f in list(futures):
            if rng.random() < 0.3:
                futures.remove(f)
                f.set_result(rng.random() < 0.9)


def _lookup(rng, rank):
    return SimpleNamespace(
        key=_KEY, hashes=["h1", "h2", "h3"], start=0, anchor_node_id=0,
        anchor_lock="lock", timeout_s=1.0, wait_started=None,
        future=_done([rng.choice([2, 3])]) if rank == 0 else None,
    )


@pytest.mark.parametrize("seed", list(range(1, 21)))
def test_pp_early_reads_late_completions_never_finish_a_pending_read(seed):
    """Real EarlyReads per stage; early reads land, fail or arrive late at random,
    direct reads complete late, requests reach stage k k rounds after PP0. No stage
    may apply a finish to an unfinished read, every stage publishes the same, and
    every staging slot comes back."""
    import random

    rng = random.Random(seed)
    size, n_requests = 4, 12
    stages, pending_early, pending_direct = _early_read_stages(size)

    handles = [_Handle(f"r{i}") for i in range(n_requests)]
    arrive = {h: 2 * i for i, h in enumerate(handles)}
    for rnd in range(400):
        for k, (relay, w) in enumerate(stages):
            for h in handles:
                if arrive[h] + k == rnd:
                    relay.begin(SimpleNamespace(cache_request_handle=h))
                    w.lookups[h] = _lookup(rng, k)
            # As ingest_requests: this round's requests first, then the actions held for them.
            relay.receive([])
        _complete_some(rng, pending_early, pending_direct)
        _relay_round(stages)
        if all(len(w.finished) == n_requests for _, w in stages) and not pending_early:
            break
    finished = [w.finished for _, w in stages]
    assert all(len(f) == n_requests for f in finished), [len(f) for f in finished]
    assert all(f == finished[0] for f in finished)
    for _, w in stages:
        assert w.early.counts["voted_direct"] == 0 and w.early.counts["voted_unused"] == 0
        assert len(w.early.slab.free["kv"]) == 6, (w.pp_rank, w.early.counts)


@pytest.mark.parametrize("seed", list(range(1, 21)))
def test_pp_early_reads_every_stage_admits_pp0s_batches(seed):
    """Liveness under early reads: PP0's requests and actions of pass r reach stage
    k at its pass r + k; with reads landing, failing and arriving late at random,
    every stage's admission pass r + k admits exactly PP0's batch of pass r."""
    import random

    rng = random.Random(seed)
    size, n_requests, lag = 4, 12, 3
    stages, pending_early, pending_direct = _early_read_stages(size)
    handles = [_Handle(f"r{i}") for i in range(n_requests)]
    arrive = {h: lag * i for i, h in enumerate(handles)}
    relayed = []  # PP0's pass r: (requests, actions)
    batches = [[] for _ in range(size)]
    admitted = [set() for _ in range(size)]
    order = [[] for _ in range(size)]
    for rnd in range(600):
        for k, (relay, w) in enumerate(stages):
            if k == 0:
                new = [h for h in handles if arrive[h] == rnd]
            elif rnd - k >= 0:
                new = relayed[rnd - k][0]
            else:
                continue
            for h in new:
                relay.begin(SimpleNamespace(cache_request_handle=h))
                w.lookups[h] = _lookup(rng, k)
                order[k].append(h)
            relay.poll()
            if k == 0:
                actions = relay.take_actions()
                relayed.append((new, actions.actions if actions is not None else []))
            relay.receive(relayed[rnd - k][1])
            batch = [h for h in order[k] if h not in admitted[k] and relay.progress(h, lambda h: False)]
            admitted[k].update(batch)
            batches[k].append(batch)
        _complete_some(rng, pending_early, pending_direct)
        if len(admitted[-1]) == n_requests and not pending_early:
            break
    assert len(admitted[-1]) == n_requests, [len(a) for a in admitted]
    for k in range(1, size):  # batches[k][i] is stage k's pass k + i
        assert batches[k] == batches[0][: len(batches[k])], k


def test_pp_prefetch_relay_settles_a_request_with_nothing_to_look_up():
    stages = _relay_stages()
    handle = _Handle("a")
    for relay, _ in stages:
        relay.begin(SimpleNamespace(cache_request_handle=handle))
        relay.settle(handle)
    assert _progress(stages, handle) == [True, True]
    _relay_round(stages)
    assert stages[0][0].take_actions() is None


def test_pp_prefetch_relay_waits_for_the_request_before_its_action():
    stages = _relay_stages()
    handle = _Handle("a")
    pp0, later = stages
    pp0[0].begin(SimpleNamespace(cache_request_handle=handle))
    pp0[1].lookups[handle] = SimpleNamespace(
        key=_KEY, hashes=["h1"], start=0, anchor_node_id=0, anchor_lock="lock",
        timeout_s=1.0, wait_started=None, future=_done([1]),
    )
    pp0[0].poll()
    actions = pp0[0].take_actions()
    later[0].receive(actions.actions)
    assert later[0].inbox and later[1].started == []
    _queue_lookup([later], handle, [1])
    later[0].receive([])
    assert later[1].started == [(handle, [1])] and not later[0].inbox


def test_mooncake_prefetch_read_ends_its_sessions_and_reports_failure():
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.storage = MagicMock()
    linker.storage._get_hybrid_page_component_keys.side_effect = (
        lambda keys, transfer: (list(keys), 1)
    )
    linker.storage._tag_keys.side_effect = list
    linker.pools = {PoolName.KV: MagicMock(prepare_locations=lambda idx: [0, 1])}
    linker._session_refs = MagicMock()
    linker._session_refs.start.return_value = [0, 0]
    linker.load_per_key = True
    linker.log_load_timing = False
    linker.stats = {}
    transfer = PoolTransfer(
        name=PoolName.KV, keys=["p1", "p2"], host_indices=torch.arange(4)
    )
    linker._load_per_key = MagicMock()
    assert linker._prefetch_read([transfer], MagicMock(), time.monotonic()) is True
    linker._load_per_key = MagicMock(side_effect=RuntimeError("-704"))
    # A failed read is not True; it returns its reason (here the error text).
    assert linker._prefetch_read([transfer], MagicMock(), time.monotonic()) == "read:-704"
    assert linker._session_refs.end.call_args_list == [call(["p1", "p2"])] * 2
    assert linker.stats["prefetch_failed"] == 1


def test_mooncake_prefetch_read_through_the_helper_sends_hashes_and_slots():
    from concurrent.futures import Future

    def answer(value=None, error=None):
        future = Future()
        future.set_exception(error) if error else future.set_result(value)
        return future

    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.helper = MagicMock()
    linker.log_load_timing = False
    linker.stats = {}
    transfer = PoolTransfer(name=PoolName.KV, keys=["p1", "p2"], host_indices=torch.arange(4))
    linker.helper.submit.return_value = answer((True, {"objects": 2}))
    assert linker._prefetch_read_helper([transfer], MagicMock(), time.monotonic()) is True
    op, pools = linker.helper.submit.call_args.args
    assert op == "read" and len(pools) == 1
    name, keys, slots = pools[0]
    assert (name, keys) == (str(PoolName.KV), ["p1", "p2"])
    assert str(slots.dtype) == "int64" and slots.tolist() == [0, 1, 2, 3]
    linker.helper.submit.return_value = answer(("read:failed by code {-704: 1}", {}))
    assert linker._prefetch_read_helper([transfer], MagicMock(), time.monotonic()).startswith("read:")
    linker.helper.submit.return_value = answer(error=RuntimeError("linker helper exited"))
    assert "helper exited" in linker._prefetch_read_helper([transfer], MagicMock(), time.monotonic())
    assert linker.stats["prefetch_failed"] == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_prefetch_timeline_intervals_skips_and_batch_interval():
    from sglang.srt.mem_cache.unified_cache import unified_cache_linker as ucl

    tl = ucl._PrefetchTimeline()
    h = object()
    for stage, t in (
        ("enqueue", 1.000),
        ("lookup_done", 1.010),
        ("load_issued", 1.030),
        ("read_start", 1.035),
        ("landed", 1.130),
        ("voted", 1.140),
        ("finish_issued", 1.330),
        ("ready", 1.530),
    ):
        tl.mark(h, stage, t)
    tl.mark(h, "enqueue", 9.0)  # a later mark of a recorded stage keeps the first
    with patch.object(ucl.time, "perf_counter", return_value=1.600):
        tl.pass_begin()
    tl.candidate(False)
    tl.candidate(False)
    tl.candidate(True)
    with patch.object(ucl.time, "perf_counter", return_value=1.600):
        tl.admitted(h)
    with patch.object(ucl.time, "perf_counter", return_value=1.800):
        tl.pass_begin()
    tl.admitted(object())  # admitted without marks: counts toward the pass only
    with patch.object(ucl.time, "perf_counter", return_value=2.000):
        tl.pass_begin()

    got = {pair: list(v) for pair, v in tl.intervals.items()}
    assert got[("enqueue", "lookup_done")] == [pytest.approx(10.0)]
    assert got[("load_issued", "landed")] == [pytest.approx(100.0)]
    assert got[("enqueue", "ready")] == [pytest.approx(530.0)]
    assert got[("ready", "admitted")] == [pytest.approx(70.0)]
    assert got[("enqueue", "read_start")] == [pytest.approx(35.0)]
    assert got[("landed", "voted")] == [pytest.approx(10.0)]
    assert h not in tl.marks
    assert tl.pass_skips == [2, 0]
    assert tl.skip_positions == [0, 1]
    line = tl.summary()
    assert "batch interval n=1 mean 200 ms" in line
    assert "not-ready skips per pass p50 2" in line
    assert "skipped queue position p50 1" in line
    assert all(not v for v in tl.intervals.values()) and not tl.pass_skips


def test_prefetch_timeline_wall_line_for_sampled_rid():
    from sglang.srt.mem_cache.unified_cache import unified_cache_linker as ucl

    tl = ucl._PrefetchTimeline()
    tl.wall_offset = 1000.0
    class _Handle:
        def __init__(self, rid):
            self.rid = rid

    sampled, other = _Handle("rid-9"), _Handle("rid-0")
    for h in (sampled, other):
        tl.mark(h, "enqueue", 1.0)
        tl.mark(h, "read_start", 1.5)
    with patch.object(ucl.logger, "info") as info:
        tl.admitted(sampled)
        tl.admitted(other)
    assert info.call_count == 1
    fmt, rid, stamps = info.call_args.args
    assert rid == "rid-9" and stamps.startswith("enqueue=1001.000 read_start=1001.500")
