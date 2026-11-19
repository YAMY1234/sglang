"""Unit tests for the external linker's L3 lease refresh of device-served prefixes."""

import logging
import random
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.mem_cache.storage import lease_refresh
from sglang.srt.mem_cache.storage.lease_refresh import StorageLeaseRefresher
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    TreePathNodeInfo,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeStopEvent:
    """Advances the fake clock instead of sleeping."""

    def __init__(self, clock):
        self.clock = clock

    def wait(self, timeout):
        self.clock.now += timeout
        return False

    def is_set(self):
        return False


def _refresher(clock, *, interval_s=60.0, max_keys_per_s=0, renew=None, **kwargs):
    return StorageLeaseRefresher(
        build_keys=lambda pages: [f"{p}_c1" for p in pages]
        + [f"{p}_c2" for p in pages],
        renew=renew or (lambda keys: [1] * len(keys)),
        interval_s=interval_s,
        max_keys_per_s=max_keys_per_s,
        clock=clock,
        **kwargs,
    )


class TestChainStamps(CustomTestCase):
    def test_a_turn_renews_only_the_pages_it_appended(self):
        clock = FakeClock()
        r = _refresher(clock)
        self.assertEqual(r.stale_start([1, 2, 3]), (0, 1000.0))
        r.submit([1, 2, 3], 1000.0, ["a", "b", "c"])

        clock.now += 10
        self.assertEqual(r.stale_start([1, 2, 3])[0], 3)
        # The appended node takes the chain's stamp, not now.
        self.assertEqual(r.stale_start([1, 2, 3, 4]), (3, 1000.0))

    def test_every_page_on_a_hit_path_is_renewed_within_the_interval(self):
        rng = random.Random(0)
        clock = FakeClock()
        interval = 60.0
        r = _refresher(clock, interval_s=interval)
        parent = {1: None}
        last_renewed: dict[int, float] = {}
        for _ in range(5000):
            clock.now += rng.expovariate(1 / 7.0)
            node = rng.choice(list(parent))
            if rng.random() < 0.3:
                child = len(parent) + 1
                parent[child] = node
                node = child
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            start, stamp = r.stale_start(path)
            r.submit(path[start:], stamp, [f"p{n}" for n in path[start:]])
            for n in path[start:]:
                last_renewed[n] = clock.now
            for n in path:
                self.assertGreaterEqual(last_renewed[n], clock.now - interval)

    def test_prune_drops_only_expired_stamps(self):
        clock = FakeClock()
        r = _refresher(clock)
        r.submit([1], 1000.0, ["a"])
        clock.now += 50
        r.submit([2], clock.now, ["b"])
        clock.now += 20
        r.stale_start([])
        self.assertEqual(set(r._chain_stamps), {2})


class TestRenewal(CustomTestCase):
    def test_batch_builds_every_pool_key_and_counts_present(self):
        clock = FakeClock()
        seen = []

        def renew(keys):
            seen.append(list(keys))
            return [1 if k.endswith("_c1") else 0 for k in keys]

        r = _refresher(clock, renew=renew)
        r._renew_batch([["a", "b"], ["c"]])
        self.assertEqual(seen, [["a_c1", "b_c1", "a_c2", "b_c2", "c_c1", "c_c2"]])
        self.assertEqual(r.stats["prefixes"], 2)
        self.assertEqual(r.stats["pages"], 3)
        self.assertEqual(r.stats["keys"], 6)
        self.assertEqual(r.stats["present"], 3)

    def test_large_batches_split_into_bounded_queries(self):
        clock = FakeClock()
        sizes = []
        r = _refresher(clock, renew=lambda keys: sizes.append(len(keys)) or [])
        with patch.object(lease_refresh, "_KEYS_PER_QUERY", 4):
            r._renew_batch([["a", "b", "c", "d", "e"]])
        self.assertEqual(sizes, [4, 4, 2])

    def test_query_failure_is_counted_not_raised(self):
        clock = FakeClock()

        def renew(keys):
            raise RuntimeError("master down")

        r = _refresher(clock, renew=renew)
        with self.assertLogs(lease_refresh.logger, level=logging.ERROR):
            r._renew_batch([["a"]])
        r._renew_batch([["b"]])
        self.assertEqual(r.stats["errors"], 2)
        self.assertEqual(r.stats["keys"], 0)

    def test_key_rate_is_capped(self):
        clock = FakeClock()
        r = _refresher(clock, max_keys_per_s=100)
        r._stopping = FakeStopEvent(clock)
        start = clock.now
        with patch.object(lease_refresh, "_KEYS_PER_QUERY", 50):
            r._renew_batch([[f"p{i}" for i in range(250)]])
        # 500 keys at 100/s with a 100-key burst: 4 s of waiting.
        self.assertEqual(r.stats["keys"], 500)
        self.assertAlmostEqual(clock.now - start, 4.0)

    def test_backlog_over_cap_drops_the_oldest_requests(self):
        clock = FakeClock()
        r = _refresher(clock, max_pending_pages=4)
        r.submit([1], clock.now, ["a", "b"])
        r.submit([2], clock.now, ["c", "d"])
        r.submit([3], clock.now, ["e"])
        self.assertEqual(r.stats["dropped_pages"], 2)
        self.assertEqual(r._take_pending(timeout=0), [["c", "d"], ["e"]])

    def test_worker_thread_drains_and_stops(self):
        renewed = []
        r = StorageLeaseRefresher(
            build_keys=lambda pages: pages,
            renew=lambda keys: renewed.extend(keys) or [1] * len(keys),
            interval_s=60.0,
            max_keys_per_s=0,
        )
        r.start()
        r.submit([1], 0.0, ["a", "b"])
        for _ in range(200):
            if len(renewed) == 2:
                break
            r._stopping.wait(0.01)
        r.stop()
        self.assertEqual(renewed, ["a", "b"])
        self.assertIsNone(r._thread)


def _entry(name, source):
    return DevicePoolEntry(
        name=name,
        indices_from_pool=source,
        device_pool=None,
        components=[[torch.zeros((8, 4), dtype=torch.uint8)]],
        layer_mapping={0: 0},
        page_size=2,
        rows_are_pages=False,
        packed=False,
    )


def _linker(*, owner=True):
    """A direct linker over DSv4-like pools: two KV-sourced pools plus a trailing SWA pool."""
    group = DevicePoolGroup(
        [
            _entry(PoolName.KV, PoolName.KV),
            _entry(PoolName.DEEPSEEK_V4_C4, PoolName.KV),
            _entry(PoolName.SWA, PoolName.SWA),
        ],
        num_layers=1,
        page_size=2,
        rank_replicated=True,
    )
    store = MooncakeStore.__new__(MooncakeStore)
    store.mem_pool_host = group
    store.registered_pools = group.entry_map
    store.pool_suffixes = {}
    store.mla_suffix = ""
    store.config_prefix = "dsv41"
    store.store = MagicMock()
    store.store.batch_is_exist.side_effect = lambda keys: [1] * len(keys)
    linker = object.__new__(MooncakeDirectLinker)
    linker.pool_group = group
    linker.storage = store
    linker.offload_owner = owner
    linker.stats = {"lookup": 0, "load": 0, "offload": 0}
    return linker, store


class TestDirectLinkerKeys(CustomTestCase):
    def test_renews_exactly_the_every_page_keys_a_lookup_reads(self):
        linker, store = _linker()
        pages = ["h0", "h1", "h2"]
        build_keys, renew = linker.lease_refresh_fns()
        renewed = build_keys(pages)
        renew(renewed)
        self.assertEqual(store.store.batch_is_exist.call_args.args[0], renewed)

        store.store.batch_is_exist.reset_mock()
        linker.lookup(
            "rid",
            [
                PoolTransfer(name=PoolName.KV, keys=pages),
                PoolTransfer(
                    name=PoolName.SWA,
                    keys=pages[-1:],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                ),
            ],
        )
        queried = store.store.batch_is_exist.call_args.args[0]
        # The lookup also reads the trailing SWA window, which the refresh leaves out.
        self.assertEqual([k for k in queried if not k.endswith("_swa")], renewed)
        self.assertEqual(len(renewed), 6)
        self.assertEqual(renewed[0], "dsv41_h0__k")

    def test_a_replica_rank_renews_nothing(self):
        linker, _ = _linker(owner=False)
        self.assertIsNone(linker.lease_refresh_fns())


class FakeTreeCore:
    def __init__(self, parents, hashes):
        self.parents = parents
        self.hashes = hashes

    def is_root(self, node_id):
        return node_id == 0

    def collect_path_info(self, node_ids):
        (node,) = node_ids
        path = []
        while node != 0:
            path.append(TreePathNodeInfo(node, self.parents[node], 0, 0, None, 0))
            node = self.parents[node]
        return (tuple(reversed(path)),)

    def get_hash_values(self, node_id):
        return self.hashes.get(node_id, [])


class TestWrapperHook(CustomTestCase):
    def _wrapper(self, refresher, tree_core):
        wrapper = object.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache = SimpleNamespace(tree_core=tree_core)
        wrapper._lease_refresher = refresher
        return wrapper

    def test_renews_the_stale_suffix_and_stops_at_unhashed_nodes(self):
        clock = FakeClock()
        refresher = _refresher(clock)
        refresher.submit([1], 990.0, ["a0"])
        refresher._take_pending(timeout=0)
        tree = FakeTreeCore(
            parents={1: 0, 2: 1, 3: 2, 4: 3},
            hashes={1: ["a0"], 2: ["b0", "b1"], 4: ["d0"]},
        )
        self._wrapper(refresher, tree)._refresh_leases(4)
        self.assertEqual(refresher._take_pending(timeout=0), [["b0", "b1"]])
        self.assertEqual(refresher._chain_stamps, {1: 990.0, 2: 990.0})
        self.assertEqual(refresher.stats["host_calls"], 1)

    def test_a_fresh_chain_submits_nothing(self):
        clock = FakeClock()
        refresher = _refresher(clock)
        tree = FakeTreeCore(parents={1: 0, 2: 1}, hashes={1: ["a0"], 2: ["b0"]})
        wrapper = self._wrapper(refresher, tree)
        wrapper._refresh_leases(2)
        self.assertEqual(refresher._take_pending(timeout=0), [["a0", "b0"]])
        clock.now += 30
        wrapper._refresh_leases(2)
        self.assertEqual(refresher._take_pending(timeout=0), [])
        clock.now += 31
        wrapper._refresh_leases(2)
        self.assertEqual(refresher._take_pending(timeout=0), [["a0", "b0"]])

    def test_root_never_touches_the_path(self):
        tree = MagicMock()
        tree.is_root.return_value = True
        self._wrapper(_refresher(FakeClock()), tree)._refresh_leases(0)
        tree.collect_path_info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
