import logging
import random
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolTransfer,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.storage import lease_refresh
from sglang.srt.mem_cache.storage.lease_refresh import (
    StorageLeaseRefresher,
    create_storage_lease_refresher,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    TreePathNodeInfo,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
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
        self.waited = []

    def wait(self, timeout):
        self.waited.append(timeout)
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


def _mooncake_store(*, local_rank=0, kv_buffer=None):
    store = MooncakeStore.__new__(MooncakeStore)
    store.registered_pools = {
        name: SimpleNamespace()
        for name in (
            "deepseek_v4_c1",
            "deepseek_v4_c1_indexer",
            "deepseek_v4_c2",
            "deepseek_v4_c2_indexer",
        )
    }
    store.mem_pool_host = SimpleNamespace(kv_buffer=kv_buffer)
    store.is_mla_backend = True
    store.mla_suffix = ""
    store.dcp_size = 1
    store.local_rank = local_rank
    store.config_prefix = "deepseek-ai-DeepSeek-V4.1-Flash"
    store.store = MagicMock()
    return store


_DSV4_SIDECARS = [
    SidecarPoolSpec(name, "kv")
    for name in (
        "deepseek_v4_c1",
        "deepseek_v4_c1_indexer",
        "deepseek_v4_c2",
        "deepseek_v4_c2_indexer",
    )
] + [SidecarPoolSpec("deepseek_v4_c4_state", "swa", PoolHitPolicy.TRAILING_PAGES)]


class TestFactory(CustomTestCase):
    def test_non_mooncake_backend_is_off(self):
        with self.assertLogs(lease_refresh.logger, level=logging.WARNING):
            refresher = create_storage_lease_refresher(
                backend=object(),
                sidecar_pool_specs=_DSV4_SIDECARS,
                interval_s=60.0,
                max_keys_per_s=1000,
            )
        self.assertIsNone(refresher)

    def test_replicated_mla_objects_are_renewed_by_rank_0_only(self):
        refresher = create_storage_lease_refresher(
            backend=_mooncake_store(local_rank=1),
            sidecar_pool_specs=_DSV4_SIDECARS,
            interval_s=60.0,
            max_keys_per_s=1000,
        )
        self.assertIsNone(refresher)

    def test_renews_the_keys_a_prefetch_existence_query_reads(self):
        store = _mooncake_store()
        refresher = create_storage_lease_refresher(
            backend=store,
            sidecar_pool_specs=_DSV4_SIDECARS,
            interval_s=60.0,
            max_keys_per_s=1000,
        )
        self.addCleanup(refresher.stop)
        pages = ["h0", "h1"]
        refresher._renew_batch([pages])
        renewed = store.store.batch_is_exist.call_args.args[0]

        store.store.batch_is_exist.reset_mock()
        store.store.batch_is_exist.return_value = [1] * len(renewed)
        store.batch_exists_v2(
            pages,
            [
                PoolTransfer(name=spec.pool_name, hit_policy=spec.hit_policy)
                for spec in _DSV4_SIDECARS[:4]
            ],
        )
        queried = store.store.batch_is_exist.call_args.args[0]
        self.assertEqual(renewed, queried)
        self.assertEqual(len(renewed), 8)
        self.assertEqual(
            renewed[0], "deepseek-ai-DeepSeek-V4.1-Flash_h0__deepseek_v4_c1"
        )

    def test_physical_kv_objects_are_renewed_with_their_batch_exists_keys(self):
        store = _mooncake_store(kv_buffer=object())
        keys = store.lease_refresh_keys(["h0"], pool_names=())
        self.assertEqual(keys, store._kv_exist_keys(["h0"])[0])
        self.assertEqual(keys, ["deepseek-ai-DeepSeek-V4.1-Flash_h0__k"])


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


class TestUnifiedRadixCacheHook(CustomTestCase):
    def _cache(self, refresher, tree_core, interval=60.0):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_core = tree_core
        cache.storage_lease_refresh_interval = interval
        cache._storage_lease_refresher = lambda: refresher
        return cache

    def test_renews_the_stale_suffix_and_stops_at_unhashed_nodes(self):
        clock = FakeClock()
        refresher = _refresher(clock)
        refresher.submit([1], 990.0, ["a0"])
        refresher._take_pending(timeout=0)
        tree = FakeTreeCore(
            parents={1: 0, 2: 1, 3: 2, 4: 3},
            hashes={1: ["a0"], 2: ["b0", "b1"], 4: ["d0"]},
        )
        self._cache(refresher, tree).refresh_storage_leases(4)
        self.assertEqual(refresher._take_pending(timeout=0), [["b0", "b1"]])
        self.assertEqual(refresher._chain_stamps, {1: 990.0, 2: 990.0})

    def test_off_and_root_never_touch_the_tree(self):
        tree = MagicMock()
        tree.is_root.return_value = False
        self._cache(None, tree, interval=0.0).refresh_storage_leases(4)
        tree.collect_path_info.assert_not_called()
        tree.is_root.return_value = True
        self._cache(None, tree).refresh_storage_leases(0)
        tree.collect_path_info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
