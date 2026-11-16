"""The Mooncake direct linker shares one object per pool along the axes whose
pages are byte-identical: attention CP (interleave PCP) and attention TP
(MLA-family pools, which have no head dimension to shard). A hybrid K3.1 group
holds both kinds at once, so the choice is per pool: KV, indexer and an MLA
draft are shared, while TP-sharded KDA state keeps its own rank namespace."""

import queue
import unittest
from collections import Counter
from types import SimpleNamespace

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.mooncake_store.linker_lease_refresh import (
    LinkerLeaseRefresher,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
    _shared_pool_suffix,
    _storage_sharing,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _k31_pools(dcp_sharded=False):
    """A K3.1 hybrid group: replicated KV / indexer / MLA draft, sharded KDA."""
    return {
        name: SimpleNamespace(
            pcp_replicated=pcp, tp_replicated=tp, dcp_sharded=dcp_sharded and tp
        )
        for name, pcp, tp in (
            (PoolName.KV, True, True),
            (PoolName.INDEXER, True, True),
            (PoolName.DRAFT, False, True),
            (PoolName.MAMBA, False, False),
        )
    }


def _sharing(dcp_size=1, **kwargs):
    args = dict(
        rank_replicated=False,
        tp_rank=0,
        tp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        pp_rank=0,
        offload_owner=True,
    )
    args.update(kwargs)
    return _storage_sharing(
        _k31_pools(dcp_size > 1),
        dcp_size=dcp_size,
        dcp_rank=args["tp_rank"] % dcp_size,
        **args,
    )


class TestSharedPoolSuffix(CustomTestCase):
    def test_each_dropped_axis_collapses_its_ranks(self):
        for axes, ranks in (
            (dict(tp_shared=True, cp_shared=False), {"tpshared_v1_cp0_pp0"}),
            (dict(tp_shared=True, cp_shared=True), {"pcp_shared_v1_tpshared_v1_pp0"}),
        ):
            with self.subTest(**axes):
                self.assertEqual(
                    {
                        _shared_pool_suffix(
                            tp_rank=rank,
                            attn_cp_rank=0,
                            pp_rank=0,
                            dcp_sharded=False,
                            dcp_size=1,
                            dcp_rank=0,
                            **axes,
                        )
                        for rank in range(8)
                    },
                    ranks,
                )

    def test_the_cp_only_term_keeps_its_original_spelling(self):
        # A store written before the TP axis existed must stay readable.
        self.assertEqual(
            _shared_pool_suffix(
                tp_shared=False,
                cp_shared=True,
                tp_rank=3,
                attn_cp_rank=1,
                pp_rank=2,
                dcp_sharded=False,
                dcp_size=1,
                dcp_rank=0,
            ),
            "pcp_shared_v1_tp3_pp2",
        )

    def test_a_pool_shared_on_no_axis_keeps_every_rank_term(self):
        suffixes = {
            _shared_pool_suffix(
                tp_shared=False,
                cp_shared=False,
                tp_rank=rank,
                attn_cp_rank=1,
                pp_rank=2,
                dcp_sharded=False,
                dcp_size=1,
                dcp_rank=0,
            )
            for rank in range(4)
        }
        self.assertEqual(len(suffixes), 4)
        self.assertIn("tp3_cp1_pp2", suffixes)


class TestStorageSharing(CustomTestCase):
    def test_pure_tp_shares_the_replicated_pools_and_writes_them_once(self):
        # The case this fixes: 8 attention-TP ranks each wrote a full copy of
        # the same MLA KV, indexer and draft pages.
        keys, owners = {}, {}
        for rank in range(8):
            suffixes, offload, cp_shared, tp_shared = _sharing(tp_rank=rank, tp_size=8)
            self.assertEqual(cp_shared, set())
            self.assertEqual(tp_shared, {PoolName.KV, PoolName.INDEXER, PoolName.DRAFT})
            for name, suffix in suffixes.items():
                keys.setdefault(name, set()).add(suffix)
            owners[rank] = offload
        for name in (PoolName.KV, PoolName.INDEXER, PoolName.DRAFT):
            self.assertEqual(keys[name], {"tpshared_v1_cp0_pp0"}, name)
        self.assertNotIn(PoolName.MAMBA, keys)  # keeps the group's tp{rank} suffix
        self.assertEqual(
            owners[0], {PoolName.KV, PoolName.INDEXER, PoolName.DRAFT, PoolName.MAMBA}
        )
        for rank in range(1, 8):
            self.assertEqual(owners[rank], {PoolName.MAMBA}, rank)

    def test_sharded_state_stays_one_object_per_rank(self):
        # Every rank must still write its own KDA shard, or ranks 1-7 would
        # read rank 0's state.
        total = sum(
            PoolName.MAMBA in _sharing(tp_rank=rank, tp_size=8)[1] for rank in range(8)
        )
        self.assertEqual(total, 8)

    def test_both_axes_shared_is_written_by_rank_zero_of_each(self):
        writers = {
            (tp, cp)
            for tp in range(2)
            for cp in range(2)
            if PoolName.KV
            in _sharing(tp_rank=tp, tp_size=2, attn_cp_rank=cp, attn_cp_size=2)[1]
        }
        self.assertEqual(writers, {(0, 0)})

    def test_tp_size_one_changes_nothing(self):
        # v9's shape: attention TP is 1, so there is nothing to collapse and
        # only the CP axis is shared.
        suffixes, offload, cp_shared, tp_shared = _sharing(
            tp_size=1, attn_cp_rank=0, attn_cp_size=2
        )
        self.assertEqual(tp_shared, set())
        self.assertEqual(cp_shared, {PoolName.KV, PoolName.INDEXER})
        self.assertEqual(suffixes[PoolName.KV], "pcp_shared_v1_tp0_pp0")
        self.assertEqual(
            offload, {PoolName.KV, PoolName.INDEXER, PoolName.DRAFT, PoolName.MAMBA}
        )

    def test_a_rank_replicated_group_is_left_to_its_group_suffix(self):
        # No sharded pool, so the group's own suffix already drops the tp term.
        suffixes, _, _, tp_shared = _sharing(rank_replicated=True, tp_rank=3, tp_size=8)
        self.assertEqual(tp_shared, set())
        self.assertEqual(suffixes, {})

    def test_a_non_owner_rank_offloads_nothing(self):
        _, offload, _, _ = _sharing(tp_rank=3, tp_size=8, offload_owner=False)
        self.assertEqual(offload, set())

    def test_dcp_keys_each_rank_and_the_first_block_writes(self):
        sharded = {PoolName.KV, PoolName.INDEXER, PoolName.DRAFT}
        for rank in range(16):
            suffixes, offload, _, _ = _sharing(tp_rank=rank, tp_size=16, dcp_size=8)
            for name in sharded:
                self.assertEqual(suffixes[name], f"tpshared_v1_dcp8r{rank % 8}_cp0_pp0")
            self.assertNotIn(PoolName.MAMBA, suffixes)
            self.assertEqual(
                offload, sharded | {PoolName.MAMBA} if rank < 8 else {PoolName.MAMBA}
            )


def _stage_schema(pp_rank):
    """A PP8 TP1 K3.1 stage: KV, indexer and KDA state under its own pp suffix."""
    suffix = f"tp0_cp0_pp{pp_rank}"
    return [
        ("kv", "k", suffix, False),
        ("indexer", "indexer", suffix, False),
        ("mamba", "mamba", suffix, True),
    ]


class _PresentKeys:
    """A store holding a fixed key set; records every existence query."""

    def __init__(self, present):
        self.present = set(present)
        self.queries = []

    def _tag_keys(self, keys):
        return [f"model_{key}" for key in keys]

    def _batch_exist(self, keys):
        self.queries.append(list(keys))
        return [1 if key in self.present else 0 for key in keys]


def _linker_over_stages(present, num_stages=3):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.storage = _PresentKeys(present)
    linker.stage_schemas = [_stage_schema(stage) for stage in range(num_stages)]
    linker.stats = {"lookup": 0, "load": 0, "offload": 0}
    linker.lookup_counts = Counter()
    return linker


def _stored(stages, pages, states):
    """Every stage's KV and indexer of ``pages`` and its state at ``states``."""
    keys = set()
    for stage in stages:
        for page in pages:
            keys |= {f"model_p{page}_tp0_cp0_pp{stage}_{obj}" for obj in ("k", "indexer")}
        keys |= {f"model_p{page}_tp0_cp0_pp{stage}_mamba" for page in states}
    return keys


class TestLookupStages(CustomTestCase):
    """Under PP, PP0 checks every stage's objects so the hit it relays is one
    every stage can load: a stop point needs every stage's state there and
    every stage's KV and indexer pages up to it."""

    def _lookup(self, linker):
        page_keys = [f"p{page}" for page in range(1, 5)]
        transfers = [PoolTransfer(name=PoolName.KV, keys=page_keys)]
        return linker.lookup_stages("rid", transfers)

    def test_a_stop_point_needs_every_stage_state(self):
        present = _stored(range(3), range(1, 5), states={2, 4})
        present.discard("model_p4_tp0_cp0_pp2_mamba")  # stage 2 lost its state at 4
        self.assertEqual(self._lookup(_linker_over_stages(present)), [2])

    def test_a_missing_page_on_one_stage_caps_the_hit(self):
        present = _stored(range(3), range(1, 5), states={2, 4})
        present.discard("model_p2_tp0_cp0_pp1_indexer")
        self.assertEqual(self._lookup(_linker_over_stages(present)), [])
        present = _stored(range(3), range(1, 5), states={2, 4})
        present.discard("model_p3_tp0_cp0_pp1_k")
        self.assertEqual(self._lookup(_linker_over_stages(present)), [2])

    def test_all_page_objects_are_queried_only_up_to_the_deepest_stop(self):
        linker = _linker_over_stages(_stored(range(3), range(1, 5), states={2}))
        self.assertEqual(self._lookup(linker), [2])
        trailing, all_pages, past_hit = linker.storage.queries
        self.assertEqual(len(trailing), 3 * 4)  # every stage's state, every page
        self.assertEqual(len(all_pages), 3 * 2 * 2)  # stages x pools x pages 1-2
        # Truncation count: stage 0's KV past the hit, pages 3-4.
        self.assertEqual(past_hit, ["model_p3_tp0_cp0_pp0_k", "model_p4_tp0_cp0_pp0_k"])

    def test_no_common_state_skips_the_page_query(self):
        linker = _linker_over_stages(_stored(range(2), range(1, 5), states={2}))
        self.assertEqual(self._lookup(linker), [])
        # The state query, then only the truncation count's KV query.
        self.assertEqual(len(linker.storage.queries), 2)
        self.assertEqual(linker.lookup_counts["truncated_pages"], 4)

    def test_lease_keys_renew_every_stage_pages_and_the_last_state(self):
        linker = _linker_over_stages(set(), num_stages=2)
        keys = linker._lease_keys(["p1", "p2"])
        self.assertEqual(
            sorted(keys),
            sorted(
                f"model_p{page}_tp0_cp0_pp{stage}_{obj}"
                for stage in range(2)
                for page, obj in (
                    (1, "k"), (2, "k"), (1, "indexer"), (2, "indexer"), (2, "mamba")
                )
            ),
        )


class TestLookupTruncation(CustomTestCase):
    def test_kv_past_the_last_common_state_counts_as_truncated(self):
        # KV and indexer on every stage up to page 4, states only at page 2.
        linker = _linker_over_stages(_stored(range(3), range(1, 5), states={2}))
        transfers = [PoolTransfer(name=PoolName.KV, keys=[f"p{p}" for p in range(1, 5)])]
        self.assertEqual(linker.lookup_stages("rid", transfers), [2])
        self.assertEqual(
            dict(linker.lookup_counts),
            {"lookups": 1, "hit_pages": 2, "truncated_pages": 2, "truncated_lookups": 1},
        )


class _RemovingStore:
    def __init__(self, refuse=()):
        self.removed, self.refuse = [], set(refuse)

    def remove(self, key):
        self.removed.append(key)
        return -1 if key in self.refuse else 0


def _capped_linker(cap):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.state_path_cap = cap
    linker.state_parent, linker.state_children = {}, Counter()
    linker.state_removed = linker.state_remove_failed = 0
    return linker


class TestStatePathCap(CustomTestCase):
    """A conversation resumes from its deepest KDA states; each older linear
    state past SGLANG_LINKER_STATE_PATH_CAP is dead weight in the store."""

    def test_cap_one_drops_the_previous_turn_state_but_not_the_first(self):
        linker = _capped_linker(1)
        self.assertEqual(linker._cap_state_path(["s1"], ["a", "s1"]), [])
        # s1 is the path's first state: another session may branch there.
        self.assertEqual(linker._cap_state_path(["s2"], ["a", "s1", "b", "s2"]), [])
        self.assertEqual(
            linker._cap_state_path(["s3"], ["a", "s1", "b", "s2", "c", "s3"]), ["s2"]
        )
        self.assertEqual(linker.state_parent, {"s1": None, "s3": "s1"})

    def test_cap_three_keeps_the_three_deepest(self):
        linker = _capped_linker(3)
        path = []
        dropped = []
        for turn in range(1, 6):
            path += [f"x{turn}", f"s{turn}"]
            dropped += linker._cap_state_path([f"s{turn}"], list(path))
        self.assertEqual(dropped, ["s2"])
        self.assertEqual(linker.state_parent, {"s1": None, "s3": "s1", "s4": "s3", "s5": "s4"})

    def test_a_shared_prompt_state_stays_and_its_branches_relink(self):
        linker = _capped_linker(1)
        linker._cap_state_path(["sys"], ["p", "sys"])
        linker._cap_state_path(["a1"], ["p", "sys", "a", "a1"])  # 'sys' has one child
        linker._cap_state_path(["b1"], ["p", "sys", "b", "b1"])  # now two
        self.assertEqual(linker.state_children["sys"], 2)
        # a2 drops a1 but stops at the branch point, and hangs from it.
        self.assertEqual(
            linker._cap_state_path(["a2"], ["p", "sys", "a", "a1", "a", "a2"]), ["a1"]
        )
        self.assertEqual(linker.state_parent["a2"], "sys")
        self.assertEqual(linker.state_children["sys"], 2)

    def test_removal_uses_the_lookup_key_schema_and_counts_refusals(self):
        linker = _linker_over_stages(set(), num_stages=1)
        linker.storage.store = _RemovingStore(refuse={"model_s1_tp0_cp0_pp0_mamba"})
        linker.storage.pool_suffixes = {
            name: "tp0_cp0_pp0" for name in (PoolName.KV, PoolName.MAMBA, PoolName.INDEXER)
        }
        linker.storage.mla_suffix = "tp0_cp0_pp0"
        linker.pools = {PoolName.KV: None, PoolName.MAMBA: None, PoolName.INDEXER: None}
        linker.state_removed = linker.state_remove_failed = 0
        linker._remove_states(["s1", "s2"])
        self.assertEqual(
            linker.storage.store.removed,
            ["model_s1_tp0_cp0_pp0_mamba", "model_s2_tp0_cp0_pp0_mamba"],
        )
        self.assertEqual((linker.state_removed, linker.state_remove_failed), (1, 1))


class _SettingStorage:
    def __init__(self, ok=True):
        self.ok, self.store = ok, _RemovingStore()
        self.pool_suffixes = {name: "tp0_cp0_pp0" for name in (PoolName.KV, PoolName.MAMBA)}
        self.mla_suffix = "tp0_cp0_pp0"

    def _tag_keys(self, keys):
        return [f"model_{key}" for key in keys]

    def batch_set_v2(self, transfers, extra_info):
        return {t.name: [self.ok] * len(t.keys) for t in transfers}


class TestOffloadAccounting(CustomTestCase):
    def _run(self, ok):
        linker = _capped_linker(1)
        linker.storage = _SettingStorage(ok)
        linker.pools = {PoolName.KV: None, PoolName.MAMBA: None}
        linker.object_bytes = {PoolName.KV: 100, PoolName.MAMBA: 5000}
        linker.write_bytes, linker.lookup_counts = Counter(), Counter()
        linker.stats, linker.stats_logged_at = {"offload": 0}, 0.0
        linker.offload_queue, linker.offload_results = queue.Queue(), queue.Queue()
        expanded = [
            PoolTransfer(name=PoolName.KV, keys=["p1", "p2", "p3"]),
            PoolTransfer(name=PoolName.MAMBA, keys=["p3"]),
        ]
        linker.offload_queue.put((expanded, 192, [], None, ["p1"]))
        linker.offload_queue.put(None)
        linker.offload_thread_func()
        return linker

    def test_writes_split_by_pool_and_old_states_go_after_the_new_lands(self):
        linker = self._run(ok=True)
        self.assertEqual(linker.offload_results.get_nowait(), True)
        # Logged once and reset: the window started at 0.
        self.assertEqual(linker.write_bytes, Counter())
        self.assertEqual(linker.storage.store.removed, ["model_p1_tp0_cp0_pp0_mamba"])

    def test_a_failed_write_keeps_the_older_state(self):
        linker = self._run(ok=False)
        self.assertEqual(linker.offload_results.get_nowait(), False)
        self.assertEqual(linker.storage.store.removed, [])


class TestLinkerLeaseRefresher(CustomTestCase):
    def _refresher(self, clock, renewed, max_keys_per_s=0):
        return LinkerLeaseRefresher(
            build_keys=lambda pages: [f"{page}_k" for page in pages],
            renew=lambda keys: renewed.append(list(keys)) or [1] * len(keys),
            interval_s=10.0,
            max_keys_per_s=max_keys_per_s,
            clock=clock,
        )

    def test_an_anchor_is_due_once_per_interval(self):
        now = [100.0]
        refresher = self._refresher(lambda: now[0], [])
        self.assertTrue(refresher.is_due("node"))
        refresher.submit("node", ["p1"])
        self.assertFalse(refresher.is_due("node"))
        now[0] += 10.5
        self.assertTrue(refresher.is_due("node"))

    def test_renewal_queries_in_bounded_chunks_and_counts_present(self):
        renewed = []
        refresher = self._refresher(lambda: 0.0, renewed)
        refresher.renew_batch([[f"p{i}" for i in range(5000)], ["q"]])
        self.assertEqual([len(chunk) for chunk in renewed], [4096, 905])
        self.assertEqual(refresher.stats["keys"], 5001)
        self.assertEqual(refresher.stats["present"], 5001)
        self.assertEqual(refresher.stats["prefixes"], 2)


if __name__ == "__main__":
    unittest.main()
