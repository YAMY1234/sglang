"""The Mooncake direct linker shares one object per pool along the axes whose
pages are byte-identical: attention CP (interleave PCP) and attention TP
(MLA-family pools, which have no head dimension to shard). A hybrid K3.1 group
holds both kinds at once, so the choice is per pool: KV, indexer and an MLA
draft are shared, while TP-sharded KDA state keeps its own rank namespace."""

import queue
import random
import threading
import unittest
from collections import Counter, OrderedDict, deque
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.mooncake_store.linker_lease_refresh import (
    LinkerLeaseRefresher,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
    _PutTrace,
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
    linker.lookup_samples = []
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
    every stage's KV and indexer pages up to it, probed at the stop."""

    def _lookup(self, linker):
        page_keys = [f"p{page}" for page in range(1, 5)]
        transfers = [PoolTransfer(name=PoolName.KV, keys=page_keys)]
        return linker.lookup_stages("rid", transfers)

    def test_a_stop_point_needs_every_stage_state(self):
        present = _stored(range(3), range(1, 5), states={2, 4})
        present.discard("model_p4_tp0_cp0_pp2_mamba")  # stage 2 lost its state at 4
        self.assertEqual(self._lookup(_linker_over_stages(present)), [2])

    def test_a_missing_page_at_a_stop_caps_the_hit(self):
        present = _stored(range(3), range(1, 5), states={2, 4})
        present.discard("model_p4_tp0_cp0_pp1_indexer")
        self.assertEqual(self._lookup(_linker_over_stages(present)), [2])
        present.discard("model_p2_tp0_cp0_pp2_k")
        self.assertEqual(self._lookup(_linker_over_stages(present)), [])

    def test_a_hole_below_a_probed_stop_is_left_to_the_admission_pin(self):
        # Only stops are probed: page 3 is gone, page 4 is there, so the hit
        # reaches 4 and pin_hit's session start finds the hole at admission.
        present = _stored(range(3), range(1, 5), states={2, 4})
        present.discard("model_p3_tp0_cp0_pp1_k")
        self.assertEqual(self._lookup(_linker_over_stages(present)), [2, 4])

    def test_one_stage_states_every_page_the_others_only_at_its_stops(self):
        linker = _linker_over_stages(_stored(range(3), range(1, 5), states={2}))
        self.assertEqual(self._lookup(linker), [2])
        (states, stops_and_probe) = linker.storage.queries
        self.assertEqual(states, [f"model_p{p}_tp0_cp0_pp0_mamba" for p in range(1, 5)])
        # Stages 1-2's state at stop 2, then every stage's KV and indexer at 2.
        self.assertEqual(
            stops_and_probe,
            ["model_p2_tp0_cp0_pp1_mamba", "model_p2_tp0_cp0_pp2_mamba"]
            + [f"model_p2_tp0_cp0_pp{s}_{o}" for s in range(3) for o in ("k", "indexer")],
        )

    def test_no_common_state_skips_the_page_query(self):
        linker = _linker_over_stages(_stored(range(2), range(1, 5), states={2}))
        self.assertEqual(self._lookup(linker), [])
        self.assertEqual(len(linker.storage.queries), 2)
        self.assertEqual(dict(linker.lookup_counts), {"lookups": 1, "hit_pages": 0})

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


def _full_scan(present, num_pages, num_stages):
    """Stops where every stage holds its state, up to the first page any stage
    lacks KV or indexer: the restorable prefixes a lookup must report."""
    stops = [
        p
        for p in range(1, num_pages + 1)
        if all(f"model_p{p}_tp0_cp0_pp{s}_mamba" in present for s in range(num_stages))
    ]
    contiguous = next(
        (
            p - 1
            for p in range(1, num_pages + 1)
            if not all(
                f"model_p{p}_tp0_cp0_pp{s}_{o}" in present
                for s in range(num_stages)
                for o in ("k", "indexer")
            )
        ),
        num_pages,
    )
    return [p for p in stops if p <= contiguous]


class TestMergedIndexerKeys(CustomTestCase):
    def test_a_merged_page_object_is_one_key_per_page_and_stage(self):
        # SGLANG_LINKER_MERGE_INDEXER: KV and the DSA indexer of a page are one
        # object ("ki"), so lookups, pins and lease renewals ask half the keys.
        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.pools = {
            PoolName.KV: SimpleNamespace(object_name="ki"),
            PoolName.MAMBA: SimpleNamespace(),
        }
        linker.storage = SimpleNamespace(pool_suffixes={}, mla_suffix="tp0_cp0_pp0")
        self.assertEqual(
            linker._pool_key_schema(),
            [("kv", "ki", "tp0_cp0_pp0", False), ("mamba", "mamba", "tp0_cp0_pp0", True)],
        )


class TestLookupCost(CustomTestCase):
    """Rhea 60076: about 24 objects queried per page held PP0 on the master's
    ExistKey ceiling. A lookup queries one stage's state per page and a
    handful of keys per probed stop."""

    def _lookup(self, linker, num_pages):
        transfers = [
            PoolTransfer(name=PoolName.KV, keys=[f"p{p}" for p in range(1, num_pages + 1)])
        ]
        return linker.lookup_stages("rid", transfers)

    def test_the_search_matches_a_full_scan_on_contiguous_prefixes(self):
        rng = random.Random(0)
        for _ in range(300):
            num_pages, num_stages = rng.randint(1, 40), rng.randint(1, 4)
            present = set()
            for stage in range(num_stages):
                for obj in ("k", "indexer"):  # contiguous from page 1, no holes
                    extent = rng.randint(0, num_pages)
                    present |= {f"model_p{p}_tp0_cp0_pp{stage}_{obj}" for p in range(1, extent + 1)}
                states = rng.sample(range(1, num_pages + 1), rng.randint(0, min(5, num_pages)))
                present |= {f"model_p{p}_tp0_cp0_pp{stage}_mamba" for p in states}
            linker = _linker_over_stages(present, num_stages)
            self.assertEqual(
                self._lookup(linker, num_pages), _full_scan(present, num_pages, num_stages)
            )
            # Only the first stage's state is asked for at every page.
            self.assertEqual(len(linker.storage.queries[0]), num_pages)

    def test_keys_per_lookup_are_bounded(self):
        # 4,000 pages, states at two stops on every stage, everything stored.
        num_pages, stages = 4000, 8
        present = _stored(range(stages), range(1, num_pages + 1), states={1000, num_pages})
        linker = _linker_over_stages(present, stages)
        self.assertEqual(self._lookup(linker, num_pages), [1000, num_pages])
        queried = sum(len(query) for query in linker.storage.queries)
        # One stage's states, 7 stages x 2 stops, 8 stages x 2 pools at one probe.
        self.assertEqual(queried, num_pages + 7 * 2 + 8 * 2)
        self.assertLessEqual(len(linker.storage.queries), 2)
        self.assertEqual(linker.lookup_samples[0][0], queried)


class _PinStore:
    """Read sessions over a fixed key set; records starts and ends."""

    def __init__(self, present):
        self.present, self.started, self.ended, self.threads = set(present), [], [], []

    def batch_get_session_start(self, keys):
        self.started.append(list(keys))
        self.threads.append(threading.current_thread().name)
        return [0 if key in self.present else -704 for key in keys]

    def batch_get_session_end(self, keys):
        self.ended.append(list(keys))


def _pinning_linker(present, num_stages=1):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.mla_broadcaster = None
    linker.pinned, linker.pin_samples, linker.pin_holes = {}, [], 0
    linker.pinned_others, linker.pin_expiry = {}, deque()
    linker.prepins, linker.prepin_pool = {}, None
    # Stage 0's own keys come from its pools below; later stages' from schemas.
    linker.stage_schemas = [None] + [_stage_schema(s) for s in range(1, num_stages)]
    # This stage's pools: KV and indexer from the KV transfer, the state trailing.
    linker.pool_group = SimpleNamespace(
        resolve_transfers=lambda transfers, **kwargs: [
            replace(t, name=name, hit_policy=PoolHitPolicy.ALL_PAGES)
            for t in transfers
            if t.name == PoolName.KV
            for name in (PoolName.KV, PoolName.INDEXER)
        ]
        + [t for t in transfers if t.name == PoolName.MAMBA]
    )
    linker.storage = SimpleNamespace(
        store=_PinStore(present),
        _get_hybrid_page_component_keys=lambda keys, t: ([f"{k}_{t.name}" for k in keys], 1),
        _tag_keys=lambda keys: keys,
    )
    return linker


class TestPinHit(CustomTestCase):
    """At admission each stage starts the read sessions of its objects of the
    hit: that checks them and holds them until the load ends the sessions."""

    pages = ["p1", "p2", "p3", "p4"]

    def _pin(self, present, num_stages=1):
        linker = _pinning_linker(present, num_stages)
        transfers = [
            PoolTransfer(name=PoolName.KV, keys=self.pages),
            PoolTransfer(
                name=PoolName.MAMBA, keys=["p4"], hit_policy=PoolHitPolicy.TRAILING_PAGES
            ),
        ]
        return linker, linker.pin_hit("rid", transfers)

    def _all(self):
        return {f"{p}_{pool}" for p in self.pages for pool in ("kv", "indexer")} | {
            "p4_mamba"
        }

    def test_a_complete_hit_stays_pinned_for_its_load(self):
        linker, pages = self._pin(self._all())
        self.assertEqual(pages, 4)
        self.assertEqual(sorted(linker.pinned["rid"]), sorted(self._all()))
        self.assertEqual(linker.storage.store.ended, [])
        linker.unpin("rid")
        self.assertEqual(linker.storage.store.ended, linker.storage.store.started)

    def test_a_missing_object_returns_the_pages_before_it_and_pins_nothing(self):
        linker, pages = self._pin(self._all() - {"p3_indexer"})
        self.assertEqual(pages, 2)
        self.assertEqual((linker.pinned, linker.pin_holes), ({}, 1))
        self.assertEqual(linker.storage.store.ended, linker.storage.store.started)

    def test_a_missing_end_state_rules_out_the_whole_hit(self):
        _, pages = self._pin(self._all() - {"p4_mamba"})
        self.assertEqual(pages, 3)

    def _all_stages(self, num_stages):
        later = set()
        for stage in range(1, num_stages):
            suffix = f"tp0_cp0_pp{stage}"
            later |= {f"{p}_{suffix}_{o}" for p in self.pages for o in ("k", "indexer")}
            later.add(f"p4_{suffix}_mamba")
        return self._all() | later

    def test_pp0_pins_every_stage_and_a_hole_on_a_later_stage_shrinks_the_hit(self):
        # Hecate H106: under PP the keep relay carries PP0's value only, so a
        # hole a later stage's own pin found was overwritten by PP0's.
        linker, pages = self._pin(self._all_stages(3) - {"p3_tp0_cp0_pp2_indexer"}, 3)
        self.assertEqual(pages, 2)
        self.assertEqual((linker.pinned, linker.pinned_others, linker.pin_holes), ({}, {}, 1))
        (started,) = linker.storage.store.started
        self.assertEqual(len(started), 3 * (2 * 4 + 1))  # every stage's objects at once
        self.assertEqual(linker.storage.store.ended, [started])

    def _transfers(self, last="p4"):
        return [
            PoolTransfer(name=PoolName.KV, keys=self.pages[:-1] + [last]),
            PoolTransfer(
                name=PoolName.MAMBA, keys=[last], hit_policy=PoolHitPolicy.TRAILING_PAGES
            ),
        ]

    def test_a_prepin_pins_off_the_scheduler_thread_and_admission_takes_it(self):
        # Hecate H112: the all-stage session start at admission was about 70 ms
        # (p90 about 210 ms) of each hit's admission on PP0's scheduler thread.
        linker = _pinning_linker(self._all_stages(3), 3)
        linker.start_prepin("rid", self._transfers())
        self.assertEqual(linker.prepins["rid"][1].result(timeout=5), (4, linker.pin_samples[0][1]))
        self.assertTrue(linker.storage.store.threads[0].startswith("mooncake-prepin"))
        self.assertEqual(linker.take_prepin("rid", self._transfers()), 4)
        self.assertEqual(linker.last_pin_rpc_s, 0.0)  # admission did not wait
        self.assertEqual(sorted(linker.pinned["rid"]), sorted(self._all()))
        self.assertEqual(linker.prepins, {})

    def test_a_prepin_of_another_hit_is_not_taken_and_a_stale_one_is_released(self):
        linker = _pinning_linker(self._all_stages(3), 3)
        linker.start_prepin("rid", self._transfers())
        linker.prepins["rid"][1].result(timeout=5)
        self.assertIsNone(linker.take_prepin("rid", self._transfers(last="p5")))
        tag, future, pages, own, others, _ = linker.prepins["rid"]
        linker.prepins["rid"] = (tag, future, pages, own, others, -1e9)
        linker.release_stale_prepins()
        self.assertEqual(linker.prepins, {})
        self.assertEqual(linker.storage.store.ended, linker.storage.store.started)

    def test_later_stages_pins_are_held_apart_from_pp0s_own(self):
        linker, pages = self._pin(self._all_stages(3), 3)
        self.assertEqual(pages, 4)
        self.assertEqual(sorted(linker.pinned["rid"]), sorted(self._all()))
        self.assertEqual(len(linker.pinned_others["rid"]), 2 * (2 * 4 + 1))
        linker.unpin("rid")
        self.assertEqual(sorted(linker.storage.store.ended[0]), sorted(self._all_stages(3)))


class TestHoleTrace(CustomTestCase):
    """SGLANG_LINKER_HOLE_TRACE: a hole at admission is matched against when
    each stage queued, landed or removed the puts of the pages behind it."""

    pages = TestPinHit.pages
    _log = "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker"

    def _traced(self, present, num_stages=3, stage=0):
        linker = _pinning_linker(present, num_stages)
        linker.hole_trace = _PutTrace(stage)
        linker.hole_sessions, linker.pp_rank, linker.offload_owner = OrderedDict(), stage, True
        return linker

    def test_pp0_logs_the_missing_objects_with_their_stage_and_times(self):
        present = TestPinHit._all_stages(TestPinHit(), 3) - {"p3_tp0_cp0_pp2_indexer"}
        linker = self._traced(present)
        transfers = TestPinHit._transfers(TestPinHit())
        self.assertEqual(linker.pin_hit("rid", transfers), 2)
        with self.assertLogs(self._log, level="INFO") as logs:
            linker.trace_hole("rid", transfers, 2, looked_up=123.0)
            linker.hole_trace.lines.join()
        pin = next(line for line in logs.output if "hole trace PP0 pin" in line)
        self.assertIn("rid=rid page 2 of 4", pin)
        self.assertIn("(2, 'p3_tp0_cp0_pp2_indexer', -704)", pin)
        self.assertIn("looked up at 123.000", pin)
        hole = next(line for line in logs.output if "hole trace PP0: rid=rid" in line)
        self.assertIn("p3: never written; p4: never written", hole)

    def test_a_stage_reports_each_page_once_and_its_late_put_once(self):
        trace = _PutTrace(stage=2)
        trace.note(["p3"], "queued")
        trace.note(["p3"], "submitted")
        trace.note(["p4"], "completed")
        with self.assertLogs(self._log, level="INFO") as logs:
            trace.report("rid", ["p3", "p4"], hole=50.0)
            trace.report("rid2", ["p3", "p4"], hole=51.0)  # reported already
            trace.note(["p3"], "completed")
            trace.note(["p3", "p4"], "completed")  # p4 never was a late put
            trace.lines.join()
        self.assertEqual(len(logs.output), 2)
        self.assertRegex(logs.output[0], r"PP2: rid=rid hole at 50.000; p3: put queued at "
                         r"[\d.]+, put submitted at [\d.]+; p4: put completed at")
        self.assertRegex(logs.output[1], r"PP2: page p3 put completed at [\d.]+, .* ms "
                         r"after its hole at 50.000")

    def test_a_removed_page_reads_as_removed_and_the_map_is_bounded(self):
        trace = _PutTrace(stage=1, capacity=2)
        trace.note(["a", "b"], "completed")
        trace.note(["a"], "removed", "budget")
        self.assertIn("removed (budget) at", _PutTrace.describe(trace.pages["a"]))
        trace.note(["c"], "queued")
        self.assertEqual(list(trace.pages), ["a", "c"])

    def test_off_by_default_records_nothing(self):
        linker = _pinning_linker(set(), 3)
        self.assertIsNone(linker.hole_trace)
        linker.trace_hole("rid", TestPinHit._transfers(TestPinHit()), 0, None)


class _RemovingStore:
    def __init__(self, codes=None):
        self.removed, self.codes = [], dict(codes or {})

    def remove(self, key):
        self.removed.append(key)
        return self.codes.get(key, 0)

    def batch_remove(self, keys):
        self.batches = getattr(self, "batches", 0) + 1
        return [self.remove(key) for key in keys]


def _capped_linker(cap):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.state_path_cap = cap
    linker.state_parent, linker.state_children = {}, Counter()
    linker.state_removed = linker.state_remove_failed = 0
    linker.state_remove_counts, linker.state_remove_retries = Counter(), deque()
    return linker


class TestStatePathCap(CustomTestCase):
    """A conversation resumes from its deepest KDA states; each older linear
    state past SGLANG_LINKER_STATE_PATH_CAP is dead weight in the store."""

    def test_cap_one_keeps_only_the_deepest_state(self):
        linker = _capped_linker(1)
        self.assertEqual(linker._cap_state_path(["s1"], ["a", "s1"]), [])
        self.assertEqual(linker._cap_state_path(["s2"], ["a", "s1", "b", "s2"]), ["s1"])
        self.assertEqual(
            linker._cap_state_path(["s3"], ["a", "s1", "b", "s2", "c", "s3"]), ["s2"]
        )
        self.assertEqual(linker.state_parent, {"s3": None})

    def test_cap_three_keeps_the_three_deepest(self):
        linker = _capped_linker(3)
        path = []
        dropped = []
        for turn in range(1, 6):
            path += [f"x{turn}", f"s{turn}"]
            dropped += linker._cap_state_path([f"s{turn}"], list(path))
        self.assertEqual(dropped, ["s1", "s2"])
        self.assertEqual(linker.state_parent, {"s3": None, "s4": "s3", "s5": "s4"})

    def test_a_branch_point_stays_and_its_branches_relink(self):
        linker = _capped_linker(2)
        linker._cap_state_path(["sys"], ["p", "sys"])
        linker._cap_state_path(["a1"], ["p", "sys", "a", "a1"])
        linker._cap_state_path(["b1"], ["p", "sys", "b", "b1"])
        self.assertEqual(linker.state_children["sys"], 2)
        path = ["p", "sys", "a", "a1", "a", "a2"]
        self.assertEqual(linker._cap_state_path(["a2"], path), [])  # past the cap: 'sys'
        # a3 drops a1 but stops at the branch point, and hangs a2 from it.
        self.assertEqual(linker._cap_state_path(["a3"], path + ["a", "a3"]), ["a1"])
        self.assertEqual(linker.state_parent["a2"], "sys")
        self.assertEqual(linker.state_children["sys"], 2)

    def test_cap_one_drops_a_shared_state_that_has_one_child_so_far(self):
        linker = _capped_linker(1)
        linker._cap_state_path(["sys"], ["p", "sys"])
        self.assertEqual(linker._cap_state_path(["a1"], ["p", "sys", "a", "a1"]), ["sys"])

    def test_removal_retries_a_leased_state_once_its_lease_can_expire(self):
        self._removal_retries_a_leased_state(batch=False)

    def test_batch_removal_retries_a_leased_state_once_its_lease_can_expire(self):
        # SGLANG_LINKER_BATCH_REMOVE: the same outcome, one store call per pass.
        linker = self._removal_retries_a_leased_state(batch=True)
        self.assertEqual(linker.storage.store.batches, 2)

    def _removal_retries_a_leased_state(self, batch):
        # Mooncake refuses an object under a live lease (-706); a lookup or the
        # lease refresher may have touched the old state within the lease TTL.
        linker = _linker_over_stages(set(), num_stages=1)
        linker.batch_remove = batch
        leased, gone = "model_s1_tp0_cp0_pp0_mamba", "model_s2_tp0_cp0_pp0_mamba"
        linker.storage.store = _RemovingStore({leased: -706, gone: -704})
        linker.storage.pool_suffixes = {
            name: "tp0_cp0_pp0" for name in (PoolName.KV, PoolName.MAMBA, PoolName.INDEXER)
        }
        linker.storage.mla_suffix = "tp0_cp0_pp0"
        linker.pools = {PoolName.KV: None, PoolName.MAMBA: None, PoolName.INDEXER: None}
        linker.state_removed = linker.state_remove_failed = 0
        linker.state_remove_counts, linker.state_remove_retries = Counter(), deque()
        clock = "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker.time.monotonic"
        with patch(clock, return_value=100.0):
            linker._remove_states(["s1", "s2", "s3"])
        self.assertEqual(linker.state_removed, 1)  # s3
        self.assertEqual(linker.state_remove_counts["already_gone"], 1)  # s2
        self.assertEqual(len(linker.state_remove_retries), 1)  # s1
        del linker.storage.store.codes[leased]  # its lease has run out
        with patch(clock, return_value=105.0):
            linker._remove_states([])  # not due yet
        self.assertEqual(linker.state_removed, 1)
        with patch(clock, return_value=112.0):
            linker._remove_states([])
        self.assertEqual(linker.state_removed, 2)
        self.assertEqual(linker.storage.store.removed.count(leased), 2)
        self.assertEqual(linker.state_remove_failed, 0)
        return linker


class _SettingStorage:
    def __init__(self, ok=True):
        self.ok, self.store = ok, _RemovingStore()
        self.last_exist_skipped = 0
        self.pool_suffixes = {name: "tp0_cp0_pp0" for name in (PoolName.KV, PoolName.MAMBA)}
        self.mla_suffix = "tp0_cp0_pp0"

    def _tag_keys(self, keys):
        return [f"model_{key}" for key in keys]

    def batch_set_v2(self, transfers, extra_info):
        # p1 and p2 were stored already: only p3's KV and state are put.
        self.last_put_objects = {PoolName.KV: 1, PoolName.MAMBA: 1}
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
        self.assertEqual(linker.storage.store.removed, ["model_p1_tp0_cp0_pp0_mamba"])

    def test_write_counts_only_what_the_store_put(self):
        # 57360/58614: offered bytes (15.1 GB/s) were 7x what the store grew by
        # (2.2 GB/s): batch_set_v2 skips objects it already holds.
        with patch.object(MooncakeDirectLinker, "_maybe_log_stats"):
            linker = self._run(ok=True)
        self.assertEqual(
            linker.write_bytes,
            Counter({PoolName.KV: 100, PoolName.MAMBA: 5000, "offered": 5300, "states": 1}),
        )

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
