"""Exercise production ACK consumers/producers with real CPU tensors.

Only heavyweight scheduler/model imports are omitted. Queue drain, logical
backup completion, release envelopes, belief storage and bridge are real code.
"""

import ast
import importlib.util
import logging
import os
import queue
import sys
import time
import types
import unittest
from enum import Enum
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


core = load("sglang.srt.mem_cache.pp_commit", "mem_cache/pp_commit.py")
transport = load(
    "sglang.srt.mem_cache.pp_commit_transport", "mem_cache/pp_commit_transport.py"
)
tags = load(
    "sglang.srt.distributed.communication_tags", "distributed/communication_tags.py"
)
proposals_module = load(
    "sglang.srt.mem_cache.pp_belief_proposals", "mem_cache/pp_belief_proposals.py"
)
bridge_module = load(
    "sglang.srt.mem_cache.pp_commit_bridge", "mem_cache/pp_commit_bridge.py"
)
belief_module = load(
    "phase1_belief", "mem_cache/buffer_mode/storage_existence_cache.py"
)


class PoolName(str, Enum):
    KV = "kv"
    MAMBA = "mamba"

    def __str__(self):
        return self.value


@lru_cache
def method(relative, name):
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    ns = {
        "torch": torch,
        "os": os,
        "PoolName": PoolName,
        "logger": logging.getLogger(__name__),
        "time": time,
        "P2PTag": tags.P2PTag,
    }
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(  # noqa: S102 -- extract checked-in production methods, not external input
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])),
            str(path),
            "exec",
        ),
        ns,
    )
    return ns[name]


class FakeHost:
    def __init__(self):
        self.freed = []
        self.entry_map = {
            name: types.SimpleNamespace(
                host_pool=types.SimpleNamespace(get_size_per_token=lambda: 8)
            )
            for name in PoolName
        }

    def free(self, indices, pool=PoolName.KV):
        self.freed.append((str(pool), indices.tolist()))


class Cluster:
    def __init__(self, enabled=True):
        self.reports = {}
        self.frame = None
        self.caches = []
        for rank in range(3):
            cc = types.SimpleNamespace(
                prefetch_hit_queue=queue.Queue(),
                ack_prefetch_queue=queue.Queue(),
                ack_backup_queue=queue.Queue(),
                host_mem_release_queue=queue.Queue(),
                extra_host_mem_release_queues={PoolName.MAMBA: queue.Queue()},
                mem_pool_host=FakeHost(),
                backup_queue=queue.Queue(),
            )
            c = types.SimpleNamespace(
                pp_rank=rank,
                pp_size=3,
                cache_controller=cc,
                host_memory_mode="cache",
                _prefetch_stats_last_log=time.monotonic(),
                _l3_tier_stats={},
                _pp_commit=None,
                ongoing_backup={},
                _write_behind_inflight={},
                enable_storage_metrics=False,
                storage_metrics_collector=None,
                storage_existence_cache=belief_module.StorageExistenceCache(),
                unlocks=[],
            )
            c.dec_host_lock_ref = lambda node, lock, c=c: c.unlocks.append((node, lock))
            c._all_reduce_attn_groups = lambda tensor, op: None
            for name in ("_drain_storage_control_queues_impl", "_commit_backup_ack"):
                setattr(
                    c,
                    name,
                    types.MethodType(
                        method("mem_cache/unified_radix_cache.py", name), c
                    ),
                )
            cc._append_host_mem_release_pages = types.MethodType(
                method(
                    "mem_cache/hybrid_cache/hybrid_cache_controller.py",
                    "_append_host_mem_release_pages",
                ),
                cc,
            )
            if enabled:
                mailbox = types.SimpleNamespace(
                    poll=lambda: self.reports,
                    publish=lambda report, rank=rank: (
                        self.reports.__setitem__(rank, dict(report)) if rank else None
                    ),
                    close=lambda: True,
                )
                with patch.object(
                    bridge_module, "PreviousRoundReports", return_value=mailbox
                ):
                    c._pp_commit = bridge_module.PPCommitBridge(c, None)
                c._pp_commit._frame = lambda frame, rank=rank: self.broadcast(
                    frame, rank
                )
                cc.pp_commit_bridge = c._pp_commit
                c.storage_existence_cache.defer_mutation = c._pp_commit.defer_belief
            self.caches.append(c)

    def broadcast(self, frame, rank):
        if rank == 0:
            self.frame = frame
        return self.frame

    def backup(self, rank):
        c = self.caches[rank]
        # Deliberately different node IDs, local operation IDs and page addresses.
        op = types.SimpleNamespace(
            id=100 + rank,
            hash_value=["shared-page"],
            pool_transfers=[],
            host_indices=torch.tensor([rank * 10]),
            completed_tokens=1,
        )
        if c._pp_commit:
            c._pp_commit.note_backup(op)
        c.ongoing_backup[op.id] = (1000 + rank, "locked")
        c._write_behind_inflight[op.id] = 1
        c.cache_controller.ack_backup_queue.put(op)
        return op

    def drain(self, rank):
        c = self.caches[rank]
        cc = c.cache_controller
        c._drain_storage_control_queues_impl(
            0,
            0,
            cc.ack_backup_queue.qsize(),
            cc.host_mem_release_queue.qsize(),
            {PoolName.MAMBA: cc.extra_host_mem_release_queues[PoolName.MAMBA].qsize()},
            False,
        )

    def settle(self, rounds=16):
        for _ in range(rounds):
            self.tick()

    def tick(self):
        for c in self.caches:
            c._pp_commit.tick()


class IntegrationTest(unittest.TestCase):
    def test_reset_rejects_owned_release_before_ack_is_prepared(self):
        c = Cluster().caches[0]
        bridge = c._pp_commit
        identity = bridge.note_release(
            ["pending-release"], PoolName.KV, torch.tensor([1, 2]), 1
        )
        self.assertFalse(bridge.state.prepared)
        with self.assertRaisesRegex(RuntimeError, "physical ACK ownership"):
            bridge.reset()
        self.assertIn(identity, bridge.issued_releases)
        self.assertEqual(bridge.state.epoch, 0)
        self.assertEqual(c.cache_controller.mem_pool_host.freed, [])

    def test_reset_rejects_owned_backup_before_ack_is_prepared(self):
        cluster = Cluster()
        cluster.backup(0)
        c = cluster.caches[0]
        bridge = c._pp_commit
        self.assertFalse(bridge.state.prepared)
        with self.assertRaisesRegex(RuntimeError, "physical ACK ownership"):
            bridge.reset()
        self.assertEqual(len(bridge.issued_backups), 1)
        self.assertEqual(bridge.state.epoch, 0)
        self.assertEqual(c.unlocks, [])

    def test_physical_backup_drains_but_lock_and_belief_wait_for_all_stages(self):
        cluster = Cluster()
        for rank in (0, 2):
            cluster.backup(rank)
            cluster.drain(rank)
        cluster.tick()
        cluster.tick()
        for rank in (0, 2):
            c = cluster.caches[rank]
            self.assertTrue(c.cache_controller.ack_backup_queue.empty())
            self.assertEqual(c._l3_tier_stats["pp_backup_drained"], 1)
            self.assertEqual(c.unlocks, [])
            self.assertEqual(len(c.storage_existence_cache), 0)
            self.assertEqual(len(c.ongoing_backup), 1)
        cluster.backup(1)
        cluster.drain(1)
        cluster.tick()
        cluster.tick()
        for rank, c in enumerate(cluster.caches):
            self.assertEqual(c.unlocks, [(1000 + rank, "locked")])
            self.assertTrue(c.storage_existence_cache.contains("kv", "shared-page"))
            self.assertEqual(c.ongoing_backup, {})
            self.assertEqual(c._write_behind_inflight, {})

    def test_release_producer_tags_logical_range_not_local_page_address(self):
        cluster = Cluster()
        for rank, c in enumerate(cluster.caches):
            cc = c.cache_controller
            cc._append_host_mem_release_pages(
                cc.host_mem_release_queue,
                torch.tensor([10 * rank, 10 * rank + 1]),
                1,
                commit_key=["request", "tail", 128],
                pool=PoolName.KV,
            )
            cluster.drain(rank)
            self.assertEqual(cc.mem_pool_host.freed, [])
        cluster.tick()
        self.assertTrue(
            all(not c.cache_controller.mem_pool_host.freed for c in cluster.caches)
        )
        cluster.tick()
        for rank, c in enumerate(cluster.caches):
            self.assertEqual(
                c.cache_controller.mem_pool_host.freed,
                [("kv", [rank * 10, rank * 10 + 1])],
            )

    def test_belief_add_and_delete_are_common_effects_and_lru_is_deferred(self):
        cluster = Cluster()
        for c in cluster.caches:
            c.storage_existence_cache.add("kv", ["a", "b"])
            self.assertEqual(len(c.storage_existence_cache), 0)
        cluster.settle()
        before = cluster.caches[0].storage_existence_cache.commit_digest
        for c in cluster.caches:
            self.assertEqual(len(c.storage_existence_cache), 2)
            c.storage_existence_cache.invalidate_beyond("kv", ["a", "b"], 1)
            self.assertTrue(c.storage_existence_cache.contains("kv", "b"))
        cluster.settle()
        for c in cluster.caches:
            self.assertFalse(c.storage_existence_cache.contains("kv", "b"))
            self.assertNotEqual(c.storage_existence_cache.commit_digest, before)

    def test_absent_belief_delete_repeats_do_not_create_orphan_operations(self):
        cluster = Cluster()
        # A prefetch miss may be reported again at a later PP stage. Deleting
        # an absent advisory entry has no logical effect or owned resources.
        for _ in range(3):
            cluster.caches[1].storage_existence_cache.invalidate_beyond(
                "kv", ["absent-page"], 0
            )
        cluster.settle()
        for c in cluster.caches:
            self.assertEqual(c._pp_commit.state.snapshot()["pending"], 0)
        # Extra no-op reports must not shift generations for the next real delete.
        for c in cluster.caches:
            c.storage_existence_cache.add("kv", ["absent-page"])
        cluster.settle()
        for c in cluster.caches:
            c.storage_existence_cache.invalidate_beyond("kv", ["absent-page"], 0)
        cluster.settle()
        for c in cluster.caches:
            self.assertEqual(len(c.storage_existence_cache), 0)
            self.assertEqual(c._pp_commit.state.snapshot()["pending"], 0)

    def test_duplicate_belief_deletes_coalesce_until_common_commit(self):
        cluster = Cluster()
        for c in cluster.caches:
            c.storage_existence_cache.add("kv", ["a", "b"])
        cluster.settle()
        for rank, c in enumerate(cluster.caches):
            for _ in range(rank + 1):
                c.storage_existence_cache.invalidate_beyond("kv", ["b"], 0)
        cluster.settle()
        for c in cluster.caches:
            self.assertEqual(len(c.storage_existence_cache), 1)
            self.assertEqual(c._pp_commit.state.snapshot()["pending"], 0)

    def test_belief_dedup_preserves_intervening_add_delete_order(self):
        cluster = Cluster()
        for c in cluster.caches:
            c.storage_existence_cache.add("kv", ["a"])
            c.storage_existence_cache.invalidate_beyond("kv", ["a"], 0)
            c.storage_existence_cache.add("kv", ["a"])
            c.storage_existence_cache.invalidate_beyond("kv", ["a"], 0)
        cluster.settle()
        for c in cluster.caches:
            self.assertEqual(len(c.storage_existence_cache), 0)
            self.assertEqual(c._pp_commit.state.snapshot()["pending"], 0)
            self.assertEqual(c._pp_commit.belief_proposals.positive, {})

    def test_downstream_belief_proposal_is_assigned_by_leader_for_all_ranks(self):
        cluster = Cluster()
        cluster.caches[2].storage_existence_cache.add("kv", ["only-downstream"])
        self.assertTrue(all(not c._pp_commit.state.prepared for c in cluster.caches))
        cluster.settle()
        for c in cluster.caches:
            self.assertTrue(
                c.storage_existence_cache.peek_present("kv", "only-downstream")
            )
            self.assertEqual(c._pp_commit.belief_proposals.snapshot()["unassigned"], 0)
        cluster.caches[1].storage_existence_cache.invalidate_beyond(
            "kv", ["only-downstream"], 0
        )
        cluster.settle()
        for c in cluster.caches:
            self.assertFalse(
                c.storage_existence_cache.peek_present("kv", "only-downstream")
            )
            self.assertEqual(c._pp_commit.state.committed, 2)
            self.assertEqual(c._pp_commit.belief_proposals.snapshot()["pending"], 0)

    def test_proposal_timeout_keeps_origin_evidence_and_does_not_apply_locally(self):
        cluster = Cluster()
        c = cluster.caches[1]
        c.storage_existence_cache.add("kv", ["one-sided"])
        proposal, born = next(iter(c._pp_commit.belief_proposals.pending.values()))
        with (
            patch.object(proposals_module.time, "monotonic", return_value=born + 121),
            self.assertRaisesRegex(RuntimeError, "origin_site"),
        ):
            c._pp_commit.belief_proposals.check_age()
        self.assertEqual(len(c.storage_existence_cache), 0)
        self.assertEqual(proposal["origin"], 1)

    def test_downstream_only_physical_ack_gets_manifest_but_never_fake_completion(self):
        cluster = Cluster()
        cluster.backup(2)
        cluster.drain(2)
        cluster.settle()
        self.assertTrue(cluster.caches[0]._pp_commit.state.manifest)
        self.assertFalse(cluster.caches[0]._pp_commit.state.prepared)
        self.assertTrue(all(not c.unlocks for c in cluster.caches))
        self.assertTrue(all(c._pp_commit.state.committed == 0 for c in cluster.caches))

    def test_bounded_chunked_proposals_fit_ready_and_finish_with_no_residue(self):
        cluster = Cluster()
        c = cluster.caches[2]
        hashes = [f"{i:064x}" for i in range(33)]
        c.storage_existence_cache.add("kv", hashes)
        self.assertEqual(len(c._pp_commit.belief_proposals.pending), 5)
        cluster.settle(40)
        for c in cluster.caches:
            self.assertEqual(len(c.storage_existence_cache), 33)
            self.assertEqual(c._pp_commit.belief_proposals.snapshot()["unassigned"], 0)
        for report in cluster.reports.values():
            self.assertEqual(
                transport.PreviousRoundReports.decode(
                    transport.PreviousRoundReports.encode(
                        {"wire_seq": 1, "report": report}
                    )
                )["report"],
                report,
            )

    def test_three_origin_burst_drains_before_unchanged_120s_deadline(self):
        cluster = Cluster()
        now = [1000.0]
        hashes = [f"{i:064x}" for i in range(8000)]
        with patch.object(proposals_module.time, "monotonic", side_effect=lambda: now[0]):
            for c in cluster.caches:
                c._pp_commit.state.clock = lambda: now[0]
                c.storage_existence_cache.add("kv", hashes)
                stats = c._pp_commit.belief_proposals.snapshot()
                self.assertEqual(stats["pending"], 1000)
                self.assertEqual(stats["batch_overflow"], 984)
                self.assertEqual(c._pp_commit.belief_proposals.timeout, 120.0)
            for step in range(250):
                # Approximate the observed 2.7 logical rounds/s while prefill
                # runs. The old one-head protocol needs >120s for this burst.
                now[0] += 0.37
                cluster.tick()
                for report in cluster.reports.values():
                    self.assertLessEqual(len(report["belief_proposals"]), 16)
                    transport.PreviousRoundReports.encode(
                        {"wire_seq": step + 1, "report": report}
                    )
                self.assertLessEqual(len(core.canonical(cluster.frame)), 32768 * 8)
                if all(not c._pp_commit.belief_proposals.pending for c in cluster.caches):
                    break
            else:
                self.fail("bounded burst did not finish in 92.5 simulated seconds")
        for c in cluster.caches:
            self.assertEqual(len(c.storage_existence_cache), 8000)
            self.assertEqual(c._pp_commit.state.committed, 3000)
            self.assertFalse(c._pp_commit.state.prepared)
            self.assertEqual(c._pp_commit.belief_proposals.completed, 1000)

    def test_coalesced_batch_keeps_serial_order_and_rejects_oversize(self):
        cluster = Cluster()
        c = cluster.caches[2]
        c.storage_existence_cache.add("kv", [f"{i:064x}" for i in range(256)])
        first = c._pp_commit.belief_proposals.batch()
        self.assertEqual([x["serial"] for x in first], list(range(1, 17)))
        self.assertEqual(c._pp_commit.belief_proposals.batch(), first)
        leader = cluster.caches[0]._pp_commit
        with self.assertRaisesRegex(RuntimeError, "batch bound exceeded"):
            leader._assign_belief_proposals({2: {"belief_proposals": first + [first[-1]]}})
        self.assertFalse(leader.state.prepared)
        with self.assertRaisesRegex(RuntimeError, "sequence gap"):
            leader._assign_belief_proposals({2: {"belief_proposals": first[1:]}})
        self.assertFalse(leader.state.prepared)

    def test_default_off_preserves_immediate_backup_release_and_belief(self):
        cluster = Cluster(enabled=False)
        for rank, c in enumerate(cluster.caches):
            cluster.backup(rank)
            cc = c.cache_controller
            cc._append_host_mem_release_pages(
                cc.host_mem_release_queue, torch.tensor([7]), 1
            )
            cluster.drain(rank)
            self.assertEqual(c.unlocks, [(1000 + rank, "locked")])
            self.assertEqual(cc.mem_pool_host.freed, [("kv", [7])])
            self.assertEqual(len(c.storage_existence_cache), 1)

    def test_missing_release_identity_fails_instead_of_freeing_page(self):
        cluster = Cluster()
        c = cluster.caches[0]
        c.cache_controller.host_mem_release_queue.put(torch.tensor([7]))
        with self.assertRaisesRegex(TypeError, "lacks producer identity"):
            cluster.drain(0)
        self.assertEqual(c.cache_controller.mem_pool_host.freed, [])

    def test_same_count_different_release_parent_does_not_advance(self):
        cluster = Cluster()
        for rank, c in enumerate(cluster.caches):
            cc = c.cache_controller
            cc._append_host_mem_release_pages(
                cc.host_mem_release_queue,
                torch.tensor([7]),
                1,
                commit_key=["request" if rank != 1 else "other-request"],
            )
            cluster.drain(rank)
        cluster.tick()
        cluster.tick()
        self.assertTrue(
            all(not c.cache_controller.mem_pool_host.freed for c in cluster.caches)
        )

    def test_all_cache_release_producers_supply_identity_context(self):
        for relative in (
            "mem_cache/unified_radix_cache.py",
            "mem_cache/hybrid_cache/hybrid_cache_controller.py",
        ):
            tree = ast.parse((ROOT / relative).read_text())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "append_host_mem_release"
                ):
                    self.assertIn(
                        "commit_key",
                        {kw.arg for kw in node.keywords},
                        ast.unparse(node),
                    )

    def test_partial_release_range_cannot_confirm_until_all_physical_pages_arrive(self):
        cluster = Cluster()
        for rank, c in enumerate(cluster.caches):
            cc = c.cache_controller
            cc._append_host_mem_release_pages(
                cc.host_mem_release_queue,
                torch.tensor([1, 2]),
                1,
                commit_key=["req", "range"],
            )
            if rank == 1:
                first = cc.host_mem_release_queue.get_nowait()
                c._pp_commit.stage_release(first)
            else:
                cluster.drain(rank)
        cluster.tick()
        cluster.tick()
        self.assertTrue(
            all(not c.cache_controller.mem_pool_host.freed for c in cluster.caches)
        )
        cluster.drain(1)
        cluster.tick()
        cluster.tick()
        self.assertTrue(
            all(
                c.cache_controller.mem_pool_host.freed == [("kv", [1, 2])]
                for c in cluster.caches
            )
        )

    def test_pending_write_coverage_is_private_and_removed_only_on_commit(self):
        cluster = Cluster()
        for rank, c in enumerate(cluster.caches):
            cluster.backup(rank)
            self.assertTrue(c._pp_commit.backup_pending("kv", ["shared-page"]))
            self.assertFalse(c.storage_existence_cache.contains("kv", "shared-page"))
            cluster.drain(rank)
        cluster.tick()
        cluster.tick()
        for c in cluster.caches:
            self.assertFalse(c._pp_commit.backup_pending("kv", ["shared-page"]))
            self.assertTrue(c.storage_existence_cache.contains("kv", "shared-page"))

    def test_orphan_behind_pp0_physical_budget_is_detected(self):
        cluster = Cluster()
        op = cluster.backup(1)
        op.pp_commit_ack_at = time.monotonic() - 121
        with self.assertRaisesRegex(RuntimeError, "orphan backup ACK"):
            cluster.caches[1]._pp_commit.tick()

    def test_belief_enum_and_string_equal_keys_keep_same_digest(self):
        cluster = Cluster()
        c = cluster.caches[0]
        c._pp_commit.applying = True
        c.storage_existence_cache.add(PoolName.KV, ["a"])
        self.assertNotEqual(c.storage_existence_cache.commit_digest, 0)
        c.storage_existence_cache.invalidate_beyond("kv", ["a"], 0)
        self.assertEqual(c.storage_existence_cache.commit_digest, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
