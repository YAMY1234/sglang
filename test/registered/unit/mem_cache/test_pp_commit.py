"""No CUDA or scheduler import: execute the production commit state machine."""

import importlib.util
import sys
import unittest
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[4] / "python/sglang/srt/mem_cache/pp_commit.py"
)
spec = importlib.util.spec_from_file_location("pp_commit_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
CommitBoundary, OperationId = module.CommitBoundary, module.OperationId


class Cluster:
    def __init__(self, **kwargs):
        self.now = 0.0
        self.ranks = [
            CommitBoundary(i, 3, clock=lambda: self.now, **kwargs) for i in range(3)
        ]
        self.effects = [[] for _ in self.ranks]
        self.reports = {}

    def stage(self, rank, name="page-key", generation=0, **kwargs):
        op = OperationId(0, "backup", name, generation)
        self.ranks[rank].stage(
            op,
            [name, generation],
            lambda: self.effects[rank].append((name, generation)),
            **kwargs,
        )
        return op

    def tick(self, publish=True):
        frame = self.ranks[0].leader_frame(self.reports)
        for rank in self.ranks:
            rank.accept_frame(frame)
        if publish:
            self.reports = {i: rank.ready() for i, rank in enumerate(self.ranks) if i}
        return frame


class CommitTest(unittest.TestCase):
    def test_normal_has_one_logical_round_delay_and_exactly_once_effects(self):
        c = Cluster()
        for rank in range(3):
            c.stage(rank)
        first = c.tick()
        self.assertEqual(first["commit"], 0)
        self.assertEqual(c.effects, [[], [], []])
        self.assertEqual(c.tick()["commit"], 1)
        for _ in range(10):
            c.tick()
        self.assertEqual(c.effects, [[("page-key", 0)]] * 3)

    def test_late_one_round_ack_keeps_unlock_and_belief_private(self):
        c = Cluster()
        for rank in (0, 2):
            c.stage(rank, pinned_bytes=100)
        c.tick()
        c.tick()
        self.assertEqual(c.effects, [[], [], []])
        self.assertEqual(c.ranks[0].pinned_bytes, 100)
        c.stage(1, pinned_bytes=100)
        c.tick()  # receive previous, still not-ready PP1 snapshot
        self.assertEqual(c.effects, [[], [], []])
        c.tick()
        self.assertEqual(c.effects, [[("page-key", 0)]] * 3)
        self.assertTrue(all(r.pinned_bytes == 0 for r in c.ranks))

    def test_permanent_missing_operation_warns_once_then_fails_not_blocks(self):
        c = Cluster(stall_seconds=120)
        for rank in (0, 2):
            c.stage(rank)
        with self.assertLogs(module.logger, level="WARNING") as logs:
            for _ in range(12):
                c.tick()
        self.assertEqual(len(logs.output), 3)
        self.assertIn("page-key", logs.output[1])
        self.assertEqual(c.ranks[1].snapshot()["missing"][0][0], 1)
        self.assertEqual(c.effects, [[], [], []])
        c.now = 121
        with self.assertRaisesRegex(RuntimeError, "frontier stalled"):
            c.tick()
        self.assertEqual(c.ranks[0].stats["commit_stall"], 1)

    def test_equal_counts_different_keys_do_not_confirm(self):
        c = Cluster()
        c.stage(0, "A")
        c.stage(1, "B")
        c.stage(2, "A")
        c.tick()
        self.assertEqual([r.confirmed for r in c.ranks], [1, 0, 1])
        self.assertEqual(c.tick()["commit"], 0)

    def test_ack_reordering_matches_identity_not_fifo_position(self):
        c = Cluster()
        for rank, names in enumerate((("A", "B"), ("B", "A"), ("A", "B"))):
            for name in names:
                c.stage(rank, name)
        c.tick()
        c.tick()
        self.assertEqual(c.effects, [[("A", 0), ("B", 0)]] * 3)

    def test_hole_prevents_ready_later_operation_committing_early(self):
        c = Cluster()
        for rank in (0, 2):
            c.stage(rank, "A")
        for rank in range(3):
            c.stage(rank, "B")
        c.tick()
        c.tick()
        self.assertEqual(c.effects, [[], [], []])
        c.stage(1, "A")
        c.tick()
        c.tick()
        self.assertEqual(c.effects, [[("A", 0), ("B", 0)]] * 3)

    def test_duplicate_ack_before_and_after_commit_is_idempotent(self):
        c = Cluster()
        for rank in range(3):
            c.stage(rank)
            c.stage(rank)
        c.tick()
        c.tick()
        for rank in range(3):
            c.stage(rank)
        c.tick()
        self.assertEqual(c.effects, [[("page-key", 0)]] * 3)
        self.assertTrue(all(r.stats["duplicates"] == 2 for r in c.ranks))

    def test_conflicting_payload_is_not_a_duplicate(self):
        c = Cluster()
        op = c.stage(0)
        with self.assertRaisesRegex(RuntimeError, "conflicting duplicate"):
            c.ranks[0].stage(op, ["wrong"], lambda: None)

    def test_cross_rank_payload_mismatch_fails_before_side_effect(self):
        c = Cluster()
        op = c.stage(0)
        c.ranks[1].stage(op, ["wrong"], lambda: c.effects[1].append("bad"))
        with self.assertRaisesRegex(RuntimeError, "payload mismatch"):
            c.tick()
        self.assertEqual(c.effects, [[], [], []])

    def test_repeated_content_uses_new_generation(self):
        c = Cluster()
        for generation in range(2):
            for rank in range(3):
                c.stage(rank, generation=generation)
            c.tick()
            c.tick()
        self.assertEqual(c.effects, [[("page-key", 0), ("page-key", 1)]] * 3)
        self.assertTrue(all(len(r.hashes) == 1 for r in c.ranks))

    def test_reset_cannot_drop_pinned_work_or_accept_previous_epoch(self):
        c = Cluster()
        op = c.stage(0)
        with self.assertRaisesRegex(RuntimeError, "pinned uncommitted"):
            c.ranks[0].reset(1)
        for rank in (1, 2):
            c.stage(rank)
        c.tick()
        c.tick()
        for rank in c.ranks:
            rank.reset(1)
            with self.assertRaisesRegex(RuntimeError, "stale epoch"):
                rank.stage(op, None, lambda: None)

    def test_bounded_pending_rejects_more_resources_without_early_free(self):
        c = Cluster(max_pending=1)
        c.stage(0, "A", pinned_bytes=100)
        self.assertFalse(c.ranks[0].admission_open)
        with self.assertRaisesRegex(RuntimeError, "resource bound"):
            c.stage(0, "B")
        self.assertEqual(c.ranks[0].pinned_bytes, 100)
        self.assertEqual(c.effects[0], [])

    def test_stale_ready_does_not_roll_back_or_skip_and_bad_digest_fails(self):
        c = Cluster()
        for rank in range(3):
            c.stage(rank)
        c.tick()
        stale = c.reports.copy()
        c.tick()
        for rank in range(3):
            c.stage(rank, "B")
        c.reports = stale
        self.assertEqual(c.tick()["commit"], 1)
        c.reports[1]["digest"] = "different-manifest"
        with self.assertRaisesRegex(RuntimeError, "manifest digest mismatch"):
            c.tick()

    def test_frame_sequence_and_unconfirmed_commit_fail_closed(self):
        c = Cluster()
        with self.assertRaisesRegex(RuntimeError, "sequence/epoch"):
            c.ranks[1].accept_frame(
                {"epoch": 0, "round": 2, "commit": 0, "entries": []}
            )
        with self.assertRaisesRegex(RuntimeError, "unconfirmed"):
            c.ranks[1].accept_frame(
                {"epoch": 0, "round": 1, "commit": 1, "entries": []}
            )

    def test_idle_time_is_not_stall(self):
        c = Cluster()
        c.now = 10000
        for _ in range(20):
            c.tick()
        self.assertTrue(all(r.stats["commit_stall"] == 0 for r in c.ranks))

    def test_reordered_generations_do_not_discard_an_older_real_ack(self):
        c = Cluster()
        for generation in (1, 0):
            for rank in range(3):
                c.stage(rank, generation=generation)
            c.tick()
            c.tick()
        self.assertEqual(c.effects, [[("page-key", 1), ("page-key", 0)]] * 3)
        self.assertTrue(
            all(r.completed[("backup", "page-key")] == (1, set()) for r in c.ranks)
        )

    def test_single_ack_cannot_overshoot_pinned_byte_bound(self):
        c = Cluster(max_pinned_bytes=10)
        with self.assertRaisesRegex(RuntimeError, "resource bound"):
            c.stage(0, pinned_bytes=11)
        self.assertEqual(c.ranks[0].pinned_bytes, 0)

    def test_orphan_ack_age_is_not_hidden_by_other_commits(self):
        c = Cluster(stall_seconds=10)
        c.stage(1, "orphan")
        for rank in range(3):
            c.stage(rank, "normal")
        c.tick()
        c.now = 11
        with self.assertRaisesRegex(RuntimeError, "aged local ACK"):
            c.tick()


if __name__ == "__main__":
    unittest.main(verbosity=2)
