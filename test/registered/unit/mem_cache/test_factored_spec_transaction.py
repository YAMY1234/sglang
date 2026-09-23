"""CPU transaction/lifecycle tests; production-kernel equality is a GPU guard."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch


def load_transaction():
    path = Path(__file__).resolve().parents[4] / "python/sglang/srt/mem_cache/gdn_factored_spec.py"
    spec = importlib.util.spec_from_file_location("factored_spec_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FactoredGDNVerifyState


Transaction = load_transaction()


def pool():
    return SimpleNamespace(
        cfg=SimpleNamespace(r=8, m=8),
        a=torch.arange(2*8*2*4, dtype=torch.float32).reshape(2, 8, 2, 4),
        U=torch.zeros(2, 8, 2, 16, 4, dtype=torch.bfloat16),
        W=torch.zeros(2, 8, 2, 16, 4, dtype=torch.bfloat16),
        count=torch.full((2, 8, 2), 8, dtype=torch.int32),
        stale=torch.zeros(8, dtype=torch.int32),
        dense_of=torch.arange(8, dtype=torch.int32),
        dense_required=torch.ones(8, dtype=torch.int32),
        prefix_valid=torch.ones(8, dtype=torch.int32),
    )


class TestFactorTransaction(unittest.TestCase):
    def setUp(self):
        self.pool = pool()
        self.tx = Transaction(self.pool, 4, 4)
        self.slots = torch.tensor([2, 5])

    def candidate(self):
        # Deliberately distinguish every input/layer/head and rejected suffix.
        for step in range(4):
            for name, t in self.tx.working.items():
                t[:, :2].add_(1 if name == "count" else step+1)
            for layer in range(2):
                self.tx.record_step(layer, step, 2)

    def state(self):
        return {name: getattr(self.pool, name).clone() for name in
                (*Transaction.names, "stale", "dense_of", "dense_required", "prefix_valid")}

    def assert_state(self, expected):
        for name, value in expected.items():
            self.assertTrue(torch.equal(getattr(self.pool, name), value), name)

    def test_rollback_does_not_publish_candidates(self):
        before = self.state()
        ticket = self.tx.snapshot_commit(self.slots)
        self.candidate()
        self.assert_state(before)
        self.tx.rollback(ticket)
        self.assert_state(before)
        with self.assertRaises(RuntimeError):
            self.tx.commit(ticket, torch.tensor([0, 0]))

    def test_accept_prefix_and_tracking_with_padding(self):
        before = self.state()
        ticket = self.tx.snapshot_commit(self.slots)
        self.candidate()
        self.tx.commit(ticket, torch.tensor([0, 2]),
                       track_slots=torch.tensor([-1, 7]), track_steps=torch.tensor([-1, 1]))
        for name in Transaction.names:
            expected = before[name]
            expected[:, 2] = self.tx.checkpoints[name][:, 0, 0]
            expected[:, 5] = self.tx.checkpoints[name][:, 1, 2]
            expected[:, 7] = self.tx.checkpoints[name][:, 1, 1]
        for name, value in (("stale", 1), ("dense_of", -1), ("dense_required", 0), ("prefix_valid", 0)):
            before[name][[2, 5, 7]] = value
        self.assert_state(before)
        with self.assertRaises(RuntimeError):
            self.tx.commit(ticket, torch.tensor([0, 2]))

    def test_consecutive_zero_drafts_consumes_one_target_input(self):
        expected = self.pool.a[:, self.slots].clone()
        for _ in range(17):
            ticket = self.tx.snapshot_commit(self.slots)
            self.candidate()
            self.tx.commit(ticket, torch.tensor([0, 0]))
            expected += 1
            self.assertTrue(torch.equal(self.pool.a[:, self.slots], expected))

    def test_reused_slot_missing_state_and_invalid_tracking_rejected(self):
        ticket = self.tx.snapshot_commit(self.slots)
        before = self.state()
        with self.assertRaises(RuntimeError):
            self.tx.commit(ticket, torch.tensor([0, 0]))
        self.candidate()
        with self.assertRaises(RuntimeError):
            self.tx.commit(ticket, torch.tensor([0, 0]),
                           track_slots=torch.tensor([6, 7]), track_steps=torch.tensor([1, 1]))
        self.tx.invalidate_slots(self.slots[:1])
        with self.assertRaises(RuntimeError):
            self.tx.commit(ticket, torch.tensor([0, 0]))
        self.assert_state(before)
        self.tx.rollback(ticket)

    def test_only_one_live_ticket_and_unique_slots(self):
        with self.assertRaises(RuntimeError):
            self.tx.snapshot_commit(torch.tensor([2, 2]))
        ticket = self.tx.snapshot_commit(self.slots)
        with self.assertRaises(RuntimeError):
            self.tx.snapshot_commit(self.slots)
        self.tx.rollback(ticket)
        self.tx.snapshot_commit(torch.tensor([3]))
        self.assertEqual(self.tx.work_indices.tolist(), [0, -1, -1, -1])


if __name__ == "__main__":
    unittest.main()
