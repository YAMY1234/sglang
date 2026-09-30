"""CPU checks of sglang.srt.duet.state_pool.SlotSidecar: cadence, warm basis, slot lifecycle.

Run from this directory:  python -m unittest test_duet_state_pool
"""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load as _load  # noqa: E402

state_pool = _load("state_pool")


class CadenceTests(unittest.TestCase):
    def test_boundary_counts_as_step_one_and_prune_every_w(self):
        sc = state_pool.SlotSidecar(4, [10, 20], heads=2, warm_dim=5, rank=3, every=4)
        sc.note_prefill(10, [0, 2])
        self.assertEqual(sc.take_pending_prefix(10, [0, 1, 2]).tolist(), [0, 2])
        self.assertEqual(sc.take_pending_prefix(10, [0, 1, 2]).tolist(), [])       # cleared once taken
        due = [sc.note_step(10, [0, 2]).tolist() for _ in range(8)]
        self.assertEqual(due, [[], [], [], [0, 2], [], [], [], [0, 2]])              # steps 4 and 8 prune
        self.assertEqual(sc.count[0, [0, 2]].tolist(), [8, 8])
        self.assertEqual(sc.count[1, 0].item(), 0)                                    # other layer untouched

    def test_zero_cadence_or_rank_never_prunes_at_decode(self):
        sc = state_pool.SlotSidecar(2, [0], heads=1, warm_dim=4, rank=2, every=0)
        sc.note_prefill(0, [0])
        self.assertEqual(sc.take_pending_prefix(0, [0]).tolist(), [0])               # prompt-final prune still owed
        self.assertTrue(all(sc.note_step(0, [0]).numel() == 0 for _ in range(10)))
        exact = state_pool.SlotSidecar(2, [0], heads=1, warm_dim=4, rank=0, every=4)
        self.assertTrue(all(exact.note_step(0, [0]).numel() == 0 for _ in range(8)))
        self.assertIsNone(exact.warm_basis(0, [0]))
        exact.store_warm(0, [0], None)


class WarmTests(unittest.TestCase):
    def test_warm_basis_round_trip_and_shape_check(self):
        sc = state_pool.SlotSidecar(3, [7], heads=2, warm_dim=5, rank=3, every=2)
        self.assertIsNone(sc.warm_basis(7, [0]))
        basis = torch.randn(2, 2, 5, 3)
        sc.store_warm(7, [0, 2], basis)
        torch.testing.assert_close(sc.warm_basis(7, [0, 2]), basis, rtol=0, atol=0)
        self.assertIsNone(sc.warm_basis(7, [0, 1]))                                   # slot 1 cold -> whole batch cold
        with self.assertRaises(ValueError):
            sc.store_warm(7, [0], torch.randn(1, 2, 5, 4))
        sc.note_prefill(7, [0])                                                       # a new prompt resets the warm start
        self.assertIsNone(sc.warm_basis(7, [0]))


class LifecycleTests(unittest.TestCase):
    def test_reset_copy_and_host_round_trip(self):
        sc = state_pool.SlotSidecar(4, [1, 2], heads=1, warm_dim=3, rank=2, every=3)
        sc.note_prefill(1, [0]); sc.note_step(1, [0]); sc.store_warm(1, [0], torch.ones(1, 1, 3, 2))
        sc.note_prefill(2, [0])
        sc.copy_slots([0], [3])
        self.assertEqual(sc.count[0, 3].item(), 1)
        self.assertTrue(sc.pending_prefix[1, 3].item())
        self.assertTrue(sc.warm_valid[0, 3].item())
        host = sc.get_cpu_slots([3])
        sc.reset_slots([0, 3])
        self.assertEqual(sc.count[0, [0, 3]].tolist(), [0, 0])
        self.assertFalse(sc.warm_valid[0, 0].item())
        sc.load_cpu_slots(host, [1])
        self.assertEqual(sc.count[0, 1].item(), 1)
        torch.testing.assert_close(sc.warm[0, 1], torch.ones(1, 3, 2), rtol=0, atol=0)
        self.assertEqual(len(list(sc.iter_transfer_state_entries())), 2 * len(sc.fields))
        self.assertTrue(all(axis is None for _, _, axis, _ in sc.iter_transfer_state_entries()))   # MambaPool slice_axis
        self.assertGreater(sc.nbytes(), 0)
        with self.assertRaises(IndexError):
            sc.note_step(1, [4])
        with self.assertRaises(ValueError):
            sc.copy_slots([0, 1], [2])


if __name__ == "__main__":
    unittest.main()
