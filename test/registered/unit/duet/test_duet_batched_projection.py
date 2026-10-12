"""CPU checks of sglang.srt.duet.state_factor.project_state_batched: every row of a batched cut equals the per-slot
reference cut (project_state, B=1, the shared Omega draw) up to fp32 rounding, for the right sink (KDA / GDN layout) and
the left sink (Mamba-2), fresh and warm; the left cut is also the stored form of factorize_left (the Lightning
functional pool).

Run from this directory:  python -m unittest test_duet_batched_projection
"""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]   # test/registered/unit/duet -> repo root
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/sglang/srt/models"))

from lightning_duet._common import load as _load  # noqa: E402  (regular import inside the image, file load on a bare CPU box)

from glm_duet import _state_factor as state_factor


def rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-12))


def states_with_sink(batch, heads, p, n, direction, side):
    """Random content plus a strong sink-like part along `direction` (H, P) on the left or (H, N) on the right."""
    content = torch.randn(batch, heads, p, n)
    if side == "left":
        return content + 3.0 * direction[None, :, :, None] * torch.randn(batch, heads, 1, n)
    return content + 3.0 * torch.randn(batch, heads, p, 1) * direction[None, :, None, :]


class BatchedProjectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def check_rows(self, side, p, n, rank, explicit):
        heads, batch = 4, 5
        direction = torch.randn(heads, p if side == "left" else n)
        states = states_with_sink(batch, heads, p, n, direction, side)
        fresh, bases = state_factor.project_state_batched(states, direction, rank, [None] * batch, explicit=explicit, side=side)
        for b in range(batch):
            ref, basis = state_factor.project_state(states[b:b + 1], direction, rank, None, explicit=explicit, side=side)
            self.assertLess(rel(fresh[b:b + 1], ref), 1e-5)
            self.assertLess(rel(bases[b], basis), 1e-4)
        moved = states + 0.05 * torch.randn_like(states)
        warm, _ = state_factor.project_state_batched(moved, direction, rank, bases, explicit=explicit, side=side)
        for b in range(batch):
            ref, _ = state_factor.project_state(moved[b:b + 1], direction, rank, bases[b], explicit=explicit, side=side)
            self.assertLess(rel(warm[b:b + 1], ref), 1e-5)

    def test_right_sink_rows_equal_the_per_slot_cut(self):
        for explicit in (True, False):
            self.check_rows("right", 32, 32, 6, explicit)

    def test_left_sink_rows_equal_the_per_slot_cut(self):
        for explicit in (True, False):
            self.check_rows("left", 16, 32, 6, explicit)    # Mamba-2 geometry in miniature: P < N

    def test_left_sink_is_the_factorized_stored_form(self):
        heads, batch, p, n, rank = 4, 3, 16, 32, 6
        direction = torch.randn(heads, p)
        states = states_with_sink(batch, heads, p, n, direction, "left")
        projected, bases = state_factor.project_state_batched(states, direction, rank, [None] * batch, side="left")
        for b in range(batch):      # factorize_left draws Omega with its batch shape: one slot per call, as the pool calls it
            coeff, left, right, warm = state_factor.factorize_left(states[b:b + 1], direction, rank)
            stored = direction[None, :, :, None] * coeff[:, :, None, :] + left @ right
            self.assertLess(rel(projected[b:b + 1], stored), 1e-5)
            self.assertLess(rel(bases[b], warm), 1e-4)

    def test_rows_of_different_layers_take_their_own_direction(self):
        heads, batch, p, n, rank = 4, 4, 16, 32, 6
        directions = torch.randn(batch, heads, p)
        states = torch.stack([states_with_sink(1, heads, p, n, directions[b], "left")[0] for b in range(batch)])
        projected, _ = state_factor.project_state_batched(states, directions, rank, [None] * batch, side="left")
        for b in range(batch):
            ref, _ = state_factor.project_state(states[b:b + 1], directions[b], rank, None, side="left")
            self.assertLess(rel(projected[b:b + 1], ref), 1e-5)

    def test_warm_bases_hold_only_their_own_storage(self):
        heads, batch, p, n, rank = 4, 6, 16, 32, 6
        direction = torch.randn(heads, p)
        states = states_with_sink(batch, heads, p, n, direction, "left")
        _, bases = state_factor.project_state_batched(states, direction, rank, [None] * batch, side="left")
        for basis in bases:     # a view of the chunk would hold batch x the bytes of one basis
            self.assertEqual(basis.untyped_storage().nbytes(), basis.numel() * basis.element_size())

    def test_unknown_side_is_refused(self):
        with self.assertRaises(ValueError):
            state_factor.project_state_batched(torch.zeros(1, 1, 4, 4), torch.ones(1, 4), 2, [None], side="middle")


if __name__ == "__main__":
    unittest.main()
