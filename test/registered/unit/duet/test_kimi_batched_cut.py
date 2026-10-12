"""The Kimi-Linear batched state cut equals the per-slot reference cut (shared B=1 Omega) up to fp32 rounding, fresh and warm;
rows of different layers in one call equal per-layer calls.  kimi_duet_math.project_state_batched is
sglang.srt.duet.state_factor.project_state_batched (checked on the source below), loaded here without SGLang's runtime.
Earlier path: python/sglang/srt/models/kimi_duet/tests/test_batched_cut.py (same cases and tolerances as unittest, except the
rank check's rtol 1e-6 -> 1e-5: on CPU fp32 the cut's 7th singular value is 1.2e-6 of the largest, rounding, not a component).

Run from this directory:  python -m unittest test_kimi_batched_cut
"""
import ast
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]   # test/registered/unit/duet -> repo root
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/sglang/srt/models"))

from lightning_duet._common import load as _load  # noqa: E402  (regular import inside the image, file load on a bare CPU box)

from glm_duet import _state_factor as state_factor
project_state, project_state_batched = state_factor.project_state, state_factor.project_state_batched


def rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-12))


class KimiBatchedCutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_kimi_math_serves_the_common_batched_cut(self):
        tree = ast.parse((ROOT / "test/registered/unit/duet/fixtures/seed_v35_kimi/kimi_duet_math.py").read_text())
        imports = {a.asname or a.name: f"{node.module}.{a.name}" for node in tree.body if isinstance(node, ast.ImportFrom)
                   for a in node.names}
        self.assertEqual(imports.get("_state"), "sglang.srt.duet.state_factor")
        sources = [ast.unparse(node.value) for node in tree.body if isinstance(node, ast.Assign)
                   and "project_state_batched" in ast.unparse(node.targets[0])]
        self.assertEqual(sources, ["_state.project_state_batched"])

    def test_rows_equal_the_per_slot_cut_fresh_and_warm(self):
        H, D, r, B = 4, 32, 6, 5
        direction = torch.randn(H, D)
        states = torch.randn(B, H, D, D) + 3.0 * torch.randn(B, H, D, 1) * direction[None, :, None, :]   # content + a sink-like part
        for explicit in (True, False):
            # fresh (no warm basis): per slot vs batched
            ref, bases_ref = [], []
            for i in range(B):
                p, b = project_state(states[i:i + 1], direction, r, None, explicit=explicit)
                ref.append(p); bases_ref.append(b)
            proj, bases = project_state_batched(states, direction, r, [None] * B, explicit=explicit)
            worst = max(rel(proj[i:i + 1], ref[i]) for i in range(B))
            worst_b = max(rel(bases[i], bases_ref[i]) for i in range(B))
            self.assertLess(worst, 1e-5, f"explicit={explicit} fresh state")
            self.assertLess(worst_b, 1e-4, f"explicit={explicit} fresh basis")
            # warm: update the states a little, use the previous bases (mixed: slot 2 fresh again)
            states2 = states + 0.05 * torch.randn_like(states)
            prevs = [bases_ref[i] for i in range(B)]
            prevs[2] = None
            ref2 = [project_state(states2[i:i + 1], direction, r, prevs[i], explicit=explicit)[0] for i in range(B)]
            proj2, _ = project_state_batched(states2, direction, r, prevs, explicit=explicit)
            worst2 = max(rel(proj2[i:i + 1], ref2[i]) for i in range(B))
            self.assertLess(worst2, 1e-5, f"explicit={explicit} warm (slot 2 fresh)")
            # the content part of the cut has rank <= r (fp32 rounding leaves singular values ~1e-6 of the largest)
            if explicit:
                n2 = (direction * direction).sum(-1)
                coeff = torch.einsum("bhkv,hv->bhk", proj, direction) / n2[None, :, None]
                content = proj - coeff[..., None] * direction[None, :, None, :]
                ranks = torch.linalg.matrix_rank(content.double(), rtol=1e-5)
                self.assertLessEqual(int(ranks.max()), r)

    def test_rows_of_different_layers_equal_per_layer_calls(self):
        H, D, r, B, L = 4, 32, 6, 5, 3
        directions = torch.randn(L, H, D)
        layer_states = torch.randn(L, B, H, D, D)
        rows = layer_states.reshape(L * B, H, D, D)
        row_dirs = directions[:, None].expand(L, B, H, D).reshape(L * B, H, D)
        prevs = [None] * (L * B)
        prevs[1] = project_state(rows[1:2], row_dirs[1], r, None, explicit=True)[1]               # one warm row
        multi, multi_bases = project_state_batched(rows, row_dirs, r, prevs, explicit=True)
        worst = 0.0
        for l in range(L):
            per_layer, per_bases = project_state_batched(layer_states[l], directions[l], r, prevs[l * B:(l + 1) * B], explicit=True)
            worst = max(worst, rel(multi[l * B:(l + 1) * B], per_layer))
            worst = max(worst, max(rel(multi_bases[l * B + i], per_bases[i]) for i in range(B)))
        self.assertLess(worst, 1e-5)


if __name__ == "__main__":
    unittest.main()
