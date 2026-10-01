"""Prove the common DUET modules against independently loaded reference Git objects."""
import sys
from pathlib import Path
import types
import unittest
from types import SimpleNamespace

import torch
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/sglang/srt/models"))
from lightning_duet._common import load

codec, state_factor = load("latent_codec"), load("state_factor")
ResidualCode, quantize = codec.ResidualCode, codec.quantize
project_state = state_factor.project_state

REF = "dd9c7bdbd9550a5d86781ebfa3d965d01fc1e78a"


def reference(name, injected=None):
    src = (Path(__file__).parent / "fixtures/kimi_reference" / f"{name}.py").read_text()
    # Supply the pinned format module without changing installed packages.
    src = src.replace("from twinstar.duet import latentfmt", "")
    module = types.ModuleType("reference_" + name)
    if injected: module.__dict__.update(injected)
    import sys
    sys.modules[module.__name__] = module
    exec(compile(src, f"{REF}/{name}.py", "exec"), module.__dict__)
    return module


class MathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fmt = reference("latentfmt")
        cls.latent = reference("latent", {"latentfmt": cls.fmt})
        cls.state = reference("state")
        torch.set_num_threads(1)

    def test_quantizers_match_reference_including_ties_and_zero(self):
        torch.manual_seed(7)
        x = torch.cat([torch.randn(8, 32), torch.zeros(1, 32), torch.linspace(-6, 6, 32)[None]])
        for fmt in ("fp32", "bf16", "fp8", "nvfp4"):
            self.assertTrue(torch.equal(quantize(x, fmt), self.fmt.quantize(x, fmt)), fmt)

    def test_code_side_channel_fp32_and_bf16(self):
        spec = dict(latent_rank=16, latent_spikes=4, latent_id_side=True, latent_z_format="nvfp4",
                    latent_value_format="bf16", latent_index_format="gap8")
        torch.manual_seed(23)
        port = ResidualCode(32, spec)
        ref = self.latent.ResidualCode(32, SimpleNamespace(**spec))
        for p in ref.parameters(): p.data.normal_(std=.1)
        ref.code.mu.normal_(std=.1)
        port.load_state_dict(ref.state_dict())
        for dtype in (torch.float32, torch.bfloat16):
            h, base = torch.randn(1, 19, 32).to(dtype), torch.randn(1, 19, 32).to(dtype)
            self.assertTrue(torch.equal(port(h, base), ref(h, base)))
            self.assertFalse(torch.equal(port(h, base), port(h, torch.zeros_like(base))))

    def test_prefix_and_three_warm_cuts_match_reference(self):
        torch.manual_seed(27)
        direction = torch.randn(2, 32)
        ref = self.state.StateFactor(1, 2, 32, "right", 8, True, 16)
        ref.sink_dir[0].copy_(direction)
        state = torch.randn(1, 2, 32, 32)
        previous = None
        for i in range(4):
            expected = ref(0, state, warm=i > 0)
            actual, previous = project_state(state, direction, 8, previous)
            self.assertTrue(torch.equal(actual, expected), f"cut {i}")
            state = expected + .01 * torch.randn_like(state)
        # A fresh request must restart the random subspace, not reuse warm data.
        actual, _ = project_state(state, direction, 8)
        self.assertTrue(torch.equal(actual, ref(0, state, warm=False)))

    def test_both_sink_sides_rectangular_cold_and_three_warm_cuts(self):
        # Non-square states catch a transpose-based left implementation, which
        # changes both random-subspace geometry and the saved right basis.
        for side, dim in (("right", 24), ("left", 32)):
            for dtype in (torch.float32, torch.bfloat16):
                for explicit in (False, True):
                    with self.subTest(side=side, dtype=dtype, explicit=explicit):
                        torch.manual_seed(51)
                        ref = self.state.StateFactor(1, 2, dim, side, 7, explicit, 13)
                        direction = torch.randn(2, dim)
                        ref.sink_dir[0].copy_(direction)
                        state = torch.randn(1, 2, 32, 24).to(dtype)
                        previous = None
                        for cut in range(4):
                            actual, previous = project_state(state, direction, 7, previous,
                                                             explicit=explicit, side=side)
                            expected = ref(0, state, warm=cut > 0)
                            self.assertTrue(torch.equal(actual, expected), f"cut {cut}")
                            self.assertTrue(torch.equal(previous, ref._warm[0]), f"basis {cut}")
                            self.assertEqual(previous.shape, (1, 2, 24, 7))
                            state = (expected + .01 * torch.randn_like(expected)).to(dtype)
                        actual, previous = project_state(state, direction, 7, side=side, explicit=explicit)
                        self.assertTrue(torch.equal(actual, ref(0, state, warm=False)))
                        self.assertTrue(torch.equal(previous, ref._warm[0]))

    def test_implicit_sink_and_disabled_code_state_match_reference(self):
        torch.manual_seed(42)
        state = torch.randn(1, 2, 32, 32)
        direction = torch.randn(2, 32)
        ref = self.state.StateFactor(1, 2, 32, "right", 7, False, 11)
        previous = None
        for i in range(3):
            actual, previous = project_state(state, direction, 7, previous, explicit=False)
            expected = ref(0, state, warm=i > 0)
            self.assertTrue(torch.equal(actual, expected))
            state = expected + .01 * torch.randn_like(state)
        actual, previous = project_state(state, direction, 0)
        self.assertIs(actual, state)
        self.assertIsNone(previous)
        h = torch.randn(1, 3, 32)
        code = ResidualCode(32, {"latent_rank": 0})
        self.assertIs(code(h, None), h)


if __name__ == "__main__":
    unittest.main()
