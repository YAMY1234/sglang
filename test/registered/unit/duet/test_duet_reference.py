"""CPU byte equality against minma/0913; set DUET_REFERENCE_REPO to its checkout.

Reads pinned Git objects, not working-tree files. No reference download or GPU.
The public modules are loaded directly to avoid importing serving dependencies.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[4]
REF = "dd9c7bdbd9550a5d86781ebfa3d965d01fc1e78a"


def common(name):
    key = f"sglang.srt.duet.{name}"
    if key not in sys.modules:
        path = ROOT / "python/sglang/srt/duet" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


codec, state_factor = common("latent_codec"), common("state_factor")


def reference(name):
    repo = os.environ.get("DUET_REFERENCE_REPO")
    if not repo:
        raise unittest.SkipTest("set DUET_REFERENCE_REPO to a repo containing the pinned minma/0913 commit")
    source = subprocess.check_output(
        ["git", "-C", repo, "show", f"{REF}:twinstar/duet/{name}.py"], text=True,
    )
    module = ModuleType(f"duet_reference_{name}")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{REF}/twinstar/duet/{name}.py", "exec"), module.__dict__)
    return module


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.fmt = reference("latentfmt")
        cls.state = reference("state")
        # Supply only the pinned import dependency; execute latent.py unchanged.
        package = ModuleType("twinstar.duet")
        package.latentfmt = cls.fmt
        with patch.dict(sys.modules, {"twinstar.duet": package,
                                      "twinstar.duet.latentfmt": cls.fmt}):
            cls.latent = reference("latent")

    def assert_bits(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8),
                                    expected.contiguous().view(torch.uint8)))

    def test_quantizers_match_reference_bytes(self):
        torch.manual_seed(7)
        x = torch.cat([torch.randn(8, 32), torch.zeros(1, 32), torch.linspace(-6, 6, 32)[None]])
        for fmt in ("fp32", "bf16", "fp8", "nvfp4"):
            with self.subTest(fmt=fmt):
                self.assert_bits(codec.quantize(x, fmt), self.fmt.quantize(x, fmt))

    def test_residual_code_matches_reference_bytes(self):
        for side_input in (False, True):
            spec = dict(latent_rank=16, latent_spikes=4, latent_id_side=side_input,
                        latent_z_format="nvfp4", latent_value_format="bf16", latent_index_format="gap8")
            torch.manual_seed(23)
            served = codec.ResidualCode(32, spec)
            ref = self.latent.ResidualCode(32, SimpleNamespace(**spec))
            for p in ref.parameters():
                p.data.normal_(std=.1)
            ref.code.mu.normal_(std=.1)
            served.load_state_dict(ref.state_dict())
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(side_input=side_input, dtype=dtype):
                    h, base = torch.randn(1, 19, 32).to(dtype), torch.randn(1, 19, 32).to(dtype)
                    self.assert_bits(served(h, base), ref(h, base))

    def test_orthonormalize_matches_reference_bytes(self):
        torch.manual_seed(27)
        for y in (torch.randn(2, 3, 24, 7), torch.zeros(1, 2, 24, 7)):
            self.assert_bits(state_factor._orthonormalize(y), self.state._orthonormalize(y))
        self.assertIs(state_factor._orthonormalize, state_factor.orthonormalize)

    def test_truncate_rank_cold_and_three_warm_bases_match_reference_bytes(self):
        torch.manual_seed(27)
        state = torch.randn(1, 2, 32, 24)
        actual_prev = ref_prev = None
        for _ in range(4):
            actual, actual_prev = state_factor.truncate_rank(state, 7, actual_prev)
            expected, ref_prev = self.state.truncate_rank(state, 7, ref_prev)
            self.assert_bits(actual, expected)
            self.assert_bits(actual_prev, ref_prev)
            state = expected + .01 * torch.randn_like(expected)

    def test_project_both_sides_cold_and_three_warm_cuts_match_reference_bytes(self):
        for side, side_dim in (("right", 24), ("left", 32)):
            for dtype in (torch.float32, torch.bfloat16):
                for explicit in (False, True):
                    with self.subTest(side=side, dtype=dtype, explicit=explicit):
                        torch.manual_seed(51)
                        direction = torch.randn(2, side_dim)
                        ref = self.state.StateFactor(1, 2, side_dim, side, 7, explicit, 13)
                        ref.sink_dir[0].copy_(direction)
                        state = torch.randn(1, 2, 32, 24).to(dtype)
                        previous = None
                        for cut in range(4):
                            actual, previous = state_factor.project_state(
                                state, direction, 7, previous, side=side, explicit=explicit,
                            )
                            expected = ref(0, state, warm=cut > 0)
                            self.assert_bits(actual, expected)
                            self.assert_bits(previous, ref._warm[0])
                            self.assertEqual(previous.shape, (1, 2, 24, 7))
                            state = (expected + .01 * torch.randn_like(expected)).to(dtype)
                        actual, previous = state_factor.project_state(
                            state, direction, 7, side=side, explicit=explicit,
                        )
                        self.assert_bits(actual, ref(0, state, warm=False))
                        self.assert_bits(previous, ref._warm[0])

    def test_disabled_full_rank_and_zero_direction_match_reference(self):
        torch.manual_seed(63)
        state = torch.randn(1, 2, 16, 24)
        for side, dim in (("right", 24), ("left", 16)):
            for rank in (0, 7, 16):
                with self.subTest(side=side, rank=rank):
                    direction = torch.zeros(2, dim)
                    actual, warm = state_factor.project_state(state, direction, rank, side=side)
                    ref = self.state.StateFactor(1, 2, dim, side, rank, True, 5)
                    self.assert_bits(actual, ref(0, state))
                    if rank in (0, 16):
                        self.assertIsNone(warm)
        with self.assertRaisesRegex(ValueError, "sink side"):
            state_factor.project_state(state, None, 0, side="typo")

    def test_left_project_matches_existing_factor_storage(self):
        torch.manual_seed(73)
        state, direction = torch.randn(1, 2, 32, 24), torch.randn(2, 32)
        dense_prev = factored_prev = None
        for _ in range(4):
            dense, dense_prev = state_factor.project_state(state, direction, 7, dense_prev, side="left")
            coeff, left, right, factored_prev = state_factor.factorize_left(state, direction, 7, factored_prev)
            self.assert_bits(dense, direction[None, :, :, None] * coeff[:, :, None, :] + left @ right)
            self.assert_bits(dense_prev, factored_prev)
            state = dense + .01 * torch.randn_like(dense)


if __name__ == "__main__":
    unittest.main()
