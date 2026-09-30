"""CPU checks of the model-independent DUET layer (sglang.srt.duet) and of the line shims that import from it.

Run from this directory:  python -m unittest test_duet_common
"""
import argparse
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]   # test/registered/unit/duet -> repo root
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import load as _load  # noqa: E402  (regular import inside the image, file load on a bare CPU box)

latent_codec, options, duet_spec, state_factor = (_load(n) for n in ("latent_codec", "options", "spec", "state_factor"))

SPEC = dict(model="lightning", prefill_depth=33, latent_rank=2048, latent_spikes=128, latent_id_side=True,
            latent_z_format="nvfp4", latent_value_format="bf16", latent_index_format="gap8",
            state_rank=16, state_every=16, state_sink="explicit", latent_init="", state_init="", name="x")


class SpecTests(unittest.TestCase):
    def test_accepts_release_and_rejects_unknown_or_wrong_model(self):
        duet_spec.validate_spec(SPEC)
        duet_spec.validate_spec({k: v for k, v in SPEC.items() if k not in ("name", "latent_init", "state_init")})
        with self.assertRaises(ValueError):
            duet_spec.validate_spec({**SPEC, "state_evry": 8})
        with self.assertRaises(ValueError):
            duet_spec.validate_spec(SPEC, model="kimi-linear")
        with self.assertRaises(ValueError):
            duet_spec.validate_spec({**SPEC, "latent_rank": 2050})   # nvfp4 needs a multiple of 16
        with self.assertRaises(ValueError):
            duet_spec.validate_spec({**SPEC, "latent_init": "stats.pt"})
        self.assertIn("k=33", duet_spec.describe(SPEC))


class OptionsTests(unittest.TestCase):
    def test_precedence_and_zero_semantics(self):
        o = options.DuetOptions.resolve(SPEC, environ={})
        self.assertEqual((o.prefill_layer_trim, o.prefill_saving_policy, o.decode_ssm_r, o.decode_ssm_w), (True, "kv-and-ssm", 16, 16))
        parser = argparse.ArgumentParser(); options.add_arguments(parser)
        args = parser.parse_args(["--no-prefill-layer-trim", "--decode-ssm-r", "0", "--decode-ssm-w", "0"])
        o = options.DuetOptions.resolve(SPEC, args, {"SGLANG_DUET_DECODE_SSM_R": "6"})
        self.assertEqual((o.prefill_layer_trim, o.decode_ssm_r, o.decode_ssm_w), (False, 0, 0))   # CLI beats env; 0 allowed (#002-4)
        self.assertEqual(o.effective_spec(SPEC)["state_rank"], 0)
        self.assertEqual(o.state_dtype_environment(), {})
        with self.assertRaises(ValueError):
            options.DuetOptions.resolve(SPEC, environ={"SGLANG_DUET_DECODE_SSM_W": "-1"})
        with self.assertRaises(ValueError):
            options.DuetOptions.resolve(SPEC, environ={"SGLANG_DUET_DECODE_SSM_R": "200"}, state_dim=128)
        for policy in ("latent-only", "latent-and-kv", "latent-and-ssm"):
            with self.assertRaises(NotImplementedError):
                options.DuetOptions.resolve(SPEC, environ={"SGLANG_DUET_PREFILL_SAVING_POLICY": policy})
        self.assertEqual(options.resolve_options(SPEC, state_dim=128).decode_ssm_r, 16)


class StateFactorTests(unittest.TestCase):
    def test_warm_truncation_recovers_a_low_rank_state(self):
        torch.manual_seed(0)
        u = torch.randn(2, 3, 32, 4); v = torch.randn(2, 3, 4, 24)
        state = u @ v                                            # rank 4 <= r: the r+8 subspace spans the range
        out, warm = state_factor.truncate_rank(state, 8)
        self.assertEqual(warm.shape, (2, 3, 24, 8))
        torch.testing.assert_close(out, state, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(state_factor.truncate_rank_exact(state, 8), state, rtol=1e-4, atol=1e-4)

    def test_explicit_right_sink_is_preserved(self):
        torch.manual_seed(1)
        state = torch.randn(1, 2, 16, 12)
        direction = torch.randn(2, 12)
        out, _ = state_factor.project_state(state, direction, 4)
        n2 = (direction * direction).sum(-1)
        a_in = torch.einsum("bhkv,hv->bhk", state, direction) / n2[None, :, None]
        a_out = torch.einsum("bhkv,hv->bhk", out, direction) / n2[None, :, None]
        torch.testing.assert_close(a_out, a_in, rtol=1e-4, atol=1e-4)

    def test_left_factorization_matches_its_stored_form(self):
        torch.manual_seed(2)
        state = torch.randn(1, 2, 8, 12)
        direction = torch.randn(2, 8)
        coeff, left, right, warm = state_factor.factorize_left(state, direction, 3)
        stored = direction[None, :, :, None] * coeff[:, :, None, :] + left @ right
        self.assertEqual(stored.shape, state.shape)
        self.assertEqual(warm.shape, (1, 2, 12, 3))
        # the sink coefficient of the stored form equals the input's
        n2 = direction.square().sum(-1).clamp_min(1e-12)
        torch.testing.assert_close(torch.einsum("bhpn,hp->bhn", stored, direction) / n2[None, :, None], coeff, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
