"""CPU checks of sglang.srt.duet.{adapters,numerics} (docs/167 P1): release spec -> adapter selection, the
registry hook's transition behaviour, and the numerics profile precedence / refusal rules.

Run from this directory:  python -m unittest test_duet_adapters_numerics
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/sglang/srt/models"))
from lightning_duet._common import load as _load  # noqa: E402

adapters, numerics, duet_spec = (_load(n) for n in ("adapters", "numerics", "spec"))

SPEC = dict(model="lightning", prefill_depth=33, latent_rank=2048, latent_spikes=128, latent_id_side=True,
            latent_z_format="nvfp4", latent_value_format="bf16", latent_index_format="gap8",
            state_rank=16, state_every=16, state_sink="explicit", latent_init="", state_init="", name="x")


def release_dir(root, model, **over):
    d = Path(root) / model
    d.mkdir()
    (d / "spec.json").write_text(json.dumps({**SPEC, "model": model, **over}))
    return d


class FakeRegistry:
    def __init__(self):
        self.calls = []

    def register(self, package, overwrite=False):
        self.calls.append((package, overwrite))


class AdapterSelection(unittest.TestCase):
    def test_table_covers_every_spec_model_and_names_base_architectures(self):
        self.assertEqual(set(adapters.ADAPTERS), set(duet_spec.MODELS))
        self.assertEqual(adapters.ADAPTERS["lightning"].architecture, "NemotronHForCausalLM")
        self.assertEqual(adapters.ADAPTERS["kimi-linear"].architecture, "KimiLinearForCausalLM")
        self.assertEqual(adapters.ADAPTERS["flash-next"].architecture, "Qwen4ExpForConditionalGeneration")

    def test_select_reads_spec_model_from_a_directory_and_rejects_unknown_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            for model in duet_spec.MODELS:
                d = release_dir(tmp, model)
                directory, spec, adapter = adapters.select(str(d))
                self.assertEqual((Path(directory), spec["model"], adapter.model), (d, model, model))
            bad = release_dir(tmp, "bad", deep_form="x")
            with self.assertRaises(ValueError):
                adapters.select(str(bad))

    def test_release_value_precedence_cli_over_environment(self):
        self.assertIsNone(adapters.release_value(SimpleNamespace(duet_release=None), {}))
        self.assertEqual(adapters.release_value(SimpleNamespace(duet_release="/a"), {"SGLANG_DUET_DIR": "/b"}), "/a")
        self.assertEqual(adapters.release_value(None, {"SGLANG_DUET_DIR": "/b"}), "/b")

    def test_register_uses_the_fork_package_when_present_and_only_warns_otherwise(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = release_dir(tmp, "kimi-linear")
            original = adapters.package_available
            try:
                adapters.package_available = lambda adapter: True
                registry = FakeRegistry()
                adapter = adapters.register(registry, str(d))
                self.assertEqual(registry.calls, [("sglang.srt.models.kimi_linear_duet", True)])
                self.assertEqual(adapter.model, "kimi-linear")
                adapters.package_available = lambda adapter: False
                registry = FakeRegistry()
                self.assertIsNone(adapters.register(registry, str(d)))
                self.assertEqual(registry.calls, [])
                with self.assertRaises(ImportError):
                    adapters.register(registry, str(d), strict=True)
            finally:
                adapters.package_available = original

    def test_register_from_environment_never_raises(self):
        registry = FakeRegistry()
        self.assertIsNone(adapters.register_from_environment(registry, {}))
        self.assertIsNone(adapters.register_from_environment(registry, {"SGLANG_DUET_DIR": "/nonexistent/release"}))
        self.assertEqual(registry.calls, [])

    def test_describe_is_cheap_and_reports_errors_instead_of_raising(self):
        self.assertEqual(adapters.describe(None, {}), {"release": None, "enabled": False})
        report = adapters.describe(None, {"SGLANG_DUET_DIR": "/nonexistent/release"})
        self.assertTrue(report["enabled"] and "error" in report)
        with tempfile.TemporaryDirectory() as tmp:
            d = release_dir(tmp, "flash-next")
            report = adapters.describe(SimpleNamespace(duet_release=str(d)), {})
            self.assertEqual((report["model"], report["adapter"]), ("flash-next", "sglang.srt.models.flash_next_duet"))
            self.assertIn("adapter_in_tree", report)


class NumericsProfiles(unittest.TestCase):
    def test_profile_precedence_and_validation(self):
        self.assertEqual(numerics.profile_name(None, {}), "production")
        self.assertEqual(numerics.profile_name(None, {"SGLANG_DUET_NUMERICS": "reference"}), "reference")
        self.assertEqual(numerics.profile_name(SimpleNamespace(duet_numerics="reference"), {"SGLANG_DUET_NUMERICS": "production"}), "reference")
        with self.assertRaises(ValueError):
            numerics.profile_name(SimpleNamespace(duet_numerics="fast"), {})

    def test_tables_cover_both_profiles(self):
        for table in (numerics.SWITCHES, numerics.CONTROLS):
            for name, per_profile in table.items():
                self.assertEqual(set(per_profile), set(numerics.PROFILES), name)
        self.assertEqual(numerics.defaults("reference")["duet_emitter_precision"], "fp32")
        self.assertEqual(numerics.defaults("production")["duet_prefix_state"], "factored")
        self.assertFalse(numerics.controls("reference")["cuda_graph"])

    def test_apply_defaults_fills_only_unset_fields_and_respects_environment(self):
        args = SimpleNamespace(duet_numerics=None, duet_emitter_precision=None, duet_prefix_state="exact")
        self.assertEqual(numerics.apply_defaults(args, {}), "production")
        self.assertEqual((args.duet_emitter_precision, args.duet_prefix_state), ("bf16", "exact"))
        args = SimpleNamespace(duet_numerics="reference", duet_emitter_precision=None, duet_prefix_state=None)
        numerics.apply_defaults(args, {"SGLANG_DUET_PREFIX_STATE": "factored"})
        self.assertEqual(args.duet_emitter_precision, "fp32")
        self.assertIsNone(args.duet_prefix_state, "an environment value is left to the options layer")
        bare = SimpleNamespace(duet_numerics=None)
        self.assertEqual(numerics.apply_defaults(bare, {}), "production")

    def test_require_profile_refuses_unvalidated_production_and_names_the_fix(self):
        self.assertEqual(numerics.require_profile("flash-next", None, {}, production_supported=True), "production")
        self.assertEqual(numerics.require_profile("lightning", SimpleNamespace(duet_numerics="reference"), {}, production_supported=False), "reference")
        with self.assertRaises(ValueError) as ctx:
            numerics.require_profile("lightning", None, {}, production_supported=False)
        self.assertIn("--duet-numerics reference", str(ctx.exception))

    def test_describe(self):
        d = numerics.describe(SimpleNamespace(duet_numerics="reference"), {})
        self.assertEqual(d["profile"], "reference")
        self.assertEqual(d["controls"]["mamba_state_dtype"], "float32")


if __name__ == "__main__":
    unittest.main()
