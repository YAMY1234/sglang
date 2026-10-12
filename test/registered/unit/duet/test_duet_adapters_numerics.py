"""CPU checks of sglang.srt.duet.{adapters,numerics}: release spec -> adapter selection, the registry hook's
refusal to serve a requested release with the base model, and the numerics profile precedence / refusal rules.

Run from this directory:  python -m unittest test_duet_adapters_numerics
"""
import json
import os
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
        self.assertEqual(set(adapters.ADAPTERS), set(duet_spec.MODELS) | {"glm5-next"})
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

    def test_register_uses_the_fork_package_when_present_and_refuses_otherwise(self):
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
                with self.assertRaises(adapters.UnservedRelease):
                    adapters.register(registry, str(d))
                self.assertEqual(registry.calls, [])
            finally:
                adapters.package_available = original

    def test_register_from_environment_without_a_release_is_a_noop(self):
        registry = FakeRegistry()
        self.assertIsNone(adapters.register_from_environment(registry, {}))
        self.assertEqual(registry.calls, [])

    def test_unserved_release_refuses_to_fall_back_to_the_base_model(self):
        """A requested release is never answered by the stock class: an absent package or a release that does not resolve stops registration.
        Available in-tree adapters take precedence over external packages."""
        with tempfile.TemporaryDirectory() as tmp:
            d = release_dir(tmp, "kimi-linear")
            original = adapters.package_available
            try:
                adapters.package_available = lambda adapter: False
                for env in ({"SGLANG_DUET_DIR": str(d)},
                            {"SGLANG_DUET_DIR": str(d), "SGLANG_EXTERNAL_MODEL_PACKAGE": "twinstar_sgl"}):
                    with self.assertRaises(adapters.UnservedRelease):
                        adapters.register_from_environment(FakeRegistry(), env)
                adapters.package_available = lambda adapter: True
                fn = release_dir(tmp, "flash-next")
                registry = FakeRegistry()
                # The unified branch retains the in-tree performance adapter;
                # an external package must not override its canonical release route.
                selected = adapters.register_from_environment(
                    registry, {"SGLANG_DUET_DIR": str(fn),
                               "SGLANG_EXTERNAL_MODEL_PACKAGE": "twinstar_sgl"})
                self.assertEqual(selected.package, "sglang.srt.models.flash_next_duet")
                self.assertEqual(registry.calls, [("sglang.srt.models.flash_next_duet", True)])
            finally:
                adapters.package_available = original
            for env in ({"SGLANG_DUET_DIR": os.path.join(tmp, "missing")},
                        {"SGLANG_DUET_DIR": os.path.join(tmp, "missing"), "SGLANG_EXTERNAL_MODEL_PACKAGE": "twinstar_sgl"}):
                with self.assertRaises(adapters.UnservedRelease):
                    adapters.register_from_environment(FakeRegistry(), env)

    def test_describe_is_cheap_and_reports_errors_instead_of_raising(self):
        self.assertEqual(adapters.describe(None, {}), {"release": None, "enabled": False})
        report = adapters.describe(None, {"SGLANG_DUET_DIR": "/nonexistent/release"})
        self.assertTrue(report["enabled"] and "error" in report)
        with tempfile.TemporaryDirectory() as tmp:
            d = release_dir(tmp, "flash-next")
            report = adapters.describe(SimpleNamespace(duet_release=str(d)), {})
            self.assertEqual((report["model"], report["adapter"]), ("flash-next", "sglang.srt.models.flash_next_duet"))
            self.assertIn("adapter_in_tree", report)


class AdapterExtensions(unittest.TestCase):
    def test_default_hf_root_precedence(self):
        self.assertEqual(adapters.default_hf_root({"HF_HOME": "/h"}), "/h")
        self.assertEqual(adapters.default_hf_root({"HF_HUB_CACHE": "/c/huggingface/hub"}), "/c/huggingface")
        self.assertTrue(adapters.default_hf_root({"XDG_CACHE_HOME": "/x"}).endswith("/x/huggingface"))

    def test_resolution_hook_is_a_noop_without_release_or_adapter_and_dispatches_when_present(self):
        self.assertIsNone(adapters.run_resolution_hook(SimpleNamespace(duet_release=None), {}))
        with tempfile.TemporaryDirectory() as tmp:
            d = release_dir(tmp, "flash-next")
            calls = []
            original = adapters.adapter_extension
            try:
                adapters.adapter_extension = lambda adapter, name: None
                self.assertIsNone(adapters.run_resolution_hook(SimpleNamespace(duet_release=str(d)), {}))
                adapters.adapter_extension = lambda adapter, name: (lambda sa, spec: calls.append((name, spec["model"]))) if name == "resolve_server_numerics" else None
                got = adapters.run_resolution_hook(SimpleNamespace(duet_release=str(d)), {})
                self.assertEqual((got.model, calls), ("flash-next", [("resolve_server_numerics", "flash-next")]))
            finally:
                adapters.adapter_extension = original

    def test_only_v3_qualified_adapters_enable_production(self):
        self.assertEqual(
            {name for name, adapter in adapters.ADAPTERS.items() if adapter.production_profile},
            {"kimi-linear", "flash-next"},
        )
        self.assertTrue(all(a.production_notes for a in adapters.ADAPTERS.values()))


class LaunchDuetChecks(unittest.TestCase):
    """benchmark/duet_eval/launch_duet.py: the two arms differ only in the DUET enablement, and a DUET arm that does
    not serve the release's cut and component file is stopped."""

    @staticmethod
    def launcher():
        import importlib.util
        spec = importlib.util.spec_from_file_location("launch_duet", ROOT / "benchmark/duet_eval/launch_duet.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_extra_accepts_only_the_duet_switches(self):
        ld = self.launcher()
        self.assertEqual(ld.parse_extra("--no-prefill-layer-trim --decode-ssm-r 0 --decode-ssm-w=4"),
                         {"prefill_layer_trim": False, "decode_ssm_r": 0, "decode_ssm_w": 4})
        for extra in ("--mem-fraction-static 0.8", "--decode-ssm-r", "--decode-ssm-r -1", "--prefill-layer-trim"):
            with self.assertRaises(SystemExit):
                ld.parse_extra(extra)

    def test_arm_check_reads_the_servers_cut_line_not_the_launchers_print(self):
        ld = self.launcher()
        sha = "c" * 64
        ident = {"name": "x", "sha256": sha, "spec": {"prefill_depth": 19, "state_rank": 32, "state_every": 32}}
        markers = "TwinStar Kimi Linear: P layers 0..18\nHF DUET loaded: x tensors=1 sha256=%s\n" % sha
        served = markers + "DUET served cut: k=19 trim=True r=32 W=32 sha256=%s\n" % sha
        self.assertTrue(ld.check_arm("kimi", "duet", served, ident, {})[0])
        launcher_print = markers + ('{"duet_runtime_options": {"SGLANG_DUET_PREFILL_LAYER_TRIM": "1", '
                                    '"SGLANG_DUET_PREFILL_SAVING_POLICY": "kv-and-ssm", "SGLANG_DUET_DECODE_SSM_R": "32", '
                                    '"SGLANG_DUET_DECODE_SSM_W": "32"}}\n')
        self.assertFalse(ld.check_arm("kimi", "duet", launcher_print, ident, {})[0])
        self.assertFalse(ld.check_arm("kimi", "duet", served.replace("k=19", "k=18"), ident, {})[0])
        self.assertFalse(ld.check_arm("kimi", "duet", served, ident, {"decode_ssm_r": 0})[0])
        other = served.replace("W=32 sha256=" + sha, "W=32 sha256=" + "d" * 64)
        self.assertFalse(ld.check_arm("kimi", "duet", other, ident, {})[0])
        self.assertFalse(ld.check_arm("kimi", "stock", served, ident, {})[0])
        self.assertTrue(ld.check_arm("kimi", "stock", '{"duet_runtime_options": {}}', ident, {})[0])

    def test_glm_base_arm_needs_no_release_and_its_duet_arm_adds_only_the_release(self):
        import json
        import tempfile
        ld = self.launcher()
        with tempfile.TemporaryDirectory() as d:
            model, out = Path(d) / "glm", Path(d) / "out"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next"}))
            self.assertEqual(ld.main(["--model", str(model), "--arm", "stock", "--port", "30600", "--gpus", "0,1,2,3",
                                      "--max-running-requests", "128", "--dry-run", "--out", str(out)]), 0)
            record = json.loads((out / "launch.json").read_text())
            self.assertEqual((record["family"], record["release"], record["release_identity"]), ("glm", None, None))
            argv = record["argv"]
            for flag in ld.GLM_SERVER_FLAGS + ("--disable-radix-cache", "--context-length", "262144"):
                self.assertIn(flag, argv)
            self.assertEqual(argv[argv.index("--tp-size") + 1], "4")
            self.assertEqual(record["env"]["SGLANG_MAMBA_SSM_DTYPE"], "float32")
            with self.assertRaises(SystemExit):          # a release is required for the DUET arm
                ld.main(["--model", str(model), "--arm", "duet", "--port", "30600", "--dry-run", "--out", str(out)])
            flags, env = ld.glm_duet_cli(Path(d), {})     # the GLM adapter (models/glm_duet): release + reference numerics
            self.assertEqual(flags, ["--duet-release", str(Path(d)), "--duet-numerics", "reference"])
            self.assertEqual(env, {"SGLANG_DUET_DIR": str(Path(d))})
            kimi = Path(d) / "kimi"
            kimi.mkdir()
            (kimi / "config.json").write_text(json.dumps({"architectures": ["KimiLinearForCausalLM"]}))
            with self.assertRaises(SystemExit):          # the other families still need --duet-release on both arms
                ld.main(["--model", str(kimi), "--arm", "stock", "--port", "30600", "--dry-run", "--out", str(out)])

    def test_server_environment_drops_inherited_duet_variables(self):
        ld = self.launcher()
        env, dropped = ld.server_environment({
            "SGLANG_DUET_CODE_PRECISION": "tf32", "TWINSTAR_BOUNDARY": "extend", "SGLANG_FLASHNEXT_DECODE_EPILOGUE": "1",
            "SGLANG_QWEN4_PREFILL_GRAPH": "1", "SGLANG_EXTERNAL_MODEL_PACKAGE": "twinstar_sgl",
            "SGLANG_DUET_HOOK_PROFILE": "1", "SGLANG_MAMBA_SSM_DTYPE": "float32", "PATH": "/bin"})
        self.assertEqual(dropped, ["SGLANG_DUET_CODE_PRECISION", "SGLANG_EXTERNAL_MODEL_PACKAGE",
                                   "SGLANG_FLASHNEXT_DECODE_EPILOGUE", "SGLANG_QWEN4_PREFILL_GRAPH", "TWINSTAR_BOUNDARY"])
        self.assertEqual(env, {"SGLANG_DUET_HOOK_PROFILE": "1", "SGLANG_MAMBA_SSM_DTYPE": "float32", "PATH": "/bin"})


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

    def test_unvalidated_production_override(self):
        # default off: refused
        with self.assertRaises(ValueError):
            numerics.require_profile("lightning", None, {}, production_supported=False)
        # CLI field or env turns the refusal into a warning + profile_validated=false
        cli = SimpleNamespace(duet_numerics=None, duet_allow_unvalidated_profile=True)
        self.assertEqual(numerics.require_profile("lightning", cli, {}, production_supported=False), "production")
        self.assertEqual(numerics.require_profile("lightning", None, {"SGLANG_DUET_ALLOW_UNVALIDATED": "1"}, production_supported=False), "production")
        self.assertFalse(numerics.profile_validated(cli, {}, production_supported=False))
        self.assertTrue(numerics.profile_validated(cli, {}, production_supported=True))
        self.assertTrue(numerics.profile_validated(SimpleNamespace(duet_numerics="reference"), {}, production_supported=False))
        d = numerics.describe(cli, {}, production_supported=False)
        self.assertEqual((d["allow_unvalidated"], d["profile_validated"]), (True, False))
        self.assertNotIn("profile_validated", numerics.describe(cli, {}))

    def test_code_precision_switch_and_validation_opt_in(self):
        options = _load("options")
        self.assertEqual(options.resolve_code_precision(None, {}), "fp32")
        self.assertEqual(options.resolve_code_precision(SimpleNamespace(duet_code_precision="tf32"), {}), "tf32")
        self.assertEqual(options.resolve_code_precision(None, {"SGLANG_DUET_CODE_PRECISION": "tf32"}), "tf32")
        with self.assertRaises(ValueError):
            options.resolve_code_precision(SimpleNamespace(duet_code_precision="bf16"), {})
        args = SimpleNamespace(duet_numerics=None, duet_code_precision=None)
        numerics.apply_defaults(args, {})
        self.assertEqual(args.duet_code_precision, "tf32")
        self.assertTrue(numerics.controls("production")["async_h2d"])
        self.assertFalse(numerics.controls("reference")["async_h2d"])
        # the validation opt-in starts an unvalidated production profile instead of refusing
        opted = SimpleNamespace(duet_numerics="production", duet_allow_unvalidated_profile=True)
        self.assertEqual(numerics.require_profile("lightning", opted, {}, production_supported=False), "production")
        self.assertEqual(numerics.require_profile("lightning", None, {"SGLANG_DUET_ALLOW_UNVALIDATED": "1"}, production_supported=False), "production")
        with self.assertRaises(ValueError):
            numerics.require_profile("lightning", SimpleNamespace(duet_numerics="production", duet_allow_unvalidated_profile=False), {}, production_supported=False)

    def test_codec_tf32_flag_is_scoped_and_value_preserving_on_cpu(self):
        import torch
        codec = _load("latent_codec")
        spec = dict(SPEC, latent_rank=32, latent_spikes=2)
        torch.manual_seed(0)
        E, D, mu = torch.randn(1, 32, 64), torch.randn(1, 64, 32), torch.randn(1, 64)
        plain = codec.PackedResidualCode(E, D, mu, 2)
        fast = codec.PackedResidualCode(E, D, mu, 2, tf32=True)
        h, emb = torch.randn(3, 64), torch.randn(3, 64)
        ids = torch.arange(3)
        before = torch.backends.cuda.matmul.allow_tf32
        a = plain.decode(plain.encode(h, emb, ids), emb)
        b = fast.decode(fast.encode(h, emb, ids), emb)
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, before, "the flag must be restored after the code")
        self.assertTrue(torch.equal(a, b), "on CPU the tf32 flag is a no-op; values must match exactly")
        code = codec.ResidualCode(64, spec, tf32=True)
        self.assertTrue(code.tf32)

    def test_describe(self):
        d = numerics.describe(SimpleNamespace(duet_numerics="reference"), {})
        self.assertEqual(d["profile"], "reference")
        self.assertEqual(d["controls"]["mamba_state_dtype"], "float32")


if __name__ == "__main__":
    unittest.main()
