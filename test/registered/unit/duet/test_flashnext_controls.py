"""CPU contracts for Flash-Next precision and existing prefix representations."""
import argparse
from dataclasses import asdict
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from sglang.srt.arg_groups.arg_utils import add_cli_args_from_dataclass, resolvable_fields
from sglang.srt.arg_groups.fields.exec_ import ExecMamba
from sglang.srt.duet.options import add_arguments, resolve_emitter_precision, resolve_prefix_state
from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNConfig
from sglang.srt.model_executor.duet_policy import apply_duet_options
from sglang.srt.model_executor import fullstack_policy as policy


class PrecisionOptionsTest(unittest.TestCase):
    def test_default_and_legacy_alias(self):
        self.assertEqual(resolve_emitter_precision(environ={}), "fp32")
        for old, expected in (("0", "bf16"), ("1", "fp32")):
            self.assertEqual(resolve_emitter_precision(environ={"TWINSTAR_EMITTER_FP32": old}), expected)

    def test_cli_then_canonical_environment_then_alias(self):
        env = {"TWINSTAR_EMITTER_FP32": "1", "SGLANG_DUET_EMITTER_PRECISION": "bf16"}
        self.assertEqual(resolve_emitter_precision(environ=env), "bf16")
        self.assertEqual(resolve_emitter_precision(NS(duet_emitter_precision="fp32"), env), "fp32")
        for env in ({"SGLANG_DUET_EMITTER_PRECISION": "fp16"}, {"TWINSTAR_EMITTER_FP32": "invalid"}):
            with self.assertRaises(ValueError):
                resolve_emitter_precision(environ=env)

    def test_prefix_default_environment_cli_and_validation(self):
        self.assertEqual(resolve_prefix_state(environ={}), "exact")
        env = {"SGLANG_DUET_PREFIX_STATE": "factored"}
        self.assertEqual(resolve_prefix_state(environ=env), "factored")
        self.assertEqual(resolve_prefix_state(NS(duet_prefix_state="exact"), env), "exact")
        with self.assertRaises(ValueError):
            resolve_prefix_state(environ={"SGLANG_DUET_PREFIX_STATE": "dense-ish"})

    def test_server_and_standalone_cli_are_equivalent_and_resolvable(self):
        for register in (add_arguments, lambda p: add_cli_args_from_dataclass(p, ExecMamba)):
            parser = argparse.ArgumentParser()
            register(parser)
            args = parser.parse_args(["--duet-emitter-precision", "bf16", "--duet-prefix-state", "factored"])
            self.assertEqual(resolve_emitter_precision(args, {}), "bf16")
            self.assertEqual(resolve_prefix_state(args, {}), "factored")
            args = parser.parse_args([])
            self.assertEqual(resolve_emitter_precision(args, {}), "fp32")
            self.assertEqual(resolve_prefix_state(args, {}), "exact")
        self.assertTrue({"duet_emitter_precision", "duet_prefix_state"} <= resolvable_fields(ExecMamba))


class PrefixWiringTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"SGLANG_EXTERNAL_MODEL_PACKAGE": "twinstar_sgl",
                                     "SGLANG_DUET_NUMERICS": "reference"}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        spec = dict(model="flash-next", prefill_depth=31, state_rank=12, state_every=4, latent_rank=2048, latent_spikes=128,
                    latent_z_format="nvfp4", state_sink="explicit", latent_id_side=True,
                    latent_value_format="bf16", latent_index_format="gap8")
        self.fs = dict(release='/release', version=3, duet_spec=spec, latent="on", latent_rank=2048, latent_sparse=128,
                       latent_store="nvfp4", state_sink="explicit", latent_id_side=True,
                       latent_value_format="bf16", latent_index_format="gap8", latent_rms=False,
                       gdn_state="rank:12", gdn_rank=12, gdn_every=4, prefill_saving_policy="kv-and-ssm",
                       state_sink_vbar=__file__)
        self.config = NS(hf_config=NS(twinstar={"fullstack": self.fs}))

    def test_default_exact_and_explicit_factored_keep_algorithm_fields(self):
        apply_duet_options(self.config, NS())
        exact = FactoredGDNConfig.parse(policy.fullstack_state_config(self.config, radix=True))
        self.assertEqual((self.fs["duet_emitter_precision"], self.fs["duet_prefix_state"]), ("fp32", "exact"))
        self.assertEqual((exact.exact_prefix, exact.factored_prefix), (1, 0))
        apply_duet_options(self.config, NS(duet_emitter_precision="bf16", duet_prefix_state="factored"))
        factored = FactoredGDNConfig.parse(policy.fullstack_state_config(self.config, radix=True))
        self.assertEqual((factored.exact_prefix, factored.factored_prefix), (0, 1))
        self.assertEqual(self.fs["duet_emitter_precision"], "bf16")
        a, b = asdict(exact), asdict(factored)
        for field in ("exact_prefix", "factored_prefix", "raw"):
            a.pop(field); b.pop(field)
        self.assertEqual(a, b)
        disabled = FactoredGDNConfig.parse(policy.fullstack_state_config(self.config, radix=False))
        self.assertEqual((disabled.exact_prefix, disabled.factored_prefix), (0, 0))

    def test_release_name_cannot_select_a_legacy_schema(self):
        self.fs.pop("duet_spec")
        self.fs["release_name"] = "duet-fn-v3-r4096"
        with self.assertRaisesRegex(ValueError, "spec-driven"):
            policy.fullstack_v3_config(self.config)


    def test_flag_off_ignores_prefix_configuration(self):
        self.config.hf_config.twinstar = None
        os.environ.pop("SGLANG_DUET_DIR", None)
        os.environ["SGLANG_DUET_PREFIX_STATE"] = "invalid"
        self.assertIsNone(policy.fullstack_state_config(self.config, radix=True))


if __name__ == "__main__":
    unittest.main()
