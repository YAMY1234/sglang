"""CPU contracts for in-tree Flash-Next (no GPU/runtime imports required)."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/sglang/srt/models"))
from lightning_duet._common import load

common = load("release")
options = load("options")
load("adapters")
load("numerics")
from flash_next_duet import config, diagnostics, latent, release

SPEC = dict(model="flash-next", prefill_depth=2, latent_rank=16, latent_spikes=4,
            latent_id_side=True, latent_z_format="nvfp4", latent_value_format="bf16",
            latent_index_format="gap8", state_rank=4, state_every=4, state_sink="explicit",
            latent_init="", state_init="", name="arbitrary-release-name")
TEXT = dict(num_hidden_layers=4, hidden_size=16, hc_count=4, hc_lowrank=8,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
            indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=4,
            num_experts=8, num_experts_per_tok=2,
            layer_types=["linear_attention", "full_attention"] * 2)
BASE = dict(architectures=["Qwen4ExpForConditionalGeneration"], text_config=TEXT)


def make_release(directory):
    geometry = release.FlashNextGeometry(BASE)
    contract = common.build_contract(SPEC, geometry)
    tensors = {name: torch.randn(shape) for name, (_, shape) in contract.expected.items()}
    path = directory / common.COMPONENT_FILE
    save_file(tensors, str(path))
    manifest = dict(name="display-name", base_model="test/base", file_bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    total_params=sum(t.numel() for t in tensors.values()),
                    tensors={k: dict(shape=list(t.shape), bytes=t.numel() * 4) for k, t in tensors.items()},
                    provenance=dict(spec=SPEC, trained_spec=SPEC))
    (directory / "spec.json").write_text(json.dumps(SPEC))
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return tensors


class FlashNextAdapterTests(unittest.TestCase):
    def test_native_fp8_ple_storage_needs_no_release(self):
        cfg = SimpleNamespace(architectures=BASE["architectures"], text_config=SimpleNamespace(**TEXT),
                              quantization_config=dict(quantized_layers={
                                  "model.language_model.layers.2.ple.ple_embedding.ngram_embedding": dict(quant_algo="FP8")}))
        config.prepare_base_config(cfg)
        self.assertEqual(cfg.text_config.ple_embedding_dtype, "float8_e4m3fn")
        self.assertFalse(hasattr(cfg, "twinstar"))
        self.assertFalse(hasattr(cfg, "_duet_identity"))

    def test_bf16_and_nvfp4_base_validation(self):
        self.assertEqual(config.validate_base(BASE), {})
        quantized = copy.deepcopy(BASE)
        quantized["quantization_config"] = dict(quant_method="modelopt", quant_algo="NVFP4", ignore=["*"])
        self.assertEqual(config.validate_base(quantized)["quant_algo"], "NVFP4")
        quantized["quantization_config"]["ignore"] = []
        with self.assertRaisesRegex(ValueError, "may be quantized"):
            config.validate_base(quantized)
        invalid = copy.deepcopy(BASE)
        invalid["text_config"].pop("layer_types")
        with self.assertRaisesRegex(ValueError, "layer_type"):
            config.validate_base(invalid)

    def test_geometry_identity_and_sink_cache(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            tensors = make_release(directory)
            identity = release.validate_release(directory, BASE, base_model="test/base")
            self.assertEqual(identity.spec["name"], "arbitrary-release-name")
            path = release.sink_cache(identity)
            cached = torch.load(path, weights_only=True)
            torch.testing.assert_close(cached["vbar"][2], tensors["P.state.sink_dir"][2].bfloat16().float(), rtol=0, atol=0)
            self.assertEqual(release.sink_cache(identity), path)
            with self.assertRaisesRegex(ValueError, "trained for"):
                release.validate_release(directory, BASE, base_model="different/base")
            bad = copy.deepcopy(BASE)
            bad["text_config"]["head_dim"] *= 2
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                release.validate_release(directory, bad)

    def test_readonly_release_cache_falls_back_to_hf_home(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            make_release(directory)
            identity = release.validate_release(directory, BASE)
            original = Path.mkdir
            def mkdir(path, *a, **kw):
                if path == directory / ".cache":
                    raise PermissionError("read-only release")
                return original(path, *a, **kw)
            with patch.object(Path, "mkdir", mkdir), patch.dict(os.environ, HF_HOME=str(directory / "hf")):
                output = release.sink_cache(identity)
            self.assertIn(f"duet-releases/.cache/{identity.sha256}/", output)

    def test_reference_view_matches_offline_tool_every_key(self):
        source = Path(os.environ.get("TWINSTAR_REFERENCE_ROOT", ROOT.parent / "twinstar-review-wt")) / "twinstar/fullstack_hf_release.py"
        if not source.is_file():
            self.skipTest("offline reference tool must be provided for G1 parity")
        spec = importlib.util.spec_from_file_location("offline_fullstack", source)
        offline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(offline)
        identity = SimpleNamespace(spec=SPEC, path="/release", sha256="a" * 64)
        base = copy.deepcopy(BASE)
        base["quantization_config"] = dict(quant_method="modelopt", quant_algo="NVFP4", ignore=["*"])
        resolved = options.DuetOptions.resolve(SPEC, environ={})
        expected = offline.model_overrides(base, vars(identity), "/sink.pt")
        actual = config.derive_fullstack(identity, base, resolved, "reference", sink_file="/sink.pt")
        self.assertEqual(actual, expected)

    def test_zero_options_and_production_view(self):
        identity = SimpleNamespace(spec=SPEC, path="/release", sha256="a" * 64)
        args = SimpleNamespace(decode_ssm_r=0, decode_ssm_w=0, prefill_layer_trim=False)
        resolved = options.DuetOptions.resolve(SPEC, args, environ={}, state_dim=8)
        view = config.derive_fullstack(identity, BASE, resolved, "production", sink_file="/sink.pt")
        fs = view["twinstar"]["fullstack"]
        self.assertEqual((fs["gdn_state"], fs["gdn_every"], fs["prefill_layer_trim"]), ("dense", 0, False))
        self.assertEqual(fs["latent_compute_precision"], "tf32")

    def test_codec_matches_reference_and_encodes_position_zero(self):
        torch.manual_seed(42)
        served = latent.FlashNextLatentCodec(device="cpu", spec=SPEC, width=64)
        reference = load("latent_codec").ResidualCode(64, SPEC)
        with torch.no_grad():
            for key, shape in (("E", (1, 16, 64)), ("D", (1, 64, 16)), ("mu", (1, 64))):
                value = torch.randn(shape) / 8
                served.load(key, value)
                getattr(reference.code, key).copy_(value.bfloat16().float())
        streams, base = torch.randn(4, 64).bfloat16(), torch.randn(4, 64).bfloat16()
        record, actual = served.encode_and_decode(streams, torch.arange(4), base)
        expected = reference(streams, base)
        self.assertEqual(record.codes.shape[0], 4)
        self.assertFalse(torch.equal(actual[0], streams[0]))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(served.decode(record, base), expected, rtol=0, atol=0)

    def test_missing_diagnostics_are_optional(self):
        diagnostics.optional.cache_clear()
        with patch.object(diagnostics.importlib.util, "find_spec", side_effect=ModuleNotFoundError):
            self.assertIsNone(diagnostics.optional("pd_shallow_install"))
            self.assertIsNone(diagnostics.optional("fullstack_state_audit"))


if __name__ == "__main__":
    unittest.main()
