"""CPU contract checks against both pinned reference release manifests."""
import copy
import json
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/sglang/srt/models"))
from lightning_duet._common import load
load("spec")
release = load("release")
from kimi_linear_duet.checkpoint import KimiGeometry, tensor_contract, validate_spec


def audit_metadata(spec, manifest, header, config):
    validate_spec(spec, config)
    if spec != manifest["provenance"]["spec"]:
        raise ValueError("spec differs from provenance")
    return release.audit_metadata(release.build_contract(spec, KimiGeometry(config)), manifest, header)

REF = "dd9c7bdbd9550a5d86781ebfa3d965d01fc1e78a"
CONFIG = {
    "model_type": "kimi_linear", "mla_use_nope": True, "q_lora_rank": None,
    "hidden_size": 2304, "num_hidden_layers": 27, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
    "linear_attn_config": {"num_heads": 32, "head_dim": 128, "short_conv_kernel_size": 4,
        "kda_layers": [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26]},
}


def fixture(version="duet-kimi-v2"):
    raw = (Path(__file__).parent / "fixtures" / f"{version}-manifest.json").read_text()
    manifest = json.loads(raw)
    spec, header, offset = manifest["provenance"]["spec"], {}, 0
    for key, value in manifest["tensors"].items():
        header[key] = {"shape": value["shape"], "dtype": "F32", "data_offsets": [offset, offset + value["bytes"]]}
        offset += value["bytes"]
    return spec, manifest, header


class CheckpointTests(unittest.TestCase):
    def test_both_published_manifests_have_exact_mapping(self):
        for version, depth, rank, count in (("duet-kimi", 17, 2048, 83), ("duet-kimi-v2", 19, 2304, 63)):
            spec, manifest, header = fixture(version)
            report = audit_metadata(spec, manifest, header, CONFIG)
            self.assertEqual((spec["prefill_depth"], spec["latent_rank"], report["tensor_count"]), (depth, rank, count))
            self.assertEqual(len({r["target"] for r in report["mapping"]}), count)
            self.assertTrue(all("q_proj" not in r["source"] for r in report["mapping"]))
            self.assertEqual(len(report["inherited_base_tensors"]), 14 if depth == 17 else 10)

    def test_missing_extra_shape_dtype_and_offsets_rejected(self):
        spec, manifest, original = fixture()
        key = "latent.code.E"
        for defect in ("missing", "extra", "shape", "dtype", "offset"):
            header = copy.deepcopy(original)
            if defect == "missing": header.pop(key)
            if defect == "extra": header["not_a_weight"] = header[key]
            if defect == "shape": header[key]["shape"] = [1, 2048, 2304]
            if defect == "dtype": header[key]["dtype"] = "BF16"
            if defect == "offset": header[key]["data_offsets"][0] += 1
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                audit_metadata(spec, manifest, header, CONFIG)

    def test_provenance_and_unknown_spec_rejected(self):
        spec, manifest, header = fixture()
        wrong = dict(spec, state_every=8)
        with self.assertRaises(ValueError): audit_metadata(wrong, manifest, header, CONFIG)
        with self.assertRaises(ValueError): tensor_contract(dict(spec, extra_flag=True), CONFIG)
        with self.assertRaises(ValueError): tensor_contract(dict(spec, latent_rank=2303), CONFIG)

    def test_dimensions_sink_and_disabled_formats_come_from_spec(self):
        spec, _, _ = fixture()
        alternate = dict(spec, prefill_depth=13, latent_rank=1536, latent_spikes=31,
                         state_sink="implicit", state_rank=9, state_every=5)
        contract = tensor_contract(alternate, CONFIG)
        self.assertEqual(contract["latent.code.E"][1], (1, 1536, CONFIG["hidden_size"]))
        self.assertIn("emitters.13.norm.weight", contract)
        exact = tensor_contract(dict(alternate, latent_rank=0, state_rank=0, state_every=0), CONFIG)
        self.assertFalse(any(k.startswith("latent.") for k in exact))


if __name__ == "__main__":
    unittest.main()
