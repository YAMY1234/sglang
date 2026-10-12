"""CPU checks of the Kimi-Linear release verification (sglang.srt.models.kimi_duet.kimi_duet_checkpoint) on a synthetic
Kimi-shaped release: the Kimi spec / base checks and the tensor contract of the module, the per-tensor audit, header and sha256 of
sglang.srt.duet.release.  A good release verifies with its inherited q tensors listed; every tampered one is refused.

Run from this directory:  python -m unittest test_kimi_checkpoint
"""
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/sglang/srt/models"))
from lightning_duet._common import load as _load  # noqa: E402  (registers the light sglang.srt.duet package on a bare CPU box)

_load("release")
_spec = importlib.util.spec_from_file_location("kimi_duet_checkpoint", ROOT / "test/registered/unit/duet/fixtures/seed_v35_kimi/kimi_duet_checkpoint.py")
ckpt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ckpt)

CONFIG = {"model_type": "kimi_linear", "mla_use_nope": True, "q_lora_rank": None, "hidden_size": 32, "num_hidden_layers": 4,
          "kv_lora_rank": 4, "qk_rope_head_dim": 2,
          "linear_attn_config": {"num_heads": 2, "head_dim": 4, "short_conv_kernel_size": 3, "kda_layers": [1, 3]}}
SPEC = dict(model="kimi-linear", prefill_depth=2, latent_rank=16, latent_spikes=2, latent_id_side=True, latent_z_format="nvfp4",
            latent_value_format="bf16", latent_index_format="gap8", state_rank=2, state_every=4, state_sink="explicit",
            latent_init="", state_init="", name="toy")


def write_release(root, *, spec=SPEC, base=ckpt.BASE_REPO, dtype=torch.float32, tensors=None, manifest=None):
    root.mkdir(parents=True)
    torch.manual_seed(0)
    t = {k: torch.randn(*shape).to(dtype) for k, (_, shape) in ckpt.tensor_contract(SPEC, CONFIG).items()}
    if tensors:
        tensors(t)
    path = root / "duet_components.safetensors"
    save_file(t, str(path))
    m = {"name": "toy", "base_model": base, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "file_bytes": path.stat().st_size,
         "total_params": sum(x.numel() for x in t.values()),
         "tensors": {k: {"shape": list(v.shape), "bytes": v.numel() * 4} for k, v in t.items()}, "provenance": {"spec": dict(spec)}}
    if manifest:
        manifest(m)
    (root / "spec.json").write_text(json.dumps(spec))
    (root / "manifest.json").write_text(json.dumps(m))
    return root


class KimiCheckpointTests(unittest.TestCase):
    def test_good_release_verifies_with_the_inherited_q_path(self):
        with tempfile.TemporaryDirectory() as d:
            report = ckpt.verify_release(write_release(Path(d) / "r"), CONFIG)
        # k = 2: emitters for layers 2 (KDA, 1-based kda_layers [1, 3] -> 0-based 0, 2) and 3 (MLA); E, D, mu and the sink
        self.assertEqual(report["tensor_count"], 4 + 10 + 3)
        self.assertEqual([r["source"] for r in report["inherited_base_tensors"]],
                         ["model.layers.2.self_attn.q_proj.weight", "model.layers.2.self_attn.q_conv1d.weight"])
        self.assertTrue(report["verified"])

    def test_tampered_releases_are_refused(self):
        cases = {
            "another base": dict(base="other/base"),
            "manifest sha256": dict(manifest=lambda m: m.update(sha256="0" * 64)),
            "file size": dict(manifest=lambda m: m.update(file_bytes=m["file_bytes"] + 1)),
            "parameter total": dict(manifest=lambda m: m.update(total_params=m["total_params"] + 1)),
            "missing tensor": dict(tensors=lambda t: t.pop("state.sink_dir")),
            "extra tensor": dict(tensors=lambda t: t.update(extra=torch.zeros(2))),
            "shape": dict(tensors=lambda t: t.update({"state.sink_dir": torch.zeros(4, 2, 5)})),
            "bf16 tensors": dict(dtype=torch.bfloat16),
            "edited spec": dict(manifest=lambda m: m["provenance"]["spec"].update(state_rank=3)),
            "unknown spec key": dict(spec={**SPEC, "extra": 1}),
            "init path": dict(spec={**SPEC, "latent_init": "x"}),
        }
        with tempfile.TemporaryDirectory() as d:
            for i, (name, kw) in enumerate(cases.items()):
                with self.subTest(name), self.assertRaises(ValueError):
                    ckpt.verify_release(write_release(Path(d) / str(i), **kw), CONFIG)
            root = write_release(Path(d) / "truncated")
            path = root / "duet_components.safetensors"
            path.write_bytes(path.read_bytes()[:-5])
            with self.assertRaises(ValueError):
                ckpt.verify_release(root, CONFIG)

    def test_another_base_config_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = write_release(Path(d) / "r")
            for change in ({"model_type": "other"}, {"mla_use_nope": False}, {"q_lora_rank": 8}):
                with self.subTest(change), self.assertRaises(ValueError):
                    ckpt.verify_release(root, {**CONFIG, **change})


class KimiLauncherReleaseArgument(unittest.TestCase):
    """`--duet <owner>/<repo>@<rev>` pins the revision (not a repository named '<repo>@<rev>'); `--revision` gives it apart."""

    def _fetch_args(self, *argv):
        from unittest import mock
        try:
            from sglang.srt.models.kimi_duet import launch
        except Exception as exc:  # noqa: BLE001 -- the sglang runtime is not importable on this machine
            self.skipTest(f"sglang.srt.models.kimi_duet.launch is not importable here: {exc}")
        calls = []

        class Fetched(Exception):
            pass

        def fetch(repo_or_path, hf_root, revision):
            calls.append((repo_or_path, revision))
            raise Fetched

        with tempfile.TemporaryDirectory() as d, mock.patch.object(launch, "fetch_release", fetch), \
                mock.patch.object(sys, "argv", ["launch", "--model", d, "--hf-cache", d, "--verify-only", *argv]):
            with self.assertRaises(Fetched):
                launch.main()
        return calls[0]

    def test_inline_and_separate_revisions(self):
        self.assertEqual(self._fetch_args("--duet", "o/rel@abc123"), ("o/rel", "abc123"))
        self.assertEqual(self._fetch_args("--duet", "o/rel", "--revision", "abc123"), ("o/rel", "abc123"))
        self.assertEqual(self._fetch_args("--duet", "o/rel"), ("o/rel", None))
        with self.assertRaises(SystemExit):
            self._fetch_args("--duet", "o/rel@abc123", "--revision", "def456")


if __name__ == "__main__":
    unittest.main()
