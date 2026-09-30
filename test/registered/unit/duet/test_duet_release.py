"""CPU checks of sglang.srt.duet.release on a synthetic release directory (no Hub access, no model).

Run from this directory:  python -m unittest test_duet_release
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/sglang/srt/models"))
from lightning_duet._common import load as _load  # noqa: E402

release = _load("release")

SPEC = dict(model="lightning", prefill_depth=2, latent_rank=16, latent_spikes=2, latent_id_side=True,
            latent_z_format="nvfp4", latent_value_format="bf16", latent_index_format="gap8",
            state_rank=3, state_every=4, state_sink="explicit", latent_init="", state_init="", name="toy-k2")


class Geometry:
    """A 4-layer toy: state / attention / state / attention; k = 2 -> emitters for layers 2 and 3."""
    residual_dim = 8
    num_layers = 4
    state_heads = 2
    state_side_dim = 3

    def memory_kind(self, layer):
        return ("state", "attention", "state", "attention")[layer]

    def emitter_tensors(self, layer):
        if self.memory_kind(layer) == "state":
            return {"norm.weight": (8,), "mixer.w.weight": (6, 8)}
        return {"norm.weight": (8,), "k_proj.weight": (4, 8)}

    def inherited_tensors(self, layer):
        if self.memory_kind(layer) == "state":
            return {f"model.layers.{layer}.q.weight": (f"emitters.{layer}.q.weight", (6, 8))}
        return {}


def make_release(root, spec=SPEC, *, mutate=None, dtype=torch.float32, base_model="toy/base"):
    contract = release.build_contract(spec, Geometry())
    torch.manual_seed(0)
    tensors = {name: torch.randn(*shape).to(dtype) for name, (_, shape) in contract.expected.items()}
    if mutate:
        mutate(tensors)
    path = Path(root) / release.COMPONENT_FILE
    save_file(tensors, str(path), metadata={"format": "pt"})
    manifest = {"name": spec["name"], "base_model": base_model,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "file_bytes": path.stat().st_size,
                "total_params": sum(t.numel() for t in tensors.values()),
                "tensors": {k: {"shape": list(v.shape), "bytes": v.numel() * 4} for k, v in sorted(tensors.items())},
                "provenance": {"spec": dict(spec), "trained_spec": dict(spec), "decode_spec_differs_from_trained": False}}
    (Path(root) / "spec.json").write_text(json.dumps(spec))
    (Path(root) / "manifest.json").write_text(json.dumps(manifest))
    return contract, tensors, manifest


class ContractTests(unittest.TestCase):
    def test_contract_is_derived_from_spec_and_geometry(self):
        c = release.build_contract(SPEC, Geometry())
        self.assertIn("state.sink_dir", c.expected)
        self.assertEqual(c.expected["state.sink_dir"][1], (4, 2, 3))
        self.assertEqual(c.expected["latent.code.E"][1], (1, 16, 8))
        self.assertEqual(c.expected["latent.code.D"][1], (1, 8, 16))
        self.assertEqual(set(n for n in c.expected if n.startswith("emitters.")),
                         {"emitters.2.norm.weight", "emitters.2.mixer.w.weight", "emitters.3.norm.weight", "emitters.3.k_proj.weight"})
        self.assertEqual(c.inherited, {"model.layers.2.q.weight": ("emitters.2.q.weight", (6, 8))})
        self.assertEqual(release.build_contract({**SPEC, "latent_rank": 0}, Geometry()).expected.keys() & {"latent.code.E"}, set())
        with self.assertRaises(ValueError):
            release.build_contract({**SPEC, "latent_spikes": 99}, Geometry())


class VerifyTests(unittest.TestCase):
    def test_good_release_verifies_and_loads(self):
        with tempfile.TemporaryDirectory() as d:
            contract, tensors, manifest = make_release(d)
            ident = release.verify_release(d, geometry=Geometry(), base_model="toy/base", model="lightning")
            self.assertEqual((ident.name, ident.sha256, ident.file_bytes), ("toy-k2", manifest["sha256"], manifest["file_bytes"]))
            self.assertFalse(ident.decode_differs_from_trained)
            self.assertEqual(ident.audit["tensor_count"], len(contract.expected))
            self.assertIn("k=2", ident.describe())
            loaded = release.load_components(d, contract)
            for name, (target, _) in contract.expected.items():
                torch.testing.assert_close(loaded[target], tensors[name], rtol=0, atol=0)
                self.assertEqual(loaded[target].dtype, torch.float32)
            base = {"model.layers.2.q.weight": torch.ones(6, 8, dtype=torch.bfloat16)}
            inherited = release.inherit_from_base(contract, base.get)
            self.assertEqual((inherited["emitters.2.q.weight"].dtype, tuple(inherited["emitters.2.q.weight"].shape)), (torch.float32, (6, 8)))
            with self.assertRaises(ValueError):
                release.inherit_from_base(contract, lambda n: None)
            # identity only: verify without geometry still checks spec / sha / size
            self.assertEqual(release.verify_release(d).sha256, manifest["sha256"])

    def test_tampered_releases_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            make_release(d)
            m = json.loads((Path(d) / "manifest.json").read_text())
            (Path(d) / "spec.json").write_text(json.dumps({**SPEC, "state_evry": 4}))
            with self.assertRaises(ValueError):
                release.verify_release(d, geometry=Geometry())
            (Path(d) / "spec.json").write_text(json.dumps({**SPEC, "state_every": 8}))   # edited spec != provenance
            with self.assertRaises(ValueError):
                release.verify_release(d, geometry=Geometry())
            (Path(d) / "spec.json").write_text(json.dumps(SPEC))
            with self.assertRaises(ValueError):
                release.verify_release(d, geometry=Geometry(), base_model="other/base")
            with self.assertRaises(ValueError):
                release.verify_release(d, geometry=Geometry(), model="kimi-linear")
            bad = dict(m); bad["sha256"] = "0" * 64
            (Path(d) / "manifest.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                release.verify_release(d)
            bad = json.loads(json.dumps(m)); bad["tensors"]["latent.code.E"]["shape"] = [1, 16, 9]
            (Path(d) / "manifest.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                release.verify_release(d, geometry=Geometry())
        with tempfile.TemporaryDirectory() as d:   # a tensor missing from the file and the manifest
            make_release(d, mutate=lambda t: t.pop("emitters.3.k_proj.weight"))
            with self.assertRaises(ValueError):
                release.verify_release(d, geometry=Geometry())
            self.assertIsNotNone(release.verify_release(d))    # identity alone still consistent
        with tempfile.TemporaryDirectory() as d:   # bf16 file
            make_release(d, dtype=torch.bfloat16)
            with self.assertRaises(ValueError):
                release.verify_release(d, geometry=Geometry())


class SwitchTests(unittest.TestCase):
    def test_release_argument_and_directory_precedence(self):
        parser = argparse.ArgumentParser(); release.add_release_argument(parser)
        self.assertEqual(parser.parse_args(["--duet-release", "o/r@abc"]).duet_release, "o/r@abc")
        self.assertEqual(parser.parse_args([]).duet_release, None)
        self.assertEqual(release.parse_release_arg("o/r@abc"), ("o/r", "abc"))
        self.assertEqual(release.parse_release_arg("o/r"), ("o/r", None))
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(release.parse_release_arg(d), (d, None))
        self.assertEqual(release.resolve_release_dir("cli", environ={"SGLANG_DUET_DIR": "env"}), "cli")
        self.assertEqual(release.resolve_release_dir(None, environ={"SGLANG_DUET_DIR": "env", "TWINSTAR_KIMI_DUET_DIR": "old"}), "env")
        self.assertEqual(release.resolve_release_dir(None, environ={"TWINSTAR_LIGHTNING_DUET_DIR": "old"}), "old")
        self.assertIsNone(release.resolve_release_dir(None, environ={}))

    def test_fetch_uses_pinned_commit_and_rejects_symlinks(self):
        calls = []
        def info(repo, rev):
            calls.append(("info", repo, rev)); return SimpleNamespace(sha="deadbeef")
        def download(**kw):
            calls.append(("download", kw["revision"], kw["allow_patterns"]))
            Path(kw["local_dir"]).mkdir(parents=True, exist_ok=True)
            for f in release.RELEASE_FILES:
                (Path(kw["local_dir"]) / f).write_text("x")
        with tempfile.TemporaryDirectory() as root:
            path = release.fetch_release("owner/repo", root, "main", model_info=info, snapshot_download=download)
            self.assertEqual(path, Path(root) / "duet-releases" / "owner--repo" / "deadbeef")
            self.assertEqual(calls[0], ("info", "owner/repo", "main"))
            self.assertEqual(calls[1][1], "deadbeef")
            self.assertEqual(release.fetch_release(str(path), root), path)          # a directory is used as is
            os.symlink(path / "spec.json", path / "link.json")
            def download2(**kw): Path(kw["local_dir"]).mkdir(parents=True, exist_ok=True)
            with self.assertRaises(ValueError):
                release.fetch_release("owner/repo", root, "main", model_info=info, snapshot_download=download2)
        with self.assertRaises(ValueError):
            release.fetch_release("owner/repo", None, model_info=info, snapshot_download=download)


if __name__ == "__main__":
    unittest.main()
