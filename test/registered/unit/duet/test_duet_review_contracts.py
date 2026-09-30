"""CPU regressions for the five common-layer review notes on Lightning #20."""

import ast
import math
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/sglang/srt/models"))
from lightning_duet._common import load

options, release, spec, pools = (
    load(n) for n in ("options", "release", "spec", "state_pool")
)


class LoaderTests(unittest.TestCase):
    def test_clean_cpu_loaders_share_package_and_relative_imports(self):
        source = r"""
import argparse, importlib.util, sys
from pathlib import Path
root = Path(sys.argv[1]) / "python/sglang/srt"
def direct(key, path):
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module
policy = direct("cpu_fullstack_policy", root / "model_executor/fullstack_policy.py")
if sys.argv[2] == "release":
    release = direct("sglang.srt.duet.release", root / "duet/release.py")
    options = release._sibling("options")
elif sys.argv[2] == "policy":
    options = policy._duet_options()
else:
    sys.path.insert(0, str(root / "models"))
    from lightning_duet._common import load
    options = load("options")
parser = argparse.ArgumentParser()
options.add_arguments(parser)  # relative .release import used to fail here
assert parser.parse_args(["--duet-release", "/release"]).duet_release == "/release"
sys.path.insert(0, str(root / "models"))
from lightning_duet._common import load
assert options is policy._duet_options() is load("release")._sibling("options") is load("options")
assert sys.modules["sglang.srt.duet"].__path__
assert "sglang" not in sys.modules and "sglang.srt" not in sys.modules
assert "triton" not in sys.modules
"""
        for first in ("release", "policy", "adapter"):
            with self.subTest(first=first):
                subprocess.run(
                    [sys.executable, "-c", source, str(ROOT), first], check=True
                )


class OptionsTests(unittest.TestCase):
    def test_design_release_entry_preserves_resolution_and_verification(self):
        identity = object()
        geometry = object()
        info = object()
        download = object()
        with (
            patch.object(release, "fetch_release", return_value="/verified") as fetch,
            patch.object(release, "verify_release", return_value=identity) as verify,
        ):
            result = release.open_release(
                "owner/duet@pinned",
                hf_root="/cache",
                geometry=geometry,
                base_model="base",
                model="lightning",
                model_info=info,
                snapshot_download=download,
            )
            self.assertIs(result, identity)
            fetch.assert_called_once_with(
                "owner/duet",
                "/cache",
                "pinned",
                model_info=info,
                snapshot_download=download,
            )
            verify.assert_called_once_with(
                "/verified", geometry=geometry, base_model="base", model="lightning"
            )
        with patch.object(release, "open_release", return_value=identity) as opened:
            self.assertIsNone(release.release_from_args(environ={}))
            opened.assert_not_called()
            args = types.SimpleNamespace(duet_release="cli")
            env = {"SGLANG_DUET_DIR": "env", "OLD_DIR": "old"}
            self.assertIs(
                release.release_from_args(
                    args, environ=env, legacy_directory="OLD_DIR", model="lightning"
                ),
                identity,
            )
            opened.assert_called_once_with("cli", model="lightning")
            opened.reset_mock()
            release.release_from_args(
                environ={"OLD_DIR": "old"}, legacy_directory="OLD_DIR"
            )
            opened.assert_called_once_with("old")
        with self.assertRaises(ValueError):
            release.open_release(None)

    def test_release_wrapper_has_one_precedence_and_warning_contract(self):
        aliases = options.LEGACY_DIR_ALIASES
        self.assertEqual(release.LEGACY_DIR_ALIASES, aliases)
        for cli, env, expected in (
            ("cli", {"SGLANG_DUET_DIR": "canonical", aliases[0]: "old"}, "cli"),
            (None, {"SGLANG_DUET_DIR": "canonical", aliases[0]: "old"}, "canonical"),
            (None, {aliases[0]: "kimi", aliases[1]: "lightning"}, "kimi"),
            (None, {aliases[1]: "lightning"}, "lightning"),
            (None, {}, None),
        ):
            with self.subTest(cli=cli, env=env):
                before = dict(env)
                args = types.SimpleNamespace(duet_release=cli)
                self.assertEqual(
                    options.resolve_release(args, env, legacy_directory=aliases),
                    expected,
                )
                self.assertEqual(
                    release.resolve_release_dir(cli, environ=env), expected
                )
                self.assertEqual(env, before)
        with self.assertLogs(options.__name__, level="WARNING") as messages:
            release.resolve_release_dir(environ={aliases[1]: "legacy"})
        self.assertIn("deprecated", messages.output[0])
        # Adapter-specific aliases do not select another model's old directory.
        self.assertIsNone(
            options.resolve_release(
                environ={aliases[0]: "kimi"}, legacy_directory=aliases[1]
            )
        )
        self.assertFalse(options.duet_enabled(environ={aliases[1]: "lightning"}))

    def test_cli_preview_is_pure_and_export_is_explicit_and_idempotent(self):
        args = types.SimpleNamespace(
            duet_release="cli", decode_ssm_r=0, prefill_layer_trim=False
        )
        ambient = {"SGLANG_DUET_DIR": "ambient", "SGLANG_DUET_DECODE_SSM_W": "5"}
        with patch.dict(os.environ, ambient, clear=True):
            overrides = options.cli_environment(args)
            self.assertEqual(dict(os.environ), ambient)
            target = dict(ambient)
            options.export_cli_environment(args, target)
            self.assertEqual(dict(os.environ), ambient)
            self.assertEqual(target, dict(ambient, **overrides))
            self.assertEqual(target["SGLANG_DUET_DECODE_SSM_W"], "5")
            options.export_cli_environment(args, target)
            self.assertEqual(target, dict(ambient, **overrides))
            options.export_cli_environment(types.SimpleNamespace())
            self.assertEqual(dict(os.environ), ambient)
            options.export_cli_environment(args)
            self.assertEqual(dict(os.environ), target)

    def test_reference_models_are_informational_not_an_allowlist(self):
        self.assertEqual(spec.MODELS, ("flash-next", "lightning", "kimi-linear"))
        release_spec = dict(
            model="new-adapter",
            prefill_depth=1,
            latent_rank=0,
            latent_spikes=0,
            latent_id_side=False,
            latent_z_format="bf16",
            latent_value_format="bf16",
            latent_index_format="gap8",
            state_rank=0,
            state_every=0,
            state_sink="explicit",
        )
        self.assertIs(
            spec.validate_spec(release_spec, model="new-adapter"), release_spec
        )
        with self.assertRaises(ValueError):
            spec.validate_spec(release_spec, model="lightning")


def transfer_consumer(sibling):
    """Execute the real MambaPool metadata consumers without importing GPU allocators."""
    path = ROOT / "python/sglang/srt/mem_cache/memory_pool.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "MambaPool"
    )
    names = {
        "_iter_transfer_state_entries",
        "get_contiguous_buf_infos",
        "get_state_dim_per_tensor",
        "get_state_layer_ids",
        "get_state_slice_outer_counts",
    }
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert len(methods) == len(names)
    scope = {"math": math}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), scope)
    consumer = type("MambaTransferConsumer", (), {n: scope[n] for n in names})()
    consumer.mamba_cache = types.SimpleNamespace()
    consumer._slot_siblings = [sibling]
    return consumer


class TransferTests(unittest.TestCase):
    def test_sidecar_metadata_matches_mamba_consumers(self):
        sidecar = pools.SlotSidecar(
            3, [4, 19, 33], heads=2, warm_dim=5, rank=3, every=4
        )
        consumer = transfer_consumer(sidecar)
        entries = list(sidecar.iter_transfer_state_entries())
        dims = consumer.get_state_dim_per_tensor()
        self.assertEqual(len(entries), 3 * len(sidecar.fields))
        for (name, tensor, axis, layer), dim in zip(entries, dims):
            self.assertIn(layer, (4, 19, 33))
            self.assertEqual(tensor.shape[0], 3)
            self.assertIsNone(axis)
            self.assertEqual(dim, 0)
        self.assertEqual(consumer.get_state_layer_ids(), [e[3] for e in entries])
        self.assertEqual(consumer.get_state_slice_outer_counts(), [1] * len(entries))
        pointers, lengths, item_lengths = consumer.get_contiguous_buf_infos()
        self.assertEqual(pointers, [e[1].data_ptr() for e in entries])
        self.assertEqual(lengths, [e[1].nbytes for e in entries])
        self.assertEqual(item_lengths, [e[1][0].nbytes for e in entries])

    def test_exact_and_zero_cadence_do_not_advertise_empty_buffers(self):
        sidecar = pools.SlotSidecar(3, [4, 19], heads=2, warm_dim=5, rank=0, every=0)
        self.assertNotIn(
            "duet_sidecar_warm", [e[0] for e in sidecar.iter_transfer_state_entries()]
        )
        for rank, window in ((0, 0), (0, 4), (2, 0), (2, 4)):
            left = pools.LeftSinkStatePool.for_test(
                3, torch.ones(2, 2, 5), n=4, rank=rank, window=window
            )
            for pool in (left, sidecar):
                with self.subTest(pool=type(pool).__name__, rank=rank, window=window):
                    consumer = transfer_consumer(pool)
                    _, lengths, item_lengths = consumer.get_contiguous_buf_infos()
                    self.assertTrue(all(x > 0 for x in lengths + item_lengths))
                    self.assertEqual(
                        consumer.get_state_dim_per_tensor(), [0] * len(lengths)
                    )


if __name__ == "__main__":
    unittest.main()
