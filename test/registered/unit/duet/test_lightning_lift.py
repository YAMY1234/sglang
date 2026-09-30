"""CPU integration of common DUET entry, invariants and left-sink slot state."""

import ast
import json
import os
import runpy
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/sglang/srt/models"))
from lightning_duet import components, latent
from lightning_duet import options as shim_options
from lightning_duet import state
from lightning_duet._common import load

options, spec, codec, factors, pools, invariants = (
    load(n)
    for n in (
        "options",
        "spec",
        "latent_codec",
        "state_factor",
        "state_pool",
        "invariants",
    )
)

SPEC = dict(
    model="external-adapter",
    prefill_depth=2,
    latent_rank=32,
    latent_spikes=3,
    latent_id_side=True,
    latent_z_format="nvfp4",
    latent_value_format="bf16",
    latent_index_format="gap8",
    state_rank=3,
    state_every=5,
    state_sink="explicit",
)


class EntryTests(unittest.TestCase):
    def test_release_enables_and_old_flag_alias(self):
        self.assertFalse(options.duet_enabled(environ={}))
        self.assertTrue(
            options.duet_enabled(
                environ={"SGLANG_DUET_DIR": "/release", "SGLANG_DUET_ENABLED": "0"}
            )
        )
        self.assertFalse(
            options.duet_enabled(
                environ={"SGLANG_DUET_DIR": "", "SGLANG_DUET_ENABLED": "0"}
            )
        )
        self.assertTrue(options.duet_enabled(environ={"SGLANG_DUET_ENABLED": "1"}))
        self.assertTrue(
            options.duet_enabled(
                environ={"OLD_ENABLE": "1"}, legacy_enabled="OLD_ENABLE"
            )
        )
        env = {"SGLANG_DUET_DIR": "/env", "OLD_DIR": "/old"}
        self.assertEqual(
            options.resolve_release(
                types.SimpleNamespace(duet_release="/cli"),
                env,
                legacy_directory="OLD_DIR",
            ),
            "/cli",
        )
        self.assertEqual(
            options.resolve_release(environ=env, legacy_directory="OLD_DIR"), "/env"
        )
        self.assertEqual(
            options.resolve_release(
                environ={"OLD_DIR": "/old"}, legacy_directory="OLD_DIR"
            ),
            "/old",
        )
        # A legacy directory on its own was never an enable switch.
        self.assertFalse(options.duet_enabled(environ={"OLD_DIR": "/old"}))

    def test_launcher_exports_canonical_names_and_keeps_other_args(self):
        from lightning_duet import launch_server

        argv = [
            "lightning",
            "--duet-release",
            "/cli",
            "--no-prefill-layer-trim",
            "--decode-ssm-r",
            "0",
            "--duet-emitter-precision",
            "bf16",
            "--duet-prefix-state",
            "factored",
            "--model-path",
            "/base",
        ]
        with (
            patch.dict(
                os.environ,
                {"SGLANG_DUET_DIR": "/env", "SGLANG_DUET_DECODE_SSM_W": "5"},
                clear=True,
            ),
            patch.object(sys, "argv", argv),
            patch.object(runpy, "run_module") as run,
        ):
            launch_server.main()
            self.assertEqual(
                sys.argv, ["sglang.launch_server", "--model-path", "/base"]
            )
            self.assertEqual(os.environ["SGLANG_DUET_DIR"], "/cli")
            self.assertEqual(os.environ["SGLANG_DUET_EMITTER_PRECISION"], "bf16")
            self.assertEqual(os.environ["SGLANG_DUET_PREFIX_STATE"], "factored")
            self.assertNotIn("SGLANG_DUET_DUET_RELEASE", os.environ)
            self.assertNotIn("SGLANG_DUET_DUET_EMITTER_PRECISION", os.environ)
            effective = options.DuetOptions.resolve(SPEC)
            self.assertEqual(
                (
                    effective.prefill_layer_trim,
                    effective.decode_ssm_r,
                    effective.decode_ssm_w,
                ),
                (False, 0, 5),
            )
            run.assert_called_once_with("sglang.launch_server", run_name="__main__")

    def test_native_cli_declares_the_same_release_option(self):
        source = r"""
import argparse, importlib.machinery, json, sys, types
from pathlib import Path
root = Path(sys.argv[1]) / "python/sglang"
for name, path in (("sglang", root), ("sglang.srt", root / "srt")):
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = mod
from sglang.srt.arg_groups.arg_utils import add_cli_args_from_dataclass
from sglang.srt.arg_groups.fields.exec_ import ExecMamba
parser = argparse.ArgumentParser()
add_cli_args_from_dataclass(parser, ExecMamba)
args = parser.parse_args(["--duet-release", "/native", "--no-prefill-layer-trim", "--decode-ssm-r", "0"])
print(json.dumps(dict(release=args.duet_release, trim=args.prefill_layer_trim, rank=args.decode_ssm_r)))
"""
        output = subprocess.check_output(
            [sys.executable, "-c", source, str(ROOT)], text=True
        )
        self.assertEqual(
            json.loads(output), dict(release="/native", trim=False, rank=0)
        )

    def test_native_resolution_publishes_cli_before_model_resolution(self):
        path = ROOT / "python/sglang/srt/server_args.py"
        cls = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "ServerArgs"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "resolve_once"
        )
        scope = {}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            scope,
        )
        calls = []
        pipeline = types.ModuleType("sglang.srt.arg_groups.pipeline")
        pipeline.run_resolution_pipeline = lambda args: calls.append(dict(os.environ))
        args = types.SimpleNamespace(
            duet_release="/native", prefill_layer_trim=False, decode_ssm_r=0
        )
        with (
            patch.dict(sys.modules, {pipeline.__name__: pipeline}),
            patch.dict(os.environ, {}, clear=True),
        ):
            scope["resolve_once"](args)
            scope["resolve_once"](args)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["SGLANG_DUET_DIR"], "/native")
        self.assertEqual(calls[0]["SGLANG_DUET_DECODE_SSM_R"], "0")
        self.assertEqual(calls[0]["SGLANG_DUET_PREFILL_LAYER_TRIM"], "False")
        self.assertTrue(args._resolution_finished)

    def test_control_server_clears_inherited_release(self):
        sys.path.insert(0, str(ROOT / "benchmark"))
        import lightning_sgl_stage2 as stage2

        args = types.SimpleNamespace(gpu=0, duet="/duet", model="/base")
        for mode in ("stock", "off", "duet"):
            proc = MagicMock(pid=123, poll=lambda: None)
            with (
                patch.dict(os.environ, {"SGLANG_DUET_DIR": "/inherited"}, clear=True),
                patch.object(Path, "mkdir"),
                patch.object(Path, "open", mock_open()),
                patch.object(stage2.subprocess, "Popen", return_value=proc) as start,
                patch.object(
                    stage2, "request", return_value={"data": [{"id": "lightning"}]}
                ),
                patch.object(stage2, "save"),
                patch.object(stage2.os, "killpg"),
            ):
                with stage2.server(args, mode, Path("/unused")):
                    env = start.call_args.kwargs["env"]
                    self.assertEqual(
                        options.duet_enabled(
                            environ=env, legacy_enabled="TWINSTAR_LIGHTNING_DUET"
                        ),
                        mode == "duet",
                    )
                    if mode != "duet":
                        self.assertNotIn("SGLANG_DUET_DIR", env)
                        self.assertNotIn("TWINSTAR_LIGHTNING_DUET_DIR", env)

    def test_stock_entry_identity_after_clearing_release(self):
        # Evaluate the actual registry selector without importing GPU kernels.
        path = ROOT / "python/sglang/srt/models/lightning_duet/engine.py"
        selector = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "EntryClass" for t in n.targets)
        )
        duet, stock = object(), object()
        for env, expected in (
            ({}, stock),
            ({"SGLANG_DUET_ENABLED": "0"}, stock),
            ({"SGLANG_DUET_DIR": "/r", "SGLANG_DUET_ENABLED": "0"}, duet),
            ({"TWINSTAR_LIGHTNING_DUET": "1"}, duet),
        ):
            scope = dict(
                _options=options, NemotronHForCausalLM=duet, StockNemotronH=stock
            )
            with patch.dict(os.environ, env, clear=True):
                exec(
                    compile(
                        ast.Module(body=[selector], type_ignores=[]), str(path), "exec"
                    ),
                    scope,
                )
            self.assertIs(scope["EntryClass"], expected)

    def test_adapter_declares_model_and_transport_capability(self):
        self.assertIs(spec.validate_spec(SPEC, model="external-adapter"), SPEC)
        with self.assertRaises(ValueError):
            components.validate_spec(SPEC)
        components.validate_spec(dict(SPEC, model="lightning"))
        with self.assertRaises(NotImplementedError):
            components.validate_spec(
                dict(SPEC, model="lightning", latent_z_format="bf16")
            )
        with self.assertRaises(NotImplementedError):
            components.validate_spec(dict(SPEC, model="lightning", latent_rank=0))
        self.assertIs(shim_options.DuetOptions, options.DuetOptions)
        self.assertIs(latent.ResidualCode, codec.PackedResidualCode)
        self.assertIs(state.LightningMambaStatePool, pools.LeftSinkStatePool)
        self.assertIs(state.factorize, factors.factorize_left)
        self.assertIs(components.rms_norm, invariants.reference_rms_norm)


class BoundaryAndSlotTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        torch.set_num_threads(1)

    def test_fresh_lookup_and_packed_roundtrip(self):
        table = torch.randn(7, 24).bfloat16()
        ids = torch.tensor([1, 3, 5])
        stale = table[ids].clone()
        stale.add_(7)
        hidden = stale + torch.randn_like(stale)
        code = codec.PackedResidualCode(
            torch.randn(1, 16, 24), torch.randn(1, 24, 16), torch.randn(1, 24), 4
        )
        calls = []

        def lookup(values):
            calls.append(values.clone())
            return table[values]

        record, decoded = invariants.reconstruct_boundary(
            code, hidden, ids, embedding_lookup=lookup
        )
        expected_record = code.encode(hidden, table[ids], ids)
        torch.testing.assert_close(
            decoded, code.decode(expected_record, table[ids]), rtol=0, atol=0
        )
        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(calls[1], record.token_ids.long(), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "aliases"):
            invariants.reconstruct_boundary(
                code, hidden, ids, embedding_lookup=lambda _: hidden
            )
        with self.assertRaisesRegex(ValueError, "shape/device"):
            invariants.reconstruct_boundary(
                code, hidden, ids, embedding_lookup=lambda _: table
            )

    def test_norm_order_bf16_and_fp32(self):
        for dtype in (torch.bfloat16, torch.float32):
            x, weight = torch.randn(9, 48).to(dtype), torch.randn(48).to(dtype)
            expected = weight * (
                x.float()
                * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
            ).to(dtype)
            torch.testing.assert_close(
                invariants.reference_rms_norm(x, weight, 1e-5), expected, rtol=0, atol=0
            )

    def test_hot_slot_snapshot_preserves_warm_ring_and_transfer_contract(self):
        pool = pools.LeftSinkStatePool(
            4,
            (1,),
            torch.randn(2, 2, 12),
            n=16,
            rank=3,
            window=5,
            conv_dim=4,
            conv_width=2,
            transfer_prefix="lightning_",
        )
        conv = torch.randn(4, 2)
        pool.initialize(1, 1, torch.randn(2, 12, 16), conv)
        for _ in range(8):
            pool.step(1, 1, torch.rand(2), torch.randn(2, 12), torch.randn(2, 16), conv)
        snapshot = pool.get_cpu_slots(torch.tensor([1]))
        self.assertEqual(snapshot["count"].item(), 3)
        pool.copy_slots(torch.tensor([1]), torch.tensor([2]))
        pool.reset_slots(torch.tensor([1]))
        pool.load_cpu_slots(snapshot, torch.tensor([1]))
        for _ in range(9):
            decay, x, b = torch.rand(2), torch.randn(2, 12), torch.randn(2, 16)
            a = pool.step(1, 1, decay, x, b, conv)
            c = pool.step(1, 2, decay, x, b, conv)
            torch.testing.assert_close(a, c, rtol=0, atol=0)
            for field in pool.fields:
                torch.testing.assert_close(
                    getattr(pool, field)[:, 1],
                    getattr(pool, field)[:, 2],
                    rtol=0,
                    atol=0,
                )
        entries = list(pool.iter_transfer_state_entries())
        self.assertEqual(
            {x[0] for x in entries}, {"lightning_" + f for f in pool.fields}
        )
        self.assertTrue(all(x[3] == 1 for x in entries))
        with self.assertRaises(ValueError):
            pool.load_cpu_slots({"warm": snapshot["warm"]}, torch.tensor([3]))


if __name__ == "__main__":
    unittest.main()
