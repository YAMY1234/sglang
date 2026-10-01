"""CPU checks of Kimi release selection, all-off delegation and profile controls."""
import ast
import copy
import itertools
import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/sglang/srt/models"))
from lightning_duet._common import load

load("spec")
release, options, numerics = (load(n) for n in ("release", "options", "numerics"))
from kimi_linear_duet.checkpoint import base_model_name
from kimi_linear_duet.controls import code_precision, configure_state_dtype, reject_legacy_overrides
from test_kimi_duet_checkpoint import CONFIG, fixture


class EntryTests(unittest.TestCase):
    def test_batched_boundary_keeps_request_slots_and_single_token_prompt(self):
        source = ROOT / "python/sglang/srt/models/kimi_linear_duet/model.py"
        cls = next(n for n in ast.parse(source.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "KimiLinearForCausalLM")
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("_sub_batch", "_decode_batch")]
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *methods], type_ignores=[])
        ns = dict(torch=torch, copy=copy, itertools=itertools, ForwardMode=SimpleNamespace(DECODE="decode"))
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), ns)
        fb = SimpleNamespace(extend_seq_lens_cpu=[1, 3], extend_prefix_lens_cpu=[0, 0],
                             req_pool_indices=torch.tensor([9, 4]), seq_lens=torch.tensor([1, 3]),
                             seq_lens_cpu=torch.tensor([1, 3]), orig_seq_lens=None,
                             out_cache_loc=torch.tensor([20, 30, 31, 32]),
                             extend_seq_lens=torch.tensor([1, 3]), extend_prefix_lens=torch.tensor([0, 0]),
                             extend_start_loc=torch.tensor([0, 1]), extend_logprob_start_lens_cpu=None)
        ids, pos = torch.tensor([10, 11, 12, 13]), torch.tensor([0, 0, 1, 2])
        shallow, index = ns["_sub_batch"](None, fb, ids, pos, [1, 1], "p")
        self.assertEqual(index.tolist(), [1, 2])
        self.assertEqual(shallow.req_pool_indices.tolist(), [4])
        self.assertEqual(shallow.out_cache_loc.tolist(), [30, 31])
        boundary = ns["_decode_batch"](None, fb, ids, pos, torch.tensor([0, 3]), [0, 1], [1, 3])
        self.assertEqual(boundary.input_ids.tolist(), [10, 13])
        self.assertEqual(boundary.positions.tolist(), [0, 2])
        self.assertEqual(boundary.req_pool_indices.tolist(), [9, 4])
        self.assertEqual(boundary.out_cache_loc.tolist(), [20, 32])

    def test_codec_precision_does_not_leak_to_state_or_emitters(self):
        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            for profile, enabled in (("reference", False), ("production", True)):
                torch.backends.cuda.matmul.allow_tf32 = not enabled
                with self.assertRaisesRegex(RuntimeError, "codec failed"):
                    with code_precision(profile):
                        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, enabled)
                        raise RuntimeError("codec failed")
                self.assertEqual(torch.backends.cuda.matmul.allow_tf32, not enabled)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    def test_local_base_identity_and_conflicting_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps({"base_model": "owner/Example"}))
            self.assertEqual(base_model_name({}, "/models/Example", root), "owner/Example")
            with self.assertRaisesRegex(ValueError, "does not match"):
                base_model_name({}, "/models/Other", root)
            self.assertEqual(base_model_name({"_name_or_path": "different/Example"},
                                             "/models/Example", root), "different/Example")

    def test_profile_gate_and_dtype_order(self):
        spec, _, _ = fixture()
        active = options.DuetOptions.resolve(spec, environ={})
        args = SimpleNamespace(duet_numerics="reference")
        env = {}
        configure_state_dtype(args, active, environ=env)
        self.assertEqual(env, {"SGLANG_MAMBA_CONV_DTYPE": "float32", "SGLANG_MAMBA_SSM_DTYPE": "float32"})
        with self.assertRaisesRegex(RuntimeError, "before pool creation"):
            configure_state_dtype(args, active, pool=object(), environ={})
        args.duet_numerics = "production"
        with self.assertRaisesRegex(ValueError, "not validated"):
            numerics.require_profile("kimi-linear", args, production_supported=False)
        env = {}
        configure_state_dtype(args, active, environ=env)
        self.assertEqual(env, {})

    def test_rejects_diagnostic_overrides(self):
        for name in ("SGLANG_KDA_STATE_PRUNE_RANK", "SGLANG_KDA_STATE_PRUNE_CALIB", "LATENT_OFF"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                reject_legacy_overrides({name: "1"})
        reject_legacy_overrides({"LATENT_OFF": "0"})

    def test_alloff_retains_adapter_and_calls_stock_without_duet_allocations(self):
        # Execute the real model class with only the GPU base model replaced.
        # Neither an options-only assertion nor a mocked constructor can prove
        # that emitter/code/policy construction and component loading are skipped.
        source = ROOT / "python/sglang/srt/models/kimi_linear_duet/model.py"
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == "KimiLinearForCausalLM")
        class Stock(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.loaded = []
            def load_weights(self, weights):
                self.loaded = list(weights)
            def forward(self, *args):
                return "stock-output"
        args = SimpleNamespace(duet_numerics="reference", duet_release="/release",
                               model_path="owner/Example", prefill_layer_trim=False,
                               decode_ssm_r=0, decode_ssm_w=0)
        cfg = SimpleNamespace(**copy.deepcopy(CONFIG))
        cfg.to_dict = lambda: CONFIG
        spec, _, _ = fixture()
        identity = SimpleNamespace(spec=spec, audit={}, sha256="test")
        ns = dict(__name__="kimi_linear_duet.model", __package__="kimi_linear_duet",
                  Path=Path, torch=torch, nn=torch.nn, _stock=SimpleNamespace(KimiLinearForCausalLM=Stock),
                  get_parallel=lambda: SimpleNamespace(tp_rank=0), get_server_args=lambda: args,
                  release=release, options=options, numerics=numerics, os=os,
                  logger=logging.getLogger(__name__), _E_PREFIX="model.emitters.", _B_PREFIX="model.bridge.")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(source), "exec"), ns)
        with patch.dict(os.environ, {}, clear=True), patch.object(release, "release_from_args", return_value=identity):
            model = ns["KimiLinearForCausalLM"](cfg)
            self.assertIs(model.duet_release, identity)
            self.assertFalse(model.duet_active)
            self.assertFalse(model.emitters)
            for name in ("latent", "duet_sink_dir", "kda_state_pruner_factory"):
                self.assertFalse(hasattr(model, name), name)
            self.assertNotIn("SGLANG_MAMBA_CONV_DTYPE", os.environ)
            tensor = torch.tensor([1.])
            model.load_weights([("base.weight", tensor)])
            self.assertEqual(model.model.loaded[0][0], "base.weight")
            self.assertIs(model.model.loaded[0][1], tensor)
            fb = SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True))
            self.assertEqual(model.forward(None, None, fb), "stock-output")


if __name__ == "__main__":
    unittest.main()
