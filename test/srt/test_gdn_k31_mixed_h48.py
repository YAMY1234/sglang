"""CPU execution of the production layer packer and its TP1/TP2 admission.

Both frozen and candidate modules are imported, not AST-extracted. Their real
factorize_dense/k31 arithmetic runs on CPU. CPU small_eigh resolves to torch:
these tests prove packing, admission, and unchanged-path bytes, NOT CUDA Jacobi
accuracy, NLL, needle quality, or performance.
"""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

ROOT = Path(os.environ.get("PFACTOR075_SOURCE", Path(__file__).resolve().parents[2]))
BASE = Path(os.environ["PFACTOR075_BASE"])
sys.path.insert(0, str(ROOT / "python"))
pool = importlib.import_module("sglang.srt.mem_cache.gdn_factored_pool")
spec = importlib.util.spec_from_file_location(
    "_pfactor075_frozen_pool", BASE / "python/sglang/srt/mem_cache/gdn_factored_pool.py"
)
frozen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = frozen
spec.loader.exec_module(frozen)
assert Path(pool.__file__).resolve() == (ROOT / "python/sglang/srt/mem_cache/gdn_factored_pool.py").resolve()
TRACE = []


def digest(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def data(h, *, layers=36, batch=1, r=16, oversample=8):
    g = torch.Generator().manual_seed(75)
    # Nontrivial head/layer variation, with more content directions than r.
    left = torch.randn(layers, batch, h, 128, 32, generator=g)
    right = torch.randn(layers, batch, h, 32, 128, generator=g)
    state = (left * torch.logspace(0, -3, 32)) @ right
    vb = torch.randn(layers, h, 128, generator=g)
    omega = torch.randn(batch, h, 128, r + oversample, generator=g)
    return list(state.unbind(0)), vb, omega


def run(module, inputs, flag, **changes):
    cfg = module.FactoredGDNConfig(r=16, m=16, dtype=torch.float16,
                                  init_method="k31", init_oversample=8, decode_method="iter")
    for name, value in changes.items():
        setattr(cfg, name, value)
    states, vb, omega = inputs
    env = dict(os.environ)
    if flag is None:
        env.pop("SGLANG_GDN_K31_MIXED_EIGH", None)
    else:
        env["SGLANG_GDN_K31_MIXED_EIGH"] = str(flag)
    # wraps runs the REAL function and records the selected solver argument.
    with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
        module, "factorize_dense", wraps=module.factorize_dense
    ) as call:
        output = module.factorize_layers(states, vb, cfg, omega=omega)
    assert call.call_count == 1
    mixed = call.call_args.kwargs["mixed_eigh"]
    assert all(torch.isfinite(t).all() for row in output for t in row)
    hashes = [[digest(t) for t in row] for row in output]
    TRACE.append(dict(module=str(module.__file__), h=states[0].shape[1],
                      layers=len(states), batch=states[0].shape[0], flag=flag,
                      selected_mixed=mixed, hashes=hashes,
                      cpu_solver="torch.linalg.eigh (CUDA solver not executed)"))
    return hashes, mixed


class MixedH48Test(unittest.TestCase):
    def test_h24_two_modes_match_frozen_bytes(self):
        inp = data(24)
        for flag in (0, 1):
            with self.subTest(flag=flag):
                old, old_mixed = run(frozen, inp, flag)
                new, new_mixed = run(pool, inp, flag)
                self.assertEqual((new, new_mixed), (old, old_mixed))
                self.assertEqual(new_mixed, bool(flag))

    def test_h48_off_matches_frozen_bytes(self):
        inp = data(48)
        self.assertEqual(run(pool, inp, 0), run(frozen, inp, 0))

    def test_h48_on_admission(self):
        inp = data(48)
        _, old_mixed = run(frozen, inp, 1)
        _, new_mixed = run(pool, inp, 1)
        self.assertFalse(old_mixed)
        self.assertTrue(new_mixed, "h48 must reach the existing mixed solver")

    def test_unset_retains_h24_and_extends_only_h48(self):
        for h in (24, 48):
            with self.subTest(h=h):
                inp = data(h)
                implicit = run(pool, inp, None)
                explicit = run(pool, inp, 1)
                self.assertEqual(implicit, explicit)
                self.assertTrue(implicit[1])

    def test_other_geometry_guards_unchanged(self):
        cases = [
            (dict(h=12), {}),
            (dict(h=48, layers=35), {}),
            (dict(h=24, batch=2), {}),
            (dict(h=48, r=8), dict(r=8, m=8)),
            (dict(h=48), dict(dtype=torch.bfloat16)),
            (dict(h=48, oversample=4), dict(init_oversample=4)),
            (dict(h=48), dict(decode_method="warm")),
        ]
        for shape, config in cases:
            with self.subTest(shape=shape, config=str(config)):
                inp = data(**shape)
                old = run(frozen, inp, 1, **config)
                new = run(pool, inp, 1, **config)
                self.assertEqual(new, old)
                self.assertFalse(new[1])

    def test_h48_fp32_storage_admitted(self):
        _, selected = run(pool, data(48), 1, dtype=torch.float32)
        self.assertTrue(selected)


if __name__ == "__main__":
    torch.set_num_threads(2)
    suite = (unittest.TestSuite([MixedH48Test("test_h48_on_admission")])
             if os.environ.get("PFACTOR075_REVERSE_PROBE") == "1"
             else unittest.defaultTestLoader.loadTestsFromTestCase(MixedH48Test))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    out = Path(os.environ["PFACTOR075_UNIT_RECEIPT"])
    out.write_text(json.dumps(dict(passed=result.wasSuccessful(), tests=result.testsRun,
        skipped=len(result.skipped), failures=len(result.failures), errors=len(result.errors),
        cases=TRACE, gpu_executed=False), indent=2) + "\n")
    sys.exit(not result.wasSuccessful())
