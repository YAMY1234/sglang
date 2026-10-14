"""CPU pool lifecycle and actual Runtime decode method with Triton interpreter.

Run with TRITON_INTERPRET=1; native convolution is the existing CPU stand-in.
"""

import copy
import contextlib
from unittest.mock import patch
import sys
import types
import unittest
from pathlib import Path

import torch
from test_mamba2_factored import kernel, rel

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "test/registered/unit"))
from test_lightning_dense_bridge import load, runtime_class

Pool = load("mamba2_factored_pool").Mamba2FactoredStatePool
sys.modules["sglang.srt.layers.attention.linear.kernels.mamba2_factored"] = kernel


class FactorPoolTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.pool = Pool(
            4,
            (0,),
            torch.randn(1, 2, 64),
            n=128,
            rank=16,
            window=16,
            conv_dim=640,
            conv_width=3,
        )
        self.initial = torch.randn(2, 64, 128)
        self.pool.initialize(0, 1, self.initial, torch.randn(640, 3))

    def test_initial_product_lifecycle_and_successor(self):
        pool = self.pool
        coeff, left, right, _ = load("state_factor").factorize_left(
            self.initial[None], pool.directions[0], 16
        )
        expected = (
            pool.directions[0, :, :, None] * coeff[0, :, None, :] + left[0] @ right[0]
        )
        self.assertLess(rel(pool.materialize(0, 1), expected), 1e-5)
        self.assertEqual(pool.dense.numel(), 0)
        self.assertFalse(hasattr(pool, "warm"))
        self.assertFalse(hasattr(pool, "left"))
        slots = torch.tensor([1], dtype=torch.int32)
        for t in range(17):
            x, b, c = (
                torch.randn(1, 2, 64),
                torch.randn(1, 2, 128),
                torch.randn(1, 2, 128),
            )
            dt, decay, d = torch.rand(1, 2), torch.rand(1, 2), torch.randn(2)
            before = pool.materialize(0, 1).clone()
            expected = (
                before * decay[0, :, None, None]
                + (dt[0, :, None] * x[0])[..., None] * b[0, :, None, :]
            )
            y = kernel.step(
                x,
                b,
                c,
                dt,
                decay,
                d,
                pool.directions[0],
                pool.coeff[0],
                pool.u[0],
                pool.v[0],
                pool.fcount[0],
                slots,
            )
            ref = (expected @ c[0, ..., None]).squeeze(-1) + d[:, None] * x[0]
            self.assertLess(rel(y[0], ref), 1e-5)
            if t in (0, 14, 15, 16):
                saved = pool.get_cpu_slots(torch.tensor([1]))
                pool.copy_slots(torch.tensor([1]), torch.tensor([2]))
                pool.copy_slots(torch.tensor([1, 2]), torch.tensor([2, 1]))
                pool.reset_slots(torch.tensor([1]))
                pool.load_cpu_slots(saved, torch.tensor([1]))
                torch.testing.assert_close(
                    pool.materialize(0, 1), pool.materialize(0, 2), rtol=0, atol=0
                )
                broken = copy.deepcopy(saved)
                broken["fcount"].fill_(32)
                with self.assertRaises(ValueError):
                    pool.load_cpu_slots(broken, torch.tensor([1]))

    def test_byte_budget_matches_allocations(self):
        pool = self.pool
        shape = types.SimpleNamespace(temporal=(2, 64, 128), conv=[(640, 3)])
        expected = load("mamba2_factored_pool").per_slot_bytes(shape, 1, 16, 16)
        self.assertEqual(pool.resident_bytes(), expected * pool.size)
        with self.assertRaises(ValueError):
            Pool(
                4,
                (0,),
                pool.directions,
                n=128,
                rank=16,
                window=16,
                conv_dim=640,
                conv_width=3,
                native_dense=torch.zeros(1, 4, 2, 64, 128),
            )

    def test_runtime_factored_branch(self):
        runtime = runtime_class().__new__(runtime_class())
        runtime.factored_decode = True
        runtime.pool = self.pool
        runtime.decode_slots = torch.tensor([1], dtype=torch.int32)
        runtime.mamba_ids = (0,)
        runtime.factor_trunc_stream = types.SimpleNamespace(wait_stream=lambda _: None)
        runtime.geometry = types.SimpleNamespace(
            heads=2,
            head_dim=64,
            state_dim=128,
            groups=2,
            intermediate=128,
            bc_dim=256,
            conv_dim=640,
        )
        mixer = types.SimpleNamespace(
            dt_bias=torch.randn(2), A=-torch.rand(2), D=torch.randn(2),
        )
        runtime.body = types.SimpleNamespace(
            layers=[types.SimpleNamespace(mixer=mixer)]
        )
        runtime.components = types.SimpleNamespace(eps=1e-5)
        before = self.pool.dense.clone()
        for _ in range(17):
            out = torch.empty(1, 2, 64, dtype=torch.bfloat16)
            # CPU stand-in checks routing/lifecycle only; GPU ordering is
            # exercised by test_mamba2_factored_async.py.
            with patch.object(torch.cuda, "current_stream", return_value=None), patch.object(
                torch.cuda, "stream", return_value=contextlib.nullcontext()
            ):
                result = runtime.mamba_factor_update(
                    0, torch.randn(1, 2, 64).bfloat16(),
                    torch.randn(1, 2, 128).bfloat16(), torch.randn(1, 2, 128).bfloat16(),
                    torch.randn(1, 2).bfloat16(), out, runtime.decode_slots,
                )
            self.assertTrue(bool(torch.isfinite(result).all()))
        torch.testing.assert_close(self.pool.dense, before, rtol=0, atol=0)
        self.assertTrue(bool((self.pool.fcount[0, 1] == 17).all()))


if __name__ == "__main__":
    unittest.main()
