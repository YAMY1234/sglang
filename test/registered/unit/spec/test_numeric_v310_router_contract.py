"""Data-only router ABI; no CUDA or package import is required."""

import dataclasses
import importlib.util
import sys
import unittest
from pathlib import Path
from typing import get_args

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

_PATH = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/kernels/plugin_kernels/nvidia/numeric.py"
)
_spec = importlib.util.spec_from_file_location("_numeric_v310_router_contract", _PATH)
n = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = n
_spec.loader.exec_module(n)


class TestRouterQuantContract(unittest.TestCase):
    def test_distinct_operation_and_complete_unions(self):
        self.assertEqual(int(n.NumericOperation.ROUTER_QUANT), 43)
        self.assertEqual(
            len(n.NumericOperation), len({int(x) for x in n.NumericOperation})
        )
        self.assertIn(n.RouterQuantSignature, get_args(n.OperatorSignature))
        self.assertIn(n.RouterQuantIO, get_args(n.NumericIO))

    def test_operation_scoped_exact_signature_keys(self):
        args = (n.NumericOperation.ROUTER_QUANT, 4, 2560, 512, n.ScalarType.BF16)
        keys = [n.RouterQuantSignature(*args, variant=v) for v in (0, 1, 2)]
        self.assertEqual(len(set(keys)), 3)
        self.assertEqual(keys[0].variant, 0)
        self.assertNotEqual(keys[0], dataclasses.replace(keys[0], rows=1))
        self.assertEqual(
            [f.name for f in dataclasses.fields(keys[0])],
            [
                "operation",
                "rows",
                "hidden_size",
                "expert_count",
                "scalar_type",
                "variant",
            ],
        )

    def test_all_signature_fields_are_integer_serializable(self):
        sig = n.RouterQuantSignature(
            n.NumericOperation.ROUTER_QUANT, 8, 2560, 512, n.ScalarType.BF16, 2
        )
        for f in dataclasses.fields(sig):
            value = getattr(sig, f.name)
            self.assertIsInstance(value, int)
            self.assertNotIsInstance(value, bool)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            sig.variant = 1

    def test_io_borrows_exact_six_buffers(self):
        values = [object() for _ in range(6)]
        io = n.RouterQuantIO(*values)
        self.assertEqual(
            [f.name for f in dataclasses.fields(io)],
            [
                "hidden_states",
                "router_weight",
                "global_scale",
                "logits_out",
                "packed_out",
                "scale_out",
            ],
        )
        for f, value in zip(dataclasses.fields(io), values):
            self.assertIs(getattr(io, f.name), value)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            io.packed_out = object()


if __name__ == "__main__":
    unittest.main()
