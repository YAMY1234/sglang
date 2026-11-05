"""Additive numeric ABI: old callers remain valid; new outputs are explicit."""

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
_spec = importlib.util.spec_from_file_location("_numeric_v38_contract", _PATH)
n = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = n
_spec.loader.exec_module(n)


class TestNumericV38Contract(unittest.TestCase):
    def test_permuted_finalize_is_distinct_from_ordered_rows(self):
        self.assertEqual(int(n.NumericOperation.FINALIZE_PERMUTED_SHARED), 42)
        self.assertNotEqual(
            n.NumericOperation.FINALIZE_PERMUTED_SHARED,
            n.NumericOperation.FINALIZE_SHARED,
        )
        self.assertIn(n.FinalizePermutedSignature, get_args(n.OperatorSignature))
        self.assertIn(n.FinalizePermutedSharedIO, get_args(n.NumericIO))

    def test_permuted_finalize_keeps_seven_borrowed_tensor_identities(self):
        items = [object() for _ in range(7)]
        io = n.FinalizePermutedSharedIO(*items)
        self.assertEqual(len(dataclasses.fields(io)), 7)
        for f, value in zip(dataclasses.fields(io), items):
            self.assertIs(getattr(io, f.name), value)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            io.output = object()

    def test_permuted_finalize_exact_launch_and_weight_keys(self):
        base = n.FinalizePermutedSignature(
            n.NumericOperation.FINALIZE_PERMUTED_SHARED,
            32,
            2560,
            10,
            n.ScalarType.BF16,
            n.ScalarType.BF16,
        )
        keys = [
            dataclasses.replace(
                base, weight_scalar_type=weight, num_warps=warps, use_pdl=pdl
            )
            for weight in (n.ScalarType.BF16, n.ScalarType.FP32)
            for warps in (8, 16)
            for pdl in (0, 1)
        ]
        self.assertEqual(len(set(keys)), 8)
        self.assertIs(type(base.use_pdl), int)
        self.assertEqual(base.use_pdl, 1)
        self.assertEqual(base.variant, 0)
        self.assertNotEqual(base, dataclasses.replace(base, variant=1))

    def test_old_front_positional_constructor_remains_valid(self):
        inputs = [object() for _ in range(13)]
        io = n.GdnFrontIO(*inputs)
        self.assertIs(io.x, inputs[0])
        self.assertIs(io.intermediate_conv_window_out, inputs[-1])
        self.assertIsNone(io.packed_out)
        self.assertEqual(io.pad_slot, -1)

    def test_front_preserves_borrowed_output_and_explicit_pad(self):
        old = [object() for _ in range(13)]
        packed = object()
        io = n.GdnFrontIO(*old, packed_out=packed, pad_slot=-7)
        self.assertIs(io.packed_out, packed)
        self.assertEqual(io.pad_slot, -7)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            io.pad_slot = -1

    def test_core_raw_and_norm_are_distinct_registry_keys(self):
        args = (n.NumericOperation.GDN_CORE, 4, 4, 12, 128, 128, 4, n.ScalarType.BF16)
        raw = n.GdnSignature(*args, gate_scalar_type=n.ScalarType.BF16)
        norm = dataclasses.replace(raw, variant=1)
        self.assertNotEqual(raw, norm)
        self.assertEqual({raw: "raw", norm: "norm"}[raw], "raw")
        self.assertEqual({raw: "raw", norm: "norm"}[norm], "norm")
        self.assertEqual(raw.variant, 0)
        self.assertEqual(norm.variant, 1)

    def test_front_compile_families_do_not_collide(self):
        args = (n.NumericOperation.GDN_FRONT, 4, 4, 12, 128, 128, 4, n.ScalarType.BF16)
        keys = [n.GdnSignature(*args, variant=v) for v in n.GdnFrontVariant]
        self.assertEqual(len(set(keys)), 4)
        self.assertEqual([int(v) for v in n.GdnFrontVariant], [0, 1, 2, 3])

    def test_activation_values_match_source_norm_encoding(self):
        self.assertEqual(int(n.GdnActivation.NONE), 0)
        self.assertEqual(int(n.GdnActivation.SILU), 1)
        self.assertEqual(int(n.GdnActivation.SIGMOID), 2)
        self.assertEqual(int(n.GdnDotPrecision.TF32), 0)
        with self.assertRaises(ValueError):
            n.GdnActivation(99)


if __name__ == "__main__":
    unittest.main()
