"""Borrowed BS1 publication ABI, no executable provider or CUDA."""

import dataclasses
import importlib.util
import sys
import unittest
from pathlib import Path
from typing import get_args

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

p = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/kernels/plugin_kernels/nvidia/numeric.py"
)
s = importlib.util.spec_from_file_location("_mtp_seed_contract", p)
n = importlib.util.module_from_spec(s)
sys.modules[s.name] = n
s.loader.exec_module(n)


class TestMtpCaptureSeedContract(unittest.TestCase):
    def key(self):
        return n.MtpCaptureSeedSignature(
            n.NumericOperation.MTP_CAPTURE_SEED,
            4,
            2051,
            2051,
            2,
            2055,
            2055,
            1,
            1,
            n.ScalarType.INT32,
            n.ScalarType.INT32,
            n.ScalarType.INT64,
            n.ScalarType.INT32,
            1,
        )

    def test_distinct_operation_and_unions(self):
        self.assertEqual(int(n.NumericOperation.MTP_CAPTURE_SEED), 7)
        self.assertIn(n.MtpCaptureSeedSignature, get_args(n.OperatorSignature))
        self.assertIn(n.MtpCaptureSeedIO, get_args(n.NumericIO))
        self.assertEqual(
            len(n.NumericOperation), len({int(x) for x in n.NumericOperation})
        )

    def test_original_seven_pointer_order_and_identity(self):
        v = [object() for _ in range(7)]
        io = n.MtpCaptureSeedIO(*v)
        self.assertEqual(
            [x.name for x in dataclasses.fields(io)],
            [
                "topk_indices",
                "extend_lengths",
                "accept_counts",
                "requests",
                "expanded_lengths",
                "seed_out",
                "captured_lengths_out",
            ],
        )
        self.assertEqual(
            [
                getattr(io, x.name) is value
                for x, value in zip(dataclasses.fields(io), v)
            ],
            [True] * 7,
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            io.seed_out = object()

    def test_metadata_keys_keep_dtype_stride_and_pdl_distinct(self):
        key = self.key()
        variants = [
            dataclasses.replace(key, **{f.name: getattr(key, f.name) + 1})
            for f in dataclasses.fields(key)
            if f.name != "operation"
        ]
        self.assertEqual(len({key, *variants}), len(variants) + 1)
        for f in dataclasses.fields(key):
            self.assertIsInstance(getattr(key, f.name), int)
            self.assertNotIsInstance(getattr(key, f.name), bool)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            key.pdl = 0

    def test_registry_borrows_callable_and_preserves_prior_contracts(self):
        key = self.key()
        calls = []
        offer = n.NumericOffer(key, lambda x: calls.append(x))
        reg = n.NumericRegistry()
        reg.register(offer)
        io = n.MtpCaptureSeedIO(*[object() for _ in range(7)])
        reg.get(key).run(io)
        self.assertIs(calls[0], io)
        self.assertIn(n.NativeGdnPdlIO, get_args(n.NumericIO))
        self.assertIn(n.RouterQuantIO, get_args(n.NumericIO))


if __name__ == "__main__":
    unittest.main()
