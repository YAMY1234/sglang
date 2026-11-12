"""Narrow borrowed deferred-KV ABI; no provider, model or CUDA execution."""

import dataclasses
import importlib.util
import sys
import unittest
from pathlib import Path
from typing import get_args, get_type_hints

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

p = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/kernels/plugin_kernels/nvidia/numeric.py"
)
s = importlib.util.spec_from_file_location("_mtp_kv_contract", p)
n = importlib.util.module_from_spec(s)
sys.modules[s.name] = n
s.loader.exec_module(n)


class TestMtpDeferredKvContract(unittest.TestCase):
    def key(self):
        return n.MtpDeferredKvSignature(
            n.NumericOperation.MTP_DEFERRED_KV_PREPARE,
            1,
            1,
            256,
            n.ScalarType.BF16,
            2055,
            2112,
            4,
            7,
            4096,
            7,
            262144,
            262144,
            2112,
            2112,
            17,
        )

    def test_distinct_operation_and_complete_unions(self):
        self.assertEqual(int(n.NumericOperation.MTP_DEFERRED_KV_PREPARE), 8)
        self.assertIn(n.MtpDeferredKvSignature, get_args(n.OperatorSignature))
        self.assertIn(n.MtpDeferredKvIO, get_args(n.NumericIO))
        self.assertEqual(
            len(n.NumericOperation), len({int(x) for x in n.NumericOperation})
        )

    def test_all15_fields_borrow_identity_and_no_optional_counts(self):
        values = [object() for _ in range(15)]
        io = n.MtpDeferredKvIO(*values)
        self.assertEqual(
            [f.name for f in dataclasses.fields(io)],
            [
                "k_cache",
                "v_cache",
                "new_k",
                "new_v",
                "locations",
                "req_to_token",
                "row_requests",
                "sequence_lengths",
                "indices_pool",
                "lookup_requests",
                "captured_lengths",
                "lookup_positions",
                "counts_out",
                "packed_k_out",
                "packed_v_out",
            ],
        )
        for field, value in zip(dataclasses.fields(io), values):
            self.assertIs(getattr(io, field.name), value)
        hints = get_type_hints(
            n.MtpDeferredKvIO, globalns={**vars(n), "Tensor": object}
        )
        self.assertTrue(all(x is object for x in hints.values()))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            io.counts_out = None

    def test_exact_integer_dtype_masks_and_strides_separate_keys(self):
        key = self.key()
        self.assertEqual(
            len({dataclasses.replace(key, metadata_dtype_mask=x) for x in range(64)}),
            64,
        )
        for name in (
            "index_pool_rows",
            "cache_slots",
            "request_pool_rows",
            "map_columns",
            "map_row_stride",
            "new_k_row_stride",
            "new_v_row_stride",
        ):
            self.assertNotEqual(
                dataclasses.replace(key, **{name: getattr(key, name) + 1}), key
            )
        for field in dataclasses.fields(key):
            self.assertIsInstance(getattr(key, field.name), int)
            self.assertNotIsInstance(getattr(key, field.name), bool)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            key.kv_heads = 2

    def test_schema_has_no_object_plan_or_unused_qpos(self):
        names = {f.name for f in dataclasses.fields(n.MtpDeferredKvIO)}
        self.assertNotIn("query_positions", names)
        self.assertNotIn("plan", names)
        self.assertNotIn("backend", names)
        self.assertIn(n.MtpCaptureSeedIO, get_args(n.NumericIO))
        self.assertIn(n.NativeGdnPdlIO, get_args(n.NumericIO))
        self.assertIn(n.SparseSelectIO, get_args(n.NumericIO))


if __name__ == "__main__":
    unittest.main()
