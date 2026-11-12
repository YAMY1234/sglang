"""Native strided ReplaySSM ABI; no CUDA objects or executable provider."""

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
s = importlib.util.spec_from_file_location("_native_gdn_pdl_contract", p)
n = importlib.util.module_from_spec(s)
sys.modules[s.name] = n
s.loader.exec_module(n)


class TestNativeGdnPdlContract(unittest.TestCase):
    def signature(self):
        return n.NativeGdnPdlSignature(
            n.NumericOperation.GDN_NATIVE_PDL,
            32,
            16,
            48,
            128,
            128,
            4,
            n.ScalarType.BF16,
        )

    def test_distinct_numeric_operation_and_complete_unions(self):
        self.assertEqual(int(n.NumericOperation.GDN_NATIVE_PDL), 33)
        self.assertNotEqual(
            n.NumericOperation.GDN_NATIVE_PDL, n.NumericOperation.GDN_CORE
        )
        self.assertIn(n.NativeGdnPdlSignature, get_args(n.OperatorSignature))
        self.assertIn(n.NativeGdnPdlIO, get_args(n.NumericIO))
        self.assertEqual(
            len(n.NumericOperation), len({int(x) for x in n.NumericOperation})
        )

    def test_original21_borrowed_pointers_have_distinct_offsets_and_cursors(self):
        values = [object() for _ in range(21)]
        io = n.NativeGdnPdlIO(*values)
        self.assertEqual(
            [f.name for f in dataclasses.fields(io)],
            [
                "q",
                "k",
                "v",
                "a",
                "b",
                "a_log",
                "dt_bias",
                "out",
                "checkpoint_state",
                "d_cache",
                "k_cache",
                "g_cache",
                "rawv_cache",
                "rawk_cache",
                "beta_cache",
                "query_start_loc",
                "ssm_state_indices",
                "replay_indices",
                "write_pos",
                "cache_base",
                "is_flush",
            ],
        )
        for f, v in zip(dataclasses.fields(io), values):
            self.assertIs(getattr(io, f.name), v)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            io.out = object()

    def test_only_three_original_ring_extras_are_optional(self):
        hints = get_type_hints(n.NativeGdnPdlIO, globalns={**vars(n), "Tensor": object})
        self.assertEqual(
            {k for k, v in hints.items() if v == object | None},
            {"rawv_cache", "rawk_cache", "beta_cache"},
        )
        values = [object() for _ in range(21)]
        for indexes in [(12, 13, 14), (14,), ()]:
            frame = values.copy()
            for i in indexes:
                frame[i] = None
            io = n.NativeGdnPdlIO(*frame)
            for f, v in zip(dataclasses.fields(io), frame):
                self.assertIs(getattr(io, f.name), v)

    def test_source_metadata_flags_and_modes_are_distinct_integer_keys(self):
        base = self.signature()
        keys = {
            dataclasses.replace(
                base,
                storage_mode=mode,
                wait_for_gate_producer=w,
                state_prefetch=p,
                dot_precision=d,
                state_scalar_type=st,
                dt_bias_scalar_type=dt,
            )
            for mode in (0, 1, 2)
            for w in (0, 1)
            for p in (0, 1)
            for d in (0, 2)
            for st in (1, 3)
            for dt in (1, 3)
        }
        self.assertEqual(len(keys), 96)
        for key in keys:
            for f in dataclasses.fields(key):
                self.assertIsInstance(getattr(key, f.name), int)
                self.assertNotIsInstance(getattr(key, f.name), bool)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            base.rows = 64

    def test_source_launch_defaults_and_storage_keys_keep_old_contracts(self):
        sig = self.signature()
        self.assertEqual(
            (sig.tokens_per_request, sig.cache_len, sig.gate_scalar_type), (4, 16, 1)
        )
        self.assertEqual(sig.storage_mode, 1)
        storage_keys = tuple(
            dataclasses.replace(sig, storage_mode=mode) for mode in (0, 1, 2)
        )
        self.assertEqual(tuple(key.storage_mode for key in storage_keys), (0, 1, 2))
        self.assertEqual(len(set(storage_keys)), 3)
        self.assertIn(n.GdnCoreIO, get_args(n.NumericIO))
        self.assertIn(n.GdnVerifyGlueIO, get_args(n.NumericIO))


if __name__ == "__main__":
    unittest.main()
