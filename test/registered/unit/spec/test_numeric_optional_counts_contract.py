"""A caller can omit unconsumed counts without changing ordered-index IO."""

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
s = importlib.util.spec_from_file_location("_optional_counts_contract", p)
n = importlib.util.module_from_spec(s)
sys.modules[s.name] = n
s.loader.exec_module(n)


class TestOptionalCountsContract(unittest.TestCase):
    def test_omitted_counts_keep_all_other_borrowed_operands(self):
        values = [object() for _ in range(5)]
        io = n.SparseSelectIO(values[0], values[1], 512, values[2], None, values[3])
        self.assertIsNone(io.counts_out)
        self.assertIs(io.scores, values[0])
        self.assertIs(io.lengths, values[1])
        self.assertIs(io.selected_indices_out, values[2])
        self.assertIs(io.scratch_out, values[3])
        self.assertIn(n.SparseSelectIO, get_args(n.NumericIO))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            io.counts_out = object()

    def test_provided_counts_keep_original_pointer_and_field_order(self):
        values = [object() for _ in range(5)]
        io = n.SparseSelectIO(
            values[0], values[1], 512, values[2], values[3], values[4]
        )
        self.assertIs(io.counts_out, values[3])
        self.assertEqual(
            [f.name for f in dataclasses.fields(io)],
            [
                "scores",
                "lengths",
                "selected_count",
                "selected_indices_out",
                "counts_out",
                "scratch_out",
            ],
        )

    def test_annotation_is_only_optional_count_not_indices_or_scratch(self):
        hints = get_type_hints(n.SparseSelectIO, globalns={**vars(n), "Tensor": object})
        self.assertEqual(hints["counts_out"], object | None)
        self.assertIs(hints["selected_indices_out"], object)
        self.assertIs(hints["scratch_out"], object)


if __name__ == "__main__":
    unittest.main()
