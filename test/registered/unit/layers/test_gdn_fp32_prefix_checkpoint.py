"""Execute the actual eager/graph checkpoint assignments without GPU imports."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/layers/attention/linear/gdn_backend.py"
)
TREE = ast.parse(SOURCE.read_text())
FORWARD = next(
    n
    for n in ast.walk(TREE)
    if isinstance(n, ast.FunctionDef) and n.name == "forward_extend"
)
ASSIGNMENTS = [
    n
    for n in ast.walk(FORWARD)
    if isinstance(n, ast.Assign)
    and any(
        ast.unparse(t) == "conv_states[forward_metadata.conv_states_mask_indices]"
        for t in n.targets
    )
]
assert len(ASSIGNMENTS) == 2


class PrefixCheckpointDTypeTest(unittest.TestCase):
    def check_storage(self, source_dtype, cache_dtype):
        torch.manual_seed(22)
        projected = torch.randn(8, 6, dtype=source_dtype)
        original = projected.clone()
        metadata = SimpleNamespace(
            conv_states_mask_indices=torch.tensor([1, 3]),
            track_conv_indices=torch.tensor([[0, 1, 2], [5, 6, 7]]),
        )
        tracked = projected.T[:, metadata.track_conv_indices].transpose(0, 1)
        expected = tracked.to(cache_dtype)
        for assignment in ASSIGNMENTS:
            graph_path = "mixed_qkv_to_track" not in ast.unparse(assignment.value)
            with self.subTest(path="graph" if graph_path else "eager"):
                cache = torch.zeros(4, 6, 3, dtype=cache_dtype)
                namespace = dict(
                    conv_states=cache,
                    forward_metadata=metadata,
                    mixed_qkv=projected if graph_path else projected.T,
                    mixed_qkv_to_track=tracked,
                )
                unit = ast.fix_missing_locations(
                    ast.Module(body=[assignment], type_ignores=[])
                )
                exec(compile(unit, str(SOURCE), "exec"), namespace)
                self.assertTrue(torch.equal(cache[[1, 3]], expected))
                self.assertEqual(cache.dtype, cache_dtype)
                self.assertEqual(torch.count_nonzero(cache[[0, 2]]).item(), 0)
                self.assertTrue(torch.equal(projected, original))

    def test_fp32_emitter_into_bf16_cache(self):
        self.check_storage(torch.float32, torch.bfloat16)

    def test_native_bf16_cache_is_unchanged(self):
        self.check_storage(torch.bfloat16, torch.bfloat16)

    def test_explicit_fp32_cache_preserves_precision(self):
        self.check_storage(torch.float32, torch.float32)


if __name__ == "__main__":
    unittest.main()
