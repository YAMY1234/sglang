"""GPU regression for the optional Q/metadata gather; runnable before model load."""
import json
import unittest

import torch

from sglang.srt.layers.attention.triton_ops.trtllm_split_gather import gather_split_inputs


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSplitGather(unittest.TestCase):
    def assert_bits_equal(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)))

    def make_inputs(self, batch, qlen, pages, dtype, strided):
        step = 2 if strided else 1
        query = torch.randn(batch, qlen, 32, 256 * step, device="cuda", dtype=torch.bfloat16).to(dtype)[..., ::step]
        table = torch.randint(0, 100000, (batch, pages * step), device="cuda", dtype=torch.int32)[:, ::step]
        lens = torch.randint(1, 262144, (batch * step,), device="cuda", dtype=torch.int32)[::step]
        return query, table, lens

    def check(self, inputs, indices):
        outputs = gather_split_inputs(*inputs, indices)
        for actual, source in zip(outputs, inputs):
            self.assert_bits_equal(actual, source.index_select(0, indices))
        return outputs

    def test_representative_shapes_and_strides(self):
        torch.manual_seed(42)
        cases = [
            (1, 1, 1, torch.float8_e4m3fn, False),
            (3, 7, 17, torch.bfloat16, True),
            (36, 7, 4096, torch.float8_e4m3fn, False),
            (58, 7, 4096, torch.float8_e4m3fn, True),
            (64, 8, 4096, torch.float8_e4m3fn, False),
            (117, 8, 4096, torch.float8_e4m3fn, False),
            (4, 1, 8, torch.bfloat16, False),
            (53, 1, 4096, torch.bfloat16, True),
        ]
        for case in cases:
            with self.subTest(case=case):
                inputs = self.make_inputs(*case)
                order = torch.argsort(inputs[2])
                for indices in torch.tensor_split(order, min(4, case[0])):
                    self.check(inputs, indices)

    def test_graph_replay_updates_indices_and_inputs(self):
        inputs = self.make_inputs(56, 8, 4096, torch.float8_e4m3fn, False)
        indices = torch.randperm(56, device="cuda")[:14].clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                gather_split_inputs(*inputs, indices)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = gather_split_inputs(*inputs, indices)
        for _ in range(3):
            indices.copy_(torch.randperm(56, device="cuda")[:14])
            inputs[1].random_(0, 100000)
            inputs[2].random_(1, 262144)
            graph.replay()
            for actual, source in zip(outputs, inputs):
                self.assert_bits_equal(actual, source.index_select(0, indices))

    def test_duplicate_and_empty_indices_and_fp8_bits(self):
        inputs = self.make_inputs(4, 1, 256, torch.float8_e4m3fn, False)
        # Exercise every FP8 bit pattern, including signed zero/NaNs.
        inputs[0].view(torch.uint8).copy_(torch.arange(inputs[0].numel(), device="cuda").to(torch.uint8).view_as(inputs[0]))
        self.check(inputs, torch.tensor([3, 1, 1, 0], dtype=torch.int32, device="cuda"))
        self.check(inputs, torch.empty(0, dtype=torch.int64, device="cuda"))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSplitGather)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"split_gather_tests": result.testsRun, "passed": result.wasSuccessful(),
                      "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped)}))
    raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
