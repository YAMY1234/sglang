"""DSpark aux means on the mHC stats stream match the inline mean bit for bit,
eagerly and inside a captured CUDA graph."""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="nightly", runner_config="1-gpu-large")

import unittest

import torch

from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestAuxMeanSideStreamGpu(CustomTestCase):
    def _check(self, rows):
        torch.manual_seed(rows)
        residual = torch.randn(rows, 4, 5120, device="cuda", dtype=torch.bfloat16)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        aux = mhc.mean_on_side_stream(residual, side)
        torch.cuda.current_stream().wait_stream(side)
        self.assertTrue(torch.equal(aux, residual.mean(dim=1)))

    def test_eager_rows(self):
        for rows in (1, 6, 16):
            self._check(rows)

    def test_graph_capture(self):
        rows = 6
        residual = torch.randn(rows, 4, 5120, device="cuda", dtype=torch.bfloat16)
        side = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        with torch.cuda.stream(capture_stream):
            # Warm the allocator and kernels outside capture.
            side.wait_stream(capture_stream)
            mhc.mean_on_side_stream(residual, side)
            capture_stream.wait_stream(side)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=capture_stream):
            side.wait_stream(torch.cuda.current_stream())
            aux = mhc.mean_on_side_stream(residual, side)
            torch.cuda.current_stream().wait_stream(side)
        residual.copy_(torch.randn_like(residual))
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(aux, residual.mean(dim=1)))


if __name__ == "__main__":
    unittest.main()
