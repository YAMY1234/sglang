"""copy_mla_rows_into_pack gathers each layer's rows into a contiguous pack. Row blocks
run on grid x, so a chunk past 65,535 KiB per layer (past the grid-y limit) packs
in one launch and must match a plain torch gather byte for byte.
"""

import unittest

import torch

from sglang.kernels.ops.kvcache.pd_dcp_gather import copy_mla_rows_into_pack
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


class TestCopyMlaRowsIntoPack(CustomTestCase):
    def _check(self, n, widths, pool_rows):
        gen = torch.Generator(device="cuda").manual_seed(0)
        srcs = [
            torch.randint(0, 256, (pool_rows, w), dtype=torch.uint8, device="cuda", generator=gen)
            for w in widths
        ]
        rows = torch.randperm(pool_rows, device="cuda", generator=gen)[:n].to(torch.int64)
        pack = torch.zeros(n * sum(widths), dtype=torch.uint8, device="cuda")
        copy_mla_rows_into_pack([s.data_ptr() for s in srcs], rows, pack, widths)
        offset = 0
        for src, w in zip(srcs, widths):
            got = pack[offset : offset + n * w].view(n, w)
            self.assertTrue(torch.equal(got, src.index_select(0, rows)))
            offset += n * w

    def test_small_chunk(self):
        self._check(1_000, [576, 576, 132], 4_096)

    def test_chunk_past_grid_y_limit(self):
        # 120,000 x 576 B = 67,500 blocks of 1 KiB per layer (> 65,535)
        self._check(120_000, [576, 576, 132], 160_000)


if __name__ == "__main__":
    unittest.main()
