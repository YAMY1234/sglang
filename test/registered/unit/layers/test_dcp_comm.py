import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.dcp import comm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDCPComm(unittest.TestCase):
    def test_mha_merge_allocates_output_in_symmetric_memory(self):
        active = False
        allocation_contexts = []
        original_nan_to_num = torch.nan_to_num

        @contextmanager
        def symmetric_memory(_group):
            nonlocal active
            active = True
            try:
                yield
            finally:
                active = False

        def tracked_nan_to_num(*args, **kwargs):
            allocation_contexts.append(active)
            return original_nan_to_num(*args, **kwargs)

        group = SimpleNamespace(
            world_size=2,
            rank_in_group=0,
            all_reduce=lambda tensor: tensor,
        )
        attn_out = torch.ones((2, 4, 8), dtype=torch.float32)
        attn_lse = torch.zeros((2, 4), dtype=torch.float32)

        with (
            patch.object(comm, "_ag_lse", return_value=torch.stack([attn_lse] * 2)),
            patch.object(comm, "use_symmetric_memory", symmetric_memory),
            patch.object(comm.torch, "nan_to_num", tracked_nan_to_num),
        ):
            output = comm.cp_lse_ag_out_rs_mha(attn_out, attn_lse, group)

        self.assertEqual(allocation_contexts, [False, True])
        self.assertEqual(output.shape, (2, 2, 8))


if __name__ == "__main__":
    unittest.main()
