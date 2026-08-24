import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTextOnlyMropePositions(CustomTestCase):
    def test_extend_reuses_device_positions(self):
        positions = torch.tensor([5, 6, 7, 11, 12], dtype=torch.int64)
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=2,
            input_ids=torch.arange(5),
            req_pool_indices=torch.tensor([0, 1]),
            seq_lens=torch.tensor([8, 13]),
            out_cache_loc=torch.arange(5),
            seq_lens_sum=21,
            positions=positions,
            seq_lens_cpu=torch.tensor([8, 13]),
        )
        model_runner = SimpleNamespace(device=torch.device("cpu"))
        schedule_batch = SimpleNamespace(
            multimodal_inputs=[None, None],
            extend_lens=[3, 2],
            prefix_lens=[5, 11],
        )
        runtime = SimpleNamespace(
            deterministic=SimpleNamespace(rl_on_policy_target=None)
        )

        with patch(
            "sglang.srt.model_executor.forward_batch_info.get_exec",
            return_value=runtime,
        ):
            forward_batch._compute_mrope_positions_extend(model_runner, schedule_batch)

        expected = positions.unsqueeze(0).repeat(3, 1)
        torch.testing.assert_close(forward_batch.mrope_positions, expected)


if __name__ == "__main__":
    unittest.main()
