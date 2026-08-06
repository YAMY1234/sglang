import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
    CuteDslFp4MoeQuantInfo,
    fused_experts_flashinfer_to_flashinfer_cutedsl_fp4,
)
from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
    FlashinferDispatchOutput,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlashinferA2ACuteDslContract(unittest.TestCase):
    def setUp(self):
        self.wrapper = Mock()
        self.wrapper.run.return_value = torch.ones((2, 4), dtype=torch.bfloat16)
        scalar = torch.tensor(1.0)
        weight = torch.empty((1, 1), dtype=torch.uint8)
        self.quant_info = CuteDslFp4MoeQuantInfo(
            wrapper=self.wrapper,
            w13_weight=weight,
            w2_weight=weight,
            w13_weight_sf=weight,
            w2_weight_sf=weight,
            w1_alpha=scalar,
            w2_alpha=scalar,
            fc2_input_scale=scalar,
            input_scale=scalar,
        )
        self.runner_config = MoeRunnerConfig(
            num_experts=4,
            num_local_experts=1,
            hidden_size=4,
            top_k=1,
        )
        self.topk_output = StandardTopKOutput(
            topk_weights=torch.ones((2, 1), dtype=torch.float32),
            topk_ids=torch.zeros((2, 1), dtype=torch.int32),
            router_logits=None,
        )

    def test_prefill_returns_standard_combine_format(self):
        hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=self.topk_output,
        )
        x_fp4 = torch.empty((2, 2), dtype=torch.uint8)
        x_sf = torch.empty((2, 1), dtype=torch.uint8)

        with patch(
            "sglang.srt.layers.quantization.fp4_utils.fp4_quantize",
            return_value=(x_fp4, x_sf),
        ):
            combine_input = fused_experts_flashinfer_to_flashinfer_cutedsl_fp4(
                dispatch_output, self.quant_info, self.runner_config
            )

        self.assertTrue(combine_input.format.is_standard())
        self.assertIs(self.wrapper.run.call_args.kwargs["x"], x_fp4)
        self.assertIs(self.wrapper.run.call_args.kwargs["x_sf"], x_sf)

    def test_decode_preserves_flashinfer_combine_format(self):
        x_fp4 = torch.empty((2, 2), dtype=torch.uint8)
        x_sf = torch.empty((2, 1), dtype=torch.uint8)
        dispatch_output = FlashinferDispatchOutput(
            hidden_states=x_fp4,
            hidden_states_scale=x_sf,
            topk_output=self.topk_output,
            moe_output=None,
        )

        combine_input = fused_experts_flashinfer_to_flashinfer_cutedsl_fp4(
            dispatch_output, self.quant_info, self.runner_config
        )

        self.assertTrue(combine_input.format.is_flashinfer())
        self.assertIs(self.wrapper.run.call_args.kwargs["x"], x_fp4)
        self.assertIs(self.wrapper.run.call_args.kwargs["x_sf"], x_sf)


if __name__ == "__main__":
    unittest.main()
