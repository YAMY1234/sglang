import unittest
from unittest.mock import patch

import torch

import sglang.srt.layers.layernorm as layernorm_module
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestGemmaRMSNormAllReduceGroup(unittest.TestCase):
    def test_forwards_requested_parallel_group(self):
        layer = GemmaRMSNorm(hidden_size=8)
        hidden_states = torch.zeros(2, 8)
        residual = torch.ones_like(hidden_states)

        for use_attn_tp_group in (False, True):
            expected = object()
            with self.subTest(use_attn_tp_group=use_attn_tp_group), patch.object(
                layernorm_module,
                "_forward_with_allreduce_fusion",
                return_value=expected,
            ) as fused:
                actual = layer.forward_with_allreduce_fusion(
                    hidden_states,
                    residual,
                    use_attn_tp_group=use_attn_tp_group,
                )

                self.assertIs(actual, expected)
                args = fused.call_args.args
                self.assertIs(args[0], layer)
                self.assertIs(args[1], hidden_states)
                self.assertIs(args[2], residual)
                self.assertIsNone(args[3])
                self.assertIs(args[4], layer.gemma_weight)
                self.assertEqual(
                    fused.call_args.kwargs, {"use_attn_tp_group": use_attn_tp_group}
                )


if __name__ == "__main__":
    unittest.main()
