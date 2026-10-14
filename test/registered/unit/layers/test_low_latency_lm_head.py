"""Low-M LM head: which calls take FlashInfer cute-dsl mm_bf16, and with what operands."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from sglang.srt.layers import low_latency_lm_head as ll
from sglang.test.test_utils import CustomTestCase


def _head(tagged=True, dtype=torch.bfloat16):
    head = nn.Module()
    head.weight = nn.Parameter(torch.zeros(64, 32, dtype=dtype))
    if tagged:
        head.low_latency_lm_head = True
    return head


class TestLowLatencyLmHead(CustomTestCase):
    def setUp(self):
        self.mm_bf16 = MagicMock(return_value="ll")
        fake = types.ModuleType("flashinfer")
        fake.mm_bf16 = self.mm_bf16
        for patcher in (
            patch.dict(sys.modules, {"flashinfer": fake}),
            patch.object(ll, "_use_pdl", return_value=True),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_eligible_call_passes_column_major_weight(self):
        head = _head()
        x = torch.zeros(6, 32, dtype=torch.bfloat16)
        self.assertEqual(ll.maybe_low_latency_lm_head(x, head), "ll")
        args, kwargs = self.mm_bf16.call_args
        # Detached views of the operands (DLPack refuses tensors that require grad).
        self.assertEqual(args[0].data_ptr(), x.data_ptr())
        self.assertFalse(args[1].requires_grad)
        # mm_bf16 wants B as (K, N) column-major: the (N, K) weight transposed, not copied.
        self.assertEqual(args[1].shape, (32, 64))
        self.assertTrue(args[1].T.is_contiguous())
        self.assertEqual(kwargs, {"pdl": True, "backend": "cute-dsl"})

    def test_ineligible_calls_fall_back(self):
        bf16 = torch.bfloat16
        cases = {
            "untagged": (_head(tagged=False), torch.zeros(6, 32, dtype=bf16)),
            "17 rows": (_head(), torch.zeros(17, 32, dtype=bf16)),
            "fp32 hidden": (_head(), torch.zeros(6, 32)),
            "fp32 weight": (_head(dtype=torch.float32), torch.zeros(6, 32)),
            "non-contiguous": (_head(), torch.zeros(32, 6, dtype=bf16).T),
            "3-D": (_head(), torch.zeros(1, 6, 32, dtype=bf16)),
        }
        for name, (head, x) in cases.items():
            with self.subTest(name):
                self.assertIsNone(ll.maybe_low_latency_lm_head(x, head))
        self.mm_bf16.assert_not_called()

    def test_autotune_covers_every_bucket_up_to_16(self):
        ll.autotune_low_latency_lm_head(_head())
        rows = [call.args[0].shape[0] for call in self.mm_bf16.call_args_list]
        self.assertEqual(rows, [1, 2, 4, 8, 16])


if __name__ == "__main__":
    unittest.main()
