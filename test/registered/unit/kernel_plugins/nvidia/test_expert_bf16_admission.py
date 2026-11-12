"""Actual installed binding admission, meta tensors and CUDA metadata doubles.

No GPU/compiler/provider execution. A lookup spy records prepared=True/False.
"""

import os
import sys
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.kernel_plugins.nvidia import local_experts as impl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class MetadataTensor(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def tensor(shape, dtype):
    return torch.empty(shape, device="meta", dtype=dtype).as_subclass(MetadataTensor)


class TestActualAdmission(unittest.TestCase):
    def kwargs(self, rows, dtype):
        t = tensor
        return dict(
            hidden_states=t((rows, 1280), torch.uint8),
            gemm1_weights=t((512, 384, 1280), torch.uint8),
            gemm2_weights=t((512, 2560, 96), torch.uint8),
            hidden_states_scale=t((rows, 160), torch.float8_e4m3fn),
            routing_logits=t((rows, 512), dtype),
            gemm1_weights_scale=t((512, 61440), torch.float8_e4m3fn),
            gemm2_weights_scale=t((512, 30720), torch.float8_e4m3fn),
            output1_scale_gate_scalar=t((512,), torch.float32),
            output1_scale_scalar=t((512,), torch.float32),
            output2_scale_scalar=t((512,), torch.float32),
            intermediate_size=192,
            num_experts=512,
            local_num_experts=512,
            local_expert_offset=0,
            top_k=10,
            routing_method_type=4,
            activation_type=1,
            routing_bias=None,
            per_token_scale=None,
            routed_scaling_factor=1.0,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
        )

    def evaluate(self, rows, dtype, mode=None, batch=0, real_facade=False):
        records = []
        context = types.SimpleNamespace(moe_forward_mode=mode, moe_batch_size=batch)
        fake_context = types.ModuleType("sglang.srt.model_executor.forward_context")
        fake_context.has_forward_context = lambda: mode is not None
        fake_context.get_forward_context = lambda: context
        fake_fi = types.ModuleType("flashinfer.tllm_enums")
        fake_fi.ActivationType = types.SimpleNamespace(Swiglu=1)
        with (
            patch(
                "sglang.srt.runtime_context.get_spec",
                return_value=types.SimpleNamespace(speculative_num_draft_tokens=4),
            ),
            patch.dict(
                sys.modules,
                {
                    "flashinfer.tllm_enums": fake_fi,
                    "sglang.srt.model_executor.forward_context": fake_context,
                },
            ),
            patch.dict(
                os.environ,
                {
                    "SGLANG_Q38_MK_MOE": "1",
                    "SGLANG_Q38_MK_MOE_SF_PACK": "1",
                    "SGLANG_Q38_MKM_WIDE": "1",
                    "SGLANG_Q38_MKM_WIDE_BATCH34": "1",
                    "SGLANG_Q38_MKM_WIDE_M8": "0",
                },
            ),
        ):
            original = impl.numeric_operator

            def choose(signature, *, prepared):
                records.append((signature, prepared))
                return original(signature, prepared=prepared) if real_facade else None

            with patch.object(impl, "numeric_operator", side_effect=choose):
                result = impl.try_numeric_local_experts(self.kwargs(rows, dtype))
        self.assertIsNone(result)
        return records

    def test_bf16_raw_abi_admitted_before_registry(self):
        for rows in (1, 4):
            with self.subTest(rows=rows):
                records = self.evaluate(rows, torch.bfloat16)
                self.assertGreaterEqual(len(records), 1)
                self.assertTrue(records[0][1])

    def test_fp32_raw_pointer_layout_is_rejected(self):
        for rows in (1, 4):
            with self.subTest(rows=rows):
                records = self.evaluate(rows, torch.float32)
                self.assertGreaterEqual(len(records), 1)
                self.assertFalse(records[0][1])

    def test_wide_missing_context_rejected_before_facade(self):
        for rows in (12, 16):
            with self.subTest(rows=rows):
                self.assertTrue(
                    all(
                        sig.variant != 2
                        for sig, _ in self.evaluate(rows, torch.bfloat16)
                    )
                )

    def test_wide_real_phase_batch_recovers_same_bf16_admission(self):
        mode = types.SimpleNamespace(is_target_verify=lambda: True)
        for rows in (12, 16):
            with self.subTest(rows=rows):
                bf = self.evaluate(rows, torch.bfloat16, mode, rows // 4)
                fp = self.evaluate(rows, torch.float32, mode, rows // 4)
                self.assertEqual(bf[0][0].variant, 2)
                self.assertTrue(bf[0][1])
                self.assertFalse(fp[0][1])

    def test_fp32_never_queries_registry_and_bf16_does(self):
        from sglang.kernels.plugin_kernels.nvidia.numeric import numeric_registry

        for dtype, calls in ((torch.bfloat16, 1), (torch.float32, 0)):
            with (
                self.subTest(dtype=dtype),
                patch.object(numeric_registry, "get", return_value=None) as lookup,
            ):
                self.evaluate(4, dtype, real_facade=True)
                self.assertEqual(bool(lookup.call_count), bool(calls))


if __name__ == "__main__":
    unittest.main(verbosity=2)
