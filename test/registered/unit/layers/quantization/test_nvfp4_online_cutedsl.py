import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
    resolve_cutedsl_standard_scales,
)
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    MoeA2ABackend,
    MoeRunnerBackend,
)
from sglang.srt.layers.quantization.nvfp4_online import (
    ModelOptNvFp4OnlineFusedMoEMethod,
    NvFp4OnlineConfig,
)


class TestNvFp4OnlineCuteDsl(CustomTestCase):
    def _verify_quantization(
        self, requested: str, detected: str, *, is_draft_model: bool = True
    ) -> str:
        config = ModelConfig.__new__(ModelConfig)
        config.quantization = requested
        config.is_draft_model = is_draft_model
        config._parse_quant_hf_config = lambda: {
            "quant_method": detected,
            "quant_algo": "NVFP4",
            "ignore": ["mtp.layers.0*"],
        }
        config._find_quant_modelslim_config = lambda: None

        with patch("sglang.srt.layers.deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0", False):
            config._verify_quantization()

        return config.quantization

    def test_explicit_online_quantization_survives_modelopt_draft_config(self):
        self.assertEqual(
            self._verify_quantization("nvfp4_online", "modelopt"),
            "nvfp4_online",
        )

    def test_explicit_online_survives_normalized_modelopt_fp4_config(self):
        # ModelOpt metadata is normalized before this verifier runs.
        self.assertEqual(
            self._verify_quantization("nvfp4_online", "modelopt_fp4"),
            "nvfp4_online",
        )

    def test_main_serialized_fp4_does_not_use_online_conversion(self):
        self.assertEqual(
            self._verify_quantization(
                "nvfp4_online", "modelopt_fp4", is_draft_model=False
            ),
            "modelopt_fp4",
        )

    def test_generic_modelopt_still_auto_detects_serialized_fp4(self):
        self.assertEqual(
            self._verify_quantization("modelopt", "modelopt_fp4"),
            "modelopt_fp4",
        )

    @contextmanager
    def _make_method(
        self,
        backend: MoeRunnerBackend,
        a2a_backend: MoeA2ABackend = MoeA2ABackend.NONE,
        deepep_mode: DeepEPMode = DeepEPMode.LOW_LATENCY,
    ):
        with (
            patch(
                "sglang.srt.layers.quantization.modelopt_quant."
                "get_moe_runner_backend",
                return_value=backend,
            ),
            patch(
                "sglang.srt.layers.moe.get_moe_runner_backend",
                return_value=backend,
            ),
            patch(
                "sglang.srt.layers.moe.get_moe_a2a_backend",
                return_value=a2a_backend,
            ),
            patch(
                "sglang.srt.layers.moe.get_deepep_mode",
                return_value=deepep_mode,
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant."
                "is_blackwell_supported",
                return_value=True,
            ),
        ):
            yield ModelOptNvFp4OnlineFusedMoEMethod(
                NvFp4OnlineConfig(), "model.layers.0.mlp.experts"
            )

    def test_cutedsl_runner_is_accepted(self):
        with self._make_method(
            MoeRunnerBackend.FLASHINFER_CUTEDSL,
            MoeA2ABackend.FLASHINFER,
        ) as method:
            self.assertTrue(method.supports_nvfp4_online_moe)
            self.assertFalse(method.enable_flashinfer_trtllm_moe)

    def test_cutedsl_deepep_low_latency_is_accepted(self):
        with self._make_method(
            MoeRunnerBackend.FLASHINFER_CUTEDSL,
            MoeA2ABackend.DEEPEP,
            DeepEPMode.LOW_LATENCY,
        ) as method:
            self.assertTrue(method.supports_nvfp4_online_moe)

    def test_cutedsl_deepep_auto_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires --deepep-mode low_latency"):
            with self._make_method(
                MoeRunnerBackend.FLASHINFER_CUTEDSL,
                MoeA2ABackend.DEEPEP,
                DeepEPMode.AUTO,
            ):
                pass

    def test_cutedsl_deepep_normal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires --deepep-mode low_latency"):
            with self._make_method(
                MoeRunnerBackend.FLASHINFER_CUTEDSL,
                MoeA2ABackend.DEEPEP,
                DeepEPMode.NORMAL,
            ):
                pass

    def test_trtllm_runner_remains_accepted(self):
        with self._make_method(MoeRunnerBackend.FLASHINFER_TRTLLM) as method:
            self.assertTrue(method.supports_nvfp4_online_moe)
            self.assertTrue(method.enable_flashinfer_trtllm_moe)

    def test_unrelated_runner_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires --moe-runner-backend"):
            with self._make_method(MoeRunnerBackend.TRITON):
                pass

    def test_online_converter_rejects_packed_integer_source(self):
        with self.assertRaisesRegex(ValueError, "floating-point source"):
            ModelOptNvFp4OnlineFusedMoEMethod._quantize_weight_nvfp4(
                torch.zeros((2, 16), dtype=torch.uint8)
            )

    def test_online_cutedsl_scalar_scale_contract(self):
        class Layer:
            pass

        layer = Layer()
        layer.num_experts = 2
        layer.num_local_experts = 2
        layer.moe_ep_rank = 0
        layer.g1_alphas = torch.tensor([0.25, 0.5], dtype=torch.float32)
        layer.g2_alphas = torch.tensor([0.75, 1.0], dtype=torch.float32)
        layer.w13_weight_scale_2 = torch.tensor(
            [[0.25, 0.25], [0.5, 0.5]], dtype=torch.float32
        )
        layer.w2_weight_scale_2 = torch.tensor([0.75, 1.0], dtype=torch.float32)
        layer.w13_input_scale_quant = torch.ones(2, dtype=torch.float32)
        layer.w2_input_scale_quant = torch.ones(2, dtype=torch.float32)

        w1_alpha, a2_scale, w2_alpha, a1_scale = resolve_cutedsl_standard_scales(layer)

        torch.testing.assert_close(a1_scale, torch.ones(1))
        torch.testing.assert_close(a2_scale, torch.ones(1))
        torch.testing.assert_close(w1_alpha, layer.w13_weight_scale_2[:, 0])
        torch.testing.assert_close(w2_alpha, layer.w2_weight_scale_2)


if __name__ == "__main__":
    unittest.main()
