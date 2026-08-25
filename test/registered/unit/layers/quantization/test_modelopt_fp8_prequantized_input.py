import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import layernorm
from sglang.srt.layers.quantization import fp8_utils, modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestModelOptFp8PrequantizedInput(CustomTestCase):
    @staticmethod
    def _method(*, use_marlin=False, cutlass_fp8_supported=True):
        method = object.__new__(ModelOptFp8LinearMethod)
        method.use_marlin = use_marlin
        method.cutlass_fp8_supported = cutlass_fp8_supported
        return method

    def test_static_scale_helper_recognizes_modelopt_fp8(self):
        method = self._method()
        scale = torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)
        linear = SimpleNamespace(quant_method=method, input_scale=scale)

        self.assertIs(layernorm._fp8_static_input_scale(linear), scale)

        method.use_marlin = True
        self.assertIsNone(layernorm._fp8_static_input_scale(linear))

    def test_modelopt_apply_passes_prequantized_tuple_to_scaled_mm(self):
        method = self._method()
        layer = SimpleNamespace(
            use_flashinfer_bmm=False,
            weight=torch.empty((4, 3), dtype=torch.float8_e4m3fn),
            weight_scale=torch.tensor(0.5),
            input_scale=torch.tensor(0.25),
        )
        qx = torch.empty((2, 4), dtype=torch.float8_e4m3fn)
        x_scale = torch.tensor(0.25)
        expected = torch.empty((2, 3), dtype=torch.bfloat16)

        with patch.object(
            modelopt_quant, "apply_fp8_linear", return_value=expected
        ) as apply_fp8:
            actual = method.apply(layer, (qx, x_scale, torch.bfloat16))

        self.assertIs(actual, expected)
        kwargs = apply_fp8.call_args.kwargs
        self.assertIs(kwargs["input"], qx)
        self.assertIs(kwargs["input_scale"], x_scale)
        self.assertIs(kwargs["pre_quant_output_dtype"], torch.bfloat16)

    def test_flashinfer_bmm_skips_requantizing_tuple(self):
        qx = torch.empty((2, 4), dtype=torch.float8_e4m3fn)
        x_scale = torch.tensor(0.25)
        weight = torch.empty((4, 3), dtype=torch.float8_e4m3fn)
        expected = torch.empty((2, 3), dtype=torch.bfloat16)

        with patch.object(fp8_utils, "static_quant_fp8") as static_quant, patch.object(
            fp8_utils,
            "flashinfer_bmm_fp8",
            return_value=expected,
            create=True,
        ) as bmm:
            actual = fp8_utils.apply_fp8_linear_bmm_flashinfer(
                (qx, x_scale, torch.bfloat16),
                weight,
                torch.tensor(0.5),
                torch.tensor(0.25),
            )

        self.assertIs(actual, expected)
        static_quant.assert_not_called()
        self.assertIs(bmm.call_args.args[0], qx)
        self.assertIs(bmm.call_args.args[2], x_scale)
        self.assertIs(bmm.call_args.args[4], torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
