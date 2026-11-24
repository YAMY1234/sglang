import unittest
from unittest import mock

import torch

from sglang.kernels.ops.moe import moe_front
from sglang.kernels.ops.moe.moe_front import NUM_EXPERTS, TOPK, fused_front
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_HIDDEN_SIZE = 7168


class TestMoeFront(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

    def test_moe_front(self):
        torch.manual_seed(4)
        num_tokens, latent_dim = 1, 128
        hidden = (
            torch.randn(
                num_tokens,
                _HIDDEN_SIZE,
                device="cuda",
                dtype=torch.bfloat16,
            )
            / 32
        )
        weight = (
            torch.randn(
                NUM_EXPERTS + latent_dim,
                _HIDDEN_SIZE,
                device="cuda",
                dtype=torch.bfloat16,
            )
            / 32
        )
        bias = torch.randn(NUM_EXPERTS, device="cuda")

        weights, ids, routed = fused_front(
            hidden,
            weight,
            bias,
            latent_dim,
            renormalize=True,
            routed_scaling_factor=2.5,
            apply_routed_scaling_factor_on_output=True,
        )
        merged = torch.mm(hidden, weight.t(), out_dtype=torch.float32)
        ref_weights, ref_ids = moe_fused_gate(
            merged[:, :NUM_EXPERTS],
            bias,
            topk=TOPK,
            scoring_func="sigmoid",
            renormalize=True,
            routed_scaling_factor=2.5,
            apply_routed_scaling_factor_on_output=True,
        )
        order = ids.argsort(dim=-1)
        ref_order = ref_ids.argsort(dim=-1)
        self.assertTrue(
            torch.equal(
                ids.gather(1, order),
                ref_ids.to(torch.int32).gather(1, ref_order),
            )
        )
        torch.testing.assert_close(
            weights.gather(1, order),
            ref_weights.gather(1, ref_order),
            rtol=1e-6,
            atol=0,
        )
        self.assertTrue(
            torch.equal(
                routed,
                merged[:, NUM_EXPERTS:].to(torch.bfloat16),
            )
        )

    def test_vr_epilogue_table_resolves_from_device_name(self):
        # VR reports "NVIDIA VR NVL72"; its table is built in (no JSON file), so a
        # broken lookup would fall back to the default silently.
        with (
            mock.patch.object(
                moe_front.torch.cuda, "get_device_name", return_value="NVIDIA VR NVL72"
            ),
            mock.patch.dict(moe_front._tables, clear=True),
        ):
            picks = {
                m: moe_front.get_config(
                    "epilogue", m, 0, moe_front.DEFAULT_EPILOGUE_CONFIG
                )
                for m in (384, 447, 448, 1024)
            }
        self.assertEqual(picks[384]["block_size"], 448)
        self.assertEqual(picks[447]["block_size"], 448)
        self.assertEqual(picks[448]["block_size"], 224)
        self.assertTrue(picks[1024]["cast_first"])


if __name__ == "__main__":
    unittest.main()
