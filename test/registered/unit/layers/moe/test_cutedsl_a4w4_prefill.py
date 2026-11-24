"""SGLANG_K31_PREFILL_MOE_A4W4 routed experts.

The direct path hands FlashInfer views of one persistent workspace sized per call (moe_sort outputs, the NVFP4 FC1
output and its MMA-layout scales); with the same tactic it must equal FlashInfer's functional API bit for bit.
"""

import unittest

import torch

from sglang.srt.layers.moe.moe_runner import cutedsl_a4w4_prefill as a4w4
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

_E, _H, _I = 8, 512, 256


def _mxfp4_experts(seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w13 = torch.randint(
        0, 256, (_E, 2 * _I, _H // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    w2 = torch.randint(
        0, 256, (_E, _H, _I // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    s13 = torch.randint(
        118,
        122,
        (_E, 2 * _I, _H // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=gen,
    )
    s2 = torch.randint(
        118, 122, (_E, _H, _I // 32), dtype=torch.uint8, device="cuda", generator=gen
    )
    return w13, s13, w2, s2


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "needs an SM10x GPU (FlashInfer CuTe DSL MoE)",
)
class TestCuteDslA4W4DirectPath(CustomTestCase):
    def test_direct_path_matches_functional_api_bitwise(self):
        from flashinfer import cute_dsl_fused_moe
        from flashinfer.fused_moe.cute_dsl.tuner import _get_default_tactic
        from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets
        from flashinfer.tllm_enums import ActivationType

        weights = a4w4.prepare_weights(*_mxfp4_experts(0), beta=4.0, lbeta=25.0)
        tactic = _get_default_tactic()
        max_tokens = 1024
        saved = a4w4._table, a4w4._workspace
        a4w4._table = a4w4._TacticTable(
            max_tokens=max_tokens,
            tactics={b: tactic for b in get_hybrid_num_tokens_buckets(max_tokens)},
        )
        a4w4._workspace = None
        self.addCleanup(lambda: setattr(a4w4, "_table", saved[0]))
        self.addCleanup(lambda: setattr(a4w4, "_workspace", saved[1]))
        gen = torch.Generator(device="cuda").manual_seed(1)
        # top_k 1: every output row takes one atomic add, so the finalize is deterministic.
        # Sizes cross the workspace's tile padding (128 rows) and reuse it at a smaller size afterwards.
        for num_tokens in (5, 300, 1024, 37):
            x = torch.randn(num_tokens, _H, device="cuda", generator=gen).bfloat16()
            ids = torch.randint(
                0, _E, (num_tokens, 1), device="cuda", generator=gen, dtype=torch.int32
            )
            w = torch.rand(num_tokens, 1, device="cuda", generator=gen) + 0.5
            direct = torch.empty(num_tokens, _H, dtype=torch.bfloat16, device="cuda")
            a4w4.run(weights, x, ids, w, output=direct)
            x_fp4, x_sf = a4w4._quantize_tokens(weights, x)
            functional = cute_dsl_fused_moe(
                x_fp4,
                x_sf,
                ids,
                w,
                weights.w13,
                weights.w13_sf,
                weights.w1_alpha,
                weights.unit_scale,
                weights.w2,
                weights.w2_sf,
                weights.w2_alpha,
                num_experts=_E,
                top_k=1,
                activation_type=ActivationType.Swiglu.value,
                situ_beta=weights.beta,
                situ_linear_beta=weights.lbeta,
                quant_mode="w4a4",
                tactic=tactic,
            )
            self.assertTrue(torch.equal(direct, functional), num_tokens)
            self.assertGreater(direct.float().abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
