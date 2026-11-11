import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# Rows above 16 take FlashInfer's fused single-cluster routing (the math the fused
# kernel reproduces) only with the block-per-token split forced off, as the recipes
# set it; FlashInfer reads this once per process.
os.environ["FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN"] = "0"

import torch

# isort: off
# mxfp4 first: importing flashinfer_trtllm first hits a circular import.
from sglang.srt.layers.quantization import mxfp4  # noqa: F401
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm

# isort: on
from sglang.kernels.ops.moe import q38_fused_routing as op
from sglang.srt import runtime_context
from sglang.srt.layers.moe.moe_runner import q38_fused_routing
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import BypassedTopKOutput, TopKConfig
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HIDDEN = op.HIDDEN_SIZE
NUM_EXPERTS = op.NUM_EXPERTS
TOPK = op.TOP_K
INTERMEDIATE = 512
# 1-32 rows run the single-CTA kernel, 33-512 the cluster kernel (high-concurrency verify).
ROWS = (1, 2, 4, 8, 16, 32, 33, 128, 512)


def _make_quant_info(seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def u8(*shape: int, lo: int = 0, hi: int = 256) -> torch.Tensor:
        return torch.randint(
            lo, hi, shape, generator=g, device="cuda", dtype=torch.uint8
        )

    # Random NVFP4 bytes: both paths read the same weights, so parity needs no
    # meaningful values. E4M3 block scales of 2^-3 .. 2^0 keep the GEMMs finite.
    w13_scale = u8(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 16, lo=0x20, hi=0x39)
    w2_scale = u8(NUM_EXPERTS, HIDDEN, INTERMEDIATE // 16, lo=0x20, hi=0x39)

    def per_expert(value: float) -> torch.Tensor:
        return torch.full((NUM_EXPERTS,), value, dtype=torch.float32, device="cuda")

    return flashinfer_trtllm.FlashInferTrtllmFp4MoeQuantInfo(
        w13_weight=u8(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2),
        w2_weight=u8(NUM_EXPERTS, HIDDEN, INTERMEDIATE // 2),
        w13_weight_scale=w13_scale.view(torch.float8_e4m3fn),
        w2_weight_scale=w2_scale.view(torch.float8_e4m3fn),
        g1_scale_c=per_expert(1 / 256),
        g1_alphas=per_expert(1 / 256),
        g2_alphas=per_expert(1 / 256),
        w13_input_scale_quant=torch.tensor([64.0], device="cuda"),
        global_num_experts=NUM_EXPERTS,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
        intermediate_size_per_partition=INTERMEDIATE,
        routing_method_type=RoutingMethodType.RenormalizeNaive,
    )


def _run_moe(x, logits, quant_info, *, fused: bool):
    dispatch = StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=BypassedTopKOutput(x, logits, TopKConfig(top_k=TOPK)),
    )
    with (
        patch.object(q38_fused_routing, "_ENABLED", fused),
        patch.object(q38_fused_routing, "_MAX_ROWS", op.MAX_ROWS),
        flashinfer_trtllm.flashinfer_trtllm_deferred_finalize_context(True),
    ):
        out = flashinfer_trtllm.fused_experts_none_to_flashinfer_trtllm_fp4(
            dispatch, quant_info, MoeRunnerConfig(activation="silu")
        ).hidden_states
    torch.cuda.synchronize()
    return out


class TestQ38FusedRouting(CustomTestCase):
    """The fused routing plus FlashInfer's prepared-routing body must stay bitwise
    equal to the unfused deferred-finalize call, so a FlashInfer bump that changes
    the runner kwargs, tuning config, default tile or PDL default fails here."""

    def setUp(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("trtllm-gen NVFP4 MoE needs SM10x")
        tp1 = SimpleNamespace(tp_size=1)
        for target in (
            patch.object(runtime_context, "get_parallel", lambda: tp1),
            patch.object(flashinfer_trtllm, "get_parallel", lambda: tp1),
            patch.object(
                flashinfer_trtllm,
                "get_forward",
                lambda: SimpleNamespace(sp_active=False),
            ),
        ):
            target.start()
            self.addCleanup(target.stop)

    def _assert_same(self, out, ref):
        self.assertTrue(
            torch.equal(out.expert_weights.float(), ref.expert_weights.float())
        )
        out_perm = out.expanded_idx_to_permuted_idx.view(-1, TOPK).long()
        ref_perm = ref.expanded_idx_to_permuted_idx.view(-1, TOPK).long()
        self.assertTrue(bool((ref_perm >= 0).all()))
        self.assertTrue(bool((out_perm >= 0).all()))
        # Only routed rows are written: the padding rows of each expert's tile in
        # gemm2_out are uninitialized scratch, which finalize never reads.
        self.assertTrue(bool(torch.isfinite(ref.gemm2_out[ref_perm]).all()))
        # Rows inside an expert may be ordered differently; each routed row must match.
        self.assertTrue(torch.equal(out.gemm2_out[out_perm], ref.gemm2_out[ref_perm]))

    def test_fused_routing_matches_the_unfused_call(self):
        from flashinfer.autotuner import autotune

        quant_info = _make_quant_info(seed=0)
        num_fused = 0
        misses = []
        fused_routing = op.fused_routing

        def counting_fused_routing(*args, **kwargs):
            nonlocal num_fused
            num_fused += 1
            return fused_routing(*args, **kwargs)

        with (
            patch.object(op, "fused_routing", counting_fused_routing),
            patch.object(q38_fused_routing, "_log_once", misses.append),
        ):
            # Default tactic first: autotuning caches a tactic per M bucket.
            for tuned in (False, True):
                for m in ROWS:
                    g = torch.Generator(device="cuda").manual_seed(m)
                    logits = torch.randn(m, NUM_EXPERTS, generator=g, device="cuda")
                    logits = logits.to(torch.bfloat16)
                    x = torch.randn(m, HIDDEN, generator=g, device="cuda")
                    x = x.to(torch.bfloat16)
                    if tuned:
                        with autotune(True):
                            _run_moe(x, logits, quant_info, fused=False)
                    with self.subTest(tuned=tuned, m=m):
                        ref = _run_moe(x, logits, quant_info, fused=False)
                        before = num_fused
                        out = _run_moe(x, logits, quant_info, fused=True)
                        self.assertEqual(num_fused, before + 1, misses)
                        self._assert_same(out, ref)


if __name__ == "__main__":
    unittest.main()
