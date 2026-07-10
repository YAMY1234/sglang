"""Tests for the fused SiluAndMul + per-tensor static FP8 quant path.

Covers:
- detection-level enablement rules incl. both kill-switches (CPU-safe)
- detection firing inside the real Qwen2MoeMLP.__init__ with a real
  ModelOptFp8 quant_config (GPU)
- numerical parity of the fused kernel vs the unfused reference chain
  (``SiluAndMul.forward_cuda`` then ``static_quant_fp8``) (GPU)
- gate/up column-layout guard and near-clamp behavior (GPU)
- Qwen2MoeMLP-level A/B: fused path never double-quantizes (GPU)
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.layers.communicator as communicator
from sglang.srt.environ import envs
from sglang.srt.layers.communicator import (
    FusedNormStaticFp8QuantSpec,
    detect_fused_silu_mul_static_fp8_quant,
)
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")


def _fp8_linear():
    # __new__ skips __init__ (which probes the device); isinstance is all
    # the detector needs.
    return SimpleNamespace(
        quant_method=ModelOptFp8LinearMethod.__new__(ModelOptFp8LinearMethod),
        input_scale=torch.tensor(0.01, dtype=torch.float32),
    )


def _bf16_linear():
    return SimpleNamespace(quant_method=object())


def _sm90_flags():
    return mock.patch.multiple(
        communicator,
        _is_cuda=True,
        _is_sm90_supported=True,
        _is_sm100_supported=False,
    )


class TestFusedSiluMulQuantDetection(CustomTestCase):
    """detect_fused_silu_mul_static_fp8_quant enables only under spec'd conditions."""

    def setUp(self):
        # The detector reads the process-wide ServerArgs (LoRA and
        # rl_on_policy_target gates).
        self._server_args_override = get_context().override_server_args()
        self._server_args_override.install()

    def tearDown(self):
        self._server_args_override.restore()

    def _override_server_args(self, **fields):
        self._server_args_override.restore()
        self._server_args_override = get_context().override_server_args(**fields)
        self._server_args_override.install()

    def test_enables_single_fp8_consumer(self):
        target = _fp8_linear()
        with _sm90_flags():
            spec = detect_fused_silu_mul_static_fp8_quant(target)
        self.assertIsNotNone(spec)
        self.assertIs(spec.target_linear, target)
        # Sole consumer by construction: never a dual output.
        self.assertFalse(spec.needs_bf16_out)

    def test_phase_kill_switch_env_disables(self):
        with _sm90_flags():
            with envs.SGLANG_DISABLE_FUSED_SILU_MUL_FP8_QUANT.override(True):
                self.assertIsNone(detect_fused_silu_mul_static_fp8_quant(_fp8_linear()))
            # Default polarity: fusion ON when the env is unset.
            self.assertIsNotNone(detect_fused_silu_mul_static_fp8_quant(_fp8_linear()))

    def test_global_kill_switch_env_disables(self):
        # The A/B contract: the Phase-1 global env must also disable this phase.
        with _sm90_flags():
            with envs.SGLANG_DISABLE_FUSED_NORM_STATIC_FP8_QUANT.override(True):
                self.assertIsNone(detect_fused_silu_mul_static_fp8_quant(_fp8_linear()))

    def test_non_fp8_consumer_scoping(self):
        # Other Qwen2MoeMLP users (bf16/NVFP4/other quant down_proj) stay
        # untouched: detection is consumer-driven per instance.
        with _sm90_flags():
            self.assertIsNone(detect_fused_silu_mul_static_fp8_quant(_bf16_linear()))

    def test_requires_sm90_or_sm100(self):
        with mock.patch.multiple(
            communicator,
            _is_cuda=True,
            _is_sm90_supported=False,
            _is_sm100_supported=False,
        ):
            self.assertIsNone(detect_fused_silu_mul_static_fp8_quant(_fp8_linear()))

    def test_rl_on_policy_target_disables(self):
        # rl_on_policy_target pins SiluAndMul to forward_native; the fused
        # kernel must not replace that math.
        self._override_server_args(rl_on_policy_target="fsdp")
        with _sm90_flags():
            self.assertIsNone(detect_fused_silu_mul_static_fp8_quant(_fp8_linear()))

    def test_lora_config_disables(self):
        for fields in ({"enable_lora": True}, {"lora_paths": ["dummy-adapter"]}):
            self._override_server_args(**fields)
            with _sm90_flags():
                self.assertIsNone(
                    detect_fused_silu_mul_static_fp8_quant(_fp8_linear()), fields
                )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestQwen2MoeMLPRealConstruction(CustomTestCase):
    """Detection fires inside the real Qwen2MoeMLP.__init__ with a real
    ModelOptFp8 quant_config — guards regressions the SimpleNamespace-based
    detection tests can't see (quant_method assignment order in LinearBase,
    a wrapper replacing down_proj before detection)."""

    def setUp(self):
        self._server_args_override = get_context().override_server_args()
        self._server_args_override.install()

    def tearDown(self):
        self._server_args_override.restore()

    def _mlp(self):
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8Config
        from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

        # tp_rank/tp_size given explicitly: no initialized distributed state.
        return Qwen2MoeMLP(
            hidden_size=256,
            intermediate_size=512,
            hidden_act="silu",
            quant_config=ModelOptFp8Config(is_checkpoint_fp8_serialized=True),
            prefix="mlp",
            tp_rank=0,
            tp_size=1,
        )

    def test_detection_fires_in_real_init(self):
        with _sm90_flags():
            mlp = self._mlp()
        self.assertIsInstance(mlp.down_proj.quant_method, ModelOptFp8LinearMethod)
        self.assertIsNotNone(mlp.act_fused_quant)
        self.assertIs(mlp.act_fused_quant.target_linear, mlp.down_proj)
        self.assertFalse(mlp.act_fused_quant.needs_bf16_out)

    def test_kill_switch_reaches_real_init(self):
        with _sm90_flags(), envs.SGLANG_DISABLE_FUSED_SILU_MUL_FP8_QUANT.override(
            True
        ):
            self.assertIsNone(self._mlp().act_fused_quant)


def _fp8_code_dist(a, b):
    """Distance in fp8 code space (monotone int mapping of e4m3 bits)."""

    def codes(t):
        u = t.view(torch.uint8).to(torch.int16)
        sign = u >> 7
        mag = u & 0x7F
        return torch.where(sign == 0, mag, -mag)

    return (codes(a) - codes(b)).abs()


def _silu_mul_f32(x):
    d = x.shape[-1] // 2
    g = x[..., :d].to(torch.float32)
    u = x[..., d:].to(torch.float32)
    return torch.nn.functional.silu(g) * u


def _make_case(M, I, seed, device, dtype=torch.bfloat16):
    from sglang.srt.layers.quantization.fp8_kernel import fp8_max

    torch.manual_seed(seed)
    x = torch.randn(M, 2 * I, dtype=dtype, device=device)
    # calibrated-style static scale: amax of the act output / fp8_max
    x_s = (_silu_mul_f32(x).abs().max() / fp8_max).clamp(min=1e-4).reshape(1)
    return x, x_s


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFusedSiluMulKernelNumerics(CustomTestCase):
    """Fused kernel vs the CURRENT unfused chain.

    Reference is ``SiluAndMul.forward_cuda`` (jit_kernel CUDA op) followed by
    ``static_quant_fp8`` — not ``forward_native`` (torch's bf16 silu rounds
    differently). Gate: fp8 codes >= 99.5% bitwise-exact, max 1 code off
    (skipped bf16 product rounding + tl.exp vs expf ulps). The gate is sized
    for SM100+ (precise-expf reference); on SM90 the reference is built with
    --use_fast_math, so if the gate fails there, measure and loosen per-arch
    rather than weakening the SM100 numbers.
    """

    # I = 512/4096: pow2; 768/1408: mask tails. 16384 (multi-tile loop path)
    # is exercised once in test_multi_tile.
    IS = (512, 768, 1408, 4096)
    MS = (1, 7, 128, 4096)
    SEEDS = tuple(range(5))

    def setUp(self):
        # SiluAndMul.__init__ reads the process-wide ServerArgs.
        self._server_args_override = get_context().override_server_args()
        self._server_args_override.install()
        from sglang.srt.layers.activation import SiluAndMul

        self._act = SiluAndMul()

    def tearDown(self):
        self._server_args_override.restore()

    def _reference_chain(self, x, x_s):
        from sglang.srt.layers.quantization.fp8_kernel import static_quant_fp8

        act = self._act.forward_cuda(x)
        q, _ = static_quant_fp8(act, x_s, repeat_scale=False)
        return q

    def _native_chain(self, x, x_s):
        from sglang.srt.layers.quantization.fp8_kernel import static_quant_fp8

        d = x.shape[-1] // 2
        act = torch.nn.functional.silu(x[..., :d]) * x[..., d:]
        q, _ = static_quant_fp8(act.contiguous(), x_s, repeat_scale=False)
        return q

    def _check_case(self, M, I, seed=0, dtype=torch.bfloat16):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fused_silu_mul_static_fp8_quant,
        )

        x, x_s = _make_case(M, I, seed, "cuda", dtype)
        q = fused_silu_mul_static_fp8_quant(x, x_s)
        ref_q = self._reference_chain(x, x_s)

        msg = f"M={M} I={I} seed={seed} dtype={dtype}"
        self.assertEqual(q.shape, (M, I), msg)
        d = _fp8_code_dist(q, ref_q)
        exact = (d == 0).float().mean().item()
        self.assertGreaterEqual(exact, 0.995, msg)
        self.assertLessEqual(d.max().item(), 1, msg)
        return exact, _fp8_code_dist(q, self._native_chain(x, x_s))

    def test_parity_sweep(self):
        native_exact = []
        for I in self.IS:
            for M in self.MS:
                for seed in self.SEEDS:
                    _, d_native = self._check_case(M, I, seed=seed)
                    native_exact.append((d_native == 0).float().mean().item())
        # Non-gating: document how far the fused kernel sits from the
        # forward_native (torch bf16 silu) chain.
        print(
            f"\n[fused_silu_mul] exact-match vs forward_native chain: "
            f"min={min(native_exact):.5f} mean={sum(native_exact)/len(native_exact):.5f}"
        )

    def test_multi_tile(self):
        # I=16384 > _FUSED_SILU_QUANT_MAX_SINGLE_TILE_N: loop path.
        self._check_case(64, 16384)

    def test_fp16_smoke(self):
        self._check_case(32, 768, dtype=torch.float16)

    def test_near_clamp_outliers(self):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fp8_max,
            fused_silu_mul_static_fp8_quant,
        )

        torch.manual_seed(0)
        M, I = 16, 1408
        x = torch.randn(M, 2 * I, dtype=torch.bfloat16, device="cuda")
        # Push part of the products past the clamp, both signs; keep some
        # rows gate-negative-heavy (silu asymmetry).
        x[::3, :I][:, ::7] = 40.0
        x[::3, I:][:, ::7] = 30.0
        x[1::3, :I] = -x[1::3, :I].abs()
        # Scale so |silu(g)*u| / scale straddles +-448.
        x_s = (_silu_mul_f32(x).abs().max() / fp8_max).reshape(1) * 0.5
        q = fused_silu_mul_static_fp8_quant(x, x_s)
        ref_q = self._reference_chain(x, x_s)
        self.assertTrue(torch.isfinite(q.float()).all())
        self.assertEqual(q.float().abs().max().item(), fp8_max)
        d = _fp8_code_dist(q, ref_q)
        self.assertGreaterEqual((d == 0).float().mean().item(), 0.995)
        self.assertLessEqual(d.max().item(), 1)

    def test_subnormal_magnitude_products(self):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fused_silu_mul_static_fp8_quant,
        )

        torch.manual_seed(0)
        x = torch.randn(32, 2 * 768, dtype=torch.bfloat16, device="cuda") * 0.2
        # scale 1.0 puts the products in the fp8 subnormal / zero code range
        x_s = torch.ones(1, dtype=torch.float32, device="cuda")
        q = fused_silu_mul_static_fp8_quant(x, x_s)
        d = _fp8_code_dist(q, self._reference_chain(x, x_s))
        self.assertGreaterEqual((d == 0).float().mean().item(), 0.995)
        self.assertLessEqual(d.max().item(), 1)

    def test_gate_up_layout(self):
        # Asymmetric constant halves: a swapped-halves kernel would compute
        # silu(up)*gate, which lands many fp8 codes away from silu(gate)*up.
        from sglang.srt.layers.quantization.fp8_kernel import (
            fp8_max,
            fused_silu_mul_static_fp8_quant,
        )

        I = 512
        x = torch.empty(4, 2 * I, dtype=torch.bfloat16, device="cuda")
        x[:, :I] = 3.0  # gate
        x[:, I:] = 0.5  # up
        x_s = torch.tensor([2.0 / fp8_max], device="cuda")

        q = fused_silu_mul_static_fp8_quant(x, x_s)
        ref_q = self._reference_chain(x, x_s)
        swapped = torch.cat([x[:, I:], x[:, :I]], dim=-1).contiguous()
        swapped_q = self._reference_chain(swapped, x_s)

        self.assertLessEqual(_fp8_code_dist(q, ref_q).max().item(), 1)
        self.assertGreater(_fp8_code_dist(q, swapped_q).min().item(), 1)

    def test_device_scalar_scale_no_sync(self):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fused_silu_mul_static_fp8_quant,
        )

        x, x_s = _make_case(8, 1408, 0, "cuda")
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            fused_silu_mul_static_fp8_quant(x, x_s)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()


class _StubFp8DownProj:
    """RowParallelLinear + ModelOptFp8LinearMethod.apply stand-in: quantizes a
    bf16 input with the layer-owned static input_scale (counting the calls),
    passes a pre-quantized fp8 input through, then dequant-GEMMs in fp32."""

    def __init__(self, weight, input_scale):
        self.weight = weight  # [hidden, intermediate] bf16
        self.input_scale = input_scale
        self.quant_method = ModelOptFp8LinearMethod.__new__(ModelOptFp8LinearMethod)
        self.static_quant_calls = 0
        self.seen_input_dtypes = []

    def __call__(self, x, skip_all_reduce=False):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fp8_dtype,
            static_quant_fp8,
        )

        self.seen_input_dtypes.append(x.dtype)
        if x.dtype != fp8_dtype:
            self.static_quant_calls += 1
            x, _ = static_quant_fp8(x.contiguous(), self.input_scale)
        out = (x.float() * self.input_scale.float()) @ self.weight.float().t()
        return out.to(self.weight.dtype), None


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestQwen2MoeMLPFusedAB(CustomTestCase):
    """Qwen2MoeMLP.forward fused vs unfused on the same weights."""

    def setUp(self):
        self._server_args_override = get_context().override_server_args()
        self._server_args_override.install()

    def tearDown(self):
        self._server_args_override.restore()

    def test_fused_matches_unfused_no_double_quant(self):
        from sglang.srt.layers.activation import SiluAndMul
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, fp8_max
        from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

        torch.manual_seed(0)
        device = "cuda"
        M, H, I = 16, 256, 512
        x = torch.randn(M, H, dtype=torch.bfloat16, device=device)
        w_gate_up = torch.randn(2 * I, H, dtype=torch.bfloat16, device=device) * 0.05
        w_down = torch.randn(H, I, dtype=torch.bfloat16, device=device) * 0.05

        def gate_up_proj(inp):
            return inp @ w_gate_up.t(), None

        input_scale = (
            (_silu_mul_f32(x @ w_gate_up.t()).abs().max() / fp8_max)
            .clamp(min=1e-4)
            .reshape(1)
        )

        def _mlp(fused):
            down = _StubFp8DownProj(w_down, input_scale)
            return (
                SimpleNamespace(
                    gate_up_proj=gate_up_proj,
                    act_fn=SiluAndMul(),
                    down_proj=down,
                    act_fused_quant=(
                        FusedNormStaticFp8QuantSpec(
                            target_linear=down, needs_bf16_out=False
                        )
                        if fused
                        else None
                    ),
                ),
                down,
            )

        fused_mlp, fused_down = _mlp(fused=True)
        unfused_mlp, unfused_down = _mlp(fused=False)
        out_fused = Qwen2MoeMLP.forward(fused_mlp, x)
        out_unfused = Qwen2MoeMLP.forward(unfused_mlp, x)

        # No double quant: the fused path hands down_proj a ready fp8 tensor.
        self.assertEqual(fused_down.static_quant_calls, 0)
        self.assertEqual(fused_down.seen_input_dtypes, [fp8_dtype])
        self.assertEqual(unfused_down.static_quant_calls, 1)
        self.assertEqual(unfused_down.seen_input_dtypes, [torch.bfloat16])

        # Same scale, codes within 1 on <0.5% of elements: GEMM outputs match
        # within bf16 tolerance.
        torch.testing.assert_close(
            out_fused.float(), out_unfused.float(), rtol=0.05, atol=0.05
        )


if __name__ == "__main__":
    unittest.main()
