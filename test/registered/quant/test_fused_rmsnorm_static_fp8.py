"""Tests for the fused (residual-add +) RMSNorm + per-tensor static FP8 quant path.

Covers:
- detection-level enablement rules (CPU-safe, no GPU required)
- AR-fusion precedence in LayerCommunicator.prepare_attn (CPU-safe)
- numerical parity of the fused kernel vs the unfused reference (GPU)
- apply_fp8_linear pre-quantized fp8-input fast path (GPU)
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.layers.communicator as communicator
from sglang.srt.environ import envs
from sglang.srt.layers.communicator import (
    FusedNormStaticFp8QuantSpec,
    detect_fused_norm_static_fp8_quant,
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


class TestFusedQuantDetection(CustomTestCase):
    """detect_fused_norm_static_fp8_quant enables only under spec'd conditions."""

    def setUp(self):
        # The detector reads the process-wide ServerArgs for the LoRA gate.
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
            spec = detect_fused_norm_static_fp8_quant([target])
        self.assertIsNotNone(spec)
        self.assertIs(spec.target_linear, target)
        self.assertFalse(spec.needs_bf16_out)

    def test_bf16_co_consumer_forces_dual_output(self):
        target = _fp8_linear()
        with _sm90_flags():
            spec = detect_fused_norm_static_fp8_quant([target, _bf16_linear()])
        self.assertIsNotNone(spec)
        self.assertIs(spec.target_linear, target)
        self.assertTrue(spec.needs_bf16_out)

    def test_kill_switch_env_disables(self):
        with _sm90_flags():
            with envs.SGLANG_DISABLE_FUSED_NORM_STATIC_FP8_QUANT.override(True):
                self.assertIsNone(detect_fused_norm_static_fp8_quant([_fp8_linear()]))
            # Default polarity: fusion ON when the env is unset.
            self.assertIsNotNone(detect_fused_norm_static_fp8_quant([_fp8_linear()]))

    def test_requires_sm90_or_sm100(self):
        with mock.patch.multiple(
            communicator,
            _is_cuda=True,
            _is_sm90_supported=False,
            _is_sm100_supported=False,
        ):
            self.assertIsNone(detect_fused_norm_static_fp8_quant([_fp8_linear()]))
        with mock.patch.multiple(
            communicator,
            _is_cuda=False,
            _is_sm90_supported=True,
            _is_sm100_supported=True,
        ):
            self.assertIsNone(detect_fused_norm_static_fp8_quant([_fp8_linear()]))

    def test_requires_unique_fp8_consumer(self):
        with _sm90_flags():
            self.assertIsNone(
                detect_fused_norm_static_fp8_quant([_fp8_linear(), _fp8_linear()])
            )
            self.assertIsNone(detect_fused_norm_static_fp8_quant([_bf16_linear()]))
            self.assertIsNone(detect_fused_norm_static_fp8_quant([]))

    def test_lora_config_disables(self):
        # LoRA wraps its targets only after model construction, so the gate
        # must key on the server config, not on module attributes.
        for fields in ({"enable_lora": True}, {"lora_paths": ["dummy-adapter"]}):
            self._override_server_args(**fields)
            with _sm90_flags():
                self.assertIsNone(
                    detect_fused_norm_static_fp8_quant([_fp8_linear()]), fields
                )

    def test_resolve_scale_guard(self):
        spec = FusedNormStaticFp8QuantSpec(
            target_linear=_fp8_linear(), needs_bf16_out=False
        )
        scale = spec.resolve_scale()
        self.assertIsNotNone(scale)
        self.assertIs(scale, spec.target_linear.input_scale)

        # Never-loaded float32.min placeholder disables this layer's fusion...
        bad = FusedNormStaticFp8QuantSpec(
            target_linear=SimpleNamespace(
                input_scale=torch.tensor(torch.finfo(torch.float32).min)
            ),
            needs_bf16_out=False,
        )
        self.assertIsNone(bad.resolve_scale())
        self.assertIsNone(bad.resolve_scale())  # cached: same object, no rerun
        # ...until a weight (re)load installs a new, valid Parameter object
        # (dummy-load bootstrap followed by update_weights_*).
        bad.target_linear.input_scale = torch.tensor(0.5)
        self.assertIs(bad.resolve_scale(), bad.target_linear.input_scale)
        # And a later swap to an invalid object disables again.
        bad.target_linear.input_scale = torch.tensor(-1.0)
        self.assertIsNone(bad.resolve_scale())

        # Non-scalar / non-fp32 scales must also disable.
        for scale in (torch.ones(2), torch.tensor(0.5, dtype=torch.bfloat16)):
            spec = FusedNormStaticFp8QuantSpec(
                target_linear=SimpleNamespace(input_scale=scale),
                needs_bf16_out=False,
            )
            self.assertIsNone(spec.resolve_scale())


class TestArFusionPrecedence(CustomTestCase):
    """When allreduce+rmsnorm fusion fires, the quant fusion must not."""

    def _fake_communicator(self):
        norm = mock.MagicMock()
        norm_out = (torch.randn(4, 8), torch.randn(4, 8))
        norm.return_value = norm_out
        norm.forward_with_allreduce_fusion.return_value = norm_out
        fake = SimpleNamespace(
            input_layernorm=norm,
            input_norm_fused_quant=FusedNormStaticFp8QuantSpec(
                target_linear=_fp8_linear(), needs_bf16_out=False
            ),
            _communicate_simple_fn=lambda hidden_states, forward_batch, context: (
                hidden_states
            ),
            qkv_latent_func=None,
            _context=None,
        )
        return fake, norm

    def _prepare_attn(self, fake, hidden_states, residual, ar_fusion_applies):
        with mock.patch.object(
            communicator,
            "get_attn_tp_context",
            return_value=SimpleNamespace(input_scattered=False),
        ), mock.patch.object(
            communicator, "apply_aiter_all_reduce_fusion", return_value=False
        ), mock.patch.object(
            communicator,
            "apply_flashinfer_allreduce_fusion",
            return_value=ar_fusion_applies,
        ), mock.patch.object(
            communicator,
            "moe_tensor_model_parallel_all_reduce",
            side_effect=lambda x: x,
        ), mock.patch.object(
            communicator, "_fused_rmsnorm_static_fp8_quant"
        ) as fused_mock:
            fused_mock.return_value = torch.randn(4, 8)
            result = communicator.LayerCommunicator.prepare_attn(
                fake, hidden_states, residual, forward_batch=None
            )
        return result, fused_mock

    def test_ar_fusion_wins(self):
        fake, norm = self._fake_communicator()
        hidden_states = torch.randn(4, 8)
        hidden_states._sglang_needs_allreduce_fusion = True
        _, fused_mock = self._prepare_attn(
            fake, hidden_states, torch.randn(4, 8), ar_fusion_applies=True
        )
        norm.forward_with_allreduce_fusion.assert_called_once()
        fused_mock.assert_not_called()

    def test_ar_tagged_but_not_firing_stays_plain(self):
        fake, norm = self._fake_communicator()
        hidden_states = torch.randn(4, 8)
        hidden_states._sglang_needs_allreduce_fusion = True
        _, fused_mock = self._prepare_attn(
            fake, hidden_states, torch.randn(4, 8), ar_fusion_applies=False
        )
        norm.assert_called_once()
        fused_mock.assert_not_called()

    def test_quant_fusion_on_plain_branch(self):
        fake, norm = self._fake_communicator()
        _, fused_mock = self._prepare_attn(
            fake, torch.randn(4, 8), torch.randn(4, 8), ar_fusion_applies=False
        )
        fused_mock.assert_called_once()
        norm.assert_not_called()
        norm.forward_with_allreduce_fusion.assert_not_called()

    def test_zero_token_batch_skips_fusion(self):
        fake, norm = self._fake_communicator()
        (hidden_states, residual), fused_mock = self._prepare_attn(
            fake, torch.randn(0, 8), torch.randn(0, 8), ar_fusion_applies=False
        )
        fused_mock.assert_not_called()
        norm.assert_not_called()
        self.assertIs(hidden_states, residual)


def _reference(x, weight, eps, x_s, residual, round_normed_to_input_dtype):
    """GemmaRMSNorm.forward_native math + _static_quant_fp8 math.

    round_normed_to_input_dtype=True emulates the UNFUSED pipeline (norm kernel
    writes bf16, then static_quant_fp8 quantizes the bf16); False emulates the
    fused kernel (quantize straight from fp32).
    """
    from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, fp8_max, fp8_min

    z = x.to(torch.float32)
    residual_out = None
    if residual is not None:
        z = z + residual.to(torch.float32)
        residual_out = z.to(x.dtype)
    var = z.pow(2).mean(dim=-1, keepdim=True)
    y = z * torch.rsqrt(var + eps) * (1.0 + weight.to(torch.float32))
    if round_normed_to_input_dtype:
        y = y.to(x.dtype).to(torch.float32)
    q = (y * (1.0 / x_s.to(torch.float32))).clamp(fp8_min, fp8_max).to(fp8_dtype)
    return q, y, residual_out


def _fp8_code_dist(a, b):
    """Distance in fp8 code space (monotone int mapping of e4m3 bits)."""

    def codes(t):
        u = t.view(torch.uint8).to(torch.int16)
        sign = u >> 7
        mag = u & 0x7F
        return torch.where(sign == 0, mag, -mag)

    return (codes(a) - codes(b)).abs()


def _bf16_code_dist(a, b):
    """Distance in bf16 code space (monotone int mapping of the bit pattern)."""

    def codes(t):
        u = t.view(torch.int16).to(torch.int32) & 0xFFFF
        sign = u >> 15
        mag = u & 0x7FFF
        return torch.where(sign == 0, mag, -mag)

    return (codes(a) - codes(b)).abs()


def _make_case(M, N, has_res, seed, device):
    from sglang.srt.layers.quantization.fp8_kernel import fp8_max

    torch.manual_seed(seed)
    x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
    residual = (
        torch.randn(M, N, dtype=torch.bfloat16, device=device) if has_res else None
    )
    weight = torch.rand(N, dtype=torch.bfloat16, device=device) * 2 - 1
    # calibrated-style static scale: amax of the normed activation / fp8_max
    _, y_ref, _ = _reference(
        x, weight, 1e-6, torch.ones(1, device=device), residual, False
    )
    x_s = (y_ref.abs().max() / fp8_max).clamp(min=1e-4).reshape(1)
    return x, weight, residual, x_s


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFusedKernelNumerics(CustomTestCase):
    """Fused kernel vs unfused reference.

    Tolerances (constraints doc): residual bitwise; fp8 codes >= 99.9% exact
    and max 1 code off (reduction order + fused-vs-bf16-roundtrip ulps).
    """

    # N = 2048/4096: serving hidden sizes; 3000: mask tail; 8192: single-tile
    # boundary; 12288: multi-tile loop path.
    NS = (2048, 4096, 3000, 8192, 12288)
    MS = (1, 7, 128, 4096)
    SEEDS = tuple(range(5))
    EPS = 1e-6

    def _check_case(self, M, N, has_res, seed=0, bf16_out=False):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fused_gemma_rmsnorm_static_fp8_quant,
        )

        device = "cuda"
        x, weight, residual, x_s = _make_case(M, N, has_res, seed, device)
        res_in = residual.clone() if has_res else None
        res_ptr = res_in.data_ptr() if has_res else None

        q, out_bf16 = fused_gemma_rmsnorm_static_fp8_quant(
            x, weight, self.EPS, x_s, residual=res_in, bf16_out=bf16_out
        )

        ref_q_fused, ref_y, ref_res = _reference(
            x, weight, self.EPS, x_s, residual, False
        )
        ref_q_unfused, _, _ = _reference(x, weight, self.EPS, x_s, residual, True)

        msg = f"M={M} N={N} res={has_res}"
        if has_res:
            # residual: bitwise equal and mutated in place
            self.assertEqual(res_in.data_ptr(), res_ptr, msg)
            self.assertTrue(torch.equal(res_in, ref_res), msg)

        d = _fp8_code_dist(q, ref_q_fused)
        self.assertGreaterEqual((d == 0).float().mean().item(), 0.999, msg)
        self.assertLessEqual(d.max().item(), 1, msg)
        d_u = _fp8_code_dist(q, ref_q_unfused)
        self.assertLessEqual(d_u.max().item(), 1, msg)

        if bf16_out:
            d_b = _bf16_code_dist(out_bf16, ref_y.to(torch.bfloat16))
            self.assertGreaterEqual((d_b == 0).float().mean().item(), 0.999, msg)
            self.assertLessEqual(d_b.max().item(), 1, msg)
        else:
            self.assertIsNone(out_bf16)

    def test_norm_only_parity(self):
        for N in self.NS:
            for M in self.MS:
                for seed in self.SEEDS:
                    self._check_case(M, N, has_res=False, seed=seed)

    def test_fused_add_parity(self):
        for N in self.NS:
            for M in self.MS:
                for seed in self.SEEDS:
                    self._check_case(M, N, has_res=True, seed=seed)

    def test_bf16_out(self):
        for N in (2048, 3000, 12288):
            for has_res in (False, True):
                self._check_case(64, N, has_res=has_res, bf16_out=True)

    def test_near_clamp_outliers(self):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fp8_max,
            fused_gemma_rmsnorm_static_fp8_quant,
        )

        device = "cuda"
        torch.manual_seed(0)
        M, N = 16, 2048
        x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
        x[::3, ::7] = 100.0  # push part of the normed output past the clamp
        weight = torch.rand(N, dtype=torch.bfloat16, device=device)
        x_s = torch.tensor([1e-3], device=device)
        q, _ = fused_gemma_rmsnorm_static_fp8_quant(x, weight, self.EPS, x_s)
        ref_q, _, _ = _reference(x, weight, self.EPS, x_s, None, False)
        self.assertTrue(torch.isfinite(q.float()).all())
        self.assertEqual(q.float().abs().max().item(), fp8_max)
        self.assertLessEqual(_fp8_code_dist(q, ref_q).max().item(), 1)

    def test_first_layer_alias_safety(self):
        # Replicates prepare_attn first layer: residual = hidden_states alias;
        # the no-residual fused call must leave the input untouched.
        from sglang.srt.layers.quantization.fp8_kernel import (
            fused_gemma_rmsnorm_static_fp8_quant,
        )

        x, weight, _, x_s = _make_case(8, 2048, False, 0, "cuda")
        x_snapshot = x.clone()
        fused_gemma_rmsnorm_static_fp8_quant(x, weight, self.EPS, x_s)
        self.assertTrue(torch.equal(x, x_snapshot))

    def test_device_scalar_scale_no_sync(self):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fused_gemma_rmsnorm_static_fp8_quant,
        )

        x, weight, residual, x_s = _make_case(8, 2048, True, 0, "cuda")
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            fused_gemma_rmsnorm_static_fp8_quant(
                x, weight, self.EPS, x_s, residual=residual, bf16_out=True
            )
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()


def _sm89_available():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


@unittest.skipUnless(_sm89_available(), "requires CUDA sm89+ (fp8 _scaled_mm)")
class TestApplyFp8LinearPrequantized(CustomTestCase):
    """apply_fp8_linear fp8-input fast path == unfused result, bitwise."""

    def setUp(self):
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

        torch.manual_seed(0)
        device = "cuda"
        M, K, N = 8, 128, 64
        self.x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        w_amax = w.abs().max().float()
        self.weight_scale = (w_amax / torch.finfo(fp8_dtype).max).reshape(1)
        self.weight = (
            (w.float() / self.weight_scale).to(fp8_dtype).t()
        )  # (K, N), column-major as _scaled_mm expects
        self.input_scale = torch.tensor([0.01], device=device)

    def test_prequantized_matches_unfused(self):
        from sglang.srt.layers.quantization.fp8_kernel import static_quant_fp8
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

        ref = apply_fp8_linear(
            self.x,
            self.weight,
            self.weight_scale,
            input_scale=self.input_scale,
            cutlass_fp8_supported=False,
        )
        qx, _ = static_quant_fp8(self.x, self.input_scale, repeat_scale=False)
        out = apply_fp8_linear(
            qx,
            self.weight,
            self.weight_scale,
            input_scale=self.input_scale,
            cutlass_fp8_supported=False,
            out_dtype=torch.bfloat16,
        )
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(ref, out))

    def test_out_dtype_respected(self):
        from sglang.srt.layers.quantization.fp8_kernel import static_quant_fp8
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

        qx, _ = static_quant_fp8(self.x, self.input_scale, repeat_scale=False)
        out = apply_fp8_linear(
            qx,
            self.weight,
            self.weight_scale,
            input_scale=self.input_scale,
            cutlass_fp8_supported=False,
            out_dtype=torch.float16,
        )
        self.assertEqual(out.dtype, torch.float16)

    def test_fp8_input_requires_out_dtype(self):
        from sglang.srt.layers.quantization.fp8_kernel import static_quant_fp8
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

        qx, _ = static_quant_fp8(self.x, self.input_scale, repeat_scale=False)
        with self.assertRaises(AssertionError):
            apply_fp8_linear(
                qx,
                self.weight,
                self.weight_scale,
                input_scale=self.input_scale,
                cutlass_fp8_supported=False,
            )

    def test_explicit_out_dtype_backward_compatible(self):
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

        ref = apply_fp8_linear(
            self.x,
            self.weight,
            self.weight_scale,
            input_scale=self.input_scale,
            cutlass_fp8_supported=False,
        )
        out = apply_fp8_linear(
            self.x,
            self.weight,
            self.weight_scale,
            input_scale=self.input_scale,
            cutlass_fp8_supported=False,
            out_dtype=torch.bfloat16,
        )
        self.assertTrue(torch.equal(ref, out))


def _cutlass_fp8_available():
    if not torch.cuda.is_available():
        return False
    try:
        from sglang.srt.layers.quantization.fp8_utils import cutlass_fp8_supported

        return bool(cutlass_fp8_supported())
    except Exception:
        return False


@unittest.skipUnless(_cutlass_fp8_available(), "requires cutlass fp8 (sm90+)")
class TestApplyFp8LinearPrequantizedCutlass(CustomTestCase):
    """Pre-quantized fp8-input fast path on the cutlass branch.

    Exercises the (M, 1) ``input_scale.expand().contiguous()`` materialization
    against the unfused path's ``repeat_scale=True`` (channelwise weight scale
    per ModelOptFp8's ``process_weights_after_loading`` on cutlass hardware).
    """

    def setUp(self):
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

        torch.manual_seed(0)
        device = "cuda"
        M, K, N = 8, 128, 64
        self.x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        # channelwise weight scale: one scale per output channel
        self.weight_scale = (
            w.abs().amax(dim=1).float() / torch.finfo(fp8_dtype).max
        ).reshape(N)
        self.weight = (
            (w.float() / self.weight_scale.unsqueeze(1)).to(fp8_dtype).t()
        )  # (K, N)
        self.input_scale = torch.tensor([0.01], device=device)

    def test_prequantized_matches_unfused_cutlass(self):
        from sglang.srt.layers.quantization.fp8_kernel import static_quant_fp8
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

        ref = apply_fp8_linear(
            self.x,
            self.weight,
            self.weight_scale,
            input_scale=self.input_scale,
            cutlass_fp8_supported=True,
        )
        qx, _ = static_quant_fp8(self.x, self.input_scale, repeat_scale=False)
        out = apply_fp8_linear(
            qx,
            self.weight,
            self.weight_scale,
            input_scale=self.input_scale,
            cutlass_fp8_supported=True,
            out_dtype=torch.bfloat16,
        )
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(ref, out))


if __name__ == "__main__":
    unittest.main()
