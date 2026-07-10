"""Tests for the fused (residual-add +) RMSNorm + per-tensor static FP8 quant path.

Covers:
- detection-level enablement rules (CPU-safe, no GPU required)
- AR-fusion precedence in LayerCommunicator.prepare_attn (CPU-safe)
- numerical parity of the fused kernel vs the unfused reference (GPU)
- apply_fp8_linear pre-quantized fp8-input fast path (GPU)
- Phase 2 (post-attn norm -> shared_expert.gate_up_proj): bf16 co-output
  drift bound vs the production sgl-kernel gemma norm + the bitwise
  fp8 == static_quant_fp8(bf16_out) invariant (GPU), prepare_mlp arm and
  detection gates (CPU-safe), fp8 routing through Qwen2MoeSparseMoeBlock
  (CPU-safe), the dual-stream capture branch incl. clone elision (GPU),
  and the real prepare_mlp arm feeding the MoE block end-to-end (GPU)
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

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")


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

        # The dual-output variant quantizes from the rounded bf16 co-output
        # (the Phase-2 fp8 == static_quant_fp8(bf16_out) contract), so its
        # primary reference is the bf16-roundtrip one; single-output still
        # quantizes straight from fp32.
        ref_q_primary = ref_q_unfused if bf16_out else ref_q_fused
        ref_q_other = ref_q_fused if bf16_out else ref_q_unfused
        d = _fp8_code_dist(q, ref_q_primary)
        self.assertGreaterEqual((d == 0).float().mean().item(), 0.999, msg)
        self.assertLessEqual(d.max().item(), 1, msg)
        d_u = _fp8_code_dist(q, ref_q_other)
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


def _production_norm(x, weight, eps, residual):
    """What the router consumes today: the sgl-kernel gemma (fused-add)
    rmsnorm CUDA kernel, cloned inputs (the fused-add variant mutates)."""
    from sgl_kernel import gemma_fused_add_rmsnorm, gemma_rmsnorm

    if residual is None:
        return gemma_rmsnorm(x, weight, eps), None
    x_out, res_out = x.clone(), residual.clone()
    gemma_fused_add_rmsnorm(x_out, res_out, weight, eps)
    return x_out, res_out


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestBf16CoOutputBitwise(CustomTestCase):
    """Phase-2 gates for the dual-output site (bf16 -> router/experts, fp8 ->
    shared gate_up).

    T0a: the bf16 co-output replaces the production norm kernel's output as
    the MoE router input; it is NOT bit-identical (different reduction tree,
    rsqrt-multiply vs sqrt-divide — not reconcilable in Triton), so the
    bit-identity gate is re-scoped: this test enforces the drift BOUND and
    logs the exact-match rate; router-flip risk is signed off e2e (paired-seed
    GSM8K + topk_ids drift probe) before the fused arm ships default-on.
    T0b: hard invariant — the fp8 co-output IS the static quant of the bf16
    co-output, bitwise (both consumers must see one value).
    T0c: the residual is bitwise-identical to the production kernel's.
    """

    NS = (2048, 4096, 3000, 8192, 12288)
    MS = (1, 7, 128, 4096)
    SEEDS = tuple(range(5))
    EPS = 1e-6

    def _cases(self):
        for N in self.NS:
            for M in self.MS:
                for seed in self.SEEDS:
                    for has_res in (False, True):
                        yield M, N, seed, has_res

    def test_bf16_out_vs_production_norm(self):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fused_gemma_rmsnorm_static_fp8_quant,
        )

        total = exact = 0
        for M, N, seed, has_res in self._cases():
            msg = f"M={M} N={N} seed={seed} res={has_res}"
            x, weight, residual, x_s = _make_case(M, N, has_res, seed, "cuda")
            res_fused = residual.clone() if has_res else None
            _, out_bf16 = fused_gemma_rmsnorm_static_fp8_quant(
                x, weight, self.EPS, x_s, residual=res_fused, bf16_out=True
            )
            prod, _ = _production_norm(x, weight, self.EPS, residual)
            d = _bf16_code_dist(out_bf16, prod)
            self.assertLessEqual(d.max().item(), 1, msg)
            self.assertLess((d != 0).float().mean().item(), 1e-3, msg)
            total += d.numel()
            exact += (d == 0).sum().item()
        # Decision-gate telemetry (expected: not fully bit-identical).
        print(
            f"bf16 co-output vs production norm: exact-match rate "
            f"{exact / total:.8f} ({total - exact}/{total} flipped codes)"
        )

    def test_fp8_equals_static_quant_of_bf16_out(self):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fused_gemma_rmsnorm_static_fp8_quant,
            static_quant_fp8,
        )

        for M, N, seed, has_res in self._cases():
            msg = f"M={M} N={N} seed={seed} res={has_res}"
            x, weight, residual, x_s = _make_case(M, N, has_res, seed, "cuda")
            q, out_bf16 = fused_gemma_rmsnorm_static_fp8_quant(
                x, weight, self.EPS, x_s, residual=residual, bf16_out=True
            )
            q_ref, _ = static_quant_fp8(out_bf16, x_s, repeat_scale=False)
            self.assertTrue(
                torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)), msg
            )

    def test_residual_bitwise_vs_production(self):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fused_gemma_rmsnorm_static_fp8_quant,
        )

        for M, N, seed, has_res in self._cases():
            if not has_res:
                continue
            msg = f"M={M} N={N} seed={seed}"
            x, weight, residual, x_s = _make_case(M, N, True, seed, "cuda")
            res_fused = residual.clone()
            fused_gemma_rmsnorm_static_fp8_quant(
                x, weight, self.EPS, x_s, residual=res_fused, bf16_out=True
            )
            _, res_prod = _production_norm(x, weight, self.EPS, residual)
            self.assertTrue(torch.equal(res_fused, res_prod), msg)

    def test_helper_attaches_fp8_to_carrier(self):
        # The prepare_mlp arm returns the bf16 carrier with the fp8 riding on
        # `._sglang_fp8_static` — the attach point qwen2_moe pops.
        from sglang.srt.layers.communicator import _fused_rmsnorm_static_fp8_quant
        from sglang.srt.layers.layernorm import GemmaRMSNorm
        from sglang.srt.layers.quantization.fp8_kernel import static_quant_fp8

        x, weight, residual, x_s = _make_case(4, 2048, True, 0, "cuda")
        norm = GemmaRMSNorm(2048, eps=self.EPS)
        norm.weight = torch.nn.Parameter(weight)
        out = _fused_rmsnorm_static_fp8_quant(
            x, norm, x_s, residual=residual, bf16_out=True
        )
        self.assertEqual(out.dtype, torch.bfloat16)
        fp8 = getattr(out, "_sglang_fp8_static", None)
        self.assertIsNotNone(fp8)
        self.assertEqual(fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(fp8.shape, out.shape)
        q_ref, _ = static_quant_fp8(out, x_s, repeat_scale=False)
        self.assertTrue(torch.equal(fp8.view(torch.uint8), q_ref.view(torch.uint8)))


class TestPostNormFusedQuantGates(CustomTestCase):
    """Phase-2 detection + prepare_mlp arm gating (CPU-safe)."""

    def setUp(self):
        self._server_args_override = get_context().override_server_args()
        self._server_args_override.install()

    def tearDown(self):
        self._server_args_override.restore()

    def _stub_moe_block(self, with_shared_expert=True):
        from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

        # __new__ skips __init__ (which needs a distributed context); the
        # detector only reads submodules and isinstance.
        mlp = Qwen2MoeSparseMoeBlock.__new__(Qwen2MoeSparseMoeBlock)
        mlp.shared_expert = (
            SimpleNamespace(gate_up_proj=_fp8_linear()) if with_shared_expert else None
        )
        mlp.gate = _bf16_linear()
        mlp.shared_expert_gate = SimpleNamespace()
        mlp.experts = _bf16_linear()
        return mlp

    def test_detect_post_norm_fused_quant(self):
        from sglang.srt.models.qwen3_5 import _detect_post_norm_fused_quant

        mlp = self._stub_moe_block()
        with _sm90_flags():
            spec = _detect_post_norm_fused_quant(mlp)
            self.assertIsNotNone(spec)
            self.assertIs(spec.target_linear, mlp.shared_expert.gate_up_proj)
            self.assertTrue(spec.needs_bf16_out)

            # Phase kill-switch and the global kill-switch both disable.
            with envs.SGLANG_DISABLE_FUSED_SHARED_GATEUP_FP8_QUANT.override(True):
                self.assertIsNone(_detect_post_norm_fused_quant(mlp))
            with envs.SGLANG_DISABLE_FUSED_NORM_STATIC_FP8_QUANT.override(True):
                self.assertIsNone(_detect_post_norm_fused_quant(mlp))

            # Site-existence: no separate shared expert (e.g. shared-experts
            # fusion) or a non-MoE mlp module.
            self.assertIsNone(
                _detect_post_norm_fused_quant(
                    self._stub_moe_block(with_shared_expert=False)
                )
            )
            self.assertIsNone(_detect_post_norm_fused_quant(SimpleNamespace()))

    def _make_layer_communicator(self, norm, ladder_fn):
        def fake_post_init(lc):
            lc._communicate_simple_fn = communicator.CommunicateSimpleFn._trivial
            lc._communicate_with_all_reduce_and_layer_norm_fn = ladder_fn
            lc._communicate_summable_tensor_pair_fn = None

        with mock.patch.object(
            communicator.CommunicateContext, "init_new", return_value=SimpleNamespace()
        ), mock.patch.object(
            communicator.LayerCommunicator, "_post_init_communicate", fake_post_init
        ):
            return communicator.LayerCommunicator(
                layer_scatter_modes=None,
                input_layernorm=norm,
                post_attention_layernorm=norm,
                post_norm_fused_quant=FusedNormStaticFp8QuantSpec(
                    target_linear=_fp8_linear(), needs_bf16_out=True
                ),
            )

    def test_init_gate_keeps_spec_only_on_simple_ladder(self):
        from functools import partial

        from sglang.srt.layers.layernorm import GemmaRMSNorm

        gemma = GemmaRMSNorm(8)
        simple = communicator.CommunicateWithAllReduceAndLayerNormFn._simple
        gather = partial(
            communicator.CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
            residual_input_mode=None,
        )

        self.assertIsNotNone(
            self._make_layer_communicator(gemma, simple).post_norm_fused_quant
        )
        # Gather ladder owns AR+norm fusion: our arm must never wire there.
        self.assertIsNone(
            self._make_layer_communicator(gemma, gather).post_norm_fused_quant
        )
        # Non-Gemma norm: the kernel implements (1 + weight) semantics only.
        self.assertIsNone(
            self._make_layer_communicator(
                torch.nn.LayerNorm(8), simple
            ).post_norm_fused_quant
        )

    def _fake_for_prepare_mlp(self, spec):
        ladder = mock.MagicMock(return_value=(torch.randn(4, 8), torch.randn(4, 8)))
        fake = SimpleNamespace(
            post_norm_fused_quant=spec,
            post_attention_layernorm=mock.MagicMock(),
            _communicate_with_all_reduce_and_layer_norm_fn=ladder,
            _context=SimpleNamespace(cache=None),
        )
        return fake, ladder

    def test_prepare_mlp_fused_arm_fires(self):
        spec = FusedNormStaticFp8QuantSpec(
            target_linear=_fp8_linear(), needs_bf16_out=True
        )
        fake, ladder = self._fake_for_prepare_mlp(spec)
        hidden_states, residual = torch.randn(4, 8), torch.randn(4, 8)
        with mock.patch.object(
            communicator, "_fused_rmsnorm_static_fp8_quant"
        ) as fused_mock:
            fused_mock.return_value = torch.randn(4, 8)
            out_h, out_r = communicator.LayerCommunicator.prepare_mlp(
                fake, hidden_states, residual, forward_batch=None
            )
        fused_mock.assert_called_once()
        kwargs = fused_mock.call_args.kwargs
        self.assertTrue(kwargs["bf16_out"])  # carrier must stay bf16 here
        self.assertIs(kwargs["residual"], residual)
        self.assertIs(out_h, fused_mock.return_value)
        self.assertIs(out_r, residual)
        ladder.assert_not_called()

    def test_prepare_mlp_falls_through(self):
        spec = FusedNormStaticFp8QuantSpec(
            target_linear=_fp8_linear(), needs_bf16_out=True
        )
        bad_spec = FusedNormStaticFp8QuantSpec(
            target_linear=SimpleNamespace(
                input_scale=torch.tensor(torch.finfo(torch.float32).min)
            ),
            needs_bf16_out=True,
        )
        cases = [
            (None, torch.randn(4, 8), torch.randn(4, 8)),  # no spec
            (spec, torch.randn(0, 8), torch.randn(0, 8)),  # zero-token idle rank
            (spec, torch.randn(4, 8), None),  # no residual
            (bad_spec, torch.randn(4, 8), torch.randn(4, 8)),  # invalid scale
        ]
        for spec_i, hidden_states, residual in cases:
            fake, ladder = self._fake_for_prepare_mlp(spec_i)
            with mock.patch.object(
                communicator, "_fused_rmsnorm_static_fp8_quant"
            ) as fused_mock:
                communicator.LayerCommunicator.prepare_mlp(
                    fake, hidden_states, residual, forward_batch=None
                )
            fused_mock.assert_not_called()
            ladder.assert_called_once()


class _RecordingLinear:
    """Callable stub that records every input tensor it sees."""

    def __init__(self, out_fn):
        self.inputs = []
        self._out_fn = out_fn

    def __call__(self, *args, **kwargs):
        x = args[0] if args else kwargs["hidden_states"]
        self.inputs.append(x)
        return self._out_fn(x)


def _make_recording_moe_block(M, N, device="cpu"):
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    def zeros(*shape):
        return torch.zeros(*shape, dtype=torch.bfloat16, device=device)

    blk = Qwen2MoeSparseMoeBlock.__new__(Qwen2MoeSparseMoeBlock)
    blk.shared_expert = _RecordingLinear(lambda x: zeros(M, N))
    blk.shared_expert_gate = _RecordingLinear(lambda x: zeros(x.shape[0], 1))
    # Read as a fused_gate_sigmoid_mul_add argument on the TP path.
    blk.shared_expert_gate.weight = zeros(1, N)
    blk.gate = _RecordingLinear(lambda x: (zeros(x.shape[0], 4), None))
    blk.topk = _RecordingLinear(lambda x: SimpleNamespace())
    blk.experts = _RecordingLinear(lambda x: zeros(M, N))
    blk.is_nextn = True  # skips ExpertLocationDispatchInfo
    blk.layer_id = 0
    blk.enable_shared_expert_fusion = False
    blk.tp_size = 1
    blk.alt_stream = None
    return blk


def _run_moe_forward(blk, carrier, deepep, capture_mode=False):
    import sglang.srt.models.qwen2_moe as qwen2_moe

    a2a = SimpleNamespace(is_deepep=lambda: deepep, is_flashinfer=lambda: False)
    forward_batch = SimpleNamespace(num_token_non_padded=None)
    with mock.patch.object(
        qwen2_moe, "get_moe_a2a_backend", return_value=a2a
    ), mock.patch.object(
        qwen2_moe, "get_is_capture_mode", return_value=capture_mode
    ), mock.patch.object(
        qwen2_moe, "fused_gate_sigmoid_mul_add"
    ) as fused_gate:
        out = blk.forward(carrier, forward_batch)
    return out, fused_gate


class TestSharedGateupFp8Routing(CustomTestCase):
    """The fp8 co-output must land exactly on shared_expert's input and
    nowhere else; a missing attribute degrades to bf16 (CPU-safe stubs).

    This is the "fused path actually fires" assertion the attribute-passing
    design demands: any upstream clone/view that drops the attribute shows up
    here (and as a quant-kernel count regression e2e), never as corruption.
    """

    M, N = 4, 8

    def _make_carrier(self, with_fp8):
        carrier = torch.randn(self.M, self.N, dtype=torch.bfloat16)
        if with_fp8:
            carrier._sglang_fp8_static = torch.zeros(
                self.M, self.N, dtype=torch.bfloat16
            ).to(torch.float8_e4m3fn)
        return carrier

    def _check(self, deepep):
        for with_fp8 in (True, False):
            msg = f"deepep={deepep} with_fp8={with_fp8}"
            blk = _make_recording_moe_block(self.M, self.N)
            carrier = self._make_carrier(with_fp8)
            _run_moe_forward(blk, carrier, deepep)

            expected = torch.float8_e4m3fn if with_fp8 else torch.bfloat16
            self.assertEqual(blk.shared_expert.inputs[0].dtype, expected, msg)
            # Every other consumer keeps the bf16 carrier.
            for name in ("gate", "topk", "experts"):
                for x in getattr(blk, name).inputs:
                    self.assertEqual(x.dtype, torch.bfloat16, f"{msg} {name}")
            for x in blk.shared_expert_gate.inputs:
                self.assertEqual(x.dtype, torch.bfloat16, msg)

    def test_deepep_path(self):
        self._check(deepep=True)

    def test_tp_path(self):
        # Non-deepep eager path uses the fused sigmoid gate; the gate reads
        # the bf16 carrier directly inside fused_gate_sigmoid_mul_add.
        for with_fp8 in (True, False):
            blk = _make_recording_moe_block(self.M, self.N)
            carrier = self._make_carrier(with_fp8)
            _, fused_gate = _run_moe_forward(blk, carrier, deepep=False)
            expected = torch.float8_e4m3fn if with_fp8 else torch.bfloat16
            self.assertEqual(blk.shared_expert.inputs[0].dtype, expected)
            fused_gate.assert_called_once()
            self.assertEqual(
                fused_gate.call_args.args[0].dtype, torch.bfloat16
            )  # carrier re-read stays bf16
            for name in ("gate", "topk", "experts"):
                for x in getattr(blk, name).inputs:
                    self.assertEqual(x.dtype, torch.bfloat16, name)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestSharedGateupDualStreamRouting(CustomTestCase):
    """forward_normal_dual_stream — the branch decode graph capture actually
    takes on non-deepep configs — must route the fp8 co-output like the eager
    path, and must not clone the carrier when nothing reads the clone."""

    M, N = 4, 8

    def _run(self, with_fp8):
        import sglang.srt.models.qwen2_moe as qwen2_moe

        blk = _make_recording_moe_block(self.M, self.N, device="cuda")
        blk.alt_stream = torch.cuda.Stream()
        carrier = torch.randn(self.M, self.N, dtype=torch.bfloat16, device="cuda")
        fp8 = carrier.to(torch.float8_e4m3fn) if with_fp8 else None
        if with_fp8:
            carrier._sglang_fp8_static = fp8

        shared_inputs = []
        orig = qwen2_moe.Qwen2MoeSparseMoeBlock._forward_shared_experts

        def spy(self_blk, hidden_states, apply_gate=True, gateup_input=None):
            shared_inputs.append(hidden_states)
            return orig(
                self_blk,
                hidden_states,
                apply_gate=apply_gate,
                gateup_input=gateup_input,
            )

        with mock.patch.object(
            qwen2_moe.Qwen2MoeSparseMoeBlock, "_forward_shared_experts", spy
        ):
            _, fused_gate = _run_moe_forward(
                blk, carrier, deepep=False, capture_mode=True
            )
        return blk, carrier, fp8, shared_inputs, fused_gate

    def test_fp8_routing_and_clone_elision(self):
        blk, carrier, fp8, shared_inputs, fused_gate = self._run(with_fp8=True)
        self.assertIs(blk.shared_expert.inputs[0], fp8)
        # Nothing reads the carrier inside _forward_shared_experts here
        # (fp8 supersedes it for gate_up, the fused gate reads it outside),
        # so no clone may be captured: same storage as the carrier.
        self.assertEqual(shared_inputs[0].data_ptr(), carrier.data_ptr())
        for name in ("gate", "topk", "experts"):
            for x in getattr(blk, name).inputs:
                self.assertEqual(x.dtype, torch.bfloat16, name)
        fused_gate.assert_called_once()
        self.assertEqual(fused_gate.call_args.args[0].dtype, torch.bfloat16)

    def test_bf16_fallback_keeps_clone(self):
        blk, carrier, _, shared_inputs, _ = self._run(with_fp8=False)
        self.assertEqual(blk.shared_expert.inputs[0].dtype, torch.bfloat16)
        # Without the fp8 co-output the dual-stream clone must stay.
        self.assertNotEqual(shared_inputs[0].data_ptr(), carrier.data_ptr())


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestPrepareMlpToMoeBlockIntegration(CustomTestCase):
    """Real fused prepare_mlp arm (no mocked helper) feeding its carrier into
    Qwen2MoeSparseMoeBlock.forward: the fp8 co-output must survive the hop and
    land on shared_expert bitwise-consistent with the carrier. Guards the
    attr chain end-to-end short of the decoder-layer forward itself (which
    needs a distributed context; the e2e kernel-count check covers that hop).
    """

    EPS = 1e-6

    def test_fused_carrier_routes_into_moe_block(self):
        from sglang.srt.layers.layernorm import GemmaRMSNorm
        from sglang.srt.layers.quantization.fp8_kernel import static_quant_fp8

        M, N = 4, 2048
        x, weight, residual, x_s = _make_case(M, N, True, 0, "cuda")
        norm = GemmaRMSNorm(N, eps=self.EPS)
        norm.weight = torch.nn.Parameter(weight)
        spec = FusedNormStaticFp8QuantSpec(
            target_linear=SimpleNamespace(
                quant_method=ModelOptFp8LinearMethod.__new__(ModelOptFp8LinearMethod),
                input_scale=x_s.to(torch.float32),
            ),
            needs_bf16_out=True,
        )
        ladder = mock.MagicMock()
        fake = SimpleNamespace(
            post_norm_fused_quant=spec,
            post_attention_layernorm=norm,
            _communicate_with_all_reduce_and_layer_norm_fn=ladder,
            _context=SimpleNamespace(cache=None),
        )

        out_h, out_r = communicator.LayerCommunicator.prepare_mlp(
            fake, x, residual, forward_batch=None
        )
        ladder.assert_not_called()
        self.assertIs(out_r, residual)
        self.assertEqual(out_h.dtype, torch.bfloat16)
        fp8 = getattr(out_h, "_sglang_fp8_static", None)
        self.assertIsNotNone(fp8)

        blk = _make_recording_moe_block(M, N, device="cuda")
        _run_moe_forward(blk, out_h, deepep=True)
        # Identity: the co-output tensor itself reaches gate_up's input.
        self.assertIs(blk.shared_expert.inputs[0], fp8)
        q_ref, _ = static_quant_fp8(out_h, x_s, repeat_scale=False)
        self.assertTrue(torch.equal(fp8.view(torch.uint8), q_ref.view(torch.uint8)))
        for name in ("gate", "topk", "experts"):
            for t in getattr(blk, name).inputs:
                self.assertEqual(t.dtype, torch.bfloat16, name)


if __name__ == "__main__":
    unittest.main()
