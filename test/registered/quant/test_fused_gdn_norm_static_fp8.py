"""Tests for the fused GDN gated-RMSNorm + static FP8 quant path (qwen3.5).

Covers:
- detection-level enablement rules incl. the phase kill-switch (CPU-safe)
- numerical parity of the fused kernel vs the real unfused chain
  (fla rms_norm_gated -> static_quant_fp8) and vs an fp32 reference (GPU)
- layout / stride / no-sync / no-mutation contracts of the host wrapper (GPU)
- forward-path wiring: the resolve_scale()-gated branch that hands out_proj
  fp8 codes, and its fallback to the plain fla norm (GPU)
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.layers.communicator as communicator
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod
from sglang.srt.models.qwen3_5 import _detect_gdn_norm_fused_quant
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

ACTIVATIONS = ("swish", "sigmoid")


def _fp8_linear(input_scale=None):
    # __new__ skips __init__ (which probes the device); isinstance is all
    # the detector needs.
    return SimpleNamespace(
        quant_method=ModelOptFp8LinearMethod.__new__(ModelOptFp8LinearMethod),
        input_scale=(
            input_scale
            if input_scale is not None
            else torch.tensor(0.01, dtype=torch.float32)
        ),
    )


def _norm_stub(activation="swish", norm_before_gate=True, group_size=None):
    # Only the attributes the detector pins on the RMSNormGated instance.
    return SimpleNamespace(
        activation=activation, norm_before_gate=norm_before_gate, group_size=group_size
    )


def _sm90_flags():
    return mock.patch.multiple(
        communicator,
        _is_cuda=True,
        _is_sm90_supported=True,
        _is_sm100_supported=False,
    )


class TestGdnFusedQuantDetection(CustomTestCase):
    """_detect_gdn_norm_fused_quant layers the phase gates on the shared detector."""

    def setUp(self):
        # The shared detector reads the process-wide ServerArgs (LoRA gate).
        self._server_args_override = get_context().override_server_args()
        self._server_args_override.install()

    def tearDown(self):
        self._server_args_override.restore()

    def test_enables_by_default(self):
        out_proj = _fp8_linear()
        for activation in ("swish", "silu", "sigmoid"):
            with _sm90_flags():
                spec = _detect_gdn_norm_fused_quant(out_proj, _norm_stub(activation))
            self.assertIsNotNone(spec, activation)
            self.assertIs(spec.target_linear, out_proj)
            self.assertFalse(spec.needs_bf16_out)

    def test_phase_kill_switch_disables(self):
        with _sm90_flags():
            with envs.SGLANG_DISABLE_FUSED_GDN_NORM_STATIC_FP8_QUANT.override(True):
                self.assertIsNone(
                    _detect_gdn_norm_fused_quant(_fp8_linear(), _norm_stub())
                )
            # Default polarity: fusion ON when the env is unset.
            self.assertIsNotNone(
                _detect_gdn_norm_fused_quant(_fp8_linear(), _norm_stub())
            )

    def test_global_kill_switch_disables(self):
        with _sm90_flags():
            with envs.SGLANG_DISABLE_FUSED_NORM_STATIC_FP8_QUANT.override(True):
                self.assertIsNone(
                    _detect_gdn_norm_fused_quant(_fp8_linear(), _norm_stub())
                )

    def test_unsupported_activation_disables(self):
        # fla applies NO gate for unknown strings; those configs must stay on
        # the unfused path rather than get a swish gate from the fused kernel.
        with _sm90_flags():
            for activation in (None, "gelu", ""):
                self.assertIsNone(
                    _detect_gdn_norm_fused_quant(
                        _fp8_linear(), _norm_stub(activation)
                    ),
                    activation,
                )

    def test_non_default_norm_config_disables(self):
        # The kernel hardcodes norm-before-gate single-group semantics.
        with _sm90_flags():
            self.assertIsNone(
                _detect_gdn_norm_fused_quant(
                    _fp8_linear(), _norm_stub(norm_before_gate=False)
                )
            )
            self.assertIsNone(
                _detect_gdn_norm_fused_quant(_fp8_linear(), _norm_stub(group_size=64))
            )

    def test_non_fp8_consumer_disables(self):
        with _sm90_flags():
            self.assertIsNone(
                _detect_gdn_norm_fused_quant(
                    SimpleNamespace(quant_method=object()), _norm_stub()
                )
            )


def _reference_fp32(x, z, weight, eps, x_s, activation):
    """fla rms_norm_gated math (fp32, norm-before-gate) + _static_quant_fp8 math.

    Emulates the FUSED kernel: quantizes straight from the fp32 gated value.
    """
    from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, fp8_max, fp8_min

    xf = x.to(torch.float32)
    zf = z.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    y = xf * torch.rsqrt(var + eps) * weight.to(torch.float32)
    if activation in ("swish", "silu"):
        y = y * (zf * torch.sigmoid(zf))
    elif activation == "sigmoid":
        y = y * torch.sigmoid(zf)
    q = (y * (1.0 / x_s.to(torch.float32))).clamp(fp8_min, fp8_max).to(fp8_dtype)
    return q, y


def _reference_unfused(x, z, weight, eps, x_s, activation):
    """The real unfused chain replaced by the fusion (incl. its bf16 rounding)."""
    from sglang.srt.layers.attention.fla.layernorm_gated import rms_norm_gated
    from sglang.srt.layers.quantization.fp8_kernel import static_quant_fp8

    y = rms_norm_gated(
        x=x,
        weight=weight,
        bias=None,
        z=z,
        eps=eps,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=True,
        activation=activation,
    )
    q, _ = static_quant_fp8(y, x_s, repeat_scale=False)
    return q


def _fp8_code_dist(a, b):
    """Distance in fp8 code space (monotone int mapping of e4m3 bits)."""

    def codes(t):
        u = t.view(torch.uint8).to(torch.int16)
        sign = u >> 7
        mag = u & 0x7F
        return torch.where(sign == 0, mag, -mag)

    return (codes(a) - codes(b)).abs()


def _make_case(M, N, activation, seed, device):
    from sglang.srt.layers.quantization.fp8_kernel import fp8_max

    torch.manual_seed(seed)
    x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
    z = torch.randn(M, N, dtype=torch.bfloat16, device=device)
    weight = torch.rand(N, dtype=torch.bfloat16, device=device) * 2 - 1
    # calibrated-style static scale: amax of the gated-normed activation / fp8_max
    _, y_ref = _reference_fp32(
        x, z, weight, 1e-6, torch.ones(1, device=device), activation
    )
    x_s = (y_ref.abs().max() / fp8_max).clamp(min=1e-4).reshape(1)
    return x, z, weight, x_s


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFusedGdnKernelNumerics(CustomTestCase):
    """Fused kernel vs an fp32 emulation of its math and the real unfused chain.

    Acceptance: fp8 codes >= 99.5% bitwise-exact and max 1 code off vs the fp32
    fused-math reference; only <= 1 code off vs the unfused chain, whose extra
    fp32->bf16->fp32 double rounding legitimately flips ~2% of codes by one.
    """

    # N = 128: qwen3.5 head_v_dim; 96: non-pow2 mask tail; 64/256: sweep.
    NS = (64, 96, 128, 256)
    # M = tokens * num_v_heads_local; 16384 = prefill-scale (2048 tokens x 8
    # heads at TP4); 1 and 7 cover ROWS_PER_BLOCK remainders.
    MS = (1, 7, 8, 64, 16384)
    SEEDS = tuple(range(5))
    EPS = 1e-6

    def _check_case(self, x, z, weight, x_s, activation, msg):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fused_rmsnorm_gated_static_fp8_quant,
        )

        q = fused_rmsnorm_gated_static_fp8_quant(
            x, z, weight, self.EPS, x_s, activation=activation
        )

        ref_q_unfused = _reference_unfused(
            x.contiguous(), z.contiguous(), weight, self.EPS, x_s, activation
        )
        ref_q_fp32, _ = _reference_fp32(x, z, weight, self.EPS, x_s, activation)

        d_f = _fp8_code_dist(q, ref_q_fp32)
        self.assertGreaterEqual((d_f == 0).float().mean().item(), 0.995, msg)
        self.assertLessEqual(d_f.max().item(), 1, msg)
        self.assertLessEqual(_fp8_code_dist(q, ref_q_unfused).max().item(), 1, msg)
        return q

    def test_parity_sweep(self):
        for activation in ACTIVATIONS:
            for N in self.NS:
                for M in self.MS:
                    for seed in self.SEEDS:
                        x, z, weight, x_s = _make_case(M, N, activation, seed, "cuda")
                        self._check_case(
                            x,
                            z,
                            weight,
                            x_s,
                            activation,
                            f"act={activation} M={M} N={N} seed={seed}",
                        )

    def test_near_clamp_outliers(self):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fp8_max,
            fused_rmsnorm_gated_static_fp8_quant,
        )

        for activation in ACTIVATIONS:
            x, z, weight, _ = _make_case(16, 128, activation, 0, "cuda")
            x[::3, ::7] = 100.0  # push part of the gated output past the clamp
            x_s = torch.tensor([1e-4], device="cuda")
            q = fused_rmsnorm_gated_static_fp8_quant(
                x, z, weight, self.EPS, x_s, activation=activation
            )
            ref_q = _reference_unfused(x, z, weight, self.EPS, x_s, activation)
            self.assertTrue(torch.isfinite(q.float()).all(), activation)
            self.assertEqual(q.float().abs().max().item(), fp8_max, activation)
            self.assertLessEqual(_fp8_code_dist(q, ref_q).max().item(), 1, activation)

    def test_gate_saturation(self):
        # sigmoid(z) -> 0 / 0.5 / 1 and swish z*sigmoid(z) -> 0 / 0 / z.
        for activation in ACTIVATIONS:
            for z_val in (-40.0, 0.0, 40.0):
                x, _, weight, x_s = _make_case(8, 128, activation, 0, "cuda")
                z = torch.full_like(x, z_val)
                self._check_case(
                    x, z, weight, x_s, activation, f"act={activation} z={z_val}"
                )

    def test_zero_pad_rows(self):
        # DP-attn pads core_attn_out with zero rows while z keeps real data;
        # a zero norm input must quantize to code 0 regardless of the gate.
        for activation in ACTIVATIONS:
            x, z, weight, x_s = _make_case(16, 128, activation, 0, "cuda")
            x[8:] = 0.0
            q = self._check_case(x, z, weight, x_s, activation, activation)
            self.assertEqual(q[8:].float().abs().max().item(), 0.0, activation)

    def test_strided_inputs(self):
        # Row-strided slices of a wider buffer (stride(-1) == 1) must go
        # through the row-stride path (no input copy) and still match the
        # reference on contiguous copies; the output is always contiguous.
        M, N = 64, 128
        torch.manual_seed(0)
        xw = torch.randn(M, 2 * N, dtype=torch.bfloat16, device="cuda")
        zw = torch.randn(M, 2 * N, dtype=torch.bfloat16, device="cuda")
        x, z = xw[:, :N], zw[:, :N]
        _, _, weight, x_s = _make_case(M, N, "swish", 0, "cuda")

        q = self._check_case(x, z, weight, x_s, "swish", "strided")
        self.assertEqual(q.stride(), (N, 1))

    def test_inputs_untouched(self):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fused_rmsnorm_gated_static_fp8_quant,
        )

        x, z, weight, x_s = _make_case(8, 128, "swish", 0, "cuda")
        x_snap, z_snap = x.clone(), z.clone()
        fused_rmsnorm_gated_static_fp8_quant(
            x, z, weight, self.EPS, x_s, activation="swish"
        )
        self.assertTrue(torch.equal(x, x_snap))
        self.assertTrue(torch.equal(z, z_snap))

    def test_device_scalar_scale_no_sync(self):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fused_rmsnorm_gated_static_fp8_quant,
        )

        x, z, weight, x_s = _make_case(8, 128, "swish", 0, "cuda")
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            fused_rmsnorm_gated_static_fp8_quant(
                x, z, weight, self.EPS, x_s, activation="swish"
            )
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()

    def test_out_proj_view_is_free(self):
        # The call site reshapes (M, N) -> (tokens, heads, N) -> (tokens,
        # heads*N) before out_proj; both must be views of the fp8 buffer.
        from sglang.srt.layers.quantization.fp8_kernel import (
            fused_rmsnorm_gated_static_fp8_quant,
        )

        tokens, heads, N = 16, 8, 128
        x, z, weight, x_s = _make_case(tokens * heads, N, "swish", 0, "cuda")
        q = fused_rmsnorm_gated_static_fp8_quant(
            x, z, weight, self.EPS, x_s, activation="swish"
        )
        self.assertEqual(q.stride(), (N, 1))
        v = q.reshape(tokens, heads, N).reshape(tokens, heads * N)
        self.assertEqual(v.data_ptr(), q.data_ptr())

    def test_empty_batch(self):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fp8_dtype,
            fused_rmsnorm_gated_static_fp8_quant,
        )

        x, z, weight, x_s = _make_case(1, 128, "swish", 0, "cuda")
        q = fused_rmsnorm_gated_static_fp8_quant(
            x[:0], z[:0], weight, self.EPS, x_s, activation="swish"
        )
        self.assertEqual(q.shape, (0, 128))
        self.assertEqual(q.dtype, fp8_dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestGdnNormOutProjWiring(CustomTestCase):
    """Forward-path wiring: ``Qwen3_5GatedDeltaNet._gated_norm_out`` must hand
    out_proj fp8 codes whenever the consumer scale resolves (no dead branch,
    no double quant) and fall back to the plain gated norm otherwise."""

    M, N = 32, 128

    def setUp(self):
        self._server_args_override = get_context().override_server_args()
        self._server_args_override.install()

    def tearDown(self):
        self._server_args_override.restore()

    def _make_gdn(self, input_scale):
        from sglang.srt.layers.attention.fla.layernorm_gated import (
            RMSNorm as RMSNormGated,
        )

        norm = RMSNormGated(
            self.N,
            eps=1e-6,
            group_size=None,
            norm_before_gate=True,
            device="cuda",
            dtype=torch.bfloat16,
        )
        torch.manual_seed(0)
        norm.weight.data.uniform_(-1, 1)
        out_proj = _fp8_linear(input_scale)
        with _sm90_flags():
            spec = _detect_gdn_norm_fused_quant(out_proj, norm)
        self.assertIsNotNone(spec)
        # Duck-typed layer: _gated_norm_out only touches these two attributes.
        gdn = SimpleNamespace(norm=norm, norm_fused_quant=spec)
        x = torch.randn(self.M, self.N, dtype=torch.bfloat16, device="cuda")
        z = torch.randn(self.M, self.N, dtype=torch.bfloat16, device="cuda")
        return gdn, out_proj, x, z

    def test_fused_branch_engages(self):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fp8_dtype,
            fused_rmsnorm_gated_static_fp8_quant,
        )
        from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

        scale = torch.tensor([0.02], dtype=torch.float32, device="cuda")
        gdn, _, x, z = self._make_gdn(scale)
        out = Qwen3_5GatedDeltaNet._gated_norm_out(gdn, x, z)
        # The fp8 dtype is what makes ModelOptFp8LinearMethod skip its own quant.
        self.assertEqual(out.dtype, fp8_dtype)
        ref = fused_rmsnorm_gated_static_fp8_quant(
            x, z, gdn.norm.weight, gdn.norm.eps, scale, activation=gdn.norm.activation
        )
        self.assertTrue(torch.equal(out.view(torch.uint8), ref.view(torch.uint8)))

    def test_unresolvable_scale_falls_back_then_reengages(self):
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
        from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

        # Never-loaded modelopt placeholder (float32.min) must fail the scale
        # guard and route through the plain fla norm.
        placeholder = torch.full(
            (1,), torch.finfo(torch.float32).min, dtype=torch.float32, device="cuda"
        )
        gdn, out_proj, x, z = self._make_gdn(placeholder)
        out = Qwen3_5GatedDeltaNet._gated_norm_out(gdn, x, z)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(out, gdn.norm(x, z)))

        # process_weights_after_loading replaces the Parameter object; the
        # spec must pick up the new scale on the next forward.
        out_proj.input_scale = torch.tensor([0.02], dtype=torch.float32, device="cuda")
        out = Qwen3_5GatedDeltaNet._gated_norm_out(gdn, x, z)
        self.assertEqual(out.dtype, fp8_dtype)


if __name__ == "__main__":
    unittest.main()
