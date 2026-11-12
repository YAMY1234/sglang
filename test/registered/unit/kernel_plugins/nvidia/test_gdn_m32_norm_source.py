"""Original TP4 M32 norm consumer and one-convolution recurrence fallback.

Actual imported backend methods run with tensor/launch metadata doubles; the
model consumer executes installed source. No GPU numerical equivalence claim.
"""

from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch
from test_gdn_backend_imports import setup
from test_gdn_core_producer import tensor
from test_gdn_front_producer import model as _model_fixture

from sglang.kernels.ops.attention.linear import gdn_replayssm_spec_decode as raw
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    qwen3_5_gdn_prefill_projection_views,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation
from sglang.srt.kernel_plugins.nvidia import numeric_binding
from sglang.srt.layers.attention.linear import gdn_backend as impl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

model = _model_fixture


@pytest.mark.parametrize(
    "bad",
    [None, "tp1", "hk", "off", "mkoff", "fp32", "silu", "group", "tokens", "widths"],
)
def test_m32_finite_startup_only_source_norm(model, monkeypatch, bad):
    m, _, _, _, _ = model
    monkeypatch.setenv("SGLANG_Q38_MK_GDN_M32_NORM", "0" if bad == "off" else "1")
    if bad == "tp1":
        m.attn_tp_size = 1
    elif bad == "hk":
        m.attn.num_k_heads, m.attn.num_v_heads = 16, 48
    elif bad == "mkoff":
        monkeypatch.setenv("SGLANG_Q38_MK_GDN", "0")
    elif bad == "fp32":
        m.norm.weight = tensor((128,), torch.float32)
    elif bad == "silu":
        m.norm.activation = "silu"
    elif bad == "group":
        m.norm.group_size = 128
    sigs = m.numeric_signatures(
        (4, 16) if bad == "widths" else (4, 16, 32), 8 if bad == "tokens" else 4
    )
    actual = [
        (s.operation, s.rows, s.variant, s.gate_scalar_type)
        for s in sigs
        if s.rows == 32
    ]
    assert actual == ([(NumericOperation.GDN_CORE, 32, 1, 1)] if bad is None else [])


@pytest.mark.parametrize(
    "reason",
    [
        None,
        "nooffer",
        "flagoff",
        "norm_fp32",
        "a_fp32",
        "ring",
        "cursor",
        "normalized_flagoff",
        "exception",
    ],
)
def test_actual_backend_conv_once_then_exact_norm_or_native_core_once(
    monkeypatch, reason
):
    backend, layer, fb = setup(32)
    state = backend.state_layer_cache(0)
    layer._numeric_gdn_signatures = frozenset(
        s
        for s in layer._numeric_gdn_signatures
        if s.variant == 1 and reason != "flagoff"
    )
    if reason == "flagoff":
        layer._numeric_gdn_standalone_norm_signatures = frozenset()
    backend.prefill_graph_metadata = None
    backend.requires_contiguous_prefill_state = False
    backend.req_to_token_pool = NS(size=64)
    backend._use_strided_target_verify_qkv = True
    monkeypatch.setenv("SGLANG_Q38_MK_GDN", "1")
    monkeypatch.setenv(
        "SGLANG_Q38_MK_GDN_M32_NORM", "0" if reason == "flagoff" else "1"
    )
    monkeypatch.setenv("SGLANG_Q38_MK_GDN_WIDE", "0")
    monkeypatch.setattr(impl, "pp_spec_stable_rows_enabled", lambda: False)
    events = []
    seen = []

    def conv(x, *args, **kwargs):
        events.append("conv")
        return x

    monkeypatch.setattr(impl, "causal_conv1d_update", conv)
    monkeypatch.setattr(
        raw, "gdn_replayssm_spec_decode", lambda **kw: events.append("native_core")
    )

    def lookup(sig, *, prepared):
        seen.append(sig)
        assert (sig.operation, sig.rows, sig.variant) == (
            NumericOperation.GDN_CORE,
            32,
            1,
        )
        if reason == "nooffer" or not prepared:
            return None

        def run(io):
            events.append("provider_core")
            assert io.z is z and io.norm_weight is weight and io.epsilon == 3e-6
            assert io.raw_out is not io.normalized_out
            if reason == "exception":
                raise RuntimeError("provider failed after launch")

        return run

    monkeypatch.setattr(numeric_binding, "numeric_operator", lookup)
    if reason == "ring":
        state = replace(state, replayssm_g=tensor((4, 12, 16), torch.bfloat16))
        backend.state_layer_cache = lambda _: state
    elif reason == "cursor":
        backend.state_pool.replayssm_is_flush = tensor((8,), torch.int32)
    z = tensor((32, 12, 128))
    weight = tensor((128,), torch.float32 if reason == "norm_fp32" else torch.bfloat16)
    a = tensor((32, 12), torch.float32 if reason == "a_fp32" else torch.bfloat16)
    b = tensor((32, 12))
    kwargs = dict(
        q38_mk_z=z,
        q38_mk_norm_weight=weight,
        q38_mk_norm_eps=3e-6,
        q38_m32_return_normalized=reason != "normalized_flagoff",
    )
    if reason == "exception":
        with pytest.raises(RuntimeError, match="provider failed"):
            backend.forward_extend(layer, fb, tensor((32, 2560)), a, b, **kwargs)
        assert events == ["conv", "provider_core"]
        return
    result = backend.forward_extend(layer, fb, tensor((32, 2560)), a, b, **kwargs)
    if reason is None:
        assert isinstance(result, tuple) and result[1] is True
        assert result[0].shape == (1, 32, 12, 128)
        assert events == ["conv", "provider_core"]
    else:
        assert isinstance(result, torch.Tensor) and result.shape == (1, 32, 12, 128)
        assert events == ["conv", "native_core"]
    assert all(s.variant == 1 for s in seen)  # M32 never queries raw0.


@pytest.mark.parametrize("selected", [True, False])
def test_model_source_consumer_normalizes_exactly_once(model, monkeypatch, selected):
    m, backend, fb, events, namespace = model
    monkeypatch.setenv("SGLANG_Q38_MK_GDN_M32_NORM", "1")
    monkeypatch.setenv("Q38_GDN_FRONT", "1")
    fb.batch_size = 8
    m.numeric_signatures((32,), 4)
    fb.forward_mode.is_decode = lambda: False
    fb.logical_forward_mode.is_extend_without_speculative = lambda: False
    packed = tensor((32, 4096))
    ba = tensor((32, 24))
    m._forward_input_proj = lambda x: (packed, ba)
    namespace.update(
        _is_xpu=False,
        _is_cpu=True,
        _gdn_decode_fused_proj_conv=False,
        qwen3_5_gdn_prefill_projection_views=qwen3_5_gdn_prefill_projection_views,
    )

    class Norm(NS):
        def __call__(self, x, z):
            events.append(("norm",))
            return x

    m.norm = Norm(**vars(m.norm))
    raw_out = tensor((1, 32, 12, 128))

    def forward(**kw):
        events.append(("backend",))
        assert kw["q38_mk_norm_weight"] is m.norm.weight
        assert kw["q38_m32_return_normalized"] is True
        assert (
            kw["q38_mk_z"].untyped_storage()._cdata == packed.untyped_storage()._cdata
        )
        return (raw_out, True) if selected else raw_out

    backend.forward = forward
    out = m.forward(tensor((32, 2560)), fb)
    assert out.shape == (32, 1536)
    assert [e[0] for e in events] == (
        ["backend", "out_proj"] if selected else ["backend", "norm", "out_proj"]
    )


def test_quantized_tuple_keeps_native_chain_even_when_m32_gate_enabled(
    model, monkeypatch
):
    m, backend, fb, events, namespace = model
    monkeypatch.setenv("SGLANG_Q38_MK_GDN_M32_NORM", "1")
    monkeypatch.setenv("Q38_GDN_FRONT", "1")
    fb.batch_size = 8
    fb.forward_mode.is_decode = lambda: False
    fb.logical_forward_mode.is_extend_without_speculative = lambda: False
    m._forward_input_proj = lambda x: (tensor((32, 4096)), tensor((32, 24)))
    namespace.update(
        _is_xpu=False,
        _is_cpu=True,
        _gdn_decode_fused_proj_conv=False,
        qwen3_5_gdn_prefill_projection_views=qwen3_5_gdn_prefill_projection_views,
    )

    class Attention(NS):
        def __call__(self, *args, **kwargs):
            events.append(("native_attention",))
            return tensor((1, 32, 12, 128))

    m.attn = Attention(**vars(m.attn))

    class Norm(NS):
        def __call__(self, x, z):
            events.append(("norm",))
            return x

    m.norm = Norm(**vars(m.norm))

    def bad_forward(**kwargs):
        raise AssertionError("tuple must not select M32 norm producer")

    backend.forward = bad_forward
    out = m.forward((tensor((32, 2560)), tensor((32, 2560))), fb)
    assert out.shape == (32, 1536)
    assert [e[0] for e in events] == ["native_attention", "norm", "out_proj"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
