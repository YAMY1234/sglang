"""Source-program routing parity, including ordinary CORE1 with failed FRONT."""

from types import SimpleNamespace as NS

import pytest
from test_gdn_core_producer import tensor
from test_gdn_front_producer import model as _model_fixture

from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    qwen3_5_gdn_prefill_projection_views,
)
from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as Op

model = _model_fixture
from capability_fixtures import contract_spec
from source_policy_reference import standalone_norm_route_477

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 12, 16, 32, 64])
@pytest.mark.parametrize("front", ["not_selected", "fallback"])
def test_original_row_matrix_keeps_normalized_return_program_separate(
    model, monkeypatch, rows, front
):
    m, backend, batch, events, namespace = model
    monkeypatch.setenv("SGLANG_Q38_MK_GDN_M32_NORM", "1")
    monkeypatch.setenv("SGLANG_Q38_MK_GDN_WIDE", "1")
    monkeypatch.setenv("Q38_GDN_FRONT", "1")
    callback = contract_spec().numeric_capability
    m._numeric_capability_spec = NS(
        numeric_capability=lambda query: (
            callback(query)
            and (front != "not_selected" or query.operation != Op.GDN_FRONT)
        )
    )
    batch.batch_size = max(1, rows // 4)
    batch.forward_mode.is_decode = lambda: False
    batch.logical_forward_mode.is_extend_without_speculative = lambda: False

    class Attention(NS):
        def __call__(self, *a, **kw):
            events.append(("ordinary_attention",))
            return tensor((1, rows, 12, 128))

    m.attn = Attention(**vars(m.attn))

    class Norm(NS):
        def __call__(self, x, z):
            events.append(("outer_norm",))
            return x

    m.norm = Norm(**vars(m.norm))
    m.numeric_signatures((rows,), 4)
    hidden = tensor((rows, 2560))
    packed, ba = tensor((rows, 4096)), tensor((rows, 24))
    m._forward_input_proj = lambda x: (packed, ba)
    namespace.update(
        _is_xpu=False,
        _is_cpu=True,
        _gdn_decode_fused_proj_conv=False,
        qwen3_5_gdn_prefill_projection_views=qwen3_5_gdn_prefill_projection_views,
    )

    def front_attempt(*a, **kw):
        events.append(("front_refused",))
        return None

    backend.forward_numeric_inproj_conv = front_attempt

    def normalized(**kw):
        assert kw["q38_m32_return_normalized"] is True
        events.append(("normalized_return",))
        return tensor((1, rows, 12, 128)), True

    backend.forward = normalized
    expected = standalone_norm_route_477(
        m, hidden, batch, False, tensor((rows, 12, 128))
    )
    output = m.forward(hidden, batch)
    assert output.shape == (rows, 1536)
    normalized_calls = sum(e[0] == "normalized_return" for e in events)
    assert normalized_calls == int(expected)
    assert sum(e[0] == "ordinary_attention" for e in events) == int(not expected)
    assert sum(e[0] == "outer_norm" for e in events) == int(not expected)
    ordinary_core1 = any(
        s.operation == Op.GDN_CORE and s.variant == 1 and s.rows == rows
        for s in m._numeric_gdn_signatures
    )
    if rows in (4, 8, 12, 16):
        assert ordinary_core1 and not m._numeric_gdn_standalone_norm_signatures
        assert normalized_calls == 0  # The original320 widening regression.


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
