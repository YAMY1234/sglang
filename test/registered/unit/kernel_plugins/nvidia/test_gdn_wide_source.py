"""Source-bounded TP4 M16 frontend preparation; no CUDA launches."""

import pytest
import torch
from test_gdn_front_producer import model as _model_fixture
from test_gdn_front_producer import route as _route_fixture

from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation
from sglang.srt.kernel_plugins.nvidia import numeric_binding

model = _model_fixture
route = _route_fixture
from test_gdn_core_producer import tensor

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


@pytest.mark.parametrize(
    "tp,heads,wide,expected",
    [
        (4, (4, 12), True, True),
        (4, (4, 12), False, False),
        (1, (16, 48), True, False),
        (1, (4, 12), True, False),
        (4, (16, 48), True, False),
    ],
)
def test_startup_m16_is_source_tp4_wide_and_finite(
    model, monkeypatch, tp, heads, wide, expected
):
    m, _, _, _, _ = model
    m.attn_tp_size = tp
    m.attn.num_k_heads, m.attn.num_v_heads = heads
    monkeypatch.setenv("SGLANG_Q38_MK_GDN_WIDE", "1" if wide else "0")
    sigs = m.numeric_signatures((4, 16, 32), 4)
    keys = {(s.operation, s.rows, s.variant) for s in sigs}
    assert ((NumericOperation.GDN_FRONT, 16, 0) in keys) == expected
    assert ((NumericOperation.GDN_CORE, 16, 1) in keys) == expected
    assert ((NumericOperation.GDN_CORE, 16, 0) in keys) == expected
    assert not any(s.rows == 32 for s in sigs)
    assert not any(s.rows == 16 for s in m.numeric_signatures((4, 8), 4))


@pytest.mark.parametrize("bad", ["silu", "norm_fp32", "group", "off"])
def test_wide_flag_does_not_authorize_norm(model, monkeypatch, bad):
    m, _, _, _, _ = model
    monkeypatch.setenv("SGLANG_Q38_MK_GDN_WIDE", "1")
    if bad == "silu":
        m.norm.activation = "silu"
    elif bad == "norm_fp32":
        m.norm.weight = tensor((128,), torch.float32)
    elif bad == "group":
        m.norm.group_size = 128
    else:
        monkeypatch.setenv("SGLANG_Q38_MK_GDN", "0")
    sigs = m.numeric_signatures((16,), 4)
    assert not any(
        s.operation == NumericOperation.GDN_FRONT or s.variant == 1 for s in sigs
    )


@pytest.mark.parametrize(
    "missing", [None, NumericOperation.GDN_FRONT, NumericOperation.GDN_CORE]
)
def test_m16_all_offers_before_state_writes(route, monkeypatch, missing):
    backend, state, args = route(16, (4, 12))
    monkeypatch.setenv("SGLANG_Q38_MK_GDN_WIDE", "1")
    events = []
    ios = {}

    def lookup(sig, *, prepared):
        assert prepared
        events.append(("lookup", sig.operation, sig.rows, sig.variant))
        if sig.operation == missing:
            return None

        def run(io):
            events.append(("run", sig.operation))
            ios[sig.operation] = io

        return run

    monkeypatch.setattr(numeric_binding, "numeric_operator", lookup)
    out = backend.forward_numeric_inproj_conv(**args)
    if missing is not None:
        assert out is None
        assert not any(e[0] == "run" for e in events)
    else:
        assert events == [
            ("lookup", NumericOperation.GDN_FRONT, 16, 0),
            ("lookup", NumericOperation.GDN_CORE, 16, 1),
            ("run", NumericOperation.GDN_FRONT),
            ("run", NumericOperation.GDN_CORE),
        ]
        front, core = ios[NumericOperation.GDN_FRONT], ios[NumericOperation.GDN_CORE]
        assert out is core.normalized_out and core.qkv is front.convolved_out
        assert (
            core.z.stride(0)
            == core.a.stride(0)
            == core.b.stride(0)
            == front.packed_out.stride(0)
        )
        assert core.ring.d is state.replayssm_d


@pytest.mark.parametrize(
    "reason", ["off", "heads", "norm_fp32", "metadata", "tree", "tokens"]
)
def test_m16_bad_policy_or_metadata_has_no_writes(route, monkeypatch, reason):
    backend, _, args = route(16, (16, 48) if reason == "heads" else (4, 12))
    monkeypatch.setenv("SGLANG_Q38_MK_GDN_WIDE", "0" if reason == "off" else "1")
    if reason in ("off", "heads"):
        args["layer"]._numeric_gdn_signatures = frozenset()
    if reason == "norm_fp32":
        args["norm_weight"] = tensor((128,), torch.float32)
    elif reason == "metadata":
        backend.forward_metadata.query_start_loc = tensor((9,), torch.int32)
    elif reason == "tree":
        backend.forward_metadata.retrieve_next_token = tensor((1,), torch.int32)
    elif reason == "tokens":
        args["forward_batch"].spec_info.draft_token_num = 8
    events = []
    monkeypatch.setattr(
        numeric_binding, "numeric_operator", lambda *a, **kw: events.append
    )
    assert backend.forward_numeric_inproj_conv(**args) is None
    assert events == []


def test_m16_model_skips_external_norm_only_after_source_front_success(
    model, monkeypatch
):
    m, backend, fb, calls, _ = model
    monkeypatch.setenv("SGLANG_Q38_MK_GDN_WIDE", "1")
    fb.batch_size = 4
    normalized = tensor((16, 12, 128))

    def front(*args, **kwargs):
        calls.append(("front",))
        return normalized

    backend.forward_numeric_inproj_conv = front
    m.numeric_signatures((16,), 4)
    out = m.forward(tensor((16, 2560)), fb)
    assert [c[0] for c in calls] == ["front", "out_proj"]
    assert out.untyped_storage()._cdata == normalized.untyped_storage()._cdata


def test_glue_is_not_extended_to_m32(model, monkeypatch):
    m, _, _, _, _ = model
    m.attn_tp_size = 1
    monkeypatch.setenv("SGLANG_GDN_FUSED_VERIFY_GLUE", "1")
    monkeypatch.setenv("SGLANG_Q38_MK_GDN_WIDE", "1")
    sigs = m.numeric_signatures((8, 12, 16, 32), 4)
    assert [
        s.rows for s in sigs if s.operation == NumericOperation.GDN_VERIFY_GLUE
    ] == [8, 12, 16]


@pytest.mark.parametrize("wide,selected", [(False, False), (True, True)])
def test_actual_imported_m16_raw_core_preserves_source_wide_gate(
    monkeypatch, wide, selected
):
    from test_gdn_backend_imports import setup

    from sglang.kernels.ops.attention.linear import gdn_replayssm_spec_decode as raw
    from sglang.srt.layers.attention.linear import gdn_backend as impl

    backend, layer, fb = setup(16)
    state = backend.state_layer_cache(0)
    if not wide:
        layer._numeric_gdn_signatures = frozenset(
            s for s in layer._numeric_gdn_signatures if s.rows != 16
        )
    monkeypatch.setenv("SGLANG_Q38_MK_GDN_WIDE", "1" if wide else "0")
    events = []

    def lookup(sig, *, prepared):
        events.append(("lookup", sig.variant, prepared))
        return (lambda io: events.append(("numeric", io))) if prepared else None

    monkeypatch.setattr(numeric_binding, "numeric_operator", lookup)
    monkeypatch.setattr(
        raw, "gdn_replayssm_spec_decode", lambda **kw: events.append(("native", kw))
    )
    out = backend._replayssm_target_verify(
        layer=layer,
        query=tensor((1, 16, 4, 128)),
        key=tensor((1, 16, 4, 128)),
        value=tensor((1, 16, 12, 128)),
        a=tensor((16, 12)),
        b=tensor((16, 12)),
        mamba_pool=backend.state_pool,
        layer_cache=state,
        cache_indices=backend.forward_metadata.mamba_cache_indices,
        replay_indices=fb.req_pool_indices,
        query_start_loc=backend.forward_metadata.query_start_loc,
        draft_token_num=4,
        packed_qkv=tensor((16, 2560)),
    )
    assert backend._replayssm_target_verify.__func__.__globals__ is vars(impl)
    assert [e[0] for e in events] == (["lookup", "numeric"] if selected else ["native"])
    if selected:
        assert events[0] == ("lookup", 0, True)
    if selected:
        io = events[1][1]
        assert io.z is io.norm_weight is io.normalized_out is None
        assert io.raw_out.shape == (16, 12, 128)
    assert out.shape == (1, 16, 12, 128)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
