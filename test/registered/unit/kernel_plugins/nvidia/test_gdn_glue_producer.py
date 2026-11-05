"""Alternative source glue producer; launches use explicit CUDA CPU doubles."""

from types import SimpleNamespace as NS

import pytest
import torch
from test_gdn_core_producer import tensor
from test_gdn_front_producer import model as _model_fixture
from test_gdn_front_producer import route as _route_fixture

from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation
from sglang.srt.kernel_plugins.nvidia import numeric_binding
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

model = _model_fixture
route = _route_fixture


def glue_args(args, activation="sigmoid"):
    layer = args["layer"]
    rows = args["x"].shape[0]
    h, hv = layer.num_k_heads, layer.num_v_heads
    return dict(
        layer=layer,
        forward_batch=args["forward_batch"],
        mixed_qkvz=tensor((rows, (2 * h + 2 * hv) * 128)),
        mixed_ba=tensor((rows, 2 * hv)),
        norm_weight=args["norm_weight"],
        norm_eps=args["norm_eps"],
        norm_activation=activation,
    )


@pytest.mark.parametrize("rows", [8, 12, 16])
@pytest.mark.parametrize("heads", [(4, 12), (16, 48)])
@pytest.mark.parametrize("activation,enum", [("silu", 1), ("sigmoid", 2)])
def test_complete_glue_caller_view_and_source_operands(
    route, monkeypatch, rows, heads, activation, enum
):
    backend, state, args = route(rows, heads)
    monkeypatch.setenv("SGLANG_GDN_FUSED_VERIFY_GLUE", "1")
    writes = []
    lookups = []

    def lookup(sig, *, prepared):
        lookups.append(sig)
        assert prepared
        return writes.append

    monkeypatch.setattr(numeric_binding, "numeric_operator", lookup)
    kwargs = glue_args(args, activation)
    out = backend.forward_numeric_verify_glue(**kwargs)
    assert len(writes) == len(lookups) == 1
    sig = lookups[0]
    io = writes[0]
    assert sig.operation == NumericOperation.GDN_VERIFY_GLUE and sig.variant == 0
    assert (
        sig.gate_scalar_type == 1
        and io.norm_activation == enum
        and io.conv_activation == 1
    )
    assert out is io.output and out.shape == (rows, heads[1] * 128)
    assert io.mixed_qkvz is kwargs["mixed_qkvz"] and io.mixed_ba is kwargs["mixed_ba"]
    assert io.conv_state is state.conv[0] and io.checkpoint is state.temporal
    assert io.intermediate_conv_window_out is state.intermediate_conv_window[0]
    assert io.intermediate_indices is backend.verify_intermediate_state_indices
    assert io.ring.is_flush is backend.state_pool.replayssm_is_flush
    assert (
        io.ring.is_flush.dtype == torch.int8 and io.norm_epsilon == kwargs["norm_eps"]
    )
    assert (
        io.launch.block_n,
        io.launch.num_warps,
        io.launch.num_stages,
        io.launch.hoist,
        io.launch.dot_precision,
        io.launch.prefetch,
        io.launch.historical_branch,
        io.launch.use_pdl,
        io.launch.trigger_early,
    ) == (0, 4, 1, False, 0, True, 1, True, True)


@pytest.mark.parametrize(
    "reason",
    [
        "off",
        "physical",
        "logical",
        "invalid_metadata",
        "tree",
        "fold",
        "norm_fp32",
        "ba_fp32",
        "checkpoint",
        "conv_weight",
        "cursor",
        "bad_capacity",
        "no_offer",
    ],
)
def test_glue_refusal_before_any_state_write(route, monkeypatch, reason):
    backend, state, args = route(8, (16, 48))
    kwargs = glue_args(args)
    monkeypatch.setenv("SGLANG_GDN_FUSED_VERIFY_GLUE", "1")
    if reason == "off":
        monkeypatch.setenv("SGLANG_GDN_FUSED_VERIFY_GLUE", "0")
    elif reason == "physical":
        kwargs["forward_batch"].forward_mode = NS(is_target_verify=lambda: False)
    elif reason == "logical":
        kwargs["forward_batch"].logical_forward_mode = NS(
            is_target_verify=lambda: False
        )
    elif reason == "invalid_metadata":
        backend.mis_metadata = object()
    elif reason == "tree":
        backend.forward_metadata.retrieve_next_token = tensor((1,), torch.int32)
    elif reason == "fold":
        backend.state_pool.replayssm_spec_fold = True
    elif reason == "norm_fp32":
        kwargs["norm_weight"] = tensor((128,), torch.float32)
    elif reason == "ba_fp32":
        kwargs["mixed_ba"] = tensor((8, 96), torch.float32)
    elif reason == "checkpoint":
        state.temporal = tensor((2, 48, 64, 128))
    elif reason == "conv_weight":
        args["layer"].conv_weights = tensor((10240, 4), torch.float32)
    elif reason == "cursor":
        backend.state_pool.replayssm_is_flush = tensor((8,), torch.int32)
    elif reason == "bad_capacity":
        state.replayssm_d = tensor((2, 48, 15, 128))
    calls = []
    monkeypatch.setattr(
        numeric_binding,
        "numeric_operator",
        lambda *a, **kw: None if reason == "no_offer" else calls.append,
    )
    assert backend.forward_numeric_verify_glue(**kwargs) is None
    assert calls == []


@pytest.mark.parametrize("mode", ["exact", "residual", "no_raw"])
def test_glue_preserves_exact_residual_raw_mode(route, monkeypatch, mode):
    backend, state, args = route(8, (16, 48))
    if mode == "exact":
        state.replayssm_beta = tensor((2, 48, 16), torch.float32)
    elif mode == "no_raw":
        state.replayssm_rawv = state.replayssm_rawk = None
    monkeypatch.setenv("SGLANG_GDN_FUSED_VERIFY_GLUE", "1")
    writes = []
    monkeypatch.setattr(
        numeric_binding, "numeric_operator", lambda *a, **kw: writes.append
    )
    assert backend.forward_numeric_verify_glue(**glue_args(args)) is not None
    r = writes[0].ring
    assert r.raw_v is state.replayssm_rawv and r.raw_k is state.replayssm_rawk
    assert r.beta is state.replayssm_beta


def test_startup_glue_independent_of_mk_and_front(model, monkeypatch):
    m, _, _, _, _ = model
    m.attn_tp_size = 1
    monkeypatch.setenv("SGLANG_Q38_MK_GDN", "0")
    monkeypatch.setenv("SGLANG_GDN_FUSED_VERIFY_GLUE", "1")
    sigs = m.numeric_signatures((4, 8, 12, 16, 20), 4)
    assert [(s.operation, s.rows, s.gate_scalar_type) for s in sigs] == [
        (NumericOperation.GDN_VERIFY_GLUE, m, 1) for m in (8, 12, 16)
    ]


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_model_glue_priority_skips_front_and_external_norm(
    model, route, monkeypatch, activation
):
    import sys

    m, _, _, calls, _ = model
    m.attn_tp_size = 1
    m.norm.activation = activation
    backend, state, args = route(8, (16, 48))
    m.attn = args["layer"]
    monkeypatch.setenv("SGLANG_GDN_FUSED_VERIFY_GLUE", "1")
    monkeypatch.setattr(
        sys.modules["sglang.srt.model_executor.forward_context"],
        "get_attn_backend",
        lambda: backend,
    )
    kwargs = glue_args(args, activation)

    def project(x):
        calls.append(("input_proj", x))
        return kwargs["mixed_qkvz"], kwargs["mixed_ba"]

    m._forward_input_proj = project
    writes = []

    def lookup(sig, **kw):
        assert sig.operation == NumericOperation.GDN_VERIFY_GLUE
        return writes.append

    monkeypatch.setattr(numeric_binding, "numeric_operator", lookup)
    m.numeric_signatures((args["x"].shape[0],), 4)
    result = m.forward(args["x"], args["forward_batch"])
    assert [c[0] for c in calls] == ["input_proj", "out_proj"]
    assert len(writes) == 1 and result is writes[0].output


def test_stable_row_indices_are_framework_prepared_before_glue(route, monkeypatch):
    backend, _, args = route(8, (16, 48))
    monkeypatch.setenv("SGLANG_GDN_FUSED_VERIFY_GLUE", "1")
    replacement = tensor((2,), torch.int32)
    events = []
    namespace = backend.forward_numeric_verify_glue.__func__.__globals__
    monkeypatch.setitem(namespace, "pp_spec_stable_rows_enabled", lambda: True)

    def select(indices, requests, mask, size):
        events.append("metadata")
        assert indices is backend.verify_intermediate_state_indices
        assert requests is args["forward_batch"].req_pool_indices
        assert mask.shape == (2,) and mask.dtype == torch.bool and size == 8
        return replacement

    monkeypatch.setitem(namespace, "select_verify_intermediate_state_indices", select)
    backend.req_to_token_pool = NS(size=8)

    def lookup(*a, **kw):
        events.append("lookup")

        def run(io):
            events.append("run")
            assert io.intermediate_indices is replacement

        return run

    monkeypatch.setattr(numeric_binding, "numeric_operator", lookup)
    assert backend.forward_numeric_verify_glue(**glue_args(args)) is not None
    assert events == ["lookup", "metadata", "run"]


@pytest.mark.parametrize("reason", ["off", "no_offer", "quant_tuple"])
def test_glue_intact_stock_chain_after_projection(model, route, monkeypatch, reason):
    import sys

    m, _, _, calls, namespace = model
    backend, state, args = route(8, (16, 48))
    m.attn_tp_size = 1
    monkeypatch.setenv("SGLANG_Q38_MK_GDN", "0")
    monkeypatch.setenv("SGLANG_GDN_FUSED_VERIFY_GLUE", "0" if reason == "off" else "1")
    kwargs = glue_args(args)
    m._fused_in_proj_weight = args["packed_weight"]

    class Attention(NS):
        def __call__(self, *a, **kw):
            calls.append(("stock_attention",))
            return tensor((8, 48, 128))

    m.attn = Attention(**vars(args["layer"]))

    class Norm(NS):
        def __call__(self, x, z):
            calls.append(("stock_norm",))
            return x

    m.norm = Norm(**vars(m.norm))

    def project(x):
        calls.append(("input_proj",))
        return kwargs["mixed_qkvz"], kwargs["mixed_ba"]

    m._forward_input_proj = project

    def split(*a):
        calls.append(("stock_split",))
        return (
            tensor((8, 10240)),
            tensor((8, 48, 128)),
            tensor((8, 48)),
            tensor((8, 48)),
        )

    namespace.update(
        _is_xpu=False,
        _gdn_decode_fused_proj_conv=False,
        _is_cpu=True,
        fused_qkvzba_split_reshape_cat_contiguous=split,
    )
    args["forward_batch"].logical_forward_mode.is_extend_without_speculative = lambda: (
        False
    )
    monkeypatch.setattr(
        sys.modules["sglang.srt.model_executor.forward_context"],
        "get_attn_backend",
        lambda: backend,
    )
    monkeypatch.setattr(numeric_binding, "numeric_operator", lambda *a, **kw: None)
    hidden = (args["x"], args["x"]) if reason == "quant_tuple" else args["x"]
    out = m.forward(hidden, args["forward_batch"])
    assert out.shape == (8, 6144)
    assert [c[0] for c in calls] == [
        "input_proj",
        "stock_split",
        "stock_attention",
        "stock_norm",
        "out_proj",
    ]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
