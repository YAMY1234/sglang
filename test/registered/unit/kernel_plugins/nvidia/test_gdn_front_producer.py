"""Installed producer methods; CUDA metadata/launches are explicit CPU doubles."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from source_policy_reference import gdn_front_launch
from test_gdn_core_producer import CudaMetadata, tensor

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    GdnFrontVariant,
    NumericOperation,
)
from sglang.srt.kernel_plugins.nvidia import numeric_binding
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

ROOT = Path(__file__).resolve().parents[5]


def load_methods(path, class_name, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert len(methods) == len(names)
    namespace = dict(namespace)
    node = ast.ClassDef(
        name="Installed", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["Installed"], namespace


@pytest.mark.parametrize(
    "family,rows,channels,override,legacy,variant,stages",
    [
        ("mk", 4, 2560, None, False, 0, 6),
        ("mk", 4, 2560, "12", False, 1, 12),
        ("mk", 8, 2560, "12", False, 0, 6),
        ("mk", 4, 10240, "12", False, 0, 6),
        ("legacy", 4, 10240, None, False, 2, 8),
        ("legacy", 4, 10240, None, True, 3, 12),
    ],
)
def test_exact_source_stages_and_neutral_fields(
    family, rows, channels, override, legacy, variant, stages
):
    v, launch = gdn_front_launch(family, rows, channels, override, legacy)
    assert v == variant and launch.num_stages == stages
    assert (
        launch.block_n,
        launch.num_warps,
        launch.hoist,
        launch.dot_precision,
        launch.prefetch,
        launch.historical_branch,
        launch.use_pdl,
        launch.trigger_early,
    ) == (0, 0, False, 0, False, 0, True, False)


@pytest.mark.parametrize("override", ["", "0", "7", "bad"])
def test_unsupported_stage_never_rounded(override):
    assert gdn_front_launch("mk", 4, 2560, override, False) is None


@pytest.fixture
def route(monkeypatch):
    class State:
        pass

    backend_type, _ = load_methods(
        "python/sglang/srt/layers/attention/linear/gdn_backend.py",
        "GDNAttnBackend",
        {
            "forward_numeric_inproj_conv",
            "numeric_verify_glue_supported",
            "forward_numeric_verify_glue",
        },
        dict(
            torch=torch,
            os=os,
            MambaPool=NS(SpeculativeState=State),
            pp_spec_stable_rows_enabled=lambda: False,
        ),
    )

    def build(rows=4, heads=(4, 12)):
        h, hv = heads
        c = (2 * h + hv) * 128
        batch = rows // 4
        state = State()
        state.temporal = tensor((2, hv, 128, 128))
        state.conv = [tensor((2, c, 3))]
        state.intermediate_conv_window = [tensor((4, 4, c, 3))]
        state.replayssm_d = tensor((2, hv, 16, 128))
        state.replayssm_k = tensor((2, h, 16, 128))
        state.replayssm_g = tensor((2, hv, 16), torch.float32)
        state.replayssm_rawv = tensor((2, hv, 16, 128))
        state.replayssm_rawk = tensor((2, h, 16, 128))
        state.replayssm_beta = None
        backend = backend_type()
        backend.mis_metadata = None
        backend.forward_metadata = NS(
            query_start_loc=tensor((batch + 1,), torch.int32),
            mamba_cache_indices=tensor((batch,), torch.int64),
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            retrieve_parent_token=None,
        )
        backend.verify_intermediate_state_indices = tensor((batch,), torch.int32)
        backend.state_layer_cache = lambda layer_id: state
        backend.state_pool = NS(
            replayssm_cache_base=tensor((8,), torch.int32),
            replayssm_spec_write_pos=tensor((8,), torch.int32),
            replayssm_is_flush=tensor((8,), torch.int8),
            replayssm_spec_fold=False,
            replayssm_is_kda=False,
        )
        layer = NS(
            layer_id=0,
            num_k_heads=h,
            num_q_heads=h,
            num_v_heads=hv,
            head_k_dim=128,
            head_q_dim=128,
            head_v_dim=128,
            q_dim=h * 128,
            k_dim=h * 128,
            v_dim=hv * 128,
            activation="silu",
            A_log=tensor((hv,), torch.float32),
            dt_bias=tensor((hv,), torch.bfloat16),
            conv_weights=tensor((c, 4)),
            bias=None,
        )
        mode = NS(is_target_verify=lambda: True)
        fb = NS(
            spec_info=NS(draft_token_num=4),
            batch_size=batch,
            forward_mode=mode,
            logical_forward_mode=mode,
            req_pool_indices=tensor((batch,), torch.int64),
        )
        args = dict(
            layer=layer,
            forward_batch=fb,
            x=tensor((rows, 2560)),
            packed_weight=tensor((c + hv * 130, 2560)),
            norm_weight=tensor((128,)),
            norm_eps=0.000003,
        )
        args["front_variant"], args["front_launch"] = gdn_front_launch(
            "mk", rows, c, None, False
        )
        from sglang.kernels.plugin_kernels.nvidia.numeric import GdnSignature

        layer._numeric_gdn_signatures = frozenset(
            (
                GdnSignature(
                    NumericOperation.GDN_FRONT,
                    rows,
                    h,
                    hv,
                    128,
                    128,
                    4,
                    1,
                    variant=args["front_variant"],
                    gate_scalar_type=3,
                ),
                GdnSignature(
                    NumericOperation.GDN_VERIFY_GLUE,
                    rows,
                    h,
                    hv,
                    128,
                    128,
                    4,
                    1,
                    gate_scalar_type=1,
                ),
            )
        )
        return backend, state, args

    old_empty = torch.empty

    def allocate(shape, **kwargs):
        # Real tensor strides/dtypes, meta device only; no CUDA execution.
        return old_empty(shape, **kwargs).as_subclass(CudaMetadata)

    monkeypatch.setattr(torch, "empty", allocate)
    return build


@pytest.mark.parametrize("rows", [4, 8, 12])
@pytest.mark.parametrize("heads", [(4, 12), (16, 48)])
def test_both_lookups_before_front_state_write(route, monkeypatch, rows, heads):
    backend, state, args = route(rows, heads)
    events = []
    views = {}

    def lookup(sig, *, prepared):
        assert prepared
        events.append(("lookup", sig.operation))

        def run(io):
            events.append(("run", sig.operation))
            views[sig.operation] = io

        return run

    monkeypatch.setattr(numeric_binding, "numeric_operator", lookup)
    output = backend.forward_numeric_inproj_conv(**args)
    assert events == [
        ("lookup", NumericOperation.GDN_FRONT),
        ("lookup", NumericOperation.GDN_CORE),
        ("run", NumericOperation.GDN_FRONT),
        ("run", NumericOperation.GDN_CORE),
    ]
    front = views[NumericOperation.GDN_FRONT]
    core = views[NumericOperation.GDN_CORE]
    assert output is core.normalized_out
    assert core.qkv is front.convolved_out
    assert core.checkpoint is state.temporal and core.ring.d is state.replayssm_d
    assert core.ring.d_low is state.replayssm_rawv
    assert core.ring.k_low is state.replayssm_rawk
    assert (
        core.z.stride(0)
        == core.a.stride(0)
        == core.b.stride(0)
        == front.packed_out.stride(0)
    )
    assert core.z.untyped_storage()._cdata == front.packed_out.untyped_storage()._cdata
    assert core.a.untyped_storage()._cdata == front.packed_out.untyped_storage()._cdata
    assert core.epsilon == args["norm_eps"] and front.pad_slot == -1
    assert front.intermediate_conv_window_out is state.intermediate_conv_window[0]
    assert front.conv_state is state.conv[0]


@pytest.mark.parametrize(
    "missing", [NumericOperation.GDN_FRONT, NumericOperation.GDN_CORE]
)
def test_missing_either_offer_has_no_writes(route, monkeypatch, missing):
    backend, _, args = route()
    writes = []
    monkeypatch.setattr(
        numeric_binding,
        "numeric_operator",
        lambda sig, **kw: None if sig.operation == missing else writes.append,
    )
    assert backend.forward_numeric_inproj_conv(**args) is None
    assert writes == []


@pytest.mark.parametrize(
    "bad",
    [
        "checkpoint",
        "ring",
        "cursor",
        "tree",
        "tokens",
        "weight",
        "norm",
        "stages",
        "activation",
    ],
)
def test_complete_preflight_before_any_lookup_or_write(route, monkeypatch, bad):
    backend, state, args = route()
    if bad == "checkpoint":
        state.temporal = tensor((2, 12, 64, 128))
    elif bad == "ring":
        state.replayssm_g = tensor((2, 12, 16), torch.bfloat16)
    elif bad == "cursor":
        backend.state_pool.replayssm_is_flush = tensor((8,), torch.int32)
    elif bad == "tree":
        backend.forward_metadata.retrieve_next_token = tensor((1,), torch.int32)
    elif bad == "tokens":
        args["forward_batch"].spec_info.draft_token_num = 8
    elif bad == "weight":
        args["packed_weight"] = tensor((4120, 1280))
    elif bad == "norm":
        args["norm_weight"] = tensor((128,), torch.float32)
    elif bad == "stages":
        args["front_variant"] = GdnFrontVariant.MK_STAGES_12
    elif bad == "activation":
        args["layer"].activation = "relu"
    events = []
    monkeypatch.setattr(
        numeric_binding, "numeric_operator", lambda *a, **kw: events.append(a)
    )
    assert backend.forward_numeric_inproj_conv(**args) is None
    assert events == []


@pytest.fixture
def model(monkeypatch):
    from sglang.srt.kernel_plugins.nvidia import binding

    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)
    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")
    import sys
    from types import ModuleType

    monkeypatch.setenv("SGLANG_Q38_MK_GDN", "1")
    monkeypatch.delenv("SGLANG_GDN_FUSED_VERIFY_GLUE", raising=False)
    model_type, namespace = load_methods(
        "python/sglang/srt/models/qwen3_5.py",
        "Qwen3_5GatedDeltaNet",
        {
            "numeric_signatures",
            "_numeric_gdn_norm_supported",
            "_numeric_gdn_m32_norm_supported",
            "_numeric_gdn_glue_eligible",
            "_numeric_gdn_glue_layer_supported",
            "_try_numeric_gdn_verify_glue",
            "_try_numeric_gdn_front",
            "forward",
        },
        dict(
            torch=torch,
            os=os,
            _is_cuda=True,
            ForwardBatch=object,
            _GDN_FUSED_QKVZBA_RATIOS=(2, 3, 4, 8),
            _GDN_UCACHE_FUSED_NORM=False,
            GDN_VERIFY_NORM_CFG=None,
            _fp8_tuple_input=lambda *a: False,
            get_bool_env_var=lambda key, default: (
                os.environ.get(key, default).lower() in ("true", "1")
            ),
        ),
    )
    calls = []
    backend = NS()
    module = ModuleType("sglang.srt.model_executor.forward_context")
    module.get_attn_backend = lambda: backend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    m = model_type()
    h, hv = 4, 12
    from capability_fixtures import contract_spec

    m._numeric_capability_spec = contract_spec()
    m.hidden_size = 2560
    m.attn = NS(
        num_k_heads=h, num_v_heads=hv, q_dim=h * 128, k_dim=h * 128, v_dim=hv * 128
    )
    m.head_k_dim = m.head_v_dim = 128
    m.attn_tp_size = 4
    m.num_k_heads = 16
    m.num_v_heads = 48
    m.activation = "silu"
    m._fused_in_proj_weight = tensor((4120, 2560))
    m.norm = NS(
        weight=tensor((128,)),
        activation="sigmoid",
        eps=0.000003,
        bias=None,
        group_size=None,
        norm_before_gate=True,
    )

    def project(x):
        calls.append(("out_proj", x))
        return x, None

    m.out_proj = project
    mode = NS(is_target_verify=lambda: True)
    fb = NS(forward_mode=mode, logical_forward_mode=mode, batch_size=1)
    m.numeric_signatures((4, 8, 12, 16, 32), 4)
    return m, backend, fb, calls, namespace


def test_model_publishes_finite_actual_stage_pair(model, monkeypatch):
    m, _, _, _, _ = model
    monkeypatch.setenv("Q38_GDNPROJ_FRONT_STAGES", "12")
    signatures = m.numeric_signatures((1, 4, 8, 12, 16, 32), 4)
    fronts = [s for s in signatures if s.operation == NumericOperation.GDN_FRONT]
    assert [(s.rows, s.variant, s.gate_scalar_type) for s in fronts] == [
        (4, 1, 3),
        (8, 0, 3),
        (12, 0, 3),
    ]
    norm = [
        s
        for s in signatures
        if s.operation == NumericOperation.GDN_CORE and s.variant == 1
    ]
    assert [(s.rows, s.gate_scalar_type) for s in norm] == [(4, 1), (8, 1), (12, 1)]
    monkeypatch.setenv("Q38_GDNPROJ_FRONT_STAGES", "7")
    assert [
        s.rows
        for s in m.numeric_signatures((4, 8), 4)
        if s.operation == NumericOperation.GDN_FRONT
    ] == [8]


@pytest.mark.parametrize(
    "bad",
    ["off", "norm_fp32", "norm_silu", "group", "norm_before", "weight", "noncuda"],
)
def test_startup_does_not_declare_unavailable_source_norm(model, monkeypatch, bad):
    m, _, _, _, namespace = model
    if bad == "off":
        monkeypatch.setenv("SGLANG_Q38_MK_GDN", "0")
    elif bad == "norm_fp32":
        m.norm.weight = tensor((128,), torch.float32)
    elif bad == "norm_silu":
        m.norm.activation = "silu"
    elif bad == "group":
        m.norm.group_size = 128
    elif bad == "norm_before":
        m.norm.norm_before_gate = False
    elif bad == "weight":
        m._fused_in_proj_weight = None
    elif bad == "noncuda":
        namespace["_is_cuda"] = False
    sigs = m.numeric_signatures((4, 8), 4)
    assert not any(
        s.operation == NumericOperation.GDN_FRONT or s.variant == 1 for s in sigs
    )


@pytest.mark.parametrize("rows", [4, 8, 12])
def test_model_success_skips_external_norm_once_and_keeps_output_projection(
    model, rows
):
    m, backend, fb, calls, _ = model
    normalized = tensor((rows, 12, 128))

    def front(*args, **kwargs):
        calls.append(("front", args, kwargs))
        return normalized

    backend.forward_numeric_inproj_conv = front
    x = tensor((rows, 2560))
    fb.batch_size = rows // 4
    m.numeric_signatures((rows,), 4)
    output = m.forward(x, fb)
    assert [c[0] for c in calls] == ["front", "out_proj"]
    assert output.shape == (rows, 1536)
    assert output.untyped_storage()._cdata == normalized.untyped_storage()._cdata
    assert calls[0][2]["norm_weight"] is m.norm.weight
    assert calls[0][2]["norm_eps"] == m.norm.eps
    # norm is deliberately not callable: any second norm would fail this test.


@pytest.mark.parametrize(
    "reason", ["off", "nooffer", "legacy", "glue", "wide", "badstage"]
)
def test_model_refusal_reaches_intact_stock_before_projection(
    model, monkeypatch, reason
):
    m, backend, fb, calls, _ = model

    class StockReached(Exception):
        pass

    def stock(x):
        calls.append(("stock_projection", x))
        raise StockReached

    m._forward_input_proj = stock
    backend.forward_numeric_inproj_conv = lambda *a, **kw: calls.append(
        ("front_attempt",)
    )
    rows = 4
    if reason == "off":
        monkeypatch.setenv("SGLANG_Q38_MK_GDN", "0")
    elif reason == "legacy":
        monkeypatch.setenv("SGLANG_Q38_MK_GDN", "0")
        monkeypatch.setenv("Q38_GDN_INPROJ_CONV", "1")
        m.attn_tp_size = 1
    elif reason == "glue":
        rows = 8
        m.attn_tp_size = 1
        fb.batch_size = 2
        monkeypatch.setenv("SGLANG_GDN_FUSED_VERIFY_GLUE", "1")
        backend.numeric_verify_glue_supported = lambda *a, **kw: True
    elif reason == "wide":
        rows = 16
        fb.batch_size = 4
    elif reason == "badstage":
        monkeypatch.setenv("Q38_GDNPROJ_FRONT_STAGES", "7")
    m.numeric_signatures((rows,), 4)
    with pytest.raises(StockReached):
        m.forward(tensor((rows, 2560)), fb)
    assert [c[0] for c in calls] == (
        ["front_attempt", "stock_projection"]
        if reason == "nooffer"
        else ["stock_projection"]
    )


@pytest.fixture(autouse=True)
def capability_provider_enabled(monkeypatch):
    from sglang.srt.kernel_plugins.nvidia import binding

    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)
    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
