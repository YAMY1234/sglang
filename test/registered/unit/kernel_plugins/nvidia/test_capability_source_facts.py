"""Actual startup publishers transmit dimensions; refusal leaves stock selected."""

from types import SimpleNamespace as NS

import pytest
import torch

from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation as Op
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import binding, numeric_binding


@pytest.fixture
def recorder(monkeypatch):
    seen = []
    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)
    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")
    monkeypatch.setattr(
        numeric_binding,
        "numeric_registry",
        NS(get=lambda _: pytest.fail("startup queried registry")),
    )
    return seen, NS(numeric_capability=lambda query: seen.append(query) or False)


def test_hc_passes_actual_geometry_and_does_not_create_stream_on_refusal(recorder):
    from sglang.srt.layers.hyperconnection import GatedResidual

    seen, spec = recorder
    producer = NS(
        params_dtype=torch.bfloat16,
        hc_count=3,
        hidden_size=3072,
        _numeric_mix_variant=lambda: 0,
        _gate_stream=None,
        _numeric_capability_spec=spec,
        config=NS(hc_lowrank=7),
    )
    assert GatedResidual.numeric_signatures(producer, (5, 33), 2) == ()
    assert len(seen) == 8 and producer._gate_stream is None
    assert all(
        dict(q.attributes)["stream_count"] == 3
        and dict(q.attributes)["hidden_size"] == 3072
        for q in seen
    )


@pytest.mark.parametrize("tokens", [1, 2, 4, 8])
def test_local_experts_reports_preparation_program_and_actual_dimensions(
    recorder, tokens
):
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    seen, spec = recorder
    producer = NS(
        hidden_size=3072,
        intermediate_size_per_partition=256,
        num_experts=128,
        top_k=7,
        _numeric_capability_spec=spec,
    )
    with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"SGLANG_Q38_MK_MOE": "1"}):
            assert FusedMoE.numeric_signatures(producer, (tokens * 3,), tokens) == ()
    assert len(seen) == 3
    for query in seen:
        a = dict(query.attributes)
        assert query.operation == Op.LOCAL_EXPERTS
        assert (a["hidden_size"], a["intermediate_size"], a["experts"], a["topk"]) == (
            3072,
            256,
            128,
            7,
        )
        assert (a["tokens_per_request"], a["is_target_verify"], a["batch_size"]) == (
            tokens,
            int(tokens > 1),
            3,
        )


def test_missing_callback_never_declares_hc(recorder):
    from sglang.srt.layers.hyperconnection import GatedResidual

    seen, _ = recorder
    producer = NS(
        params_dtype=torch.bfloat16,
        hc_count=4,
        hidden_size=2560,
        _numeric_mix_variant=lambda: 0,
        _gate_stream=None,
        _numeric_capability_spec=NS(),
        config=NS(hc_lowrank=8),
    )
    assert GatedResidual.numeric_signatures(producer, (1, 4, 32), 4) == ()
    assert seen == []


def forbid_forward_queries(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("forward repeated capability/spec resolution")

    monkeypatch.setattr(numeric_binding, "numeric_capability_supported", forbidden)
    monkeypatch.setattr(numeric_binding, "signature_capability_supported", forbidden)
    monkeypatch.setattr(binding, "find_spec", forbidden)


@pytest.mark.parametrize("family", ["HC", "LOCAL"])
def test_actual_forward_chains_do_not_query_or_resolve_spec(monkeypatch, family):
    forbid_forward_queries(monkeypatch)
    if family == "HC":
        from test_hc_admission import TestActualHcAdmission

        from sglang.srt.layers import hyperconnection

        test = TestActualHcAdmission()
        _, gates, _ = test.execute_mix(
            hyperconnection, test.layer(hyperconnection, True), rows=8
        )
        assert len(gates) == 1
    else:
        from test_expert_finalize_chain import exercise

        events, sink, _, _, _ = exercise()
        assert len(sink) == 1 and events.count(("finalize", "main")) == 1


def test_qsa_selected_forward_does_not_query_again(monkeypatch):
    from test_sparse_score_producer import flags, indexer, tensors

    from sglang.kernels.plugin_kernels.nvidia import numeric as api

    obj = indexer()
    calls = []
    with flags():
        signatures = obj.numeric_preparation_signatures(32768, 16)
        registry = api.NumericRegistry()
        for signature in signatures:
            registry.register(
                api.NumericOffer(signature, lambda io: calls.append(type(io)))
            )
        monkeypatch.setattr(numeric_binding, "numeric_registry", registry)
        forbid_forward_queries(monkeypatch)
        assert obj._numeric_score_select(*tensors(8192)).shape == (4, 512)
    assert calls == [api.SparseScoreIO, api.SparseSelectIO]


from test_router_quant import packet_constructor as _packet_constructor_fixture
from test_router_quant import setup as _router_setup_fixture

router_setup = _router_setup_fixture
packet_constructor = _packet_constructor_fixture


def test_router_selected_forward_does_not_query_again(
    router_setup, packet_constructor, monkeypatch
):
    from test_router_quant import tensor

    from sglang.kernels.plugin_kernels.nvidia import numeric as api
    from sglang.srt.kernel_plugins.nvidia import router_quant

    producer, registry, *_ = router_setup
    seen = []
    for signature in producer._numeric_router_quant_signatures:
        registry.register(api.NumericOffer(signature, seen.append))
    forbid_forward_queries(monkeypatch)
    logits, packet = router_quant.produce(producer, tensor((4, 2560)))
    assert logits.shape == (4, 512) and len(seen) == 1
    assert packet.x_fp4 is seen[0].packed_out


from test_gdn_front_producer import model as _gdn_model_fixture
from test_gdn_front_producer import route as _gdn_route_fixture

gdn_model = _gdn_model_fixture
gdn_route = _gdn_route_fixture


def test_gdn_selected_model_and_backend_do_not_query_again(
    gdn_model, gdn_route, monkeypatch
):
    from test_gdn_core_producer import tensor
    from test_gdn_front_producer import test_both_lookups_before_front_state_write

    model, backend, batch, events, _ = gdn_model
    backend.forward_numeric_inproj_conv = lambda *a, **kw: tensor((4, 12, 128))
    forbid_forward_queries(monkeypatch)
    assert model._try_numeric_gdn_front(tensor((4, 2560)), batch).shape == (4, 1536)
    assert events[0][0] == "out_proj"
    test_both_lookups_before_front_state_write(gdn_route, monkeypatch, 4, (4, 12))


def test_callback_call_sites_are_startup_only():
    import ast
    from pathlib import Path

    root = Path(numeric_binding.__file__).parents[3]
    allowed = {
        "numeric_binding.py": {"signature_capability_supported"},
        "mtp_capture_seed.py": {"prepare"},
        "mtp_deferred_kv.py": {"prepare"},
        "pool_attention.py": {"pool_signatures"},
        "router_quant.py": {"numeric_signatures"},
        "permuted_finalize.py": {"numeric_signatures"},
        "hyperconnection.py": {"numeric_signatures"},
        "qsa_indexer.py": {"numeric_preparation_signatures", "_qpf_signatures"},
        "gdn_backend.py": {"numeric_native_pdl_signatures"},
        "layer.py": {"numeric_signatures"},
        "qwen3_5.py": {"numeric_signatures"},
        "qwen4_exp.py": {"numeric_signatures"},
    }
    calls = []

    class Visitor(ast.NodeVisitor):
        def __init__(self, path):
            self.path = path
            self.stack = []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id in (
                "numeric_capability_supported",
                "signature_capability_supported",
            ):
                assert any(
                    name in allowed.get(self.path.name, set()) for name in self.stack
                ), (self.path, self.stack)
                calls.append((self.path.name, tuple(self.stack)))
            self.generic_visit(node)

    for path in root.rglob("*.py"):
        if path.name in allowed:
            Visitor(path).visit(ast.parse(path.read_text()))
    assert {path for path, _ in calls} == set(allowed)


from test_native_gdn_pdl_producer import context as _native_context_fixture

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

native_context = _native_context_fixture


def test_native_gdn_selected_forward_does_not_query_again(native_context, monkeypatch):
    from test_native_gdn_pdl_producer import prepare, setup, tensor

    from sglang.kernels.plugin_kernels.nvidia import numeric as api
    from sglang.srt.layers.attention.linear import gdn_backend

    model, backend, layer, batch, _ = setup(32, (4, 12))
    (signature,) = prepare(model, backend, monkeypatch, widths=(32,)).numeric_signatures
    calls = []
    native_context.register(api.NumericOffer(signature, calls.append))
    monkeypatch.setattr(gdn_backend, "causal_conv1d_update", lambda x, *a, **kw: x)
    forbid_forward_queries(monkeypatch)
    result = backend.forward_extend(
        layer, batch, tensor((32, 2560)), tensor((32, 12)), tensor((32, 12))
    )
    assert result.shape == (1, 32, 12, 128) and len(calls) == 1


def test_permuted_finalize_prepares_and_consumes_without_query_after_startup(
    monkeypatch,
):
    import sys

    from test_permuted_finalize import inputs

    import sglang.srt.runtime_context as runtime
    from sglang.kernels.plugin_kernels.nvidia import numeric as api
    from sglang.srt.kernel_plugins.nvidia import permuted_finalize as impl

    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)
    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")
    monkeypatch.setattr(impl, "_SOURCE_ENABLED", True)
    monkeypatch.setattr(impl, "_SOURCE_EXTEND", False)
    monkeypatch.setattr(runtime, "get_parallel", lambda: NS(moe_ep_size=1))
    monkeypatch.setitem(
        sys.modules, "sglang.kernels.jit.utils", NS(is_arch_support_pdl=lambda: True)
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.batch_invariant_ops",
        NS(is_batch_invariant_mode_enabled=lambda: False),
    )
    producer, hidden, gate, shared, deferred = inputs()
    registry = api.NumericRegistry()
    calls = []
    for signature in producer._numeric_permuted_finalize_signatures:
        registry.register(api.NumericOffer(signature, calls.append))
    monkeypatch.setattr(numeric_binding, "numeric_registry", registry)
    forbid_forward_queries(monkeypatch)
    plan = impl.prepare(
        producer, hidden, use_fused_gate=True, ordered_finalize_sink=None
    )
    assert plan is not None
    assert impl.finish(plan, hidden, gate, shared, deferred).shape == hidden.shape
    assert len(calls) == 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hc_actual_dtype_preserves_stream_lifecycle_and_fp16_stock(monkeypatch, dtype):
    from capability_fixtures import contract_spec

    from sglang.srt.layers import hyperconnection as hc

    monkeypatch.setattr(torch, "compile", lambda f: f)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = hc.HyperConnectionConfig(
        hidden_size=2560,
        hc_count=4,
        hc_lowrank=8,
        hc_per_branch_norm=True,
        params_dtype=dtype,
    )
    layer = hc.GatedResidual(config, use_combine=True)
    assert layer.params_dtype == config.params_dtype
    seen = []
    stream_calls = []
    callback = contract_spec().numeric_capability
    layer._numeric_capability_spec = NS(
        numeric_capability=lambda query: seen.append(query) or callback(query)
    )
    layer._numeric_mix_variant = lambda: 0
    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)
    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def make_stream(device):
        stream_calls.append(device)
        return object()

    monkeypatch.setattr(hc, "_hc_gate_stream", make_stream)
    signatures = layer.numeric_signatures((1, 4, 32), 4)
    assert bool(signatures) == (dtype == torch.bfloat16)
    assert len(stream_calls) == int(dtype == torch.bfloat16)
    assert bool(seen) == (dtype == torch.bfloat16)
    if dtype == torch.float16:
        assert layer._gate_stream is None
        forbid_forward_queries(monkeypatch)
        monkeypatch.setattr(
            numeric_binding,
            "numeric_operator",
            lambda *a, **kw: pytest.fail("FP16 queried numeric offer"),
        )
        monkeypatch.setattr(hc, "flydsl_hc_mix_supported", lambda *a: False)
        monkeypatch.setattr(hc, "fused_hc_mix_supported", lambda *a: False)
        x = torch.zeros((4, 10240), dtype=dtype)
        result = layer.mix(x, normalized_input=x)
        assert result[0].shape == (4, 2560) and result[0].dtype == dtype


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
