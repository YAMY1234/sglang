"""Actual F producer/facade control flow; CUDA/FI/kernel work is CPU doubled."""

import __future__

import ast
import sys
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    FinalizePermutedSharedIO,
    FinalizePermutedSignature,
    NumericOffer,
    NumericOperation,
    NumericRegistry,
    ScalarType,
)
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import binding, numeric_binding
from sglang.srt.kernel_plugins.nvidia import permuted_finalize as impl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class Metadata(torch.Tensor):
    events = []

    @property
    def is_cuda(self):
        return True

    def record_stream(self, stream):
        self.events.append(("record", self.label, stream.name))


def tensor(shape, dtype=torch.bfloat16, label="tensor"):
    value = torch.zeros(shape, dtype=dtype).as_subclass(Metadata)
    value.label = label
    return value


def inputs(rows=4, weight_type=torch.bfloat16):
    hidden = tensor((rows, 2560), label="hidden")
    gate = tensor((2560,), label="gate")
    shared = tensor((rows, 2560), label="shared")
    deferred = NS(
        gemm2_out=tensor((rows * 10, 2560), label="gemm2"),
        expert_weights=tensor((rows, 10), weight_type, "weights"),
        expanded_idx_to_permuted_idx=tensor((rows * 10,), torch.int32, "permutation"),
        top_k=10,
    )
    if rows:
        deferred.expanded_idx_to_permuted_idx[0] = -1
    producer = NS(
        shared_expert=object(),
        shared_expert_gate=NS(weight=gate.view(1, -1)),
        _uses_nvfp4_trtllm_glue=lambda: True,
        gate=NS(weight=tensor((512, 2560))),
        experts=NS(top_k=10),
        tp_size=1,
    )
    from capability_fixtures import contract_spec

    producer._numeric_capability_spec = contract_spec()
    impl.numeric_signatures(producer, (hidden.shape[0],))
    return producer, hidden, gate, shared, deferred


def extracted(name, scope=None):
    path = Path(impl.__file__).parents[2] / "models/qwen2_moe.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen2MoeSparseMoeBlock"
    )
    node = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    scope = {} if scope is None else scope
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        scope,
    )
    return scope[name]


@pytest.fixture(autouse=True)
def source_launch(monkeypatch):
    monkeypatch.setattr(impl, "_SOURCE_ENABLED", True)
    monkeypatch.setattr(impl, "_SOURCE_EXTEND", False)
    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")
    import sglang.srt.runtime_context as ctx

    monkeypatch.setattr(ctx, "get_parallel", lambda: NS(moe_ep_size=1))
    monkeypatch.setitem(
        sys.modules, "sglang.kernels.jit.utils", NS(is_arch_support_pdl=lambda: True)
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.batch_invariant_ops",
        NS(is_batch_invariant_mode_enabled=lambda: False),
    )


@pytest.mark.parametrize("batch_invariant", [False, True])
@pytest.mark.parametrize("pdl", [False, True])
def test_actual_startup_finite_widths_and_source_launch_values(
    monkeypatch, batch_invariant, pdl
):
    monkeypatch.setitem(
        sys.modules, "sglang.kernels.jit.utils", NS(is_arch_support_pdl=lambda: pdl)
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.batch_invariant_ops",
        NS(is_batch_invariant_mode_enabled=lambda: batch_invariant),
    )
    obj, *_ = inputs()
    fn = extracted("numeric_signatures")
    signatures = fn(obj, (0, 1, 4, 16, 32, 64, 65), 4)
    assert isinstance(signatures, tuple) and len(signatures) == 10
    assert {s.rows for s in signatures} == {1, 4, 16, 32, 64}
    assert {s.weight_scalar_type for s in signatures} == {
        ScalarType.BF16,
        ScalarType.FP32,
    }
    for s in signatures:
        assert isinstance(s, FinalizePermutedSignature)
        assert s.operation == NumericOperation.FINALIZE_PERMUTED_SHARED
        assert (
            s.hidden_size == 2560
            and s.topk == 10
            and s.scalar_type == ScalarType.BF16
            and s.variant == 0
        )
        assert s.num_warps == (8 if batch_invariant else 16) and s.use_pdl == int(pdl)
        assert type(s.use_pdl) is int


@pytest.mark.parametrize(
    "reason", ["master", "source", "shared", "gate", "gate_dtype", "native"]
)
def test_source_disabled_never_declares_or_requests_deferred(monkeypatch, reason):
    obj, hidden, *_ = inputs()
    if reason == "master":
        monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "0")
    if reason == "source":
        monkeypatch.setattr(impl, "_SOURCE_ENABLED", False)
    if reason == "shared":
        obj.shared_expert = None
    if reason == "gate":
        obj.shared_expert_gate = None
    if reason == "gate_dtype":
        obj.shared_expert_gate.weight = obj.shared_expert_gate.weight.float()
    if reason == "native":
        obj._uses_nvfp4_trtllm_glue = lambda: False

    def forbidden(*a, **k):
        raise AssertionError("no capability probe after source rejection")

    monkeypatch.setattr(numeric_binding, "numeric_operator", forbidden)
    assert impl.numeric_signatures(obj, (1, 4, 64)) == ()
    assert (
        impl.prepare(obj, hidden, use_fused_gate=True, ordered_finalize_sink=None)
        is None
    )


def test_extend_fast_path_offers_only_rows_below_its_floor(monkeypatch):
    # The source's extend path adds a register cap from its floor upward.
    monkeypatch.setattr(impl, "_SOURCE_EXTEND", True)
    monkeypatch.setattr(impl, "_SOURCE_EXTEND_MIN_ROWS", 16)
    obj, hidden, *_ = inputs()
    signatures = impl.numeric_signatures(obj, (1, 4, 15, 16, 32, 64))
    assert {s.rows for s in signatures} == {1, 4, 15}
    seen = []

    def lookup(signature, *, prepared):
        seen.append(signature.rows)
        return lambda io: None

    monkeypatch.setattr(numeric_binding, "numeric_operator", lookup)
    assert impl.prepare(obj, hidden, use_fused_gate=True, ordered_finalize_sink=None)
    assert seen == [4, 4]
    wide = tensor((16, 2560), label="hidden")
    assert (
        impl.prepare(obj, wide, use_fused_gate=True, ordered_finalize_sink=None) is None
    )
    assert seen == [4, 4]


@pytest.mark.parametrize(
    "reason",
    [
        "rows0",
        "rows65",
        "hidden",
        "cpu",
        "dtype",
        "strided",
        "gate_shape",
        "gate_dtype",
        "gate_strided",
        "not_fused",
        "ordered_sink",
    ],
)
def test_layout_and_owner_preflight_happen_before_offer(monkeypatch, reason):
    obj, hidden, gate, *_ = inputs(
        0 if reason == "rows0" else 65 if reason == "rows65" else 4
    )
    if reason == "hidden":
        hidden = tensor((4, 1280))
    if reason == "cpu":
        hidden = torch.zeros((4, 2560), dtype=torch.bfloat16)
    if reason == "dtype":
        hidden = hidden.float()
    if reason == "strided":
        hidden = tensor((4, 5120))[:, ::2]
    if reason == "gate_shape":
        obj.shared_expert_gate.weight = tensor((1, 1280))
    if reason == "gate_dtype":
        obj.shared_expert_gate.weight = gate.float().view(1, -1)
    if reason == "gate_strided":
        obj.shared_expert_gate.weight = tensor((1, 5120))[:, ::2]

    def forbidden(*a, **k):
        raise AssertionError("lookup before complete source layout")

    monkeypatch.setattr(numeric_binding, "numeric_operator", forbidden)
    assert (
        impl.prepare(
            obj,
            hidden,
            use_fused_gate=reason != "not_fused",
            ordered_finalize_sink=[] if reason == "ordered_sink" else None,
        )
        is None
    )


@pytest.mark.parametrize("missing", [ScalarType.BF16, ScalarType.FP32])
def test_both_possible_native_weight_capabilities_required_before_deferral(
    monkeypatch, missing
):
    obj, hidden, *_ = inputs()
    seen = []

    def lookup(signature, *, prepared):
        seen.append(signature.weight_scalar_type)
        return None if signature.weight_scalar_type == missing else lambda io: None

    monkeypatch.setattr(numeric_binding, "numeric_operator", lookup)
    assert (
        impl.prepare(obj, hidden, use_fused_gate=True, ordered_finalize_sink=None)
        is None
    )
    assert seen == [ScalarType.BF16, ScalarType.FP32]


@pytest.mark.parametrize("rows", [1, 4, 16, 32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_actual_registry_facade_and_caller_output_identity(monkeypatch, rows, dtype):
    obj, hidden, gate, shared, deferred = inputs(rows, dtype)
    registry = NumericRegistry()
    seen = []

    def run(io):
        assert isinstance(io, FinalizePermutedSharedIO)
        assert (
            io.hidden is hidden
            and io.gate_weight is gate
            and io.shared is shared
            and io.gemm2_out is deferred.gemm2_out
            and io.expert_weights is deferred.expert_weights
            and io.expanded_idx_to_permuted_idx is deferred.expanded_idx_to_permuted_idx
        )
        assert io.output is not shared and io.output.shape == shared.shape
        seen.append(io)
        io.output.fill_(7)

    for signature in impl.numeric_signatures(obj, (rows,)):
        registry.register(NumericOffer(signature, run))
    monkeypatch.setattr(numeric_binding, "numeric_registry", registry)
    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)
    plan = impl.prepare(obj, hidden, use_fused_gate=True, ordered_finalize_sink=None)
    assert plan is not None
    got = impl.finish(plan, hidden, gate, shared, deferred)
    assert got is seen[0].output and len(seen) == 1 and torch.all(got == 7)
    with pytest.raises(TypeError, match="view"):
        plan.bf16(NS())


@pytest.mark.parametrize(
    "reason",
    ["dtype", "weights", "permutation", "topk", "shared", "gemm2_shape", "wrong_H"],
)
def test_invalid_native_return_metadata_fails_before_writes(monkeypatch, reason):
    obj, hidden, gate, shared, deferred = inputs()
    if reason == "dtype":
        deferred.gemm2_out = deferred.gemm2_out.float()
    if reason == "weights":
        deferred.expert_weights = deferred.expert_weights.half()
    if reason == "permutation":
        deferred.expanded_idx_to_permuted_idx = (
            deferred.expanded_idx_to_permuted_idx.long()
        )
    if reason == "topk":
        deferred.top_k = 9
    if reason == "shared":
        shared = None
    if reason == "gemm2_shape":
        deferred.gemm2_out = tensor((40, 1280))
    if reason == "wrong_H":
        deferred.expert_weights = tensor((4, 9))

    def forbidden(*a):
        raise AssertionError("provider write or fallback after invalid native ABI")

    plan = impl._Plan(forbidden, forbidden, deferred.top_k)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm",
        NS(finalize_flashinfer_trtllm_deferred_output=forbidden),
    )
    with pytest.raises(RuntimeError, match="incompatible metadata"):
        impl.finish(plan, hidden, gate, shared, deferred)


def test_noncontiguous_native_output_attempts_original_native_rounding_then_gate_once(
    monkeypatch,
):
    obj, hidden, gate, shared, deferred = inputs()
    events = []
    deferred.gemm2_out = tensor((40, 5120))[:, ::2]
    routed = tensor((4, 2560))

    def native(d, s):
        assert d is deferred and s is None
        events.append("native-finalize-BF16")
        return routed

    def add(h, g, s, out):
        assert h is hidden and g is gate and s is shared and out is routed
        events.append("original-gate-add")

    def forbidden(*a):
        raise AssertionError("numeric provider after unsupported layout")

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm",
        NS(finalize_flashinfer_trtllm_deferred_output=native),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.kernels.ops.elementwise.elementwise",
        NS(fused_gate_sigmoid_mul_add=add),
    )
    assert (
        impl.finish(
            impl._Plan(forbidden, forbidden, deferred.top_k),
            hidden,
            gate,
            shared,
            deferred,
        )
        is routed
    )
    assert events == ["native-finalize-BF16", "original-gate-add"]


def test_after_native_protocol_rejection_or_provider_partial_write_never_retry(
    monkeypatch,
):
    obj, hidden, gate, shared, deferred = inputs()
    events = []

    def fail(io):
        events.append("provider-partial")
        raise RuntimeError("partial write")

    def native(*a):
        events.append("native")
        raise RuntimeError("native protocol refused")

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm",
        NS(finalize_flashinfer_trtllm_deferred_output=native),
    )
    with pytest.raises(RuntimeError, match="partial write"):
        impl.finish(
            impl._Plan(fail, fail, deferred.top_k), hidden, gate, shared, deferred
        )
    assert events == ["provider-partial"]
    deferred.gemm2_out = tensor((40, 5120))[:, ::2]
    with pytest.raises(RuntimeError, match="native protocol refused"):
        impl.finish(
            impl._Plan(fail, fail, deferred.top_k), hidden, gate, shared, deferred
        )
    assert events == ["provider-partial", "native"]


@pytest.mark.parametrize("rows", [1, 4, 16, 32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "mode",
    [
        "active",
        "nooffer",
        "half",
        "masteroff",
        "sourceoff",
        "nonbypassed",
        "ordered",
        "nocapture",
        "ineligible",
        "providerfail",
    ],
)
def test_actual_complete_moe_forward_to_native_dispatch_and_numeric_consumer(
    monkeypatch, rows, dtype, mode
):
    obj, hidden, gate, shared, deferred = inputs(rows, dtype)
    events = []
    Metadata.events = events

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            events.append(("wait", self.name, other.name))

    main, alt = Stream("main"), Stream("alt")
    active = [main]

    @contextmanager
    def stream(s):
        old = active[0]
        active[0] = s
        try:
            yield
        finally:
            active[0] = old

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: active[0])
    monkeypatch.setattr(torch.cuda, "stream", stream)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    if mode == "masteroff":
        monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "0")
    if mode == "sourceoff":
        monkeypatch.setattr(impl, "_SOURCE_ENABLED", False)
    if mode == "ineligible":
        obj._uses_nvfp4_trtllm_glue = lambda: False
    routed = tensor((rows, 2560), label="routed")

    class Experts:
        top_k = 10
        intermediate_size_per_partition = 192
        num_experts = 512

        def __call__(self, *a):
            events.append(("experts-stock", active[0].name))
            return routed

        def forward_deferred_finalize(self, *a):
            events.append(("experts-deferred", active[0].name))
            return deferred

    obj.experts = Experts()
    obj.supports_deferred_finalize = True
    obj.enable_shared_expert_fusion = False
    obj.gate = lambda h: (tensor((rows, 512)), None)
    obj.gate.weight = tensor((512, 2560))
    obj.topk = lambda *a: object()
    obj._use_mega_moe = False
    obj.alt_stream = alt
    obj.reduce_results = False
    obj._shared_expert_input = lambda hidden: hidden.clone()
    obj._forward_shared_experts = lambda *a, **k: shared
    registry = NumericRegistry()

    def provider(io):
        events.append(("provider", active[0].name))
        assert (
            io.hidden.data_ptr() == hidden.data_ptr()
            and io.shared is shared
            and io.gemm2_out is deferred.gemm2_out
        )
        if mode == "providerfail":
            raise RuntimeError("partial kernel failure")
        io.output.fill_(1)

    for sig in impl.numeric_signatures(obj, (rows,)):
        if mode == "nooffer" or (
            mode == "half" and sig.weight_scalar_type == ScalarType.FP32
        ):
            continue
        registry.register(NumericOffer(sig, provider))
    monkeypatch.setattr(numeric_binding, "numeric_registry", registry)
    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)

    @contextmanager
    def capture(sink, **kwargs):
        yield

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.kernel_plugins.nvidia.local_experts",
        NS(
            prepare_local_shared_finalize=lambda *a, **kw: None,
            capture_local_finalize=capture,
        ),
    )
    backend = NS(
        **{
            name: lambda: False
            for name in ["is_deepep", "is_deepep_v2", "is_mori", "is_ascend_fuseep"]
        }
    )
    # The runtime routed-input prequant has its own test; keep this chain's stock quantize.
    monkeypatch.setenv("SGLANG_OPT_MOE_NVFP4_PREQUANT_OVERLAP", "0")
    scope = {
        "torch": torch,
        "envs": envs,
        "_use_aiter": False,
        "_SGLANG_EXPERIMENTAL_LORA_OPTI": False,
        "TopKOutputChecker": NS(
            format_is_bypassed=lambda _: mode != "nonbypassed",
            format_is_standard=lambda _: mode == "nonbypassed",
        ),
        "get_moe_a2a_backend": lambda: backend,
        "get_is_capture_mode": lambda: mode != "nocapture",
        "get_exec": lambda: NS(moe=NS(enable_paged_experts=False)),
        "use_intel_amx_backend": lambda _: False,
        "is_npu": lambda: False,
        "get_bool_env_var": lambda *a: mode == "ordered",
        "fused_gate_sigmoid_mul_add": lambda *a: events.append(
            ("stock-gate-add", active[0].name)
        ),
    }
    obj._forward_router_experts = MethodType(
        extracted("_forward_router_experts", scope), obj
    )
    obj.forward_normal_dual_stream = MethodType(
        extracted("forward_normal_dual_stream", scope), obj
    )
    fn = extracted("forward", scope)
    if mode == "providerfail":
        with pytest.raises(RuntimeError, match="partial kernel failure"):
            fn(obj, hidden)
        assert (
            events.count(("experts-deferred", "alt")) == 1
            and ("stock-gate-add", "main") not in events
        )
        return
    output = fn(obj, hidden)
    assert output.shape == hidden.shape
    if mode == "active":
        assert (
            events.count(("experts-deferred", "alt")) == 1
            and ("experts-stock", "alt") not in events
        )
        assert events.count(("provider", "main")) == 1 and not any(
            e[0] == "stock-gate-add" for e in events
        )
        assert (
            events.index(("experts-deferred", "alt"))
            < events.index(("wait", "main", "alt"))
            < events.index(("provider", "main"))
        )
        assert [e for e in events if e[0] == "record"] == [
            ("record", "gemm2", "main"),
            ("record", "weights", "main"),
            ("record", "permutation", "main"),
        ]
    else:
        assert not any(e[0] in ("provider", "experts-deferred") for e in events)
        assert events.count(("stock-gate-add", "main")) == 1


@pytest.mark.parametrize(
    "reason",
    [
        None,
        "cuda",
        "tp",
        "H_E",
        "topk",
        "ep",
        "lora",
        "a2a",
        "subclass",
        "quant",
        "backend_none",
        "backend_other",
        "per_token",
        "padded",
        "mega",
        "scale",
        "shared_fusion",
    ],
)
def test_actual_native_eligibility_keeps_exact_source_type_and_all_predicates(
    monkeypatch, reason
):
    class FusedMoE:
        pass

    class Other(FusedMoE):
        pass

    class Quant:
        pass

    quant = Quant() if reason != "quant" else NS()
    quant.quant_config = NS(use_per_token_activation=reason == "per_token")
    experts = (Other if reason == "subclass" else FusedMoE)()
    experts.top_k = 9 if reason == "topk" else 10
    experts.quant_method = quant
    experts.runner_backend = (
        None
        if reason == "backend_none"
        else NS(is_flashinfer_trtllm=lambda: reason != "backend_other")
    )
    experts.trtllm_padded_hidden_size = 0 if reason == "padded" else None
    experts._mega_moe_nvfp4 = reason == "mega"
    experts.w13_input_scale_quant = None if reason == "scale" else object()
    obj = NS(
        tp_size=4 if reason == "tp" else 1,
        gate=NS(weight=NS(shape=(512, 1280) if reason == "H_E" else (512, 2560))),
        experts=experts,
        enable_shared_expert_fusion=reason == "shared_fusion",
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.quantization.modelopt_quant",
        NS(ModelOptNvFp4FusedMoEMethod=Quant),
    )
    scope = {
        "_is_cuda": reason != "cuda",
        "_SGLANG_EXPERIMENTAL_LORA_OPTI": reason == "lora",
        "FusedMoE": FusedMoE,
        "get_parallel": lambda: NS(moe_ep_size=2 if reason == "ep" else 1),
        "get_moe_a2a_backend": lambda: NS(is_none=lambda: reason != "a2a"),
    }
    fn = extracted("_uses_nvfp4_trtllm_glue", scope)
    assert fn(obj) == (reason in (None, "tp", "ep", "H_E", "topk"))
    obj.experts = None
    assert fn(obj) == (reason in (None, "tp", "ep", "H_E", "topk"))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("narrow_view", [False, True])
def test_native_gemm2_actual_positive_rowpitch_and_negative_map_are_borrowed(
    monkeypatch, dtype, narrow_view
):
    obj, hidden, gate, shared, deferred = inputs(weight_type=dtype)
    gemm2 = tensor((40, 2624), label="gemm2")
    deferred.gemm2_out = gemm2[:, :2560] if narrow_view else gemm2
    seen = []

    def run(io):
        seen.append(io)
        assert io.gemm2_out is deferred.gemm2_out and io.gemm2_out.stride() == (2624, 1)
        assert io.expanded_idx_to_permuted_idx is deferred.expanded_idx_to_permuted_idx
        assert io.expanded_idx_to_permuted_idx[0] == -1

    output = impl.finish(
        impl._Plan(run, run, deferred.top_k), hidden, gate, shared, deferred
    )
    assert output is seen[0].output and output.shape == shared.shape


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, (True, False)),
        ({"SGLANG_QWEN_MOE_FUSED_FINALIZE_GATE": "0"}, (False, False)),
        ({"SGLANG_QWEN_MOE_FUSED_FINALIZE_GATE": "true"}, (True, False)),
        ({"SGLANG_MOE_FUSED_FINALIZE_GATE": "0"}, (False, False)),
        ({"SGLANG_MOE_FUSED_FINALIZE_GATE": "true"}, (True, False)),
        (
            {
                "SGLANG_MOE_FUSED_FINALIZE_GATE": "0",
                "SGLANG_QWEN_MOE_FUSED_FINALIZE_GATE": "true",
            },
            (False, False),
        ),
        (
            {
                "SGLANG_MOE_FUSED_FINALIZE_GATE": "true",
                "SGLANG_QWEN_MOE_FUSED_FINALIZE_GATE": "0",
            },
            (True, False),
        ),
        ({"SGLANG_Q38_EXTEND_FASTPATH": "1"}, (True, True)),
        (
            {
                "SGLANG_Q38_EXTEND_FASTPATH": "1",
                "SGLANG_X_MOE_EXTEND_FUSED_FINALIZE_DISABLE": "TRUE",
            },
            (True, False),
        ),
        (
            {
                "SGLANG_Q38_EXTEND_FASTPATH": "0",
                "SGLANG_X_MOE_EXTEND_FUSED_FINALIZE_DISABLE": "0",
            },
            (True, False),
        ),
    ],
)
def test_real_fresh_import_keeps_original_source_default_and_extend_policy(
    changes, expected
):
    import json
    import os
    import subprocess

    environment = os.environ.copy()
    for key in [
        "SGLANG_MOE_FUSED_FINALIZE_GATE",
        "SGLANG_QWEN_MOE_FUSED_FINALIZE_GATE",
        "SGLANG_Q38_EXTEND_FASTPATH",
        "SGLANG_X_MOE_EXTEND_FUSED_FINALIZE_DISABLE",
    ]:
        environment.pop(key, None)
    environment.update(changes)
    environment["PYTHONPATH"] = str(Path(impl.__file__).parents[4])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    code = "import json; from sglang.srt.kernel_plugins.nvidia import permuted_finalize as f; print(json.dumps([f._SOURCE_ENABLED,f._SOURCE_EXTEND]))"
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert tuple(json.loads(result.stdout)) == expected


@pytest.mark.parametrize("rows", [1, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("batch_invariant", [False, True])
def test_real_producer_collect_spec_registry_facade_chain(
    monkeypatch, rows, pdl, batch_invariant
):
    from sglang.kernels.plugin_kernels.nvidia import spec

    monkeypatch.setitem(
        sys.modules, "sglang.kernels.jit.utils", NS(is_arch_support_pdl=lambda: pdl)
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.batch_invariant_ops",
        NS(is_batch_invariant_mode_enabled=lambda: batch_invariant),
    )
    obj, hidden, gate, shared, deferred = inputs(rows)
    # Use the model's actual declaration method and the real collector. These
    # are the startup boundaries previously bypassed by direct registry tests.
    obj.numeric_signatures = MethodType(extracted("numeric_signatures"), obj)
    widths = (0, rows, 65)
    declared = numeric_binding.collect_numeric_signatures((obj,), widths, 4)
    assert len(declared) == 2
    assert all(type(s.use_pdl) is int and s.use_pdl == int(pdl) for s in declared)
    registry = NumericRegistry()
    calls = []

    def run(io):
        calls.append(io)
        io.output.fill_(7)

    def build(model, config, context):
        assert model is obj and config is None
        assert context.numeric_signatures == declared
        for signature in context.numeric_signatures:
            registry.register(NumericOffer(signature, run))
        return NS(widths=())

    descriptor = spec.MegakernelSpec(
        name="permuted-finalize-cpu",
        requires=spec.Requirements(isas=frozenset(), tp_sizes=frozenset()),
        features=frozenset(),
        numerics=spec.Numerics(spec.NumericsKind.BITWISE),
        max_rows=0,
        numeric_max_rows=64,
        build=build,
        api_version=spec.MEGAKERNEL_API_VERSION,
    )
    context = spec.BuildContext(
        widths=(), tokens_per_request=4, deadline_s=1.0, numeric_signatures=declared
    )
    descriptor.build(obj, None, context)
    monkeypatch.setattr(numeric_binding, "numeric_registry", registry)
    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)
    plan = impl.prepare(obj, hidden, use_fused_gate=True, ordered_finalize_sink=None)
    result = impl.finish(plan, hidden, gate, shared, deferred)
    assert len(calls) == 1 and result is calls[0].output
    assert torch.all(result == 7)


def test_collector_covers_all_canonical_signature_types():
    from dataclasses import fields
    from typing import get_args

    from sglang.kernels.plugin_kernels.nvidia import numeric as api

    signatures = tuple(
        cls(**{field.name: 1 for field in fields(cls)})
        for cls in get_args(api.OperatorSignature)
    )

    class Producer:
        def numeric_signatures(self, widths, tokens_per_request):
            return signatures

    # This test follows the public union so a new DTO cannot silently disappear
    # from the real startup collector whitelist again.
    assert (
        numeric_binding.collect_numeric_signatures((Producer(),), (1,), 4) == signatures
    )


@pytest.fixture(autouse=True)
def capability_provider_enabled(monkeypatch):
    from sglang.srt.kernel_plugins.nvidia import binding

    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)
    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
