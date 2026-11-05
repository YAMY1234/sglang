"""Real binding producer/collector/facade and typed packet; CUDA/raw launches are doubles."""

from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericOffer,
    NumericOperation,
    NumericRegistry,
    RouterQuantIO,
    RouterQuantSignature,
)
from sglang.srt.kernel_plugins.nvidia import binding, numeric_binding
from sglang.srt.kernel_plugins.nvidia import router_quant as impl
from sglang.srt.layers.moe.quantized_input import Nvfp4RoutedInputPreQuant
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class DeviceTensor(torch.Tensor):
    @property
    def is_cuda(self):
        return True

    @property
    def device(self):
        return torch.device("cuda:0")


def tensor(shape, dtype=torch.bfloat16):
    return torch.zeros(shape, dtype=dtype).as_subclass(DeviceTensor)


def installed(name, monkeypatch, scope=None):
    import sglang.srt.models.qwen2_moe as module

    assert (
        Path(module.__file__).resolve()
        == Path(impl.__file__).resolve().parents[2] / "models/qwen2_moe.py"
    )
    for key, value in (scope or {}).items():
        monkeypatch.setattr(module, key, value, raising=False)
    return getattr(module.Qwen2MoeSparseMoeBlock, name)


@pytest.fixture
def packet_constructor(monkeypatch):
    """Count construction of the real binding-owned packet; never replace its type."""
    calls = []

    def construct(*args, **kwargs):
        packet = Nvfp4RoutedInputPreQuant(*args, **kwargs)
        calls.append(packet)
        return packet

    monkeypatch.setattr(impl, "Nvfp4RoutedInputPreQuant", construct)
    return NS(packet_type=Nvfp4RoutedInputPreQuant, calls=calls)


@pytest.fixture
def setup(monkeypatch, packet_constructor):
    import sglang.srt.layers.moe as moe
    import sglang.srt.runtime_context as ctx
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptNvFp4FusedMoEMethod,
    )
    from sglang.srt.model_executor.runner_backend_utils import (
        tc_piecewise_cuda_graph as pw,
    )

    monkeypatch.setattr(impl, "_SOURCE_ENABLED", True)
    monkeypatch.setattr(impl, "_SOURCE_WIDE", True)
    monkeypatch.setattr(impl, "_SOURCE_K4", False)
    monkeypatch.setattr(impl, "_SOURCE_LAYERS", "all")
    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")
    parallel = NS(moe_ep_size=1)
    lora = NS(enable_lora=False)
    backend = NS(is_flashinfer_trtllm=lambda: True)
    a2a = NS(is_none=lambda: True)
    monkeypatch.setattr(ctx, "get_parallel", lambda: parallel)
    monkeypatch.setattr(ctx, "get_lora", lambda: lora)
    monkeypatch.setattr(moe, "get_moe_runner_backend", lambda: backend)
    monkeypatch.setattr(moe, "get_moe_a2a_backend", lambda: a2a)
    monkeypatch.setattr(pw, "is_in_tc_piecewise_cuda_graph", lambda: False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    experts = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(experts)
    experts.quant_method = ModelOptNvFp4FusedMoEMethod.__new__(
        ModelOptNvFp4FusedMoEMethod
    )
    experts.quant_method.quant_config = NS(use_per_token_activation=False)
    experts.w13_input_scale_quant = tensor((1,), torch.float32)
    experts.trtllm_padded_hidden_size = None
    producer = NS(
        experts=experts,
        gate=NS(weight=tensor((512, 2560))),
        tp_size=4,
        is_nextn=False,
        enable_shared_expert_fusion=False,
        layer_id=0,
    )
    from capability_fixtures import contract_spec

    producer._numeric_capability_spec = contract_spec()
    registry = NumericRegistry()
    monkeypatch.setattr(numeric_binding, "numeric_registry", registry)
    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)
    impl.numeric_signatures(producer, (1, 4, 8, 16))
    # Allocator double: real CPU tensors with explicit CUDA metadata, no CUDA math.
    empty = torch.empty
    monkeypatch.setattr(
        torch,
        "empty",
        lambda shape, *, dtype, device: empty(shape, dtype=dtype).as_subclass(
            DeviceTensor
        ),
    )
    return producer, registry, parallel, lora, backend, a2a


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("k4", [False, True])
def test_finite_startup_and_real_collector(setup, monkeypatch, wide, k4):
    obj, *_ = setup
    monkeypatch.setattr(impl, "_SOURCE_WIDE", wide)
    monkeypatch.setattr(impl, "_SOURCE_K4", k4)
    # Execute the installed producer method; old F source capability remains off.
    obj.numeric_signatures = MethodType(
        installed("numeric_signatures", monkeypatch), obj
    )
    from sglang.srt.kernel_plugins.nvidia import permuted_finalize

    monkeypatch.setattr(permuted_finalize, "_SOURCE_ENABLED", False)
    sigs = numeric_binding.collect_numeric_signatures(
        (obj,), (0, 1, 4, 8, 16, 32, True), 4
    )
    assert {s.rows for s in sigs} == ({1, 4, 8, 16} if wide else {1, 4})
    assert all(isinstance(s, RouterQuantSignature) for s in sigs)
    assert {s.variant for s in sigs if s.rows in (1, 4)} == {int(k4)}
    assert all(s.variant == 2 for s in sigs if s.rows in (8, 16))
    assert obj._numeric_router_quant_signatures == frozenset(sigs)


@pytest.mark.parametrize(
    "reason",
    [
        "source",
        "master",
        "tp1",
        "ep",
        "nextn",
        "lora",
        "a2a",
        "backend",
        "method",
        "per_token",
        "padded",
        "shared_fused",
        "weight_cpu",
        "weight_dtype",
        "weight_shape",
        "scale_dtype",
        "scale_shape",
        "layer",
    ],
)
def test_source_negative_no_registration_or_offer_probe(setup, monkeypatch, reason):
    obj, reg, parallel, lora, backend, a2a = setup
    if reason == "source":
        monkeypatch.setattr(impl, "_SOURCE_ENABLED", False)
    elif reason == "master":
        monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "0")
    elif reason == "tp1":
        obj.tp_size = 1
    elif reason == "ep":
        parallel.moe_ep_size = 2
    elif reason == "nextn":
        obj.is_nextn = True
    elif reason == "lora":
        lora.enable_lora = True
    elif reason == "a2a":
        a2a.is_none = lambda: False
    elif reason == "backend":
        backend.is_flashinfer_trtllm = lambda: False
    elif reason == "method":
        obj.experts.quant_method = object()
    elif reason == "per_token":
        obj.experts.quant_method.quant_config.use_per_token_activation = True
    elif reason == "padded":
        obj.experts.trtllm_padded_hidden_size = 3072
    elif reason == "shared_fused":
        obj.enable_shared_expert_fusion = True
    elif reason == "weight_cpu":
        obj.gate.weight = torch.zeros((512, 2560), dtype=torch.bfloat16)
    elif reason == "weight_dtype":
        obj.gate.weight = obj.gate.weight.float()
    elif reason == "weight_shape":
        obj.gate.weight = tensor((256, 2560))
    elif reason == "scale_dtype":
        obj.experts.w13_input_scale_quant = tensor((1,), torch.bfloat16)
    elif reason == "scale_shape":
        obj.experts.w13_input_scale_quant = tensor((2,), torch.float32)
    elif reason == "layer":
        monkeypatch.setattr(impl, "_SOURCE_LAYERS", "2,4")
    monkeypatch.setattr(
        numeric_binding,
        "numeric_operator",
        lambda *a, **k: pytest.fail("rejected source looked up offer"),
    )
    assert impl.numeric_signatures(obj, (1, 4, 8, 16)) == ()
    assert impl.produce(obj, tensor((4, 2560))) is None


@pytest.mark.parametrize("rows", [1, 4, 8, 16])
@pytest.mark.parametrize("k4", [False, True])
def test_facade_buffers_and_real_typed_packet(
    setup, monkeypatch, packet_constructor, rows, k4
):
    obj, reg, *_ = setup
    monkeypatch.setattr(impl, "_SOURCE_K4", k4)
    sigs = impl.numeric_signatures(obj, (rows,))
    seen = []

    def run(io):
        assert isinstance(io, RouterQuantIO)
        seen.append(io)
        io.logits_out.fill_(1)

    reg.register(NumericOffer(sigs[0], run))
    hidden = tensor((rows, 2560))
    logits, packet = impl.produce(obj, hidden)
    assert isinstance(packet, packet_constructor.packet_type) and packet.ready is None
    assert packet_constructor.calls == [packet]
    assert len(seen) == 1 and seen[0].hidden_states is hidden
    assert seen[0].global_scale is obj.experts.w13_input_scale_quant
    assert seen[0].router_weight is obj.gate.weight
    assert logits is seen[0].logits_out and packet.x_fp4 is seen[0].packed_out
    assert packet.x_sf is seen[0].scale_out
    assert packet.x_sf.shape == (rows, 160) and packet.x_sf.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize(
    "shape,eligible",
    [((), True), ((1,), True), ((1, 1), False), ((2,), False), ((0,), False)],
)
def test_global_scale_domain_is_scalar_or_existing_vector(setup, shape, eligible):
    obj, *_ = setup
    obj.experts.w13_input_scale_quant = tensor(shape, torch.float32)
    assert impl._source_eligible(obj) == eligible
    assert bool(impl.numeric_signatures(obj, (1, 4))) == eligible


@pytest.mark.parametrize("rows", [1, 4, 8, 16])
@pytest.mark.parametrize("old_parameter", [False, True])
def test_source_reduced_scalar_becomes_only_caller_view(
    setup, packet_constructor, rows, old_parameter
):
    from sglang.srt.layers.utils.common import copy_or_rebind_param

    obj, reg, *_ = setup
    experts = obj.experts
    if old_parameter:
        del experts.w13_input_scale_quant
        experts.w13_input_scale_quant = torch.nn.Parameter(
            tensor((1,), torch.float32), requires_grad=False
        )
    experts.w13_input_scale = tensor((2, 3), torch.float32).fill_(4)
    # The actual ModelOpt TRTLLM reduction and installed parameter binder.
    w13_input_scale = experts.w13_input_scale.max().to(torch.float32)
    copy_or_rebind_param(
        experts, "w13_input_scale_quant", (1 / w13_input_scale).to(torch.float32)
    )
    scalar = experts.w13_input_scale_quant
    assert scalar.shape == () and scalar.item() == 0.25
    (sig,) = impl.numeric_signatures(obj, (rows,))
    seen = []

    def run(io):
        assert io.global_scale.shape == (1,)
        assert io.global_scale.data_ptr() == scalar.data_ptr()
        assert (
            io.global_scale.untyped_storage().data_ptr()
            == scalar.untyped_storage().data_ptr()
        )
        assert (
            io.global_scale.dtype == scalar.dtype
            and io.global_scale.device == scalar.device
        )
        assert io.global_scale.item() == scalar.item()
        seen.append(io)

    reg.register(NumericOffer(sig, run))
    hidden = tensor((rows, 2560))
    logits, packet = impl.produce(obj, hidden)
    assert len(seen) == 1 and seen[0].hidden_states is hidden
    assert logits is seen[0].logits_out and packet.x_fp4 is seen[0].packed_out
    assert packet.x_sf is seen[0].scale_out and packet.ready is None
    assert experts.w13_input_scale_quant is scalar and scalar.shape == ()
    assert packet_constructor.calls == [packet]


@pytest.mark.parametrize(
    "reason",
    [
        "no_startup",
        "tail",
        "cpu",
        "dtype",
        "shape",
        "strided",
        "no_offer",
        "compile",
        "piecewise",
    ],
)
def test_incomplete_source_leaves_stock_unwritten(setup, monkeypatch, reason):
    obj, reg, *_ = setup
    impl.numeric_signatures(obj, (1, 4))
    hidden = tensor((4, 2560))
    if reason == "no_startup":
        del obj._numeric_router_quant_signatures
    elif reason == "tail":
        hidden = tensor((8, 2560))
    elif reason == "cpu":
        hidden = torch.zeros((4, 2560), dtype=torch.bfloat16)
    elif reason == "dtype":
        hidden = hidden.float()
    elif reason == "shape":
        hidden = tensor((4, 1280))
    elif reason == "strided":
        hidden = tensor((4, 5120))[:, ::2]
    elif reason == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif reason == "piecewise":
        from sglang.srt.model_executor.runner_backend_utils import (
            tc_piecewise_cuda_graph as pw,
        )

        monkeypatch.setattr(pw, "is_in_tc_piecewise_cuda_graph", lambda: True)
    assert impl.produce(obj, hidden) is None


def test_selected_failure_propagates_no_stock_retry(setup, packet_constructor):
    obj, reg, *_ = setup
    sig = impl.numeric_signatures(obj, (4,))[0]

    def bad(io):
        raise RuntimeError("cold capture bundle")

    reg.register(NumericOffer(sig, bad))
    with pytest.raises(RuntimeError, match="cold capture bundle"):
        impl.produce(obj, tensor((4, 2560)))
    assert packet_constructor.calls == []


def test_collector_refuses_bool_before_key_collision():
    sig = RouterQuantSignature(NumericOperation.ROUTER_QUANT, True, 2560, 512, 1)
    producer = NS(numeric_signatures=lambda *_: (sig,))
    with pytest.raises(TypeError, match="bool"):
        numeric_binding.collect_numeric_signatures((producer,), (1,), 4)


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_router_experts_keeps_hidden_topk_and_packet_once(
    monkeypatch, selected, defer, fused
):
    events = []
    hidden = tensor((4, 2560))
    logits = tensor((4, 512))
    packet = object()
    topk = object()

    class Experts:
        def __call__(self, h, t, **kw):
            events.append(("experts", h, t, kw))
            return h

        def forward_deferred_finalize(self, h, t, **kw):
            events.append(("defer", h, t, kw))
            return h

    def gate(h):
        events.append(("gate", h))
        return logits, None

    def top(h, l):
        events.append(("topk", h, l))
        return topk

    obj = NS(
        gate=gate,
        topk=top,
        experts=Experts(),
        enable_shared_expert_fusion=False,
        supports_deferred_finalize=True,
    )
    fn = installed(
        "_forward_router_experts",
        monkeypatch,
        {
            "_use_aiter": False,
            "TopKOutputChecker": NS(
                format_is_bypassed=lambda t: True, format_is_standard=lambda t: False
            ),
        },
    )
    out = fn(
        obj,
        hidden,
        defer_finalize=defer,
        fused_finalize=fused,
        precomputed_router=(logits, packet) if selected else None,
    )
    assert out is hidden
    assert len([e for e in events if e[0] == "gate"]) == (0 if selected else 1)
    assert len([e for e in events if e[0] in ("experts", "defer")]) == 1
    assert events[-2] == ("topk", hidden, logits)
    assert events[-1][1] is hidden and events[-1][2] is topk
    assert events[-1][3] == ({"pre_quant_input": packet} if selected else {})


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("defer", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_outer_produces_before_side_join_and_threads_one_packet(
    monkeypatch, dual, defer, selected
):
    import sys

    from sglang.srt.kernel_plugins.nvidia import permuted_finalize
    from sglang.srt.layers.layer_boundary.fusions.moe_finalize import (
        MoeDeferredFinalize,
    )

    events = []
    hidden = tensor((4, 2560))
    logits = tensor((4, 512))
    shared = tensor((4, 2560))
    packet = object()

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
    monkeypatch.setattr(permuted_finalize, "prepare", lambda *a, **k: None)
    monkeypatch.setattr(
        MoeDeferredFinalize,
        "from_flashinfer",
        classmethod(lambda cls, *a, **k: NS(deferred=a[0])),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.kernels.ops.elementwise.elementwise",
        NS(fused_gate_sigmoid_mul_add=lambda *a, **k: None),
    )

    def produce(obj, h):
        events.append(("produce", active[0].name, h))
        return (logits, packet) if selected else None

    monkeypatch.setattr(impl, "produce", produce)

    def gate(h):
        events.append(("gate", h))
        return logits, None

    def top(h, l):
        events.append(("topk", h, l))
        return object()

    class Experts:
        def __call__(self, h, t, **kw):
            events.append(("experts", active[0].name, h, kw))
            return h

        def forward_deferred_finalize(self, h, t, **kw):
            events.append(("experts", active[0].name, h, kw))
            return object()

    obj = NS(
        gate=gate,
        topk=top,
        experts=Experts(),
        enable_shared_expert_fusion=False,
        supports_deferred_finalize=True,
        _use_mega_moe=False,
        shared_expert=object(),
        shared_expert_gate=NS(weight=tensor((1, 2560))),
        alt_stream=alt,
        reduce_results=False,
        _shared_expert_input=lambda hidden: hidden.clone(),
        _forward_shared_experts=lambda *a, **k: shared,
        _gate_shared_output_out_of_place=lambda *a: shared,
    )
    checker = NS(format_is_bypassed=lambda t: True, format_is_standard=lambda t: False)
    obj._forward_router_experts = MethodType(
        installed(
            "_forward_router_experts",
            monkeypatch,
            {"_use_aiter": False, "TopKOutputChecker": checker},
        ),
        obj,
    )
    scope = {
        "torch": torch,
        "get_moe_a2a_backend": lambda: NS(
            is_deepep=lambda: False,
            is_deepep_v2=lambda: False,
            is_mori=lambda: False,
            is_ascend_fuseep=lambda: False,
        ),
        "use_intel_amx_backend": lambda _: False,
        "is_npu": lambda: False,
        "_SGLANG_EXPERIMENTAL_LORA_OPTI": False,
        "get_bool_env_var": lambda *a: False,
        "get_is_capture_mode": lambda: dual,
        "get_exec": lambda: NS(moe=NS(enable_paged_experts=False)),
        "fused_gate_sigmoid_mul_add": lambda *a, **k: None,
        "is_in_breakable_cuda_graph": lambda: False,
        "sum_post_experts_output": lambda x: x,
    }
    obj.forward_normal_dual_stream = MethodType(
        installed("forward_normal_dual_stream", monkeypatch, scope), obj
    )
    installed("forward", monkeypatch, scope)(obj, hidden, defer_finalize=defer)
    produce_events = [e for e in events if e[0] == "produce"]
    expert_events = [e for e in events if e[0] == "experts"]
    assert len(produce_events) == len(expert_events) == 1
    assert (
        produce_events[0][1] == "main"
        and produce_events[0][2].data_ptr() == hidden.data_ptr()
    )
    assert expert_events[0][2].data_ptr() == hidden.data_ptr()
    assert expert_events[0][3] == ({"pre_quant_input": packet} if selected else {})
    assert len([e for e in events if e[0] == "gate"]) == (0 if selected else 1)
    if dual:
        first_wait = next(i for i, e in enumerate(events) if e[0] == "wait")
        assert events.index(produce_events[0]) < first_wait
        assert ("wait", "main", "alt") in events


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
