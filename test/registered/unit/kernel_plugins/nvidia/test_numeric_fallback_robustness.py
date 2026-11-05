"""Real HC producer/facade fallback; CUDA streams and launches are CPU doubles.

Provider absence is independent of source flags. Other-family slot controls use
real canonical signatures and the same registry/facade as their producers.
"""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.kernels.ops.elementwise import hc_combine_decode as decode
from sglang.kernels.plugin_kernels.nvidia import numeric as api
from sglang.srt.kernel_plugins.nvidia import binding, numeric_binding
from sglang.srt.layers import hyperconnection as hc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class Metadata(torch.Tensor):
    @property
    def is_cuda(self):
        return True

    def record_stream(self, stream):
        pass


def tensor(shape, dtype=torch.bfloat16):
    return torch.zeros(shape, dtype=dtype).as_subclass(Metadata)


def setup_layer(monkeypatch, rows, enabled, *, macro=True, cap=16, registry=None):
    registry = registry or api.NumericRegistry()
    monkeypatch.setattr(numeric_binding, "numeric_registry", registry)
    monkeypatch.setattr(binding, "megakernel_enabled", lambda: enabled)
    monkeypatch.setenv(
        "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1" if enabled else "0"
    )
    monkeypatch.setenv("SGLANG_HC_MACRO", "1" if macro else "0")
    monkeypatch.setenv("SGLANG_HC_MACRO_MIN_ROWS", "2")
    monkeypatch.setenv("SGLANG_HC_GATE_MAX_ROWS", str(cap))
    monkeypatch.setattr(torch, "compile", lambda fn: fn)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    layer = hc.GatedResidual(
        hc.HyperConnectionConfig(
            hidden_size=2560,
            hc_count=4,
            hc_lowrank=8,
            hc_per_branch_norm=True,
        ),
        use_combine=True,
    )
    layer._jit_mix_ok = layer._jit_combine_ok = False
    layer._numeric_mix_variant = lambda: 1 if macro else 0
    layer._flydsl_mix_ok = lambda: False
    layer._gate_stream = NS(wait_stream=lambda _: None)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: NS(wait_stream=lambda _: None)
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(hc, "flydsl_hc_mix_supported", lambda *_: False)
    monkeypatch.setattr(hc, "fused_hc_mix_supported", lambda *_: False)
    layer._mix_compute = lambda normed, *_: normed[:, :2560].clone()
    events = []

    def general(block, residual, normed, *_):
        events.append("main_combine")
        assert normed is not None
        return residual + block.repeat(1, 4)

    layer._combine_compute = general

    def gate(normed, weight, partials):
        events.append(("gate", normed.shape[0]))
        partials.zero_()

    def apply_norm(block, residual, partials, weight, combined, normalized, eps):
        events.append(("stock_decode", block.shape[0]))
        assert 0 < block.shape[0] <= 4, "hc_combine_decode.cuh:147"
        combined.copy_(residual)
        normalized.zero_()

    def apply(block, residual, partials, out):
        events.append(("stock_apply", block.shape[0]))
        out.copy_(residual)

    monkeypatch.setattr(
        decode, "_jit_module", lambda: NS(gate=gate, apply_norm=apply_norm, apply=apply)
    )
    # Allocation stays CPU; metadata advertises CUDA explicitly for preflight.
    empty = torch.empty
    monkeypatch.setattr(
        torch,
        "empty",
        lambda shape, *, dtype, device: empty(shape, dtype=dtype).as_subclass(Metadata),
    )
    x = tensor((rows, 10240))
    block = tensor((rows, 2560))
    norm = NS(group_size=2560, weight=tensor((10240,)), variance_epsilon=1e-6)
    return layer, registry, x, block, norm, events


def offer(registry, operation, rows, events):
    def run(io):
        events.append(("provider", operation))
        if operation == api.NumericOperation.HC_APPLY_NORM:
            io.combined_out.copy_(io.residual)
            io.normalized_out.zero_()

    registry.register(
        api.NumericOffer(
            api.HcSignature(operation, rows, 4, 2560, api.ScalarType.BF16),
            run,
        )
    )


@pytest.mark.parametrize("rows", [1, 4, 8, 16, 32])
@pytest.mark.parametrize(
    "mode", ["not_installed", "master_off", "six_slots_off", "all_slots_off"]
)
def test_hc_no_consumer_preserves_main_rows(monkeypatch, rows, mode):
    layer, reg, x, block, norm, events = setup_layer(
        monkeypatch, rows, mode != "master_off"
    )
    if mode in ("six_slots_off", "master_off"):
        offer(reg, api.NumericOperation.HC_APPLY_NORM_GATE, rows, events)
    if mode == "not_installed":
        # A plugin is never imported by the stock producer/facade.
        original_import = importlib.import_module

        def no_plugin(name, *args, **kwargs):
            assert not name.startswith("sgl_nvidia_mega")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", no_plugin)
    _, residuals = layer.mix(x, normalized_input=x)
    assert (residuals[2] is not None) == (rows == 1)
    combined, normalized, next_partials = layer.combine_normalize_and_gate(
        block, residuals, norm, None
    )
    assert combined.shape == x.shape and next_partials is None
    if rows > 4:
        assert ("stock_decode", rows) not in events
    if rows > 1:
        assert normalized is None and "main_combine" in events
        assert not any(
            isinstance(event, tuple) and event[0] == "gate" for event in events
        )


@pytest.mark.parametrize("rows", [4, 8, 16, 32])
def test_exact_consumer_allows_extended_macro_partials(monkeypatch, rows):
    layer, reg, x, block, norm, events = setup_layer(monkeypatch, rows, True)
    offer(reg, api.NumericOperation.HC_APPLY_NORM, rows, events)
    offer(reg, api.NumericOperation.HC_APPLY_NORM_GATE, rows, events)
    _, residuals = layer.mix(x, normalized_input=x)
    assert residuals[2].shape == (rows, 8, 4)
    # The terminal/no-inject boundary must use op11, never the small stock path.
    layer.combine_normalize_and_gate(block, residuals, norm, None)
    assert ("provider", api.NumericOperation.HC_APPLY_NORM) in events
    assert not any(
        isinstance(event, tuple) and event[0] == "stock_decode" for event in events
    )


@pytest.mark.parametrize("rows", [8, 16, 32])
@pytest.mark.parametrize(
    "reason", ["missing_offer", "strided_weight", "strided_partials", "cpu_block"]
)
def test_carried_partials_recheck_actual_consumer_metadata(monkeypatch, rows, reason):
    layer, reg, x, block, norm, events = setup_layer(monkeypatch, rows, True)
    partials = tensor((rows, 8, 4), torch.float32)
    if reason != "missing_offer":
        offer(reg, api.NumericOperation.HC_APPLY_NORM, rows, events)
    if reason == "strided_weight":
        norm.weight = tensor((20480,))[::2]
    if reason == "strided_partials":
        partials = tensor((rows, 8, 8), torch.float32)[:, :, ::2]
    if reason == "cpu_block":
        block = torch.zeros((rows, 2560), dtype=torch.bfloat16)
    result, normalized = layer.combine_and_normalize(block, (x, x, partials), norm)
    assert result.shape == x.shape and normalized is None
    assert events == ["main_combine"]


def test_selected_consumer_exception_does_not_retry(monkeypatch):
    rows = 32
    layer, reg, x, block, norm, events = setup_layer(monkeypatch, rows, True)

    def fail(io):
        io.combined_out.zero_()
        raise RuntimeError("selected write")

    reg.register(
        api.NumericOffer(
            api.HcSignature(
                api.NumericOperation.HC_APPLY_NORM,
                rows,
                4,
                2560,
                api.ScalarType.BF16,
            ),
            fail,
        )
    )
    with pytest.raises(RuntimeError, match="selected write"):
        layer.combine_and_normalize(
            block, (x, x, tensor((rows, 8, 4), torch.float32)), norm
        )
    assert "main_combine" not in events


def signature_for(operation):
    if operation in (
        api.NumericOperation.HC_GATE,
        api.NumericOperation.HC_APPLY_NORM,
        api.NumericOperation.HC_APPLY_NORM_GATE,
        api.NumericOperation.HC_MIX,
    ):
        return api.HcSignature(operation, 4, 4, 2560, api.ScalarType.BF16)
    if operation in (
        api.NumericOperation.GDN_FRONT,
        api.NumericOperation.GDN_CORE,
        api.NumericOperation.GDN_VERIFY_GLUE,
    ):
        return api.GdnSignature(operation, 4, 4, 12, 128, 128, 4, api.ScalarType.BF16)
    if operation == api.NumericOperation.LOCAL_EXPERTS:
        return api.ExpertSignature(
            operation, 4, 2560, 192, 512, 10, api.ScalarType.PACKED_FP4
        )
    if operation == api.NumericOperation.FINALIZE_PERMUTED_SHARED:
        return api.FinalizePermutedSignature(
            operation, 4, 2560, 10, api.ScalarType.BF16, api.ScalarType.FP32
        )
    if operation == api.NumericOperation.ROUTER_QUANT:
        return api.RouterQuantSignature(operation, 4, 2560, 512, api.ScalarType.BF16)
    if operation == api.NumericOperation.MTP_CAPTURE_SEED:
        return api.MtpCaptureSeedSignature(
            operation, 4, 2051, 2051, 8, 2055, 2055, 0, 7, 5, 5, 4, 5, 1
        )
    if operation == api.NumericOperation.MTP_DEFERRED_KV_PREPARE:
        return api.MtpDeferredKvSignature(
            operation,
            1,
            1,
            256,
            api.ScalarType.BF16,
            2055,
            2112,
            4,
            8,
            64,
            7,
            8192,
            8192,
            256,
            256,
            17,
        )
    return api.NumericSignature(operation, 4, 8, 1, 128, 512, 4, api.ScalarType.FP32)


@pytest.mark.parametrize("disabled", list(api.NumericOperation))
@pytest.mark.parametrize("rows", [8, 16, 32])
def test_individual_slot_off_cannot_break_hc_fallback(monkeypatch, disabled, rows):
    layer, reg, x, block, norm, events = setup_layer(monkeypatch, rows, True)
    for op in (
        api.NumericOperation.HC_APPLY_NORM,
        api.NumericOperation.HC_APPLY_NORM_GATE,
    ):
        if op != disabled:
            offer(reg, op, rows, events)
    _, residuals = layer.mix(x, normalized_input=x)
    result, _, _ = layer.combine_normalize_and_gate(block, residuals, norm, None)
    assert result.shape == x.shape
    assert ("stock_decode", rows) not in events
    if disabled == api.NumericOperation.HC_APPLY_NORM:
        assert residuals[2] is None and "main_combine" in events


@pytest.mark.parametrize("operation", list(api.NumericOperation))
def test_each_operation_absent_keeps_facade_stock(monkeypatch, operation):
    # These are canonical/facade controls, not model/GPU invocation evidence.
    registry = api.NumericRegistry()
    monkeypatch.setattr(numeric_binding, "numeric_registry", registry)
    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)
    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")
    signatures = []
    for other in api.NumericOperation:
        sig = signature_for(other)
        signatures.append(sig)
        if other != operation:
            registry.register(api.NumericOffer(sig, lambda io: None))
    for sig in signatures:
        assert (numeric_binding.numeric_operator(sig, prepared=True) is None) == (
            sig.operation == operation
            or sig.operation == api.NumericOperation.PACKED_ATTENTION
        )


def test_prefill_capture_metadata_exact_six_slots_off(monkeypatch):
    layer, reg, x, block, norm, events = setup_layer(
        monkeypatch, 32, True, macro=True, cap=16
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    offer(reg, api.NumericOperation.HC_APPLY_NORM_GATE, 32, events)
    _, residuals = layer.mix(x, normalized_input=x)
    assert residuals[2] is None
    layer.combine_normalize_and_gate(block, residuals, norm, None)
    assert events == ["main_combine"]


@pytest.mark.parametrize("reason", ["no_norm", "norm_dtype", "norm_group"])
def test_carried_partials_without_norm_consumer_uses_unbounded_plain_apply(
    monkeypatch, reason
):
    layer, reg, x, block, norm, events = setup_layer(monkeypatch, 32, True)
    if reason == "no_norm":
        norm = None
    elif reason == "norm_dtype":
        norm.weight = norm.weight.float()
    else:
        norm.group_size = 128
    result, normalized, next_partials = layer.combine_normalize_and_gate(
        block,
        (x, x, tensor((32, 8, 4), torch.float32)),
        norm,
        None,
    )
    assert result.shape == x.shape and normalized is next_partials is None
    assert ("stock_apply", 32) in events and ("stock_decode", 32) not in events


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
