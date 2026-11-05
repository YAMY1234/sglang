"""Execute the actual runner with real meta tensors and explicit FI/CUDA doubles."""

import importlib
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.moe.quantized_input import Nvfp4RoutedInputPreQuant
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

runner = importlib.import_module("sglang.srt.layers.moe.moe_runner.flashinfer_trtllm")
topk_module = importlib.import_module("sglang.srt.layers.moe.topk")


def tensor(shape, dtype=torch.bfloat16, device="meta"):
    return torch.empty(shape, dtype=dtype, device=device)


@pytest.fixture
def setup_runner(monkeypatch):
    fused = ModuleType("flashinfer.fused_moe")
    launch = Mock(
        side_effect=lambda **kw: (tensor((kw["hidden_states"].shape[0], 2560)),)
    )
    fused.trtllm_fp4_block_scale_moe = launch
    fused.trtllm_fp4_block_scale_routed_moe = launch
    enums = ModuleType("flashinfer.tllm_enums")
    enums.ActivationType = SimpleNamespace(Swiglu=0)
    fi = ModuleType("flashinfer")
    fi.__path__ = []
    fi.SfLayout = SimpleNamespace(layout_linear=0)
    fi.nvfp4_quantize = Mock(
        side_effect=lambda x, *a, **k: (
            tensor((x.shape[0], x.shape[1] // 2), torch.uint8),
            tensor((x.shape[0], x.shape[1] // 16), torch.uint8),
            tensor((x.shape[0],), torch.float32),
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "flashinfer", fi)
    monkeypatch.setitem(__import__("sys").modules, "flashinfer.fused_moe", fused)
    monkeypatch.setitem(__import__("sys").modules, "flashinfer.tllm_enums", enums)
    quantize = Mock(
        side_effect=lambda x, scale: (
            tensor((x.shape[0], x.shape[1] // 2), torch.uint8),
            tensor((x.shape[0], x.shape[1] // 16), torch.float8_e4m3fn),
        )
    )
    monkeypatch.setattr(runner, "quantize_hidden_states_fp4", quantize)
    monkeypatch.setattr(
        runner,
        "_get_fp4_moe_output_buffer",
        lambda **kw: tensor((kw["num_tokens"], kw["hidden_size"]), kw["dtype"]),
    )
    monkeypatch.setattr(runner, "get_activation_type", lambda *a, **k: 0)
    monkeypatch.setattr(runner, "trtllm_moe_enable_pdl", lambda _: False)
    monkeypatch.setattr(runner, "_make_deferred_finalize_output", lambda x, **k: x)
    monkeypatch.setattr(
        runner,
        "_get_routing_for_flashinfer_routed",
        lambda _: tensor((4, 10), torch.int32),
    )
    monkeypatch.setattr(runner, "_routing_top_k", lambda _: 10)
    monkeypatch.setattr(
        topk_module.TopKOutputChecker, "format_is_standard", lambda x: x.standard
    )
    monkeypatch.setattr(
        topk_module.TopKOutputChecker, "format_is_packed", lambda x: False
    )
    monkeypatch.setattr(
        topk_module.TopKOutputChecker, "format_is_bypassed", lambda x: not x.standard
    )
    stream = Mock()
    current_stream = Mock(return_value=stream)
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    yield SimpleNamespace(
        launch=launch,
        quantize=quantize,
        per_token=fi.nvfp4_quantize,
        stream=stream,
        current_stream=current_stream,
    )


def inputs(rows=4, standard=False):
    hidden = tensor((rows, 2560))
    packet = Nvfp4RoutedInputPreQuant(
        tensor((rows, 1280), torch.uint8), tensor((rows, 160), torch.float8_e4m3fn)
    )
    dispatch = SimpleNamespace(
        hidden_states=hidden,
        hidden_states_scale=None,
        hidden_states_pre_quant=packet,
        topk_output=SimpleNamespace(
            standard=standard,
            router_logits=tensor((rows, 512)),
            topk_config=SimpleNamespace(
                correction_bias=None, top_k=10, num_expert_group=1, topk_group=1
            ),
        ),
    )
    info = SimpleNamespace(
        padded_hidden_size=None,
        use_per_token_activation=False,
        w13_input_scale_quant=tensor((1,), torch.float32),
        w13_weight=tensor((1,)),
        w2_weight=tensor((1,)),
        w13_weight_scale=tensor((1,), torch.uint8),
        w2_weight_scale=tensor((1,), torch.uint8),
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        g1_scale_c=tensor((1,), torch.float32),
        g1_alphas=tensor((1,), torch.float32),
        g2_alphas=tensor((1,), torch.float32),
        global_num_experts=512,
        intermediate_size_per_partition=192,
        local_expert_offset=0,
        local_num_experts=512,
        routing_method_type=None,
    )
    config = SimpleNamespace(
        activation="silu", is_gated=True, routed_scaling_factor=None
    )
    return dispatch, info, config


@pytest.mark.parametrize("rows", [1, 4, 8, 16])
@pytest.mark.parametrize("standard", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("event", [False, True])
def test_typed_input_skips_quant_once_through_actual_runner(
    setup_runner, rows, standard, deferred, event
):
    d, q, c = inputs(rows, standard)
    ready = object() if event else None
    d.hidden_states_pre_quant = d.hidden_states_pre_quant._replace(ready=ready)
    token = runner._deferred_finalize_enabled.set(deferred)
    try:
        runner.fused_experts_none_to_flashinfer_trtllm_fp4(d, q, c)
    finally:
        runner._deferred_finalize_enabled.reset(token)
    s = setup_runner
    s.quantize.assert_not_called()
    s.per_token.assert_not_called()
    s.launch.assert_called_once()
    kw = s.launch.call_args.kwargs
    assert kw["hidden_states"] is d.hidden_states_pre_quant.x_fp4
    assert kw["hidden_states_scale"].shape == (rows, 160)
    assert kw["hidden_states_scale"].dtype == torch.float8_e4m3fn
    assert kw["do_finalize"] is not deferred
    if event:
        s.stream.wait_event.assert_called_once_with(ready)
    else:
        s.current_stream.assert_not_called()


@pytest.mark.parametrize("other", [None, ("unrelated", "tuple")])
def test_absent_or_other_packet_preserves_native_quantization(setup_runner, other):
    d, q, c = inputs()
    d.hidden_states_pre_quant = other
    runner.fused_experts_none_to_flashinfer_trtllm_fp4(d, q, c)
    setup_runner.quantize.assert_called_once_with(
        d.hidden_states, q.w13_input_scale_quant
    )
    setup_runner.launch.assert_called_once()
    setup_runner.current_stream.assert_not_called()


def test_existing_a2a_fp4_passthrough_unchanged(setup_runner):
    d, q, c = inputs()
    d.hidden_states_pre_quant = None
    d.hidden_states = tensor((4, 1280), torch.uint8)
    d.hidden_states_scale = tensor((4, 160), torch.float8_e4m3fn)
    runner.fused_experts_none_to_flashinfer_trtllm_fp4(d, q, c)
    setup_runner.quantize.assert_not_called()
    assert setup_runner.launch.call_args.kwargs["hidden_states"] is d.hidden_states


def test_existing_per_token_quantization_unchanged(setup_runner):
    d, q, c = inputs()
    d.hidden_states_pre_quant = None
    q.use_per_token_activation = True
    runner.fused_experts_none_to_flashinfer_trtllm_fp4(d, q, c)
    setup_runner.quantize.assert_not_called()
    setup_runner.per_token.assert_called_once()
    setup_runner.launch.assert_called_once()


@pytest.mark.parametrize(
    "change",
    [
        "packed_dtype",
        "packed_shape",
        "scale_dtype",
        "scale_shape",
        "device",
        "stride",
        "hidden_dtype",
        "padded",
        "per_token",
        "a2a",
    ],
)
def test_invalid_typed_input_raises_without_quant_or_launch_retry(setup_runner, change):
    d, q, c = inputs()
    packet = d.hidden_states_pre_quant
    if change == "packed_dtype":
        packet = packet._replace(x_fp4=tensor((4, 1280), torch.int8))
    if change == "packed_shape":
        packet = packet._replace(x_fp4=tensor((3, 1280), torch.uint8))
    if change == "scale_dtype":
        packet = packet._replace(x_sf=tensor((4, 160), torch.float32))
    if change == "scale_shape":
        packet = packet._replace(x_sf=tensor((4, 159), torch.float8_e4m3fn))
    if change == "device":
        packet = packet._replace(x_fp4=tensor((4, 1280), torch.uint8, device="cpu"))
    if change == "stride":
        packet = packet._replace(x_sf=tensor((160, 4), torch.float8_e4m3fn).T)
    if change == "hidden_dtype":
        d.hidden_states = tensor((4, 2560), torch.float16)
    if change == "padded":
        q.padded_hidden_size = 2688
    if change == "per_token":
        q.use_per_token_activation = True
    if change == "a2a":
        d.hidden_states_scale = packet.x_sf
    d.hidden_states_pre_quant = packet
    with pytest.raises(ValueError):
        runner.fused_experts_none_to_flashinfer_trtllm_fp4(d, q, c)
    setup_runner.quantize.assert_not_called()
    setup_runner.per_token.assert_not_called()
    setup_runner.launch.assert_not_called()
    setup_runner.current_stream.assert_not_called()


def test_selected_launch_failure_propagates_without_retry(setup_runner):
    d, q, c = inputs()
    setup_runner.launch.side_effect = RuntimeError("explicit FI launch failure")
    with pytest.raises(RuntimeError, match="explicit FI launch failure"):
        runner.fused_experts_none_to_flashinfer_trtllm_fp4(d, q, c)
    setup_runner.launch.assert_called_once()
    setup_runner.quantize.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
