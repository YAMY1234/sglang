"""Framework publication for optional local packed experts, without collectives."""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

import torch

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    ExpertSignature,
    FinalizeSharedIO,
    LocalExpertsIO,
    NumericOperation,
    ScalarType,
)
from sglang.srt.kernel_plugins.nvidia.numeric_binding import numeric_operator
from sglang.srt.utils import get_bool_env_var

_finalize_sink = ContextVar("local_expert_finalize_sink", default=None)


@contextmanager
def capture_local_finalize(sink, *, run=None):
    """Framework-only lifetime; providers never receive the capture sink."""
    token = _finalize_sink.set((sink, run))
    try:
        yield
    finally:
        _finalize_sink.reset(token)


def prepare_local_shared_finalize(
    hidden, gate_weight, shared, *, intermediate_size, expert_count, topk
):
    if shared is None or gate_weight is None:
        return None
    rows, hidden_size = hidden.shape
    cols = int(os.environ.get("SGLANG_Q38_FINALIZE_SHARED_COLS", "256"))
    return numeric_operator(
        ExpertSignature(
            NumericOperation.FINALIZE_SHARED,
            rows,
            hidden_size,
            intermediate_size,
            expert_count,
            topk,
            ScalarType.BF16,
        ),
        prepared=(
            get_bool_env_var("SGLANG_Q38_FINALIZE_SHARED", "False")
            and hidden.is_cuda
            and cols in (256, 512, 1024, 4096)
            and shared.shape == hidden.shape
            and gate_weight.numel() == hidden_size
            and all(
                t.dtype == torch.bfloat16
                and t.is_contiguous()
                and t.device == hidden.device
                for t in (hidden, gate_weight, shared)
            )
        ),
    )


def finish_local_shared_finalize(sink, hidden, gate_weight, shared, output):
    """Complete only a staged local expert; the caller retains stream joins and AR."""
    if not sink:
        return False
    if len(sink) != 1:
        raise RuntimeError("a local finalize sink must have exactly one producer")
    rows, weights, staged_output, run = sink[0]
    if (
        run is None
        or output.data_ptr() != staged_output.data_ptr()
        or output.shape != hidden.shape
        or rows.ndim != 3
        or rows.shape[0] != hidden.shape[0]
        or rows.shape[2] != hidden.shape[1]
        or weights.shape != rows.shape[:2]
    ):
        raise RuntimeError(
            "staged local finalize no longer matches its prepared consumer"
        )
    if rows.is_cuda:
        # These buffers may originate on the producer's side stream. The caller
        # has joined it; retain their allocations for this stream's consumer.
        stream = torch.cuda.current_stream(rows.device)
        for tensor in (rows, weights, output):
            tensor.record_stream(stream)
    run(
        FinalizeSharedIO(
            hidden,
            gate_weight,
            shared,
            rows,
            weights,
            int(os.environ.get("SGLANG_Q38_FINALIZE_SHARED_COLS", "256")),
            get_bool_env_var("SGLANG_Q38_FINALIZE_SHARED_PDL", "True"),
            output,
        )
    )
    return True


def local_expert_source_context(
    rows, tokens_per_request, *, is_target_verify, batch_size
):
    """Resolved framework flags; row-domain and variant acceptance are provider data."""
    return dict(
        tokens_per_request=tokens_per_request,
        sf_pack_enabled=int(get_bool_env_var("SGLANG_Q38_MK_MOE_SF_PACK", "False")),
        mkm_wide_enabled=int(get_bool_env_var("SGLANG_Q38_MKM_WIDE", "False")),
        mkm_wide_m8_enabled=int(get_bool_env_var("SGLANG_Q38_MKM_WIDE_M8", "True")),
        wide_batch34_enabled=int(
            get_bool_env_var("SGLANG_Q38_MKM_WIDE_BATCH34", "False")
        ),
        is_target_verify=int(is_target_verify),
        batch_size=batch_size,
    )


@lru_cache(None)
def _identity(device: torch.device, rows: int, topk: int):
    # Immutable mapping owned by the deferred framework consumer, not the plugin.
    return torch.arange(rows * topk, dtype=torch.int32, device=device)


def try_numeric_local_experts(kwargs):
    """Return the ordinary runner result tuple, or refuse before any writes."""
    from flashinfer.tllm_enums import ActivationType

    from sglang.srt.model_executor.forward_context import (
        get_forward_context,
        has_forward_context,
    )

    x = kwargs["hidden_states"]
    rows, packed_hidden = x.shape
    mode, batch = None, 0
    if has_forward_context():
        context = get_forward_context()
        mode = getattr(context, "moe_forward_mode", None)
        batch = getattr(context, "moe_batch_size", 0)
    if not get_bool_env_var("SGLANG_Q38_MK_MOE", "False"):
        return None
    hidden, topk = packed_hidden * 2, kwargs["top_k"]
    w13, w2 = kwargs["gemm1_weights"], kwargs["gemm2_weights"]
    inputs = tuple(
        kwargs[key]
        for key in (
            "hidden_states_scale",
            "routing_logits",
            "gemm1_weights_scale",
            "gemm2_weights_scale",
            "output1_scale_gate_scalar",
            "output1_scale_scalar",
            "output2_scale_scalar",
        )
    )
    prepared = (
        x.is_cuda
        and x.dtype == w13.dtype == w2.dtype == torch.uint8
        and kwargs["num_experts"] == kwargs["local_num_experts"]
        and kwargs["local_expert_offset"] == 0
        and int(kwargs["routing_method_type"]) == 4
        and kwargs["activation_type"] == ActivationType.Swiglu
        and kwargs["routing_bias"] is None
        and kwargs["per_token_scale"] is None
        and kwargs["routed_scaling_factor"] in (None, 1.0)
        and all(
            kwargs[name] is None
            for name in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit")
        )
        and tuple(w13.shape)
        == (kwargs["num_experts"], kwargs["intermediate_size"] * 2, packed_hidden)
        and tuple(w2.shape)
        == (kwargs["num_experts"], hidden, kwargs["intermediate_size"] // 2)
        and kwargs["routing_logits"].shape == (rows, kwargs["num_experts"])
        and kwargs["routing_logits"].dtype == torch.bfloat16
        and kwargs["hidden_states_scale"].dtype == torch.float8_e4m3fn
        and kwargs["hidden_states_scale"].shape == (rows, hidden // 16)
        and all(
            kwargs[key].dtype == torch.float8_e4m3fn
            for key in ("gemm1_weights_scale", "gemm2_weights_scale")
        )
        and all(
            kwargs[key].dtype == torch.float32
            for key in (
                "output1_scale_gate_scalar",
                "output1_scale_scalar",
                "output2_scale_scalar",
            )
        )
        and all(isinstance(t, torch.Tensor) and t.device == x.device for t in inputs)
        and all(t.is_contiguous() for t in (x, w13, w2, *inputs))
    )
    # Only exact startup offers can select a finite model geometry. Invalid
    # borrowed views never reach registry lookup, and no provider has written.
    from sglang.srt.runtime_context import get_spec

    tokens = get_spec().speculative_num_draft_tokens if mode is not None else 0
    run = None
    for variant in (2, 1, 0):
        if variant == 2 and not (
            mode is not None
            and mode.is_target_verify()
            and tokens > 0
            and rows == tokens * batch
        ):
            continue
        signature = ExpertSignature(
            NumericOperation.LOCAL_EXPERTS,
            rows,
            hidden,
            kwargs["intermediate_size"],
            kwargs["num_experts"],
            topk,
            ScalarType.PACKED_FP4,
            variant=variant,
        )
        run = numeric_operator(signature, prepared=prepared)
        if run is not None or not prepared:
            break
    if run is None:
        return None
    expert_rows = torch.empty(
        (rows, topk, hidden), dtype=torch.bfloat16, device=x.device
    )
    weights = torch.empty((rows, topk), dtype=torch.bfloat16, device=x.device)
    ids = torch.empty((rows, topk), dtype=torch.int32, device=x.device)
    finalize = kwargs["do_finalize"]
    output = kwargs.get("output") if finalize else None
    if finalize and output is None:
        output = torch.empty((rows, hidden), dtype=torch.bfloat16, device=x.device)
    permutation = None if finalize else _identity(x.device, rows, topk)
    sink = _finalize_sink.get() if finalize else None
    # No hidden scratch requirement is currently advertised by this raw ABI.
    scratch = x.new_empty((0,))
    run(
        LocalExpertsIO(
            w13,
            w2,
            kwargs["gemm1_weights_scale"],
            kwargs["gemm2_weights_scale"],
            kwargs["output1_scale_gate_scalar"],
            kwargs["output1_scale_scalar"],
            kwargs["output2_scale_scalar"],
            x,
            kwargs["hidden_states_scale"],
            kwargs["routing_logits"],
            kwargs["num_experts"],
            kwargs["intermediate_size"],
            hidden,
            topk,
            int(kwargs["activation_type"]),
            kwargs["local_expert_offset"],
            kwargs["local_num_experts"],
            1.0,
            weights,
            ids,
            expert_rows,
            output if sink is None else None,
            scratch,
        )
    )
    if finalize:
        if sink is not None:
            target, consumer = sink
            target.append((expert_rows, weights, output, consumer))
        return (output,)
    return expert_rows.view(rows * topk, hidden), weights, permutation
