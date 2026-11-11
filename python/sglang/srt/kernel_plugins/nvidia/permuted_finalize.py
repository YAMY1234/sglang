"""Framework policy for native permuted MoE finalize plus shared gate/add.

Native expert dispatch, streams and fallback stay here; a provider receives
only the frozen seven-tensor view and writes the framework's output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from sglang.srt.utils import get_bool_env_var

_SOURCE_ENABLED = get_bool_env_var(
    "SGLANG_MOE_FUSED_FINALIZE_GATE",
    os.environ.get("SGLANG_QWEN_MOE_FUSED_FINALIZE_GATE", "true"),
)
# The source's extend fast path launches rows at or above its floor with a
# register cap; below the floor its launch is the plain one offered here.
_SOURCE_EXTEND = os.environ.get("SGLANG_Q38_EXTEND_FASTPATH", "0") == "1" and (
    os.environ.get("SGLANG_X_MOE_EXTEND_FUSED_FINALIZE_DISABLE", "false").lower()
    not in ("true", "1")
)
_SOURCE_EXTEND_MIN_ROWS = int(
    os.environ.get("SGLANG_X_MOE_EXTEND_FUSED_FINALIZE_MIN_TOKENS", "1024")
)


def _source_eligible(producer):
    import torch

    from sglang.srt.environ import envs

    return (
        _SOURCE_ENABLED
        and envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get()
        and producer.shared_expert is not None
        and producer.shared_expert_gate is not None
        and producer.shared_expert_gate.weight.dtype == torch.bfloat16
        and producer._uses_nvfp4_trtllm_glue()
    )


def _source_rows(rows):
    """Whether the source finalizes ``rows`` with the fused launch."""
    from sglang.srt.environ import envs

    if _SOURCE_EXTEND and rows >= _SOURCE_EXTEND_MIN_ROWS:
        # The source's extend fused finalize: the same kernel at any row count
        # (its register cap changes no value); otherwise these rows stay stock.
        return envs.SGLANG_OPT_MOE_EXTEND_FUSED_FINALIZE.get()
    return rows <= envs.SGLANG_QWEN_MOE_FUSED_FINALIZE_GATE_MAX_TOKENS.get()


def _signature(producer, rows, weight_scalar_type):
    from sglang.kernels.jit.utils import is_arch_support_pdl
    from sglang.kernels.plugin_kernels.nvidia.numeric import (
        FinalizePermutedSignature,
        NumericOperation,
        ScalarType,
    )
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

    return FinalizePermutedSignature(
        NumericOperation.FINALIZE_PERMUTED_SHARED,
        rows,
        producer.gate.weight.shape[1],
        producer.experts.top_k,
        ScalarType.BF16,
        weight_scalar_type,
        # Rows 0 stands for every row count above the graph widths (>= 1024).
        num_warps=8
        if rows == 0 or rows >= 1024 or is_batch_invariant_mode_enabled()
        else 16,
        use_pdl=int(is_arch_support_pdl()),
    )


def numeric_signatures(producer, widths):
    """Ask for the actual native geometry before any expert deferral.

    Captured prefill runs wider than the graph widths; one rows-0 declaration
    covers those rows when the source would still fuse the first of them.
    """
    from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType
    from sglang.srt.kernel_plugins.nvidia.numeric_binding import (
        signature_capability_supported,
    )
    from sglang.srt.runtime_context import get_parallel

    if not _source_eligible(producer):
        return ()
    widest = max((rows for rows in widths if type(rows) is int), default=0)
    declared_rows = [
        rows for rows in widths if type(rows) is int and rows > 0 and _source_rows(rows)
    ]
    if widest > 0 and (
        _source_rows(widest + 1)
        or _source_rows(max(widest + 1, _SOURCE_EXTEND_MIN_ROWS))
    ):
        declared_rows.append(0)
    signatures = tuple(
        signature
        for rows in declared_rows
        for weight_type in (ScalarType.BF16, ScalarType.FP32)
        if signature_capability_supported(
            signature := _signature(producer, rows, weight_type),
            spec=getattr(producer, "_numeric_capability_spec", None),
            tp_size=producer.tp_size,
            ep_size=get_parallel().moe_ep_size,
            expert_count=producer.gate.weight.shape[0],
        )
    )

    producer._numeric_permuted_finalize_signatures = frozenset(signatures)
    producer._numeric_permuted_finalize_widest = widest
    return signatures


@dataclass(frozen=True, slots=True)
class _Plan:
    bf16: object
    fp32: object
    topk: int


def prepare(producer, hidden, *, use_fused_gate, ordered_finalize_sink):
    """Resolve all possible native output capabilities BEFORE deferring experts."""
    import torch

    from sglang.kernels.plugin_kernels.nvidia.numeric import ScalarType
    from sglang.srt.kernel_plugins.nvidia.numeric_binding import numeric_operator

    if ordered_finalize_sink is not None or not use_fused_gate:
        return None
    if not _source_eligible(producer):
        return None
    rows, hidden_size = hidden.shape
    gate = producer.shared_expert_gate.weight.squeeze(0)
    if not (
        rows > 0
        and hidden_size == producer.gate.weight.shape[1]
        and hidden.is_cuda
        and hidden.dtype == torch.bfloat16
        and hidden.is_contiguous()
        and gate.shape == (hidden_size,)
        and gate.dtype == torch.bfloat16
        and gate.device == hidden.device
        and gate.is_contiguous()
    ):
        return None
    if not _source_rows(rows):
        return None
    declared = getattr(producer, "_numeric_permuted_finalize_signatures", ())
    # Rows above the graph widths take the one rows-0 declaration.
    key = (
        rows if rows <= getattr(producer, "_numeric_permuted_finalize_widest", 0) else 0
    )
    if any(
        _signature(producer, key, dtype) not in declared
        for dtype in (ScalarType.BF16, ScalarType.FP32)
    ):
        return None
    # FlashInfer accepts either weight dtype. Require BOTH exact offers before
    # requesting do_finalize=False rather than guessing its runtime output type.
    bf16 = numeric_operator(_signature(producer, key, ScalarType.BF16), prepared=True)
    fp32 = numeric_operator(_signature(producer, key, ScalarType.FP32), prepared=True)
    if bf16 is None or fp32 is None:
        return None
    return _Plan(bf16, fp32, producer.experts.top_k)


def _native_layout(hidden, gate, shared, deferred):
    """Preflight native output metadata without reading a GPU permutation map."""
    import torch

    gemm2 = deferred.gemm2_out
    weights = deferred.expert_weights
    permutation = deferred.expanded_idx_to_permuted_idx
    rows, hidden_size = hidden.shape
    return (
        shared is not None
        and deferred.top_k > 0
        and shared.shape == hidden.shape
        and shared.dtype == torch.bfloat16
        and gate.shape == (hidden_size,)
        and gate.dtype == torch.bfloat16
        and gemm2.ndim == 2
        and gemm2.dtype == torch.bfloat16
        and gemm2.shape[1] >= hidden_size
        and weights.shape == (rows, deferred.top_k)
        and weights.dtype in (torch.bfloat16, torch.float32)
        and permutation.dtype == torch.int32
        and permutation.numel() == rows * deferred.top_k
        and all(
            t.is_cuda and t.device == hidden.device
            for t in (hidden, gate, shared, gemm2, weights, permutation)
        )
    )


def finish(plan, hidden, gate, shared, deferred):
    """Consume once after the caller's real join and record_stream operations."""
    import torch

    from sglang.kernels.plugin_kernels.nvidia.numeric import FinalizePermutedSharedIO

    if deferred.top_k != plan.topk or not _native_layout(
        hidden, gate, shared, deferred
    ):
        raise RuntimeError("native permuted finalize returned incompatible metadata")
    gemm2 = deferred.gemm2_out
    weights = deferred.expert_weights
    permutation = deferred.expanded_idx_to_permuted_idx
    if not (
        hidden.is_contiguous()
        and gate.is_contiguous()
        and shared.is_contiguous()
        and weights.is_contiguous()
        and permutation.is_contiguous()
        and gemm2.stride(1) == 1
        and gemm2.stride(0) > 0
    ):
        # The original native deferred protocol rounds routed to BF16 first.
        # Never rerun experts, and never add gated shared in FP32 before that
        # boundary. Only its unpadded native output can feed the original add.
        if gemm2.shape[1] != hidden.shape[1]:
            raise RuntimeError("native fallback cannot preserve the unpadded output")
        from sglang.kernels.ops.elementwise.elementwise import (
            fused_gate_sigmoid_mul_add,
        )
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            finalize_flashinfer_trtllm_deferred_output,
        )

        routed = finalize_flashinfer_trtllm_deferred_output(deferred, None)
        fused_gate_sigmoid_mul_add(hidden, gate, shared, routed)
        return routed
    output = torch.empty_like(shared)
    run = plan.bf16 if weights.dtype == torch.bfloat16 else plan.fp32
    # A provider failure is fatal: a partial write must never replay stock.
    run(
        FinalizePermutedSharedIO(
            hidden,
            gate,
            shared,
            gemm2,
            permutation,
            weights,
            output,
        )
    )
    return output
