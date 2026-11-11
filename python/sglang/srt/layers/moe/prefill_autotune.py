"""FlashInfer autotune of the NVFP4 TRT-LLM MoE at prefill token counts.

The startup autotune runs a decode-shaped dummy forward, so FlashInfer profiles
``trtllm_fp4_block_scale_moe`` only up to the decode token count; every larger
prefill MoE call misses the tactic cache and runs the default tactic. Like the
GLM and DeepSeek hooks, this makes one routed-experts call of the first eligible
``Qwen2MoeSparseMoeBlock`` at the prefill ceiling inside the runner's autotune
context, through the block's own routed path with the routed-input pre-quant a
captured prefill segment uses, so it stores the cache keys served prefill calls
look up. The router GEMM and the shared expert are not called, so no other op is
tuned at prefill sizes.

Only the choice among FlashInfer's precompiled tactics changes. Each boot with
the hook on runs one extra tuning pass over the prefill buckets, unless the
autotune cache already holds them.
"""

from __future__ import annotations

import functools
import logging
import time

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_MOE_OP = "flashinfer::trtllm_fp4_block_scale_moe"
# The dual-stream MoE forward's router-front paths serve at most 16 tokens; the
# hook mirrors its larger-M branch.
_MIN_TOKENS = 17


@functools.lru_cache(maxsize=None)
def _log_inert_once(reason: str) -> None:
    logger.debug("MoE prefill autotune inactive: %s", reason)


def _eligible_block(model):
    """First NVFP4 TRT-LLM MoE block on the TP1/EP1 glue path, or None."""
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    for module in model.modules():
        if (
            isinstance(module, Qwen2MoeSparseMoeBlock)
            and module._uses_nvfp4_trtllm_glue()
        ):
            return module
    return None


def wants_moe_prefill_autotune(model) -> bool:
    if not envs.SGLANG_OPT_MOE_PREFILL_AUTOTUNE.get():
        return False
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled
    from sglang.srt.layers.moe.utils import (
        get_moe_a2a_backend,
        get_moe_runner_backend,
    )

    if not get_moe_runner_backend().is_flashinfer_trtllm():
        _log_inert_once("the MoE runner backend is not flashinfer_trtllm")
        return False
    if not get_moe_a2a_backend().is_none() or is_dp_attention_enabled():
        _log_inert_once("needs no MoE all-to-all and no DP attention")
        return False
    if _eligible_block(model) is None:
        _log_inert_once("no NVFP4 TRT-LLM MoE block on the TP1/EP1 path")
        return False
    return True


def _route_one_call(block, num_tokens: int, dtype: torch.dtype) -> bool:
    """The routed-experts call of the TP1 dual-stream branch; returns whether the
    routed input was pre-quantized."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        quantize_nvfp4_routed_input,
    )

    prequant = envs.SGLANG_OPT_MOE_NVFP4_PREQUANT_OVERLAP.get() and block.tp_size == 1
    device = block.gate.weight.device
    with torch.no_grad():
        hidden_states = torch.randn(
            (num_tokens, block.gate.weight.shape[1]), dtype=dtype, device=device
        )
        # The router output's shape and dtype; the tuner profiles on its own
        # synthesized logits, and skipping the GEMM keeps it untuned at these sizes.
        router_logits = torch.randn(
            (num_tokens, block.gate.weight.shape[0]),
            dtype=block.gate.weight.dtype,
            device=device,
        )
        routed_input = (
            quantize_nvfp4_routed_input(
                hidden_states, block.experts.w13_input_scale_quant
            )
            if prequant
            else None
        )
        block._forward_router_experts(
            hidden_states, precomputed_router=(router_logits, routed_input)
        )
        torch.cuda.current_stream().synchronize()
    return prequant


def _moe_entries(winners: dict) -> set:
    return {key for key in winners if key.custom_op == _MOE_OP}


def autotune_moe_prefill(model, num_tokens: int, *, dtype: torch.dtype) -> int:
    from flashinfer.autotuner import AutoTuner

    block = _eligible_block(model)
    if block is None or num_tokens < _MIN_TOKENS:
        _log_inert_once(f"no eligible block, or {num_tokens} tokens < {_MIN_TOKENS}")
        return 0

    tuner = AutoTuner.get()
    before = _moe_entries(tuner._winner_cache())
    started = time.perf_counter()
    oom = False
    prequant = None
    try:
        prequant = _route_one_call(block, num_tokens, dtype)
    except torch.OutOfMemoryError:
        from sglang.srt.model_executor.runner.flashinfer_autotune import (
            _autotune_tactic_sync_group,
        )
        from sglang.srt.runtime_context import get_parallel

        if _autotune_tactic_sync_group(get_parallel().tp_group) is not None:
            raise  # Tuning is collective: a rank that stops alone hangs its peers.
        oom = True
    finally:
        torch.cuda.empty_cache()  # Free the dummy rows before graph capture sizes memory.

    with tuner._lock:
        tuned = len(_moe_entries(tuner._winner_cache()) - before)
    logger.info(
        "MoE prefill autotune (SGLANG_OPT_MOE_PREFILL_AUTOTUNE): %d tokens, %d "
        "buckets tuned this boot, prequant=%s, %.1f s%s",
        num_tokens,
        tuned,
        prequant,
        time.perf_counter() - started,
        ", skipped after OOM" if oom else "",
    )
    if oom:
        return 0
    if envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND.get():
        return 0  # An explicitly requested whole-model EXTEND pass still runs.
    return max(1, tuned)
