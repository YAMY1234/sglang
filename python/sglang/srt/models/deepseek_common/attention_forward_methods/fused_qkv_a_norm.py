"""MLA's q_a and kv_a RMSNorms as one launch (SGLANG_OPT_FUSED_MLA_QKV_A_NORM).

At decode, target verify and draft extend the two norms are two small launches
(on two streams under CUDA graphs). ``fused_qkv_a_rmsnorm`` replaces them with
one Triton launch, returning what the unfused path would.
"""

from typing import Optional, Tuple

import torch

from sglang.kernels.fused_op import get_fused_op_backend
from sglang.kernels.ops.layernorm.mla_qkv_a_rmsnorm import mla_qkv_a_rmsnorm
from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def fused_qkv_a_rmsnorm(
    attn: torch.nn.Module, forward_batch: ForwardBatch
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """``(q, k_nope, latent_cache)`` as the unfused q_a/kv_a norms leave them
    (``q`` and ``k_nope`` normalized and contiguous, ``latent_cache`` the
    projection's ``[k_nope | k_pe]`` columns), or None where the unfused path
    must run."""
    mode = forward_batch.forward_mode
    if not (
        (mode.is_decode() or mode.is_target_verify() or mode.is_draft_extend_v2())
        and _is_plain_rmsnorm(attn.q_a_layernorm)
        and _is_plain_rmsnorm(attn.kv_a_layernorm)
    ):
        return None
    latent = get_attn_tp_context().fetch_qkv_latent()
    if latent.dtype != torch.bfloat16:
        return None
    q, k_nope = mla_qkv_a_rmsnorm(
        latent,
        q_weight=attn.q_a_layernorm.weight,
        q_eps=attn.q_a_layernorm.variance_epsilon,
        kv_weight=attn.kv_a_layernorm.weight,
        kv_eps=attn.kv_a_layernorm.variance_epsilon,
    )
    return q, k_nope, latent[..., attn.q_lora_rank :]


def _is_plain_rmsnorm(norm: torch.nn.Module) -> bool:
    return (
        type(norm) is RMSNorm
        and norm.has_weight
        and not norm.cast_x_before_out_mul
        and norm.variance_size_override is None
        and norm.override_orig_dtype is None
        and norm._quant_linear is None
        and norm.weight.dtype == torch.bfloat16
        and not norm.is_torch_compile
        and get_fused_op_backend() is None
        and not is_batch_invariant_mode_enabled()
    )
