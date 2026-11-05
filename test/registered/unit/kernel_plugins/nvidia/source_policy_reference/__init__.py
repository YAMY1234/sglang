"""Frozen pre-capability source oracle; test-only, never a runtime dependency."""

import torch


def gdn_front_source_family(
    *,
    mk_enabled: bool,
    front_enabled: bool,
    wide_enabled: bool,
    glue_eligible: bool,
    target_verify: bool,
    rows: int,
    hidden_size: int,
    input_rank: int,
    input_dtype: torch.dtype,
    tp_size: int,
    batch_size: int,
    packed_weight_shape: tuple[int, ...] | None,
    packed_weight_dtype: torch.dtype | None,
    activation: str,
) -> str | None:
    """Resolve the original projection/Conv branches, before provider lookup.

    Family labels are framework policy, not provider signature variant numbers.
    The caller still proves the backend metadata and every mutable output view.
    """
    if (
        mk_enabled
        and not glue_eligible
        and target_verify
        and input_rank == 2
        and input_dtype == torch.bfloat16
        and (
            rows in (4, 8, 12)
            or (rows == 16 and tp_size == 4 and batch_size >= 2 and wide_enabled)
        )
        and hidden_size == 2560
        and tp_size in (1, 4)
        and packed_weight_shape is not None
        and packed_weight_dtype == torch.bfloat16
        and activation in ("silu", "swish")
    ):
        return "mk"
    if (
        front_enabled
        and target_verify
        and input_rank == 2
        and input_dtype == torch.bfloat16
        and (rows, hidden_size) == (4, 2560)
        and tp_size == 1
        and packed_weight_shape == (16480, 2560)
        and packed_weight_dtype == torch.bfloat16
        and activation in ("silu", "swish")
    ):
        return "legacy"
    return None


def gdn_front_launch(family, rows, channels, stage_override, legacy_front12):
    """Freeze only source-resolved supported GEMM stages, never round an override."""
    from sglang.kernels.plugin_kernels.nvidia.numeric import (
        GdnDotPrecision,
        GdnFrontVariant,
        GdnLaunchConfig,
    )

    if family == "mk":
        try:
            stages = (
                int("6" if stage_override is None else stage_override)
                if (rows == 4 and channels == 2560)
                else 6
            )
        except ValueError:
            return None
        variants = {6: GdnFrontVariant.MK_STAGES_6, 12: GdnFrontVariant.MK_STAGES_12}
    elif family == "legacy":
        stages = 12 if legacy_front12 else 8
        variants = {8: GdnFrontVariant.TGV_STAGES_8, 12: GdnFrontVariant.TGV_STAGES_12}
    else:
        return None
    if stages not in variants:
        return None
    return variants[stages], GdnLaunchConfig(
        block_n=0,
        num_warps=0,
        num_stages=stages,
        hoist=False,
        dot_precision=GdnDotPrecision.TF32,
        prefetch=False,
        historical_branch=0,
        use_pdl=True,
        trigger_early=False,
    )


# Frozen477 standalone normalized-return source protocol and exact route guard.
import os

_is_cuda = True  # explicit host CUDA-metadata boundary in the oracle


def gdn_m32_protocol_477(self):
    return (
        _is_cuda
        and self.attn_tp_size == 4
        and (os.environ.get("SGLANG_Q38_MK_GDN", "0") == "1")
        and (os.environ.get("SGLANG_Q38_MK_GDN_M32_NORM", "0") == "1")
        and ((self.attn.num_k_heads, self.attn.num_v_heads) == (4, 12))
        and (self.head_k_dim == self.head_v_dim == 128)
        and (self.norm.weight.shape == (128,))
        and (self.norm.weight.dtype == torch.bfloat16)
        and (self.norm.activation == "sigmoid")
        and (self.norm.group_size is None)
        and self.norm.norm_before_gate
        and (self.norm.bias is None)
    )


def standalone_norm_route_477(
    self, hidden_states, forward_batch, use_fused_proj_conv, z
):
    return (
        isinstance(hidden_states, torch.Tensor)
        and hidden_states.shape[0] == 32
        and gdn_m32_protocol_477(self)
        and forward_batch.forward_mode.is_target_verify()
        and (not use_fused_proj_conv)
        and (z is not None)
        and (hidden_states.dtype == z.dtype == torch.bfloat16)
    )
