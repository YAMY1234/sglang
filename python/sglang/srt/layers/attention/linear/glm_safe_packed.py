"""GLM P4 shape qualification; host metadata only, safe during graph capture."""

import math

import torch


def covered(layer, qkv, a, b, state, indices):
    """The GLM TP4 H16/128-dim dense FP32 path, one token per request.

    Unsupported layouts fall back to the existing safe-gate generic kernel.
    Slot validity is a device concern (-1 padding); this guard never reads it.
    """
    lower_bound = getattr(layer, "lower_bound", None)
    if (
        not isinstance(lower_bound, (int, float))
        or not math.isfinite(lower_bound)
        or lower_bound != -5.0
    ):
        return False
    h = layer.num_v_heads
    batch = indices.numel()
    if not (
        layer.num_q_heads == layer.num_k_heads == h
        and h == 16
        and layer.head_q_dim == layer.head_k_dim == layer.head_v_dim == 128
    ):
        return False
    # GLM passes beta as (1, B, H), including row-pitched slices from the
    # fused input projection. Packed decode reshapes these views to (B, H).
    a_shapes = (
        (batch, h * 128),
        (batch, h, 128),
        (batch, 1, h * 128),
        (1, batch, h * 128),
        (1, batch, h, 128),
        (batch, 1, h, 128),
    )
    b_shapes = ((batch, h), (batch, 1, h), (1, batch, h))
    return (
        qkv.device.type == "cuda"
        and all(
            t.device == qkv.device
            for t in (a, b, state, indices, layer.A_log, layer.dt_bias)
        )
        and qkv.dtype in (torch.bfloat16, torch.float32)
        and a.dtype in (torch.bfloat16, torch.float32)
        and b.dtype in (torch.bfloat16, torch.float32)
        and state.dtype == layer.A_log.dtype == layer.dt_bias.dtype == torch.float32
        and indices.dtype in (torch.int32, torch.int64)
        and indices.ndim == 1
        and indices.is_contiguous()
        and qkv.shape == (batch, 3 * h * 128)
        and tuple(a.shape) in a_shapes
        and tuple(b.shape) in b_shapes
        and a.numel() == batch * h * 128
        and b.numel() == batch * h
        and a.stride(-1) == b.stride(-1) == 1
        and (a.shape[-1] != 128 or a.stride(-2) == 128)
        and qkv.stride(-1) == 1
        and state.ndim == 4
        and state.shape[1:] == (h, 128, 128)
        and state.stride(-1) == 1
        and state.stride(-2) == 128
        and state.stride(-3) == 128 * 128
        and layer.A_log.numel() == h
        and layer.A_log.is_contiguous()
        and layer.dt_bias.numel() == h * 128
        and layer.dt_bias.is_contiguous()
    )
