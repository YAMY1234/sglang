"""Framework-side admission for rank-local GDN numeric operations."""

import torch


def gdn_front_metadata_supported(
    *,
    stable_rows,
    has_mis_metadata,
    tokens_per_request,
    query_start_count,
    batch_size,
    mk_enabled,
    wide_enabled,
    has_next_token,
    has_next_sibling,
    has_parent_token,
):
    """Validate linear verify metadata; finite geometry belongs to the offer."""
    return (
        not stable_rows
        and not has_mis_metadata
        and tokens_per_request > 0
        and batch_size > 0
        and query_start_count == batch_size + 1
        and not has_next_token
        and not has_next_sibling
        and not has_parent_token
    )


def gdn_front_launch_config(variant):
    from sglang.kernels.plugin_kernels.nvidia.numeric import (
        GdnDotPrecision,
        GdnFrontVariant,
        GdnLaunchConfig,
    )

    stages = {GdnFrontVariant.MK_STAGES_6: 6, GdnFrontVariant.MK_STAGES_12: 12}[variant]
    return GdnLaunchConfig(
        0, 0, stages, False, GdnDotPrecision.TF32, False, 0, True, False
    )


def bind_gdn_core(
    *,
    qkv,
    a,
    b,
    a_log,
    dt_bias,
    checkpoint,
    ring,
    key_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    tokens_per_request: int,
    raw_out,
    z=None,
    norm_weight=None,
    normalized_out=None,
    epsilon: float = 1e-6,
):
    """Preflight an exact caller-buffer recurrence before any state writes.

    The optional norm view is only constructed by the original fused front
    producer. A binding is not warmup: the selected raw specialization must
    still be executed eagerly before CUDA graph capture.
    """
    from sglang.kernels.plugin_kernels.nvidia.numeric import (
        GdnCoreIO,
        GdnSignature,
        NumericOperation,
        ScalarType,
    )
    from sglang.srt.kernel_plugins.nvidia.numeric_binding import numeric_operator

    if qkv is None:
        return None
    rows = qkv.shape[0]
    optional = (z, norm_weight, normalized_out)
    fused = any(t is not None for t in optional)
    if fused and not all(t is not None for t in optional):
        return None
    norm_prepared = not fused or (
        z.dtype == norm_weight.dtype == normalized_out.dtype == torch.bfloat16
        and z.shape == normalized_out.shape == (rows, value_heads, value_dim)
        and z.stride(-1) == 1
        and norm_weight.shape == (value_dim,)
        and norm_weight.is_contiguous()
        and normalized_out.is_contiguous()
        and all(t.device == qkv.device for t in optional)
        and a.dtype == b.dtype == torch.bfloat16
    )
    prepared = (
        qkv.is_cuda
        and qkv.dtype == checkpoint.dtype == torch.bfloat16
        and qkv.shape == (rows, (2 * key_heads + value_heads) * key_dim)
        and qkv.stride(1) == 1
        and qkv.stride(0) % 8 == 0
        and qkv.data_ptr() % 16 == 0
        and a.dtype == b.dtype
        and a.dtype in (torch.bfloat16, torch.float32)
        and a.shape == b.shape == (rows, value_heads)
        and a.stride(0) == b.stride(0)
        and a.stride(1) == b.stride(1) == 1
        # The compact ring ABI has sixteen physical entries.
        and tokens_per_request > 0
        and ring.d.shape[-2] == 16
        and key_dim == value_dim
        and ring.raw_v is not None
        and ring.raw_k is not None
        and raw_out.shape == (rows, value_heads, value_dim)
        and raw_out.dtype == torch.bfloat16
        and raw_out.is_contiguous()
        and norm_prepared
    )
    signature = GdnSignature(
        NumericOperation.GDN_CORE,
        rows,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        tokens_per_request,
        ScalarType.BF16,
        variant=int(fused),
        gate_scalar_type=(
            ScalarType.BF16 if a.dtype == torch.bfloat16 else ScalarType.FP32
        ),
    )
    run = numeric_operator(signature, prepared=prepared)
    if run is None:
        return None
    return run, GdnCoreIO(
        qkv,
        a,
        b,
        a.stride(0),
        b.stride(0),
        a_log,
        dt_bias,
        checkpoint,
        ring,
        z,
        norm_weight,
        epsilon,
        value_dim,
        raw_out,
        normalized_out,
    )
