"""Framework policy and caller workspace for the original TP1 pool attention."""

import math
import os
from functools import lru_cache

import torch

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericOperation,
    NumericSignature,
    ScalarType,
)
from sglang.srt.environ import envs


@lru_cache(maxsize=1)
def pool_decode_enabled() -> bool:
    # Match the source pool route's default-on kill switch.
    return os.environ.get("SGLANG_QSA_POOL_DECODE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


def pool_signatures(
    widths,
    *,
    tp_size,
    query_heads,
    kv_heads,
    head_dim,
    selected_count,
    compress_ratio,
    spec=None,
):
    from sglang.srt.kernel_plugins.nvidia.numeric_binding import (
        signature_capability_supported,
    )

    if not (
        envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get() and pool_decode_enabled()
    ):
        return ()
    routes = [
        (NumericOperation.DIRECT_POOL_ATTENTION, variant, 1)
        for variant in (2, verify_variant())
    ]
    if envs.SGLANG_QSA_FP8_POOL_VERIFY.get():
        # The FP8 union serves width-4 target verify only.
        routes.append((NumericOperation.POOL_UNION_ATTENTION, union_variant(), 4))
    return tuple(
        signature
        for rows in widths
        if type(rows) is int and rows > 0
        for operation, variant, width in routes
        if rows % width == 0
        and signature_capability_supported(
            signature := NumericSignature(
                operation,
                rows,
                query_heads,
                kv_heads,
                head_dim,
                selected_count,
                compress_ratio,
                ScalarType.BF16,
                variant=variant,
            ),
            spec=spec,
            tp_size=tp_size,
        )
    )


_UNION_WS = 1 << 6
_UNION_WS_PDL = 1 << 7
# DIRECT_POOL_ATTENTION variant bit 2: the output gate rides the split-K merge.
POOL_MERGE_GATE = 1 << 2


def verify_variant():
    """DIRECT_POOL_ATTENTION's width-4 verify variant: 1, with POOL_MERGE_GATE under
    the source's SGLANG_ENABLE_QSA_POOL_MERGE_GATE (decode, 2, never takes the gate)."""
    return 1 | (POOL_MERGE_GATE if envs.SGLANG_ENABLE_QSA_POOL_MERGE_GATE.get() else 0)


def union_variant():
    """POOL_UNION_ATTENTION's source options: bits 0-5 = the token-union pass's warp
    count (0 for the source's 8), bit 6 = the warp-specialized attention kernel, bit 7
    = PDL into it (with bit 6 only)."""
    warps = envs.SGLANG_QSA_FP8_POOL_VERIFY_UNION_WARPS.get()
    if warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError(
            "SGLANG_QSA_FP8_POOL_VERIFY_UNION_WARPS must be 1, 2, 4, 8, 16 or 32, "
            f"got {warps}"
        )
    ws = _UNION_WS if envs.SGLANG_QSA_FP8_POOL_VERIFY_WS.get() else 0
    if ws and envs.SGLANG_QSA_FP8_POOL_VERIFY_WS_PDL.get():
        ws |= _UNION_WS_PDL
    return (0 if warps == 8 else warps) | ws


@lru_cache(maxsize=None)
def pool_device_sms(device):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("TP1 pool SM count must be prepared before capture")
    return torch.cuda.get_device_properties(device).multi_processor_count


def pool_scratch_elements(
    rows, sms, *, kv_heads, head_dim, query_heads, selected_count, compress_ratio
):
    # Same auto choose_config as the source: 512 blocks * 4 + 3 tail tokens.
    rh = rows * kv_heads
    bn = 32 if rh >= 32 else 64
    waves = 3.0 if rh >= 32 else 1.0
    splits = max(1, round(waves * sms / rh))
    columns = selected_count * compress_ratio + compress_ratio - 1
    chunk = math.ceil(math.ceil(columns / splits) / bn) * bn
    splits = math.ceil(columns / chunk)
    padded_heads = 1 << ((query_heads // kv_heads) - 1).bit_length()
    return rows * kv_heads * splits * padded_heads * (head_dim + 2)


def union_workspace(
    rows,
    sms,
    *,
    kv_heads,
    head_dim,
    query_heads,
    selected_count,
    compress_ratio,
    variant=0,
):
    """The source W4 union plan: (splits, INT32 elements, FP32 elements)."""
    requests = rows // 4
    pairs = max(1, requests * kv_heads)
    if variant & _UNION_WS:
        # The warp-specialized kernel: SGLANG_QSA_FP8_POOL_VERIFY_WS_SPLITS, else the
        # most splits that keep one wave, at most 4.
        splits = envs.SGLANG_QSA_FP8_POOL_VERIFY_WS_SPLITS.get()
        if splits is None:
            splits = max(1, min(4, sms // pairs))
        elif not 1 <= splits <= 16:
            raise ValueError(
                f"SGLANG_QSA_FP8_POOL_VERIFY_WS_SPLITS must be in 1..16, got {splits}"
            )
    elif pairs > sms:
        splits = 2
    else:
        # Least wave-quantized work per CTA, each split charged 10% for its merge.
        splits = min(
            range(1, 17),
            key=lambda s: (-(-pairs * s // sms) / s * (1.0 + 0.1 * s), s),
        )
    sort_width = 1 << (4 * selected_count - 1).bit_length()
    tail = max(16, 1 << (4 + 2 * compress_ratio - 1).bit_length())
    tokens = sort_width * compress_ratio + tail
    padded_rows = max(16, 1 << (4 * query_heads // kv_heads - 1).bit_length())
    stats = requests * kv_heads * splits * padded_rows
    integers = requests * (2 * tokens + sort_width + 1) + rows
    return splits, integers, stats * (head_dim + 2)
