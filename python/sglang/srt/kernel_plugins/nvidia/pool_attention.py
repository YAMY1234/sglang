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
    return tuple(
        signature
        for rows in widths
        if type(rows) is int and rows > 0
        for variant in (2, 1)
        if signature_capability_supported(
            signature := NumericSignature(
                NumericOperation.DIRECT_POOL_ATTENTION,
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
