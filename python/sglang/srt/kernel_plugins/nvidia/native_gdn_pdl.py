"""Framework-owned admission for the original circular native GDN verify."""

import os
from dataclasses import fields

import torch

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NativeGdnPdlIO,
    NativeGdnPdlSignature,
    NumericOperation,
    ScalarType,
)


def _view(tensor, tail, dtype, rows=None):
    if (
        not isinstance(tensor, torch.Tensor)
        or not tensor.is_cuda
        or tensor.dtype != dtype
        or tensor.ndim != len(tail) + 1
        or tuple(tensor.shape[1:]) != tail
        or (rows is not None and tensor.shape[0] != rows)
    ):
        return False
    stride = 1
    for size, actual in zip(reversed(tail), reversed(tensor.stride()[1:])):
        if actual != stride:
            return False
        stride *= size
    return tensor.shape[0] > 0 and tensor.stride(0) >= stride


def state_signature(layer, params, pool, rows, *, prefetch, precision):
    """Read actual allocated state; never enumerate guessed dtype/storage modes."""
    tokens = getattr(layer, "_numeric_tokens_per_request", 0)
    if not (
        os.environ.get("Q38_GDN_PDL") == "1"
        and type(rows) is int
        and rows > 0
        and tokens > 0
        and rows % tokens == 0
        and layer.head_k_dim > 0
        and layer.head_v_dim > 0
        and not pool.replayssm_is_kda
        and not pool.replayssm_spec_fold
        and params.replayssm_d is not None
        and pool.replayssm_cache_base is not None
        and params.temporal.dtype in (torch.bfloat16, torch.float32)
        and layer.dt_bias.dtype in (torch.bfloat16, torch.float32)
    ):
        return None
    extras = tuple(
        t is not None
        for t in (
            params.replayssm_rawv,
            params.replayssm_rawk,
            params.replayssm_beta,
        )
    )
    modes = {(False, False, False): 0, (True, True, False): 1, (True, True, True): 2}
    if extras not in modes:
        return None
    h, hv, slots = layer.num_k_heads, layer.num_v_heads, params.replayssm_d.shape[0]
    dk, dv, length = layer.head_k_dim, layer.head_v_dim, params.replayssm_d.shape[-2]
    if any(
        not _view(t, tail, dtype, count)
        for t, tail, dtype, count in (
            (params.temporal, (hv, dk, dv), params.temporal.dtype, None),
            (params.replayssm_d, (hv, length, dv), torch.bfloat16, slots),
            (params.replayssm_k, (h, length, dk), torch.bfloat16, slots),
            (params.replayssm_g, (hv, length), torch.float32, slots),
        )
    ):
        return None
    for t, tail, dtype in (
        (params.replayssm_rawv, (hv, length, dv), torch.bfloat16),
        (params.replayssm_rawk, (h, length, dk), torch.bfloat16),
        (params.replayssm_beta, (hv, length), torch.float32),
    ):
        if t is not None and not _view(t, tail, dtype, slots):
            return None
    return NativeGdnPdlSignature(
        NumericOperation.GDN_NATIVE_PDL,
        rows,
        layer.num_k_heads,
        layer.num_v_heads,
        dk,
        dv,
        tokens,
        ScalarType.BF16,
        cache_len=length,
        state_scalar_type=(
            ScalarType.BF16
            if params.temporal.dtype == torch.bfloat16
            else ScalarType.FP32
        ),
        dt_bias_scalar_type=(
            ScalarType.BF16
            if layer.dt_bias.dtype == torch.bfloat16
            else ScalarType.FP32
        ),
        storage_mode=modes[extras],
        state_prefetch=prefetch,
        dot_precision=precision,
    )


def bind_native_gdn_pdl(
    layer,
    params,
    pool,
    *,
    q,
    k,
    v,
    a,
    b,
    out,
    query_start_loc,
    state_indices,
    replay_indices,
    wait_for_gate_producer=False,
):
    """All tensor/metadata checks precede lookup and the first ring write.

    Native convolution has already run once. A miss leaves its output for the
    ordinary native core; a selected exception must propagate without retry.
    """
    if os.environ.get("Q38_GDN_PDL") != "1":
        return None
    candidates = getattr(layer, "_numeric_native_pdl_signatures", ())
    signature = next(
        (
            s
            for s in candidates
            if s.rows == q.shape[0]
            and s.wait_for_gate_producer == int(wait_for_gate_producer)
        ),
        None,
    )
    if signature is None:
        return None
    if (
        state_signature(
            layer,
            params,
            pool,
            q.shape[0],
            prefetch=signature.state_prefetch,
            precision=signature.dot_precision,
        )
        != signature
    ):
        return None
    m, h, hv = signature.rows, signature.key_heads, signature.value_heads
    dk, dv = signature.key_dim, signature.value_dim
    slots = params.replayssm_d.shape[0]
    bias_dtype = torch.bfloat16 if signature.dt_bias_scalar_type == 1 else torch.float32
    if any(
        not _view(t, tail, dtype, rows)
        for t, tail, dtype, rows in (
            (q, (h, dk), torch.bfloat16, m),
            (k, (h, dk), torch.bfloat16, m),
            (v, (hv, dv), torch.bfloat16, m),
            (a, (hv,), torch.bfloat16, m),
            (b, (hv,), torch.bfloat16, m),
            (out, (hv, dv), torch.bfloat16, m),
        )
    ):
        return None
    for t, dtype in ((layer.A_log, torch.float32), (layer.dt_bias, bias_dtype)):
        if (
            not isinstance(t, torch.Tensor)
            or t.shape != (hv,)
            or t.dtype != dtype
            or not t.is_contiguous()
        ):
            return None
    batch = m // signature.tokens_per_request
    if any(
        not isinstance(t, torch.Tensor)
        or t.ndim != 1
        or t.numel() < size
        or t.dtype not in (torch.int32, torch.int64)
        or t.stride(0) < 1
        for t, size in (
            (query_start_loc, batch + 1),
            (state_indices, batch),
            (replay_indices, batch),
        )
    ):
        return None
    if query_start_loc.numel() != batch + 1:
        return None
    for t, dtype in (
        (pool.replayssm_spec_write_pos, torch.int32),
        (pool.replayssm_cache_base, torch.int32),
        (pool.replayssm_is_flush, torch.int8),
    ):
        if (
            not isinstance(t, torch.Tensor)
            or t.ndim != 1
            or t.numel() < slots
            or t.dtype != dtype
            or not t.is_contiguous()
        ):
            return None
    io = NativeGdnPdlIO(
        q,
        k,
        v,
        a,
        b,
        layer.A_log,
        layer.dt_bias,
        out,
        params.temporal,
        params.replayssm_d,
        params.replayssm_k,
        params.replayssm_g,
        params.replayssm_rawv,
        params.replayssm_rawk,
        params.replayssm_beta,
        query_start_loc,
        state_indices[:batch],
        replay_indices[:batch],
        pool.replayssm_spec_write_pos,
        pool.replayssm_cache_base,
        pool.replayssm_is_flush,
    )
    if any(
        not t.is_cuda or t.device != q.device
        for f in fields(io)
        if (t := getattr(io, f.name)) is not None
    ):
        return None
    from sglang.srt.kernel_plugins.nvidia.numeric_binding import numeric_operator

    run = numeric_operator(signature, prepared=True)
    return (run, io) if run is not None else None
