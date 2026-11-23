# SPDX-License-Identifier: Apache-2.0
"""KdaFlow: the persistent KDA verify with RecoverSSM records and the deferred fold
(SGLANG_OPT_KDA_FLOW), host side.

Records live in the KDA ReplaySSM ring buffers of each layer (no new memory):
corrections u_t [slot, H, q, 128] FP32 in rows 0..q-1 of ``replayssm_g``, and the
conv outputs k_t with the raw gates [slot, H, q, 256] BF16 in ``replayssm_rawk``
viewed as [slot, H, L / 2, 256]. The record slot is the mamba slot; the pending
count per slot is MambaPool.kda_fold_pending (set by the commit, cleared when a
slot changes owner).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

HEAD_DIM = 128
_COMPILED: dict = {}
_LOGGED: set = set()


def kda_flow_records(*, g_ring: torch.Tensor, rawk_ring: torch.Tensor, tokens: int):
    """(corrections, key_gate) views of one layer's rings, or of all layers' when
    the rings carry a leading layer dim."""
    lead = g_ring.shape[:-3]
    heads, ring = g_ring.shape[-3], g_ring.shape[-2]
    if ring % 2 or ring // 2 < tokens or rawk_ring.shape[-2] != ring:
        raise ValueError(f"KdaFlow records need ring length >= 2 q, got {ring}")
    corrections = g_ring[..., :tokens, :]
    key_gate = rawk_ring.view(*lead, heads, ring // 2, 2 * HEAD_DIM)[..., :tokens, :]
    return corrections, key_gate


def prepare_conv_weight(conv_weights: torch.Tensor) -> torch.Tensor:
    """[3 D, 4] FP32 (q, k, v) -> the kernel's [3, 4, D] layout."""
    width = conv_weights.shape[-1]
    return conv_weights.detach().float().view(3, -1, width).transpose(1, 2).contiguous()


def _cute_dtype(dtype):
    import cutlass

    return {
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
        torch.int32: cutlass.Int32,
    }[dtype]


def _layout(t: torch.Tensor):
    return tuple(t.shape), tuple(t.stride()), t.dtype


def _check_spills(compiled, device_index: int, key) -> None:
    """Fail loudly when the compiled kernel spills: KdaFlow at q = 5 sits at the
    255-register cap with no local memory, and a toolchain that spills would run
    it slowly instead of failing."""
    import cuda.bindings.driver as cuda

    def checked(result):
        status, *values = result
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"KdaFlow: CUDA driver call failed: {status}")
        return values

    compiled.to(device_index)
    library = compiled.library
    (count,) = checked(cuda.cuLibraryGetKernelCount(library))
    (kernels,) = checked(cuda.cuLibraryEnumerateKernels(count, library))
    attr = cuda.CUfunction_attribute
    for kernel in kernels:
        (function,) = checked(cuda.cuKernelGetFunction(kernel))
        (regs,) = checked(cuda.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_NUM_REGS, function))
        (local,) = checked(
            cuda.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function)
        )
        if local > 0:
            raise RuntimeError(
                f"KdaFlow {key[:6]} spills: {regs} registers, {local} B local memory "
                "per thread; it would run slowly (SGLANG_OPT_KDA_FLOW)."
            )
        tag = ("regs",) + key[:6]
        if tag not in _LOGGED:
            _LOGGED.add(tag)
            logger.info("KdaFlow %s: %d registers, no local memory", key[:6], regs)


def _compile(args, *, heads, tokens, requests, fused_fb, device_index):
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    from .kernel import KdaFlow, max_items_per_cta, shared_bytes, smem_budget

    props = torch.cuda.get_device_properties(device_index)
    items = heads * requests
    ctas = min(items, props.multi_processor_count)
    if -(-items // ctas) > max_items_per_cta(requests):
        raise ValueError(
            f"KdaFlow: {requests} requests x {heads} heads exceed "
            f"{max_items_per_cta(requests)} items per CTA"
        )
    conv_size = args[2].element_size()
    budget = smem_budget(props.major, props.minor, props.shared_memory_per_block_optin)
    smem_bytes = shared_bytes(tokens, conv_size)
    if smem_bytes > budget:
        raise ValueError(f"KdaFlow q{tokens}: shared memory over {budget} B")
    fakes = []
    for a in args:
        if isinstance(a, torch.Tensor):
            fakes.append(
                make_fake_tensor(_cute_dtype(a.dtype), tuple(a.shape), tuple(a.stride()),
                                 assumed_align=16)  # fmt: skip
            )
        else:
            import cutlass

            fakes.append(cutlass.Float32(a))
    kernel = KdaFlow(heads, tokens, requests, ctas, fused_fb, smem_bytes)
    with torch.cuda.device(device_index):
        return cute.compile(kernel, *fakes, make_fake_stream())


def kda_flow_verify(
    *,
    x: torch.Tensor,  # [R, 3 D] BF16, q | k | v rows (row stride free)
    conv_weight: torch.Tensor,  # [3, 4, D] FP32 (prepare_conv_weight)
    conv_state: torch.Tensor,  # [pool, 3, 3 D] BF16, tap-major
    gate_in: torch.Tensor,  # [R, D] BF16 raw gate, or [R, 128] F_A when f_b is given
    beta: torch.Tensor,  # [R, H] BF16, before the sigmoid
    a_log: torch.Tensor,  # [H] FP32
    dt_bias: torch.Tensor,  # [D] FP32
    slots: torch.Tensor,  # [n] int32 mamba slots (<= 0: padding)
    state: torch.Tensor,  # [pool, H, 128, 128] FP32
    out_gate: torch.Tensor,  # [R, D] BF16 output-norm gate
    norm_weight: torch.Tensor,  # [128] FP32
    f_b: torch.Tensor | None,  # [D, 128] BF16, or None (gate_in is the raw gate)
    conv_windows: torch.Tensor,  # [scratch, >= q, 3, 3 D] BF16 per-token windows
    scratch_rows: torch.Tensor,  # [n] int32
    corrections: torch.Tensor,  # [slots, H, q, 128] FP32 (kda_flow_records)
    key_gate: torch.Tensor,  # [slots, H, q, 256] BF16
    pending: torch.Tensor,  # [slots + 1] int32
    lower_bound: float,
    eps: float,
) -> torch.Tensor:
    """One KDA layer's verify over n requests of q rows: folds each slot's pending
    accepted rows into ``state``, writes the conv windows and this verify's records,
    returns the gated-RMSNorm output [R, D] BF16."""
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack

    rows, n = x.shape[0], slots.shape[0]
    tokens = rows // n
    heads = a_log.shape[0]
    if rows != n * tokens or not 2 <= tokens <= 5:
        raise ValueError(f"KdaFlow: {rows} rows over {n} requests (2..5 rows each)")
    fused_fb = f_b is not None
    if conv_weight.dtype != torch.float32 or dt_bias.dtype != torch.float32:
        raise ValueError("KdaFlow needs FP32 conv weights and dt bias")
    out = torch.empty((rows, heads * HEAD_DIM), dtype=torch.bfloat16, device=x.device)
    args = (
        x, conv_weight, conv_state, gate_in, beta, a_log, dt_bias, slots, state, out,
        out_gate, norm_weight, f_b if fused_fb else conv_weight, conv_windows,
        scratch_rows, corrections, key_gate, pending,
        float(lower_bound), HEAD_DIM**-0.5, float(eps),
    )  # fmt: skip
    device_index = x.device.index
    key = (
        device_index, heads, tokens, n, fused_fb,
        *(_layout(a) if isinstance(a, torch.Tensor) else ("s",) for a in args),
    )  # fmt: skip
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = _compile(args, heads=heads, tokens=tokens, requests=n,
                            fused_fb=fused_fb, device_index=device_index)  # fmt: skip
        _check_spills(compiled, device_index, key)
        _COMPILED[key] = compiled
    import cutlass

    views = tuple(
        from_dlpack(a.detach(), assumed_align=16) if isinstance(a, torch.Tensor) else cutlass.Float32(a)
        for a in args
    )
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled(*views, stream)
    return out


# ---------------------------------------------------------------- reader flush
@triton.jit
def _kda_flow_fold_kernel(
    state_ptr, corr_ptr, kg_ptr, a_log_ptr, dt_ptr, slots_ptr, pending_ptr,
    lower_bound,
    s_state_l, s_state_s, s_state_h,
    s_corr_l, s_corr_s, s_corr_h, s_corr_t,
    s_kg_l, s_kg_s, s_kg_h, s_kg_t,
    H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, BV: tl.constexpr,
):  # fmt: skip
    """S = S_0 * D_n + sum_{t<n} (u_t * kn_t) * P_t for one (layer, head), slot and
    V tile, accumulated over t = q - 1 .. 0 as the KdaFlow fold does."""
    pid_v, pid_b, pid_lh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    layer, head = pid_lh // H, pid_lh % H
    slot = tl.load(slots_ptr + pid_b).to(tl.int64)
    if slot <= 0:
        return
    n = tl.minimum(tl.load(pending_ptr + slot), T)
    if n <= 0:
        return
    offs_k = tl.arange(0, K)
    offs_v = pid_v * BV + tl.arange(0, BV)
    a = tl.exp(tl.load(a_log_ptr + layer * H + head))
    dt = tl.load(dt_ptr + (layer * H + head) * K + offs_k)
    corr = corr_ptr + layer * s_corr_l + slot * s_corr_s + head * s_corr_h
    kg = kg_ptr + layer * s_kg_l + slot * s_kg_s + head * s_kg_h
    final = tl.full([K], 1.0, tl.float32)
    fc = tl.zeros([BV, K], dtype=tl.float32)
    for i in tl.static_range(T):
        t = T - 1 - i
        if t < n:
            k = tl.load(kg + t * s_kg_t + offs_k).to(tl.float32)
            g = tl.load(kg + t * s_kg_t + K + offs_k).to(tl.float32)
            kn = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
            u = tl.load(corr + t * s_corr_t + offs_v)
            fc += u[:, None] * (kn * final)[None, :]
            final = final * tl.exp(lower_bound * tl.sigmoid(a * (g + dt)))
    s_ptr = (
        state_ptr + layer * s_state_l + slot * s_state_s + head * s_state_h
        + offs_v[:, None] * K + offs_k[None, :]
    )  # fmt: skip
    s0 = tl.load(s_ptr)
    tl.store(s_ptr, s0 * final[None, :] + fc)


def fold_kda_flow_pending(
    *,
    state: torch.Tensor,  # [L, slots, H, 128, 128] FP32
    corrections: torch.Tensor,  # [L, slots, H, q, 128] FP32
    key_gate: torch.Tensor,  # [L, slots, H, q, 256] BF16
    a_log: torch.Tensor,  # [L, H] FP32
    dt_bias: torch.Tensor,  # [L, H * 128] FP32
    lower_bound: float,
    slots: torch.Tensor,  # [B] slots to fold (<= 0: skipped)
    pending: torch.Tensor,  # [slots + 1] int32
) -> None:
    """Fold the pending records of ``slots`` into ``state`` (every layer), for a
    reader outside the verify; the caller resets the counts."""
    layers, _, heads, v_dim, k_dim = state.shape
    tokens = corrections.shape[-2]
    assert v_dim == k_dim == HEAD_DIM and state.stride()[-2:] == (k_dim, 1)
    slots = slots.reshape(-1).to(torch.int32)
    if slots.numel() == 0:
        return
    bv = 32
    grid = (v_dim // bv, slots.numel(), layers * heads)
    _kda_flow_fold_kernel[grid](
        state, corrections, key_gate, a_log, dt_bias, slots, pending, float(lower_bound),
        state.stride(0), state.stride(1), state.stride(2),
        corrections.stride(0), corrections.stride(1), corrections.stride(2), corrections.stride(3),
        key_gate.stride(0), key_gate.stride(1), key_gate.stride(2), key_gate.stride(3),
        H=heads, T=tokens, K=k_dim, BV=bv,
    )  # fmt: skip


__all__ = [
    "fold_kda_flow_pending",
    "kda_flow_records",
    "kda_flow_verify",
    "prepare_conv_weight",
]
