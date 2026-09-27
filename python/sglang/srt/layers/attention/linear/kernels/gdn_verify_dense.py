"""Directive 828: temporary FP32 stock recurrence, unchanged factor commit.

The dense tensor is request scratch. It is reconstructed each verification
window and is never published to the persistent r8/W8 pool.
"""
import os
import torch
import triton
import triton.language as tl
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update_kernel as _stock_recurrent_kernel,
)


@triton.jit(do_not_specialize=['T'])
def _stock_dense_verify_record_kernel(
    Q, KIN, VINPUT, A, BETA, ALOG, BIAS, INITIAL, INDICES, OUT,
    RM, RA, RB, WRITTEN, SCALE, T,
    BATCH: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    QS: tl.constexpr, KS: tl.constexpr, VS: tl.constexpr,
    AS: tl.constexpr, BS: tl.constexpr, INITIAL_STRIDE: tl.constexpr,
    RMS0: tl.constexpr, RMS1: tl.constexpr,
    RAS0: tl.constexpr, RAS1: tl.constexpr, WS0: tl.constexpr,
    SPLIT: tl.constexpr, USE_GDC: tl.constexpr = False,
):
    # Inline the stock implementation verbatim. The independent admission
    # compares against its ordinary wrapper, including each captured replay.
    _stock_recurrent_kernel(
        A_log=ALOG, a=A, dt_bias=BIAS, softplus_beta=1.0,
        softplus_threshold=20.0, lower_bound=0.0,
        q=Q, k=KIN, v=VINPUT, b=BETA, o=OUT,
        h0_source=INITIAL, h0_indices=INDICES, stride_h0_source=INITIAL_STRIDE,
        cu_seqlens=None, intermediate_states_buffer=None,
        intermediate_state_indices=None, cache_steps=0, retrieve_parent_token_ptr=None,
        stride_retrieve_parent_token_seq=0, stride_retrieve_parent_token_token=0,
        scale=SCALE, T=T, stride_a=AS, stride_q=QS, stride_k=KS,
        stride_v=VS, stride_b=BS, NP2_T=4, B=BATCH, H=H, HV=HV,
        K=K, V=V, BK=BK, BV=BV, USE_INITIAL_STATE=True,
        USE_QK_L2NORM_IN_KERNEL=True, IS_VARLEN=False, IS_KDA=False,
        USE_LOWER_BOUND=False, DISABLE_STATE_UPDATE=True,
        ROUND_BETA_TO_INPUT_DTYPE=True, SPLIT_N_HV_GRID=SPLIT, USE_GDC=USE_GDC,
    )
    if SPLIT:
        iv, row, hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    else:
        iv = tl.program_id(1)
        row, hv = tl.program_id(2) // HV, tl.program_id(2) % HV
    h = hv // (HV // H)
    kk = tl.arange(0, BK)
    vv = iv*BV + tl.arange(0, BV)
    for step in range(0, T):
        token = row*T + step
        if iv == 0:
            if hv % (HV // H) == 0:
                q = tl.load(Q + token*QS + h*K + kk, kk < K, 0)
                k = tl.load(KIN + token*KS + h*K + kk, kk < K, 0)
                tl.store(RM + row*RMS0 + step*RMS1 + h*K + kk, q, kk < K)
                tl.store(RM + row*RMS0 + step*RMS1 + H*K + h*K + kk, k, kk < K)
            ga = tl.load(A + token*AS + hv)
            gb = tl.load(BETA + token*BS + hv)
            tl.store(RA + row*RAS0 + step*RAS1 + hv, ga)
            tl.store(RB + row*RAS0 + step*RAS1 + hv, gb)
            if hv == 0:
                tl.store(WRITTEN + row*WS0 + step, 1)
        v = tl.load(VINPUT + token*VS + hv*V + vv, vv < V, 0)
        tl.store(RM + row*RMS0 + step*RMS1 + 2*H*K + hv*V + vv, v, vv < V)


@triton.jit
def _dense_verify_record_kernel(M, A, B, RM, RA, RB, WRITTEN,
                                WIDTH: tl.constexpr, HEADS: tl.constexpr,
                                MS0: tl.constexpr, MS1: tl.constexpr,
                                AS0: tl.constexpr, AS1: tl.constexpr,
                                BS0: tl.constexpr, BS1: tl.constexpr,
                                RMS0: tl.constexpr, RMS1: tl.constexpr,
                                RAS0: tl.constexpr, RAS1: tl.constexpr,
                                WS0: tl.constexpr, BLOCK: tl.constexpr):
    row, step = tl.program_id(0), tl.program_id(1)
    x = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    m = tl.load(M + row*MS0 + step*MS1 + x, x < WIDTH, 0)
    a = tl.load(A + row*AS0 + step*AS1 + x, x < HEADS, 0)
    b = tl.load(B + row*BS0 + step*BS1 + x, x < HEADS, 0)
    tl.store(RM + row*RMS0 + step*RMS1 + x, m, x < WIDTH)
    tl.store(RA + row*RAS0 + step*RAS1 + x, a, x < HEADS)
    tl.store(RB + row*RAS0 + step*RAS1 + x, b, x < HEADS)
    if tl.program_id(2) == 0:
        tl.store(WRITTEN + row*WS0 + step, 1)


def restore_dense_layers(working, vbar, output, batch):
    # Keep the original FP32 densify expression. Independent layer/head
    # matrices are batched together; no persistent factor is changed.
    layers, _, heads, rank, key = working['U'].shape
    value = working['W'].shape[-1]
    def flatten(name):
        t = working[name][:, :batch]
        return t.transpose(0, 1).reshape(batch, layers*heads, *t.shape[3:])
    a, u, w, count = (flatten(n) for n in ('a', 'U', 'W', 'count'))
    rows = torch.arange(rank, device=u.device)[None, None, :] < count[:, :, None]
    uf = u.float() * rows[..., None]
    wf = w.float() * rows[..., None]
    dense = torch.einsum('bhrv,bhrk->bhvk', wf, uf)
    dense = dense + vbar.float().reshape(layers*heads, value)[None, :, :, None] * a.float()[:, :, None, :]
    output[:, :batch].copy_(dense.reshape(batch, layers, heads, value, key).transpose(0, 1))


def stock_dense_verify(mixed, a, b, initial, indices, arguments, recording=None):
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    batch, tokens, width = mixed.shape
    h, hv = arguments['num_q_heads'], arguments['num_v_heads']
    k, v = arguments['head_k_dim'], arguments['head_v_dim']
    if tokens != 4 or width != 2*h*k + hv*v:
        raise ValueError('dense verification requires packed four-input NEXTN chain')
    fused_record = os.environ.get('SGLANG_GDN_VERIFY_DENSE_RECORD_FUSED', '0') == '1'
    if fused_record and recording is None:
        raise ValueError('fused stock recording requires owned input buffers')
    if recording is not None and not fused_record:
        rm, ra, rb, rw = (recording[n] for n in ('mixed', 'a', 'b', 'written'))
        if ra.stride() != rb.stride():
            raise ValueError('recorded gates must have identical strides')
        _dense_verify_record_kernel[(batch, tokens, triton.cdiv(width, 256))](
            mixed, a, b, rm, ra, rb, rw, width, hv,
            *mixed.stride()[:2], *a.stride()[:2], *b.stride()[:2],
            *rm.stride()[:2], *ra.stride()[:2], rw.stride(0), 256,
            num_warps=4,
        )
    # Strided packed Q/K/V views are accepted by the stock wrapper. Preserve
    # its normalization, BF16 beta rounding and FP32 recurrence verbatim.
    q = mixed[..., :h*k].reshape(batch, tokens, h, k)
    key = mixed[..., h*k:2*h*k].reshape(batch, tokens, h, k)
    value = mixed[..., 2*h*k:].reshape(batch, tokens, hv, v)
    if fused_record:
        from sglang.kernels.jit.utils import is_arch_support_pdl
        split = mixed.is_cuda
        bv, bk = min(triton.next_power_of_2(v), 32), triton.next_power_of_2(k)
        grid = (triton.cdiv(v, bv), batch, hv) if split else (1, triton.cdiv(v, bv), batch*hv)
        pdl = dict(USE_GDC=True, launch_pdl=True) if split and is_arch_support_pdl() else {}
        rm, ra, rb, rw = (recording[n] for n in ('mixed', 'a', 'b', 'written'))
        output = mixed.new_empty(batch, tokens, hv, v)
        compiled = _stock_dense_verify_record_kernel[grid](
            q, key, value, a, b, arguments['A_log'], arguments['dt_bias'], initial, indices,
            output, rm, ra, rb, rw, arguments['scale'], tokens,
            batch, h, hv, k, v, bk, bv, q.stride(1), key.stride(1), value.stride(1),
            a.stride(1), b.stride(1), initial.stride(0),
            *rm.stride()[:2], *ra.stride()[:2], rw.stride(0), split,
            num_warps=1, num_stages=3, **pdl,
        )
        if os.environ.get('SGLANG_GDN_VERIFY_DIAGNOSTICS') == '1' and compiled is not None:
            global DENSE_VERIFY_LAST_RESOURCES
            DENSE_VERIFY_LAST_RESOURCES = dict(registers=compiled.n_regs, spills=compiled.n_spills,
                shared=compiled.metadata.shared, warps=1, stock_recurrence=True, record_fused=True)
        return output
    return fused_sigmoid_gating_delta_rule_update(
        A_log=arguments['A_log'], a=a, dt_bias=arguments['dt_bias'],
        softplus_beta=1.0, softplus_threshold=20.0,
        q=q, k=key, v=value, b=b, initial_state_source=initial,
        initial_state_indices=indices, scale=arguments['scale'],
        use_qk_l2norm_in_kernel=True, disable_state_update=True,
        round_beta_to_input_dtype=True,
    )
