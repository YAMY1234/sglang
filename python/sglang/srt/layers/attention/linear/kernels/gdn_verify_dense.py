"""Directive 828: temporary FP32 stock recurrence, unchanged factor commit.

The dense tensor is request scratch. It is reconstructed each verification
window and is never published to the persistent r8/W8 pool.
"""
import torch
import triton
import triton.language as tl


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
    if recording is not None:
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
    return fused_sigmoid_gating_delta_rule_update(
        A_log=arguments['A_log'], a=a, dt_bias=arguments['dt_bias'],
        softplus_beta=1.0, softplus_threshold=20.0,
        q=q, k=key, v=value, b=b, initial_state_source=initial,
        initial_state_indices=indices, scale=arguments['scale'],
        use_qk_l2norm_in_kernel=True, disable_state_update=True,
        round_beta_to_input_dtype=True,
    )
