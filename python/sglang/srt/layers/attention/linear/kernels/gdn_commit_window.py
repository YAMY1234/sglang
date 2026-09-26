"""Deferred accepted replay: restore, four ordered appends, cut and publish.

One program owns a complete (layer, request, value head). Tracked prefixes
retain the existing path. Every primitive keeps its typed memory boundary.
"""
import triton
import triton.language as tl

from .gdn_factored import (
    GS_EPS, MGS_REL_TOL, TRUNC_ITERS, _step_warps,
    _factored_packed_step_kernel, _factored_expiry_truncate_kernel,
)


@triton.jit
def _factored_commit_window_kernel(
    mixed, ga, gb, alog, bias, vbar,
    pa, pu, pw, pc, wa, wu, ww, wc, stale, slots, rows, steps,
    SCALE: tl.constexpr, EPS: tl.constexpr, REL_TOL: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    M_L: tl.constexpr, M_B: tl.constexpr, M_T: tl.constexpr,
    G_L: tl.constexpr, G_B: tl.constexpr, G_T: tl.constexpr,
    LOG_L: tl.constexpr, BIAS_L: tl.constexpr, VB_L: tl.constexpr,
    PA_L: tl.constexpr, PU_L: tl.constexpr, PW_L: tl.constexpr, PC_L: tl.constexpr,
    WA_L: tl.constexpr, WU_L: tl.constexpr, WW_L: tl.constexpr, WC_L: tl.constexpr,
    INDEX_STRIDE: tl.constexpr, ITERS: tl.constexpr,
):
    pid=tl.program_id(0)
    layer=tl.program_id(1).to(tl.int64)
    batch, head=pid//HV, pid%HV
    slot=tl.load(slots+batch).to(tl.int64)
    if slot<0:
        return
    row=tl.load(rows+batch*INDEX_STRIDE).to(tl.int64)
    ik, iv, ir=tl.arange(0,K), tl.arange(0,V), tl.arange(0,32)
    source=slot*HV+head
    target=row*HV+head
    psa=pa+layer*PA_L+source*K+ik
    psu=pu+layer*PU_L+source*32*K+ir[:,None]*K+ik[None,:]
    psw=pw+layer*PW_L+source*32*V+ir[:,None]*V+iv[None,:]
    psc=pc+layer*PC_L+source
    pwa=wa+layer*WA_L+target*K+ik
    pwu=wu+layer*WU_L+target*32*K+ir[:,None]*K+ik[None,:]
    pww=ww+layer*WW_L+target*32*V+ir[:,None]*V+iv[None,:]
    pwc=wc+layer*WC_L+target
    tl.store(pwa,tl.load(psa))
    tl.store(pwu,tl.load(psu))
    tl.store(pww,tl.load(psw))
    tl.store(pwc,tl.load(psc))
    tl.debug_barrier()
    for step in tl.static_range(4):
        _factored_packed_step_kernel(
            mixed+step*M_T, ga+step*G_T, gb+step*G_T,
            alog,bias,vbar,wa,wu,ww,wc,stale,rows,mixed,SCALE,EPS,
            M_B,G_B,G_B,INDEX_STRIDE,H,HV,K,V,32,20.0,
            wa,wu,ww,wc,False,0,WRITE_OUTPUT=False,
            LAYER_MIXED=M_L,LAYER_GATE_A=G_L,LAYER_GATE_B=G_L,
            LAYER_LOG=LOG_L,LAYER_BIAS=BIAS_L,LAYER_VBAR=VB_L,
            LAYER_A=WA_L,LAYER_U=WU_L,LAYER_W=WW_L,LAYER_COUNT=WC_L,
            CONDITIONAL_STEP=True,accepted_steps=steps,INPUT_STEP=step)
        tl.debug_barrier()
    _factored_expiry_truncate_kernel(
        wu,ww,wc,rows,INDEX_STRIDE,HV,K,V,32,8,16,ITERS,REL_TOL,
        STRIDE_LAYER_U=WU_L,STRIDE_LAYER_W=WW_L,STRIDE_LAYER_COUNT=WC_L,
        DEFERRED_CUT=True)
    tl.debug_barrier()
    tl.store(psa,tl.load(pwa))
    tl.store(psu,tl.load(pwu))
    tl.store(psw,tl.load(pww))
    tl.store(psc,tl.load(pwc))


def factored_commit_window(pool,working,inputs,constants,stale,slots,rows,steps,arguments):
    if pool.U.shape[-2]!=32 or _step_warps(32)!=4:
        raise ValueError('fused commit requires deferred capacity 32 and consistent four-warp append')
    if (arguments.get('trunc_warps') or 4)!=4:
        raise ValueError('fused commit preserves the original capacity-32 compression warp layout')
    mixed,ga,gb=(inputs[name] for name in ('mixed','a','b'))
    if ga.stride()!=gb.stride():
        raise ValueError('replay gate layouts differ')
    wa,wu,ww,wc=(working[name] for name in ('a','U','W','count'))
    alog,bias=constants['A_log'],constants['dt_bias']
    layers,_,hv,_,k=pool.U.shape
    _factored_commit_window_kernel[(slots.numel()*hv,layers)](
        mixed,ga,gb,alog,bias,pool.vbar,pool.a,pool.U,pool.W,pool.count,
        wa,wu,ww,wc,stale,slots,rows,steps,
        arguments['scale'],GS_EPS,MGS_REL_TOL,arguments['num_q_heads'],hv,k,pool.W.shape[-1],
        *mixed.stride()[:3],*ga.stride()[:3],alog.stride(0),bias.stride(0),pool.vbar.stride(0),
        pool.a.stride(0),pool.U.stride(0),pool.W.stride(0),pool.count.stride(0),
        wa.stride(0),wu.stride(0),ww.stride(0),wc.stride(0),rows.stride(0),
        arguments.get('trunc_iters') or TRUNC_ITERS,num_warps=4)
