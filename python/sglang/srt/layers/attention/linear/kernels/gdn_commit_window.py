"""Deferred accepted replay: restore, four ordered appends, cut and publish.

One program owns a complete (layer, request, value head). Tracked prefixes
retain the existing path. Every primitive keeps its typed memory boundary.
"""
import os
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
    INDEX_STRIDE: tl.constexpr, ITERS: tl.constexpr, PREFIX_CUT: tl.constexpr = False, COMPACT_STEP: tl.constexpr = False,
):
    pid=tl.program_id(0)
    layer=tl.program_id(1).to(tl.int64)
    batch, head=pid//HV, pid%HV
    slot=tl.load(slots+batch).to(tl.int64)
    if slot<0:
        return
    row=tl.load(rows+batch*INDEX_STRIDE).to(tl.int64)
    tile: tl.constexpr = 16 if COMPACT_STEP else 32
    ik, iv, ir=tl.arange(0,K), tl.arange(0,V), tl.arange(0,tile)
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
            M_B,G_B,G_B,INDEX_STRIDE,H,HV,K,V,tile,20.0,
            wa,wu,ww,wc,False,0,WRITE_OUTPUT=False,
            LAYER_MIXED=M_L,LAYER_GATE_A=G_L,LAYER_GATE_B=G_L,
            LAYER_LOG=LOG_L,LAYER_BIAS=BIAS_L,LAYER_VBAR=VB_L,
            LAYER_A=WA_L,LAYER_U=WU_L,LAYER_W=WW_L,LAYER_COUNT=WC_L,
            CONDITIONAL_STEP=True,accepted_steps=steps,INPUT_STEP=step,STORAGE_RMAX=32)
        tl.debug_barrier()
        if PREFIX_CUT:
            # The cut happens in this accepted commit, exactly after the
            # eighth consumed input; replay any remaining accepted suffix.
            _factored_expiry_truncate_kernel(
                wu,ww,wc,rows,INDEX_STRIDE,HV,K,V,16,8,16,ITERS,REL_TOL,
                STRIDE_LAYER_U=WU_L,STRIDE_LAYER_W=WW_L,STRIDE_LAYER_COUNT=WC_L,
                STORAGE_RMAX=32)
            tl.debug_barrier()
    if not PREFIX_CUT:
        _factored_expiry_truncate_kernel(
            wu,ww,wc,rows,INDEX_STRIDE,HV,K,V,32,8,16,ITERS,REL_TOL,
            STRIDE_LAYER_U=WU_L,STRIDE_LAYER_W=WW_L,STRIDE_LAYER_COUNT=WC_L,
            DEFERRED_CUT=True)
    tl.debug_barrier()
    tl.store(psa,tl.load(pwa))
    tl.store(psu,tl.load(pwu))
    tl.store(psw,tl.load(pww))
    tl.store(psc,tl.load(pwc))


@triton.jit
def _factored_commit_window_kernel_loop(
    mixed, ga, gb, alog, bias, vbar,
    pa, pu, pw, pc, wa, wu, ww, wc, stale, slots, rows, steps,
    SCALE: tl.constexpr, EPS: tl.constexpr, REL_TOL: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    M_L: tl.constexpr, M_B: tl.constexpr, M_T: tl.constexpr,
    G_L: tl.constexpr, G_B: tl.constexpr, G_T: tl.constexpr,
    LOG_L: tl.constexpr, BIAS_L: tl.constexpr, VB_L: tl.constexpr,
    PA_L: tl.constexpr, PU_L: tl.constexpr, PW_L: tl.constexpr, PC_L: tl.constexpr,
    WA_L: tl.constexpr, WU_L: tl.constexpr, WW_L: tl.constexpr, WC_L: tl.constexpr,
    INDEX_STRIDE: tl.constexpr, ITERS: tl.constexpr, PREFIX_CUT: tl.constexpr = False, COMPACT_STEP: tl.constexpr = False,
):
    pid=tl.program_id(0)
    layer=tl.program_id(1).to(tl.int64)
    batch, head=pid//HV, pid%HV
    slot=tl.load(slots+batch).to(tl.int64)
    if slot<0:
        return
    row=tl.load(rows+batch*INDEX_STRIDE).to(tl.int64)
    tile: tl.constexpr = 16 if COMPACT_STEP else 32
    ik, iv, ir=tl.arange(0,K), tl.arange(0,V), tl.arange(0,tile)
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
    for step in range(4):
        _factored_packed_step_kernel(
            mixed+step*M_T, ga+step*G_T, gb+step*G_T,
            alog,bias,vbar,wa,wu,ww,wc,stale,rows,mixed,SCALE,EPS,
            M_B,G_B,G_B,INDEX_STRIDE,H,HV,K,V,tile,20.0,
            wa,wu,ww,wc,False,0,WRITE_OUTPUT=False,
            LAYER_MIXED=M_L,LAYER_GATE_A=G_L,LAYER_GATE_B=G_L,
            LAYER_LOG=LOG_L,LAYER_BIAS=BIAS_L,LAYER_VBAR=VB_L,
            LAYER_A=WA_L,LAYER_U=WU_L,LAYER_W=WW_L,LAYER_COUNT=WC_L,
            CONDITIONAL_STEP=True,accepted_steps=steps,INPUT_STEP=step,STORAGE_RMAX=32)
        tl.debug_barrier()
        if PREFIX_CUT:
            # The cut happens in this accepted commit, exactly after the
            # eighth consumed input; replay any remaining accepted suffix.
            _factored_expiry_truncate_kernel(
                wu,ww,wc,rows,INDEX_STRIDE,HV,K,V,16,8,16,ITERS,REL_TOL,
                STRIDE_LAYER_U=WU_L,STRIDE_LAYER_W=WW_L,STRIDE_LAYER_COUNT=WC_L,
                STORAGE_RMAX=32)
            tl.debug_barrier()
    if not PREFIX_CUT:
        _factored_expiry_truncate_kernel(
            wu,ww,wc,rows,INDEX_STRIDE,HV,K,V,32,8,16,ITERS,REL_TOL,
            STRIDE_LAYER_U=WU_L,STRIDE_LAYER_W=WW_L,STRIDE_LAYER_COUNT=WC_L,
            DEFERRED_CUT=True)
    tl.debug_barrier()
    tl.store(psa,tl.load(pwa))
    tl.store(psu,tl.load(pwu))
    tl.store(psw,tl.load(pww))
    tl.store(psc,tl.load(pwc))


@triton.jit
def _factored_packed_step_kernel_commit_segment(
    mixed, ga, gb, alog, bias, vbar,
    pa, pu, pw, pc, wa, wu, ww, wc, stale, slots, rows, steps,
    SCALE: tl.constexpr, EPS: tl.constexpr, REL_TOL: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    M_L: tl.constexpr, M_B: tl.constexpr, M_T: tl.constexpr,
    G_L: tl.constexpr, G_B: tl.constexpr, G_T: tl.constexpr,
    LOG_L: tl.constexpr, BIAS_L: tl.constexpr, VB_L: tl.constexpr,
    PA_L: tl.constexpr, PU_L: tl.constexpr, PW_L: tl.constexpr, PC_L: tl.constexpr,
    WA_L: tl.constexpr, WU_L: tl.constexpr, WW_L: tl.constexpr, WC_L: tl.constexpr,
    INDEX_STRIDE: tl.constexpr, ITERS: tl.constexpr, PREFIX_CUT: tl.constexpr = False, COMPACT_STEP: tl.constexpr = False, SEGMENT: tl.constexpr = 0,
):
    pid=tl.program_id(0)
    layer=tl.program_id(1).to(tl.int64)
    batch, head=pid//HV, pid%HV
    slot=tl.load(slots+batch).to(tl.int64)
    if slot<0:
        return
    row=tl.load(rows+batch*INDEX_STRIDE).to(tl.int64)
    tile: tl.constexpr = 16 if COMPACT_STEP else 32
    ik, iv, ir=tl.arange(0,K), tl.arange(0,V), tl.arange(0,tile)
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
    # Persistent count stays unchanged until segment 1 publishes. Thus both
    # launches derive the same original accepted-prefix boundary.
    boundary = 16 - tl.load(psc)
    consumed = tl.load(steps+batch) + 1
    if SEGMENT == 0:
        tl.store(pwa,tl.load(psa))
        tl.store(pwu,tl.load(psu))
        tl.store(pww,tl.load(psw))
        tl.store(pwc,tl.load(psc))
        first, last = 0, tl.minimum(consumed, boundary)
    else:
        first, last = boundary, consumed
    tl.debug_barrier()
    for step in range(first, last):
        _factored_packed_step_kernel(
            mixed+step*M_T, ga+step*G_T, gb+step*G_T,
            alog,bias,vbar,wa,wu,ww,wc,stale,rows,mixed,SCALE,EPS,
            M_B,G_B,G_B,INDEX_STRIDE,H,HV,K,V,tile,20.0,
            wa,wu,ww,wc,False,0,WRITE_OUTPUT=False,
            LAYER_MIXED=M_L,LAYER_GATE_A=G_L,LAYER_GATE_B=G_L,
            LAYER_LOG=LOG_L,LAYER_BIAS=BIAS_L,LAYER_VBAR=VB_L,
            LAYER_A=WA_L,LAYER_U=WU_L,LAYER_W=WW_L,LAYER_COUNT=WC_L,
            CONDITIONAL_STEP=False,STORAGE_RMAX=32)
        tl.debug_barrier()
    if SEGMENT == 1:
        # The caller has run the original truncation once between segments.
        # A prefix of at most four inputs can cross at most one W8 boundary.
        tl.debug_barrier()
        tl.store(psa,tl.load(pwa))
        tl.store(psu,tl.load(pwu))
        tl.store(psw,tl.load(pww))
        tl.store(psc,tl.load(pwc))


def factored_commit_window(pool,working,inputs,constants,stale,slots,rows,steps,arguments,*,prefix_cut=False):
    compact_step = os.environ.get('SGLANG_GDN_VERIFY_COMMIT_COMPACT', '0') == '1'
    loop = os.environ.get('SGLANG_GDN_VERIFY_COMMIT_LOOP', '0') == '1'
    split_cut = os.environ.get('SGLANG_GDN_VERIFY_COMMIT_SPLIT_CUT', '0') == '1'
    cut_warps = int(os.environ.get('SGLANG_GDN_VERIFY_COMMIT_CUT_WARPS', '1'))
    if cut_warps not in (1, 2, 4, 8):
        raise ValueError('split compression warps must be 1, 2, 4 or 8')
    if split_cut and not (loop and compact_step and prefix_cut):
        raise ValueError('split cut requires looped compact prefix replay')
    if loop and not compact_step:
        raise ValueError('looped commit is confined to compact prefix replay')
    if compact_step and not prefix_cut:
        raise ValueError('compact accepted step requires prefix cuts')
    warps=1 if prefix_cut else 4
    if pool.U.shape[-2]!=32 or _step_warps(32)!=warps:
        raise ValueError('fused commit requires capacity 32 and a matching append/cut warp layout')
    if (arguments.get('trunc_warps') or warps)!=warps:
        raise ValueError('fused commit preserves the original capacity-32 compression warp layout')
    mixed,ga,gb=(inputs[name] for name in ('mixed','a','b'))
    if ga.stride()!=gb.stride():
        raise ValueError('replay gate layouts differ')
    wa,wu,ww,wc=(working[name] for name in ('a','U','W','count'))
    alog,bias=constants['A_log'],constants['dt_bias']
    layers,_,hv,_,k=pool.U.shape
    selected = _factored_commit_window_kernel_loop if loop else _factored_commit_window_kernel
    launch_args = (
        mixed,ga,gb,alog,bias,pool.vbar,pool.a,pool.U,pool.W,pool.count,
        wa,wu,ww,wc,stale,slots,rows,steps,
        arguments['scale'],GS_EPS,MGS_REL_TOL,arguments['num_q_heads'],hv,k,pool.W.shape[-1],
        *mixed.stride()[:3],*ga.stride()[:3],alog.stride(0),bias.stride(0),pool.vbar.stride(0),
        pool.a.stride(0),pool.U.stride(0),pool.W.stride(0),pool.count.stride(0),
        wa.stride(0),wu.stride(0),ww.stride(0),wc.stride(0),rows.stride(0),
        arguments.get('trunc_iters') or TRUNC_ITERS)
    grid = (slots.numel()*hv,layers)
    segments = []
    if split_cut:
        segments.append(_factored_packed_step_kernel_commit_segment[grid](
            *launch_args,PREFIX_CUT=True,COMPACT_STEP=True,SEGMENT=0,num_warps=1))
        segments.append(_factored_expiry_truncate_kernel[grid](
            wu,ww,wc,rows,rows.stride(0),hv,k,pool.W.shape[-1],16,8,16,
            arguments.get('trunc_iters') or TRUNC_ITERS,MGS_REL_TOL,
            STRIDE_LAYER_U=wu.stride(0),STRIDE_LAYER_W=ww.stride(0),
            STRIDE_LAYER_COUNT=wc.stride(0),STORAGE_RMAX=32,num_warps=cut_warps))
        segments.append(_factored_packed_step_kernel_commit_segment[grid](
            *launch_args,PREFIX_CUT=True,COMPACT_STEP=True,SEGMENT=1,num_warps=1))
        compiled = segments[-1]
    else:
        compiled = selected[grid](*launch_args,PREFIX_CUT=prefix_cut,
            COMPACT_STEP=compact_step,num_warps=warps)

    if os.environ.get('SGLANG_GDN_VERIFY_DIAGNOSTICS') == '1' and compiled is not None:
        global COMMIT_LAST_RESOURCES
        COMMIT_LAST_RESOURCES = dict(registers=getattr(compiled,'n_regs',None),
            spills=getattr(compiled,'n_spills',None),shared=getattr(compiled.metadata,'shared',None),
            warps=warps,prefix_cut=prefix_cut,compact_step=compact_step,loop=loop,
            split_cut=split_cut,cut_warps=cut_warps,segments=[dict(registers=getattr(c,'n_regs',None),
                spills=getattr(c,'n_spills',None),shared=getattr(c.metadata,'shared',None))
                for c in segments if c is not None])
