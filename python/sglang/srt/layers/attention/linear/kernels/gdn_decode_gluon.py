"""Decode candidate with explicit frozen reductions and distributed factor rows.

The one-warp layouts match the admitted packed-step sm103 TTGIR. Extra warps
share factor rows, but every reduction over rows returns to the original
layout before summation. CUDA bitwise admission is required for every variant.
"""
import triton.experimental.gluon as g
import triton.experimental.gluon.language as l
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


@g.jit
def _rows(matrix, vector, WARPS:l.constexpr):
    # Convert operands before multiplying so the frozen local FMA chain
    # survives. Converting products rounded them before the reduction.
    original:l.constexpr=BlockedLayout([1,8],[2,16],[1,WARPS],[1,0])
    return l.sum(l.convert_layout(matrix,original) *
                 l.convert_layout(vector,SliceLayout(1,original))[:,None],axis=0)


@g.jit
def packed_step(mixed, gate_a, gate_b, A_log, dt_bias, vbar, fa, fu, fw,
                count, stale, indices, output, prefix, scale, gs_eps,
                MIXED_ROW:l.constexpr, A_ROW:l.constexpr, B_ROW:l.constexpr,
                INDEX_STRIDE:l.constexpr, OUTPUT_ROW:l.constexpr,
                H:l.constexpr, HV:l.constexpr, WARPS:l.constexpr,
                INVALIDATE:l.constexpr):
    S:l.constexpr=BlockedLayout([1,8],[2,16],[WARPS,1],[1,0])
    O:l.constexpr=BlockedLayout([1,8],[2,16],[1,WARPS],[1,0])
    Q:l.constexpr=BlockedLayout([4],[32],[WARPS],[0])
    pid=l.program_id(0);row=pid//HV;head=pid%HV;qh=head//(HV//H)
    x=l.arange(0,128,layout=Q)
    sx=l.arange(0,128,layout=SliceLayout(0,S))
    sr=l.arange(0,16,layout=SliceLayout(1,S))
    slot=l.load(indices+row*INDEX_STRIDE).to(l.int64)
    if INVALIDATE:
        if head==0:l.store(prefix+l.maximum(slot,0),0)
    op=output+row*OUTPUT_ROW+head*128+x
    if slot<0:
        l.store(op,0)
        return
    base=slot*HV+head
    mp=mixed+row*MIXED_ROW
    q=l.load(mp+qh*128+x).to(l.float32)
    k=l.load(mp+H*128+qh*128+x).to(l.float32)
    v=l.load(mp+2*H*128+head*128+x).to(l.float32)
    av=l.load(gate_a+row*A_ROW+head).to(l.float32)
    bv=l.load(gate_b+row*B_ROW+head).to(l.float32)
    log=l.load(A_log+head).to(l.float32);bias=l.load(dt_bias+head).to(l.float32)
    z=av+bias
    soft=l.where(z<=20.0,l.log(1.0+l.exp(z)),z)
    gv=-l.exp(log)*soft
    beta=(1/(1+l.exp(-bv))).to(gate_b.dtype.element_ty).to(l.float32)
    gt=l.exp(gv)
    qn=q/l.sqrt(l.sum(q*q,0)+1e-6)*scale
    kn=k/l.sqrt(l.sum(k*k,0)+1e-6)
    vb=l.load(vbar+head*128+x).to(l.float32)
    a=l.load(fa+base*128+x)
    an=gt*(a-beta*kn*l.sum(kn*a,axis=0))+beta*kn
    l.store(fa+base*128+x,an)
    out=vb*l.sum(an*qn,axis=0)
    cnt=l.load(count+base)
    up=fu+base*16*128+sr[:,None]*128+sx[None,:]
    wp=fw+base*16*128+sr[:,None]*128+sx[None,:]
    U=l.load(up,mask=(sr<cnt)[:,None],other=0.0).to(l.float32)
    W=l.load(wp,mask=(sr<cnt)[:,None],other=0.0).to(l.float32)
    qn=l.convert_layout(qn,SliceLayout(0,S));kn=l.convert_layout(kn,SliceLayout(0,S))
    c=l.sum(U*kn[None,:],axis=1)
    kp=kn-l.convert_layout(_rows(U,c,WARPS),SliceLayout(0,S))
    nrm2=l.sum(kp*kp,axis=0)
    if nrm2<0.25:
        c2=l.sum(U*kp[None,:],axis=1)
        kp=kp-l.convert_layout(_rows(U,c2,WARPS),SliceLayout(0,S))
        c=c+c2;nrm2=l.sum(kp*kp,axis=0)
    nrm=l.sqrt(nrm2);keep=nrm>gs_eps
    khat=l.where(keep,kp/l.maximum(nrm,gs_eps),0.0)
    clast=l.where(keep,nrm,0.0)
    mvec=l.convert_layout(_rows(W,c,WARPS),SliceLayout(0,S))
    delta=beta*(l.convert_layout(v-vb,SliceLayout(0,S))-gt*mvec)
    is_new=sr==cnt;cfull=l.where(is_new,clast,c)
    cq=l.sum(U*qn[None,:],axis=1)+l.where(is_new,l.sum(khat*qn,axis=0),0.0)
    cc=l.convert_layout(cfull,SliceLayout(1,O))*l.convert_layout(cq,SliceLayout(1,O))
    out=l.convert_layout(out,SliceLayout(0,S))+gt*l.convert_layout(_rows(W,cq,WARPS),SliceLayout(0,S))+delta*l.sum(cc,axis=0)
    l.store(wp,(gt*W+cfull[:,None]*delta[None,:]).to(fw.dtype.element_ty),mask=(sr<=cnt)[:,None])
    l.store(fu+base*16*128+cnt*128+x,l.convert_layout(khat,Q).to(fu.dtype.element_ty),mask=x<128*(cnt<16))
    l.store(count+base,cnt+1);l.store(stale+slot,1)
    l.store(op,l.convert_layout(out,Q).to(output.dtype.element_ty))
