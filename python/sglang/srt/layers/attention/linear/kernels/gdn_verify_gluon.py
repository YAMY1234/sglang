"""Experimental register residency with frozen one-warp reduction layouts.

Layouts are copied from the admitted sm100 r8/K128 step and expiry TTGIR.
Changing the layouts can change summation order, hence requires bitwise gates.
"""
import triton.experimental.gluon as g
import triton.experimental.gluon.language as l
from triton.experimental.gluon.language import BlockedLayout, SliceLayout, DotOperandLayout


@g.jit
def _dot(a,b,LAYOUT:l.constexpr):
    aa=l.convert_layout(a,DotOperandLayout(0,LAYOUT,0))
    bb=l.convert_layout(b,DotOperandLayout(1,LAYOUT,0))
    return l.dot_fma(aa,bb,l.full((a.shape[0],b.shape[1]),0,l.float32,LAYOUT))


@g.jit
def _mgs(Q,R:l.constexpr,REL_TOL:l.constexpr,GATHER:l.constexpr):
    C:l.constexpr=BlockedLayout([2,2],[4,8],[1,1],[1,0])
    cols=l.arange(0,16,layout=SliceLayout(0,C))
    n0=l.sqrt(l.sum(Q*Q,axis=0))
    for p in l.static_range(2):
        for j in l.static_range(R):
            colj=cols==j
            if GATHER:
                y=l.gather(Q,l.full((16,1),j,l.int32,C),axis=1).reshape((16,))
            else:
                y=l.sum(l.where(colj[None,:],Q,0.0),axis=1)
            proj=l.where(cols<j,l.sum(Q*y[:,None],axis=0),0.0)
            y=y-l.sum(Q*proj[None,:],axis=1)
            n=l.sqrt(l.sum(y*y,axis=0))
            n0j=l.sum(l.where(colj,n0,0.0),axis=0)
            ok=n>1e-12
            if p==0:ok=ok&(n>REL_TOL*n0j)
            y=l.where(ok,y/l.maximum(n,1e-30),0.0)
            Q=l.where(colj[None,:],y[:,None],Q)
    return Q


@g.jit
def _cut(U,W,R:l.constexpr,ITERS:l.constexpr,REL_TOL:l.constexpr,GATHER:l.constexpr):
    C:l.constexpr=BlockedLayout([2,2],[4,8],[1,1],[1,0])
    P:l.constexpr=BlockedLayout([4,4],[1,32],[1,1],[1,0])
    S:l.constexpr=BlockedLayout([1,8],[2,16],[1,1],[1,0])
    rows=l.arange(0,16,layout=SliceLayout(1,C))
    cols=l.arange(0,16,layout=SliceLayout(0,C))
    G=_dot(W.to(l.float32),l.trans(W.to(l.float32)),C)
    d=l.sum(l.where(rows[:,None]==cols[None,:],G,0.0),axis=1)
    dc=l.convert_layout(d,SliceLayout(0,C))
    better=(dc[None,:]>d[:,None])|((dc[None,:]==d[:,None])&(cols[None,:]<rows[:,None]))
    rank=l.sum(better.to(l.int32),axis=1)
    Z=l.where((rank[:,None]==cols[None,:])&(cols[None,:]<R),1.0,0.0)
    for _ in range(ITERS):
        Z=_dot(G,Z,C)
        Z=_mgs(Z,R,REL_TOL,GATHER)
    Un=_dot(l.trans(Z),U.to(l.float32),P).to(U.dtype)
    Wn=_dot(l.trans(Z),W.to(l.float32),P).to(W.dtype)
    sr=l.arange(0,16,layout=SliceLayout(1,S))
    return (l.where((sr<R)[:,None],l.convert_layout(Un,S),U),
            l.where((sr<R)[:,None],l.convert_layout(Wn,S),W))


@g.jit
def verify(
    mixed,gate_a,gate_b,A_log,dt_bias,vbar,fa,fu,fw,count,stale,indices,output,scale,gs_eps,
    MIXED_ROW:l.constexpr,MIXED_STEP:l.constexpr,A_ROW:l.constexpr,A_STEP:l.constexpr,
    B_ROW:l.constexpr,B_STEP:l.constexpr,INDEX_STRIDE:l.constexpr,
    H:l.constexpr,HV:l.constexpr,K:l.constexpr,V:l.constexpr,RMAX:l.constexpr,R:l.constexpr,
    RFULL:l.constexpr,ITERS:l.constexpr,REL_TOL:l.constexpr,TOKENS:l.constexpr,
    BATCH:l.constexpr,GATHER:l.constexpr,HEAD_MAJOR:l.constexpr,
):
    l.static_assert(K==128 and V==128 and RMAX==16 and RFULL==16)
    S:l.constexpr=BlockedLayout([1,8],[2,16],[1,1],[1,0])
    Q:l.constexpr=BlockedLayout([4],[32],[1],[0])
    N:l.constexpr=BlockedLayout([1],[32],[1],[0])
    pid=l.program_id(0)
    if HEAD_MAJOR:i_n,i_hv=pid%BATCH,pid//BATCH
    else:i_n,i_hv=pid//HV,pid%HV
    i_h=i_hv//(HV//H)
    x=l.arange(0,128,layout=Q)
    sx=l.arange(0,128,layout=SliceLayout(0,S))
    sr=l.arange(0,16,layout=SliceLayout(1,S))
    slot=l.load(indices+i_n*INDEX_STRIDE).to(l.int64)
    if slot<0:
        for step in range(TOKENS):l.store(output+(i_n*TOKENS+step)*HV*V+i_hv*V+x,0)
        return
    base=slot*HV+i_hv
    up=fu+base*16*128+sr[:,None]*128+sx[None,:]
    wp=fw+base*16*128+sr[:,None]*128+sx[None,:]
    U_all,W_all=l.load(up),l.load(wp)
    a=l.load(fa+base*128+x);cnt=l.load(count+base)
    log=l.load(A_log+i_hv).to(l.float32);bias=l.load(dt_bias+i_hv).to(l.float32)
    vb=l.load(vbar+i_hv*128+x).to(l.float32)
    for step in range(TOKENS):
        mp=mixed+i_n*MIXED_ROW+step*MIXED_STEP
        q=l.load(mp+i_h*128+x).to(l.float32)
        k=l.load(mp+H*128+i_h*128+x).to(l.float32)
        v=l.load(mp+2*H*128+i_hv*128+x).to(l.float32)
        av=l.load(gate_a+i_n*A_ROW+step*A_STEP+i_hv).to(l.float32)
        bv=l.load(gate_b+i_n*B_ROW+step*B_STEP+i_hv).to(l.float32)
        z=av+bias
        soft=l.where(z<=20.0,l.log(1.0+l.exp(z)),z)
        gv=-l.exp(log)*soft
        beta=l.sigmoid(bv).to(gate_b.dtype.element_ty).to(l.float32)
        gt=l.exp(gv)
        qn=q/l.sqrt(l.sum(l.convert_layout(q*q,N),0)+1e-6)*scale
        kn=k/l.sqrt(l.sum(l.convert_layout(k*k,N),0)+1e-6)
        a_new=gt*(a-beta*kn*l.sum(kn*a,axis=0))+beta*kn
        out=vb*l.sum(a_new*qn,axis=0)
        qn=l.convert_layout(qn,SliceLayout(0,S));kn=l.convert_layout(kn,SliceLayout(0,S))
        U=l.where((sr<cnt)[:,None],U_all,0.0).to(l.float32)
        W=l.where((sr<cnt)[:,None],W_all,0.0).to(l.float32)
        c=l.sum(U*kn[None,:],axis=1)
        kp=kn-l.sum(U*c[:,None],axis=0)
        nrm2=l.sum(kp*kp,axis=0)
        if nrm2<0.25:
            c2=l.sum(U*kp[None,:],axis=1)
            kp=kp-l.sum(U*c2[:,None],axis=0)
            c=c+c2;nrm2=l.sum(kp*kp,axis=0)
        nrm=l.sqrt(nrm2);keep=nrm>gs_eps
        khat=l.where(keep,kp/l.maximum(nrm,gs_eps),0.0)
        clast=l.where(keep,nrm,0.0)
        mvec=l.sum(W*c[:,None],axis=0)
        delta=beta*(l.convert_layout(v-vb,SliceLayout(0,S))-gt*mvec)
        is_new=sr==cnt;cfull=l.where(is_new,clast,c)
        cq=l.sum(U*qn[None,:],axis=1)+l.where(is_new,l.sum(khat*qn,axis=0),0.0)
        out=l.convert_layout(out,SliceLayout(0,S))+gt*l.sum(W*cq[:,None],axis=0)+delta*l.sum(cfull*cq,axis=0)
        U_all=l.where(is_new[:,None],khat[None,:].to(fu.dtype.element_ty),U_all)
        W_all=l.where((sr<=cnt)[:,None],(gt*W+cfull[:,None]*delta[None,:]).to(fw.dtype.element_ty),W_all)
        a=a_new;cnt+=1
        l.store(output+(i_n*TOKENS+step)*HV*V+i_hv*V+x,l.convert_layout(out,Q).to(output.dtype.element_ty))
        if cnt>=RFULL:
            U_all,W_all=_cut(U_all,W_all,R,ITERS,REL_TOL,GATHER)
            cnt=cnt*0+R
    l.store(fa+base*128+x,a);l.store(up,U_all);l.store(wp,W_all)
    l.store(count+base,cnt);l.store(stale+slot,1)
