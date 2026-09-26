"""Decode convolution helpers with deferred, all-layer state publication."""
import triton
import triton.language as tl


@triton.jit
def conv_values(x, state, weight, bias, pending, slot, channel, token,
                XS:tl.constexpr, SS:tl.constexpr, SD:tl.constexpr, ST:tl.constexpr,
                WD:tl.constexpr, WT:tl.constexpr, DIM:tl.constexpr,
                HAS_BIAS:tl.constexpr, WRITE_STATE:tl.constexpr):
    base=state+slot*SS+channel*SD
    col0=tl.load(base)
    col1=tl.load(base+ST)
    col2=tl.load(base+2*ST)
    value=tl.load(x+token*XS+channel)
    wbase=weight+channel*WD
    w0=tl.load(wbase);w1=tl.load(wbase+WT)
    w2=tl.load(wbase+2*WT);w3=tl.load(wbase+3*WT)
    if HAS_BIAS:
        acc=tl.load(bias+channel).to(tl.float32)
    else:
        acc=tl.full(channel.shape,0,tl.float32)
    acc+=col0*w0
    acc+=col1*w1
    acc+=col2*w2
    acc+=value*w3
    acc=acc/(1+tl.exp(-acc))
    if WRITE_STATE:
        dst=pending+(token*DIM+channel)*3
        tl.store(dst,col1)
        tl.store(dst+1,col2)
        tl.store(dst+2,value)
    return acc.to(x.dtype.element_ty).to(tl.float32)


@triton.jit
def _publish(pending, state, indices, mask, destinations,
             PL:tl.constexpr, SL:tl.constexpr, SS:tl.constexpr,
             SD:tl.constexpr, ST:tl.constexpr, DIM:tl.constexpr,
             TRACK:tl.constexpr, BLOCK:tl.constexpr):
    layer=tl.program_id(1).to(tl.int64)
    slot=tl.load(indices).to(tl.int64)
    if slot<0:return
    offset=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    channel=offset//3;column=offset%3
    value=tl.load(pending+layer*PL+offset,channel<DIM,other=0)
    tl.store(state+layer*SL+slot*SS+channel*SD+column*ST,value,channel<DIM)
    if TRACK:
        if tl.load(mask)!=0:
            dst=tl.load(destinations).to(tl.int64)
            if dst>=0 and dst!=slot:
                tl.store(state+layer*SL+dst*SS+channel*SD+column*ST,value,channel<DIM)


def publish(pending,state,indices,mask,destinations):
    assert indices.numel()==1 and pending.shape[1]==1 and state.shape[-1]==3
    _publish[(triton.cdiv(state.shape[-2]*3,256),state.shape[0])](
        pending,state,indices,indices if mask is None else mask,
        indices if destinations is None else destinations,
        pending.stride(0),state.stride(0),state.stride(1),state.stride(2),state.stride(3),
        state.shape[-2],mask is not None,256,num_warps=4)
