"""Store the rounded decode output through the existing RMS/gate expression."""
import triton
import triton.language as tl


@triton.jit
def store_normalized(value, output, z, weight, token, head,
                     ZT:tl.constexpr, ZH:tl.constexpr, V:tl.constexpr,
                     EPS:tl.constexpr, ROWS:tl.constexpr, ACTIVATION:tl.constexpr):
    columns=tl.arange(0,V)
    rows=tl.arange(0,ROWS)
    # Preserve the original BF16 output store/load boundary before normalization.
    x=value.to(output.dtype.element_ty).to(tl.float32)
    x=tl.broadcast_to(x[None,:],(ROWS,V))
    variance=tl.sum(x*x,axis=1)/V
    rstd=tl.rsqrt(variance+EPS)
    w=tl.load(weight+columns).to(tl.float32)
    gate=tl.load(z+token*ZT+head*ZH+columns).to(tl.float32)
    y=(x*rstd[:,None])*w[None,:]
    if ACTIVATION=='sigmoid':
        y*=tl.sigmoid(gate)[None,:]
    else:
        y*=(gate*tl.sigmoid(gate))[None,:]
    # ROWS follows the original production/deterministic normalization layout.
    tl.store(output[None,:]+tl.zeros((ROWS,1),tl.int32),y,rows[:,None]==0)
