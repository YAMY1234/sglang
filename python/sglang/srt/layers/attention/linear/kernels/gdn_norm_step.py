"""Store the rounded decode output through the existing RMS/gate expression."""
import triton
import triton.language as tl
import os

ORIGINAL_TREE = tl.constexpr(os.environ.get('SGLANG_GDN_NORM_ORIGINAL_TREE', '0') == '1')
INTERPRETER = tl.constexpr(os.environ.get('TRITON_INTERPRET', '0') == '1')


@triton.jit
def original_variance(x, ROWS:tl.constexpr, V:tl.constexpr):
    # The served V=128/one-warp norm groups four adjacent columns per lane,
    # then combines lanes in xor 16,8,4,2,1 order. The recurrence's inferred
    # layout instead puts eight columns on each of sixteen lanes. Express
    # the logical tree explicitly so its floating-point order cannot follow
    # the recurrence layout. f32x2 lowering fuses the x0 and x2 squares.
    quads=tl.reshape(x,(ROWS,V//4,2,2))
    even,odd=tl.split(quads)
    x0,x2=tl.split(even)
    x1,x3=tl.split(odd)
    if INTERPRETER:
        p1=x1*x1
        p3=x3*x3
    else:
        p1=tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $1;',
            constraints='=f,f',args=[x1],dtype=tl.float32,is_pure=True,pack=1)
        p3=tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $1;',
            constraints='=f,f',args=[x3],dtype=tl.float32,is_pure=True,pack=1)
    partial=tl.fma(x0,x0,p1)
    partial=tl.fma(x2,x2,partial)+p3
    lanes=tl.arange(0,V//4)
    for shift in tl.static_range(5):
        peer=tl.broadcast_to((lanes^(16>>shift))[None,:],(ROWS,V//4))
        partial=partial+tl.gather(partial,peer,axis=1)
    return tl.sum(tl.where(lanes[None,:]==0,partial,0.),axis=1)/V


@triton.jit
def store_normalized(value, output, z, weight, token, head,
                     ZT:tl.constexpr, ZH:tl.constexpr, V:tl.constexpr,
                     EPS:tl.constexpr, ROWS:tl.constexpr, ACTIVATION:tl.constexpr):
    columns=tl.arange(0,V)
    rows=tl.arange(0,ROWS)
    # Preserve the original BF16 output store/load boundary before normalization.
    x=value.to(output.dtype.element_ty).to(tl.float32)
    x=tl.broadcast_to(x[None,:],(ROWS,V))
    if ORIGINAL_TREE and V==128:
        variance=original_variance(x,ROWS,V)
    else:
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
