"""HC4 boundary device helpers. No synchronization or implicit current slot.

Shapes/rounding/ownership: q38cc/lanes/hc2/HC-INTERFACE.md. Callers complete
native output reductions and side-stream dependencies before invoking these.
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def _fma_rn(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [v.ir_value(loc=loc, ip=ip) for v in (a, b, c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _add_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "add.rn.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def _mul_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def inject_gate(partials: cute.Tensor, row, branch):
    total = cutlass.Float32(0.0)
    for split in cutlass.range_constexpr(8):
        total = _add_rn(total, partials[row, split, branch])
    return 2.0 / (1.0 + cmath.exp(-total * 0.25))


@cute.jit
def combine(block, residual, gate):
    return _fma_rn(gate, block.to(cutlass.Float32),
                   residual.to(cutlass.Float32)).to(cutlass.BFloat16)


@cute.jit
def sumsq64(x0, x1):
    """One complete warp, columns lane/lane+32; full result only in lane 0."""
    a, b = x0.to(cutlass.Float32), x1.to(cutlass.Float32)
    total = _add_rn(_mul_rn(a, a), _mul_rn(b, b))
    for step in cutlass.range_constexpr(5):
        total = _add_rn(total, cute.arch.shuffle_sync_down(total, offset=16 >> step))
    return total


@cute.jit
def inv_rms(stats: cute.Tensor, row, branch, eps: cutlass.Constexpr):
    total = cutlass.Float32(0.0)
    for part in cutlass.range_constexpr(40):
        total = _add_rn(total, stats[row, branch, part])
    return cmath.rsqrt(_add_rn(_mul_rn(total, cutlass.Float32(1.0 / 2560.0)),
                              cutlass.Float32(eps)), fastmath=True)


@cute.jit
def normalize(value, raw_gamma, inv):
    scale = _add_rn(cutlass.Float32(1.0), raw_gamma.to(cutlass.Float32))
    return _mul_rn(_mul_rn(value.to(cutlass.Float32), inv), scale).to(cutlass.BFloat16)
