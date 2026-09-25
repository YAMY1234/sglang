"""Exact Scheme-C decode epilogue; the projection GEMM stays unchanged."""
import torch
import triton
import triton.language as tl


@triton.jit
def _epilogue(P, M, C, R, B, O, WIDTH: tl.constexpr,
              PS: tl.constexpr, PC: tl.constexpr, CS: tl.constexpr, CC: tl.constexpr,
              RS: tl.constexpr, BS: tl.constexpr, BC: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < WIDTH
    projection = tl.load(P + row * PS + col * PC, mask, other=0)
    mean = tl.load(M + col, mask, other=0)
    correction = tl.load(C + row * CS + col * CC, mask, other=0)
    rms = tl.load(R + row * RS)
    base = tl.load(B + row * BS + col * BC, mask, other=0).to(tl.float32)
    # Match the four distinct FP32 PyTorch operations. Compilation must not
    # contract the multiply and final add into an FMA.
    reconstructed = mean + projection
    corrected = reconstructed + correction
    residual = corrected * rms
    result = residual + base
    tl.store(O + row * WIDTH + col, result, mask)


def decode_epilogue(projection, mean, correction, rms, base):
    if (projection.dtype != torch.float32 or correction.dtype != torch.float32
            or mean.dtype != torch.float32 or rms.dtype != torch.float32
            or base.dtype != torch.bfloat16 or projection.shape != base.shape
            or correction.shape != base.shape or mean.shape != (base.shape[1],)
            or rms.shape != (base.shape[0], 1)):
        raise ValueError('decode epilogue requires the original FP32/BF16 operands')
    n, width = base.shape
    # Retain the native final BF16 conversion and sink override in the caller.
    # This also lets CPU interpretation verify all intermediate FP32 bits.
    out = torch.empty((n, width), dtype=torch.float32, device=base.device)
    if n:
        _epilogue[(n, triton.cdiv(width, 1024))](projection, mean, correction, rms, base, out,
            width, *projection.stride(), *correction.stride(), rms.stride(0), *base.stride(),
            1024, enable_fp_fusion=False)
    return out
