"""Private opt-in pointwise fusion; projections and reductions stay in torch.

All launches disable FP fusion: a tap product and its subsequent sum, and
collapse's stream products and sums, must round separately in FP32. Keeping
RMS/L2 reductions in torch preserves their training reduction order. This is
ordinary current-stream work, with no graph cache or cross-stream writers.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_round(x):
    # Explicit round-to-nearest-even keeps the CPU interpreter and device
    # contract identical (interpreter BF16 casts truncate in Triton 3.8).
    bits = x.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    rounded = tl.where((bits & 0x7FFFFFFF) > 0x7F800000, 0x7FC00000, rounded)
    return rounded.to(tl.float32, bitcast=True)


@triton.jit
def _taps(Full, W, Idx, Y, C: tl.constexpr, T: tl.constexpr, K: tl.constexpr,
          SF0: tl.constexpr, SF1: tl.constexpr, SW0: tl.constexpr, SW1: tl.constexpr,
          BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // C, i % C
    idx = tl.load(Idx + row, row < T, other=K - 1) - (K - 1)
    x = tl.load(Full + idx * SF0 + col * SF1, i < T * C, other=0).to(tl.float32)
    w = tl.load(W + col * SW0, i < T * C, other=0).to(tl.float32)
    y = x * w
    for k in tl.static_range(1, K):
        x = tl.load(Full + (idx + k) * SF0 + col * SF1, i < T * C, other=0).to(tl.float32)
        w = tl.load(W + col * SW0 + k * SW1, i < T * C, other=0).to(tl.float32)
        product = x * w
        y = y + product
    # SiLU is intentionally outside this kernel, so its exp implementation and
    # subsequent BF16 rounding remain the serving/training torch operation.
    tl.store(Y + i, y, i < T * C)


@triton.jit
def _gate(F, B, A, Bias, G, Beta, H: tl.constexpr, D: tl.constexpr, N: tl.constexpr,
          LOWER: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    hd, head = i % (H * D), (i % (H * D)) // D
    f = tl.load(F + i, i < N, other=0).to(tl.float32)
    bias = tl.load(Bias + hd, i < N, other=0).to(tl.float32)
    a = tl.load(A + head, i < N, other=0).to(tl.float32)
    f = f + bias
    f = tl.exp(a) * f
    g = LOWER * (1.0 / (1.0 + tl.exp(-f)))
    tl.store(G + i, g, i < N)
    # One writer per beta element (D evenly divides the row).
    bidx = i // D
    bm = (i < N) & (i % D == 0)
    b = tl.load(B + bidx, bm, other=0).to(tl.float32)
    beta = 1.0 / (1.0 + tl.exp(-b))
    tl.store(Beta + bidx, beta, bm)


@triton.jit
def _collapse_acc(Hb, Pre, Acc, D: tl.constexpr, HC: tl.constexpr,
                  T: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // D, i % D
    acc = tl.full((BLOCK,), 0, tl.float32)
    for n in tl.static_range(HC):
        pre = tl.load(Pre + row * HC + n, i < T * D, other=0).to(tl.float32)
        h = tl.load(Hb + row * HC * D + n * D + col, i < T * D, other=0).to(tl.float32)
        product = pre * h
        if n == 0:
            acc = product
        else:
            acc = acc + product
    tl.store(Acc + i, acc, i < T * D)


@triton.jit
def _collapse_norm(Acc, R, W, Out, D: tl.constexpr, T: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Training computes RMS from FP32 acc, but multiplies its BF16 copy.
    acc = _bf16_round(tl.load(Acc + i, i < T * D, other=0).to(tl.float32))
    r = tl.load(R + i // D, i < T * D, other=0).to(tl.float32)
    w = tl.load(W + i % D, i < T * D, other=0).to(tl.float32)
    y = acc * r
    y = y * w
    tl.store(Out + i, _bf16_round(y), i < T * D)


def conv_products(full, w, idx, K):
    T, C = idx.numel(), full.shape[1]
    y = torch.empty((T, C), dtype=torch.float32, device=full.device)
    _taps[(triton.cdiv(T * C, 256),)](full, w, idx, y, C, T, K,
                                    *full.stride(), *w.stride(), 256, enable_fp_fusion=False)
    return y


def gate_values(f, b, A_log, dt_bias, heads, head_dim, lower_bound):
    g = torch.empty(f.shape, dtype=torch.float32, device=f.device)
    beta = torch.empty(b.shape, dtype=torch.float32, device=b.device)
    _gate[(triton.cdiv(f.numel(), 256),)](f, b, A_log, dt_bias, g, beta, heads, head_dim,
                                       f.numel(), lower_bound, 256, enable_fp_fusion=False)
    return g, beta


def collapse_acc(hb, pre, hc_mult):
    T, D = hb.shape[0], hb.shape[1] // hc_mult
    acc = torch.empty((T, D), dtype=torch.float32, device=hb.device)
    _collapse_acc[(triton.cdiv(T * D, 256),)](hb, pre, acc, D, hc_mult, T,
                                           256, enable_fp_fusion=False)
    return acc


def collapse_norm(acc, r, norm_w):
    out = torch.empty(acc.shape, dtype=torch.bfloat16, device=acc.device)
    T, D = acc.shape
    _collapse_norm[(triton.cdiv(T * D, 256),)](acc, r, norm_w, out, D, T, 256, enable_fp_fusion=False)
    return out
