"""Scheme-C byte operations. E/D GEMMs and their FP32 arithmetic stay native.

No approximate division or FMA contraction: byte parity includes e2m1 midpoint
ties, e4m3 scales, negative zero, gap escapes, and zeroed storage padding.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack_nv(X, Z, S, G, XS: tl.constexpr, XC: tl.constexpr,
             R: tl.constexpr, BLOCKS: tl.constexpr):
    row = tl.program_id(0)
    b = tl.arange(0, BLOCKS)
    c = tl.arange(0, 16)
    col = b[:, None] * 16 + c[None, :]
    x = tl.load(X + row * XS + col * XC, col < R, other=0).to(tl.float32)
    maximum = tl.max(tl.max(tl.abs(x), 1), 0)
    # PyTorch's scalar division kernel multiplies by the FP32 reciprocal.
    g = tl.maximum(maximum, 1.0e-12) * (1.0 / 2688.0)
    blk = tl.div_rn(x, g)
    scale = (tl.max(tl.abs(blk), 1) * (1.0 / 6.0)).to(tl.float8e4nv)
    sb = tl.maximum(scale.to(tl.float32), 0.001953125)
    y = tl.minimum(tl.maximum(tl.div_rn(blk, sb[:, None]), -6.0), 6.0)
    a = tl.abs(y)
    mag = ((a > .25).to(tl.int32) + (a > .75).to(tl.int32)
           + (a > 1.25).to(tl.int32) + (a > 1.75).to(tl.int32)
           + (a > 2.5).to(tl.int32) + (a > 3.5).to(tl.int32)
           + (a > 5.0).to(tl.int32))
    code = mag | ((y < 0).to(tl.int32) << 3)
    pairs = tl.reshape(code, (BLOCKS, 8, 2))
    packed = tl.sum(pairs << (tl.arange(0, 2)[None, None, :] * 4), 2)
    pc = b[:, None] * 8 + tl.arange(0, 8)[None, :]
    tl.store(Z + row * (R // 2) + pc, packed, pc < R // 2)
    tl.store(S + row * (R // 16) + b, scale, b < R // 16)
    tl.store(G + row, g)


@triton.jit
def _unpack_nv(Z, S, G, X, ZS: tl.constexpr, ZC: tl.constexpr,
               SS: tl.constexpr, SC: tl.constexpr, GS: tl.constexpr,
               R: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    packed = tl.load(Z + row * ZS + (col // 2) * ZC, col < R, other=0).to(tl.int32)
    code = (packed >> ((col % 2) * 4)) & 15
    mag = code & 7
    level = tl.where(mag < 5, mag.to(tl.float32) * .5,
                     tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0)))
    # Multiplication, rather than subtraction from zero, preserves -0.
    q = level * tl.where((code & 8) != 0, -1.0, 1.0)
    scale = tl.load(S + row * SS + (col // 16) * SC, col < R, other=0).to(tl.float32)
    g = tl.load(G + row * GS)
    value = (q * tl.maximum(scale, .001953125)) * g
    tl.store(X + row * R + col, value, col < R)


@triton.jit
def _pack_gap(I, O, L, P, V, IS: tl.constexpr, IC: tl.constexpr,
              N: tl.constexpr, WIDTH: tl.constexpr, CAP: tl.constexpr,
              BLOCK: tl.constexpr, PAD: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    idx = tl.load(I + row * IS + c * IC, c < N, other=0).to(tl.int64)
    key = tl.where(c < N, idx * BLOCK + c, 9223372036854775807)
    key = tl.sort(key, descending=False)
    sorted_idx = key // BLOCK
    order = key % BLOCK
    previous = tl.gather(sorted_idx, tl.maximum(c - 1, 0), 0)
    gap = sorted_idx - tl.where(c == 0, -1, previous) - 1
    valid = tl.sum(((c < N) & ((sorted_idx < 0) | (sorted_idx >= WIDTH) | (gap < 0))).to(tl.int32), 0) == 0
    escape = gap >= 255
    sizes = tl.where(c < N, 1 + 2 * escape.to(tl.int32), 0)
    offsets = tl.cumsum(sizes, 0) - sizes
    length = tl.sum(sizes, 0)
    valid = valid & (length <= CAP)
    pad = tl.arange(0, PAD)
    tl.store(O + row * CAP + pad, 0, (pad < CAP) & ((pad >= length) | ~valid))
    tl.store(O + row * CAP + offsets, tl.where(escape, 255, gap), (c < N) & valid)
    tl.store(O + row * CAP + offsets + 1, gap & 255, (c < N) & escape & valid)
    tl.store(O + row * CAP + offsets + 2, gap >> 8, (c < N) & escape & valid)
    tl.store(P + row * N + c, order, c < N)
    tl.store(L + row, length)
    tl.store(V + row, valid)


@triton.jit
def _unpack_gap(I, L, O, V, IS: tl.constexpr, IC: tl.constexpr, LS: tl.constexpr,
                CAP: tl.constexpr, N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    byte = tl.load(I + row * IS + c * IC, c < CAP, other=0).to(tl.int32)
    prev = tl.load(I + row * IS + (c - 1) * IC, (c > 0) & (c <= CAP), other=0).to(tl.int32)
    marker = (byte == 255) & (prev != 255)
    following = tl.gather(marker, tl.maximum(c - 1, 0), 0) & (c >= 1)
    following2 = tl.gather(marker, tl.maximum(c - 2, 0), 0) & (c >= 2)
    length = tl.load(L + row * LS).to(tl.int32)
    live = (c < CAP) & (c < length)
    starts = live & ~following & ~following2
    low = tl.load(I + row * IS + (c + 1) * IC, c + 1 < CAP, other=0).to(tl.int32)
    high = tl.load(I + row * IS + (c + 2) * IC, c + 2 < CAP, other=0).to(tl.int32)
    gap = tl.where(marker, low + (high << 8), byte)
    ranks = tl.cumsum(starts.to(tl.int32), 0) - 1
    coords = tl.cumsum(tl.where(starts, gap + 1, 0), 0) - 1
    malformed = (tl.sum(starts.to(tl.int32), 0) != N)
    malformed = malformed | (tl.sum((marker & live & (c + 2 >= length)).to(tl.int32), 0) != 0)
    malformed = malformed | (tl.sum((starts & ((coords < 0) | (coords >= WIDTH))).to(tl.int32), 0) != 0)
    malformed = malformed | (length < 0) | (length > CAP)
    tl.store(O + row * N + ranks, coords, starts & (ranks >= 0) & (ranks < N))
    tl.store(V + row, ~malformed)


def pack_nvfp4(x):
    n, r = x.shape
    z = torch.empty((n, r//2), dtype=torch.uint8, device=x.device)
    s = torch.empty((n, r//16), dtype=torch.float8_e4m3fn, device=x.device)
    g = torch.empty((n, 1), dtype=torch.float32, device=x.device)
    if n:
        _pack_nv[(n,)](x, z, s, g, *x.stride(), r, triton.next_power_of_2(r//16), enable_fp_fusion=False)
    return z, s, g


def unpack_nvfp4(z, s, g):
    n, half = z.shape
    out = torch.empty((n, half*2), dtype=torch.float32, device=z.device)
    if n:
        _unpack_nv[(n,)](z, s, g, out, *z.stride(), *s.stride(), g.stride(0),
                         half*2, triton.next_power_of_2(half*2), enable_fp_fusion=False)
    return out


def pack_gap8(indices, width, *, validate):
    n, sparse = indices.shape
    cap = sparse + 2 * min(sparse, (width-sparse)//255)
    out = torch.empty((n, cap), dtype=torch.uint8, device=indices.device)
    lengths = torch.empty((n, 1), dtype=torch.int16, device=indices.device)
    order = torch.empty((n, sparse), dtype=torch.int64, device=indices.device)
    valid = torch.empty(n, dtype=torch.bool, device=indices.device)
    if n:
        _pack_gap[(n,)](indices, out, lengths, order, valid, *indices.stride(), sparse, width, cap,
                        triton.next_power_of_2(sparse), triton.next_power_of_2(cap))
    if validate and not bool(valid.all()):
        raise ValueError('gap8 index out of range or duplicate indices')
    return out, lengths, order


def unpack_gap8(stream, lengths, sparse, width, *, validate):
    n, cap = stream.shape
    out = torch.empty((n, sparse), dtype=torch.int64, device=stream.device)
    valid = torch.empty(n, dtype=torch.bool, device=stream.device)
    if n:
        _unpack_gap[(n,)](stream, lengths, out, valid, *stream.stride(), lengths.stride(0),
                          cap, sparse, width, triton.next_power_of_2(cap))
    if validate and not bool(valid.all()):
        raise ValueError('malformed gap8 stream or decoded index out of range')
    return out
