"""Centered dense GDN state, fp32 watermark and row/group scaled byte storage.

Based on the K0 dense recurrence (twinstar/kernels/gdn_factored.py). No SVD,
factorization or expiry work. The physical content axes are (key, value).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _read(C, SC, off, MODE: tl.constexpr):
    k = tl.arange(0, 128)
    v = tl.arange(0, 128)
    if MODE == 4:
        raw = tl.load(C + off * 8192 + k[:, None] * 64 + v[None, :] // 2).to(tl.int32)
        nibble = (raw >> ((v[None, :] % 2) * 4)) & 15
        val = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)
        scale = tl.load(SC + off * 512 + k[:, None] * 4 + v[None, :] // 32)
    else:
        raw = tl.load(C + off * 16384 + k[:, None] * 128 + v[None, :])
        if MODE == 2:
            val = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        else:
            val = raw.to(tl.int8, bitcast=True).to(tl.float32)
        scale = tl.load(SC + off * 128 + k)[:, None]
    return val * scale


@triton.jit
def _write(c, C, SC, off, MODE: tl.constexpr):
    k = tl.arange(0, 128)
    v = tl.arange(0, 128)
    if MODE == 4:
        grouped = tl.reshape(c, (128, 4, 32))
        scale = tl.maximum(tl.max(tl.abs(grouped), 2) / 7., 1.e-30)
        q = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(grouped / scale[:, :, None]), -7.), 7.).to(tl.int32)
        lo, hi = tl.split(tl.reshape(q, (128, 64, 2)))
        packed = (lo & 15) | ((hi & 15) << 4)
        tl.store(C + off * 8192 + k[:, None] * 64 + tl.arange(0, 64)[None, :], packed.to(tl.uint8))
        tl.store(SC + off * 512 + k[:, None] * 4 + tl.arange(0, 4)[None, :], scale)
    else:
        if MODE == 2:
            scale = tl.maximum(tl.max(tl.abs(c), 1) / 448., 1.e-30)
            q = (c / scale[:, None]).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        else:
            scale = tl.maximum(tl.max(tl.abs(c), 1) / 127., 1.e-30)
            q = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(c / scale[:, None]), -127.), 127.).to(tl.int8).to(tl.uint8, bitcast=True)
        tl.store(C + off * 16384 + k[:, None] * 128 + v[None, :], q)
        tl.store(SC + off * 128 + k, scale)


@triton.jit
def _pack(S, A, C, SC, VB, IDX, HV: tl.constexpr, MODE: tl.constexpr):
    row, h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(IDX + row).to(tl.int64)
    if slot < 0:
        return
    d = tl.arange(0, 128)
    s = tl.load(S + (row * HV + h) * 16384 + d[:, None] + d[None, :] * 128)
    vb = tl.load(VB + h * 128 + d)
    av = tl.sum(s * vb[None, :], 1) / tl.maximum(tl.sum(vb * vb, 0), 1.e-30)
    off = slot * HV + h
    tl.store(A + off * 128 + d, av)
    _write(s - av[:, None] * vb[None, :], C, SC, off, MODE)


@triton.jit
def _unpack(A, C, SC, VB, IDX, S, HV: tl.constexpr, MODE: tl.constexpr):
    row, h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(IDX + row).to(tl.int64)
    d = tl.arange(0, 128)
    off = tl.maximum(slot, 0) * HV + h
    c = _read(C, SC, off, MODE)
    av = tl.load(A + off * 128 + d)
    vb = tl.load(VB + h * 128 + d)
    s = tl.where(slot >= 0, c + av[:, None] * vb[None, :], 0.)
    tl.store(S + (row * HV + h) * 16384 + d[:, None] + d[None, :] * 128, s)


@triton.jit
def _dense_quant_packed_step(MIX, AG, BG, AL, DB, A, C, SC, VB, IDX, STALE, O,
                            SM: tl.constexpr, SA: tl.constexpr, SB: tl.constexpr,
                            H: tl.constexpr, HV: tl.constexpr, MODE: tl.constexpr):
    row, h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(IDX + row).to(tl.int64)
    d = tl.arange(0, 128)
    if slot < 0:
        tl.store(O + (row * HV + h) * 128 + d, 0.)
        return
    kh = h // (HV // H)
    q = tl.load(MIX + row * SM + kh * 128 + d).to(tl.float32)
    k = tl.load(MIX + row * SM + (H + kh) * 128 + d).to(tl.float32)
    v = tl.load(MIX + row * SM + 2 * H * 128 + h * 128 + d).to(tl.float32)
    ag = tl.load(AG + row * SA + h).to(tl.float32) + tl.load(DB + h).to(tl.float32)
    bg = tl.load(BG + row * SB + h).to(tl.float32)
    g = tl.exp(-tl.exp(tl.load(AL + h).to(tl.float32)) * tl.where(ag <= 20., tl.log(1. + tl.exp(ag)), ag))
    beta = tl.sigmoid(bg).to(BG.dtype.element_ty).to(tl.float32)
    q = q / tl.sqrt(tl.sum(q * q, 0) + 1.e-6) * (128. ** -.5)
    k = k / tl.sqrt(tl.sum(k * k, 0) + 1.e-6)
    off = slot * HV + h
    av = tl.load(A + off * 128 + d)
    vb = tl.load(VB + h * 128 + d)
    c = _read(C, SC, off, MODE) * g
    av = g * (av - beta * k * tl.sum(av * k, 0)) + beta * k
    c += k[:, None] * (beta * (v - vb - tl.sum(c * k[:, None], 0)))[None, :]
    out = tl.sum(c * q[:, None], 0) + vb * tl.sum(av * q, 0)
    tl.store(O + (row * HV + h) * 128 + d, out)
    tl.store(A + off * 128 + d, av)
    _write(c, C, SC, off, MODE)
    tl.store(STALE + slot, 1)


def pack(s, a, c, scales, vb, slots, mode):
    assert s.shape[-2:] == (128, 128) and s.is_contiguous()
    _pack[(slots.numel(), a.shape[1])](s, a, c, scales, vb, slots, a.shape[1], mode, num_warps=8)


def unpack(a, c, scales, vb, slots, mode):
    s = torch.empty(slots.numel(), a.shape[1], 128, 128, device=a.device, dtype=torch.float32)
    _unpack[(slots.numel(), a.shape[1])](a, c, scales, vb, slots, s, a.shape[1], mode, num_warps=8)
    return s


def packed_step(mixed, ag, bg, *, layer, a, c, scales, vb, slots, stale, mode):
    out = torch.empty(mixed.shape[0], 1, layer.num_v_heads, 128, device=mixed.device, dtype=mixed.dtype)
    _dense_quant_packed_step[(mixed.shape[0], layer.num_v_heads)](
        mixed, ag, bg, layer.A_log, layer.dt_bias, a, c, scales, vb, slots, stale, out,
        mixed.stride(0), ag.stride(0), bg.stride(0), layer.num_q_heads, layer.num_v_heads, mode, num_warps=8,
    )
    return out


@triton.jit
def _track_quant_tensor(X, SRC, MASK, DST, L: tl.constexpr, S: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row, layer, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    src, dst = tl.load(SRC + row).to(tl.int64), tl.load(DST + row).to(tl.int64)
    if tl.load(MASK + row) and src >= 0 and dst >= 0:
        d = block * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + (layer * S + src) * WIDTH + d, mask=d < WIDTH, other=0)
        tl.store(X + (layer * S + dst) * WIDTH + d, x, mask=d < WIDTH)


def track_tensor(x, src, mask, dst):
    width = x[0, 0].numel()
    _track_quant_tensor[(src.numel(), x.shape[0], triton.cdiv(width, 1024))](
        x, src, mask, dst, x.shape[0], x.shape[1], width, 1024,
    )
