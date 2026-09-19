"""Offline K1 state-precision screen. Not a serving pool or a timing path.

Dense state uses mathematical (K,V) rows; INT4 groups contain 32 V entries.
FP8 is e4m3fn, max-abs scales are fp32, rounding is nearest/even.
The factor emulator dequantizes into the existing bf16 K3-A compute buffers.
"""
import os
from functools import lru_cache

import torch
import triton
import triton.language as tl


def arm():
    path = os.environ.get("SGLANG_GDN_PRECISION_ARM_FILE")
    if not path:
        return "off"
    with open(path) as f:
        return f.read().strip()


@lru_cache(None)
def vbar_for(layer, hv, device):
    from sglang.srt.runtime_context import get_parallel
    data = torch.load(os.environ["SGLANG_GDN_PRECISION_CONSTS"], map_location="cpu", weights_only=False)
    rank = get_parallel().attn_tp_rank
    return data["vbar"][layer][rank * hv:(rank + 1) * hv].to(device=device, dtype=torch.float32).contiguous()


@triton.jit
def _qdq(x, MODE: tl.constexpr):
    # x is (K, V=128); reduce over V, or over each 32-element group.
    if MODE == 4:
        grouped = tl.reshape(x, (128, 4, 32))
        scale = tl.maximum(tl.max(tl.abs(grouped), 2) / 7., 1.e-30)
        y = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(grouped / scale[:, :, None]), -7.), 7.) * scale[:, :, None]
        return tl.reshape(y, (128, 128))
    elif MODE == 3:
        scale = tl.maximum(tl.max(tl.abs(x), 1) / 127., 1.e-30)
        return tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(x / scale[:, None]), -127.), 127.) * scale[:, None]
    elif MODE == 2:
        scale = tl.maximum(tl.max(tl.abs(x), 1) / 448., 1.e-30)
        return (x / scale[:, None]).to(tl.float8e4nv).to(tl.float32) * scale[:, None]
    else:
        return x


@triton.jit
def _dense_sequence(Q, K, V, AG, BG, AL, DB, VB, S, IDX, CU, O,
                    SQ: tl.constexpr, SK: tl.constexpr, SV: tl.constexpr,
                    SA: tl.constexpr, SB: tl.constexpr, SS: tl.constexpr,
                    H: tl.constexpr, HV: tl.constexpr, MODE: tl.constexpr):
    n, h = tl.program_id(0), tl.program_id(1)
    kh = h // (HV // H)
    d = tl.arange(0, 128)
    slot = tl.load(IDX + n).to(tl.int64)
    offsets = slot * SS + h * 16384 + d[:, None] + d[None, :] * 128
    s = tl.load(S + offsets).to(tl.float32)
    vb = tl.load(VB + h * 128 + d)
    av = tl.sum(s * vb[None, :], 1) / tl.maximum(tl.sum(vb * vb, 0), 1.e-30)
    c = _qdq(s - av[:, None] * vb[None, :], MODE)
    al = tl.load(AL + h).to(tl.float32)
    db = tl.load(DB + h).to(tl.float32)
    start, end = tl.load(CU + n), tl.load(CU + n + 1)
    for t in range(start, end):
        q = tl.load(Q + t * SQ + kh * 128 + d).to(tl.float32)
        k = tl.load(K + t * SK + kh * 128 + d).to(tl.float32)
        v = tl.load(V + t * SV + h * 128 + d).to(tl.float32)
        ag = tl.load(AG + t * SA + h).to(tl.float32) + db
        bg = tl.load(BG + t * SB + h).to(tl.float32)
        g = tl.exp(-tl.exp(al) * tl.where(ag <= 20., tl.log(1. + tl.exp(ag)), ag))
        beta = tl.sigmoid(bg).to(BG.dtype.element_ty).to(tl.float32)
        q = q / tl.sqrt(tl.sum(q * q, 0) + 1.e-6) * (128. ** -0.5)
        k = k / tl.sqrt(tl.sum(k * k, 0) + 1.e-6)
        av = g * (av - beta * k * tl.sum(av * k, 0)) + beta * k
        c = c * g
        delta = beta * (v - vb - tl.sum(c * k[:, None], 0))
        c += k[:, None] * delta[None, :]
        out = tl.sum(c * q[:, None], 0) + vb * tl.sum(av * q, 0)
        tl.store(O + (t * HV + h) * 128 + d, out)
        c = _qdq(c, MODE)
    tl.store(S + offsets, c + av[:, None] * vb[None, :])


def dense_sequence(layer, query, key, value, a, b, states, indices, cu):
    mode = {"split": 0, "P2": 2, "P3": 3, "P4": 4}[arm()]
    hv = value.shape[-2]
    vb = vbar_for(layer.layer_id, hv, str(value.device))
    out = torch.empty_like(value)
    _dense_sequence[(indices.numel(), hv)](
        query, key, value, a, b, layer.A_log, layer.dt_bias, vb,
        states, indices, cu, out, query.stride(1), key.stride(1), value.stride(1),
        a.stride(0), b.stride(0), states.stride(0), query.shape[-2], hv, mode,
        num_warps=8,
    )
    return out


@triton.jit
def _factor_qdq(U, W, COUNT, IDX, HV: tl.constexpr, R: tl.constexpr):
    bh, col = tl.program_id(0), tl.program_id(1)
    slot = tl.load(IDX + bh // HV).to(tl.int64)
    h = bh % HV
    count = tl.load(COUNT + slot * HV + h)
    if col < count:
        d = tl.arange(0, 128)
        off = ((slot * HV + h) * R + col) * 128 + d
        u, w = tl.load(U + off).to(tl.float32), tl.load(W + off).to(tl.float32)
        su, sw = tl.maximum(tl.max(tl.abs(u), 0) / 448., 1.e-30), tl.maximum(tl.max(tl.abs(w), 0) / 448., 1.e-30)
        tl.store(U + off, (u / su).to(tl.float8e4nv).to(tl.float32) * su)
        tl.store(W + off, (w / sw).to(tl.float8e4nv).to(tl.float32) * sw)


def factor_qdq(u, w, count, slots):
    _factor_qdq[(slots.numel() * u.shape[1], u.shape[2])](u, w, count, slots, u.shape[1], u.shape[2], num_warps=4)


def capture_probe(layer, query, key, value, a, b, states, indices):
    path = os.environ.get("SGLANG_GDN_PRECISION_CAPTURE")
    if not path or layer.layer_id not in (1, 13, 25, 45) or arm() != "off":
        return
    target = int(os.environ.get("SGLANG_GDN_PRECISION_CAPTURE_TOKENS", "0"))
    if target and key.shape[1] != target:
        return  # Ignore engine warmup inputs when capturing a whole prompt.
    from sglang.srt.runtime_context import get_parallel
    rank = get_parallel().attn_tp_rank
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"rank{rank}-L{layer.layer_id}.pt")
    if os.path.exists(filename):
        return
    vb = vbar_for(layer.layer_id, value.shape[-2], str(value.device))
    torch.save({"q": query.cpu(), "k": key.cpu(), "v": value.cpu(), "a_gate": a.cpu(), "b_gate": b.cpu(),
                "A_log": layer.A_log.cpu(), "dt_bias": layer.dt_bias.cpu(), "vbar": vb.cpu(),
                "S": states[indices.long()].cpu(), "layer": layer.layer_id, "rank": rank}, filename)
