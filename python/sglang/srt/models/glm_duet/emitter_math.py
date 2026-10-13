"""The GLM-5.3-Flash DUET emitters' arithmetic (torch only; twinstar docs/79).  Every function transcribes the twinstar training
modules the released components were trained with (twinstar/models/glm5_duet.py at cf92ab005: Collapse, GlmKDAEmitter -- the layer's
own arithmetic, bf16 GEMM outputs on the fp32 master weights cast to bf16, fp32 conv taps / gates / scan state --, GlmDSAEmitter --
fp32 GEMMs on the fake-quantised input --, on twinstar/models/glm5_next.py's GlmKDA, ShortConvF32, rms_norm, quant_fp8_groups, and
twinstar/kimi/model.py's l2norm): the same operation order and roundings (TF32 off), so the served prompt memories are the trained
ones.  The KDA scan itself is fla's chunk_kda (the training kernel) called by the server."""
import os

import torch
import torch.nn.functional as F

_FUSED_EMITTER = os.environ.get("SGLANG_GLM_DUET_FUSED_EMITTER", "0") not in ("", "0")

FP8_MAX = 448.0
INV448 = float(torch.tensor(1.0 / FP8_MAX, dtype=torch.float32))      # fl(1/448), the reciprocal the served kernels multiply by


def collapse(h, fn, scale, base, norm_w, *, hc_mult, eps, hc_eps):
    """Attention-site collapse + fused input norm: streams h (T, n D) bf16 -> x (T, D) bf16 (glm5_duet.Collapse: the `pre` rows only,
    the RMS statistic of the fp32 collapse applied to its bf16 copy)."""
    n = hc_mult
    D = h.shape[-1] // n
    hb = h.reshape(-1, n * D)
    rs = torch.rsqrt(hb.float().pow(2).mean(-1, keepdim=True) + eps)
    mix = (hb.to(torch.bfloat16).float() @ fn[:n].float().t()) * rs
    pre = torch.sigmoid(mix * scale[0].float() + base[:n].float()) + hc_eps
    if _FUSED_EMITTER and h.is_cuda:
        from ._emitter_kernels import collapse_acc, collapse_norm

        acc = collapse_acc(hb.contiguous(), pre, n)
        r = torch.rsqrt(acc.pow(2).mean(-1, keepdim=True) + eps)
        return collapse_norm(acc, r, norm_w)
    hs = hb.reshape(-1, n, D).float()
    acc = pre[:, 0:1] * hs[:, 0]
    for i in range(1, n):
        acc = acc + pre[:, i:i + 1] * hs[:, i]
    r = torch.rsqrt(acc.pow(2).mean(-1, keepdim=True) + eps)
    return (acc.to(torch.bfloat16).float() * r * norm_w.float()).to(torch.bfloat16)


def conv_taps(full, w, starts, lens, K, *, idx=None):
    """Causal depthwise conv + SiLU of several sequences packed with their K-1 history rows: full (R, C) fp32, request r's rows
    starts[r] .. starts[r] + K - 2 are its history and the next lens[r] rows its inputs; w (C, K) fp32.  Returns (T, C) fp32 in the
    order of the inputs (ShortConvF32._taps: tap by tap, product then sum, no FMA)."""
    if idx is None:
        idx = torch.cat([torch.arange(s + K - 1, s + K - 1 + n, device=full.device) for s, n in zip(starts, lens)])
    if _FUSED_EMITTER and full.is_cuda:
        from ._emitter_kernels import conv_products

        return F.silu(conv_products(full, w, idx, K))
    y = full[idx - (K - 1)] * w[None, :, 0]
    for k in range(1, K):
        y = y + full[idx - (K - 1) + k] * w[None, :, k]
    return F.silu(y)


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def kda_gates(x, f_a, f_b, b_w, A_log, dt_bias, *, heads, head_dim, lower_bound):
    """glm5_duet.GlmKDAEmitter's gates (x and the projection weights bf16: bf16 GEMM outputs, fp32 from there):
    g = lb sigmoid(exp(A_log) (f_b f_a x + dt_bias)) (T, H, Dk), beta = sigmoid(b x) (T, H)."""
    f = F.linear(F.linear(x, f_a), f_b).float().view(*x.shape[:-1], heads, head_dim)
    if _FUSED_EMITTER and x.is_cuda:
        from ._emitter_kernels import gate_values

        b = F.linear(x, b_w)
        return gate_values(f, b, A_log, dt_bias, heads, head_dim, lower_bound)
    g = lower_bound * torch.sigmoid(torch.exp(A_log.float()).view(heads, 1) * (f + dt_bias.float().view(heads, head_dim)))
    beta = torch.sigmoid(F.linear(x, b_w).float())
    return g, beta


def ceil_ue8m0(x):
    """Smallest power of two >= x > 0 (DeepGEMM ceil_to_ue8m0: the fp32 exponent, + 1 if any mantissa bit is set)."""
    bits = x.float().contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def fake_quant_act(x, mode, group=128):
    """The served per-token / 128-channel FP8 activation quantisation of the DSA kv_a input, returned dequantised in fp32 (glm5_duet
    fake_quant_act): mode "ue8m0" (DeepGEMM: power-of-two scales), "fp32" (fp32 scales amax / 448) or None (no quantisation)."""
    if mode is None:
        return x.float()
    shp = x.shape
    xf = x.float().reshape(*shp[:-1], shp[-1] // group, group)
    amax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-10)
    if mode == "ue8m0":
        s = ceil_ue8m0(amax * INV448)
        q = (xf * (1.0 / s)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    elif mode == "fp32":
        s = amax * INV448
        q = (xf * (FP8_MAX / amax)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    else:
        raise ValueError(f"unknown activation quantisation {mode!r}")
    return (q.float() * s).reshape(shp)


def dsa_memory(x, kv_a, kv_norm, wk, k_norm_w, k_norm_b, gate, *, eps, act_quant, k_norm_eps=1e-6):
    """x (T, D) bf16 -> (c (T, 512), k (T, 128), s (T, 128)) bf16 (glm5_duet.GlmDSAEmitter.memory on fp32 weights)."""
    xa = fake_quant_act(x, act_quant)
    c = F.linear(xa, kv_a.float()).to(torch.bfloat16)
    cf = c.float()
    c = (cf * torch.rsqrt(cf.pow(2).mean(-1, keepdim=True) + eps) * kv_norm.float()).to(torch.bfloat16)
    kk = F.linear(x.float(), wk.float()).to(torch.bfloat16).float()
    kk = F.layer_norm(kk, (kk.shape[-1],), k_norm_w.float(), k_norm_b.float(), k_norm_eps).to(torch.bfloat16)
    s = F.linear(x.float(), gate.float()).to(torch.bfloat16)
    return c, kk, s
