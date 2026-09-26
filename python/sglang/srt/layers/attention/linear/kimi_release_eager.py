"""Eager Kimi recurrence for the final DUET accuracy branch.

Keeps the emitter's FP32 convolution windows and the native [H,K,V] state.
No pruning occurs inside this scan; the model owns prompt-end/W-step pruning.
This module has no runtime imports so it can be checked in the GPU image on CPU.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.no_grad()
def forward_eager(
    mixed_qkv, forget, beta, *, conv_weight, a_log, dt_bias,
    conv_pool, state_pool, slots, lengths, prefixes, heads, head_dim, decode,
):
    if conv_pool.dtype != torch.float32 or state_pool.dtype != torch.float32:
        raise ValueError("DUET Kimi requires FP32 convolution and recurrent pools")
    if sum(lengths) != mixed_qkv.shape[0] or len(slots) != len(lengths):
        raise ValueError("invalid eager Kimi batch geometry")
    if decode and any(length != 1 for length in lengths):
        raise ValueError("DUET eager decode requires one token per request")
    if len(set(slots)) != len(slots):
        raise ValueError("duplicate Kimi state slots")
    width = conv_weight.shape[-1]
    channels = heads * head_dim
    if mixed_qkv.shape[-1] != 3 * channels:
        raise ValueError("DUET Kimi requires equal q/k/v head dimensions")
    forget = forget.reshape(-1, heads, head_dim).float()
    beta = beta.reshape(-1, heads).float()
    if decode:
        beta = beta.sigmoid()
    g = -a_log.float().reshape(1, heads, 1).exp() * F.softplus(
        forget + dt_bias.float().reshape(heads, head_dim))
    output = []
    start = 0
    for slot, length, prefix in zip(slots, lengths, prefixes):
        raw = mixed_qkv[start:start + length]
        previous = conv_pool[slot].transpose(0, 1)
        state = state_pool[slot].clone()
        if not prefix:
            previous = torch.zeros_like(previous)
            state.zero_()
        if decode:
            full = torch.cat((previous, raw.transpose(0, 1).float()), dim=-1)
            convolved = F.silu((full * conv_weight.float()).sum(-1)).unsqueeze(0)
        else:
            # Native prompt conv runs in the frozen projection's dtype. The
            # emitter path writes FP32 windows separately and never enters here.
            full = torch.cat((previous.to(raw.dtype), raw.transpose(0, 1)), dim=-1)
            convolved = F.silu(F.conv1d(
                full.unsqueeze(0), conv_weight.to(raw.dtype).unsqueeze(1),
                groups=3 * channels)).squeeze(0).transpose(0, 1)
        conv_pool[slot].copy_(full[:, -(width - 1):].transpose(0, 1))
        q, k, v = convolved.reshape(length, 3, heads, head_dim).unbind(1)
        q, k, v = q.float(), k.float(), v.float()
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        for i in range(length):
            state = state * g[start + i].exp()[..., None]
            prediction = torch.einsum("hk,hkv->hv", k[i], state)
            state = state + torch.einsum(
                "hk,hv->hkv", beta[start + i, :, None] * k[i], v[i] - prediction)
            output.append(torch.einsum("hk,hkv->hv", q[i] * head_dim ** -0.5, state))
        state_pool[slot].copy_(state)
        start += length
    return torch.stack(output).unsqueeze(0)
