# SPDX-License-Identifier: Apache-2.0
"""Eager LR-KDA pool reference, with real factor storage and C=64 propagation.

Pool [slot, head, K, 2*(r+64)+2] stores U, W, exact sink, and metadata.
This implementation uses host slot/offset reads: CUDA graphs, speculative
rollback, radix snapshots and PD transfers must be disabled by its caller.
"""
from functools import lru_cache
import importlib
import os

import torch


@lru_cache(None)
def configure_precision():
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    os.environ["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
    import triton.language as tl
    importlib.import_module("fla.ops.kda.chunk_intra").SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")
    torch.backends.cuda.matmul.allow_tf32 = False


def factors(content, rank, projector):
    if projector == "eigh":
        scale = content.abs().amax((-2, -1), keepdim=True).clamp_min(torch.finfo(content.dtype).tiny)
        z = content/scale
        gram = z @ z.transpose(-1, -2)
        try:
            _, u = torch.linalg.eigh(gram)
        except torch.linalg.LinAlgError:
            if not torch.isfinite(content).all():
                raise RuntimeError("Nonfinite LR-KDA pool content")
            _, u = torch.linalg.eigh(gram.double())
            u = u.float()
        u = u[..., -rank:]
        return u, content.transpose(-1, -2) @ u
    if projector == "svd":
        u, s, vh = torch.linalg.svd(content, full_matrices=False)
        return u[..., :rank], vh[..., :rank, :].transpose(-1, -2)*s[..., None, :rank]
    raise ValueError(projector)


def read_slot(slot, rank):
    cap = rank+64
    offset = int(slot[0, 0, -1].item())
    count = int(slot[0, 1, -1].item())
    if slot.dtype != torch.float32 or not 0 <= count <= cap:
        raise ValueError("LR-KDA pool requires fp32 factors and valid count")
    return slot[..., :count], slot[..., cap:cap+count], slot[..., 2*cap], offset


def write_slot(slot, rank, u, w, sink, offset):
    cap, count = rank+64, u.shape[-1]
    if count > cap:
        raise ValueError("factor capacity exceeded")
    slot[..., :count].copy_(u)
    slot[..., cap:cap+count].copy_(w)
    slot[..., 2*cap].copy_(sink)
    slot[:, 0, -1] = offset
    slot[:, 1, -1] = count


def decode_one(slot, q, k, value, g, beta, mean, center, rank, interval=64, projector="eigh"):
    u, w, sink, offset = read_slot(slot, rank)
    dtype = value.dtype
    q, k = q.float(), k.float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True)+1e-6) * q.shape[-1]**-0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True)+1e-6)
    decay = g.float().exp()
    sink = decay*sink
    sink = sink+beta.float()[..., None]*k*(1-(sink*k).sum(-1, keepdim=True))
    u = decay[..., None]*u
    memory = (w @ (u.transpose(-1, -2) @ k[..., None]))[..., 0]
    centered = (value.float()-center).to(dtype).float()
    delta = beta.float()[..., None]*(centered-memory)
    u = torch.cat((u, k[..., None]), -1)
    w = torch.cat((w, delta[..., None]), -1)
    content_out = (w @ (u.transpose(-1, -2) @ q[..., None]))[..., 0].to(dtype)
    sink_out = (sink*q).sum(-1, keepdim=True).to(dtype)
    out = (content_out.float()+sink_out.float()*mean).to(dtype)
    offset += 1
    if offset % interval == 0:
        u, w = factors(u @ w.transpose(-1, -2), rank, projector)
    write_slot(slot, rank, u, w, sink, offset)
    return out


def prefill_one(slot, q, k, value, g, beta, mean, center, rank, prefix=0, projector="eigh"):
    configure_precision()
    from fla.ops.kda import chunk_kda

    if prefix == 0:
        slot.zero_()
    u, w, sink, offset = read_slot(slot, rank)
    if offset != prefix:
        raise ValueError(f"LR-KDA boundary offset mismatch: {offset} != {prefix}")
    state = torch.cat((u @ w.transpose(-1, -2), sink[..., None]), -1)[None]
    outputs, pos = [], 0
    while pos < q.shape[0]:
        end = min(q.shape[0], pos+64-(offset % 64))
        vc = (value[pos:end].float()-center).to(value.dtype)
        packed = torch.cat((vc, torch.ones_like(vc[..., :1])), -1)
        out, state = chunk_kda(q[None, pos:end].contiguous(), k[None, pos:end].contiguous(),
                               packed[None].contiguous(), g[None, pos:end].contiguous(),
                               beta[None, pos:end].contiguous(), initial_state=state,
                               output_final_state=True, use_qk_l2norm_in_kernel=True)
        outputs.append((out[0, ..., :-1].float()+out[0, ..., -1:].float()*mean).to(value.dtype))
        offset += end-pos
        if offset % 64 == 0:
            uu, ww = factors(state[0, ..., :-1], rank, projector)
            state = torch.cat((uu @ ww.transpose(-1, -2), state[0, ..., -1:]), -1)[None]
        pos = end
    # The mathematical rank grows at most one per token between boundaries.
    # Eigenspace compression of a partial final block only removes roundoff.
    count = min(rank + offset % 64, q.shape[-1])
    u, w = factors(state[0, ..., :-1], count, projector)
    write_slot(slot, rank, u, w, state[0, ..., -1], offset)
    return torch.cat(outputs, 0)
