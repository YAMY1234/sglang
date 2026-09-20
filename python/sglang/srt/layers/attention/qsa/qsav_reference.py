# ruff: noqa
# Copied from YAMY1234/twinstar-pd-models cdc30c85972b0b5772a848193c8f752ecf5bce24
# twinstar/duet/qsav.py. Only private component-return hooks are added.
"""Value-side code for the sparse-attention prompt cache on Flash-Next (docs/51 §41): each stored prompt value
v (per KV head, hd = 256) is replaced by  mean + U_r U_r^T (v - mean) + the m largest residual coordinates exact.
Keys and indexer keys are untouched, so WHICH blocks are selected does not change; only what is read from them.
Selected by QSA_V="path.pt:rank:m" (basis from probes.qsavbasis); applied to the prompt states in Qwen4Prefill.forward.
Bytes per token per layer: 2 heads x (r + 2m) numbers instead of 2 x 256."""
import os
from typing import Optional

import torch

_M: Optional[dict] = None


def active() -> bool:
    return bool(os.environ.get("QSA_V"))


_MK: Optional[dict] = None


def active_k() -> bool:
    return bool(os.environ.get("QSA_K"))


def _load_k():
    global _MK
    if _MK is None:
        path, r, m = os.environ["QSA_K"].split(":")
        P = torch.load(path, map_location="cpu")
        _MK = {"U": {int(l): u[:, :, : int(r)].float() for l, u in P["UK"].items()}, "mean": {int(l): x.float() for l, x in P["meanK"].items()},
               "r": int(r), "m": int(m), "rot": int(P["rot"])}
        print(f"[qsav] sparse-attention keys: first {_MK['rot']} (rotated) dims exact, remaining dims rank {r} + {m} sparse coordinates", flush=True)
    return _MK


def apply_k(l: int, k: torch.Tensor) -> torch.Tensor:
    """k (B, Hkv, S, hd) post-RoPE -> rotated dims exact, non-rotated dims coded."""
    M = _load_k()
    if l not in M["U"]:
        return k
    rot = M["rot"]
    U = M["U"][l].to(k.device)
    mean = M["mean"][l].to(k.device)
    kn = k[..., rot:].float() - mean[None, :, None, :]
    z = torch.einsum("bhsd,hdr->bhsr", kn, U)
    rec = torch.einsum("bhsr,hdr->bhsd", z, U)
    if M["m"]:
        res = kn - rec
        idx = res.abs().topk(M["m"], dim=-1).indices
        rec = rec.scatter_add(-1, idx, res.gather(-1, idx))
    return torch.cat([k[..., :rot], (rec + mean[None, :, None, :]).to(k.dtype)], dim=-1)


def _load():
    global _M
    if _M is None:
        path, r, m = os.environ["QSA_V"].split(":")
        P = torch.load(path, map_location="cpu")
        _M = {"U": {int(l): u[:, :, : int(r)].float() for l, u in P["U"].items()}, "mean": {int(l): x.float() for l, x in P["mean"].items()},
              "r": int(r), "m": int(m)}
        print(f"[qsav] sparse-attention values: rank {r} + {m} sparse coordinates per token-head, {len(_M['U'])} layers", flush=True)
    return _M


def apply(l: int, v: torch.Tensor, *, _model=None, _return_components=False):
    """v (B, Hkv, S, hd) -> coded values, same shape/dtype."""
    M = _load() if _model is None else _model
    if l not in M["U"]:
        return v
    U = M["U"][l].to(v.device)  # (Hkv, hd, r)
    mean = M["mean"][l].to(v.device)  # (Hkv, hd)
    vf = v.float() - mean[None, :, None, :]
    z = torch.einsum("bhsd,hdr->bhsr", vf, U)
    rec = torch.einsum("bhsr,hdr->bhsd", z, U)
    if M["m"]:
        res = vf - rec
        idx = res.abs().topk(M["m"], dim=-1).indices
        rec = rec.scatter_add(-1, idx, res.gather(-1, idx))
    if _return_components:
        if not M["m"]:
            idx = torch.empty_like(z[..., :0], dtype=torch.int64)
        return z, idx
    return (rec + mean[None, :, None, :]).to(v.dtype)


_MX: Optional[dict] = None


def active_joint() -> bool:
    return bool(os.environ.get("QSA_X"))


def _load_joint():
    """QSA_X="path.pt:rK:rV:m" -- joint bases from probes.qsaxlayer over the prefix QSA layers; rK / rV joint ranks per KV
    head (0 = leave that side untouched); m = largest residual coordinates kept exact per token-head over the concatenation."""
    global _MX
    if _MX is None:
        path, rk, rv, m = os.environ["QSA_X"].split(":")
        P = torch.load(path, map_location="cpu")
        _MX = {"layers": [int(l) for l in P["layers"]], "rot": int(P["rot"]), "rk": int(rk), "rv": int(rv), "m": int(m),
               "UK": P["K"]["U"][:, :, : int(rk)].float() if int(rk) else None, "mK": P["K"]["mean"].float(),
               "UV": P["V"]["U"][:, :, : int(rv)].float() if int(rv) else None, "mV": P["V"]["mean"].float()}
        print(f"[qsav] joint code over QSA layers {_MX['layers']}: non-rotated keys rank {rk}, values rank {rv}, {m} sparse coordinates (per token-head, over the concatenation)", flush=True)
    return _MX


def _joint(x, U, mean, m):
    """x (B, Hkv, S, L*d) -> mean + U U^T (x - mean) + m exact coordinates."""
    dev = x.device
    U, mean = U.to(dev), mean.to(dev)
    xf = x.float() - mean[None, :, None, :]
    z = torch.einsum("bhsd,hdr->bhsr", xf, U)
    rec = torch.einsum("bhsr,hdr->bhsd", z, U)
    if m:
        res = xf - rec
        idx = res.abs().topk(m, dim=-1).indices
        rec = rec.scatter_add(-1, idx, res.gather(-1, idx))
    return (rec + mean[None, :, None, :]).to(x.dtype)


def apply_joint(states: dict) -> dict:
    """Code the prefix QSA layers' caches jointly (the concatenation over layers of each token's non-rotated key dims and of
    its values, per KV head); rotated key dims and indexer keys untouched.  Tensors are gathered to one device and returned
    to their layers' devices."""
    M = _load_joint()
    layers = [l for l in M["layers"] if l in states]
    if len(layers) != len(M["layers"]):
        return states
    rot = M["rot"]
    dev = states[layers[0]][0].device
    if M["UK"] is not None:
        kn = torch.cat([states[l][0][..., rot:].to(dev) for l in layers], dim=-1)
        kn = _joint(kn, M["UK"], M["mK"], M["m"])
        d = states[layers[0]][0].shape[-1] - rot
        for i, l in enumerate(layers):
            k = states[l][0]
            k2 = torch.cat([k[..., :rot], kn[..., i * d:(i + 1) * d].to(k.device, k.dtype)], dim=-1)
            states[l] = (k2,) + tuple(states[l][1:])
    if M["UV"] is not None:
        v = torch.cat([states[l][1].to(dev) for l in layers], dim=-1)
        v = _joint(v, M["UV"], M["mV"], M["m"])
        d = states[layers[0]][1].shape[-1]
        for i, l in enumerate(layers):
            vv = states[l][1]
            states[l] = (states[l][0], v[..., i * d:(i + 1) * d].to(vv.device, vv.dtype)) + tuple(states[l][2:])
    return states
