"""Verified HF release loader and the ten Lightning memory emitters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .latent import ResidualCode


EXPECTED_SPEC = {
    "model": "lightning", "prefill_depth": 33,
    "latent_rank": 2048, "latent_spikes": 128, "latent_id_side": True,
    "latent_z_format": "nvfp4", "latent_value_format": "bf16",
    "latent_index_format": "gap8", "state_rank": 16, "state_every": 16,
    "state_sink": "explicit",
}
MAMBA_EMITTERS = (35, 37, 39, 41, 44, 46, 48, 50)
ATTENTION_EMITTERS = (33, 42)


def verify_release(directory):
    root = Path(directory)
    spec = json.loads((root / "spec.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if spec != manifest["provenance"]["spec"]:
        raise ValueError("DUET spec differs from manifest provenance")
    for key, value in EXPECTED_SPEC.items():
        if spec.get(key) != value:
            raise ValueError(f"unsupported Lightning DUET spec: {key}={spec.get(key)!r}; expected {value!r}")
    extra = set(spec) - set(EXPECTED_SPEC) - {"latent_init", "state_init", "name"}
    if extra or spec.get("latent_init") or spec.get("state_init"):
        raise ValueError("release must contain its components, without extra spec overrides")
    path = root / "duet_components.safetensors"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != manifest["sha256"]:
        raise ValueError("DUET component SHA256 mismatch")
    if path.stat().st_size != manifest["file_bytes"]:
        raise ValueError("DUET component length differs from manifest")
    return spec, manifest


def rms_norm(x, weight, eps):
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return weight * normalized.to(x.dtype)


def base_rms_norm(self, x, residual=None, post_residual_addition=None, quant_linear=None):
    """Reference BF16 residual addition and cast-before-weight norm semantics.

    Stock fused add+norm can compute statistics on an unrounded FP32 residual
    sum and multiply the weight before the BF16 cast. Both rounding points
    differ from the released reference, so the DUET functional path explicitly
    preserves them. Off-mode retains the original native norm implementation.
    """
    if post_residual_addition is not None or quant_linear is not None:
        raise ValueError("Lightning DUET norm supports ordinary BF16 layers only")
    if residual is not None:
        x = x + residual
        residual = x
    out = rms_norm(x, self.weight, self.variance_epsilon)
    return (out, residual) if residual is not None else out


def final_mamba_state(x, dt, a, b, chunk=128):
    """FP32 write-only half of reference ssd_chunked (no C/output read)."""
    batch, length, heads, width = x.shape
    groups, state_dim = b.shape[-2:]
    x, dt, a, b = x.float(), dt.float(), a.float(), b.float()
    pad = (chunk - length % chunk) % chunk
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))
        b = F.pad(b, (0, 0, 0, 0, 0, pad))
    chunks = (length + pad) // chunk
    b = b.repeat_interleave(heads // groups, dim=2)
    xd = (x * dt[..., None]).view(batch, chunks, chunk, heads, width)
    adt = (dt * a).view(batch, chunks, chunk, heads).permute(0, 3, 1, 2)
    cumulative = adt.cumsum(-1)
    bc = b.view(batch, chunks, chunk, heads, state_dim)
    decay = torch.exp(cumulative[..., -1:] - cumulative)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", bc, decay, xd)
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    ends = F.pad(cumulative[..., -1], (1, 0)).cumsum(-1)
    differences = ends[..., :, None] - ends[..., None, :]
    mask = torch.tril(torch.ones(chunks + 1, chunks + 1, device=x.device, dtype=torch.bool))
    decay_chunks = torch.exp(differences.masked_fill(~mask, float("-inf")))
    # Same einsum and full result as the reference, then select the final row.
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunks, states)
    return new_states[:, -1]


class Components:
    def __init__(self, directory, config, device):
        from safetensors.torch import load_file

        self.spec, self.manifest = verify_release(directory)
        if config.hidden_size != 2688 or config.num_hidden_layers != 52:
            raise ValueError("Lightning DUET requires the 52-layer hidden=2688 BF16 base")
        if tuple(i for i, kind in enumerate(config.layers_block_type) if kind == "attention") != (5, 12, 19, 26, 33, 42):
            raise ValueError("base attention layer geometry differs from Lightning checkpoint")
        weights = load_file(str(Path(directory) / "duet_components.safetensors"), device="cpu")
        expected = {
            "latent.code.E": (1, 2048, 2688), "latent.code.D": (1, 2688, 2048),
            "latent.code.mu": (1, 2688), "state.sink_dir": (52, 64, 64),
        }
        for layer in MAMBA_EMITTERS:
            for name, shape in {
                "norm.weight": (2688,), "mixer.in_proj.weight": (10304, 2688),
                "mixer.conv1d.weight": (6144, 1, 4), "mixer.conv1d.bias": (6144,),
                "mixer.dt_bias": (64,), "mixer.A_log": (64,),
            }.items():
                expected[f"emitters.{layer}.{name}"] = shape
        for layer in ATTENTION_EMITTERS:
            for name, shape in {"norm.weight": (2688,), "k_proj.weight": (256, 2688), "v_proj.weight": (256, 2688)}.items():
                expected[f"emitters.{layer}.{name}"] = shape
        if set(weights) != set(expected) or set(weights) != set(self.manifest["tensors"]):
            raise ValueError("DUET release has missing or unexpected used tensors")
        for name, shape in expected.items():
            tensor = weights[name]
            entry = self.manifest["tensors"][name]
            if (tuple(tensor.shape) != shape or tuple(entry["shape"]) != shape
                    or tensor.dtype != torch.float32 or tensor.nbytes != entry["bytes"]):
                raise ValueError(f"invalid DUET tensor: {name}")
        self.weights = {name: tensor.to(device=device, dtype=torch.float32) for name, tensor in weights.items()}
        self.code = ResidualCode(self.weights["latent.code.E"], self.weights["latent.code.D"], self.weights["latent.code.mu"])
        self.directions = self.weights["state.sink_dir"]
        self.eps = config.layer_norm_epsilon
        self.chunk = config.mamba_chunk_size

    def emitter(self, layer, residual):
        prefix = f"emitters.{layer}."
        weight = lambda key: self.weights[prefix + key]
        normalized = rms_norm(residual.float(), weight("norm.weight"), self.eps)
        if layer in ATTENTION_EMITTERS:
            return {
                "k": F.linear(normalized, weight("k_proj.weight")).reshape(-1, 2, 128),
                "v": F.linear(normalized, weight("v_proj.weight")).reshape(-1, 2, 128),
            }
        projected = F.linear(normalized, weight("mixer.in_proj.weight"))
        _, xbc, raw_dt = projected.split([4096, 6144, 64], -1)
        history = F.pad(xbc.T.unsqueeze(0), (3, 0))
        conv = history[0, :, -3:].contiguous()
        convolved = F.silu(F.conv1d(history, weight("mixer.conv1d.weight"), weight("mixer.conv1d.bias"), groups=6144))[0].T
        x, b, _ = convolved.split([4096, 1024, 1024], -1)
        dt = F.softplus(raw_dt.float() + weight("mixer.dt_bias"))
        state = final_mamba_state(
            x.reshape(1, -1, 64, 64), dt[None], -weight("mixer.A_log").exp(),
            b.reshape(1, -1, 8, 128), self.chunk,
        )[0]
        return {"state": state, "conv": conv}
