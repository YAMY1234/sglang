"""Verified HF release loader and the ten Lightning memory emitters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .latent import ResidualCode


from ._common import load as _load

_spec = _load("spec")
SPEC_FIELDS = set(_spec.SPEC_FIELDS)
_validate_common_spec = _spec.validate_spec


def validate_spec(spec):
    """Common DuetSpec contract (sglang.srt.duet.spec) plus what this transport does not implement yet."""
    _validate_common_spec(spec, model="lightning")
    if spec["latent_rank"] == 0:
        raise NotImplementedError("uncoded residual checkpoints need an exact latent transport")
    if tuple(spec.get(k) for k in ("latent_z_format", "latent_value_format", "latent_index_format")) != ("nvfp4", "bf16", "gap8"):
        raise NotImplementedError("this transport currently supports scheme C NVFP4/BF16/gap8")


class Geometry:
    """Only base config determines structural dimensions and memory layer IDs."""
    def __init__(self, config, spec):
        self.hidden = config.hidden_size
        self.k = spec["prefill_depth"]
        self.layers = tuple(config.layers_block_type)
        if len(self.layers) != config.num_hidden_layers or not 0 < self.k <= len(self.layers):
            raise ValueError("prefill cut or base layer table is inconsistent")
        if spec["latent_spikes"] > self.hidden:
            raise ValueError("exact coordinate count exceeds residual dimension")
        self.mamba_ids = tuple(i for i, kind in enumerate(self.layers) if kind == "mamba")
        self.mamba_emitters = tuple(i for i in self.mamba_ids if i >= self.k)
        self.attention_emitters = tuple(i for i, kind in enumerate(self.layers) if kind == "attention" and i >= self.k)
        self.heads, self.head_dim = config.mamba_num_heads, config.mamba_head_dim
        self.groups, self.state_dim = config.mamba_n_groups, config.ssm_state_size
        self.intermediate = self.heads * self.head_dim
        self.bc_dim = self.groups * self.state_dim
        self.conv_dim = self.intermediate + 2 * self.bc_dim
        self.conv_width = config.conv_kernel - 1
        self.kv_heads, self.attn_dim = config.num_key_value_heads, config.head_dim
        if self.heads % self.groups or self.intermediate % self.groups or self.conv_width < 1:
            raise ValueError("unsupported Mamba grouped/conv geometry")
        if config.mamba_proj_bias or not config.use_conv_bias or config.mamba_hidden_act != "silu":
            raise NotImplementedError("current emitter tensor contract requires bias-free projection, biased SiLU convolution")


def verify_release(directory):
    root = Path(directory)
    spec = json.loads((root / "spec.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if spec != manifest["provenance"]["spec"]:
        raise ValueError("DUET spec differs from manifest provenance")
    validate_spec(spec)
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


def final_mamba_state(x, dt, a, b, chunk):
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
        self.geometry = g = Geometry(config, self.spec)
        self.mamba_emitters, self.attention_emitters = g.mamba_emitters, g.attention_emitters
        weights = load_file(str(Path(directory) / "duet_components.safetensors"), device="cpu")
        m = self.spec["latent_rank"]
        expected = {
            "latent.code.E": (1, m, g.hidden), "latent.code.D": (1, g.hidden, m),
            "latent.code.mu": (1, g.hidden),
        }
        # The reference StateFactor exports its registered buffer for both
        # sink modes and even when rank=0. Implicit mode must not use it.
        expected["state.sink_dir"] = (len(g.layers), g.heads, g.head_dim)
        for layer in g.mamba_emitters:
            for name, shape in {
                "norm.weight": (g.hidden,),
                "mixer.in_proj.weight": (g.intermediate + g.conv_dim + g.heads, g.hidden),
                "mixer.conv1d.weight": (g.conv_dim, 1, g.conv_width + 1),
                "mixer.conv1d.bias": (g.conv_dim,),
                "mixer.dt_bias": (g.heads,), "mixer.A_log": (g.heads,),
            }.items():
                expected[f"emitters.{layer}.{name}"] = shape
        for layer in g.attention_emitters:
            for name, shape in {"norm.weight": (g.hidden,),
                                "k_proj.weight": (g.kv_heads * g.attn_dim, g.hidden),
                                "v_proj.weight": (g.kv_heads * g.attn_dim, g.hidden)}.items():
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
        self.code = ResidualCode(self.weights["latent.code.E"], self.weights["latent.code.D"], self.weights["latent.code.mu"], self.spec["latent_spikes"], self.spec["latent_id_side"])
        self.directions = self.weights["state.sink_dir"] if self.spec["state_sink"] == "explicit" else torch.zeros_like(self.weights["state.sink_dir"])
        self.eps = config.layer_norm_epsilon
        self.chunk = config.mamba_chunk_size

    def emitter(self, layer, residual):
        g = self.geometry
        prefix = f"emitters.{layer}."
        weight = lambda key: self.weights[prefix + key]
        normalized = rms_norm(residual.float(), weight("norm.weight"), self.eps)
        if layer in self.attention_emitters:
            return {
                "k": F.linear(normalized, weight("k_proj.weight")).reshape(-1, g.kv_heads, g.attn_dim),
                "v": F.linear(normalized, weight("v_proj.weight")).reshape(-1, g.kv_heads, g.attn_dim),
            }
        projected = F.linear(normalized, weight("mixer.in_proj.weight"))
        _, xbc, raw_dt = projected.split([g.intermediate, g.conv_dim, g.heads], -1)
        history = F.pad(xbc.T.unsqueeze(0), (g.conv_width, 0))
        conv = history[0, :, -g.conv_width:].contiguous()
        convolved = F.silu(F.conv1d(history, weight("mixer.conv1d.weight"), weight("mixer.conv1d.bias"), groups=g.conv_dim))[0].T
        x, b, _ = convolved.split([g.intermediate, g.bc_dim, g.bc_dim], -1)
        dt = F.softplus(raw_dt.float() + weight("mixer.dt_bias"))
        state = final_mamba_state(
            x.reshape(1, -1, g.heads, g.head_dim), dt[None], -weight("mixer.A_log").exp(),
            b.reshape(1, -1, g.groups, g.state_dim), self.chunk,
        )[0]
        return {"state": state, "conv": conv}
