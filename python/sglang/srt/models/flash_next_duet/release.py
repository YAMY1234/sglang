"""Flash-Next geometry and explicit sink cache; identity belongs to duet.release."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

from sglang.srt.duet.release import COMPONENT_FILE, build_contract, verify_release


def config_dict(config):
    if isinstance(config, dict):
        return config
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return {k: config_dict(v) if hasattr(v, "__dict__") else v
            for k, v in vars(config).items() if not k.startswith("_duet_")}


class FlashNextGeometry:
    latent_key = "P.latent.core"
    state_key = "P.state.sink_dir"
    emitter_prefix = "P.emitters."

    def __init__(self, config):
        config = config_dict(config)
        self.text = t = config.get("text_config", config)
        self.num_layers = t["num_hidden_layers"]
        self.residual_dim = t["hidden_size"] * t["hc_count"]
        self.state_heads = t["linear_num_value_heads"]
        self.state_side_dim = t["linear_value_head_dim"]

    def memory_kind(self, layer):
        return "state" if self.text["layer_types"][layer] == "linear_attention" else "attention"

    def emitter_tensors(self, layer):
        t = self.text
        h, width, low = t["hidden_size"], self.residual_dim, t["hc_lowrank"]
        result = {"attn_hyper_connection.hc_norm.weight": (width,),
                  "attn_hyper_connection.input_mix_weight_down.weight": (low, width),
                  "attn_hyper_connection.input_mix_weight_up.weight": (width, low)}
        if self.memory_kind(layer) == "state":
            heads = self.state_heads
            conv = (2 * t["linear_num_key_heads"] * t["linear_key_head_dim"]
                    + heads * t["linear_value_head_dim"])
            result.update({"linear_attn.in_proj_qkv.weight": (conv, h),
                           "linear_attn.in_proj_a.weight": (heads, h),
                           "linear_attn.in_proj_b.weight": (heads, h),
                           "linear_attn.conv1d.weight": (conv, 1, t["linear_conv_kernel_dim"]),
                           "linear_attn.A_log": (heads,), "linear_attn.dt_bias": (heads,)})
        else:
            kv = t["num_key_value_heads"] * t["head_dim"]
            index = (t["indexer_n_heads"] + t["indexer_kv_heads"]) * t["indexer_head_dim"]
            result.update({"self_attn.k_proj.weight": (kv, h),
                           "self_attn.v_proj.weight": (kv, h),
                           "self_attn.k_norm.weight": (t["head_dim"],),
                           "self_attn.indexer.index_qk_proj.weight": (index, h)})
        return result

    def inherited_tensors(self, layer):
        if self.memory_kind(layer) != "attention":
            return {}
        name = "self_attn.indexer.k_layernorm.weight"
        return {f"model.language_model.layers.{layer}.{name}":
                (f"P.emitters.{layer}.{name}", (self.text["indexer_head_dim"],))}


def validate_release(directory, hf_config, *, base_model=None):
    geometry = FlashNextGeometry(hf_config)
    identity = verify_release(directory, geometry=geometry, base_model=base_model, model="flash-next")
    build_contract(identity.spec, geometry)
    return identity


def write_sink_vbar(directory, output_file, *, release):
    """Preserve the reference's bf16 round trip and unsharded layer/head order."""
    import torch
    from safetensors import safe_open

    with safe_open(str(Path(directory) / COMPONENT_FILE), framework="pt", device="cpu") as weights:
        sink = weights.get_tensor("P.state.sink_dir")
    sink = sink.to(torch.bfloat16).float()
    torch.save({"vbar": {layer: sink[layer].clone() for layer in range(sink.shape[0])},
                "source": release.name, "sha256": release.sha256}, output_file)
    return sink.shape[0]


def sink_cache(identity):
    """Use an atomic, hash-checked cache; read-only releases use the HF cache root."""
    import torch

    primary = Path(identity.path) / ".cache"
    fallback = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    fallback = fallback / "duet-releases" / ".cache" / identity.sha256
    for directory in (primary, fallback):
        temporary = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            output = directory / "state_sink_vbar.pt"
            if output.is_file():
                cached = torch.load(output, map_location="cpu", weights_only=True)
                if cached.get("sha256") == identity.sha256:
                    return str(output)
            fd, temporary = tempfile.mkstemp(prefix="state_sink_vbar-", suffix=".pt", dir=directory)
            os.close(fd)
            write_sink_vbar(identity.path, temporary, release=identity)
            os.replace(temporary, output)
            return str(output)
        except OSError:
            if directory == fallback:
                raise
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
    raise RuntimeError("unable to cache Flash-Next sink directions")
