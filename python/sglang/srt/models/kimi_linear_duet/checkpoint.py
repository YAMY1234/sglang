"""Kimi geometry and destination mapping for the shared DUET release validator."""
from __future__ import annotations

import json
import os
from pathlib import Path

from sglang.srt.duet import release, spec as common_spec


def validate_spec(spec, config):
    common_spec.validate_spec(spec, model="kimi-linear")
    if spec["model"] != "kimi-linear" or config.get("model_type") != "kimi_linear":
        raise ValueError("Kimi DUET requires the unchanged Kimi Linear base")
    if not config.get("mla_use_nope") or config.get("q_lora_rank") is not None:
        raise ValueError("Kimi DUET requires NoPE MLA without q_lora")
    for key in ("prefill_depth", "latent_rank", "latent_spikes", "state_rank", "state_every"):
        if type(spec[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if not 1 <= spec["prefill_depth"] < config["num_hidden_layers"]:
        raise ValueError("DUET cut must be inside the base layer stack")
    if not 0 <= spec["latent_rank"] <= config["hidden_size"] or not 0 <= spec["latent_spikes"] <= config["hidden_size"]:
        raise ValueError("invalid residual code dimensions")
    if type(spec["latent_id_side"]) is not bool:
        raise ValueError("latent_id_side must be boolean")
    if spec["latent_z_format"] not in ("fp32", "bf16", "fp8", "nvfp4"):
        raise ValueError("unsupported latent z format")
    if spec["latent_z_format"] == "nvfp4" and spec["latent_rank"] % 16:
        raise ValueError("NVFP4 residual rank must be divisible by 16")
    if spec["latent_value_format"] not in ("fp32", "bf16", "fp8") or spec["latent_index_format"] not in ("uint16", "gap8", "packed"):
        raise ValueError("unsupported spike format")
    if spec["state_sink"] not in ("explicit", "implicit") or not 0 <= spec["state_rank"] <= config["linear_attn_config"]["head_dim"] or spec["state_every"] < 0:
        raise ValueError("invalid sink form or content rank/cadence")
    if spec["latent_init"] or spec["state_init"]:
        raise ValueError("released components must carry code and sink tensors, not external init paths")


def tensor_contract(spec, config):
    """Every released tensor -> (destination name, exact unsharded shape).

    Frozen q tensors are deliberately absent. frozen_q_contract lists the two
    tensors inherited from each base KDA layer instead of inventing zero weights.
    """
    validate_spec(spec, config)
    d, n = config["hidden_size"], config["num_hidden_layers"]
    la = config["linear_attn_config"]
    h, dh, kc = la["num_heads"], la["head_dim"], la["short_conv_kernel_size"]
    r, inner = spec["latent_rank"], h * dh
    out = {
        "state.sink_dir": ("state.sink_dir", (n, h, dh)),
    }
    if r:
        out.update({"latent.code.E": ("latent.code.E", (1, r, d)),
                    "latent.code.D": ("latent.code.D", (1, d, r)),
                    "latent.code.mu": ("latent.code.mu", (1, d))})
    kda = {int(l) - 1 for l in la["kda_layers"]}
    for l in range(spec["prefill_depth"], n):
        src, dst = f"emitters.{l}.", f"model.emitters.{l}."
        out[src + "norm.weight"] = (dst + "input_layernorm.weight", (d,))
        if l in kda:
            shapes = {
                "k_proj.weight": (inner, d), "v_proj.weight": (inner, d),
                "k_conv1d.weight": (inner, 1, kc), "v_conv1d.weight": (inner, 1, kc),
                "f_a_proj.weight": (dh, d), "f_b_proj.weight": (inner, dh),
                "b_proj.weight": (h, d), "A_log": (1, 1, h, 1), "dt_bias": (inner,),
            }
            for name, shape in shapes.items():
                out[src + "mixer." + name] = (dst + "self_attn." + name, shape)
        else:
            out[src + "kv_a_proj_with_mqa.weight"] = (
                dst + "self_attn.kv_a_proj_with_mqa.weight",
                (config["kv_lora_rank"] + config["qk_rope_head_dim"], d))
            out[src + "kv_a_layernorm.weight"] = (dst + "self_attn.kv_a_layernorm.weight", (config["kv_lora_rank"],))
    return out


def frozen_q_contract(spec, config):
    la = config["linear_attn_config"]
    inner = la["num_heads"] * la["head_dim"]
    out = {}
    for one_based in la["kda_layers"]:
        l = int(one_based) - 1
        if l < spec["prefill_depth"]:
            continue
        for name, shape in (("q_proj.weight", (inner, config["hidden_size"])),
                            ("q_conv1d.weight", (inner, 1, la["short_conv_kernel_size"]))):
            out[f"model.layers.{l}.self_attn.{name}"] = (f"model.emitters.{l}.self_attn.{name}", shape)
    return out



class KimiGeometry:
    latent_key = "latent.code"
    state_key = "state.sink_dir"
    emitter_prefix = "emitters."

    def __init__(self, config):
        self.config = config
        self.residual_dim = config["hidden_size"]
        self.num_layers = config["num_hidden_layers"]
        la = config["linear_attn_config"]
        self.state_heads, self.state_side_dim = la["num_heads"], la["head_dim"]
        self.kda_layers = {int(l) - 1 for l in la["kda_layers"]}

    def memory_kind(self, layer):
        return "state" if layer in self.kda_layers else "attention"

    def emitter_tensors(self, layer):
        # Shapes come from the same contract used by weight destinations.
        # Public build_contract keeps the wire names, not the model names.
        spec = dict(model="kimi-linear", prefill_depth=layer, latent_rank=0,
                    latent_spikes=0, latent_id_side=True, latent_z_format="fp32",
                    latent_value_format="fp32", latent_index_format="uint16",
                    state_rank=0, state_every=0, state_sink="explicit",
                    latent_init="", state_init="", name="geometry")
        prefix = f"emitters.{layer}."
        return {name[len(prefix):]: shape
                for name, (_, shape) in tensor_contract(spec, self.config).items()
                if name.startswith(prefix)}

    def inherited_tensors(self, layer):
        prefix = f"model.layers.{layer}."
        return {name: value for name, value in frozen_q_contract(
            {"prefill_depth": layer}, self.config).items() if name.startswith(prefix)}


def base_model_name(config, model_path, directory):
    """Validate Hub identity or the basename of a metadata-free local base.

    HF may replace _name_or_path with the local path. Prefer the original
    config.json when it carries a Hub id. Without an owner in base metadata,
    require the exact repository basename; never invent a model owner.
    """
    path = Path(model_path)
    original = config
    if (path / "config.json").is_file():
        original = json.loads((path / "config.json").read_text())
    for value in (original.get("_name_or_path"), original.get("name_or_path"), str(model_path)):
        if value and not Path(value).is_absolute() and len(value.split("/")) == 2:
            return value
    # The manifest supplies only the namespace absent from the local path.
    repo_or_path, revision = release.parse_release_arg(directory)
    root = Path(release.fetch_release(repo_or_path, hf_root=os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")), revision=revision))
    declared = json.loads((root / "manifest.json").read_text()).get("base_model", "")
    if not declared or declared.rsplit("/", 1)[-1] != path.name:
        raise ValueError(f"local base {path.name!r} does not match release base {declared!r}")
    return declared


def verify_release(directory, config, *, model_path):
    identity = release.verify_release(
        directory, geometry=KimiGeometry(config),
        base_model=base_model_name(config, model_path, directory), model="kimi-linear")
    validate_spec(identity.spec, config)
    return identity
