"""Strict, torch-free contract for the HF Kimi DUET component release.

The component repo is separate from the unchanged base checkpoint. Names and
shapes follow twinstar dd9c7bdbd, twinstar/{kimi/emitters,duet/components}.py.
Nothing here silently substitutes another Hub repo or a different depth.
The Kimi-specific parts (spec and base checks, the tensor contract) are here; reading the safetensors header, the per-tensor
audit, the sha256 and the pinned Hub download are sglang.srt.duet.release's (read_header, audit_metadata, sha256_of,
fetch_release), the functions this module's copies were generalised into.
"""
from __future__ import annotations

import json
from pathlib import Path

from sglang.srt.duet import release as _release

BASE_REPO = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
SPEC_KEYS = {
    "model", "prefill_depth", "latent_rank", "latent_spikes", "latent_id_side",
    "latent_z_format", "latent_value_format", "latent_index_format", "state_rank",
    "state_every", "state_sink", "latent_init", "state_init", "name",
}


def validate_spec(spec, config):
    if set(spec) != SPEC_KEYS:
        raise ValueError(f"DUET spec keys: missing={SPEC_KEYS-set(spec)}, extra={set(spec)-SPEC_KEYS}")
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


def audit_metadata(spec, manifest, header, config):
    """The Kimi base and the spec's provenance, then the common per-tensor audit (names, shapes, fp32, contiguous byte ranges,
    parameter total) against this module's contract."""
    contract = _release.TensorContract(expected=tensor_contract(spec, config), inherited=frozen_q_contract(spec, config))
    if spec != manifest["provenance"]["spec"] or manifest["base_model"] != BASE_REPO:
        raise ValueError("spec/base differs from manifest provenance")
    return _release.audit_metadata(contract, manifest, header)


def verify_release(directory, config):
    root = Path(directory)
    spec = json.loads((root / "spec.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    weights = root / _release.COMPONENT_FILE
    header, offset = _release.read_header(weights)
    report = audit_metadata(spec, manifest, header, config)
    if weights.stat().st_size != offset + report["payload_bytes"] or weights.stat().st_size != manifest["file_bytes"]:
        raise ValueError("component file size mismatch")
    sha = _release.sha256_of(weights)
    if sha != manifest["sha256"]:
        raise ValueError("component SHA256 mismatch")
    report.update(spec=spec, sha256=sha, verified=True)
    return report
