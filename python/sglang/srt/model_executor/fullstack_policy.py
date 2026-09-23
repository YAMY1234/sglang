"""Host-side policy for the external Flash-Next shallow-prefill model."""
from __future__ import annotations

import os
from pathlib import Path

FULLSTACK_R8_STATE = "r=8,m=8,dtype=fp32,ring=16,init_iters=2,async=1,strict_chunk=1"
FULLSTACK_R8_RADIX_STATE = "r=8,m=8,dtype=fp16,ring=16,init_iters=2,async=1,strict_chunk=1,factored_prefix=1"


def fullstack_r8_state(*, radix, disaggregation_mode="null"):
    if radix:
        return FULLSTACK_R8_RADIX_STATE
    if disaggregation_mode != "null":
        # D disables radix but must receive P's fp16 wire representation.
        return FULLSTACK_R8_STATE.replace("dtype=fp32", "dtype=fp16")
    return FULLSTACK_R8_STATE


def fullstack_config(model_config):
    hf = model_config.hf_config
    text = getattr(hf, "text_config", hf)
    ts = getattr(hf, "twinstar", None) or getattr(text, "twinstar", None)
    return ts.get("fullstack") if ts else None


def fullstack_enabled(model_config):
    # A release config can also be loaded by the stock class for flag-off
    # comparisons; it must not alter that class's scheduler or cache policy.
    if os.environ.get("TWINSTAR_FULLSTACK", "1") == "0":
        return False
    if "twinstar_sgl" not in os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE", "").split(","):
        return False
    return bool(fullstack_config(model_config))


def fullstack_v3_config(model_config):
    if not fullstack_enabled(model_config):
        return None
    fs = fullstack_config(model_config)
    if fs.get("version") not in (2, 3):
        return None
    latent = fs.get("latent")
    if latent not in ("on", "off"):
        raise ValueError("v3 requires an explicit latent on/off choice")
    expected = {"status": "latent-serving-candidate" if latent == "on" else "component-candidate",
                "release_name": "duet-fn-v3-r4096",
                "latent": latent, "latent_id_side": True, "latent_store": "fp8",
                "latent_weight_precision": "bf16-roundtrip-fp32", "latent_rank": 4096,
                "latent_sparse": 512, "latent_payload_bytes": 7176,
                "qsa_code": "off", "gdn_state": "rank:8", "gdn_rank": 8, "gdn_every": 8}
    if fs["version"] == 3:
        expected.update(release_name="duet-fn-v3-r4096-b", latent_store="nvfp4",
                        latent_value_format="bf16", latent_index_format="gap8",
                        latent_payload_bytes=3848, deep_gdn_prefix=True, qad=True)
    for key, value in expected.items():
        if fs.get(key) != value:
            raise ValueError(f"invalid v3 serving policy {key}: {fs.get(key)!r}")
    if latent == "off":
        if any(key in fs for key in ("deep_private_tokens", "materialization_chunk")):
            raise ValueError("v3 latent-off must not reserve private materialization pools")
        return fs
    for key in ("deep_private_tokens", "materialization_chunk"):
        if type(fs.get(key)) is not int or fs[key] <= 0 or fs[key] % 64:
            raise ValueError(f"v3 {key} must be a positive multiple of 64")
    return fs


def fullstack_latent_config(model_config):
    fs = fullstack_v3_config(model_config)
    return fs if fs and fs["latent"] == "on" else None


def fullstack_state_config(model_config, *, radix=False, disaggregation_mode="null"):
    if not fullstack_enabled(model_config):
        return None
    state = fullstack_config(model_config).get("gdn_state")
    if state == "dense":
        return None
    if state == "rank:8":
        value = fullstack_r8_state(radix=radix, disaggregation_mode=disaggregation_mode)
        fs = fullstack_config(model_config)
        method = fs.get("gdn_prefill_truncation", "service-iter")
        if method == "paper-ns8-power2-eigh" and fs.get("version") in (2, 3):
            value += ",init_method=paper"
        elif method != "service-iter":
            raise ValueError("unsupported fullstack prefill truncation algorithm")
        return value
    raise ValueError(f"unsupported Flash-Next fullstack GDN state: {state!r}")


def fullstack_qsa_config(model_config):
    """Resolve the release's QSA choice only for the enabled external model."""
    if not fullstack_enabled(model_config):
        return None
    fs = fullstack_config(model_config)
    mode = fs.get("qsa_code")
    if mode == "off":
        return {"qsa_code_prefix": False, "qsa_code_release": None}
    if mode != "on" or fs.get("generated_token_code_delay") != 256:
        raise ValueError("unsupported Flash-Next QSA code policy")
    fraction = fs.get("qsa_code_exact_fraction", 0.25)
    if not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError("fullstack QSA exact fraction must be in (0,1)")
    return {"qsa_code_prefix": True,
            "qsa_code_release": str(Path(fs["release"]).resolve()),
            "qsa_code_exact_fraction": fraction}


def fullstack_qsa_environment(model_config):
    policy = fullstack_qsa_config(model_config)
    if not policy or not policy["qsa_code_prefix"]:
        return {}
    fs = fullstack_config(model_config)
    fmt = fs.get("qsa_code_spike_format", "original")
    tuned = fs.get("qsa_code_read_tuning", False)
    if fmt not in ("original", "bitmap") or type(tuned) is not bool:
        raise ValueError("unsupported fullstack QSA representation/read configuration")
    return {"SGLANG_QSA_CODE_SPIKE_FORMAT": fmt,
            "SGLANG_QSA_CODE_READ_TUNING": str(int(tuned))}


def prompt_p_extent(req):
    """Only P tokens can donate reusable shallow-prefill state to radix."""
    length = req.extend_range.length
    final = req.extend_range.end >= len(req.origin_input_ids)
    return max(0, length - int(final))
