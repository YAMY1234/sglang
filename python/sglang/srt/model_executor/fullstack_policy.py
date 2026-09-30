"""Host-side policy for the external Flash-Next shallow-prefill model."""
from __future__ import annotations

import os
from pathlib import Path

K31_RELEASE_NAME = "duet-fn-k31-r4096-u"
FULLSTACK_R8_STATE = "r=8,m=8,dtype=fp32,ring=16,init_iters=2,async=1,strict_chunk=1"
FULLSTACK_R8_RADIX_STATE = "r=8,m=8,dtype=fp16,ring=16,init_iters=2,async=1,strict_chunk=1,factored_prefix=1"


def factored_batch_layers_enabled(cfg):
    """Keep rank-16 k31 functional admission on the per-layer expiry kernel.

    Strict prefix continuation is independent of the grouped LU optimization.
    Existing r8 and non-k31 policies retain their prior automatic selection.
    """
    if cfg is None or getattr(cfg, "decode_method", "iter") == "warm" or cfg.r not in (8, 16):
        return False
    requested = os.environ.get("SGLANG_GDN_FACTORED_BATCH_LAYERS")
    if requested is not None:
        if requested not in ("0", "1"):
            raise ValueError("SGLANG_GDN_FACTORED_BATCH_LAYERS must be 0 or 1")
        return requested == "1"
    return bool(cfg.strict_chunk) and not (cfg.r == 16 and cfg.init_method == "k31")


def fullstack_r8_state(*, radix, disaggregation_mode="null", prefix_state="factored"):
    if radix:
        if prefix_state == "exact":
            return FULLSTACK_R8_STATE + ",exact_prefix=1"
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
    if "duet_spec" in fs:
        from .duet_policy import validate_duet_config
        return validate_duet_config(fs)
    allocation = fs.get("deep_private_allocation", "fixed")
    if allocation not in ("fixed", "shared-arena") or (allocation == "shared-arena" and fs["version"] != 3):
        raise ValueError("unsupported deep private allocation policy")
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
        if fs.get("latent_compute_precision", "fp32") not in ("fp32", "tf32", "bf16"):
            raise ValueError("unsupported final E/D compute precision")
        expected.update(release_name="duet-fn-v3-r4096-b", latent_store="nvfp4",
                        latent_value_format="bf16", latent_index_format="gap8",
                        latent_payload_bytes=3848, deep_gdn_prefix=True, qad=True)
        if fs.get("release_name") == K31_RELEASE_NAME:
            # #873: Mingyuan's final release (spec.json): no per-token RMS in the code (3,844 B nominal; our wire keeps
            # a constant rms field), explicit sink + rank-8 content with his prompt-final truncation.
            expected.update(release_name=K31_RELEASE_NAME, latent_payload_bytes=3844, latent_rms=False)
            if fs.get("gdn_state") != "dense":
                expected.update(state_sink="explicit", gdn_prefill_truncation="k31-warm-subspace")
    # #624/#626: explicit P31+emitter control with the ordinary dense pool.
    # Keep the released r8 policy strict unless BOTH the process opt-in and
    # the independent ablation config declare this control arm.
    dense_ablation = os.environ.get("SGLANG_FLASHNEXT_DENSE_STATE_ABLATION", "0")
    if dense_ablation not in ("0", "1"):
        raise ValueError("SGLANG_FLASHNEXT_DENSE_STATE_ABLATION must be 0 or 1")
    if dense_ablation == "1":
        if (fs["version"] != 3 or latent != "off"
                or fs.get("state_ablation") not in ("dense-bf16", "dense-stock")
                or (fs.get("state_ablation") == "dense-stock" and fs.get("release_name") != K31_RELEASE_NAME)
                or fs.get("gdn_state") != "dense"):
            raise ValueError("dense state ablation requires explicit v3 latent-off dense-bf16 (or k31 dense-stock) config")
        expected.update(gdn_state="dense", gdn_rank=0, gdn_every=0)
    elif fs.get("state_ablation") is not None:
        raise ValueError("state ablation config requires its explicit process opt-in")
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
    return fs if fs and fs["latent"] == "on" and fs.get("prefill_saving_policy", "latent-and-ssm") != "kv-and-ssm" else None


def validate_dense_state_ablation_dtype(model_config, ssm_dtype):
    if (fullstack_enabled(model_config)
            and os.environ.get("SGLANG_FLASHNEXT_DENSE_STATE_ABLATION", "0") == "1"):
        fs = fullstack_v3_config(model_config)
        # dense-bf16 = the #624 control; dense-stock = #873 arm 2 (layer cut + emitters, the stock fp32 state path)
        if fs.get("state_ablation") == "dense-bf16" and ssm_dtype != "bfloat16":
            raise ValueError("dense state ablation requires --mamba-ssm-dtype bfloat16")


def fullstack_state_config(model_config, *, radix=False, disaggregation_mode="null"):
    if not fullstack_enabled(model_config):
        return None
    fs = fullstack_config(model_config)
    from sglang.srt.duet.options import resolve_prefix_state
    from types import SimpleNamespace
    prefix_state = resolve_prefix_state(SimpleNamespace(duet_prefix_state=fs.get("duet_prefix_state")))
    if "duet_spec" in fs:
        fullstack_v3_config(model_config)
        path = fs.get("state_sink_vbar")
        if not path or not Path(path).is_file():
            raise ValueError("explicit state sink requires state_sink_vbar")
        # Accuracy first: full-precision factors, reference warm projection and
        # Exact dense P checkpoints are the default; factored selects the existing compact snapshot.
        return (f"r={fs['gdn_rank']},m={fs['gdn_every']},dtype=fp32,ring=16,async=0,"
                f"strict_chunk=1,init_method=k31,decode_method=warm,vbar={path}"
                + (f",{prefix_state}_prefix=1" if radix else ""))
    state = fullstack_config(model_config).get("gdn_state")
    if state == "dense":
        return None
    if state == "rank:8":
        value = fullstack_r8_state(radix=radix, disaggregation_mode=disaggregation_mode,
                                  prefix_state=prefix_state)
        fs = fullstack_config(model_config)
        method = fs.get("gdn_prefill_truncation", "service-iter")
        if method == "paper-ns8-power2-eigh" and fs.get("version") in (2, 3):
            value += ",init_method=paper"
        elif method == "k31-warm-subspace" and fs.get("version") == 3:
            value += ",init_method=k31"  # #873: k31-r4096-u prompt-final truncation
        elif method != "service-iter":
            raise ValueError("unsupported fullstack prefill truncation algorithm")
        sink = fs.get("state_sink", "implicit")
        if sink == "explicit":
            # #873: the release's per-head sink directions (P.state.sink_dir) exported next to the view
            path = fs.get("state_sink_vbar")
            if not path or not Path(path).is_file():
                raise ValueError("explicit state sink requires the view's state_sink_vbar file")
            value += f",vbar={path}"
        elif sink != "implicit":
            raise ValueError(f"unsupported fullstack state sink {sink!r}")
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
