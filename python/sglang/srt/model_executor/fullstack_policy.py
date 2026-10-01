"""Host-side policy for the external Flash-Next shallow-prefill model."""
from __future__ import annotations

import os
from pathlib import Path

K31_RELEASE_NAME = "duet-fn-k31-r4096-u"
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



def initialize_prompt_only_state_cache(pool, cfg=None):
    """Parse once during pool construction, before serving or graph capture."""
    if hasattr(pool, "_prompt_only_state_cache_flag"):
        return
    raw = os.environ.get("SGLANG_GDN_PROMPT_ONLY_STATE_CACHE")
    flag = "1" if raw is None else raw
    if flag not in ("0", "1", "all"):
        raise ValueError("SGLANG_GDN_PROMPT_ONLY_STATE_CACHE must be 0, 1 or all")
    pool._prompt_only_state_cache_flag = flag
    pool._prompt_only_state_cache_source = "unset(default=1)" if raw is None else raw
    pool._generic_prompt_only_state_cache = bool(
        flag != "0" and cfg is not None and cfg.strict_chunk
        and (cfg.factored_prefix or cfg.exact_prefix)
    )


def initialize_checkpoint_policy(model_config, req_to_token_pool, *, speculative_algorithm=None):
    """Finalize model-dependent policy once, before any scheduler work.

    Dense `all` changes decode/publication only. Fullstack keeps its existing
    prefill policy, including when the generic-factor override is zero.
    """
    if req_to_token_pool is None:
        return
    if hasattr(req_to_token_pool, "_checkpoint_policy_reason"):
        return
    factor_pool = getattr(req_to_token_pool, "factored_gdn_pool", None)
    policy_pool = factor_pool if factor_pool is not None else req_to_token_pool
    # FactoredGDNPool parses in its constructor; this also admits custom dense
    # request pools through the common post-allocation startup hook.
    initialize_prompt_only_state_cache(policy_pool, getattr(factor_pool, "cfg", None))
    flag = policy_pool._prompt_only_state_cache_flag
    generic = policy_pool._generic_prompt_only_state_cache
    fullstack = fullstack_enabled(model_config)
    if generic and not fullstack and speculative_algorithm:
        raise ValueError(
            "generic prompt_only_state_cache does not support speculative_algorithm; "
            "disable speculative decoding or set SGLANG_GDN_PROMPT_ONLY_STATE_CACHE=0"
        )
    req_to_token_pool._prefill_prompt_only_state_cache = fullstack or generic
    req_to_token_pool._prompt_only_state_cache = fullstack or generic or flag == "all"
    req_to_token_pool._prompt_only_state_cache_flag = flag
    req_to_token_pool._prompt_only_state_cache_source = policy_pool._prompt_only_state_cache_source
    req_to_token_pool._checkpoint_policy_reason = (
        "fullstack" if fullstack else "explicit-all-control" if flag == "all"
        else "eligible-factor-prefix" if generic else "explicit-zero" if flag == "0"
        else "ineligible-or-dense-default"
    )


def generic_prompt_only_state_cache(req_to_token_pool=None):
    """Read the immutable factor-pool policy; no environment access at decode."""
    pool = getattr(req_to_token_pool, "factored_gdn_pool", None)
    return pool is not None and pool._generic_prompt_only_state_cache


def prompt_only_state_cache(model_config, req_to_token_pool=None):
    """Read the startup-selected decode/publication policy."""
    return req_to_token_pool is not None and req_to_token_pool._prompt_only_state_cache


def prefill_prompt_only_state_cache(model_config, req_to_token_pool=None):
    """Read startup policy; stock `all` preserves its original prefill depth."""
    return req_to_token_pool is not None and req_to_token_pool._prefill_prompt_only_state_cache


def report_checkpoint_policy(model_config, req_to_token_pool, *, speculative_algorithm=None):
    """Initialize and log either effective policy once before graph capture."""
    if not hasattr(req_to_token_pool, "mamba_allocator"):
        return
    initialize_checkpoint_policy(
        model_config, req_to_token_pool, speculative_algorithm=speculative_algorithm
    )
    if getattr(req_to_token_pool, "_checkpoint_policy_reported", False):
        return
    import logging

    logging.getLogger(__name__).info(
        "MAMBA_CHECKPOINT_POLICY prompt_only_state_cache=%s (%s) "
        "p_only_radix=%s prefill_p_only=%s "
        "SGLANG_GDN_PROMPT_ONLY_STATE_CACHE=%s state_slots=%s",
        int(req_to_token_pool._prompt_only_state_cache),
        req_to_token_pool._checkpoint_policy_reason,
        int(req_to_token_pool._prompt_only_state_cache),
        int(req_to_token_pool._prefill_prompt_only_state_cache),
        req_to_token_pool._prompt_only_state_cache_source,
        req_to_token_pool.mamba_allocator.size,
    )
    req_to_token_pool._checkpoint_policy_reported = True


def fullstack_v3_config(model_config):
    if not fullstack_enabled(model_config):
        return None
    fs = fullstack_config(model_config)
    if fs.get("version") not in (2, 3):
        return None
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
    return fs if fs and fs["latent"] == "on" else None


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
    state = fullstack_config(model_config).get("gdn_state")
    if state == "dense":
        return None
    if state == "rank:8":
        value = fullstack_r8_state(radix=radix, disaggregation_mode=disaggregation_mode)
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
