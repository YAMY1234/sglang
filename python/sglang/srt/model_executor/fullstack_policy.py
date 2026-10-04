"""Host-side policy for the in-tree Flash-Next shallow-prefill adapter."""

from __future__ import annotations

import os
from pathlib import Path


def _duet_options():
    """The common DUET options module (sglang.srt.duet.options).

    This file is also loaded by file path from CPU tests on boxes without the sglang runtime; when `sglang.srt`
    is not initialised the light `sglang.srt.duet` package is registered by path (one convention with
    sglang.srt.duet.release._sibling and models/lightning_duet/_common.load) and the module imported normally.
    """
    import importlib
    import importlib.util
    import sys

    key = "sglang.srt.duet.options"
    if key in sys.modules:
        return sys.modules[key]
    if "sglang.srt" not in sys.modules:
        package = "sglang.srt.duet"
        if package not in sys.modules:
            path = Path(__file__).resolve().parents[1] / "duet"
            spec = importlib.util.spec_from_file_location(
                package, path / "__init__.py", submodule_search_locations=[str(path)]
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[package] = module
            spec.loader.exec_module(module)
    return importlib.import_module(key)


K31_RELEASE_NAME = "duet-fn-k31-r4096-u"
FULLSTACK_R8_STATE = "r=8,m=8,dtype=fp32,ring=16,init_iters=2,async=1,strict_chunk=1"
FULLSTACK_R8_RADIX_STATE = (
    "r=8,m=8,dtype=fp16,ring=16,init_iters=2,async=1,strict_chunk=1,factored_prefix=1"
)


def factored_batch_layers_enabled(cfg):
    """Keep rank-16 k31 functional admission on the per-layer expiry kernel.

    Strict prefix continuation is independent of the grouped LU optimization.
    Existing r8 and non-k31 policies retain their prior automatic selection.
    """
    if (
        cfg is None
        or getattr(cfg, "decode_method", "iter") == "warm"
        or cfg.r not in (8, 16)
    ):
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
    if (
        os.environ.get("SGLANG_DUET_DIR")
        and getattr(hf, "architectures", []) == ["Qwen4ExpForConditionalGeneration"]
        and getattr(hf, "_duet_identity", None) is None
    ):
        from sglang.srt.models.flash_next_duet.config import install_config

        install_config(hf)
    text = getattr(hf, "text_config", hf)
    ts = getattr(hf, "twinstar", None) or getattr(text, "twinstar", None)
    return ts.get("fullstack") if ts else None


def fullstack_enabled(model_config):
    # Canonical release presence is the master switch. An explicit historical
    # hf_config.twinstar view remains a one-version compatibility entry.
    fs = fullstack_config(model_config)
    return bool(fs and (os.environ.get("SGLANG_DUET_DIR") or fs.get("release")))



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
    if "duet_spec" in fs:
        from .duet_policy import validate_duet_config

        return validate_duet_config(fs)
    raise ValueError("Flash-Next serving requires a spec-driven DUET release view")


def fullstack_latent_config(model_config):
    fs = fullstack_v3_config(model_config)
    return (
        fs
        if fs
        and fs["latent"] == "on"
        and fs.get("prefill_saving_policy", "latent-and-ssm") != "kv-and-ssm"
        else None
    )


def validate_dense_state_ablation_dtype(model_config, ssm_dtype):
    if (
        fullstack_enabled(model_config)
        and os.environ.get("SGLANG_FLASHNEXT_DENSE_STATE_ABLATION", "0") == "1"
    ):
        fs = fullstack_v3_config(model_config)
        # dense-bf16 = the #624 control; dense-stock = #873 arm 2 (layer cut + emitters, the stock fp32 state path)
        if fs.get("state_ablation") == "dense-bf16" and ssm_dtype != "bfloat16":
            raise ValueError("dense state ablation requires --mamba-ssm-dtype bfloat16")


def fullstack_state_config(model_config, *, radix=False, disaggregation_mode="null"):
    if not fullstack_enabled(model_config):
        return None
    fs = fullstack_config(model_config)
    resolve_prefix_state = _duet_options().resolve_prefix_state
    from types import SimpleNamespace

    prefix_state = resolve_prefix_state(
        SimpleNamespace(duet_prefix_state=fs.get("duet_prefix_state"))
    )
    fullstack_v3_config(model_config)
    r, w = fs["gdn_rank"], fs["gdn_every"]
    # r=0: untouched stock dense recurrence. W=0: dense recurrence with a
    # single prompt-end projection; neither allocates a factored decode pool.
    if r == 0 or w == 0:
        return None
    if r + w > 32:
        raise NotImplementedError("factored recurrence supports r + W <= 32")
    path = fs.get("state_sink_vbar")
    if not path or not Path(path).is_file():
        raise ValueError("explicit state sink requires state_sink_vbar")
    reference = fs.get("duet_state_truncation", "reference-warm") == "reference-warm"
    # A D worker disables radix but receives the same factor tensors as P.
    # Cache metadata stays role-local; numerical/wire precision must not change.
    wire_factors = disaggregation_mode in ("prefill", "decode")
    precision = (
        "fp32"
        if reference or prefix_state == "exact" or not (radix or wire_factors)
        else "fp16"
    )
    return (
        f"r={r},m={w},dtype={precision},ring=16,async={int(not reference)},"
        f"strict_chunk=1,init_method=k31,decode_method={'warm' if reference else 'iter'},vbar={path}"
        + (f",{prefix_state}_prefix=1" if radix else "")
    )


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
    return {
        "qsa_code_prefix": True,
        "qsa_code_release": str(Path(fs["release"]).resolve()),
        "qsa_code_exact_fraction": fraction,
    }


def fullstack_qsa_environment(model_config):
    policy = fullstack_qsa_config(model_config)
    if not policy or not policy["qsa_code_prefix"]:
        return {}
    fs = fullstack_config(model_config)
    fmt = fs.get("qsa_code_spike_format", "original")
    tuned = fs.get("qsa_code_read_tuning", False)
    if fmt not in ("original", "bitmap") or type(tuned) is not bool:
        raise ValueError("unsupported fullstack QSA representation/read configuration")
    return {
        "SGLANG_QSA_CODE_SPIKE_FORMAT": fmt,
        "SGLANG_QSA_CODE_READ_TUNING": str(int(tuned)),
    }


def prompt_p_extent(req):
    """Only P tokens can donate reusable shallow-prefill state to radix."""
    length = req.extend_range.length
    if getattr(req, "_pfactor_agg_contract", False):
        # The opt-in full-N collector publishes S_N without a boundary step.
