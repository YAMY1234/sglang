"""Host-side policy for the external Flash-Next shallow-prefill model."""
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
            spec = importlib.util.spec_from_file_location(package, path / "__init__.py", submodule_search_locations=[str(path)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[package] = module
            spec.loader.exec_module(module)
    return importlib.import_module(key)


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
    # Canonical release presence is the master switch. An explicit historical
    # hf_config.twinstar view remains a one-version compatibility entry.
    fs = fullstack_config(model_config)
    return bool(fs and (os.environ.get("SGLANG_DUET_DIR") or fs.get("release")))


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
    resolve_prefix_state = _duet_options().resolve_prefix_state
    from types import SimpleNamespace
    prefix_state = resolve_prefix_state(SimpleNamespace(duet_prefix_state=fs.get("duet_prefix_state")))
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
    precision = "fp32" if reference or prefix_state == "exact" or not radix else "fp16"
    return (f"r={r},m={w},dtype={precision},ring=16,async={int(not reference)},"
            f"strict_chunk=1,init_method=k31,decode_method={'warm' if reference else 'iter'},vbar={path}"
            + (f",{prefix_state}_prefix=1" if radix else ""))


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
