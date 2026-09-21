"""Host-side policy for the external Flash-Next shallow-prefill model."""
from __future__ import annotations

import os

FULLSTACK_R8_STATE = "r=8,m=8,dtype=fp32,ring=16,init_iters=2,async=1,strict_chunk=1"


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


def fullstack_state_config(model_config):
    if not fullstack_enabled(model_config):
        return None
    state = fullstack_config(model_config).get("gdn_state")
    if state == "dense":
        return None
    if state == "rank:8":
        return FULLSTACK_R8_STATE
    raise ValueError(f"unsupported Flash-Next fullstack GDN state: {state!r}")


def prompt_p_extent(req):
    """Only P tokens can donate reusable shallow-prefill state to radix."""
    length = req.extend_range.length
    final = req.extend_range.end >= len(req.origin_input_ids)
    return max(0, length - int(final))
