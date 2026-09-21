"""Host-side policy for the external Flash-Next shallow-prefill model."""
from __future__ import annotations

import os


def fullstack_enabled(model_config):
    # A release config can also be loaded by the stock class for flag-off
    # comparisons; it must not alter that class's scheduler or cache policy.
    if os.environ.get("TWINSTAR_FULLSTACK", "1") == "0":
        return False
    if "twinstar_sgl" not in os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE", "").split(","):
        return False
    hf = model_config.hf_config
    text = getattr(hf, "text_config", hf)
    ts = getattr(hf, "twinstar", None) or getattr(text, "twinstar", None)
    return bool(ts and ts.get("fullstack"))


def prompt_p_extent(req):
    """Only P tokens can donate reusable shallow-prefill state to radix."""
    length = req.extend_range.length
    final = req.extend_range.end >= len(req.origin_input_ids)
    return max(0, length - int(final))
