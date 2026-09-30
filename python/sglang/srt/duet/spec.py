"""The DuetSpec contract (origin/minma/0913 twinstar/duet/spec.py L21-70), model independent.

A spec names the algorithm's per-checkpoint hyper-parameters and nothing else.  Everything a served model does is read
from it; there are no environment switches for the algorithm.  Unknown keys raise (a misspelled field would otherwise
take its default silently -- spec.py L67-69).
"""

from __future__ import annotations

# Informational reference set, not an allowlist: adapters declare model=.
MODELS = ("flash-next", "lightning", "kimi-linear")
SINKS = ("explicit", "implicit")
Z_FORMATS = ("fp32", "bf16", "fp8", "nvfp4")
VALUE_FORMATS = ("fp32", "bf16", "fp8")
INDEX_FORMATS = ("uint16", "gap8", "packed")
NVFP4_BLOCK = 16

SPEC_FIELDS = (
    "model",
    "prefill_depth",
    "latent_rank",
    "latent_spikes",
    "latent_id_side",
    "latent_z_format",
    "latent_value_format",
    "latent_index_format",
    "state_rank",
    "state_every",
    "state_sink",
    "latent_init",
    "state_init",
    "name",
)
# Fields a release must carry (latent_init / state_init / name may be absent or empty).
REQUIRED_FIELDS = SPEC_FIELDS[:11]


def validate_spec(spec, *, model=None, latent_formats=None, allow_exact_latent=True):
    """Raise on anything a release spec may not be; return the spec unchanged.

    model: when given, the adapter's model name -- a spec for another model is rejected here rather than by a
    shape mismatch deep in the loader.
    """
    if not isinstance(spec, dict):
        raise ValueError("DUET spec must be a JSON object")
    unknown = set(spec) - set(SPEC_FIELDS)
    if unknown:
        raise ValueError(f"unknown DUET spec fields: {sorted(unknown)}")
    missing = [k for k in REQUIRED_FIELDS if k not in spec]
    if missing:
        raise ValueError(f"DUET spec lacks fields: {missing}")
    if not isinstance(spec["model"], str) or not spec["model"].strip():
        raise ValueError("model must be a nonempty adapter-declared name")
    if model is not None and spec["model"] != model:
        raise ValueError(
            f"spec is for {spec['model']!r}; this adapter implements {model!r}"
        )
    for key in (
        "prefill_depth",
        "latent_rank",
        "latent_spikes",
        "state_rank",
        "state_every",
    ):
        if type(spec.get(key)) is not int or spec[key] < 0:
            raise ValueError(f"invalid DUET integer: {key}")
    if spec["prefill_depth"] < 1:
        raise ValueError("prefill_depth must be >= 1")
    if type(spec.get("latent_id_side")) is not bool:
        raise ValueError("latent_id_side must be boolean")
    if spec["state_sink"] not in SINKS:
        raise ValueError(f"state_sink {spec['state_sink']!r}: expected one of {SINKS}")
    if spec["latent_z_format"] not in Z_FORMATS:
        raise ValueError(
            f"latent z format {spec['latent_z_format']!r}: expected one of {Z_FORMATS}"
        )
    if spec["latent_value_format"] not in VALUE_FORMATS:
        raise ValueError(
            f"latent value format {spec['latent_value_format']!r}: expected one of {VALUE_FORMATS}"
        )
    if spec["latent_index_format"] not in INDEX_FORMATS:
        raise ValueError(
            f"latent index format {spec['latent_index_format']!r}: expected one of {INDEX_FORMATS}"
        )
    if spec["latent_z_format"] == "nvfp4" and spec["latent_rank"] % NVFP4_BLOCK:
        raise ValueError(
            f"latent rank {spec['latent_rank']} must be a multiple of {NVFP4_BLOCK} for nvfp4 storage"
        )
    if spec.get("latent_init") or spec.get("state_init"):
        raise ValueError(
            "a release spec must not carry training initializer paths (latent_init / state_init)"
        )
    if not allow_exact_latent and spec["latent_rank"] == 0:
        raise NotImplementedError(
            "uncoded residual checkpoints need an exact latent transport"
        )
    actual_formats = tuple(
        spec[k]
        for k in ("latent_z_format", "latent_value_format", "latent_index_format")
    )
    if latent_formats is not None and actual_formats != tuple(latent_formats):
        raise NotImplementedError(
            f"this adapter's latent transport requires {tuple(latent_formats)}"
        )
    return spec


def describe(spec) -> str:
    """twinstar/duet/spec.py L91-94."""
    if spec["latent_rank"]:
        code = (
            f"code {spec['latent_rank']}+{spec['latent_spikes']}{' +id' if spec['latent_id_side'] else ''} "
            f"({spec['latent_z_format']}/{spec['latent_value_format']}/{spec['latent_index_format']})"
        )
    else:
        code = "exact residual"
    state = (
        f"state {spec['state_sink']} sink + rank {spec['state_rank']} every {spec['state_every']}"
        if spec["state_rank"]
        else "state exact"
    )
    return f"{spec['model']} k={spec['prefill_depth']} {code}; {state}"
