"""`--duet-numerics {production,reference}` (docs/162 §3.2, docs/167 §4 C; DUET-INFORK P1).

A profile is a table of defaults for the per-item switches.  Explicit CLI / environment values of a switch always
win; the profile only fills what the launcher left unset.  `production` is the serving default (lead #1531);
guards pass `--duet-numerics reference` explicitly.  An adapter that has not validated the production profile
refuses to start under it (docs/162 §3.2: unimplemented combinations fail at startup, they are not mapped to
something else) -- until docs/167 P4 lands that is Kimi and Lightning, whose users pass `reference`.
"""
import os

ENV = "SGLANG_DUET_NUMERICS"
PROFILES = ("production", "reference")
DEFAULT_PROFILE = "production"

# switch -> {profile: default}; keys are ServerArgs field names (fields/exec_.py) or adapter-side controls.
SWITCHES = {
    "duet_emitter_precision": {"production": "bf16", "reference": "fp32"},
    "duet_prefix_state": {"production": "factored", "reference": "exact"},
    "duet_state_truncation": {"production": "factored-iter", "reference": "reference-warm"},
}
# Controls that are not ServerArgs fields yet (adapters read them through `controls()`).
CONTROLS = {
    "cuda_graph": {"production": True, "reference": False},
    "batched_decode": {"production": True, "reference": False},
    "mamba_state_dtype": {"production": None, "reference": "float32"},  # Kimi reference: fp32 conv/ssm pools
    "radix_cache": {"production": True, "reference": False},
}


def profile_name(args=None, environ=None):
    """CLI > SGLANG_DUET_NUMERICS > production; invalid names raise."""
    env = os.environ if environ is None else environ
    value = getattr(args, "duet_numerics", None) if args is not None else None
    value = value or env.get(ENV) or DEFAULT_PROFILE
    if value not in PROFILES:
        raise ValueError(f"unknown DUET numerics profile {value!r}; choose one of {PROFILES}")
    return value


def defaults(profile):
    return {name: table[profile] for name, table in SWITCHES.items()}


def controls(profile):
    return {name: table[profile] for name, table in CONTROLS.items()}


def apply_defaults(args, environ=None):
    """Fill unset switch fields of a ServerArgs-like object from the profile; return the profile name.

    Only attributes the object actually has are touched, and only when they are None and no environment value
    exists for them (the common options layer resolves environment values itself).
    """
    env = os.environ if environ is None else environ
    profile = profile_name(args, env)
    for name, value in defaults(profile).items():
        if not hasattr(args, name):
            continue
        if getattr(args, name) is not None:
            continue
        if env.get("SGLANG_" + name.upper()) or env.get("SGLANG_DUET_" + name[len("duet_"):].upper()):
            continue
        setattr(args, name, value)
    return profile


def require_profile(adapter_model, args=None, environ=None, *, production_supported):
    """Adapters call this at construction: refuse an unvalidated profile rather than degrade silently."""
    profile = profile_name(args, environ)
    if profile == "production" and not production_supported:
        raise ValueError(
            f"{adapter_model}: the production numerics profile is not validated on this adapter yet "
            f"(docs/167 P4); start with --duet-numerics reference (or SGLANG_DUET_NUMERICS=reference)"
        )
    return profile


def describe(args=None, environ=None):
    profile = profile_name(args, environ)
    return {"profile": profile, "defaults": defaults(profile), "controls": controls(profile)}
