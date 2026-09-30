"""Model-independent DUET serving options: CLI value > SGLANG_DUET_* environment > checkpoint spec > default.

Moved from models/lightning_duet/options.py and twinstar_sgl/duet_options.py (Kimi line); the two agreed on every
name.  Lead rulings folded in (docs/162 §3.2, #002-4): decode_ssm_r / decode_ssm_w may be 0 (0 = exact state /
prune only at the prompt end, the reference `state_every=0` semantics); a factored-storage capacity limit such as
r + W <= 32 belongs to the adapter that selects factored storage, not here.  Unsupported saving policies fail before
the server acquires a GPU.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import os

POLICIES = ("latent-only", "latent-and-kv", "latent-and-ssm", "kv-and-ssm")
ENV_PREFIX = "SGLANG_DUET_"
IMPLEMENTED_POLICIES = ("kv-and-ssm",)
EMITTER_PRECISIONS = ("fp32", "bf16")
PREFIX_STATES = ("exact", "factored")


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("1", "true", "yes", "on"):
        return True
    if str(value).lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"invalid DUET boolean: {value!r}")


def _named_option(name, choices, default, args=None, environ=None, *, alias=None):
    env = os.environ if environ is None else environ
    value = getattr(args, name, None) if args is not None else None
    if value is None:
        value = env.get("SGLANG_" + name.upper())
    if value is None and alias is not None and alias in env:
        value = "fp32" if boolean(env[alias]) else "bf16"
    value = default if value is None else value
    if value not in choices:
        raise ValueError(f"{name.replace('_', '-')} must be one of {choices}; got {value!r}")
    return value


def resolve_emitter_precision(args=None, environ=None):
    """Flash-Next reference precision by default; bf16 is the production option."""
    return _named_option("duet_emitter_precision", EMITTER_PRECISIONS, "fp32", args, environ,
                         alias="TWINSTAR_EMITTER_FP32")


def resolve_prefix_state(args=None, environ=None):
    """Select an existing prefix checkpoint representation without changing its algorithm."""
    return _named_option("duet_prefix_state", PREFIX_STATES, "exact", args, environ)


def add_arguments(parser):
    """Serving switches for a stand-alone launcher (the fork's ServerArgs declares the same names in
    arg_groups/fields/exec_.py)."""
    from .release import add_release_argument

    add_release_argument(parser)
    # Lightning-line form: accepts --prefill-layer-trim, --prefill-layer-trim=false and --no-prefill-layer-trim.
    parser.add_argument("--prefill-layer-trim", type=boolean, nargs="?", const=True, default=None,
                        help="DUET shallow prefill (default on; SGLANG_DUET_PREFILL_LAYER_TRIM)")
    parser.add_argument("--no-prefill-layer-trim", dest="prefill_layer_trim", action="store_false", default=None)
    parser.add_argument("--prefill-saving-policy", choices=POLICIES, default=None,
                        help="what persists for prefix reuse (default kv-and-ssm; SGLANG_DUET_PREFILL_SAVING_POLICY)")
    parser.add_argument("--decode-ssm-r", type=int, default=None,
                        help="decode content rank; spec state_rank by default; 0 retains the full recurrent state")
    parser.add_argument("--decode-ssm-w", type=int, default=None,
                        help="decode pruning cadence; spec state_every by default; 0 prunes only the prompt-final state")
    parser.add_argument("--duet-emitter-precision", choices=EMITTER_PRECISIONS, default=None,
                        help="Flash-Next emitter: fp32 reference default (including dt_bias), bf16 production; "
                             "SGLANG_DUET_EMITTER_PRECISION; legacy TWINSTAR_EMITTER_FP32 alias")
    parser.add_argument("--duet-prefix-state", choices=PREFIX_STATES, default=None,
                        help="Flash-Next prefix checkpoint: exact dense default or factored; SGLANG_DUET_PREFIX_STATE")


def resolve_release(args=None, environ=None, *, legacy_directory=None):
    """CLI > canonical directory > adapter's one-version directory alias."""
    env = os.environ if environ is None else environ
    cli = getattr(args, "duet_release", None) if args is not None else None
    return cli or env.get("SGLANG_DUET_DIR") or (env.get(legacy_directory) if legacy_directory else None)


def duet_enabled(args=None, environ=None, *, legacy_enabled=None):
    """A canonical release enables DUET, even if the deprecated flag is false.

    Without a canonical release, SGLANG_DUET_ENABLED (then the adapter's legacy
    switch) is supported for one version. Turning everything off requires
    clearing the release option/DIR as well as the compatibility switches.
    """
    env = os.environ if environ is None else environ
    if resolve_release(args, env):
        return True
    value = env.get("SGLANG_DUET_ENABLED")
    if value is None and legacy_enabled:
        value = env.get(legacy_enabled)
    return boolean(value) if value is not None else False


def export_cli_environment(args, environ=None):
    """Publish explicit CLI values before model registration / worker spawning.

    Used by native ServerArgs resolution and compatibility launchers for older
    images. Never exports absent CLI defaults over the caller's environment.
    """
    env = os.environ if environ is None else environ
    mapping = {"duet_release": "SGLANG_DUET_DIR"}
    for name in ("prefill_layer_trim", "prefill_saving_policy", "decode_ssm_r", "decode_ssm_w"):
        mapping[name] = ENV_PREFIX + name.upper()
    for name in ("duet_emitter_precision", "duet_prefix_state"):
        mapping[name] = "SGLANG_" + name.upper()
    for name, key in mapping.items():
        value = getattr(args, name, None)
        if value is not None:
            env[key] = str(value)


@dataclass(frozen=True)
class DuetOptions:
    prefill_layer_trim: bool
    prefill_saving_policy: str
    decode_ssm_r: int
    decode_ssm_w: int

    def environment(self):
        return {ENV_PREFIX + k.upper(): str(int(v)) if isinstance(v, bool) else str(v)
                for k, v in asdict(self).items()}

    def effective_spec(self, spec):
        return dict(spec, state_rank=self.decode_ssm_r, state_every=self.decode_ssm_w)

    def state_dtype_environment(self):
        # Emitter weights and projected states are fp32. With both algorithmic
        # paths off, preserve stock/user dtype selection exactly.
        if self.prefill_layer_trim or self.decode_ssm_r:
            return {"SGLANG_MAMBA_CONV_DTYPE": "float32", "SGLANG_MAMBA_SSM_DTYPE": "float32"}
        return {}

    @classmethod
    def resolve(cls, spec, args=None, environ=None, *, state_dim=None,
                implemented_policies=IMPLEMENTED_POLICIES, unimplemented_message=None):
        env = os.environ if environ is None else environ
        defaults = dict(prefill_layer_trim=True, prefill_saving_policy="kv-and-ssm",
                        decode_ssm_r=spec["state_rank"], decode_ssm_w=spec["state_every"])
        values = {}
        for name, default in defaults.items():
            cli = getattr(args, name, None) if args is not None else None
            values[name] = cli if cli is not None else env.get(ENV_PREFIX + name.upper(), default)
        values["prefill_layer_trim"] = boolean(values["prefill_layer_trim"])
        for key in ("decode_ssm_r", "decode_ssm_w"):
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                raise ValueError(f"{key} must be an integer")
            values[key] = int(value)
        if values["decode_ssm_r"] < 0 or values["decode_ssm_w"] < 0:
            raise ValueError("DUET decode r/W must be nonnegative; zero follows release exact/no-prune semantics")
        if state_dim is not None and values["decode_ssm_r"] > state_dim:
            raise ValueError("invalid DUET state rank/cadence")
        if values["prefill_saving_policy"] not in POLICIES:
            raise ValueError("unknown DUET saving policy")
        if values["prefill_saving_policy"] not in implemented_policies:
            raise NotImplementedError(unimplemented_message or
                                      f"{values['prefill_saving_policy']}: prefix reconstruction is not implemented")
        return cls(**values)


def resolve_options(spec, *, state_dim=None, args=None, environ=None, **kw):
    """Kimi-line entry name (twinstar_sgl/duet_options.py)."""
    return DuetOptions.resolve(spec, args, environ, state_dim=state_dim, **kw)
