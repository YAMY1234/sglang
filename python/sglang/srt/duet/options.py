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


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("1", "true", "yes", "on"):
        return True
    if str(value).lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"invalid DUET boolean: {value!r}")


def add_arguments(parser):
    """The four serving switches for a stand-alone launcher (the fork's ServerArgs declares the same names in
    arg_groups/fields/exec_.py)."""
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
