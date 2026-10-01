"""Kimi-specific profile application before the worker allocates its pools."""

import os

from sglang.srt.duet import numerics


def reject_legacy_overrides(environ=None):
    env = os.environ if environ is None else environ
    if any(env.get(k) not in (None, "", "0") for k in (
        "SGLANG_KDA_STATE_PRUNE_RANK", "SGLANG_KDA_STATE_PRUNE_CALIB", "LATENT_OFF"
    )):
        raise ValueError("remove legacy/diagnostic state and latent overrides for spec-driven DUET")


def configure_state_dtype(args, options, *, config=None, pool=None, environ=None):
    """The model loader runs before KVCacheConfigurator creates the Mamba pool.

    KimiLinearConfig.mamba2_cache_params is a property, and its dtype factory
    reads the environment on each access. Reject cached params or a live pool
    rather than silently changing only future reads. All-off retains stock.
    """
    if not (options.prefill_layer_trim or options.decode_ssm_r):
        return
    dtype = numerics.controls(numerics.profile_name(args))["mamba_state_dtype"]
    if dtype is None:
        return
    if pool is not None or (config is not None and "mamba2_cache_params" in vars(config)):
        raise RuntimeError("Kimi DUET state dtype must be selected before pool creation")
    env = os.environ if environ is None else environ
    env["SGLANG_MAMBA_CONV_DTYPE"] = dtype
    env["SGLANG_MAMBA_SSM_DTYPE"] = dtype
