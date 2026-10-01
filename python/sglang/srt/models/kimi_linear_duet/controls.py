"""Kimi-specific profile application before the worker allocates its pools."""

import os
from contextlib import contextmanager

import torch

from sglang.srt.duet import numerics, options


def describe_numerics(args, *, spec):
    """Report implemented Kimi behavior, including the P2 production limits."""
    production = numerics.profile_name(args) == "production"
    control = options.DuetOptions.resolve(spec, args)
    trim = control.prefill_layer_trim
    active = trim or control.decode_ssm_r > 0
    graphs = (not getattr(args, "disable_cuda_graph", False)
              and not getattr(args, "disable_decode_cuda_graph", False)
              and getattr(args, "cuda_graph_backend_decode", None) != "disabled")
    config = getattr(args, "cuda_graph_config", None)
    if config is not None:
        if hasattr(config, "decode"):
            # CudaGraphConfig.to_dict() contains only non-default overrides,
            # so an enabled default backend is absent from that representation.
            graphs = config.decode.backend != "disabled"
        else:
            backend = config.get("decode", {}).get("backend")
            if backend is not None:
                graphs = backend != "disabled"
    return {
        "code_precision": options.resolve_code_precision(args) if trim else "unused",
        "emitter_precision": "fp32" if trim else "unused",
        "emitter_state_only": trim,
        "emitter_cuda_graph": production and trim,
        "prefill_cuda_graph": False,
        "decode_cuda_graph": graphs,
        "async_component_h2d": production and active,
        "batched_decode": getattr(args, "max_running_requests", None) != 1,
        "state_truncation": "reference-warm" if control.decode_ssm_r > 0 else "disabled",
        "state_storage": "dense-inplace",
        "prefix_state": "exact",
        "radix_cache": not getattr(args, "disable_radix_cache", False),
        "mamba_state_dtype": "float32" if active and not production else "native",
        "unimplemented": [
            "bf16 emitter (requires common emitter_runner)",
            "factored-iter state truncation and factor pool",
            "factored prefix restoration and radix reuse",
        ] if production and active else [],
    }


@contextmanager
def code_precision(precision):
    """Limit TF32 to the residual codec; preserve the caller's matmul mode."""
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@contextmanager
def component_upload(profile, device):
    """Upload pinned component tensors on a private stream in production."""
    if profile != "production" or torch.device(device).type != "cuda":
        yield lambda tensor: tensor
        return
    stream = torch.cuda.Stream(device=device)
    caller = torch.cuda.current_stream(device)
    stream.wait_stream(caller)
    pinned = []
    try:
        with torch.cuda.stream(stream):
            def upload(tensor):
                host = tensor.pin_memory()
                pinned.append(host)
                return host.to(device=device, non_blocking=True)
            yield upload
    finally:
        caller.wait_stream(stream)


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
