"""Kimi extensions for the common DUET resolution and reporting hooks."""

from sglang.srt.duet import options


def required_server_settings(args, spec):
    """Only adapter compatibility requirements; guard scheduling stays in scripts."""
    control = options.DuetOptions.resolve(spec, args)
    if not (control.prefill_layer_trim or control.decode_ssm_r):
        return {}
    backend = getattr(args, "cuda_graph_backend_prefill", None)
    graph = getattr(args, "cuda_graph_config", None)
    if graph is not None:
        graph = graph.to_dict() if hasattr(graph, "to_dict") else graph
        backend = graph.get("prefill", {}).get("backend", backend)
    if backend not in (None, "disabled"):
        raise ValueError("Kimi DUET requires generic prefill graphs disabled; production uses its emitter graph")
    settings = {"disable_radix_cache": True, "disable_prefill_cuda_graph": True}
    if getattr(args, "chunked_prefill_size", None) is None:
        settings["chunked_prefill_size"] = -1
    return settings


def resolve_server_numerics(server_args, spec):
    from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view

    settings = required_server_settings(resolving_view(server_args), spec)
    if settings:
        declare_resolution(server_args, "kimi_linear_duet.resolve_server_numerics", **settings)


def resolved_server_args(args):
    # CPU constructor fixtures are intentionally uninitialized records. Workers
    # always carry completed resolution and must read the declared values.
    if not getattr(args, "_resolution_finished", False):
        return args
    from sglang.srt.arg_groups.overrides import resolved_view

    return resolved_view(args)


def describe_numerics(args, spec):
    from .controls import describe_numerics as describe

    return describe(args, spec=spec)
