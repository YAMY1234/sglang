"""Cheaper host launches for hot custom kernels (SGLANG_LAUNCH_FASTPATH; set_enabled switches it at run time).

A prefill forward issues hundreds of launches, and on a pipeline stage whose GPU waits on its host the Python
work around each launch is on the critical path. Two launch kinds get a cached path:
- Triton: ``JITFunction.run`` rebuilds the option string, reads knobs and builds launch metadata on every call.
  Here Triton's own binder still computes the argument specialization each call, so a new specialization gets
  its own kernel, but a known one launches straight through the compiled kernel's launcher.
- CuTe DSL: a compiled function called with the same argument objects every time (e.g. device-side argument
  arrays) has those arguments marshalled once.
Launch hooks (e.g. Proton) set after a Triton kernel was cached are not called for it; turn this off to profile
with them.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Hashable, Tuple

import torch

from sglang.srt.environ import envs

enabled = envs.SGLANG_LAUNCH_FASTPATH.get()


def set_enabled(on: bool) -> None:
    global enabled
    enabled = on


# (id(fn), device, specialization, options) -> (launcher, function handle, packed metadata)
_triton_kernels: Dict[Tuple, Tuple[Callable, Any, Any]] = {}


def triton_launch(fn, grid, *args, **kwargs) -> None:
    """``fn[grid](*args, **kwargs)`` for a ``triton.jit`` function."""
    if not enabled:
        fn[grid](*args, **kwargs)
        return
    device = torch.cuda.current_device()
    # Empty until the slow path has compiled for this device; the binder is the tuple's last item.
    caches = fn.device_caches.get(device)
    if caches is None or fn.pre_run_hooks:
        fn[grid](*args, **kwargs)
        return
    bound_args, specialization, options = caches[4](*args, **kwargs)
    try:
        key = (id(fn), device, tuple(specialization), tuple(options.items()))
        entry = _triton_kernels.get(key)
    except TypeError:  # an unhashable specialization or option value
        fn[grid](*args, **kwargs)
        return
    if callable(grid):
        grid = grid(bound_args)
    if entry is None:
        kernel = fn[grid](*args, **kwargs)
        from triton import knobs

        if kernel is not None and knobs.runtime.launch_enter_hook is None:
            _triton_kernels[key] = (kernel.run, kernel.function, kernel.packed_metadata)
        return
    run, function, packed_metadata = entry
    n = len(grid)
    run(
        grid[0],
        grid[1] if n > 1 else 1,
        grid[2] if n > 2 else 1,
        torch._C._cuda_getCurrentRawStream(device),
        function,
        packed_metadata,
        None,
        None,
        None,
        *bound_args.values(),
    )


# cache key -> (execution args, adapted args, the argument objects they point into)
_cute_args: Dict[Hashable, Tuple[list, list, tuple]] = {}


def cute_static_call(compiled, make_args: Callable[[], tuple], cache_key: Hashable):
    """``compiled(*make_args())`` where every call with this cache_key passes the same argument objects."""
    if not enabled:
        return compiled(*make_args())
    entry = _cute_args.get(cache_key)
    if entry is None:
        args = make_args()
        exe_args, adapted_args = compiled.execution_args.generate_execution_args(
            args, {}
        )
        entry = _cute_args[cache_key] = (exe_args, adapted_args, args)
    executor = compiled._default_executor
    return (executor if executor is not None else compiled).run_compiled_program(
        entry[0]
    )
