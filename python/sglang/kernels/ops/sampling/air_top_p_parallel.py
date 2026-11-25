"""Parallel, bitwise-identical replacement for flashinfer's deterministic AIR top-p renorm.

``air_top_p_renorm_parallel(probs, top_p)`` returns exactly the bits of
``flashinfer.sampling.top_p_renorm_probs(probs, top_p, is_deterministic=True)`` (AIR,
``air_top_p.cuh`` sha256 e41572fb) with many CTAs per row instead of one; see
``csrc/sampling/air_top_p_parallel.cuh`` for why each stage is bitwise identical.

The kernels keep per-row state in a per-device workspace laid out by its row capacity. Its
zero-required regions (per-row global histogram, listCount) are zero when it is allocated and
return to zero at the end of every complete call; row state, seg_table and the kept-candidate list
are written before they are read in each call. A call is CUDA-graph capturable
as long as the workspace already holds the captured row count (``reserve_air_top_p_workspace``);
calls on one device must not overlap across streams (as with flashinfer's own cached workspace).
The fallback-row counter is a separate per-device int64 tensor, never replaced.

``air_top_p_worker_record`` is the observation hook: called inside the process that runs the
kernel (the scheduler serves it in its internal state), it reports the sources this process
imported, the JIT library it loaded and the cumulative counters. ``calls`` / ``total_rows`` and the
bypass counters are host-side, counted where Python runs the call: a CUDA-graph replay of a
captured call would update ``kernel_fallback_rows`` (device) but not them, so their denominators
are not graph-aware (target verify sampling runs eagerly today; capture-time calls do count).
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.jit.utils.arch import get_compile_arch, get_default_target_flags
from sglang.kernels.jit.utils.compile import toolchain
from sglang.kernels.jit.utils.compile.spec import resolve_sources
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Resolved against the in-tree csrc/ directory; an absolute path builds a copy elsewhere.
CUDA_FILE = "sampling/air_top_p_parallel.cuh"
_EXTRA_CUDA_CFLAGS = ["-use_fast_math"]

# Mirrors the checks in `AirTopPParallel::run`.
_MIN_VOCAB = 2048  # flashinfer uses a different (ternary-search) kernel below NUM_BUCKETS
_MAX_VOCAB = 1 << 20
_MAX_ROWS = 65535
_MIN_RESERVED_ROWS = 128

# Per device: (workspace, its row capacity). The kernel lays regions out by the capacity.
_workspaces: Dict[int, Tuple[torch.Tensor, int]] = {}
# Replaced workspaces stay alive: a captured graph may still point at them.
_retired: List[torch.Tensor] = []
# One int64 fallback-row counter per device, never replaced, shared by every workspace and graph.
_fallback_totals: Dict[int, torch.Tensor] = {}
# Per device, created with the fallback counter: its generation id and host [calls, total_rows].
_generations: Dict[int, str] = {}
_host_totals: Dict[int, List[int]] = {}
# Per process: flag on and opted in, but the inputs were rejected (stock kernel used): reason ->
# [calls, rows].
_BYPASS_GENERATION = uuid.uuid4().hex
_bypass_totals: Dict[str, List[int]] = {}


def _sha256(path: Union[str, pathlib.Path]) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# The sources as this process imported them; `_jit_module` fills `_loaded` once, at first load.
_CUH_PATH = resolve_sources([CUDA_FILE])[0]
_SOURCES = {
    "py_path": os.path.abspath(__file__),
    "py_sha256": _sha256(__file__),
    "cuh_path": _CUH_PATH,
    "cuh_sha256": _sha256(_CUH_PATH),
}
_loaded: Dict[str, Any] = {}


def _mapped_library(suffix: str) -> Optional[str]:
    # The path the dynamic loader mapped; /proc reflects a later rename of the cache leaf.
    with open("/proc/self/maps") as maps:
        for line in maps:
            path = "/" + line.rstrip("\n").partition("/")[2]
            if path.endswith(suffix):
                return path
    return None


def _module_identity(cuda_file: str) -> Dict[str, Any]:
    so_path = _mapped_library("sampling_air_top_p_parallel.so")
    nvcc = toolchain.device_compiler_path()
    leaf = pathlib.Path(so_path).parent if so_path else None
    # A cached build is keyed on the source contents, so this is the compiled source's hash.
    cuh_path = resolve_sources([cuda_file])[0]
    arch = get_compile_arch()  # cached: the target load_jit just built for
    return {
        "compile_target": arch.target_name,
        "compile_cuda_flags": get_default_target_flags(arch) + _EXTRA_CUDA_CFLAGS,
        # cache layout <target>/<module>/build-<build_key>/deps-<deps_key>/<module>.so
        "build_key_dir": leaf.parent.name if leaf else None,
        "deps_key_dir": leaf.name if leaf else None,
        "so_path": so_path,
        "so_sha256": _sha256(so_path) if so_path and os.path.isfile(so_path) else None,
        "cuh_path_at_load": cuh_path,
        "cuh_sha256_at_load": _sha256(cuh_path),
        "nvcc_path": nvcc,
        "nvcc_sha256": _sha256(nvcc) if os.path.isfile(nvcc) else None,
        "nvcc_version": subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True
        ).stdout.strip() if os.path.isfile(nvcc) else None,
    }


@cache_once
def _jit_module(cuda_file: str) -> Module:
    required = envs.SGLANG_AIR_TOP_P_REQUIRED_TARGET.get()
    if required and get_compile_arch().target_name != required:
        raise RuntimeError(
            f"air_top_p_parallel: JIT target {get_compile_arch().target_name!r} != required {required!r} "
            "(SGLANG_AIR_TOP_P_REQUIRED_TARGET); refusing to build or run"
        )
    # flashinfer builds AIR with -use_fast_math (gen_jit_spec default); every float op here must
    # lower to the same instruction (ftz adds / muls, approximate 1/x).
    module = load_jit(
        "sampling_air_top_p_parallel",
        cuda_files=[cuda_file],
        cuda_wrappers=[
            ("run", "sglang::air_tier1::AirTopPParallel::run"),
            ("workspace_bytes", "sglang::air_tier1::AirTopPParallel::workspace_bytes"),
        ],
        extra_cuda_cflags=_EXTRA_CUDA_CFLAGS,
    )
    _loaded.update(_module_identity(cuda_file))
    return module


def air_top_p_reject_reason(
    probs: torch.Tensor, top_p: Union[torch.Tensor, float]
) -> Optional[str]:
    """Why ``air_top_p_renorm_parallel`` does not take these inputs; None when it does."""
    if not (probs.is_cuda and probs.dtype == torch.float32 and probs.dim() == 2):
        return "not_cuda_fp32_2d"
    rows, vocab = probs.shape
    if not (_MIN_VOCAB <= vocab < _MAX_VOCAB and vocab % 4 == 0):
        return "vocab"
    if not 0 < rows <= _MAX_ROWS:
        return "rows"
    if not probs.is_contiguous() or probs.data_ptr() % 16 != 0:
        return "layout"
    if isinstance(top_p, torch.Tensor):
        if top_p.device == probs.device and top_p.numel() == rows:
            return None
        return "top_p_tensor"
    if isinstance(top_p, (float, int)) and not isinstance(top_p, bool):
        return None
    return "top_p_type"


def can_air_top_p_parallel(probs: torch.Tensor, top_p: Union[torch.Tensor, float]) -> bool:
    """Whether ``air_top_p_renorm_parallel`` accepts these inputs (else use flashinfer)."""
    return air_top_p_reject_reason(probs, top_p) is None


def air_top_p_note_bypass(reason: str, rows: int) -> None:
    """Count a call the flag and the caller sent here but the inputs sent to the stock kernel."""
    totals = _bypass_totals.setdefault(reason, [0, 0])
    totals[0] += 1
    totals[1] += rows


def _device_index(device: Optional[torch.device]) -> int:
    index = torch.device(device if device is not None else "cuda").index
    return torch.cuda.current_device() if index is None else index


def reserve_air_top_p_workspace(rows: int, device: torch.device) -> Tuple[torch.Tensor, int]:
    """The device's (workspace, row capacity), grown (zeroed) to hold ``rows``; growing is not capturable."""
    index = _device_index(device)
    entry = _workspaces.get(index)
    if entry is not None and entry[1] >= rows:
        return entry
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"air_top_p_renorm_parallel: workspace for {rows} rows must be reserved before "
            "CUDA graph capture (call it eagerly once or reserve_air_top_p_workspace)"
        )
    capacity = max(_MIN_RESERVED_ROWS, 1 << (max(rows, 1) - 1).bit_length())
    ws = torch.zeros(
        int(_jit_module(CUDA_FILE).workspace_bytes(capacity)),
        dtype=torch.uint8,
        device=torch.device("cuda", index),
    )
    if entry is not None:
        _retired.append(entry[0])
    _workspaces[index] = (ws, capacity)
    return _workspaces[index]


def _counters(index: int) -> Tuple[torch.Tensor, List[int]]:
    """The device's (fallback-row counter, host [calls, total_rows])."""
    t = _fallback_totals.get(index)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("air_top_p_renorm_parallel: first call must be eager (fallback counter)")
        t = _fallback_totals[index] = torch.zeros(1, dtype=torch.int64, device=torch.device("cuda", index))
        _generations[index] = uuid.uuid4().hex
        _host_totals[index] = [0, 0]
    return t, _host_totals[index]


def air_top_p_fallback_count(device: Optional[torch.device] = None) -> int:
    """Rows that took the dense fallback in this process on this device (syncs; metrics / tests)."""
    t = _fallback_totals.get(_device_index(device))
    return 0 if t is None else int(t.item())


def air_top_p_worker_record(device_index: int) -> Dict[str, Any]:
    """This process's sources, loaded library and cumulative counters on ``device_index``.

    Never loads, compiles or allocates anything. Once the kernel has run on the device it reads
    the fallback counter with ``.item()`` (no torch.cuda.synchronize): that waits for the reading
    thread's current stream only, so read after the measured work has drained, and per query only
    (each read stalls the calling scheduler loop for that wait).
    """
    record: Dict[str, Any] = {
        "wall_time": time.time(),
        "pid": os.getpid(),
        "device": device_index,
        "gpu_uuid": (
            str(torch.cuda.get_device_properties(device_index).uuid)
            if torch.cuda.is_available()
            else None
        ),
        **_SOURCES,
        "module_status": "LOADED" if _loaded else "NOT_LOADED",
        "module": dict(_loaded) if _loaded else None,
        "bypass_generation": _BYPASS_GENERATION,
        "bypass": {reason: {"calls": c, "rows": r} for reason, (c, r) in _bypass_totals.items()},
    }
    if device_index not in _fallback_totals:
        record.update(counter_generation=None, calls=0, total_rows=0, kernel_fallback_rows=0)
    else:
        calls, total_rows = _host_totals[device_index]
        record.update(
            counter_generation=_generations[device_index],
            calls=calls,
            total_rows=total_rows,
            kernel_fallback_rows=int(_fallback_totals[device_index].item()),
        )
    record["engaged_rows"] = record["total_rows"] - record["kernel_fallback_rows"]
    return record


def air_top_p_renorm_parallel(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    out: Optional[torch.Tensor] = None,
    fallback_rows: Optional[torch.Tensor] = None,
    grid_limit: int = 0,
) -> torch.Tensor:
    """Bitwise ``flashinfer.sampling.top_p_renorm_probs(probs, top_p, is_deterministic=True)``.

    ``probs``: fp32 ``[rows, vocab]``, contiguous, 16 B aligned, ``vocab`` a multiple of 4 in
    ``[2048, 2^20)``. ``top_p``: fp32-convertible tensor ``[rows]`` or a Python float (narrowed
    to fp32 as flashinfer does). ``fallback_rows``: optional uint8 ``[rows]``; set to 1 for rows
    that took the dense per-row fallback (same result, AIR's single-CTA loop). ``grid_limit``
    (tests): caps the persistent pass-0/1 grids; 0 means every CTA the device holds at once.
    """
    rows = probs.shape[0]
    if out is None:
        out = torch.empty_like(probs)
    top_p_arr: Optional[torch.Tensor] = None
    top_p_val = 0.0
    if isinstance(top_p, torch.Tensor):
        # flashinfer: maybe_top_p_arr.float()
        top_p_arr = top_p.float().contiguous().view(-1)
    else:
        top_p_val = float(top_p)
    ws, capacity = reserve_air_top_p_workspace(rows, probs.device)
    counter, totals = _counters(_device_index(probs.device))
    _jit_module(CUDA_FILE).run(
        probs, out, top_p_arr, top_p_val, ws, capacity, counter, fallback_rows, int(grid_limit)
    )
    totals[0] += 1
    totals[1] += rows
    return out


__all__ = [
    "air_top_p_fallback_count",
    "air_top_p_note_bypass",
    "air_top_p_reject_reason",
    "air_top_p_renorm_parallel",
    "air_top_p_worker_record",
    "can_air_top_p_parallel",
    "reserve_air_top_p_workspace",
]
