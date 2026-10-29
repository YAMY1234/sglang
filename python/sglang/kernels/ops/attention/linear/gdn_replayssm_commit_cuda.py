# SPDX-License-Identifier: Apache-2.0
"""Commit all ReplaySSM GDN layers with one CUDA kernel.

This is a model-independent runtime fast path for the ReplaySSM GDN cache
contract. One CTA handles a layer, request, and value head. It replaces only the
stock compact-commit kernel; the cursor finish stays the Triton kernel. The
runtime calls it only under ``SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION``; this
module additionally requires its BF16 production layout and an SM90-or-newer
CUDA device. Unsupported inputs return to the stock Triton compact commit without
changing state.

The math matches the Triton commit, but its FP32 accumulation order differs. It
is therefore tolerance-equivalent (``rtol=2**-7, atol=1e-6``), not bitwise
equivalent. The BF16 dtype check below is an intentional precision gate. Both
track variants are one ``load_jit`` module, built by one compiler invocation and
cached on disk, where every rank and every restart on the same cache root reuses
it. The scheduler builds it for the pool before graph capture (``prebuild``), so
no verify round compiles; a failed build is a capability rejection that keeps
the Triton commit. A caller that skips the prebuild builds on its first eager
call, and capture takes the Triton commit until the module is loaded.
"""

from __future__ import annotations

import functools
import logging
import weakref
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

CUDA_COMMIT_RTOL = 2**-7
CUDA_COMMIT_ATOL = 1e-6

logger = logging.getLogger(__name__)

# (key heads, value heads) modules loaded or failed in this process.
_LOADED: set[tuple[int, int]] = set()
_BUILD_FAILURES: set[tuple[int, int]] = set()

# ReplaySSM pool tensors are stable for the model lifetime. Cache their static
# layout contract by identity so a per-round commit does not re-walk large shape
# and stride tuples. Weak references let model reload release the large buffers.
_LAYOUT_SUPPORT: dict[
    tuple[int, ...],
    tuple[tuple[weakref.ReferenceType[torch.Tensor] | None, ...], bool],
] = {}


@cache_once
def _module(hk: int, hv: int) -> Module:
    plain = make_cpp_args(hk, hv, False)
    track = make_cpp_args(hk, hv, True)
    return load_jit(
        "gdn_replayssm_commit",
        *make_cpp_args(hk, hv),
        cuda_files=["mamba/gdn_replayssm_commit.cuh"],
        cuda_wrappers=[
            ("commit", f"GdnReplaySSMCommit<{plain}>::run"),
            ("commit_track", f"GdnReplaySSMCommit<{track}>::run"),
        ],
    )


def _load(hk: int, hv: int) -> bool:
    key = (hk, hv)
    if key in _LOADED:
        return True
    if key in _BUILD_FAILURES:
        return False
    try:
        _module(hk, hv)
    except Exception as exc:
        # The build and the load happen before _commit mutates any state, so
        # this layout can safely stay on the stock path. Launch failures are
        # deliberately not caught: retrying stock after a possibly partial
        # device write would be unsafe.
        _BUILD_FAILURES.add(key)
        logger.warning(
            "ReplaySSM CUDA commit build failed; using stock commit: %s", exc
        )
        return False
    _LOADED.add(key)
    return True


def prebuild(
    *,
    checkpoint_state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    d_residual_cache: torch.Tensor | None,
    k_residual_cache: torch.Tensor | None,
) -> bool:
    """Build the kernel for these pool views ahead of graph capture.

    ``False`` means the commit keeps the stock path: the layout is unsupported or
    the build failed, and the failure is remembered.
    """
    if not _static_layout_supported(
        checkpoint_state,
        d_cache,
        k_cache,
        g_cache,
        d_residual_cache,
        k_residual_cache,
    ):
        return False
    return _load(k_cache.shape[2], checkpoint_state.shape[2])


@functools.cache
def _device_supported(device_index: int) -> bool:
    """The kernel's bulk copies and mbarriers need SM90 or newer."""
    try:
        return torch.cuda.get_device_capability(device_index) >= (9, 0)
    except (AssertionError, RuntimeError):
        return False


def _index_vector(tensor: torch.Tensor | None, length: int) -> bool:
    return bool(
        tensor is not None
        and tensor.ndim == 1
        and tensor.shape[0] == length
        and tensor.is_contiguous()
        and tensor.dtype in (torch.int32, torch.int64)
    )


def _static_layout_supported(
    checkpoint_state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    d_residual_cache: torch.Tensor | None,
    k_residual_cache: torch.Tensor | None,
) -> bool:
    tensors = (
        checkpoint_state,
        d_cache,
        k_cache,
        g_cache,
        d_residual_cache,
        k_residual_cache,
    )
    key = tuple(0 if tensor is None else id(tensor) for tensor in tensors)
    cached = _LAYOUT_SUPPORT.get(key)
    if cached is not None and all(
        (old is None and new is None) or (old is not None and old() is new)
        for old, new in zip(cached[0], tensors)
    ):
        return cached[1]

    supported = _check_static_layout(*tensors)
    # A worker normally has one set of views; bound pathological test churn.
    if len(_LAYOUT_SUPPORT) >= 16:
        _LAYOUT_SUPPORT.clear()
    _LAYOUT_SUPPORT[key] = (
        tuple(None if tensor is None else weakref.ref(tensor) for tensor in tensors),
        supported,
    )
    return supported


def _check_static_layout(
    checkpoint_state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    d_residual_cache: torch.Tensor | None,
    k_residual_cache: torch.Tensor | None,
) -> bool:
    """Cold-path dtype/shape/layout precision gate for stable pool views."""
    if not checkpoint_state.is_cuda or checkpoint_state.ndim != 5:
        return False
    device = checkpoint_state.device
    if any(
        tensor is not None and tensor.device != device
        for tensor in (
            d_cache,
            k_cache,
            g_cache,
            d_residual_cache,
            k_residual_cache,
        )
    ):
        return False
    if device.index is None or not _device_supported(device.index):
        return False

    # Precision gate: the CUDA implementation packs BF16 operands and changes
    # FP32 accumulation order. Other state/cache dtypes retain stock numerics.
    if checkpoint_state.dtype != torch.bfloat16:
        return False
    layers, _, hv, value_dim, key_dim = checkpoint_state.shape
    if (value_dim, key_dim) != (128, 128):
        return False
    if checkpoint_state.stride()[2:] != (128 * 128, 128, 1):
        return False
    if d_cache.ndim != 5 or k_cache.ndim != 5 or g_cache.ndim != 4:
        return False
    replay_slots = d_cache.shape[1]
    ring = d_cache.shape[-2]
    h = k_cache.shape[2]
    if (
        d_cache.dtype != torch.bfloat16
        or k_cache.dtype != torch.bfloat16
        or g_cache.dtype != torch.float32
        or ring != 16
        or h <= 0
        or hv % h != 0
        or tuple(d_cache.shape) != (layers, replay_slots, hv, ring, value_dim)
        or tuple(k_cache.shape) != (layers, replay_slots, h, ring, key_dim)
        or tuple(g_cache.shape) != (layers, replay_slots, hv, ring)
        or d_cache.stride()[2:] != (ring * value_dim, value_dim, 1)
        or k_cache.stride()[2:] != (ring * key_dim, key_dim, 1)
        or g_cache.stride()[2:] != (ring, 1)
    ):
        return False

    has_d_residual = d_residual_cache is not None
    has_k_residual = k_residual_cache is not None
    if has_d_residual != has_k_residual:
        return False
    if has_d_residual and (
        d_residual_cache.dtype != torch.bfloat16
        or k_residual_cache.dtype != torch.bfloat16
        or tuple(d_residual_cache.shape) != tuple(d_cache.shape)
        or tuple(k_residual_cache.shape) != tuple(k_cache.shape)
        or d_residual_cache.stride()[2:] != d_cache.stride()[2:]
        or k_residual_cache.stride()[2:] != k_cache.stride()[2:]
    ):
        return False
    return True


def _supported(
    *,
    checkpoint_state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    d_residual_cache: torch.Tensor | None,
    k_residual_cache: torch.Tensor | None,
    state_batch_indices: torch.Tensor,
    replay_indices: torch.Tensor,
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    accept_lens: torch.Tensor,
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    **_: object,
) -> bool:
    """Whether the one-launch kernel can consume this structured cache view.

    Keep this check ahead of the build and the launch: any unsupported dtype, shape,
    layout, index representation, device, or ISA must take the stock commit.
    """
    if not _static_layout_supported(
        checkpoint_state,
        d_cache,
        k_cache,
        g_cache,
        d_residual_cache,
        k_residual_cache,
    ):
        return False
    device = checkpoint_state.device
    tensors = (
        d_cache,
        k_cache,
        g_cache,
        d_residual_cache,
        k_residual_cache,
        state_batch_indices,
        replay_indices,
        write_pos,
        cache_base,
        is_flush,
        accept_lens,
        mamba_track_indices,
        mamba_steps_to_track,
    )
    if any(tensor is not None and tensor.device != device for tensor in tensors):
        return False
    replay_slots = d_cache.shape[1]
    batch = state_batch_indices.shape[0] if state_batch_indices.ndim == 1 else -1
    if batch <= 0 or not all(
        _index_vector(tensor, batch)
        for tensor in (state_batch_indices, replay_indices, accept_lens)
    ):
        return False
    has_track_indices = mamba_track_indices is not None
    has_track_steps = mamba_steps_to_track is not None
    if has_track_indices != has_track_steps:
        return False
    if has_track_indices and not (
        _index_vector(mamba_track_indices, batch)
        and _index_vector(mamba_steps_to_track, batch)
    ):
        return False
    return bool(
        write_pos.ndim == 1
        and write_pos.shape[0] >= replay_slots
        and write_pos.is_contiguous()
        and write_pos.dtype == torch.int32
        and cache_base.ndim == 1
        and cache_base.shape[0] >= replay_slots
        and cache_base.is_contiguous()
        and cache_base.dtype == torch.int32
        and is_flush.ndim == 1
        and is_flush.shape[0] >= replay_slots
        and is_flush.is_contiguous()
        and is_flush.dtype == torch.int8
    )


def try_commit(**kw) -> bool:
    """Commit when every capability gate passes; otherwise request stock fallback."""
    if not _supported(**kw):
        return False
    key = (kw["k_cache"].shape[2], kw["checkpoint_state"].shape[2])
    if key not in _LOADED and (
        torch.cuda.is_current_stream_capturing() or not _load(*key)
    ):
        return False
    _commit(**kw)
    return True


def _commit(
    *,
    checkpoint_state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    d_residual_cache: torch.Tensor | None,
    k_residual_cache: torch.Tensor | None,
    state_batch_indices: torch.Tensor,
    replay_indices: torch.Tensor,
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    accept_lens: torch.Tensor,
    mamba_track_indices: torch.Tensor | None = None,
    mamba_steps_to_track: torch.Tensor | None = None,
    null_block_id: int = -1,
) -> None:
    """The compact-commit part of ``commit_gdn_replayssm_circular`` (not its cursor finish)."""
    has_track = mamba_track_indices is not None and mamba_steps_to_track is not None
    resid = d_residual_cache is not None and k_residual_cache is not None
    module = _module(k_cache.shape[2], checkpoint_state.shape[2])
    (module.commit_track if has_track else module.commit)(
        checkpoint_state,
        d_cache,
        k_cache,
        g_cache,
        d_residual_cache if resid else None,
        k_residual_cache if resid else None,
        state_batch_indices,
        replay_indices,
        write_pos,
        cache_base,
        is_flush,
        accept_lens,
        mamba_track_indices if has_track else None,
        mamba_steps_to_track if has_track else None,
        null_block_id,
    )
