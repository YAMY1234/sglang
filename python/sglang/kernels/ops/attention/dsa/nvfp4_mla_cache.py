"""NVFP4 DSA main-cache gather/dequantization.

The persistent cache stores each token's 576-element MLA row as one packed
uint8 row: 288 bytes of E2M1 pairs (low nibble first), then one E4M3 scale per
block of 16 values (36 bytes). TRTLLM-GEN sparse MLA consumes FP8, so the
generation path uses a persistent CUDA kernel adapted from TensorRT-LLM to
gather and convert selected rows into a compact FP8 cache. The context path
deduplicates the selected slots first, then gathers each distinct row once with
the same kernel. Every path computes e2m1 * (block_scale * global_scale) in FP32
and rounds to E4M3 once, so a stored row reads back identically everywhere.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)
_cuda_gather_failed = False
_logged_gather_paths: set[str] = set()
# Workaround for flashinfer-ai/flashinfer#6484: TRTLLM-GEN sparse MLA reads ~256 rows
# from its cache start whatever the cache size (B200); doubled, as SM10.7 is unmeasured.
_TRTLLM_GEN_MIN_KV_ROWS = 512


def _log_gather_path_once(path: str) -> None:
    if path not in _logged_gather_paths:
        _logged_gather_paths.add(path)
        logger.info("NVFP4 DSA KV cache: %s gather of packed rows", path)


def nvfp4_mla_row_bytes(head_dim: int) -> int:
    """Bytes of one packed row: E2M1 pairs, then one E4M3 scale per 16 values."""
    return head_dim // 2 + head_dim // 16


def _split_packed_rows(
    kv_cache: torch.Tensor, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strided views of the E2M1 bytes and the E4M3 scales of packed rows."""
    if (
        kv_cache.dtype != torch.uint8
        or kv_cache.ndim != 3
        or kv_cache.shape[1] != 1
        or kv_cache.shape[2] != nvfp4_mla_row_bytes(head_dim)
        or kv_cache.stride(2) != 1
    ):
        raise ValueError(
            f"expected packed NVFP4 rows [slots, 1, {nvfp4_mla_row_bytes(head_dim)}] "
            f"uint8, got shape={tuple(kv_cache.shape)} dtype={kv_cache.dtype}"
        )
    data_bytes = head_dim // 2
    return kv_cache[..., :data_bytes], kv_cache[..., data_bytes:].view(
        torch.float8_e4m3fn
    )


@cache_once
def _jit_nvfp4_mla_gather_module() -> Module:
    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("NVFP4 MLA CUDA gather requires SM10x")
    return load_jit(
        "nvfp4_mla_cache_gather_packed_sm10x",
        cuda_files=["dsa/nvfp4_mla_cache_gather.cuh"],
        cuda_wrappers=[("gather", "nvfp4_mla::GatherKernel::run")],
        extra_cuda_cflags=["-O3", "-DNDEBUG"],
    )


def _cuda_gather_module() -> Module | None:
    """The JIT gather, or None once building or loading it has failed.

    Only the build is guarded; launch errors propagate. Neither the JIT nor a
    cold Triton compile may run inside CUDA Graph capture, so a build failure
    there raises instead of switching paths.
    """
    global _cuda_gather_failed

    if _cuda_gather_failed:
        return None
    try:
        return _jit_nvfp4_mla_gather_module()
    except Exception as error:  # noqa: BLE001
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "The NVFP4 MLA CUDA gather failed to build during CUDA Graph "
                "capture; call preload_nvfp4_mla_gather before capturing."
            ) from error
        _cuda_gather_failed = True
        logger.warning(
            "Using the Triton NVFP4 MLA gather; the CUDA kernel failed to build: %s",
            error,
        )
        return None


def preload_nvfp4_mla_gather(device: torch.device, head_dim: int) -> None:
    """Build the generation gather before CUDA Graph capture.

    A failed JIT build warms the Triton fallback here instead.
    """
    if head_dim == 576 and _cuda_gather_module() is not None:
        _log_gather_path_once("CUDA")
        return
    page_size = 64
    gather_dequant_nvfp4_mla_cache_generation(
        torch.zeros(
            (page_size, 1, nvfp4_mla_row_bytes(head_dim)),
            dtype=torch.uint8,
            device=device,
        ),
        torch.zeros((1, 4), dtype=torch.int32, device=device),
        torch.ones(1, dtype=torch.float32, device=device),
        head_dim=head_dim,
        page_size=page_size,
    )


@triton.jit
def _gather_dequant_nvfp4_mla_generation_kernel(
    data_ptr,
    scale_ptr,
    physical_indices_ptr,
    output_ptr,
    compact_indices_ptr,
    global_scale_ptr,
    data_row_stride: tl.constexpr,
    scale_row_stride: tl.constexpr,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
):
    row = tl.program_id(0)
    dim_block = tl.program_id(1)
    # int64: slot * 324 passes 2**31 at slot 6,628,035, inside a decode pool.
    physical_row = tl.load(physical_indices_ptr + row).to(tl.int64)
    row_valid = physical_row >= 0

    offsets = dim_block * block_dim + tl.arange(0, block_dim)
    mask = row_valid & (offsets < head_dim)
    packed = tl.load(
        data_ptr + physical_row * data_row_stride + offsets // 2,
        mask=mask,
        other=0,
    ).to(tl.uint8)
    code = tl.where((offsets & 1) == 0, packed & 0xF, packed >> 4)
    magnitude_code = code & 0x7
    # E2M1 magnitudes for codes 0..7: 0, .5, 1, 1.5, 2, 3, 4, 6.
    value = tl.where(
        magnitude_code == 0,
        0.0,
        tl.where(
            magnitude_code == 1,
            0.5,
            tl.where(
                magnitude_code == 2,
                1.0,
                tl.where(
                    magnitude_code == 3,
                    1.5,
                    tl.where(
                        magnitude_code == 4,
                        2.0,
                        tl.where(
                            magnitude_code == 5,
                            3.0,
                            tl.where(magnitude_code == 6, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    value = tl.where((code & 0x8) != 0, -value, value)
    block_scale = tl.load(
        scale_ptr + physical_row * scale_row_stride + offsets // 16,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    tl.store(
        output_ptr + row.to(tl.int64) * head_dim + offsets,
        # Same FP32 order as the CUDA kernel: e2m1 * (block_scale * global).
        value * (block_scale * global_scale),
        mask=offsets < head_dim,
    )

    tl.store(
        compact_indices_ptr + row,
        tl.where(row_valid, row, -1),
        mask=dim_block == 0,
    )


def gather_dequant_nvfp4_mla_cache_generation(
    kv_cache: torch.Tensor,
    physical_indices: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    head_dim: int,
    page_size: int,
    output: torch.Tensor | None = None,
    compact_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather/dequantize generation rows without cross-query deduplication.

    ``output`` and ``compact_indices`` allow the attention backend to reuse
    stable scratch allocations across layers and CUDA Graph replays. The CUDA
    implementation is the optimized persistent, asynchronously prefetched
    TensorRT-LLM algorithm; the Triton kernel remains as a portability fallback.
    """
    if physical_indices.ndim != 2 or physical_indices.dtype != torch.int32:
        raise ValueError("physical_indices must be contiguous int32 [rows, topk]")
    physical_indices = physical_indices.contiguous()
    data_cache, scale_cache = _split_packed_rows(kv_cache, head_dim)
    total_rows = physical_indices.numel()
    if total_rows > torch.iinfo(torch.int32).max:
        raise ValueError("NVFP4 MLA gather exceeds int32 compact indexing")
    padded_rows = ((total_rows + page_size - 1) // page_size) * page_size
    if output is None:
        output = torch.empty(
            (max(padded_rows, _TRTLLM_GEN_MIN_KV_ROWS), 1, head_dim),
            dtype=torch.float8_e4m3fn,
            device=kv_cache.device,
        )[:padded_rows]
    else:
        if (
            output.dtype != torch.float8_e4m3fn
            or output.device != kv_cache.device
            or output.ndim != 3
            or output.shape[0] < padded_rows
            or output.shape[1:] != (1, head_dim)
        ):
            raise ValueError("output scratch has incompatible shape, dtype, or device")
        output = output[:padded_rows]
    if compact_indices is None:
        compact_indices = torch.empty_like(physical_indices, dtype=torch.int32)
    else:
        if (
            compact_indices.dtype != torch.int32
            or compact_indices.device != physical_indices.device
            or compact_indices.numel() < total_rows
        ):
            raise ValueError(
                "compact_indices scratch has incompatible dtype or capacity"
            )
        compact_indices = compact_indices.reshape(-1)[:total_rows].view_as(
            physical_indices
        )

    compact_kv = output.view(padded_rows // page_size, 1, page_size, head_dim)
    if total_rows == 0:
        return compact_kv, compact_indices

    module = _cuda_gather_module() if head_dim == 576 else None
    if module is not None:
        module.gather(
            kv_cache,
            physical_indices,
            output,
            compact_indices,
            global_scale,
        )
        _log_gather_path_once("CUDA")
        return compact_kv, compact_indices

    _log_gather_path_once("Triton")
    block_dim = 256
    _gather_dequant_nvfp4_mla_generation_kernel[
        (total_rows, triton.cdiv(head_dim, block_dim))
    ](
        data_cache,
        scale_cache,
        physical_indices,
        output,
        compact_indices,
        global_scale,
        data_row_stride=data_cache.stride(0),
        scale_row_stride=scale_cache.stride(0),
        head_dim=head_dim,
        block_dim=block_dim,
        num_warps=4,
    )
    return compact_kv, compact_indices


def gather_dequant_nvfp4_mla_cache(
    kv_cache: torch.Tensor,
    physical_indices: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    head_dim: int,
    page_size: int,
    compaction: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the distinct selected NVFP4 rows into a compact FP8 paged cache.

    Args:
        kv_cache: ``[num_slots, 1, head_dim / 2 + head_dim / 16]`` packed rows.
        physical_indices: ``[num_queries, topk]`` physical token slots; ``-1``
            marks padding.
        global_scale: one FP32 dequantization scale for this layer.
        head_dim: logical MLA row width (normally 576).
        page_size: tokens per TRTLLM-GEN KV page (normally 64).
        compaction: ``compact_selected_slots(physical_indices)`` when the caller
            already has it, e.g. from a layer sharing these top-k indices.

    Returns:
        ``compact_kv`` with shape ``[num_pages, 1, page_size, head_dim]`` and
        dtype FP8 E4M3, plus remapped ``compact_indices`` with the same shape as
        ``physical_indices`` and dtype int32.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "The basic NVFP4 DSA gather uses dynamic eager allocations and cannot "
            "run during CUDA Graph capture yet."
        )
    if physical_indices.ndim != 2:
        raise ValueError(
            f"physical_indices must be 2-D, got shape={tuple(physical_indices.shape)}"
        )
    _split_packed_rows(kv_cache, head_dim)
    _log_gather_path_once("context")

    unique_slots, compact_indices = (
        compaction
        if compaction is not None
        else compact_selected_slots(physical_indices, num_slots=kv_cache.shape[0])
    )
    if unique_slots.numel() == 0:
        compact_rows = torch.zeros(
            (max(page_size, _TRTLLM_GEN_MIN_KV_ROWS), 1, head_dim),
            dtype=torch.float8_e4m3fn,
            device=kv_cache.device,
        )[:page_size]
        return compact_rows.view(1, 1, page_size, head_dim), compact_indices

    # Row i of the output is distinct slot i, the rank compact_indices refer to.
    compact_kv, _ = gather_dequant_nvfp4_mla_cache_generation(
        kv_cache,
        unique_slots.to(torch.int32).view(1, -1),
        global_scale,
        head_dim=head_dim,
        page_size=page_size,
    )
    return compact_kv, compact_indices


def compact_selected_slots(
    physical_indices: torch.Tensor, *, num_slots: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sorted distinct valid slots, and each index remapped to its position there.

    A slot map over the pool replaces sorting all ``num_queries * topk``
    entries; padding entries mark slot 0, which only adds one unread row.
    """
    flat = physical_indices.reshape(-1)
    valid = flat >= 0
    slots = torch.where(valid, flat, 0).long()
    slot_map = torch.zeros(num_slots, dtype=torch.int32, device=flat.device)
    slot_map.index_fill_(0, slots, 1)
    unique_slots = slot_map.nonzero().squeeze(1)
    slot_map[unique_slots] = torch.arange(
        unique_slots.numel(), dtype=torch.int32, device=flat.device
    )
    compact_indices = torch.where(valid, slot_map[slots], -1).to(torch.int32)
    return unique_slots, compact_indices.view_as(physical_indices)
