# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mooncake-specific utilities for custom memory pool management."""

import logging
import os
import threading
import time
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Global constants for custom memory pool types
SUPPORTED_MOONCAKE_CUSTOM_MEM_POOL_TYPES = ["NVLINK", "BAREX", "INTRA_NODE_NVLINK"]


def _cuda_expandable_segments_enabled() -> Optional[str]:
    """Return the allocator env var that enables CUDA expandable segments."""
    for var in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        for field in os.environ.get(var, "").split(","):
            key, _, value = field.partition(":")
            if key.strip() == "expandable_segments" and value.strip().lower() == "true":
                return var
    return None


def _validate_efa_allocator_compatibility(
    enable_custom_mem_pool: bool, custom_mem_pool_type: Optional[str]
) -> None:
    """Reject CUDA VMM allocators unsupported by the current libfabric EFA path."""
    if envs.MOONCAKE_PROTOCOL.get().lower() != "efa":
        return

    if enable_custom_mem_pool and custom_mem_pool_type in ("NVLINK", "BAREX"):
        raise ValueError(
            f"SGLANG_MOONCAKE_CUSTOM_MEM_POOL={custom_mem_pool_type} is "
            "incompatible with MOONCAKE_PROTOCOL=efa. Mooncake custom memory "
            "pools use CUDA VMM allocations, which the current libfabric EFA "
            "provider cannot transfer. Unset SGLANG_MOONCAKE_CUSTOM_MEM_POOL."
        )

    expandable_segments_var = _cuda_expandable_segments_enabled()
    if expandable_segments_var is not None:
        raise ValueError(
            f"{expandable_segments_var} enables expandable_segments, which is "
            "incompatible with MOONCAKE_PROTOCOL=efa because the current "
            "libfabric EFA provider cannot transfer CUDA VMM allocations. "
            "Disable expandable_segments."
        )


def init_mooncake_custom_mem_pool(
    device: str,
) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Initialize custom memory pool based on environment variable.

    Args:
        device: The device to allocate memory on

    Returns:
        Tuple of (enable_custom_mem_pool, custom_mem_pool, custom_mem_pool_type)
    """
    enable_custom_mem_pool, custom_mem_pool_type = (
        check_mooncake_custom_mem_pool_enabled()
    )

    custom_mem_pool = None

    if enable_custom_mem_pool:
        try:
            # TODO(shangming): abstract custom allocator class for more backends
            if custom_mem_pool_type == "NVLINK":
                from mooncake.allocator import NVLinkAllocator

                allocator = NVLinkAllocator.get_allocator(device)
            elif custom_mem_pool_type == "BAREX":
                from mooncake.allocator import BarexAllocator

                allocator = BarexAllocator.get_allocator(device)
            elif custom_mem_pool_type == "INTRA_NODE_NVLINK":
                return False, None, None
            else:
                # This should not happen due to the enable_custom_mem_pool check above
                raise ValueError(
                    f"Unsupported custom mem pool type: {custom_mem_pool_type}"
                )

            # MemPool binds to the current device; on a non-main thread (e.g.
            # PD bootstrap) that is device 0, so ranks on other GPUs hit the
            # CUDACachingAllocator use_count assert in use_mem_pool().
            with torch.cuda.device(device):
                custom_mem_pool = torch.cuda.MemPool(allocator.allocator())
            logger.debug(
                f"Initialized custom memory pool: {custom_mem_pool_type} on device {device}"
            )
        except ImportError as e:
            logger.warning(
                f"Failed to import mooncake allocator for {custom_mem_pool_type}: {e}. "
                f"Falling back to default memory pool."
            )
            enable_custom_mem_pool = False
            custom_mem_pool = None
            custom_mem_pool_type = None
        except Exception as e:
            logger.error(
                f"Failed to initialize custom memory pool {custom_mem_pool_type}: {e}. "
                f"Falling back to default memory pool."
            )
            enable_custom_mem_pool = False
            custom_mem_pool = None
            custom_mem_pool_type = None
    else:
        return False, None, None

    return enable_custom_mem_pool, custom_mem_pool, custom_mem_pool_type


def check_mooncake_custom_mem_pool_enabled() -> Tuple[bool, Optional[str]]:
    """
    Check if custom memory pool is enabled without importing allocators.

    Returns:
        Tuple of (enable_custom_mem_pool, custom_mem_pool_type)
    """
    custom_mem_pool_type = envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get()

    if custom_mem_pool_type is not None:
        # Handle boolean True as NVLINK
        if custom_mem_pool_type.lower() == "true":
            custom_mem_pool_type = "NVLINK"
        enable_custom_mem_pool = (
            custom_mem_pool_type in SUPPORTED_MOONCAKE_CUSTOM_MEM_POOL_TYPES
        )
    else:
        enable_custom_mem_pool = False
        custom_mem_pool_type = None

    return enable_custom_mem_pool, custom_mem_pool_type


class TransferWorkerStats:
    """Prefill transfer-worker timings of one process, logged every interval."""

    def __init__(self, interval_s: float):
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._window_start = time.monotonic()
        self._samples: List[Tuple[float, float, float, int, int, int]] = []

    def record(
        self,
        *,
        queue_s: float,
        event_s: float,
        send_s: float,
        kv_bytes: int,
        pages: int,
        runs: int,
    ) -> None:
        now = time.monotonic()
        with self._lock:
            self._samples.append((queue_s, event_s, send_s, kv_bytes, pages, runs))
            if now - self._window_start < self._interval_s:
                return
            samples, span_s = self._samples, now - self._window_start
            self._samples, self._window_start = [], now
        logger.info(format_transfer_worker_stats(samples=samples, span_s=span_s))


def format_transfer_worker_stats(
    *, samples: List[Tuple[float, float, float, int, int, int]], span_s: float
) -> str:
    # Columns: queue, event and send seconds, KV bytes, pages, contiguous runs.
    table = np.array(samples, dtype=np.float64).reshape(-1, 6)
    queue_ms, event_ms, send_ms = (table[:, :3] * 1e3).T
    gib = table[:, 3] / 2**30
    gib_per_s = gib / np.maximum(table[:, 2], 1e-9)

    def spread(values: np.ndarray) -> str:
        p50, p90 = np.percentile(values, [50, 90])
        return f"p50 {p50:.1f} p90 {p90:.1f} max {values.max():.1f}"

    return (
        f"Mooncake KV transfer worker, last {span_s:.0f}s: {len(table)} chunks, "
        f"{gib.sum():.2f} GiB KV; queue ms {spread(queue_ms)}; "
        f"event ms {spread(event_ms)}; send ms {spread(send_ms)}; "
        f"GiB/s per chunk p50 {np.percentile(gib_per_s, 50):.2f} "
        f"p90 {np.percentile(gib_per_s, 90):.2f}, "
        f"overall {gib.sum() / max(table[:, 2].sum(), 1e-9):.2f}; "
        f"pages per contiguous run {table[:, 4].sum() / max(table[:, 5].sum(), 1):.1f}"
    )
