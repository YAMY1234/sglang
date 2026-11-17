"""HiCache host pools that keep FP8 DSA KV as NVFP4 (--hicache-host-kv-dtype nvfp4).

The device pool, the attention kernels and PD transfer stay FP8. Write-back
encodes each FP8 row to NVFP4 on the device before the D2H copy; load-back
copies the NVFP4 rows to the device and decodes them to FP8 in place. Host
pages are opaque bytes to every L3 backend.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.device_pool_info import IndexKeyBufferInfo, IndexKeyLayout
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.nvfp4_host_codec import (
    NVFP4_HOST_SCRATCH_BYTES,
    IndexKeyPageCodec,
    NVFP4RowCodec,
    backup_all_layers,
    load_layer_group,
    load_one_layer,
    nvfp4_row_bytes,
    scratch_chunk_pages,
)
from sglang.srt.utils import is_cuda

if is_cuda():
    from sgl_kernel.kvcacheio import (
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_per_layer_direct_pf_lf,
    )

logger = logging.getLogger(__name__)

NVFP4_HOST_LAYOUT = "page_first_direct"
# IndexKeyLayout.PAGED_FP8_KEYS_WITH_SCALES: 128 FP8 bytes per key.
_PAGED_INDEX_KEY_HEAD_DIM = 128


def _copy_pages_to_host(staging, host_buffer, host_page_ids, staging_ids) -> None:
    # One batched copy of whole pages: each staging row is one host page, all layers.
    transfer_kv_all_layer_direct_lf_pf(
        src_ptrs=[staging],
        dst_ptrs=[host_buffer.view(host_buffer.shape[0], 1, 1, -1)],
        src_indices=staging_ids,
        dst_indices=host_page_ids,
        page_size=1,
    )


def _copy_layer_from_host(
    host_buffer, stage, host_units, layer_id, units_per_page, stage_ids
):
    transfer_kv_per_layer_direct_pf_lf(
        src_ptrs=[host_buffer],
        dst_ptrs=[stage],
        src_indices=host_units,
        dst_indices=stage_ids,
        layer_id=layer_id,
        page_size=units_per_page,
    )


def _copy_pages_from_host(host_buffer, stage, host_page_ids, first_layer, stage_ids):
    # One copy per page of host layers [first_layer, first_layer + stage row bytes / layer bytes).
    transfer_kv_per_layer_direct_pf_lf(
        src_ptrs=[host_buffer],
        dst_ptrs=[stage],
        src_indices=host_page_ids,
        dst_indices=stage_ids,
        layer_id=first_layer,
        page_size=1,
    )


def _upload_indices(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Once per L2 submission, from pinned memory: the per-layer loads then index
    # device rows without a pageable host-to-device copy each.
    if indices.device == device:
        return indices
    return indices.pin_memory().to(device, non_blocking=True)


def _require_direct_io(io_backend: str) -> None:
    if io_backend != "direct":
        raise ValueError(
            f"--hicache-host-kv-dtype nvfp4 requires the direct IO backend, got {io_backend!r}"
        )


class MLATokenToKVPoolNVFP4Host(MLATokenToKVPoolHost):
    """Raw FP8 MLA rows (latent + RoPE, e.g. 576 B) kept as NVFP4 rows (324 B)."""

    def __init__(
        self,
        buffer_info,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        *,
        override_kv_cache_dim: int | None = None,
        layer_shard_enabled: bool = False,
        dcp_size: int = 1,
        **kwargs,
    ):
        device_dim = buffer_info.kv_cache_dim
        if (
            not is_cuda()
            or layout != NVFP4_HOST_LAYOUT
            or buffer_info.dtype != torch.uint8
            or buffer_info.scales is not None
            or device_dim != buffer_info.kv_lora_rank + buffer_info.qk_rope_head_dim
            or override_kv_cache_dim not in (None, device_dim)
            or layer_shard_enabled
            or dcp_size != 1
        ):
            raise NotImplementedError(
                "--hicache-host-kv-dtype nvfp4 supports raw FP8 MLA rows on CUDA "
                f"with the {NVFP4_HOST_LAYOUT} layout, without layer sharding or DCP; "
                f"got layout={layout!r}, row={device_dim} x {buffer_info.dtype}"
            )
        self._codec = NVFP4RowCodec(device_dim, buffer_info.device)
        # Read by the DSA indexer builder, so the L3 key tag names both formats.
        self.encode_index_keys = envs.SGLANG_ENABLE_HICACHE_NVFP4_HOST_INDEXER.get()
        self.storage_format_tag = "nvfp4" if self.encode_index_keys else "nvfp4-kv"
        super().__init__(
            buffer_info,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            override_kv_cache_dim=nvfp4_row_bytes(device_dim),
            layer_shard_enabled=layer_shard_enabled,
            dcp_size=dcp_size,
            **kwargs,
        )
        self._backup_chunk_pages = scratch_chunk_pages(
            self.page_size
            * (self.layer_num * self.kv_cache_dim + self._codec.encode_scratch_bytes)
        )
        self._load_chunk_pages = scratch_chunk_pages(
            self.page_size * self._codec.decode_scratch_bytes
        )
        self._arange = torch.arange(
            max(self._load_chunk_pages * self.page_size, self._backup_chunk_pages)
        )
        self._load_group = max(1, envs.SGLANG_HICACHE_NVFP4_LOAD_GROUP.get())
        if self._load_group > 1:
            self._init_grouped_load()
        logger.info(
            "HiCache host KV stored as NVFP4, codec: %s. %d B per token-layer "
            "instead of %d, %d layers, %d tokens, index keys %s; device scratch "
            "<= %d MiB per step (write-back %d pages, load %d pages).",
            self._codec.backend,
            self.kv_cache_dim,
            device_dim,
            self.layer_num,
            self.size,
            "NVFP4" if self.encode_index_keys else "FP8",
            NVFP4_HOST_SCRATCH_BYTES >> 20,
            self._backup_chunk_pages,
            self._load_chunk_pages,
        )

    def prepare_transfer_indices(self, host_indices, device_indices, io_backend):
        return host_indices, _upload_indices(device_indices, self._transfer_device)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        _require_direct_io(io_backend)
        backup_all_layers(
            self._codec,
            self.device_kv_buffers,
            self.kv_buffer,
            host_indices,
            device_indices,
            units_per_page=self.page_size,
            copy_pages_to_host=_copy_pages_to_host,
            chunk_pages=self._backup_chunk_pages,
            arange=self._arange,
        )

    def _init_grouped_load(self):
        k, targets = self._load_group, list(self._target_layer_indices)
        assert self.kv_buffer.is_contiguous(), "grouped load needs page-first contiguous host pages"
        by_host = {self._host_layer_by_controller[layer]: layer for layer in targets}
        assert sorted(by_host) == list(range(len(targets))), by_host
        # Controller layer ids of host layers [first, first + k), per group.
        self._group_layers = {
            first: [by_host[h] for h in range(first, min(first + k, len(targets)))]
            for first in range(0, len(targets), k)
        }
        self._group_chunk_pages = scratch_chunk_pages(self.page_size * k * self.kv_cache_dim)
        self._group_loaded = None
        self._group_verify_left = envs.SGLANG_HICACHE_NVFP4_LOAD_GROUP_VERIFY.get()
        self._arange = torch.arange(max(self._arange.numel(), self._group_chunk_pages))
        logger.info(
            "HiCache NVFP4 grouped load: %d layers per page copy, %d groups, %d pages per "
            "chunk, verify %d loads",
            k, len(self._group_layers), self._group_chunk_pages, self._group_verify_left,
        )

    def _load_grouped(self, host_indices, device_indices, layer_id):
        host_layer = self._host_layer_by_controller[layer_id]
        first = host_layer - host_layer % self._load_group
        key = (host_indices.data_ptr(), host_indices.numel(), device_indices.data_ptr(), first)
        if host_layer != first:
            # Already decoded with its group's first layer, earlier in this same submission.
            assert self._group_loaded == key, ("grouped load out of order", self._group_loaded, key)
            return
        layers = self._group_layers[first]
        load_layer_group(
            self._codec,
            [self.device_kv_buffers[self._buffer_index_by_layer[layer]] for layer in layers],
            self.kv_buffer,
            first,
            host_indices,
            device_indices,
            units_per_page=self.page_size,
            chunk_pages=self._group_chunk_pages,
            arange=self._arange,
            copy_pages_from_host=_copy_pages_from_host,
        )
        self._group_loaded = key
        if self._group_verify_left > 0:
            self._verify_group(layers, first, host_indices, device_indices)

    def _verify_group(self, layers, first, host_indices, device_indices):
        # The per-layer path into a scratch tensor must give the same bytes (syncs).
        rows = torch.arange(host_indices.numel(), device=device_indices.device)
        for j, layer in enumerate(layers):
            device_layer = self.device_kv_buffers[self._buffer_index_by_layer[layer]]
            ref = torch.empty(
                (rows.numel(), *device_layer.shape[1:]),
                dtype=device_layer.dtype,
                device=device_layer.device,
            )
            load_one_layer(
                self._codec, ref, self.kv_buffer, first + j, host_indices, rows,
                units_per_page=self.page_size, copy_layer_from_host=_copy_layer_from_host,
                chunk_pages=self._load_chunk_pages, arange=self._arange,
            )
            if not torch.equal(device_layer.index_select(0, device_indices), ref):
                raise RuntimeError(f"NVFP4 grouped load differs from the per-layer load: host layer {first + j}")
        self._group_verify_left -= 1
        logger.info(
            "NVFP4_GROUPED_LOAD_VERIFY_OK host layers %d-%d, %d tokens, %d loads left",
            first, first + len(layers) - 1, rows.numel(), self._group_verify_left,
        )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if not is_draft and layer_id not in self._target_layer_indices:
            return
        _require_direct_io(io_backend)
        if self._load_group > 1 and not is_draft:
            self._load_grouped(host_indices, device_indices, layer_id)
            return
        load_one_layer(
            self._codec,
            self.device_kv_buffers[self._buffer_index_by_layer[layer_id]],
            self.kv_buffer,
            self._host_layer_by_controller[layer_id],
            host_indices,
            device_indices,
            units_per_page=self.page_size,
            copy_layer_from_host=_copy_layer_from_host,
            chunk_pages=self._load_chunk_pages,
            arange=self._arange,
        )


class DSAIndexerPoolNVFP4Host(DSAIndexerPoolHost):
    """DSA index-key pages ([FP8 keys | FP32 scales], 132 B per token) kept as
    [NVFP4 keys | FP32 scales] (76 B per token)."""

    def __init__(
        self,
        buffer_info: IndexKeyBufferInfo,
        *,
        transfer_page_size: int,
        layer_sharded: bool = False,
        **kwargs,
    ):
        if (
            buffer_info.key_layout is not IndexKeyLayout.PAGED_FP8_KEYS_WITH_SCALES
            or buffer_info.compress_ratio != 1
            or buffer_info.kernel_page_size not in (None, buffer_info.page_size)
            or transfer_page_size != buffer_info.page_size
            or layer_sharded
        ):
            raise NotImplementedError(
                "--hicache-host-kv-dtype nvfp4 index keys need FP8 keys with scales "
                "on whole KV pages (index_kpool=1), without DCP or layer sharding"
            )
        self._index_device = buffer_info.buffers[0].device
        self._codec = IndexKeyPageCodec(
            page_size=buffer_info.page_size,
            head_dim=_PAGED_INDEX_KEY_HEAD_DIM,
            device=self._index_device,
        )
        if buffer_info.kernel_page_bytes != self._codec.device_page_bytes:
            raise ValueError(
                f"DSA index page of {buffer_info.kernel_page_bytes} B "
                f"is not FP8 keys plus FP32 scales ({self._codec.device_page_bytes} B)"
            )
        super().__init__(
            buffer_info,
            transfer_page_size=transfer_page_size,
            layer_sharded=layer_sharded,
            **kwargs,
        )
        if self._is_dummy:
            return
        self._backup_chunk_pages = scratch_chunk_pages(
            self.layer_num * self._codec.host_page_bytes
            + self._codec.encode_scratch_bytes
        )
        self._load_chunk_pages = scratch_chunk_pages(self._codec.decode_scratch_bytes)
        self._arange = torch.arange(
            max(self._load_chunk_pages, self._backup_chunk_pages)
        )
        logger.info(
            "HiCache host DSA index keys stored as NVFP4, codec: %s. %d B per "
            "page-layer instead of %d; write-back %d pages, load %d pages per step.",
            self._codec.backend,
            self._codec.host_page_bytes,
            self._codec.device_page_bytes,
            self._backup_chunk_pages,
            self._load_chunk_pages,
        )

    def _initialize_host_buffers(self):
        # Host pages hold the encoded page, not the device page.
        self.indexer_page_stride_size = self._codec.host_page_bytes
        super()._initialize_host_buffers()

    def prepare_transfer_indices(self, host_indices, device_indices, io_backend):
        return host_indices, _upload_indices(device_indices, self._index_device)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        _require_direct_io(io_backend)
        host_pages, device_pages = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        backup_all_layers(
            self._codec,
            self.packed_device_index_buffers,
            self.index_k_with_scale_buffer,
            host_pages,
            device_pages,
            units_per_page=1,
            copy_pages_to_host=_copy_pages_to_host,
            chunk_pages=self._backup_chunk_pages,
            arange=self._arange,
        )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if not is_draft and not self._is_device_layer_owned(device_pool, layer_id):
            return
        _require_direct_io(io_backend)
        host_layer_id = (
            self._draft_host_layer(layer_id)
            if is_draft
            else self._host_layer_index(layer_id)
        )
        host_pages, device_pages = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        load_one_layer(
            self._codec,
            self.packed_device_index_buffers[host_layer_id],
            self.index_k_with_scale_buffer,
            host_layer_id,
            host_pages,
            device_pages,
            units_per_page=1,
            copy_layer_from_host=_copy_layer_from_host,
            chunk_pages=self._load_chunk_pages,
            arange=self._arange,
        )
