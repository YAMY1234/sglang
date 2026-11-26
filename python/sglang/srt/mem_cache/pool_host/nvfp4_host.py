"""HiCache host pools that keep FP8 MLA/DSA KV as NVFP4 (--hicache-host-kv-dtype nvfp4).

The device pool, the attention kernels and PD transfer stay FP8. Write-back
encodes each FP8 row to NVFP4 on the device before the D2H copy; load-back
copies the NVFP4 rows to the device and decodes them to FP8 in place. The
global scale is fixed at 1/6, so every block scale is its block's FP8 amax,
which E4M3 holds exactly: no calibration, and any process decodes any page.
Host pages are opaque bytes to every L3 backend.
"""

from __future__ import annotations

import logging

import torch

from sglang.kernels.ops.kvcache.nvfp4_host_kv import (
    decode_nvfp4_rows_to_fp8,
    encode_fp8_rows_to_nvfp4,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.device_pool_info import (
    IndexKeyBufferInfo,
    IndexKeyLayout,
    MLABufferInfo,
)
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()
if _is_cuda:
    from sgl_kernel.kvcacheio import (
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_per_layer_direct_pf_lf,
    )

logger = logging.getLogger(__name__)

NVFP4_HOST_LAYOUT = "page_first_direct"
NVFP4_BLOCK_SIZE = 16
_GLOBAL_SCALE = 1.0 / 6.0
# IndexKeyLayout.PAGED_FP8_KEYS_WITH_SCALES: 128 FP8 bytes per key.
_PAGED_INDEX_KEY_HEAD_DIM = 128
# Device scratch one encode or decode step may take on its transfer stream; the
# step's page count follows from it. Arbitrary.
NVFP4_HOST_SCRATCH_BYTES = 256 << 20


def nvfp4_row_bytes(dim: int) -> int:
    return dim // 2 + dim // NVFP4_BLOCK_SIZE


def _scratch_chunk_pages(scratch_bytes_per_page: int) -> int:
    return max(1, NVFP4_HOST_SCRATCH_BYTES // scratch_bytes_per_page)


class NVFP4RowCodec:
    """FP8 rows (uint8 storage, ``dim`` values) <-> NVFP4 rows (``nvfp4_row_bytes(dim)`` bytes)."""

    def __init__(self, dim: int, device: torch.device | str):
        if dim % NVFP4_BLOCK_SIZE:
            raise ValueError(f"NVFP4 rows need a multiple of 16 values, got {dim}")
        self.dim = dim
        self.row_bytes = nvfp4_row_bytes(dim)
        self.global_scale = torch.tensor(
            [_GLOBAL_SCALE], dtype=torch.float32, device=device
        )
        # Encode writes into the caller's staging; decode reads only its stage.
        self.encode_scratch_bytes = 0
        self.decode_scratch_bytes = self.row_bytes

    def encode_rows(self, src, rows, out, *, page_size: int) -> None:
        """Encode src[rows] (FP8 bytes) into out[page, unit, ..., row_bytes],
        unit = position in rows % page_size."""
        encode_fp8_rows_to_nvfp4(
            src, rows, out, self.global_scale, dim=self.dim, page_size=page_size
        )

    def decode_rows(self, src, rows, dst) -> None:
        """Decode NVFP4 rows src[i] ([n, row_bytes]) into dst[rows[i]] (FP8 bytes)."""
        decode_nvfp4_rows_to_fp8(src, rows, dst, self.global_scale, dim=self.dim)


class IndexKeyPageCodec:
    """One DSA index-key page: [keys FP8 | per-token FP32 scales] ->
    [NVFP4 key rows | the same scales]. The scales stay exact."""

    def __init__(self, page_size: int, device: torch.device | str):
        self.page_size = page_size
        self.rows = NVFP4RowCodec(_PAGED_INDEX_KEY_HEAD_DIM, device)
        self.key_bytes = page_size * _PAGED_INDEX_KEY_HEAD_DIM
        self.encoded_key_bytes = page_size * self.rows.row_bytes
        scale_bytes = page_size * 4
        self.device_page_bytes = self.key_bytes + scale_bytes
        self.host_page_bytes = self.encoded_key_bytes + scale_bytes
        # Per page: the page gather and the key copy.
        self.encode_scratch_bytes = self.device_page_bytes + self.key_bytes
        # Per page: the stage, the key-row copy, the decoded keys and the page.
        self.decode_scratch_bytes = (
            self.host_page_bytes
            + self.encoded_key_bytes
            + self.key_bytes
            + self.device_page_bytes
        )

    def encode_rows(self, src, rows, out, *, page_size: int) -> None:
        """Encode the device pages src[rows] into out[page, 1, host_page_bytes]."""
        n = rows.numel()
        pages = src.index_select(0, rows)
        keys = pages[:, : self.key_bytes].reshape(n * self.page_size, self.rows.dim)
        host = out.view(n, self.host_page_bytes)
        self.rows.encode_rows(
            keys,
            torch.arange(n * self.page_size, device=keys.device),
            host[:, : self.encoded_key_bytes].view(
                n, self.page_size, self.rows.row_bytes
            ),
            page_size=self.page_size,
        )
        host[:, self.encoded_key_bytes :].copy_(pages[:, self.key_bytes :])

    def decode_rows(self, src, rows, dst) -> None:
        """Decode host pages src[i] ([n, host_page_bytes]) into dst[rows[i]]."""
        n = rows.numel()
        key_rows = src[:, : self.encoded_key_bytes].reshape(
            n * self.page_size, self.rows.row_bytes
        )
        keys = torch.empty(
            (n * self.page_size, self.rows.dim), dtype=torch.uint8, device=src.device
        )
        self.rows.decode_rows(
            key_rows, torch.arange(n * self.page_size, device=src.device), keys
        )
        pages = torch.cat(
            (keys.view(n, self.key_bytes), src[:, self.encoded_key_bytes :]), dim=1
        )
        dst.index_copy_(0, rows, pages)


def _upload_indices(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Once per L2 submission, from pinned memory: the per-layer loads then index
    # device rows without a pageable host-to-device copy each.
    if indices.is_cuda:
        return indices
    return indices.pin_memory().to(device, non_blocking=True)


def _require_direct_io(io_backend: str) -> None:
    if io_backend != "direct":
        raise ValueError(
            f"--hicache-host-kv-dtype nvfp4 requires the direct IO backend, got {io_backend!r}"
        )


def _backup_all_layers(
    codec,
    device_layers,
    host_buffer: torch.Tensor,
    host_units: torch.Tensor,
    device_units: torch.Tensor,
    *,
    units_per_page: int,
    chunk_pages: int,
    arange: torch.Tensor,
) -> None:
    """Encode every layer's units straight into page-major staging laid out like
    the host pages ([page, layer, unit, ..., unit_bytes]), then copy whole pages
    to the host with one batched copy.

    host_units are page-aligned CPU indices; device_units index the device rows
    and already live on their device; arange is a CPU arange of >= chunk_pages.
    """
    total = host_units.numel()
    if total == 0:
        return
    step = chunk_pages * units_per_page
    staging = torch.empty(
        (
            min(total, step) // units_per_page,
            len(device_layers),
            units_per_page,
            host_buffer.shape[-1],
        ),
        dtype=torch.uint8,
        device=device_layers[0].device,
    )
    for start in range(0, total, step):
        h = host_units[start : start + step]
        d = device_units[start : start + step]
        pages = h.numel() // units_per_page
        chunk = staging[:pages]
        for j, layer in enumerate(device_layers):
            codec.encode_rows(layer, d, chunk[:, j], page_size=units_per_page)
        # Each staging row is one host page, all layers.
        transfer_kv_all_layer_direct_lf_pf(
            src_ptrs=[chunk.view(pages, -1)],
            dst_ptrs=[host_buffer.view(host_buffer.shape[0], 1, 1, -1)],
            src_indices=arange[:pages],
            dst_indices=h[::units_per_page] // units_per_page,
            page_size=1,
        )


def _load_one_layer(
    codec,
    device_layer: torch.Tensor,
    host_buffer: torch.Tensor,
    host_layer: int,
    host_units: torch.Tensor,
    device_units: torch.Tensor,
    *,
    units_per_page: int,
    chunk_pages: int,
    arange: torch.Tensor,
) -> None:
    """Copy one host layer's units to a device stage, then decode them straight
    into the device rows."""
    total = host_units.numel()
    if total == 0:
        return
    step = chunk_pages * units_per_page
    for start in range(0, total, step):
        h = host_units[start : start + step]
        d = device_units[start : start + step]
        stage = torch.empty(
            (h.numel(), *host_buffer.shape[3:]),
            dtype=torch.uint8,
            device=device_layer.device,
        )
        transfer_kv_per_layer_direct_pf_lf(
            src_ptrs=[host_buffer],
            dst_ptrs=[stage],
            src_indices=h,
            dst_indices=arange[: h.numel()],
            layer_id=host_layer,
            page_size=units_per_page,
        )
        codec.decode_rows(stage.view(h.numel(), -1), d, device_layer)


class MLATokenToKVPoolNVFP4Host(MLATokenToKVPoolHost):
    """Raw FP8 MLA rows (latent + RoPE, e.g. 576 B) kept as NVFP4 rows (324 B)."""

    def __init__(
        self,
        buffer_info: MLABufferInfo,
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
            not _is_cuda
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
        # Read by the DSA indexer builder; the L3 key tag names both formats.
        self.encode_index_keys = envs.SGLANG_ENABLE_HICACHE_NVFP4_HOST_INDEXER.get()
        self.storage_format_tag = (
            "nvfp4-kv-index" if self.encode_index_keys else "nvfp4-kv"
        )
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
        self._backup_chunk_pages = _scratch_chunk_pages(
            self.page_size
            * (self.layer_num * self.kv_cache_dim + self._codec.encode_scratch_bytes)
        )
        self._load_chunk_pages = _scratch_chunk_pages(
            self.page_size * self._codec.decode_scratch_bytes
        )
        self._arange = torch.arange(
            max(self._load_chunk_pages * self.page_size, self._backup_chunk_pages)
        )
        logger.info(
            "HiCache host KV stored as NVFP4: %d B per token-layer instead of %d, "
            "index keys %s.",
            self.kv_cache_dim,
            device_dim,
            "NVFP4" if self.encode_index_keys else "FP8",
        )

    def prepare_transfer_indices(self, host_indices, device_indices, io_backend):
        return host_indices, _upload_indices(device_indices, self._transfer_device)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        _require_direct_io(io_backend)
        _backup_all_layers(
            self._codec,
            self.device_kv_buffers,
            self.kv_buffer,
            host_indices,
            device_indices,
            units_per_page=self.page_size,
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
        if not is_draft and layer_id not in self._target_layer_indices:
            return
        _require_direct_io(io_backend)
        _load_one_layer(
            self._codec,
            self.device_kv_buffers[self._buffer_index_by_layer[layer_id]],
            self.kv_buffer,
            self._host_layer_by_controller[layer_id],
            host_indices,
            device_indices,
            units_per_page=self.page_size,
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
        buffer_device: str,
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
                "SGLANG_ENABLE_HICACHE_NVFP4_HOST_INDEXER needs FP8 index keys with "
                "scales on whole KV pages, without DCP or layer sharding"
            )
        self._codec = IndexKeyPageCodec(buffer_info.page_size, buffer_device)
        super().__init__(
            buffer_info,
            transfer_page_size=transfer_page_size,
            buffer_device=buffer_device,
            layer_sharded=layer_sharded,
            **kwargs,
        )
        self._backup_chunk_pages = _scratch_chunk_pages(
            self.layer_num * self._codec.host_page_bytes
            + self._codec.encode_scratch_bytes
        )
        self._load_chunk_pages = _scratch_chunk_pages(self._codec.decode_scratch_bytes)
        self._arange = torch.arange(
            max(self._load_chunk_pages, self._backup_chunk_pages)
        )

    def _initialize_host_buffers(self):
        # Host pages hold the encoded page, not the device page.
        self.indexer_page_stride_size = self._codec.host_page_bytes
        super()._initialize_host_buffers()

    def prepare_transfer_indices(self, host_indices, device_indices, io_backend):
        return host_indices, _upload_indices(device_indices, self._buffer_device)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        _require_direct_io(io_backend)
        host_pages, device_pages = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        _backup_all_layers(
            self._codec,
            self.packed_device_index_buffers,
            self.index_k_with_scale_buffer,
            host_pages,
            device_pages,
            units_per_page=1,
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
        _load_one_layer(
            self._codec,
            self.packed_device_index_buffers[host_layer_id],
            self.index_k_with_scale_buffer,
            host_layer_id,
            host_pages,
            device_pages,
            units_per_page=1,
            chunk_pages=self._load_chunk_pages,
            arange=self._arange,
        )
