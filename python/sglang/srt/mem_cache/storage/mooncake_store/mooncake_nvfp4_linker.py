"""HiCache's existing MLA host codec at the Mooncake direct-linker boundary.

Only prefetch/all-layer IO is supported. One registered GPU workspace is shared
by the existing load/offload workers. Their futures and PP votes remain owned
by MooncakeDirectLinker; an IO result is returned only after decoding finishes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import replace
from pathlib import Path

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolEntry
from sglang.srt.mem_cache.pool_host import nvfp4_host_codec
from sglang.srt.mem_cache.pool_host.nvfp4_host_codec import NVFP4RowCodec
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.storage.mooncake_store.nvfp4_linker_config import (
    HOST_CODEC_ABI,
    HOST_CODEC_WORKSPACE_BYTES,
)
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


def format_identity(pool_group, start_layer: int) -> dict:
    """No addresses, worker identity or local token capacity enters store keys."""
    entries = []
    for pool in pool_group.entries:
        entries.append(
            {
                "name": str(pool.name),
                "layer_mapping": sorted(
                    (start_layer + layer, value)
                    for layer, value in pool.layer_mapping.items()
                ),
                "buffer_bytes_per_token": [
                    buffer.nbytes // buffer.shape[0] // pool.page_size
                    if pool._row_span == 1
                    else buffer.nbytes // buffer.shape[0]
                    for buffer in pool.kv_buffer
                ],
            }
        )
    return dict(
        format="hicache-nvfp4-kv/1",
        page_size=pool_group.page_size,
        main_fp8_bytes=576,
        main_host_bytes=324,
        index_host_bytes=132,
        block_size=16,
        global_scale="1/6",
        nibble_order="low-first",
        codec_abi=HOST_CODEC_ABI,
        start_layer=start_layer,
        target_layers=pool_group.num_layers,
        # Physical buffer order includes the existing packed MTP components.
        pools=entries,
    )


class NVFP4MooncakeIO:
    """Storage facade; unchanged lookup/metadata methods delegate to Mooncake.

    The two IO workers serialize ownership here, never in the scheduler. Native
    synchronous Mooncake calls retain staging until completion. The finally
    fence also covers codec errors after some GPU work has already launched.
    """

    def __init__(self, storage, pool_group, start_layer: int):
        self.storage = storage
        self.pool_group = pool_group
        self.main = pool_group.entry_map[PoolName.KV]
        self.layers = self.main.get_hybrid_pool_buffer()
        self.device = self.layers[0].device
        self.page_size = pool_group.page_size
        if self.page_size != 64 or len(self.main.components) != 1:
            raise ValueError("NVFP4 linker requires page64 and one MLA component")
        if not pool_group.rank_replicated or set(pool_group.entry_map) - {
            PoolName.KV,
            PoolName.INDEXER,
        }:
            raise ValueError("NVFP4 linker supports the replicated GLM DSA pools")
        if any(
            layer.dtype != torch.uint8
            or layer.device != self.device
            or layer[0].numel() != 576
            or not layer.is_contiguous()
            for layer in self.layers
        ):
            raise ValueError("NVFP4 linker requires contiguous raw576-byte MLA rows")
        index = pool_group.entry_map.get(PoolName.INDEXER)
        if index is not None and any(
            buffer.dtype != torch.uint8
            or buffer.device != self.device
            or buffer[0].numel() != self.page_size * 132
            for buffer in index.kv_buffer
        ):
            raise ValueError("NVFP4 linker keeps the existing132-byte index format")
        if hashlib.sha256(Path(nvfp4_host_codec.__file__).read_bytes()).hexdigest() != HOST_CODEC_ABI:
            raise ValueError("NVFP4 linker codec source differs from qualified host ABI")
        self.codec = NVFP4RowCodec(576, self.device)
        self.identity = format_identity(pool_group, start_layer)
        self.digest = hashlib.sha256(
            json.dumps(self.identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        # Include the staging tensor, row IDs, codec globals and one layer's
        # codec workspace in the same64MiB pre-KV-sizing reservation.
        per_page = self.page_size * (
            len(self.layers) * self.codec.row_bytes
            + max(self.codec.encode_scratch_bytes, self.codec.decode_scratch_bytes)
            + 8
        )
        self.chunk_pages = (HOST_CODEC_WORKSPACE_BYTES - 4096) // per_page
        if self.chunk_pages < 1:
            raise ValueError("NVFP4 linker workspace cannot hold one page")
        self.workspace = torch.empty(
            (len(self.layers), self.chunk_pages, self.page_size, self.codec.row_bytes),
            dtype=torch.uint8,
            device=self.device,
        )
        self.local_indices = torch.arange(self.chunk_pages * self.page_size)
        self.stage_pool = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=self.main.device_pool,
            components=[[self.workspace[layer] for layer in range(len(self.layers))]],
            layer_mapping=self.main.layer_mapping,
            page_size=self.page_size,
            rows_are_pages=True,
        )
        self.lock = threading.Lock()
        self.stream = None
        # Diagnostic: re-read every Nth multi-chunk load (fraction), fenced, and compare with what the load decoded.
        self.get_recheck = float(os.environ.get("SGLANG_MOONCAKE_NVFP4_GET_RECHECK", "0") or 0)
        self.recheck_stats = dict(seen=0, loads=0, pages=0, bad_pages=0, bad_loads=0, read_fail=0)
        self._recheck_scratch = None
        storage.registered_pools = {**pool_group.entry_map, PoolName.KV: self.stage_pool}
        suffix = "_hostnvfp4_" + self.digest
        storage.mla_suffix += suffix
        storage.mha_suffix += suffix

    def __getattr__(self, name):
        return getattr(self.storage, name)

    def register_buffers(self):
        """Called on the CUDA startup thread, before either IO worker starts."""
        if self.device.type != "cuda" or self.codec._fused is None:
            raise ValueError("NVFP4 linker currently requires the verified fused CUDA host codec")
        self.stream = torch.cuda.Stream(device=self.device)
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        buffers = [self.workspace]
        index = self.pool_group.entry_map.get(PoolName.INDEXER)
        if index is not None:
            buffers.extend(index.kv_buffer)
        seen = set()
        for buffer in buffers:
            allocation = buffer.untyped_storage()
            pair = (int(allocation.data_ptr()), int(allocation.nbytes()))
            if pair in seen:
                continue
            seen.add(pair)
            result = self.storage.store.register_buffer(*pair)
            if result not in (0, None):
                raise RuntimeError(f"NVFP4 linker buffer registration failed: {result}")
        logger.info(
            "Mooncake host NVFP4: digest=%s chunk_pages=%d stage_bytes=%d "
            "workspace_reserve=%d codec=%s",
            self.digest, self.chunk_pages, self.workspace.nbytes,
            HOST_CODEC_WORKSPACE_BYTES, self.codec.backend,
        )

    def _chunks(self, transfer):
        pool = self.pool_group.entry_map[transfer.name]
        # Validate real page alignment/bounds before a byte of IO is submitted.
        pool.prepare_locations(transfer.host_indices)
        if len(transfer.keys) * self.page_size != transfer.host_indices.numel():
            raise ValueError("NVFP4 linker key/page count mismatch")
        for first in range(0, len(transfer.keys), self.chunk_pages):
            keys = transfer.keys[first : first + self.chunk_pages]
            indices = transfer.host_indices[
                first * self.page_size : (first + len(keys)) * self.page_size
            ]
            rows = (
                indices.to(device=self.device, dtype=torch.int64)
                if transfer.name == PoolName.KV
                else None
            )
            chunk = replace(
                transfer,
                keys=list(keys),
                host_indices=(
                    self.local_indices[: indices.numel()]
                    if transfer.name == PoolName.KV
                    else transfer.host_indices[
                        first * self.page_size : (first + len(keys)) * self.page_size
                    ]
                ),
            )
            yield chunk, rows

    def _set(self, transfers, extra_info):
        results = {}
        for transfer in transfers:
            result = results.setdefault(transfer.name, [])
            for chunk, rows in self._chunks(transfer):
                if transfer.name == PoolName.KV:
                    for layer, source in enumerate(self.layers):
                        self.codec.encode_rows(
                            source, rows, self.workspace[layer, : len(chunk.keys)],
                            page_size=self.page_size,
                        )
                    self.stream.synchronize()  # NIC must see the encoded bytes.
                got = self.storage.batch_set_v2([chunk], extra_info)
                values = got.get(transfer.name, [])
                if len(values) != len(chunk.keys):
                    raise RuntimeError("NVFP4 linker PUT returned incomplete results")
                result.extend(values)
        return results

    def _get(self, transfers):
        for transfer in transfers:
            for chunk, rows in self._chunks(transfer):
                keys, multiplier = self.storage._get_hybrid_page_component_keys(
                    list(chunk.keys), chunk
                )
                keys = self.storage._tag_keys(keys)
                ptrs, sizes = self.storage._build_pool_buffer_meta(
                    self.storage.registered_pools[chunk.name],
                    chunk.host_indices, keys, multiplier, None,
                )
                received = self.storage._get_batch_zero_copy_impl(keys, ptrs, sizes)
                expected = [sum(size) if isinstance(size, list) else size for size in sizes]
                if received is None or isinstance(received, int) or list(received) != expected:
                    return False  # Never decode a partial or stale staging page.
                if transfer.name == PoolName.KV:
                    for layer, target in enumerate(self.layers):
                        self.codec.decode_rows(
                            self.workspace[layer, : len(chunk.keys)].view(-1, self.codec.row_bytes),
                            rows, target,
                        )
        return True

    def _run(self, operation, *args):
        with self.lock, torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            try:
                return operation(*args)
            finally:
                # Future/result, reset/close and the next staging owner all wait
                # for every encode/decode, including an exceptional partial one.
                self.stream.synchronize()

    def batch_set_v2(self, transfers, extra_info=None):
        return self._run(self._set, transfers, extra_info)

    def load_all_layers(self, transfers):
        return self._run(self._get_checked, transfers)

    def _get_checked(self, transfers):
        ok = self._get(transfers)
        if ok and self.get_recheck and any(
            t.name == PoolName.KV and len(t.keys) > self.chunk_pages for t in transfers
        ):
            st = self.recheck_stats
            st["seen"] += 1
            if int(st["seen"] * self.get_recheck) != int((st["seen"] - 1) * self.get_recheck):
                self._recheck(transfers)
        return ok

    def _recheck(self, transfers):
        """Read each KV chunk of the finished load again, fenced, decode it into a scratch copy of the rows the load
        wrote and compare. Stored objects are immutable and still under the load's read lease, so a correct load matches
        byte for byte; a differing page was decoded from staging bytes the next chunk's read had overwritten."""
        st = self.recheck_stats
        rows_max = self.chunk_pages * self.page_size
        if self._recheck_scratch is None:
            self._recheck_scratch = (
                torch.empty((rows_max, *self.layers[0].shape[1:]), dtype=self.layers[0].dtype, device=self.device),
                torch.arange(rows_max, device=self.device),
            )
        scratch, local = self._recheck_scratch
        pages = bad = 0
        detail = []
        for transfer in transfers:
            if transfer.name != PoolName.KV:
                continue
            for index, (chunk, rows) in enumerate(self._chunks(transfer)):
                keys, multiplier = self.storage._get_hybrid_page_component_keys(list(chunk.keys), chunk)
                keys = self.storage._tag_keys(keys)
                ptrs, sizes = self.storage._build_pool_buffer_meta(
                    self.storage.registered_pools[chunk.name], chunk.host_indices, keys, multiplier, None,
                )
                self.stream.synchronize()  # the load's (or the last check's) decode is done before staging is reused
                received = self.storage._get_batch_zero_copy_impl(keys, ptrs, sizes)
                expected = [sum(size) if isinstance(size, list) else size for size in sizes]
                if received is None or isinstance(received, int) or list(received) != expected:
                    st["read_fail"] += 1  # gone between the passes: no verdict on this load
                    return
                n, m = len(chunk.keys), rows.numel()
                differ = torch.zeros(n, dtype=torch.bool, device=self.device)
                for layer, target in enumerate(self.layers):
                    loaded = target.index_select(0, rows)
                    scratch[:m].copy_(loaded)  # bytes the decode does not write compare equal
                    self.codec.decode_rows(
                        self.workspace[layer, :n].view(-1, self.codec.row_bytes), local[:m], scratch,
                    )
                    self.stream.synchronize()
                    differ |= (scratch[:m] != loaded).view(n, -1).any(1)
                nbad = int(differ.sum())
                pages += n
                bad += nbad
                if nbad:
                    detail.append(f"chunk {index}: {nbad} of {n} pages")
        st["loads"] += 1
        st["pages"] += pages
        st["bad_pages"] += bad
        st["bad_loads"] += bad > 0
        if bad or st["loads"] % 20 == 1:
            (logger.warning if bad else logger.info)(
                "Mooncake host NVFP4 get recheck: this load %d of %d pages differ (%s); totals %s",
                bad, pages, "; ".join(detail) or "none", st,
            )


class MooncakeNVFP4Linker(MooncakeDirectLinker):
    def __init__(self, server_args, params, **kwargs):
        config = kwargs.get("extra_config") or {}
        if config.get("load_mode") != "prefetch" or get_parallel().tp_size != 1:
            raise ValueError("NVFP4 Mooncake linker requires prefetch mode and TP1")
        self.host_start_layer = params.token_to_kv_pool_allocator.get_kvcache().start_layer
        super().__init__(server_args, params, **kwargs)

    def register_buffers(self):
        self.storage = NVFP4MooncakeIO(
            self.storage, self.pool_group, self.host_start_layer
        )
        self.storage.register_buffers()

    def load_all_layers(self, transfers):
        return self.storage.load_all_layers(transfers)
