"""Materialize hybrid device pools for external cache linkers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import partial
from itertools import chain
from typing import Any, NamedTuple

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dcp.layout import maybe_dcp_kernel_indices
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_memory, get_parallel


class DevicePoolEntry:
    """Zero-copy linker view over one physical device pool."""

    def __init__(
        self,
        *,
        name: PoolName,
        indices_from_pool: PoolName,
        device_pool: Any,
        components: Sequence[Sequence[torch.Tensor]],
        layer_mapping: dict[int, int | Sequence[int]],
        page_size: int,
        rows_are_pages: bool,
        packed: bool = True,
        pcp_replicated: bool = False,
        tp_replicated: bool = False,
        dcp_sharded: bool = False,
        index_mapper: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.name = name
        self.indices_from_pool = indices_from_pool
        self.device_pool = device_pool
        self.components = [list(component) for component in components]
        self.layer_mapping = layer_mapping
        self.page_size = page_size
        self.packed = packed
        # PCP can replicate DSA pages while the same group holds sharded KDA
        # states. Replication is a property of a physical pool, not the group.
        self.pcp_replicated = pcp_replicated
        # Same per-pool granularity along the attention-TP axis: an MLA-family
        # pool has no head dimension to shard, so every attention-TP rank holds
        # byte-identical pages, while KDA state in the same group is sharded.
        self.tp_replicated = tp_replicated
        self.dcp_sharded = dcp_sharded
        self._index_mapper = index_mapper
        self._page_offsets = torch.arange(page_size)
        self._row_span = 1 if rows_are_pages else page_size

        if not self.components or any(not component for component in self.components):
            raise ValueError(f"Device pool {name} has no storage buffers.")
        self.kv_buffer = [buffer for group in self.components for buffer in group]
        self._row_count = min(buffer.shape[0] for buffer in self.kv_buffer)

        self.buffer_meta = [
            [
                (
                    buffer.data_ptr(),
                    buffer.stride(0) * buffer.element_size(),
                    buffer.nbytes // buffer.shape[0] * self._row_span,
                )
                for buffer in component
            ]
            for component in self.components
        ]

        self._component_offsets = []
        offset = 0
        for component in self.buffer_meta:
            if not packed:
                offset = 0
            offsets = []
            for _, _, size in component:
                offsets.append(offset)
                offset += size
            self._component_offsets.append(offsets)

    def get_hybrid_pool_buffer(self) -> list[torch.Tensor]:
        return self.kv_buffer

    def translate_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self._index_mapper(indices) if self._index_mapper else indices

    # Device-tensor indices copied to the CPU here (SGLANG_LINKER_CPU_INDICES
    # keeps the linker's read and offload threads at zero).
    cuda_index_copies = 0

    def _rows(self, indices: torch.Tensor) -> list[int]:
        if indices.is_cuda:
            DevicePoolEntry.cuda_index_copies += 1
        slots = indices.detach().to(device="cpu", dtype=torch.int64).flatten()
        if slots.numel() % self.page_size:
            raise ValueError(
                f"Pool {self.name} got {slots.numel()} indices, expected a "
                f"multiple of page_size={self.page_size}."
            )
        if not slots.numel():
            return []

        pages = slots.reshape(-1, self.page_size)
        starts = pages[:, 0]
        if torch.any(starts.remainder(self.page_size)) or not torch.equal(
            pages, starts[:, None] + self._page_offsets
        ):
            raise ValueError(f"Pool {self.name} requires aligned contiguous pages.")
        rows = (
            starts.div(self.page_size, rounding_mode="floor")
            if self._row_span == 1
            else starts
        )
        first_row = int(rows.min())
        last_row = int(rows.max()) + self._row_span
        if first_row < 0 or last_row > self._row_count:
            raise ValueError(
                f"Pool {self.name} row range [{first_row}, {last_row}) exceeds "
                f"buffer shapes {[tuple(buffer.shape) for buffer in self.kv_buffer]}."
            )
        return rows.tolist()

    def get_page_buffer_meta(self, indices: torch.Tensor):
        rows = self._rows(indices)
        ptrs = [
            base_ptr + row * row_stride
            for row in rows
            for component in self.buffer_meta
            for base_ptr, row_stride, _ in component
        ]
        sizes = [
            size
            for _ in rows
            for component in self.buffer_meta
            for _, _, size in component
        ]
        return ptrs, sizes

    def prepare_locations(self, indices: torch.Tensor) -> list[int]:
        return self._rows(indices)

    def get_prepared_layer_range_meta(self, locations: list[int], layer: int):
        mapped = self.layer_mapping.get(layer)
        if mapped is None:
            return None
        buffer_indices = [mapped] if isinstance(mapped, int) else list(mapped)

        items = []
        for component, offsets in zip(self.buffer_meta, self._component_offsets):
            for buffer_index in buffer_indices:
                base_ptr, row_stride, size = component[buffer_index]
                items.append((base_ptr, row_stride, size, offsets[buffer_index]))

        ptrs, sizes, offsets = [], [], []
        for row in locations:
            row_ptrs = [
                base_ptr + row * row_stride for base_ptr, row_stride, _, _ in items
            ]
            row_sizes = [size for _, _, size, _ in items]
            row_offsets = [offset for _, _, _, offset in items]
            if self.packed:
                ptrs.append(row_ptrs)
                sizes.append(row_sizes)
                offsets.append(row_offsets)
            else:
                ptrs.extend([[value] for value in row_ptrs])
                sizes.extend([[value] for value in row_sizes])
                offsets.extend([[value] for value in row_offsets])
        return ptrs, sizes, offsets

    def _layer_items(self, layer: int):
        """(base_ptr, row_stride, size, offset) in get_prepared_layer_range_meta
        order for one layer, or None when the layer has no buffer here."""
        mapped = self.layer_mapping.get(layer)
        if mapped is None:
            return None
        buffer_indices = [mapped] if isinstance(mapped, int) else list(mapped)
        return [
            (*component[buffer_index][:2], component[buffer_index][2], offsets[buffer_index])
            for component, offsets in zip(self.buffer_meta, self._component_offsets)
            for buffer_index in buffer_indices
        ]

    def key_range_table(self, num_layers: int):
        """SGLANG_LINKER_VECTOR_META: every layer's items of one key, in the order
        the per-key load concatenates them, as int64 arrays (src, base, stride,
        size, offset); src indexes the location (always 0 here). Packed only."""
        cached = getattr(self, "_key_range_table", None)
        if cached is not None and cached[0] == num_layers:
            return cached[1]
        if not self.packed:
            return None
        rows = []
        for layer in range(num_layers):
            for item in self._layer_items(layer) or ():
                rows.append((0, *item))
        table = np.array(rows, dtype=np.int64).reshape(-1, 5)
        self._key_range_table = (num_layers, table)
        return table

    def key_range_rows(self, locations) -> np.ndarray:
        return np.asarray(locations, dtype=np.int64).reshape(-1, 1)


_CHECKSUM_WEIGHTS: dict = {}


def _weighted_byte_sum(rows: torch.Tensor) -> torch.Tensor:
    """Position-weighted sum of a slice's bytes (int64, on its device)."""
    data = rows.contiguous().view(torch.uint8).flatten()
    key = (data.numel(), data.device)
    weights = _CHECKSUM_WEIGHTS.get(key)
    if weights is None:
        weights = torch.arange(data.numel(), device=data.device, dtype=torch.int64) % 65521 + 1
        _CHECKSUM_WEIGHTS[key] = weights
    return (data.to(torch.int64) * weights).sum()


def _entry_page_checksum(entry, location) -> torch.Tensor:
    if isinstance(entry, MergedPageEntry):
        first = _entry_page_checksum(entry.first, location[0])
        return first * 1000003 + _entry_page_checksum(entry.second, location[1])
    total = None
    for index, buffer in enumerate(entry.kv_buffer):
        value = _weighted_byte_sum(buffer[location : location + entry._row_span]) * (index + 1)
        total = value if total is None else total + value
    return total


def page_checksums(entry, locations) -> list[int]:
    """SGLANG_LINKER_PAGE_CHECKSUM: one checksum per page location over every
    buffer the page's store object packs (KV, indexer, or the state)."""
    if not locations:
        return []
    return torch.stack([_entry_page_checksum(entry, loc) for loc in locations]).tolist()


def coalesce_key_range_table(table: np.ndarray) -> np.ndarray:
    """Merge consecutive items that stay contiguous for every row: same source
    location and stride, the next item's device base right after this one's bytes,
    and the next object offset right after this one's. Exact for any row."""
    if len(table) < 2:
        return table
    out = [list(table[0])]
    for src, base, stride, size, offset in table[1:].tolist():
        last = out[-1]
        if (
            src == last[0]
            and stride == last[2]
            and base == last[1] + last[3]
            and offset == last[4] + last[3]
        ):
            last[3] += size
        else:
            out.append([src, base, stride, size, offset])
    return np.array(out, dtype=np.int64).reshape(-1, 5)


class RangeArrays(NamedTuple):
    """SGLANG_LINKER_STORE_ARRAYS: per-key ranges as flat uint64 arrays; key i owns
    entries [indptr[i], indptr[i + 1]). Field order is the store's array call."""

    ptrs: np.ndarray
    sizes: np.ndarray
    offsets: np.ndarray
    indptr: np.ndarray

    @classmethod
    def from_lists(cls, ptrs, sizes, offsets) -> "RangeArrays":
        counts = np.fromiter(map(len, ptrs), dtype=np.uint64, count=len(ptrs))
        indptr = np.zeros(len(ptrs) + 1, dtype=np.uint64)
        np.cumsum(counts, out=indptr[1:])
        total = int(indptr[-1])

        def flat(rows):
            return np.fromiter(chain.from_iterable(rows), dtype=np.uint64, count=total)

        return cls(flat(ptrs), flat(sizes), flat(offsets), indptr)

    @classmethod
    def from_rows(cls, ptrs, size_row, offset_row) -> "RangeArrays":
        """``ptrs`` is [keys, items]; every key shares one size and offset row."""
        count, items = ptrs.shape
        return cls(
            np.ascontiguousarray(ptrs, dtype=np.int64).reshape(-1).view(np.uint64),
            np.tile(np.asarray(size_row, dtype=np.uint64), count),
            np.tile(np.asarray(offset_row, dtype=np.uint64), count),
            np.arange(count + 1, dtype=np.uint64) * np.uint64(items),
        )

    @classmethod
    def concat(cls, parts: list["RangeArrays"]) -> "RangeArrays":
        if len(parts) == 1:
            return parts[0]
        shifts = np.cumsum([0] + [int(p.indptr[-1]) for p in parts[:-1]], dtype=np.uint64)
        return cls(
            np.concatenate([p.ptrs for p in parts]),
            np.concatenate([p.sizes for p in parts]),
            np.concatenate([p.offsets for p in parts]),
            np.concatenate(
                [parts[0].indptr[:1]]
                + [p.indptr[1:] + shift for p, shift in zip(parts, shifts)]
            ),
        )

    def expected(self) -> list[int]:
        """Bytes per key: the store's result for a complete read."""
        total = np.zeros(len(self.sizes) + 1, dtype=np.uint64)
        np.cumsum(self.sizes, out=total[1:])
        return (total[self.indptr[1:]] - total[self.indptr[:-1]]).tolist()

    def select(self, keys: list[int]) -> "RangeArrays":
        index = np.asarray(keys, dtype=np.int64)
        starts = self.indptr[index].astype(np.int64)
        counts = self.indptr[index + 1].astype(np.int64) - starts
        indptr = np.zeros(len(index) + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        gather = np.repeat(starts - indptr[:-1], counts) + np.arange(indptr[-1])
        return RangeArrays(
            self.ptrs[gather],
            self.sizes[gather],
            self.offsets[gather],
            indptr.astype(np.uint64),
        )

    def to_lists(self):
        bounds = self.indptr.tolist()
        return tuple(
            [field[a:b] for a, b in zip(bounds, bounds[1:])]
            for field in (self.ptrs.tolist(), self.sizes.tolist(), self.offsets.tolist())
        )


def vector_key_range_meta(
    entry, locations, num_layers: int, coalesce: bool = False, arrays: bool = False
):
    """Per-key (ptrs, sizes, offsets) lists for a packed or merged entry, equal to
    concatenating get_prepared_layer_range_meta over layers 0..num_layers-1; None
    when the entry has no vector table (unpacked). With ``coalesce``
    (SGLANG_LINKER_COALESCE_RANGES) contiguous items are merged and every key
    shares one size list and one offset list (read-only for the store call). With
    ``arrays`` (SGLANG_LINKER_STORE_ARRAYS) the same ranges come as a RangeArrays."""
    table = entry.key_range_table(num_layers)
    if table is None:
        return None
    if coalesce:
        cached = getattr(entry, "_coalesced_table", None)
        if cached is None or cached[0] is not table:
            cached = (table, coalesce_key_range_table(table))
            entry._coalesced_table = cached
        table = cached[1]
    rows = entry.key_range_rows(locations)
    src, base, stride, size, offset = table.T
    ptrs = base[None, :] + rows[:, src] * stride[None, :]
    count = rows.shape[0]
    if arrays:
        return RangeArrays.from_rows(ptrs, size, offset)
    if coalesce:
        size_row, offset_row = size.tolist(), offset.tolist()
        return ptrs.tolist(), [size_row] * count, [offset_row] * count
    sizes = np.broadcast_to(size, (count, size.shape[0]))
    offsets = np.broadcast_to(offset, (count, offset.shape[0]))
    return ptrs.tolist(), sizes.tolist(), offsets.tolist()


def page_buffer_table(entry):
    """SGLANG_LINKER_VECTOR_PUT: (src, base, stride, size) per buffer of one page
    object, in get_page_buffer_meta order (``first``'s buffers, then ``second``'s
    for a merged entry); None for an entry it does not cover."""
    cached = getattr(entry, "_page_buffer_table", None)
    if cached is not None:
        return cached
    if isinstance(entry, MergedPageEntry):
        parts = [(0, entry.first), (1, entry.second)]
    elif isinstance(entry, DevicePoolEntry):
        parts = [(0, entry)]
    else:
        return None
    rows = [
        (src, base, stride, size)
        for src, part in parts
        for component in part.buffer_meta
        for base, stride, size in component
    ]
    table = np.array(rows, dtype=np.int64).reshape(-1, 4)
    entry._page_buffer_table = table
    return table


def vector_page_buffer_meta(entry, indices):
    """Per-page (ptrs, sizes) equal to get_page_buffer_meta followed by the store's
    per-key packing: one list per page when a page has several buffers, a plain
    int otherwise. None when ``entry`` has no table."""
    table = page_buffer_table(entry)
    if table is None:
        return None
    rows = entry.key_range_rows(entry.prepare_locations(indices))
    src, base, stride, size = table.T
    ptrs = base[None, :] + rows[:, src] * stride[None, :]
    count = rows.shape[0]
    if table.shape[0] == 1:
        return ptrs[:, 0].tolist(), [int(size[0])] * count
    size_row = size.tolist()
    return ptrs.tolist(), [list(size_row) for _ in range(count)]


class MergedPageEntry:
    """Two DevicePoolEntries of the same pages stored as one object per page:
    ``second``'s bytes follow ``first``'s (SGLANG_LINKER_MERGE_INDEXER, KV then
    the DSA indexer). Halves the store objects per page and stage, and with them
    the master's per-object work on lookups and loads (Rhea 60076)."""

    def __init__(self, first: DevicePoolEntry, second: DevicePoolEntry, object_name: str):
        flags = ("indices_from_pool", "page_size", "pcp_replicated", "tp_replicated", "dcp_sharded")
        if not (first.packed and second.packed) or any(
            getattr(first, f) != getattr(second, f) for f in flags
        ):
            raise ValueError(f"Cannot merge {first.name} and {second.name} into one object.")
        self.first, self.second = first, second
        self.name, self.object_name = first.name, object_name
        for flag in flags:
            setattr(self, flag, getattr(first, flag))
        self.device_pool = first.device_pool
        self.packed = True
        self.layer_mapping = {**second.layer_mapping, **first.layer_mapping}
        self.kv_buffer = first.kv_buffer + second.kv_buffer
        self.buffer_meta = first.buffer_meta + second.buffer_meta
        self._first_bytes = sum(size for c in first.buffer_meta for *_, size in c)

    def get_hybrid_pool_buffer(self) -> list[torch.Tensor]:
        return self.kv_buffer

    def translate_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.first.translate_indices(indices)

    def get_page_buffer_meta(self, indices: torch.Tensor):
        ptrs1, sizes1 = self.first.get_page_buffer_meta(indices)
        ptrs2, sizes2 = self.second.get_page_buffer_meta(indices)
        pages = indices.numel() // self.page_size
        a, b = len(ptrs1) // max(pages, 1), len(ptrs2) // max(pages, 1)
        ptrs, sizes = [], []
        for page in range(pages):
            ptrs += ptrs1[page * a : (page + 1) * a] + ptrs2[page * b : (page + 1) * b]
            sizes += sizes1[page * a : (page + 1) * a] + sizes2[page * b : (page + 1) * b]
        return ptrs, sizes

    def prepare_locations(self, indices: torch.Tensor) -> list[tuple[int, int]]:
        return list(
            zip(self.first.prepare_locations(indices), self.second.prepare_locations(indices))
        )

    def get_prepared_layer_range_meta(self, locations, layer: int):
        meta1 = self.first.get_prepared_layer_range_meta([l[0] for l in locations], layer)
        meta2 = self.second.get_prepared_layer_range_meta([l[1] for l in locations], layer)
        if meta1 is None and meta2 is None:
            return None
        empty = ([[] for _ in locations],) * 3
        (p1, s1, o1), (p2, s2, o2) = meta1 or empty, meta2 or empty
        offsets = [
            first + [offset + self._first_bytes for offset in second]
            for first, second in zip(o1, o2)
        ]
        return (
            [a + b for a, b in zip(p1, p2)],
            [a + b for a, b in zip(s1, s2)],
            offsets,
        )

    def key_range_table(self, num_layers: int):
        """SGLANG_LINKER_VECTOR_META: per layer, ``first``'s items (src 0) then
        ``second``'s (src 1, offset past ``first``'s bytes), as in
        get_prepared_layer_range_meta."""
        cached = getattr(self, "_key_range_table", None)
        if cached is not None and cached[0] == num_layers:
            return cached[1]
        rows = []
        for layer in range(num_layers):
            for item in self.first._layer_items(layer) or ():
                rows.append((0, *item))
            for base, stride, size, offset in self.second._layer_items(layer) or ():
                rows.append((1, base, stride, size, offset + self._first_bytes))
        table = np.array(rows, dtype=np.int64).reshape(-1, 5)
        self._key_range_table = (num_layers, table)
        return table

    def key_range_rows(self, locations) -> np.ndarray:
        return np.asarray(locations, dtype=np.int64).reshape(-1, 2)


class DevicePoolGroup:
    """Physical device pools sharing one logical linker layer range."""

    def __init__(
        self,
        entries: Sequence[DevicePoolEntry],
        num_layers: int,
        page_size: int,
        *,
        rank_replicated: bool = False,
    ):
        self.entries = list(entries)
        self.entry_map = {entry.name: entry for entry in entries}
        if len(self.entries) != len(self.entry_map):
            raise ValueError("DevicePoolGroup contains duplicate pool names.")
        self.sources = {entry.name: entry.indices_from_pool for entry in self.entries}
        self.num_layers = num_layers
        self.page_size = page_size
        self.rank_replicated = rank_replicated
        self.kv_buffer = None

    def resolve_transfers(
        self,
        transfers: list[PoolTransfer],
        *,
        allow_partial: bool = False,
        allow_missing_kv: bool = False,
    ) -> list[PoolTransfer]:
        """Expand logical component transfers into physical device pools."""
        by_name = {transfer.name: transfer for transfer in transfers}
        kv = by_name.get(PoolName.KV)
        if not any(transfer.keys for transfer in transfers):
            return []
        if not allow_missing_kv and (kv is None or not kv.keys):
            return []
        if not allow_partial and not set(self.sources.values()) <= set(by_name):
            return []

        resolved = []
        for name, source_name in self.sources.items():
            source = by_name.get(source_name)
            if source is None or not source.keys:
                continue
            indices = source.device_indices
            entry = self.entry_map[name]
            # CPU-known slots (load reserve) stay on the CPU unless an index
            # mapper (DCP) must run on the device copy.
            if source.cpu_indices is not None and not _maps_indices(entry):
                indices = source.cpu_indices
            resolved.append(
                replace(
                    source,
                    name=name,
                    host_indices=(
                        entry.translate_indices(indices)
                        if indices is not None
                        else None
                    ),
                    keys=list(source.keys),
                    hit_policy=(
                        PoolHitPolicy.ALL_PAGES
                        if source_name == PoolName.KV
                        else source.hit_policy
                    ),
                    indices_from_pool=None,
                )
            )
        return resolved


def attach_cpu_indices(transfers: list[PoolTransfer]) -> list[PoolTransfer]:
    """SGLANG_LINKER_CPU_INDICES: on the scheduler thread, give each transfer with
    device slot indices a CPU copy (non-blocking D2H on the current stream), so the
    linker threads read CPU indices and never sync a D2H copy against the forward.
    The caller records its ready event after this, so the copy has landed once a
    thread waits on that event."""
    out = []
    for transfer in transfers:
        indices = transfer.device_indices
        if transfer.cpu_indices is None and indices is not None and indices.is_cuda:
            cpu = torch.empty(indices.shape, dtype=indices.dtype, pin_memory=True)
            cpu.copy_(indices, non_blocking=True)
            transfer = replace(transfer, cpu_indices=cpu)
        out.append(transfer)
    return out


def _maps_indices(entry) -> bool:
    first = getattr(entry, "first", entry)
    return getattr(first, "_index_mapper", None) is not None


def _deepseek_v4_state_views(state_pools: list[Any], global_layers: list[int]):
    views = []
    for layer in global_layers:
        pool = state_pools[layer]
        state = pool.kv_score_buffer.kv_score
        ring = int(pool.ring_size)
        usable = state.shape[0] // ring * ring
        views.append(
            state.view(torch.uint8)
            .reshape(state.shape[0], -1)[:usable]
            .reshape(usable // ring, -1)
        )
    return views


def _with_packed_draft_mapping(
    layer_mapping: dict[int, int],
    *,
    target_device_layer_num: int,
    draft_layer_num: int,
) -> dict[int, int | tuple[int, ...]]:
    """Attach draft depth N to the same transfer layer as target layer N."""
    if draft_layer_num > len(layer_mapping):
        raise ValueError(
            "Packed draft layers exceed the target transfer layer count: "
            f"{draft_layer_num} > {len(layer_mapping)}."
        )
    result: dict[int, int | tuple[int, ...]] = dict(layer_mapping)
    for depth in range(draft_layer_num):
        result[depth] = (layer_mapping[depth], target_device_layer_num + depth)
    return result


def _build_deepseek_v4_device_pool_group(
    kvcache: Any,
    page_size: int,
    mtp_draft_device_pools: tuple[Any, ...] = (),
) -> DevicePoolGroup:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import HiSparseC4DevicePool
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _dsv4_compressed_region_buffers,
        _dsv4_indexer_regions,
        _dsv4_low_ratio_regions,
        _require_single_row_dsv4_swa_pages,
        _resolve_deepseek_v4_layer_mappings,
    )

    if isinstance(kvcache.c4_kv_pool, HiSparseC4DevicePool):
        raise ValueError("The direct external linker does not support HiSparse.")

    mappings = _resolve_deepseek_v4_layer_mappings(kvcache)
    is_unified_kv = getattr(kvcache, "_unified_kv", False)
    # Encoder SWA replay keeps the target window per request; the draft SWA is
    # dropped with it, a known draft accept-length cost.
    has_paged_swa = not is_unified_kv and kvcache.swa_kv_pool is not None
    entries = []
    if has_paged_swa:
        _require_single_row_dsv4_swa_pages(
            logical_page_size=kvcache.swa_page_size,
            physical_page_size=kvcache.swa_kv_pool.page_size,
            consumer="DeepSeek-V4 direct external linker",
        )
        if kvcache.swa_page_size != page_size:
            raise ValueError(
                "DeepSeek V4 SWA page size must match the tree page size: "
                f"{kvcache.swa_page_size} != {page_size}."
            )
        draft_swa_buffers = [
            buffer
            for pool in mtp_draft_device_pools
            for buffer in pool.swa_kv_pool.kv_buffer
        ]
        swa_mapping = _with_packed_draft_mapping(
            mappings.swa,
            target_device_layer_num=len(kvcache.swa_kv_pool.kv_buffer),
            draft_layer_num=len(draft_swa_buffers),
        )
        entries.append(
            DevicePoolEntry(
                name=PoolName.SWA,
                indices_from_pool=PoolName.SWA,
                device_pool=kvcache.swa_kv_pool,
                components=[[*kvcache.swa_kv_pool.kv_buffer, *draft_swa_buffers]],
                layer_mapping=swa_mapping,
                page_size=page_size,
                rows_are_pages=True,
            )
        )

    def add(name, source, pool, buffers, layer_mapping):
        if layer_mapping:
            entries.append(
                DevicePoolEntry(
                    name=name,
                    indices_from_pool=source,
                    device_pool=pool,
                    components=[buffers],
                    layer_mapping=layer_mapping,
                    page_size=page_size,
                    rows_are_pages=True,
                )
            )

    if mappings.c4:
        c4_buffers, _ = _dsv4_compressed_region_buffers(kvcache, 4)
        add(
            PoolName.DEEPSEEK_V4_C4,
            PoolName.KV,
            kvcache.c4_kv_pool,
            c4_buffers,
            mappings.c4,
        )
        for region in _dsv4_indexer_regions(kvcache, page_size):
            add(
                region.name,
                PoolName.KV,
                kvcache.c4_indexer_kv_pool,
                region.device_buffers,
                mappings.c4,
            )
    if mappings.c128:
        c128_buffers, _ = _dsv4_compressed_region_buffers(kvcache, 128)
        add(
            PoolName.DEEPSEEK_V4_C128,
            PoolName.KV,
            kvcache.c128_kv_pool,
            c128_buffers,
            mappings.c128,
        )
    for name, device_pool, buffers, layer_mapping in _dsv4_low_ratio_regions(
        kvcache, page_size
    ):
        add(name, PoolName.KV, device_pool, buffers, layer_mapping)

    # fp8 two-pool unified_kv: one row index addresses both kv_buffer (fp8 nope) and
    # kv_buffer_rope (bf16 rope), but unified_region_buffers() above returns the nope
    # pool only -- shipping that alone restores 512 of every 640 bytes and leaves stale
    # rope, which is wrong output rather than a crash. Mirrors _dsv4_rope_sibling() on
    # the HiCache side; returns None on non-fp8 layouts, so bf16 is untouched.
    for ratio, rope_name, rope_pool, rope_mapping in (
        (4, PoolName.DEEPSEEK_V4_C4_ROPE, kvcache.c4_kv_pool, mappings.c4),
        (128, PoolName.DEEPSEEK_V4_C128_ROPE, kvcache.c128_kv_pool, mappings.c128),
    ):
        rope_region = (
            kvcache.unified_rope_region_buffers(ratio) if is_unified_kv else None
        )
        if rope_region is None:
            continue
        rope_buffers, _ = rope_region
        add(rope_name, PoolName.KV, rope_pool, rope_buffers, rope_mapping)
    if not is_unified_kv:
        add(
            PoolName.DEEPSEEK_V4_C4_STATE,
            PoolName.SWA,
            kvcache.compress_state_pools,
            _deepseek_v4_state_views(
                kvcache.compress_state_pools,
                mappings.c4_state_global_layers,
            ),
            mappings.c4_state,
        )
        add(
            PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            PoolName.SWA,
            kvcache.indexer_compress_state_pools,
            _deepseek_v4_state_views(
                kvcache.indexer_compress_state_pools,
                mappings.c4_state_global_layers,
            ),
            mappings.c4_state,
        )
    if not has_paged_swa and any(
        entry.indices_from_pool == PoolName.SWA for entry in entries
    ):
        raise ValueError(
            "DeepSeek-V4 compress state pools take SWA page indices, which "
            "encoder SWA replay does not keep."
        )
    return DevicePoolGroup(
        entries,
        mappings.transfer_layer_id_max,
        page_size,
        rank_replicated=True,
    )


def _build_dsa_device_pool_group(
    kvcache: Any,
    page_size: int,
    mtp_draft_device_pools: tuple[Any, ...] = (),
) -> DevicePoolGroup:
    if kvcache.page_size != page_size:
        raise ValueError(
            "DSA KV page size must match the tree page size: "
            f"{kvcache.page_size} != {page_size}."
        )
    num_layers = kvcache.layer_num
    if any(pool.page_size != page_size for pool in mtp_draft_device_pools):
        raise ValueError("DSA MTP page size must match the tree page size.")
    draft_kv_buffers = [
        buffer for pool in mtp_draft_device_pools for buffer in pool.kv_buffer
    ]
    draft_indexer_buffers = [
        buffer
        for pool in mtp_draft_device_pools
        for buffer in pool.index_k_with_scale_buffer
    ]
    if len(draft_kv_buffers) != len(draft_indexer_buffers):
        raise ValueError("DSA MTP KV and indexer draft layer counts must match.")
    layer_mapping = _with_packed_draft_mapping(
        {layer: layer for layer in range(num_layers)},
        target_device_layer_num=num_layers,
        draft_layer_num=len(draft_kv_buffers),
    )
    entries = [
        DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=kvcache,
            components=[[*kvcache.kv_buffer, *draft_kv_buffers]],
            layer_mapping=layer_mapping,
            page_size=page_size,
            rows_are_pages=False,
        ),
        DevicePoolEntry(
            name=PoolName.INDEXER,
            indices_from_pool=PoolName.KV,
            device_pool=kvcache,
            components=[[*kvcache.index_k_with_scale_buffer, *draft_indexer_buffers]],
            layer_mapping=layer_mapping,
            page_size=page_size,
            rows_are_pages=True,
        ),
    ]
    return DevicePoolGroup(entries, num_layers, page_size, rank_replicated=True)


def _mha_device_pool_entry(
    name: PoolName,
    pool: Any,
    layer_mapping: dict[int, int],
    page_size: int,
    indices_from_pool: PoolName | None = None,
) -> DevicePoolEntry:
    # TODO(alphabetc1): support mxfp8.
    if pool.k_scale_buffer is not None or pool.native_k_scale_buffer is not None:
        raise ValueError(
            "The direct external linker does not support scaled KV caches yet."
        )
    num_layers = len(pool.k_buffer)
    return DevicePoolEntry(
        name=name,
        indices_from_pool=indices_from_pool or name,
        device_pool=pool,
        components=[[*pool.k_buffer, *pool.v_buffer]],
        layer_mapping={
            layer: (index, num_layers + index) for layer, index in layer_mapping.items()
        },
        page_size=page_size,
        rows_are_pages=False,
    )


def _mamba_device_pool_entry(
    req_to_token_pool: Any, start_layer: int
) -> DevicePoolEntry:
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _stage_local_layer_mapping,
    )

    # TODO(alphabetc1): support unified memory for hybrid Mamba models.
    if get_memory().enable_unified_memory:
        raise ValueError(
            "The direct external linker does not support --enable-unified-memory "
            "on hybrid Mamba models yet."
        )
    # TODO(alphabetc1): support int8 Mamba checkpoints and Mamba side states.
    if (
        req_to_token_pool.mamba_ckpt_pool is not None
        or req_to_token_pool.short_conv_pool.enabled
        or req_to_token_pool.ngram_pool.enabled
    ):
        raise ValueError(
            "The direct external linker does not support int8 Mamba checkpoints "
            "or Mamba side states yet."
        )

    mamba_pool = req_to_token_pool.mamba_pool
    mamba_cache = mamba_pool.mamba_cache
    layers = range(mamba_pool.num_mamba_layers)
    # One row per state slot; conv-only models carry an empty temporal state.
    components = [
        [state[layer] for layer in layers]
        for state in (mamba_cache.temporal, *mamba_cache.conv)
        if state.numel() > 0
    ]
    return DevicePoolEntry(
        name=PoolName.MAMBA,
        indices_from_pool=PoolName.MAMBA,
        device_pool=mamba_pool,
        components=components,
        layer_mapping=_stage_local_layer_mapping(
            req_to_token_pool.mamba_map, start_layer
        ),
        page_size=1,
        rows_are_pages=True,
    )


def _build_mamba_swa_device_pool_group(
    kvcache: Any, params: Any, page_size: int
) -> DevicePoolGroup:
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _swa_layer_mappings,
    )

    full_mapping, swa_mapping = _swa_layer_mappings(kvcache)
    entries = [
        _mha_device_pool_entry(
            PoolName.KV, kvcache.full_kv_pool, full_mapping, page_size
        ),
        _mha_device_pool_entry(
            PoolName.SWA, kvcache.swa_kv_pool, swa_mapping, page_size
        ),
        _mamba_device_pool_entry(params.req_to_token_pool, kvcache.start_layer),
    ]
    num_layers = max(layer for entry in entries for layer in entry.layer_mapping)
    return DevicePoolGroup(entries, num_layers + 1, page_size)


def _build_mamba_device_pool_group(
    kvcache: Any, params: Any, page_size: int
) -> DevicePoolGroup:
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _stage_local_layer_mapping,
    )
    from sglang.srt.mem_cache.memory_pool import (
        DSATokenToKVPool,
        HybridLinearKVPool,
        MLATokenToKVPool,
    )

    # TODO(alphabetc1): support other full-attention pools and packed draft KV.
    full_kv_pool = kvcache.full_kv_pool
    if (
        type(kvcache) is not HybridLinearKVPool
        or type(full_kv_pool) not in (MLATokenToKVPool, DSATokenToKVPool)
        or params.mtp_draft_device_pools
    ):
        raise ValueError(
            "The direct external linker supports FULL + MAMBA models only with a "
            "plain MLA or DSA pool, without packed draft KV."
        )
    is_dsa = type(full_kv_pool) is DSATokenToKVPool
    parallel = get_parallel()
    dcp_size = parallel.attn_dcp_size
    dcp_sharded = dcp_size > 1
    # NOTE(kpham-sgl): page_size counts widened ids; under DCP the mapper keeps
    # this rank's 1/dcp_size of a tree page as one physical page of
    # page_size // dcp_size rows.
    dcp_mapper = (
        partial(
            maybe_dcp_kernel_indices,
            dcp_size=dcp_size,
            dcp_rank=parallel.attn_dcp_rank,
        )
        if dcp_sharded
        else None
    )

    entries = [
        DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=full_kv_pool,
            components=[full_kv_pool.kv_buffer],
            layer_mapping=_stage_local_layer_mapping(
                kvcache.full_attention_layer_id_mapping, kvcache.start_layer
            ),
            page_size=page_size // dcp_size,
            rows_are_pages=False,
            pcp_replicated=is_dsa,
            tp_replicated=True,
            dcp_sharded=dcp_sharded,
            index_mapper=dcp_mapper,
        ),
        _mamba_device_pool_entry(params.req_to_token_pool, kvcache.start_layer),
    ]
    if is_dsa:
        # The indexer has one physical row per logical KV page. Shared-topk
        # layers can have zero-row placeholders; retain only owning layers.
        buffers = full_kv_pool.index_k_with_scale_buffer
        active_layers = [i for i, buffer in enumerate(buffers) if buffer.numel()]
        active = {layer: i for i, layer in enumerate(active_layers)}
        if active:
            mapping = _stage_local_layer_mapping(
                kvcache.full_attention_layer_id_mapping, kvcache.start_layer
            )
            indexer = (
                DevicePoolEntry(
                    name=PoolName.INDEXER,
                    indices_from_pool=PoolName.KV,
                    device_pool=full_kv_pool,
                    components=[[buffers[i] for i in active]],
                    layer_mapping={
                        layer: active[i] for layer, i in mapping.items() if i in active
                    },
                    page_size=page_size // dcp_size,
                    rows_are_pages=True,
                    pcp_replicated=True,
                    tp_replicated=True,
                    dcp_sharded=dcp_sharded,
                    index_mapper=dcp_mapper,
                )
            )
            if envs.SGLANG_LINKER_MERGE_INDEXER.get():
                entries[0] = MergedPageEntry(entries[0], indexer, object_name="ki")
            else:
                entries.append(indexer)
    for draft_pool in params.sidecar_draft_device_pools:
        draft_dcp_sharded = draft_pool.rows_are_dcp_collapsed
        draft_page_size = page_size // dcp_size if draft_dcp_sharded else page_size
        if draft_pool.page_size != draft_page_size:
            raise ValueError("Draft KV page size must match the tree page size.")
        # Draft layer i rides transfer layer i; the draft reads it only after
        # the batch's loads finish.
        layer_mapping = {layer: layer for layer in range(draft_pool.layer_num)}
        if type(draft_pool) is MLATokenToKVPool:
            entry = DevicePoolEntry(
                name=PoolName.DRAFT,
                indices_from_pool=PoolName.KV,
                device_pool=draft_pool,
                components=[draft_pool.kv_buffer],
                layer_mapping=layer_mapping,
                page_size=draft_page_size,
                rows_are_pages=False,
                tp_replicated=True,
                dcp_sharded=draft_dcp_sharded,
                index_mapper=dcp_mapper if draft_dcp_sharded else None,
            )
        else:
            entry = _mha_device_pool_entry(
                PoolName.DRAFT,
                draft_pool,
                layer_mapping,
                page_size,
                indices_from_pool=PoolName.KV,
            )
        entries.append(entry)
    num_layers = max(layer for entry in entries for layer in entry.layer_mapping)
    # Not rank_replicated: the Mamba state is TP-sharded.
    return DevicePoolGroup(entries, num_layers + 1, page_size)


def resolve_hybrid_device_pool_group(
    *,
    kvcache: Any,
    page_size: int,
    params: Any,
    components: set[ComponentType],
) -> DevicePoolGroup:
    """Materialize a direct-linker pool group through the hybrid registry."""
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _select_strategy,
    )

    group = _select_strategy(kvcache, components).build_direct_linker_pool_group(
        kvcache=kvcache,
        params=params,
        page_size=page_size,
    )
    if getattr(params, "sidecar_draft_device_pools", ()) and (
        PoolName.DRAFT not in group.entry_map
    ):
        raise ValueError(
            "The direct external linker does not support this model's draft KV."
        )
    return group
