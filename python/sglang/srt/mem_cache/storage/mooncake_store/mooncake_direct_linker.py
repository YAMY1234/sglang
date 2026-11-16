from __future__ import annotations

import glob
import hashlib
import logging
import os
import re
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from copy import deepcopy
from queue import Empty, Queue

import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.storage.mooncake_store.linker_lease_refresh import (
    LinkerLeaseRefresher,
)
from sglang.srt.mem_cache.unified_cache.linker_mla_dedup import (
    LinkerMLADedupBroadcaster,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LayerWiseLoadCounter,
    UnifiedCacheLinker,
)
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.srt.runtime_context import (
    get_model,
    get_parallel,
)
from sglang.srt.utils import freeze_gc, get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


def _storage_suffix(
    *, rank_replicated: bool, tp_rank: int, attn_cp_rank: int, pp_rank: int
) -> str:
    parts = []
    if not rank_replicated:
        parts.append(f"tp{tp_rank}")
    parts.extend((f"cp{attn_cp_rank}", f"pp{pp_rank}"))
    return "_".join(parts)


def _shared_pool_suffix(
    *,
    tp_shared: bool,
    cp_shared: bool,
    tp_rank: int,
    attn_cp_rank: int,
    pp_rank: int,
    dcp_sharded: bool,
    dcp_size: int,
    dcp_rank: int,
) -> str:
    """Key suffix for a pool whose pages are byte-identical along some axes.

    Dropping an axis makes one object serve every rank on it, so the store
    holds one copy instead of one per rank. Each dropped axis adds its own
    namespace term so a shared object can never alias an older per-rank one.
    The CP-only term keeps its original spelling, so a store written before
    the TP axis existed stays readable.
    """
    parts = []
    if cp_shared:
        parts.append("pcp_shared_v1")
    parts.append("tpshared_v1" if tp_shared else f"tp{tp_rank}")
    if dcp_sharded:
        parts.append(f"dcp{dcp_size}r{dcp_rank}")
    if not cp_shared:
        parts.append(f"cp{attn_cp_rank}")
    parts.append(f"pp{pp_rank}")
    return "_".join(parts)


def _storage_sharing(
    pools,
    *,
    rank_replicated: bool,
    tp_rank: int,
    tp_size: int,
    attn_cp_rank: int,
    attn_cp_size: int,
    pp_rank: int,
    offload_owner: bool,
    dcp_size: int,
    dcp_rank: int,
):
    """Per-pool key suffixes and the pools this rank offloads.

    A pool replicated along an axis is written by rank 0 of that axis only;
    every rank still reads it, with its own indices, under the shared key.
    A rank-replicated group has no sharded pool, so its group-level suffix
    already drops the tp term and nothing is overridden here.
    A DCP-sharded pool is written by the first DCP block of the TP group,
    one object per DCP rank.
    """
    dcp_sharded = {name for name, pool in pools.items() if pool.dcp_sharded}
    cp_shared = {
        name for name, pool in pools.items() if attn_cp_size > 1 and pool.pcp_replicated
    }
    tp_shared = {
        name
        for name, pool in pools.items()
        if not rank_replicated and tp_size > 1 and pool.tp_replicated
    }
    suffixes = {
        name: _shared_pool_suffix(
            tp_shared=name in tp_shared,
            cp_shared=name in cp_shared,
            tp_rank=tp_rank,
            attn_cp_rank=attn_cp_rank,
            pp_rank=pp_rank,
            dcp_sharded=name in dcp_sharded,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
        )
        for name in cp_shared | tp_shared | dcp_sharded
    }
    offload_pools = {
        name
        for name in pools
        if offload_owner
        and (name not in cp_shared or attn_cp_rank == 0)
        and (
            name not in tp_shared or tp_rank < (dcp_size if name in dcp_sharded else 1)
        )
    }
    return suffixes, offload_pools, cp_shared, tp_shared


def _numa_available() -> list[tuple[int, int, int]]:
    """(node, available bytes, MemTotal) per NUMA node, available as ARIA's NUMA
    guard counts it: MemFree + clean file pages + SReclaimable."""
    nodes = []
    for path in glob.glob("/sys/devices/system/node/node[0-9]*/meminfo"):
        m = {}
        for line in open(path):
            parts = line.split()  # Node 0 MemFree:  123 kB
            if len(parts) >= 4 and parts[3].isdigit():
                m[parts[2].rstrip(":")] = int(parts[3]) * 1024
        clean = m.get("Active(file)", 0) + m.get("Inactive(file)", 0)
        clean -= m.get("Dirty", 0) + m.get("Writeback", 0)
        available = m.get("MemFree", 0) + max(clean, 0) + m.get("SReclaimable", 0)
        node = int(re.search(r"node(\d+)/meminfo$", path).group(1))
        nodes.append((node, available, m.get("MemTotal", 0)))
    return sorted(nodes)


def _resident_by_numa(numa_maps: str = "/proc/self/numa_maps") -> dict[int, int]:
    """Bytes this process holds on each NUMA node (its pages, any page size)."""
    held = {}
    with open(numa_maps) as f:
        for line in f:
            fields = line.split()
            page = next(
                (int(x[18:]) * 1024 for x in fields if x.startswith("kernelpagesize_kB=")),
                4096,
            )
            for x in fields:
                if x[0] == "N" and "=" in x and x[1:x.index("=")].isdigit():
                    node, pages = x[1:].split("=")
                    held[int(node)] = held.get(int(node), 0) + int(pages) * page
    return held


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


class _RecentKeys:
    """Object keys read from the store lately, oldest dropped past capacity: a
    write-through of a tail that was just loaded skips their existence query
    and put. Added on the scheduler thread, read on the offload thread."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.keys: dict[str, None] = {}

    def add(self, keys: list[str]) -> None:
        for key in keys:
            self.keys.pop(key, None)
            self.keys[key] = None
        while len(self.keys) > self.capacity:
            self.keys.pop(next(iter(self.keys)))

    def __contains__(self, key: str) -> bool:
        return key in self.keys


class MooncakeDirectLinker(UnifiedCacheLinker):
    # Set False at init: the first lookup, load or offload logs NUMA headroom.
    _numa_first_request_logged = True
    # SGLANG_LINKER_STATE_PATH_CAP at init; 0 leaves the store's states alone.
    state_path_cap = 0

    def __init__(
        self,
        server_args,
        params: CacheInitParams,
        *,
        components,
        extra_config=None,
        storage=None,
        forward_stream=None,
    ):
        self.forward_stream = forward_stream
        self.log_load_timing = envs.SGLANG_LINKER_LOG_LOAD_TIMING.get()
        self.page_size = params.page_size
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=self.page_size,
            params=params,
            components=components,
        )
        self.pools = self.pool_group.entry_map
        # Bytes of one page object per pool (every buffer of the page).
        self.object_bytes = {
            name: sum(size for component in entry.buffer_meta for *_, size in component)
            for name, entry in self.pools.items()
        }
        self.write_bytes: Counter = Counter()
        self.lookup_counts: Counter = Counter()
        # (keys queried, seconds) per PP lookup and per admission pin, and the
        # pins that found a hole, since the last stats line.
        self.lookup_samples: list[tuple[int, float]] = []
        self.pin_samples: list[tuple[int, float]] = []
        self.pin_holes = 0
        # rid -> object keys whose read sessions pin_hit started; the load
        # thread ends them (load_pins: the batch's pinned rids by load index).
        self.pinned: dict[str, list[str]] = {}
        self.load_pins: dict[int, frozenset] = {}
        self.stats_logged_at = time.monotonic()
        # Store-side state path cap: each state this rank wrote -> the deepest
        # earlier state on its path (None at the root), and its child count.
        self.state_path_cap = envs.SGLANG_LINKER_STATE_PATH_CAP.get()
        self.state_parent: dict[str, str | None] = {}
        self.state_children: Counter = Counter()
        self.state_removed = self.state_remove_failed = 0
        self.state_remove_counts: Counter = Counter()
        # (retry at, key, attempts left): removals a live lease refused (-706).
        self.state_remove_retries: deque = deque()
        self.num_layers = self.pool_group.num_layers

        parallel = get_parallel()
        tp_rank = 0
        tp_size = parallel.tp_size
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(group=tp_group)
            tp_size = torch.distributed.get_world_size(group=tp_group)
        rank_replicated = self.pool_group.rank_replicated
        self.offload_owner = not rank_replicated or tp_rank == 0
        extra_config = deepcopy(extra_config or {})
        # TODO: unlike `HiCacheController._generate_storage_config`, this sets no
        # `dp_rank`. Aligning the storage-config producers changes persisted
        # storage keys, so it waits for a storage-key format change.
        storage_config = HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            is_mla_model=rank_replicated,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name=get_model().model_path,
            extra_config=extra_config,
        )
        if storage is None:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            self.storage = MooncakeStore(storage_config, mem_pool=None)
        else:
            self.storage = storage
        if self.state_path_cap and not callable(getattr(self.storage.store, "remove", None)):
            logger.warning(
                "SGLANG_LINKER_STATE_PATH_CAP needs Mooncake store.remove(); path cap off"
            )
            self.state_path_cap = 0
        self.storage.mem_pool_host = self.pool_group
        self.storage.registered_pools = self.pools
        self.storage.recently_loaded = _RecentKeys(500_000)
        storage_suffix = _storage_suffix(
            rank_replicated=rank_replicated,
            tp_rank=tp_rank,
            attn_cp_rank=params.attn_cp_rank,
            pp_rank=params.pp_rank,
        )
        self.storage.mla_suffix = storage_suffix
        self.storage.mha_suffix = storage_suffix
        # Interleave PCP materializes full DSA KV and indexer pages on every
        # rank, and an MLA-family pool holds identical pages on every
        # attention-TP rank. KDA state stays in its own rank namespace.
        self.storage.pool_suffixes, self.offload_pools, pcp_shared, tp_shared = (
            _storage_sharing(
                self.pools,
                rank_replicated=rank_replicated,
                tp_rank=tp_rank,
                tp_size=tp_size,
                attn_cp_rank=params.attn_cp_rank,
                attn_cp_size=params.attn_cp_size,
                pp_rank=params.pp_rank,
                offload_owner=self.offload_owner,
                dcp_size=parallel.attn_dcp_size,
                dcp_rank=parallel.attn_dcp_rank,
            )
        )
        logger.info(
            "Mooncake direct linker storage topology: "
            "rank_replicated=%s, tp_rank=%d/%d, offload_owner=%s, suffix=%s, "
            "pcp_shared=%s, tp_shared=%s, offload_pools=%s, pool_suffixes=%s",
            rank_replicated,
            tp_rank,
            tp_size,
            self.offload_owner,
            storage_suffix,
            sorted(pcp_shared),
            sorted(tp_shared),
            sorted(self.offload_pools),
            dict(sorted(self.storage.pool_suffixes.items())),
        )

        # Under PP, PP0 looks up every stage's objects (lookup_stages), so each
        # stage publishes how its pools name them.
        self.stage_schemas = [self._pool_key_schema()]
        if params.pp_size > 1:
            self.stage_schemas = [None] * params.pp_size
            torch.distributed.all_gather_object(
                self.stage_schemas, self._pool_key_schema(), group=params.pp_cache_group
            )
            logger.info(
                "Mooncake direct linker PP stage key schemas: %s", self.stage_schemas
            )
        self.lease_refresher = None
        interval = envs.SGLANG_LINKER_LEASE_REFRESH_INTERVAL.get()
        # PP0 renews every stage's objects; other PP ranks renew none.
        if interval > 0 and self.offload_owner and params.pp_rank == 0:
            self.lease_refresher = LinkerLeaseRefresher(
                build_keys=self._lease_keys,
                renew=self.storage._batch_exist,
                interval_s=interval,
                max_keys_per_s=envs.SGLANG_LINKER_LEASE_REFRESH_MAX_KEYS_PER_S.get(),
            )
            self.lease_refresher.start()
            logger.info(
                "Mooncake direct linker lease refresh on: every %.0f s per prefix, "
                "at most %d keys/s, e.g. %s",
                interval,
                envs.SGLANG_LINKER_LEASE_REFRESH_MAX_KEYS_PER_S.get(),
                self._lease_keys(["<page>"]),
            )

        self.register_buffers()
        # Build the native page-hash extension at startup: its first use otherwise
        # compiles it inside a request (about 3 s per node on the first store hit).
        get_storage_hash_str([0] * self.page_size, page_size=self.page_size)
        self._log_numa_available("after segment allocation", resident=True)
        self._numa_first_request_logged = False
        self.mla_broadcaster = None
        self.tp_group = tp_group
        self.broadcast_loads = {}
        if server_args.enable_linker_mla_dedup and rank_replicated and tp_size > 1:
            self.mla_broadcaster = LinkerMLADedupBroadcaster.build(
                self.pool_group, params.tp_cache_group, params.attn_tp_cache_group
            )
            logger.info(
                "MLA linker rank-0 loading enabled: tp_rank=%d/%d", tp_rank, tp_size
            )
        self.layer_done_counter = LayerWiseLoadCounter(
            self.num_layers,
            self._broadcast_loaded_layers if self.mla_broadcaster else None,
        )
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        self.pending_loads: dict[str, list[PoolTransfer]] = {}
        # rid -> when load() queued it; filled only with log_load_timing.
        self.load_queued_at: dict[str, float] = {}
        self.gc_frozen = False
        self.load_queue: Queue[
            tuple[int, dict[str, list[PoolTransfer]], object, dict[str, float]] | None
        ] = Queue()
        self.offload_queue: Queue[
            tuple[list[PoolTransfer], int, list[object], str | None] | None
        ] = Queue()
        self.offload_results: Queue[bool] = Queue()
        self.stats = {"lookup": 0, "load": 0, "offload": 0}
        self.load_thread = threading.Thread(
            target=self.load_thread_func,
            daemon=True,
            name=f"mooncake-load-tp{tp_rank}",
        )
        self.load_thread.start()
        self.offload_thread = threading.Thread(
            target=self.offload_thread_func,
            daemon=True,
            name=f"mooncake-offload-tp{tp_rank}",
        )
        self.offload_thread.start()

    def register_buffers(self) -> None:
        seen = set()
        for pool in self.pools.values():
            for buffer in pool.get_hybrid_pool_buffer():
                storage = buffer.untyped_storage()
                allocation = (int(storage.data_ptr()), int(storage.nbytes()))
                if allocation in seen:
                    continue
                seen.add(allocation)
                result = self.storage.store.register_buffer(*allocation)
                if result not in (0, None):
                    raise RuntimeError(
                        "Failed to register GPU KV buffer with Mooncake, "
                        f"error code: {result}."
                    )

    def _log_numa_available(self, when: str, resident: bool = False) -> None:
        nodes = [entry for entry in _numa_available() if entry[2] > 0]
        low = [node for node, available, total in nodes if available < 0.1 * total]
        held = ""
        if resident:
            held = "; this rank holds " + ", ".join(
                f"n{node} {size / 1e9:.1f} GB"
                for node, size in sorted(_resident_by_numa().items())
            )
        (logger.warning if low else logger.info)(
            "Linker NUMA available %s: %s%s%s",
            when,
            " | ".join(
                f"n{node} {available / 1e9:.1f} of {total / 1e9:.1f} GB "
                f"({100 * available / total:.1f}%)"
                for node, available, total in nodes
            ),
            held,
            f"; UNDER 10% on nodes {low}" if low else "",
        )

    def _log_numa_first_request(self) -> None:
        if not self._numa_first_request_logged:
            self._numa_first_request_logged = True
            self._log_numa_available("at first request")

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        self._log_numa_first_request()
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return []
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        page_keys = list(kv.keys)
        if not page_keys:
            return []
        result = self.storage.batch_exists_v2(page_keys, expanded)
        restorable = result.restorable_prefix_pages or []
        self.stats["lookup"] += 1
        if restorable:
            logger.info(
                "Mooncake direct linker lookup hit: rid=%s pages=%d candidates=%d",
                rid,
                restorable[-1],
                len(restorable),
            )
        return restorable

    def _pool_key_schema(self) -> list[tuple[str, str, str, bool]]:
        """(pool, object name, key suffix, trailing) for every pool a lookup
        checks, as _get_hybrid_page_component_keys names their objects: one
        object per page, or for a trailing pool one per stop point."""
        schema = []
        for name in self.pools:
            obj = "k" if name == PoolName.KV else str(name)
            suffix = self.storage.pool_suffixes.get(name, self.storage.mla_suffix)
            schema.append((str(name), obj, suffix, name == PoolName.MAMBA))
        return schema

    def lookup_stages(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        """lookup() over every PP stage's objects, from one rank: the prefix
        lengths (pages) at which all stages can restore every pool, so the hit
        PP0 relays is one every stage can load.

        A stop point needs every stage's state there. States are sparse, so one
        stage's are queried at every page and the other stages' only at its
        stops. The all-pages objects are then probed at stops, deepest first
        (binary search): a page with every stage's objects is taken to have all
        earlier ones. A hole below it (an eviction) fails that stage's
        admission pin (pin_hit), which shrinks the hit instead of a read.
        Rhea 60076: querying every object of every page (about 24 per page)
        held PP0 on the master's ExistKey ceiling."""
        self._log_numa_first_request()
        began = time.monotonic()
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        page_keys = list(kv.keys)
        num_pages = len(page_keys)
        if not num_pages:
            return []
        queried = 0

        def keys_for(entries, pages):
            return self.storage._tag_keys(
                [f"{page}_{suffix}_{obj}" for _, obj, suffix, _ in entries for page in pages]
            )

        def exist(keys):
            nonlocal queried
            queried += len(keys)
            return self.storage._batch_exist(keys) if keys else []

        def all_found(found, entries, count):
            # Per position: every entry's object exists (entry-major layout).
            return [
                all(found[i * count + j] == 1 for i in range(entries))
                for j in range(count)
            ]

        first = [e for e in self.stage_schemas[0] if e[3]]
        others = [e for schema in self.stage_schemas[1:] for e in schema if e[3]]
        all_pages = [e for schema in self.stage_schemas for e in schema if not e[3]]
        if first:
            found = all_found(exist(keys_for(first, page_keys)), len(first), num_pages)
            candidates = [page + 1 for page in range(num_pages) if found[page]]
        else:
            candidates = list(range(1, num_pages + 1))

        stops = []
        if candidates:
            deepest = candidates[-1]
            check = keys_for(others, [page_keys[p - 1] for p in candidates])
            found = exist(check + keys_for(all_pages, [page_keys[deepest - 1]]))
            have = all_found(found, len(others), len(candidates))
            stops = [p for p, ok in zip(candidates, have) if ok]
            probed = {deepest: all(value == 1 for value in found[len(check) :])}

        def complete(stop):
            if stop not in probed:
                found = exist(keys_for(all_pages, [page_keys[stop - 1]]))
                probed[stop] = all(value == 1 for value in found)
            return probed[stop]

        # Deepest stop whose page every stage holds; complete() is monotone
        # (True up to the first hole, then False).
        last = -1
        low, high = 0, len(stops) - 1
        if stops and complete(stops[high]):
            last = high
        else:
            high -= 1
            while low <= high:
                mid = (low + high) // 2
                if complete(stops[mid]):
                    last, low = mid, mid + 1
                else:
                    high = mid - 1
        restorable = stops[: last + 1]
        self.stats["lookup"] += 1
        self.lookup_counts["lookups"] += 1
        self.lookup_counts["hit_pages"] += restorable[-1] if restorable else 0
        self.lookup_samples.append((queried, time.monotonic() - began))
        if restorable:
            logger.info(
                "Mooncake direct linker PP lookup hit: rid=%s pages=%d candidates=%d "
                "stages=%d",
                rid,
                restorable[-1],
                len(restorable),
                len(self.stage_schemas),
            )
        return restorable

    def _object_keys(
        self, expanded: list[PoolTransfer]
    ) -> tuple[list[str], list[tuple[bool, int, int]]]:
        """This stage's object keys for ``expanded`` as a load reads them, and
        per transfer (trailing, first, end) in that list."""
        keys, spans = [], []
        for transfer in expanded:
            component_keys, _ = self.storage._get_hybrid_page_component_keys(
                list(transfer.keys), transfer
            )
            first = len(keys)
            keys.extend(self.storage._tag_keys(component_keys))
            trailing = transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES
            spans.append((trailing, first, len(keys)))
        return keys, spans

    def pin_hit(self, rid: str, transfers: list[PoolTransfer]) -> int | None:
        """Start this stage's read sessions for a hit as it is admitted. The
        session start is the admission check (an object evicted since the
        lookup shows here, before any forward reads it) and the pin that keeps
        the objects until the load ends the sessions. Returns how many leading
        pages this stage can load: all, or up to the first object that is gone
        (none of its sessions are kept then). None: no pin (MLA dedup peers
        read through the source's broadcast)."""
        if self.mla_broadcaster is not None:
            return None
        num_pages = len(
            next(transfer for transfer in transfers if transfer.name == PoolName.KV).keys
        )
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        keys, spans = self._object_keys(expanded)
        began = time.monotonic()
        result = list(self.storage.store.batch_get_session_start(keys)) if keys else []
        self.last_pin_rpc_s = time.monotonic() - began
        self.pin_samples.append((len(keys), self.last_pin_rpc_s))
        if all(value == 0 for value in result):
            self.pinned[rid] = keys
            return num_pages
        # Even a partial start may have installed some sessions.
        self.storage.store.batch_get_session_end(keys)
        self.pin_holes += 1
        pages = num_pages
        for trailing, first, end in spans:
            missing = next(
                (i for i, value in enumerate(result[first:end]) if value != 0), None
            )
            if missing is not None:
                # A trailing object is the state at the last page only.
                pages = min(pages, num_pages - 1 if trailing else missing)
        return pages

    def unpin(self, rid: str) -> None:
        keys = self.pinned.pop(rid, None)
        if keys:
            self.storage.store.batch_get_session_end(keys)

    def _lease_keys(self, page_hashes: list[str]) -> list[str]:
        """Every stage's all-pages objects of these pages, and the state at the
        last one (a device hit's stop point)."""
        keys = []
        for schema in self.stage_schemas:
            for _, obj, suffix, is_trailing in schema:
                pages = page_hashes[-1:] if is_trailing else page_hashes
                keys.extend(f"{page}_{suffix}_{obj}" for page in pages)
        return self.storage._tag_keys(keys)

    def refresh_device_prefix(self, anchor: object, hashes: Callable) -> None:
        """Renew the store leases of a device-served prefix ending at ``anchor``
        (its page hashes from ``hashes()``), at most once per interval."""
        refresher = self.lease_refresher
        if refresher is not None and refresher.is_due(anchor):
            page_hashes = hashes()
            if page_hashes:
                refresher.submit(anchor, page_hashes)

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        self._log_numa_first_request()
        # Query establishes a boundary at which every component is restorable;
        # insert then removes pages already resident in L1. Loading is therefore
        # intentionally partial and may contain only a side pool such as SWA.
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        if rid in self.pending_loads:
            raise RuntimeError(f"Mooncake load for rid={rid} is already queued.")
        self.pending_loads[rid] = expanded
        self.storage.recently_loaded.add(
            self.pinned.get(rid) or self._object_keys(expanded)[0]
        )
        if self.log_load_timing:
            self.load_queued_at[rid] = time.monotonic()
        return True

    def cancel_queued_load(self, rid: str) -> bool:
        self.load_queued_at.pop(rid, None)
        self.unpin(rid)
        return self.pending_loads.pop(rid, None) is not None

    def finish_layer_wise_loading(self, counter_index: int) -> bool:
        # An MLA dedup peer's layers complete at start; its data arrives through
        # the broadcast on the forward stream, ordered before any later reuse.
        return self.layer_done_counter.finish(counter_index)

    def freeze_gc_once(self) -> None:
        if self.gc_frozen:
            return
        # Transfer metadata creates many short-lived lists. Keep the mature
        # model graph out of cyclic GC scans before load or offload traffic.
        freeze_gc("Mooncake direct linker")
        self.gc_frozen = True

    def start_layer_wise_loading(self) -> int:
        broadcaster = self.mla_broadcaster
        if broadcaster is not None:
            # Compare the logical load order (not local slots), including empty
            # batches, before deciding collectively whether this batch can
            # broadcast.
            plan = [
                (rid, [(t.name, list(t.keys)) for t in transfers])
                for rid, transfers in self.pending_loads.items()
            ]
            digest = hashlib.sha256(repr(plan).encode()).digest()
            digests = [None] * torch.distributed.get_world_size(self.tp_group)
            torch.distributed.all_gather_object(digests, digest, group=self.tp_group)
            if any(other != digest for other in digests):
                logger.warning("MLA linker load plans differ; using all-rank reads.")
                broadcaster = None
        if not self.pending_loads:
            return -1
        self.freeze_gc_once()
        pending = self.pending_loads
        self.pending_loads = {}
        queued_at = {
            rid: self.load_queued_at.pop(rid)
            for rid in pending
            if rid in self.load_queued_at
        }

        counter_index = self.layer_done_counter.update_producer()
        pins = frozenset(rid for rid in pending if self.pinned.pop(rid, None))
        if pins:
            self.load_pins[counter_index] = pins
        if broadcaster is not None:
            indices = {}
            for transfers in pending.values():
                for transfer in transfers:
                    indices.setdefault(transfer.name, []).append(transfer.host_indices)
            prepared = broadcaster.prepare_broadcast(
                {name: torch.cat(parts) for name, parts in indices.items()},
                device_module.current_stream(),
            )
            if not broadcaster.is_src:
                for layer in range(self.num_layers):
                    self.layer_done_counter.complete(counter_index, layer)
        ready_event = device_module.Event()
        ready_event.record()
        if broadcaster is not None:
            self.broadcast_loads[counter_index] = (0, prepared, ready_event)
        if broadcaster is None or broadcaster.is_src:
            self.load_queue.put((counter_index, pending, ready_event, queued_at))
        self.stats["load"] += len(pending)
        flag = envs.SGLANG_LINKER_DEBUG_WAIT_ALL_LAYERS_FILE.get()
        if flag and os.path.exists(flag):
            # Debug only: every layer lands before the forward is launched.
            for future in self.layer_done_counter.futures[counter_index]:
                future.exception()
            device_module.synchronize()
            logger.info(
                "Linker debug: load batch %d landed all %d layers before its forward",
                counter_index,
                self.num_layers,
            )
        return counter_index

    def _broadcast_loaded_layers(self, index: int, threshold: int) -> None:
        batch = self.broadcast_loads.get(index)
        if batch is None:
            return
        first, prepared, ready_event = batch
        if first == 0:
            device_module.current_stream().wait_event(ready_event)
        # KV access may wait several times per layer, or skip sparse layers.
        # Launch each broadcast exactly once, in order, on the forward stream.
        for layer in range(first, threshold + 1):
            self.mla_broadcaster.broadcast_loaded_layer(layer, prepared)
        if threshold == self.num_layers - 1:
            del self.broadcast_loads[index]
        else:
            self.broadcast_loads[index] = (
                max(first, threshold + 1),
                prepared,
                ready_event,
            )

    def load_thread_func(self) -> None:
        while True:
            task = self.load_queue.get()
            try:
                if task is None:
                    return
                counter_index, pending, ready_event, queued_at = task
                try:
                    ready_event.synchronize()
                    self.load_layer_wise(
                        counter_index,
                        list(pending.values()),
                        [(rid, queued_at.get(rid)) for rid in pending],
                        pinned=self.load_pins.pop(counter_index, frozenset()),
                    )
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("Mooncake layer-wise load batch failed")
            finally:
                self.load_queue.task_done()

    def _get_layer_ranges(self, keys, ptrs, sizes, offsets):
        store = self.storage.store
        result = store.batch_get_into_multi_buffer_ranges(keys, ptrs, sizes, offsets)
        if result is None or isinstance(result, int):
            return result
        result = list(result)
        if len(result) != len(keys):
            return result
        expected = [sum(item) for item in sizes]
        # Mooncake ErrorCode::LEASE_EXPIRED. Session leases can expire between
        # layers of a long prefix. The synchronous read has drained its writers;
        # reacquire only those immutable content-addressed objects and retry
        # their current layer once. Other failures still abort the request.
        expired = [i for i, value in enumerate(result) if value == -707]
        if not expired or any(
            value not in (want, -707) for value, want in zip(result, expected)
        ):
            return result
        expired_keys = [keys[i] for i in expired]
        renewed = list(store.batch_get_session_start(expired_keys))
        if renewed != [0] * len(expired_keys):
            raise RuntimeError(
                f"Mooncake get session renewal failed: keys={len(expired_keys)}, "
                f"result_sample={renewed[:8]}"
            )
        retried = store.batch_get_into_multi_buffer_ranges(
            expired_keys,
            [ptrs[i] for i in expired],
            [sizes[i] for i in expired],
            [offsets[i] for i in expired],
        )
        if retried is None or isinstance(retried, int):
            return retried
        retried = list(retried)
        if len(retried) != len(expired):
            raise RuntimeError("Mooncake renewed range get returned wrong result count")
        for index, value in zip(expired, retried):
            result[index] = value
        logger.debug("Refreshed Mooncake read leases for %d objects", len(expired))
        return result

    def load_layer_wise(
        self,
        counter_index: int,
        request_transfers: list[list[PoolTransfer]],
        requests: list[tuple[str, float | None]] | None = None,
        pinned: frozenset = frozenset(),
    ) -> None:
        """``requests`` names each entry of ``request_transfers`` and when its
        load was queued; with ``log_load_timing`` each one is logged once all
        layers land. ``pinned``: rids whose sessions pin_hit already started;
        their sessions end here with the rest."""
        started = []
        began = time.monotonic()
        timed = self.log_load_timing and requests is not None
        # Per request: the key range each pool's batch holds for it.
        spans: list[list[tuple[PoolName, int, int]]] = []
        try:
            batches: dict[PoolName, tuple[list[str], list[int]]] = {}
            unpinned: dict[PoolName, list[str]] = {}
            rids = [rid for rid, _ in requests] if requests else []
            for index, transfers in enumerate(request_transfers):
                span = []
                pinned_here = index < len(rids) and rids[index] in pinned
                for transfer in transfers:
                    keys, locations = batches.setdefault(transfer.name, ([], []))
                    component_keys, _ = self.storage._get_hybrid_page_component_keys(
                        list(transfer.keys), transfer
                    )
                    first = len(keys)
                    keys.extend(self.storage._tag_keys(component_keys))
                    if not pinned_here:
                        unpinned.setdefault(transfer.name, []).extend(keys[first:])
                    span.append((transfer.name, first, len(keys)))
                    locations.extend(
                        self.pools[transfer.name].prepare_locations(
                            transfer.host_indices
                        )
                    )
                spans.append(span)
            loaded = [0] * len(request_transfers)
            # Ends every session of the batch, the pinned ones too; even a
            # partial start failure may have installed some.
            started.extend(keys for keys, _ in batches.values())
            for keys in unpinned.values():
                result = self.storage.store.batch_get_session_start(keys)
                if list(result) != [0] * len(keys):
                    raise RuntimeError(
                        f"Mooncake get session start failed: keys={len(keys)}, "
                        f"result_sample={list(result)[:8]}"
                    )

            for layer in range(self.num_layers):
                for name, (keys, locations) in batches.items():
                    meta = self.pools[name].get_prepared_layer_range_meta(
                        locations, layer
                    )
                    if meta is None:
                        continue
                    ptrs, sizes, offsets = meta
                    result = self._get_layer_ranges(keys, ptrs, sizes, offsets)
                    expected = [sum(item) for item in sizes]
                    if (
                        result is None
                        or isinstance(result, int)
                        or list(result) != expected
                    ):
                        raise RuntimeError(
                            f"Mooncake range get failed for pool={name}, "
                            f"layer={layer}: objects={len(keys)}, "
                            f"transferred_sample={result[:8] if isinstance(result, list) else result}, "
                            f"expected_sample={expected[:8]}"
                        )
                    if timed:
                        for index, span in enumerate(spans):
                            for pool, first, last in span:
                                if pool == name:
                                    loaded[index] += sum(expected[first:last])
                self.layer_done_counter.complete(counter_index, layer)
            if timed:
                self._log_load_timing(
                    request_transfers, requests, loaded, began, time.monotonic()
                )
        except BaseException as error:
            self.layer_done_counter.fail(counter_index, error)
            logger.exception("Mooncake layer-wise load batch failed")
        finally:
            for keys in started:
                try:
                    self.storage.store.batch_get_session_end(keys)
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("Mooncake layer-wise load session cleanup failed")

    @staticmethod
    def _log_load_timing(request_transfers, requests, loaded, began, landed):
        """``wait``: queued until this thread started reading, which includes
        the rest of admission and the batch launch; ``xfer``: until the last
        layer landed. Bytes are this rank's share only."""
        rate = sum(loaded) / max(landed - began, 1e-9) / 1e9
        for transfers, (rid, queued_at), nbytes in zip(
            request_transfers, requests, loaded
        ):
            pages = next((len(t.keys) for t in transfers if t.name == PoolName.KV), 0)
            queued_at = began if queued_at is None else queued_at
            logger.info(
                "Mooncake linker load rid=%s pages=%d bytes=%d wait=%.0fms "
                "xfer=%.0fms total=%.0fms batch_reqs=%d batch_rate=%.2fGB/s",
                rid,
                pages,
                nbytes,
                (began - queued_at) * 1e3,
                (landed - began) * 1e3,
                (landed - queued_at) * 1e3,
                len(request_transfers),
                rate,
            )

    def offload(
        self,
        transfers: list[PoolTransfer],
        session_id: str | None = None,
        path_hashes: Callable[[], list[str]] | None = None,
    ) -> bool:
        """``path_hashes`` gives the page hashes from the root to the node, for
        the state path cap."""
        self._log_numa_first_request()
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
        if not expanded:
            return False
        self.freeze_gc_once()
        expanded = [t for t in expanded if t.name in self.offload_pools]
        if not expanded:
            self.offload_results.put(True)
            return True
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        tokens = len(kv.keys) * self.page_size
        victims = []
        state = next((t for t in expanded if t.name == PoolName.MAMBA), None)
        if self.state_path_cap and state is not None and path_hashes is not None:
            victims = self._cap_state_path(list(state.keys), path_hashes())
        ready_events = self.record_offload_ready()
        self.offload_queue.put((expanded, tokens, ready_events, session_id, victims))
        return True

    def _cap_state_path(self, new_states: list[str], path: list[str]) -> list[str]:
        """Record the states this offload writes on their path and return the
        older ones past SGLANG_LINKER_STATE_PATH_CAP to drop once it lands: a
        turn resumes from the deepest state on its path. A state with more than
        one child is a branch point (a shared prompt) and stays."""
        written = set(new_states)
        parent = next(
            (h for h in reversed(path) if h in self.state_parent and h not in written),
            None,
        )
        for state in new_states:
            if state in self.state_parent:
                continue
            self.state_parent[state] = parent
            if parent is not None:
                self.state_children[parent] += 1
            parent = state
        kept, node = None, parent
        for _ in range(self.state_path_cap):
            if node is None:
                return []
            kept, node = node, self.state_parent.get(node)
        # node: the first state past the cap; drop it and its linear ancestors.
        victims = []
        while node is not None and self.state_children[node] <= 1:
            above = self.state_parent.pop(node, None)
            self.state_children.pop(node, None)
            victims.append(node)
            node = above
        if victims:
            # The kept chain now hangs from the surviving branch point, whose
            # child count is unchanged: one child for another.
            self.state_parent[kept] = node
        return victims

    def offload_thread_func(self) -> None:
        while True:
            task = self.offload_queue.get()
            try:
                if task is None:
                    return
                expanded, tokens, ready_events, session_id, victims = task
                for event in ready_events:
                    event.synchronize()
                extra_info = (
                    HiCacheStorageExtraInfo(extra_info={"session_id": session_id})
                    if session_id
                    else None
                )
                results = self.storage.batch_set_v2(expanded, extra_info)
                success = all(all(pool_results) for pool_results in results.values())
                if success:
                    self.stats["offload"] += 1
                    if self.stats["offload"] == 1:
                        logger.info("Mooncake direct linker offload: tokens=%d", tokens)
                    # The store skips objects it already holds: count what it
                    # put, and what was offered.
                    put = self.storage.last_put_objects
                    self.write_bytes["exist_skipped"] += self.storage.last_exist_skipped
                    for transfer in expanded:
                        size = self.object_bytes[transfer.name]
                        self.write_bytes[transfer.name] += put.get(transfer.name, 0) * size
                        self.write_bytes["offered"] += len(transfer.keys) * size
                    self.write_bytes["states"] += put.get(PoolName.MAMBA, 0)
                    if self.state_path_cap:
                        self._remove_states(victims)
                self.offload_results.put(success)
                self._maybe_log_stats()
            except BaseException:
                logger.exception("Mooncake offload failed")
                self.offload_results.put(False)
            finally:
                self.offload_queue.task_done()

    _REMOVE_RETRIES = 6

    def _remove_states(self, page_hashes: list[str]) -> None:
        """Remove superseded state objects. Mooncake refuses an object under a
        live lease (-706, e.g. one a lookup or the lease refresher touched in
        the last lease TTL), so those are retried once it can have expired."""
        now = time.monotonic()
        due = []
        while self.state_remove_retries and self.state_remove_retries[0][0] <= now:
            due.append(self.state_remove_retries.popleft())
        if page_hashes:
            _, obj, suffix, _ = next(e for e in self._pool_key_schema() if e[3])
            keys = self.storage._tag_keys([f"{h}_{suffix}_{obj}" for h in page_hashes])
            due += [(now, key, self._REMOVE_RETRIES) for key in keys]
        retry_at = now + envs.SGLANG_LINKER_LEASE_TTL_SECS.get() + 1
        for _, key, attempts in due:
            rc = self.storage.store.remove(key)
            if rc == 0:
                self.state_removed += 1
            elif rc == -704:  # OBJECT_NOT_FOUND: already evicted
                self.state_remove_counts["already_gone"] += 1
            elif rc == -706 and attempts > 0:  # OBJECT_HAS_LEASE
                self.state_remove_counts["lease_retries"] += 1
                self.state_remove_retries.append((retry_at, key, attempts - 1))
            else:
                self.state_remove_failed += 1

    def _maybe_log_stats(self, every_s: float = 60.0) -> None:
        now = time.monotonic()
        if now - self.stats_logged_at < every_s:
            return
        window, self.stats_logged_at = now - self.stats_logged_at, now
        written, self.write_bytes = self.write_bytes, Counter()
        looked, self.lookup_counts = self.lookup_counts, Counter()
        lookups, self.lookup_samples = self.lookup_samples, []
        pins, self.pin_samples = self.pin_samples, []
        holes, self.pin_holes = self.pin_holes, 0
        logger.info(
            "Linker store writes in %.0f s: %s GB/s new (%.2f GB/s offered), %d new "
            "states; states removed %d "
            "(failed %d, already gone %d, lease retries %d, pending %d) of %d "
            "tracked; lookups %d, hit pages %d",
            window,
            ", ".join(
                f"{name} {written[name] / window / 1e9:.2f}"
                for name in self.object_bytes
            ),
            written["offered"] / window / 1e9,
            written["states"],
            self.state_removed,
            self.state_remove_failed,
            self.state_remove_counts["already_gone"],
            self.state_remove_counts["lease_retries"],
            len(self.state_remove_retries),
            len(self.state_parent),
            looked["lookups"],
            looked["hit_pages"],
        )
        lookup_keys = [keys for keys, _ in lookups]
        lookup_ms = [seconds * 1e3 for _, seconds in lookups]
        pin_keys = [keys for keys, _ in pins]
        pin_ms = [seconds * 1e3 for _, seconds in pins]
        logger.info(
            "Linker store queries in %.0f s: %d lookups (%.1f/s), keys per lookup "
            "p50 %d p90 %d, lookup ms p50 %.1f p90 %.1f; %d session starts, keys "
            "p50 %d p90 %d, ms p50 %.1f p90 %.1f, load holes %d; write-through "
            "existence queries skipped for %d loaded objects",
            window,
            len(lookups),
            len(lookups) / window,
            _percentile(lookup_keys, 0.5),
            _percentile(lookup_keys, 0.9),
            _percentile(lookup_ms, 0.5),
            _percentile(lookup_ms, 0.9),
            len(pins),
            _percentile(pin_keys, 0.5),
            _percentile(pin_keys, 0.9),
            _percentile(pin_ms, 0.5),
            _percentile(pin_ms, 0.9),
            holes,
            written["exist_skipped"],
        )

    def num_completed_offloads(self) -> int:
        return self.offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self.offload_results.get_nowait()

    def reset(self) -> None:
        for rid in list(self.pinned):
            self.unpin(rid)
        self.pending_loads.clear()
        self.load_queued_at.clear()
        self.load_queue.join()
        self.offload_queue.join()
        self.broadcast_loads.clear()
        while True:
            try:
                self.offload_results.get_nowait()
            except Empty:
                break
        self.layer_done_counter.reset()

    def close(self) -> None:
        if self.lease_refresher is not None:
            self.lease_refresher.stop()
        self.reset()
        self.load_queue.put(None)
        self.offload_queue.put(None)
        self.load_thread.join()
        self.offload_thread.join()
        logger.info("Mooncake direct linker stats: %s", self.stats)
        self.storage.close()
        if self.mla_broadcaster is not None:
            self.mla_broadcaster.destroy()
