from __future__ import annotations

import glob
import hashlib
import heapq
import logging
import os
import re
import threading
import time
import zlib
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
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
    page_checksums,
    RangeArrays,
    attach_cpu_indices,
    resolve_hybrid_device_pool_group,
    vector_key_range_meta,
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
    and put. Added on the scheduler thread, read on the offload thread.
    OrderedDict: dropping a plain dict's first key scans the slots freed before
    it, which stalled Hecate H106's every load once full."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.keys: OrderedDict[str, None] = OrderedDict()

    def add(self, keys: list[str]) -> None:
        for key in keys:
            self.keys[key] = None
            self.keys.move_to_end(key)
        while len(self.keys) > self.capacity:
            self.keys.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        return key in self.keys


class _PutTrace:
    """SGLANG_LINKER_HOLE_TRACE: per page hash, the last time (wall clock, for
    correlation across nodes) this stage queued, submitted, completed or
    removed its objects; the newest ``capacity`` pages. A page reported for a
    hole is logged once, and once more if its put completes after the hole.
    Lines are formatted and logged on the trace's own thread."""

    _EVENTS = ("queued", "submitted", "completed", "failed", "removed")

    def __init__(self, stage: int, capacity: int = 200_000):
        self.stage = stage
        self.capacity = capacity
        self.pages: OrderedDict[str, dict] = OrderedDict()
        self.lock = threading.Lock()
        self.lines: Queue = Queue()
        threading.Thread(
            target=self._log_loop, daemon=True, name="mooncake-hole-trace"
        ).start()

    def _entry(self, page: str) -> dict:
        entry = self.pages.get(page)
        if entry is None:
            entry = self.pages[page] = {}
            if len(self.pages) > self.capacity:
                self.pages.popitem(last=False)
        else:
            self.pages.move_to_end(page)
        return entry

    def note(self, pages, event: str, why: str | None = None) -> None:
        now = time.time()
        with self.lock:
            for page in pages:
                entry = self._entry(page)
                entry[event] = now
                if why is not None:
                    entry["why"] = why
                hole = entry.pop("hole", None) if event == "completed" else None
                if hole is not None:
                    self.lines.put(("late", page, hole, now))

    def report(self, rid: str, pages: Sequence[str], hole: float) -> None:
        """A hole at ``hole`` (wall) on these pages: log what this stage did
        with each, unless that page was reported before."""
        states = []
        with self.lock:
            for page in pages:
                entry = self._entry(page)
                if entry.get("reported"):
                    continue
                entry["reported"] = True
                states.append((page, dict(entry)))
                if entry.get("completed", 0) <= max(
                    entry.get("removed", 0), entry.get("failed", 0)
                ):
                    entry["hole"] = hole
        if states:
            self.lines.put(("hole", rid, states, hole))

    @classmethod
    def describe(cls, entry: dict) -> str:
        done = [(entry[e], e) for e in cls._EVENTS if e in entry]
        if not done:
            return "never written"
        parts = []
        for at, event in sorted(done):
            what = "put " + event if event != "removed" else f"removed ({entry.get('why')})"
            parts.append(f"{what} at {at:.3f}")
        return ", ".join(parts)

    def _log_loop(self) -> None:
        while True:
            kind, *rest = self.lines.get()
            try:
                self._log(kind, rest)
            finally:
                self.lines.task_done()

    def _log(self, kind: str, rest: list) -> None:
        if kind == "hole":
            rid, states, hole = rest
            logger.info(
                "Linker hole trace PP%d: rid=%s hole at %.3f; %s",
                self.stage,
                rid,
                hole,
                "; ".join(f"{page}: {self.describe(e)}" for page, e in states),
            )
        else:
            page, hole, done = rest
            logger.info(
                "Linker hole trace PP%d: page %s put completed at %.3f, %.1f ms "
                "after its hole at %.3f",
                self.stage,
                page,
                done,
                (done - hole) * 1e3,
                hole,
            )


class _StoreTree:
    """SGLANG_LINKER_STORE_BUDGET_PAGES: the pages this stage offloaded, as a
    tree (page hash -> [parent hash, stored children, last-use tick, has a
    state]), evicted leaf first in least-recently-used order down to 95% of
    the budget once over it. Mooncake's own eviction removes single objects
    and leaves holes in the middle of stored prefixes (Rhea pm-lk2: holes in
    13% of PP0 lookups from the store's watermark on); this keeps prefixes
    whole. Every PP stage must pick the same victims, so the inputs are only
    what every stage sees in the same order on its scheduler thread (offload
    and load calls, page counts with a state as ``state_units`` pages) and
    ties break by tick then hash. Scheduler thread only, except ``recalled``
    which the offload thread reads before each removal.

    ``protect``: a leaf used within that many ticks is never evicted.
    ``recall``: a page loaded or stored again within that many ticks of its
    eviction goes back in the tree (with its parent link, without a state: the
    same on every stage whether or not its removals ran there), and
    ``recalled[page]`` takes the tick so its removals decided before that tick
    are dropped if they have not run yet."""

    RECALLED_CAPACITY = 1 << 16

    def __init__(
        self,
        budget: int,
        state_units: int,
        low_fraction: float = 0.95,
        protect: int = 0,
        recall: int = 0,
    ):
        self.budget, self.state_units = budget, state_units
        self.low = int(budget * low_fraction)
        self.protect, self.recall = protect, recall
        self.nodes: dict[str, list] = {}
        # Stored pages whose parent was not stored yet: parent -> count.
        self.orphans: Counter = Counter()
        self.heap: list[tuple[int, str]] = []
        self.units = self.tick = self.states = 0
        self.counts: Counter = Counter()
        # Evicted within the last ``recall`` ticks: page -> (tick, parent).
        self.evicted: OrderedDict = OrderedDict()
        self.recalled: OrderedDict = OrderedDict()

    def _insert(self, page: str, parent: str | None) -> list:
        node = [parent, self.orphans.pop(page, 0), self.tick, False]
        self.nodes[page] = node
        self.units += 1
        above = self.nodes.get(parent) if parent is not None else None
        if above is not None:
            above[1] += 1
        elif parent is not None:
            self.orphans[parent] += 1
        return node

    def _recall(self, page: str) -> list | None:
        """Put an evicted page back if it was evicted within ``recall`` ticks."""
        self._forget_evicted()
        entry = self.evicted.pop(page, None)
        if entry is None:
            return None
        self.recalled[page] = self.tick
        self.recalled.move_to_end(page)
        if len(self.recalled) > self.RECALLED_CAPACITY:
            self.recalled.popitem(last=False)
        self.counts["recalled"] += 1
        return self._insert(page, entry[1])

    def _forget_evicted(self) -> None:
        while self.evicted and next(iter(self.evicted.values()))[0] < self.tick - self.recall:
            self.evicted.popitem(last=False)

    def is_recalled(self, page: str, decided: int) -> bool:
        """Offload thread: was ``page`` used again after its eviction at ``decided``?"""
        return self.recalled.get(page, -1) > decided

    def _push_if_leaf(self, page: str, node: list) -> None:
        if node[1] == 0:
            heapq.heappush(self.heap, (node[2], page))

    def add(self, pages: list[str], parent: str | None, states: Sequence[str]) -> None:
        """An offload of ``pages`` (in path order) under ``parent``, with a
        state stored at each of ``states``."""
        self.tick += 1
        added = []
        for page in pages:
            node = self.nodes.get(page)
            if node is None and self.recall:
                node = self._recall(page)  # stored again: drop its pending removal
            if node is None:
                node = self._insert(page, parent)
            else:
                node[2] = self.tick
            added.append((page, node))
            parent = page
        # Only the pages still leaves once the whole chain is in: an entry for a
        # page that gets its child next is skipped by every evict, yet it grows
        # the heap until the O(nodes) rebuild below stalls the scheduler.
        for page, node in added:
            self._push_if_leaf(page, node)
        for page in states:
            node = self.nodes.get(page)
            if node is not None and not node[3]:
                node[3] = True
                self.units += self.state_units
                self.states += 1

    def _unlink(self, page: str, node: list) -> None:
        parent = node[0]
        above = self.nodes.get(parent) if parent is not None else None
        if above is not None:
            above[1] -= 1
            self._push_if_leaf(parent, above)
        elif parent is not None:
            self.orphans[parent] -= 1
            if self.orphans[parent] <= 0:
                del self.orphans[parent]

    def touch(self, pages: Sequence[str]) -> None:
        """A load of ``pages`` (at admission, on every stage): used now."""
        self.tick += 1
        for page in pages:
            node = self.nodes.get(page)
            if node is None and self.recall:
                node = self._recall(page)
            if node is not None:
                node[2] = self.tick
                self._push_if_leaf(page, node)

    def drop_state(self, page: str) -> None:
        node = self.nodes.get(page)
        if node is not None and node[3]:
            node[3] = False
            self.units -= self.state_units
            self.states -= 1

    def evict(self) -> list[tuple[str, bool]]:
        """(page, had a state) of the pages to remove, leaves first."""
        if self.units <= self.budget:
            return []
        victims = []
        while self.units > self.low and self.heap:
            tick, page = self.heap[0]
            node = self.nodes.get(page)
            if node is None or node[1] or node[2] != tick:
                heapq.heappop(self.heap)
                continue  # gone, not a leaf any more, or used since
            if self.protect and tick > self.tick - self.protect:
                # Judged on a live leaf only, so stale entries never change the victims.
                self.counts["protected_stops"] += 1
                break  # the oldest leaf is recent: so is every other one
            heapq.heappop(self.heap)
            del self.nodes[page]
            self.units -= 1 + (self.state_units if node[3] else 0)
            self.states -= node[3]
            victims.append((page, node[3]))
            if self.recall:
                self.evicted[page] = (self.tick, node[0])
            self._unlink(page, node)
        self.counts["evicted"] += len(victims)
        if self.recall:
            self._forget_evicted()
        if len(self.heap) > 4 * len(self.nodes) + 1024:
            self.heap = [(n[2], p) for p, n in self.nodes.items() if n[1] == 0]
            heapq.heapify(self.heap)
        return victims


class _SessionRefs:
    """This client's read sessions. Mooncake keeps one session per key and
    client, so any end closes it for every holder: with ``counted`` a key's
    session starts on its first holder and ends with its last (Hecate r5: two
    loads ending each other's shared keys failed 16 of 96 with -600). A start
    is all or nothing: on any failed key it keeps no session.

    Counted, the lock guards only the counts: no store RPC runs under it
    (Rhea pm-lk3: PP0's scheduler thread waited on it while pre-pin threads
    held it across their session starts). A key with an RPC in flight is in
    ``inflight``; a start of that key waits for it. ``end_async`` drops the
    counts on the caller's thread and ends the keys that reached zero on one
    background thread, in order; a key started again before that runs keeps
    its session (an end only closes keys still at zero holders)."""

    def __init__(self, store, counted: bool):
        self.store, self.counted = store, counted
        self.counts: Counter = Counter()
        self.lock = threading.Lock()
        self.inflight: dict[str, threading.Event] = {}
        # key -> the sequence number of the end that will close it
        self.pending_end: dict[str, int] = {}
        self.end_seq = 0
        self.ender: ThreadPoolExecutor | None = None

    def start(self, keys: list[str]) -> list[int]:
        if not keys:
            return []
        if not self.counted:
            result = list(self.store.batch_get_session_start(keys))
            if any(result):
                self.store.batch_get_session_end(keys)
            return result
        unique = list(dict.fromkeys(keys))
        while True:
            with self.lock:
                waits = [self.inflight[key] for key in unique if key in self.inflight]
                if not waits:
                    new = [key for key in unique if not self.counts[key]]
                    started = threading.Event()
                    for key in new:
                        self.inflight[key] = started
                    self.counts.update(keys)
                    break
            for event in waits:
                event.wait()
        try:
            codes = dict(zip(new, self.store.batch_get_session_start(new))) if new else {}
        except BaseException:
            codes = {key: -1 for key in new}
            failed = True
        else:
            failed = False
        result = [codes.get(key, 0) for key in keys]
        ok_started = [key for key in new if codes.get(key) == 0]
        with self.lock:
            for key in new:
                self.inflight.pop(key, None)
            if any(result):
                self.counts.subtract(keys)
                for key in unique:
                    if self.counts[key] <= 0:
                        del self.counts[key]
                # Sessions this call opened whose count is back to zero.
                close = [key for key in ok_started if not self.counts[key]]
            else:
                close = []
        started.set()
        if close:
            self.store.batch_get_session_end(close)
        if failed:
            raise RuntimeError("Mooncake get session start raised")
        return result

    def renew(self, keys: list[str]) -> list[int]:
        """Start again sessions this client holds (extends their leases)."""
        return list(self.store.batch_get_session_start(keys)) if keys else []

    def _release(self, keys: list[str]) -> list[str]:
        """Drop one hold of each key; the keys whose count reached zero."""
        last = []
        with self.lock:
            for key in keys:
                held = self.counts.get(key, 0)
                if held > 1:
                    self.counts[key] = held - 1
                elif held == 1:
                    del self.counts[key]
                    last.append(key)
        return last

    def end(self, keys: list[str]) -> None:
        if not self.counted:
            self.store.batch_get_session_end(keys)
            return
        last = self._release(keys)
        if last:
            self._end_now(last, None)

    def end_async(self, keys: list[str]) -> None:
        """``end`` without the store RPC on this thread (counted only; an
        uncounted client ends at once, as before)."""
        if not self.counted:
            self.store.batch_get_session_end(keys)
            return
        last = self._release(keys)
        if not last:
            return
        with self.lock:
            self.end_seq += 1
            seq = self.end_seq
            for key in last:
                self.pending_end[key] = seq
        if self.ender is None:
            self.ender = ThreadPoolExecutor(1, thread_name_prefix="mooncake-session-end")
        self.ender.submit(self._end_now, last, seq)

    def _end_now(self, keys: list[str], seq: int | None) -> None:
        with self.lock:
            if seq is not None:
                keys = [k for k in keys if self.pending_end.get(k) == seq and not self.counts[k]]
                for key in keys:
                    del self.pending_end[key]
            done = threading.Event()
            for key in keys:
                self.inflight[key] = done
        try:
            if keys:
                self.store.batch_get_session_end(keys)
        finally:
            with self.lock:
                for key in keys:
                    if self.inflight.get(key) is done:
                        del self.inflight[key]
            done.set()

    def flush(self) -> None:
        """Wait for queued ends (tests and shutdown)."""
        if self.ender is not None:
            self.ender.submit(lambda: None).result()


class MooncakeDirectLinker(UnifiedCacheLinker):
    # Set False at init: the first lookup, load or offload logs NUMA headroom.
    _numa_first_request_logged = True
    # Refcount read sessions per key (async admission or load overlap).
    count_sessions = False
    # SGLANG_LINKER_PAGE_CHECKSUM sample rate (0: off).
    page_checksum_rate = 0.0
    # SGLANG_LINKER_STORE_BUDGET_PAGES: this stage's stored pages (None: off).
    store_tree: "_StoreTree | None" = None
    # Held pins of launched loads are renewed this often (s); see _renew_held_pins.
    prepin_renew_s = 2.0
    prepin_pool: "ThreadPoolExecutor | None" = None
    pin_renewals = 0
    _session_refs: "_SessionRefs | None" = None
    # SGLANG_LINKER_LOAD_OVERLAP / _PER_KEY at init; the previous launch's event.
    load_overlap = False
    load_per_key = False
    # SGLANG_LINKER_STORE_ARRAYS at init; the list-call fallback logs once.
    store_arrays = False
    # SGLANG_LINKER_HELPER client (None: store I/O in this process)
    helper = None
    _store_arrays_fallback_logged = False
    prev_launch_event = None
    # SGLANG_LINKER_STATE_PATH_CAP at init; 0 leaves the store's states alone.
    state_path_cap = 0
    # SGLANG_LINKER_HOLE_TRACE at init (None: off).
    hole_trace: "_PutTrace | None" = None

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
        # Objects read again after a transient failure, and how many of them
        # landed then; a recovered object is not a failed read.
        self.read_retries = Counter()
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
        self.pp_rank = params.pp_rank
        self.hole_trace = (
            _PutTrace(params.pp_rank) if envs.SGLANG_LINKER_HOLE_TRACE.get() else None
        )
        # PP0 with the trace: first key of a session start that found objects
        # missing -> (wall start, seconds, [(key index, code)] of the first few).
        self.hole_sessions: OrderedDict[str, tuple] = OrderedDict()
        # rid -> object keys whose read sessions pin_hit started; the load
        # thread ends them (load_pins: the batch's pinned rids by load index).
        self.pinned: dict[str, list[str]] = {}
        self.load_pins: dict[int, frozenset] = {}
        # PP0: rid -> the other stages' object keys pin_hit started; once the
        # batch is launched they move to pin_expiry (end time, keys) and end
        # after _PIN_GRACE_S, when every stage's own load has started.
        self.pinned_others: dict[str, list[str]] = {}
        self.pin_expiry: deque = deque()
        # PP0, SGLANG_LINKER_PREPIN_DEPTH: rid -> (hit tag, Future of
        # _start_sessions, pages, own keys, other stages' keys, started or last
        # renewed at, object spans).
        self.prepins: dict[str, tuple] = {}
        self.prepin_pool: ThreadPoolExecutor | None = None
        self.urgent_pool: ThreadPoolExecutor | None = None
        self.prepin_renew_s = envs.SGLANG_LINKER_PREPIN_RENEW_SECS.get()
        self.load_overlap = self._overlap_enabled(
            envs.SGLANG_LINKER_LOAD_OVERLAP.get(),
            envs.SGLANG_LINKER_LOAD_RESERVE_PAGES.get(),
        )
        self.load_per_key = envs.SGLANG_LINKER_LOAD_PER_KEY.get()
        self.page_checksum_rate = envs.SGLANG_LINKER_PAGE_CHECKSUM.get()
        self.page_checksums: OrderedDict[str, int] = OrderedDict()
        self.page_checksum_counts = Counter()
        self.page_checksum_lock = threading.Lock()
        self._checksum_stream = None
        # Per-call switches, re-read by refresh_flags (POST /linker_flags);
        # flag_hits counts the calls that took each switched path.
        self.flag_hits: Counter = Counter()
        self.store_epoch = 0
        self.refresh_flags()
        self.num_load_threads = envs.SGLANG_LINKER_LOAD_THREADS.get() or (
            2 if self.load_overlap else 1
        )
        # Pre-pins held across loads, or loads in flight together, can share a
        # prefix's keys: refcount sessions.
        self.prefetch_mode = envs.SGLANG_LINKER_LOAD_MODE.get() == "prefetch"
        self.count_sessions = (
            envs.SGLANG_LINKER_ASYNC_ADMISSION.get()
            or self.load_overlap
            or self.num_load_threads > 1
            or self.prefetch_mode
        )
        logger.info(
            "Mooncake linker load mode: overlap=%s per_key=%s threads=%d "
            "session_refcount=%s",
            self.load_overlap,
            self.load_per_key,
            self.num_load_threads,
            self.count_sessions,
        )
        self.prepin_renewals = 0
        self.pin_renewals = 0
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
        # SGLANG_LINKER_STORE_BUDGET_PAGES removals: (retry at, key, attempts left).
        self.page_remove_retries: deque = deque()
        self.page_remove_counts: Counter = Counter()
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
        budget = envs.SGLANG_LINKER_STORE_BUDGET_PAGES.get()
        self.store_tree = None
        if budget > 0 and self.offload_owner:
            if callable(getattr(self.storage.store, "remove", None)):
                self.store_tree = _StoreTree(
                    budget,
                    envs.SGLANG_LINKER_STATE_PAGE_EQUIV.get(),
                    protect=envs.SGLANG_LINKER_STORE_PROTECT_TICKS.get(),
                    recall=envs.SGLANG_LINKER_STORE_RECALL_TICKS.get(),
                )
                logger.info(
                    "Linker store budget: %d pages per stage (a state counts %d), "
                    "evicted tail first",
                    budget,
                    self.store_tree.state_units,
                )
            else:
                logger.warning(
                    "SGLANG_LINKER_STORE_BUDGET_PAGES needs Mooncake store.remove(); off"
                )
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
        self.helper = self._start_helper() if envs.SGLANG_LINKER_HELPER.get() else None
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
        self.pins_hits = self.mla_broadcaster is None
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
        self.stats = {"lookup": 0, "load": 0, "offload": 0, "load_overlapped": 0}
        self.load_threads = [
            threading.Thread(
                target=self.load_thread_func,
                daemon=True,
                name=f"mooncake-load-tp{tp_rank}-{i}",
            )
            for i in range(self.num_load_threads)
        ]
        for thread in self.load_threads:
            thread.start()
        # Prefetch mode: whole-request reads, SGLANG_LINKER_LOAD_THREADS at a time.
        self.prefetch_executor = (
            ThreadPoolExecutor(
                self.num_load_threads, thread_name_prefix=f"mooncake-prefetch-tp{tp_rank}"
            )
            if self.prefetch_mode
            else None
        )
        self.prefetch_futures: set = set()
        self.offload_thread = threading.Thread(
            target=self.offload_thread_func,
            daemon=True,
            name=f"mooncake-offload-tp{tp_rank}",
        )
        self.offload_thread.start()

    def _buffer_ranges(self) -> list[tuple[int, int]]:
        """(ptr, nbytes) of every distinct storage behind the linker's pools."""
        seen = []
        for pool in self.pools.values():
            for buffer in pool.get_hybrid_pool_buffer():
                storage = buffer.untyped_storage()
                allocation = (int(storage.data_ptr()), int(storage.nbytes()))
                if allocation not in seen:
                    seen.append(allocation)
        return seen

    def register_buffers(self) -> None:
        for allocation in self._buffer_ranges():
            result = self.storage.store.register_buffer(*allocation)
            if result not in (0, None):
                raise RuntimeError(
                    "Failed to register GPU KV buffer with Mooncake, "
                    f"error code: {result}."
                )

    # ---- SGLANG_LINKER_EARLY_READS: staged reads (linker_staging.py) -----------------

    def staging_slab(self, gb: float, states: int, pages: int = 0, node_gb: float = 0.0):
        """A pinned host slab of object slots, registered with the store: ``states``
        for the state pool (MAMBA) and, per other pool, ``pages`` slots (the same on
        every stage) within this GPU's share of ``node_gb``; with ``pages`` 0, the
        rest of ``gb`` as page slots."""
        from sglang.srt.mem_cache.storage.mooncake_store.linker_staging import StagingSlab, slab_pools

        object_bytes = {name: self.object_bytes[name] for name in self.pool_group.sources}
        if pages:
            cap = int(node_gb * 2**30) // max(torch.cuda.device_count(), 1)
        else:
            pages, cap = 1 << 40, int(gb * 2**30)
        pools = slab_pools(object_bytes, PoolName.MAMBA, pages, states, cap)
        size = StagingSlab.size_of(pools)
        buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        rc = self.storage.store.register_buffer(buffer.data_ptr(), size)
        if rc not in (0, None):
            raise RuntimeError(f"early-read staging registration failed: {rc}")
        self._early_executor = ThreadPoolExecutor(2, thread_name_prefix="mooncake-early")
        # Staged loads have their own thread: never queued behind early reads.
        self._staging_executor = ThreadPoolExecutor(1, thread_name_prefix="mooncake-staging")
        self._staging_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        from sglang.srt.mem_cache.storage.mooncake_store.linker_staging_kernel import loaded_cudart

        logger.info("Mooncake linker early-read staging: %.1f GB pinned, %s; copy %s, libcudart loaded %s",
                    size / 2**30, {str(k): v for k, v in pools.items()}, envs.SGLANG_LINKER_EARLY_COPY.get(),
                    loaded_cudart())
        return StagingSlab(pools, buffer, buffer.data_ptr())

    def early_objects(self, transfers) -> dict:
        """{physical pool: (object keys, object bytes)} of LOOKUP-phase transfers
        (keys only), as a read of them would take them."""
        by_name = {t.name: t for t in transfers if t is not None}
        out = {}
        for name, source in self.pool_group.sources.items():
            transfer = by_name.get(source)
            if transfer is None or not transfer.keys:
                continue
            keys, _ = self.storage._get_hybrid_page_component_keys(
                list(transfer.keys), PoolTransfer(name=name)
            )
            out[name] = (self.storage._tag_keys(keys), self.object_bytes[name])
        return out

    def early_read(self, keys_by_pool: dict, ptrs_by_pool: dict):
        return self._early_executor.submit(self._early_read, keys_by_pool, ptrs_by_pool)

    def _early_read(self, keys_by_pool: dict, ptrs_by_pool: dict) -> bool:
        keys, ptrs, sizes = [], [], []
        for name, names in keys_by_pool.items():
            keys += names
            ptrs += [int(p) for p in ptrs_by_pool[name]]
            sizes += [self.object_bytes[name]] * len(names)
        if self.sessions.start(keys) != [0] * len(keys):
            return False
        try:
            got = list(self.storage.store.batch_get_into(keys, ptrs, sizes))
        finally:
            self.sessions.end(keys)
        return got == sizes

    def prefetch_from_staging(self, transfers, staged, slab, release):
        """Prefetch mode with a landed early read: fill the request's private slots
        from its staged objects (one scatter on the staging stream) instead of a
        store read. On the staging executor: wait for the work queued so far (the
        slot indices), build the plan from the host rows, launch the scatter; the
        future resolves then (``staged``: the finish waits for it if it applies
        first) and carries ``copy_event``, which orders any use or reuse of the
        slots after the scatter. The staging is released (and the sampled checksums
        verified) once the event completes."""
        from concurrent.futures import Future

        from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import coalesce_key_range_table
        from sglang.srt.mem_cache.storage.mooncake_store.linker_staging import (
            copy_engine_scatter,
            gpu_scatter,
            scatter_plan,
        )

        scatter = gpu_scatter if envs.SGLANG_LINKER_EARLY_COPY.get() == "kernel" else copy_engine_scatter

        if self.cpu_indices:
            transfers = attach_cpu_indices(transfers)
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True, allow_missing_kv=True)
        ready = device_module.Event()
        ready.record()
        future: Future = Future()
        future.staged = True

        def run():
            ready.synchronize()
            try:
                plans = []
                for transfer in expanded:
                    entry = self.pools[transfer.name]
                    table = getattr(entry, "_staging_table", None)
                    if table is None:
                        table = entry._staging_table = coalesce_key_range_table(entry.key_range_table(self.num_layers))
                    rows = entry.key_range_rows(entry.prepare_locations(transfer.host_indices))
                    plans.append(scatter_plan(table, rows, slab.ptrs(transfer.name, staged.slots[transfer.name])))
                done = scatter(plans, self._staging_stream)
            except Exception:
                # The finish reads these slots' data from the store instead; free the staging.
                self._staging_stream.synchronize()
                release(staged)
                raise
            future.copy_event = done
            future.set_result(True)
            done.synchronize()
            release(staged)
            if self.page_checksum_rate:
                # As _prefetch_read: the sampled pages of the request's slots, now
                # written, against their checksums recorded at offload.
                self._verify_page_checksums(self._read_batches(expanded))

        def failed(f):
            if f.exception() is None:
                return
            if not future.done():
                logger.error("Linker staged load failed before its copy was queued", exc_info=f.exception())
                future.set_exception(f.exception())
            else:
                logger.warning("Early-read staging release failed", exc_info=f.exception())

        self._staging_executor.submit(run).add_done_callback(failed)
        return future

    def _start_helper(self):
        """SGLANG_LINKER_HELPER: start the store I/O helper (None: not supported here)."""
        from sglang.srt.mem_cache.storage.mooncake_store.linker_helper import (
            HelperClient,
            HelperInit,
            mooncake_client,
            pool_table,
        )
        from sglang.srt.mem_cache.storage.mooncake_store.linker_helper_vmm import export_fabric

        if not self.prefetch_mode or self.storage._can_use_group_semantics():
            logger.warning("SGLANG_LINKER_HELPER needs prefetch mode without group ids; staying in process")
            return None
        tables = {}
        for name, entry in self.pools.items():
            keys, _ = self.storage._get_hybrid_page_component_keys(["x"], PoolTransfer(name=name))
            table = pool_table(name, entry, keys[0][1:], self.num_layers)
            if table is None:
                logger.warning("SGLANG_LINKER_HELPER: pool %s has no range table; staying in process", name)
                return None
            tables[str(name)] = table
        ranges = self._buffer_ranges()
        host, metadata, _, local_buffer, protocol, device_name, master = self.storage.setup_args
        client = HelperClient(
            HelperInit(
                tables=tables,
                prefix=self.storage.config_prefix,
                exports=export_fabric(ranges),
                store_factory=mooncake_client,
                store_args=(
                    host, metadata, envs.SGLANG_LINKER_HELPER_SEGMENT_MB.get() << 20,
                    local_buffer, protocol, device_name, master,
                ),
                coalesce=True,
                threads=envs.SGLANG_LINKER_HELPER_THREADS.get(),
                counted_sessions=self.count_sessions,
                put_config=("mooncake.store", "ReplicateConfig"),
                registered=ranges,
                device=torch.cuda.current_device(),
            )
        )
        logger.info(
            "Mooncake linker helper pid %d: %d pools, %d allocations exported",
            client.pid, len(tables), len(ranges),
        )
        return client

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
            obj = getattr(self.pools[name], "object_name", None) or (
                "k" if name == PoolName.KV else str(name)
            )
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

    _PIN_GRACE_S = 30.0
    # Read results retried once: LEASE_EXPIRED (sessions renewed first) and a
    # failed transfer.
    _RETRY_CODES = (-600, -707, -800)
    # A pre-pin not taken within this long is released (the read sessions'
    # leases are short); its request pins at admission instead.
    _PREPIN_TTL_S = 3.0

    def _pin_keys(self, transfers: list[PoolTransfer]):
        """Every stage's object keys of a hit: (pages, PP0's own keys, the other
        stages' keys, per object group (trailing, first, end) over both)."""
        page_keys = list(
            next(transfer for transfer in transfers if transfer.name == PoolName.KV).keys
        )
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        own, spans = self._object_keys(expanded)
        others = []
        for schema in self.stage_schemas[1:]:
            for _, obj, suffix, trailing in schema:
                pages = page_keys[-1:] if trailing else page_keys
                first = len(own) + len(others)
                others.extend(self.storage._tag_keys([f"{p}_{suffix}_{obj}" for p in pages]))
                spans.append((trailing, first, len(own) + len(others)))
        return len(page_keys), own, others, spans

    @property
    def sessions(self) -> _SessionRefs:
        if self._session_refs is None:
            self._session_refs = _SessionRefs(self.storage.store, self.count_sessions)
        return self._session_refs

    def _start_sessions(
        self, keys, spans, num_pages, renew: bool = False
    ) -> tuple[int, float]:
        """Start read sessions on ``keys``: (leading pages every object holds,
        seconds waited). Short of all pages, no session is kept; a renewal
        (``renew``) that finds a hole gives up the ones it held."""
        began = time.monotonic()
        result = self.sessions.renew(keys) if renew else self.sessions.start(keys)
        waited = time.monotonic() - began
        self.pin_samples.append((len(keys), waited))
        if all(value == 0 for value in result):
            return num_pages, waited
        if renew:
            self.sessions.end(keys)
        self.pin_holes += 1
        if self.hole_trace is not None and keys:
            missing = [(i, v) for i, v in enumerate(result) if v != 0][:4]
            self.hole_sessions[keys[0]] = (time.time() - waited, waited, missing)
            while len(self.hole_sessions) > 4096:
                self.hole_sessions.popitem(last=False)
        pages = num_pages
        for trailing, first, end in spans:
            missing = next(
                (i for i, value in enumerate(result[first:end]) if value != 0), None
            )
            if missing is not None:
                # A trailing object is the state at the last page only.
                pages = min(pages, num_pages - 1 if trailing else missing)
        return pages, waited

    def pin_hit(self, rid: str, transfers: list[PoolTransfer]) -> int:
        """PP0, as a relayed hit is admitted: start the read sessions of every
        stage's objects of it. The session start is the admission check (an
        object evicted since the lookup shows here, on any stage, before a
        forward reads it), and the pin that keeps the objects until each stage
        has started its own load. Returns how many leading pages every stage
        can load: all, or up to the first object that is gone (no sessions are
        kept then). Later stages take this through the PP relay, which carries
        PP0's value only (Hecate H106: per-stage pins were overwritten)."""
        num_pages, own, others, spans = self._pin_keys(transfers)
        self._end_expired_pins()
        pages, self.last_pin_rpc_s = self._start_sessions(own + others, spans, num_pages)
        if pages == num_pages:
            self._hold_pins(rid, own, others)
        return pages

    def trace_hole(
        self, rid: str, transfers: list[PoolTransfer], keep: int, looked_up: float | None
    ) -> None:
        """Every stage, as admission finds a hole past page ``keep``: report this
        stage's puts of the first pages past it (and the last, for its state);
        PP0 also logs the objects its session start found missing."""
        if self.hole_trace is None:
            return
        now = time.time()
        pages = list(next(t for t in transfers if t.name == PoolName.KV).keys)
        if self.offload_owner:
            self.hole_trace.report(rid, pages[keep : keep + 3] + pages[-1:], now)
        if self.pp_rank != 0:
            return
        num_pages, own, others, _ = self._pin_keys(transfers)
        keys = own + others
        session = self.hole_sessions.get(keys[0]) if keys else None
        # Stage of each key index: PP0's own keys, then each later stage's.
        ends, end = [len(own)], len(own)
        for schema in self.stage_schemas[1:]:
            end += sum(1 if trailing else num_pages for *_, trailing in schema)
            ends.append(end)
        missing = [
            (next(s for s, e in enumerate(ends) if i < e), keys[i], code)
            for i, code in (session[2] if session else [])
            if i < len(keys)
        ]
        logger.info(
            "Linker hole trace PP0 pin: rid=%s page %d of %d; missing (stage, key, "
            "code) %s; looked up at %s, session start at %s (%s ms), hole at %.3f",
            rid,
            keep,
            num_pages,
            missing,
            "-" if looked_up is None else f"{looked_up:.3f}",
            "-" if session is None else f"{session[0]:.3f}",
            "-" if session is None else f"{session[1] * 1e3:.1f}",
            now,
        )

    def _hold_pins(self, rid: str, own: list[str], others: list[str]) -> None:
        self.pinned[rid] = own
        if others:
            self.pinned_others[rid] = others

    @staticmethod
    def _prepin_tag(transfers: list[PoolTransfer]) -> tuple:
        keys = next(t for t in transfers if t.name == PoolName.KV).keys
        return len(keys), keys[0], keys[-1]

    def start_prepin(
        self, rid: str, transfers: list[PoolTransfer], urgent: bool = False
    ) -> None:
        """PP0: pin_hit for a request still waiting, on a worker thread
        (SGLANG_LINKER_PREPIN_DEPTH); its admission takes the result. An
        ``urgent`` pin runs on its own thread, not behind the window's."""
        tag = self._prepin_tag(transfers)
        entry = self.prepins.get(rid)
        if entry is not None:
            # A pin still queued behind the window moves to the urgent thread.
            if entry[0] == tag and not (urgent and entry[1].cancel()):
                return
            if entry[0] == tag:
                del self.prepins[rid]
            else:
                self._drop_prepin(rid)
        if self.prepin_pool is None:
            self.prepin_pool = ThreadPoolExecutor(
                max(1, envs.SGLANG_LINKER_PREPIN_THREADS.get()),
                thread_name_prefix="mooncake-prepin",
            )
        if urgent and self.urgent_pool is None:
            self.urgent_pool = ThreadPoolExecutor(1, thread_name_prefix="mooncake-prepin-urgent")
        num_pages, own, others, spans = self._pin_keys(transfers)
        future = (self.urgent_pool if urgent else self.prepin_pool).submit(
            self._start_sessions, own + others, spans, num_pages
        )
        self.prepins[rid] = (tag, future, num_pages, own, others, time.monotonic(), spans)

    def peek_prepin(self, rid: str, transfers: list[PoolTransfer]) -> int | None:
        """The pages a finished pre-pin of this same hit holds, without taking
        it; None if there is none or it is still running."""
        entry = self.prepins.get(rid)
        if entry is None or entry[0] != self._prepin_tag(transfers) or not entry[1].done():
            return None
        return entry[1].result()[0]

    def take_prepin(self, rid: str, transfers: list[PoolTransfer]) -> int | None:
        """The pages a finished pre-pin of this same hit holds, now owned like
        pin_hit's; None if there is none or it is still running."""
        entry = self.prepins.get(rid)
        if entry is None or entry[0] != self._prepin_tag(transfers) or not entry[1].done():
            return None
        del self.prepins[rid]
        _, future, num_pages, own, others, *_ = entry
        pages, _ = future.result()
        self.last_pin_rpc_s = 0.0
        if pages == num_pages:
            self._hold_pins(rid, own, others)
        return pages

    def release_stale_prepins(self) -> None:
        now = time.monotonic()
        for rid in [r for r, e in self.prepins.items() if now - e[5] > self._PREPIN_TTL_S]:
            self._drop_prepin(rid)

    def keep_prepins(self, rids: Sequence[str], urgent: Sequence[str] = ()) -> None:
        """Async admission, after a pass: end the pre-pins of requests no longer
        in ``rids`` (the next-admitted window) and renew the kept ones older
        than the renewal interval. A renewal is a session start on the same
        keys, as a LEASE_EXPIRED read renews them; it shows a hole the same way."""
        keep, held = set(rids), set(urgent)
        for rid in [r for r in self.prepins if r not in keep]:
            self._drop_prepin(rid)
        now = time.monotonic()
        self._renew_held_pins(now)
        for rid, entry in list(self.prepins.items()):
            tag, future, num_pages, own, others, started, spans = entry
            # A renewal in flight reads as not ready: never renew a hit that
            # admission is already waiting for.
            if rid in held or now - started < self.prepin_renew_s or not future.done():
                continue
            if future.result()[0] != num_pages:
                continue  # a hole: no session is held, its admission shrinks the hit
            renewed = self.prepin_pool.submit(
                self._start_sessions, own + others, spans, num_pages, True
            )
            self.prepins[rid] = (tag, renewed, num_pages, own, others, now, spans)
            self.prepin_renewals += 1

    def _drop_prepin(self, rid: str) -> None:
        entry = self.prepins.pop(rid, None)
        if entry is None:
            return
        _, future, num_pages, own, others, *_ = entry

        def end(done):
            if done.result()[0] == num_pages:
                self.sessions.end_async(own + others)

        # A renewal replaced the entry's future, so this ends after it.
        future.add_done_callback(end)

    def unpin(self, rid: str) -> None:
        keys = self.pinned.pop(rid, None) or []
        keys += self.pinned_others.pop(rid, None) or []
        if keys:
            self.sessions.end_async(keys)

    def _end_expired_pins(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        while self.pin_expiry and self.pin_expiry[0][0] <= now:
            # Off the scheduler thread (Rhea pm-lk3: 24% of PP0's).
            self.sessions.end_async(self.pin_expiry.popleft()[1])

    def _renew_held_pins(self, now: float | None = None) -> None:
        """PP0: renew the later stages' pinned objects of launched loads until
        their grace ends. A session lapses with the store lease (about 10 s):
        under a deep queue a later stage started its own load after that and
        found the objects evicted (Rhea pm-lk1: -704 at its session start).
        Renewals run off the scheduler thread; a renewal that finds an object
        gone changes nothing here, that stage's load reports it."""
        if not self.pin_expiry or self.prepin_renew_s <= 0:
            return
        now = time.monotonic() if now is None else now
        due = [entry for entry in self.pin_expiry if now - entry[2] >= self.prepin_renew_s]
        if not due:
            return
        if self.prepin_pool is None:
            self.prepin_pool = ThreadPoolExecutor(1, thread_name_prefix="mooncake-prepin")
        for entry in due:
            entry[2] = now
            self.prepin_pool.submit(self.sessions.renew, list(entry[1]))
            self.pin_renewals += 1

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
        if self.store_tree is not None:
            kv = next((t for t in transfers if t.name == PoolName.KV), None)
            if kv is not None:
                self.store_tree.touch(list(kv.keys))
        self.pending_loads[rid] = expanded
        self.storage.recently_loaded.add(
            self.pinned.get(rid) or self._object_keys(expanded)[0]
        )
        if self.log_load_timing:
            self.load_queued_at[rid] = time.monotonic()
        return True

    def prefetch(self, transfers: list[PoolTransfer]):
        """Prefetch mode: read every layer of one request's restorable prefix
        into its private slots off the scheduler thread. The slot indices are
        ready once the work queued so far on this stream has run."""
        if self.prefetch_executor is None:
            return None
        if self.mla_broadcaster is not None:
            raise RuntimeError("Mooncake linker prefetch mode does not support MLA dedup.")
        self._log_numa_first_request()
        if self.cpu_indices:
            transfers = attach_cpu_indices(transfers)
            self.flag_hits["cpu_indices_attached"] += 1
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return None
        self.freeze_gc_once()
        if self.store_tree is not None:
            kv = next((t for t in transfers if t.name == PoolName.KV), None)
            if kv is not None:
                self.store_tree.touch(list(kv.keys))
        if self.helper is None:
            # Objects a read takes are stored: their publication skips the put
            # (the helper keeps this set itself).
            self.storage.recently_loaded.add(self._object_keys(expanded)[0])
        ready = device_module.Event()
        ready.record()
        future = self.prefetch_executor.submit(
            self._prefetch_read if self.helper is None else self._prefetch_read_helper,
            expanded,
            ready,
            time.monotonic(),
        )
        self.prefetch_futures.add(future)
        future.add_done_callback(self.prefetch_futures.discard)
        self.stats["prefetch"] = self.stats.get("prefetch", 0) + 1
        return future

    def _read_batches(self, expanded: list[PoolTransfer]) -> dict:
        batches: dict[PoolName, tuple[list[str], list[int]]] = {}
        for transfer in expanded:
            keys, locations = batches.setdefault(transfer.name, ([], []))
            component_keys, _ = self.storage._get_hybrid_page_component_keys(
                list(transfer.keys), transfer
            )
            keys.extend(self.storage._tag_keys(component_keys))
            locations.extend(self.pools[transfer.name].prepare_locations(transfer.host_indices))
        return batches

    @staticmethod
    def _helper_pools(expanded: list[PoolTransfer]) -> list:
        """The helper's descriptor: page hashes and slot indices per pool."""
        return [
            (str(t.name), list(t.keys), t.host_indices.to(device="cpu", dtype=torch.int64).numpy())
            for t in expanded
        ]

    def _prefetch_read_helper(self, expanded: list[PoolTransfer], ready, queued: float) -> bool | str:
        """SGLANG_LINKER_HELPER: _prefetch_read with the keys, ranges and store
        calls in the helper; this thread only waits (without the GIL)."""
        began = time.monotonic()
        ready.synchronize()
        try:
            status, info = self.helper.submit("read", self._helper_pools(expanded)).result()
        except Exception as error:
            logger.warning("Mooncake linker helper read failed: %s", error)
            status, info = f"read:{str(error)[:80]}", {}
        if status is not True:
            self.stats["prefetch_failed"] = self.stats.get("prefetch_failed", 0) + 1
            return status
        if self.page_checksum_rate:
            self._verify_page_checksums(self._read_batches(expanded))
        if self.log_load_timing:
            logger.info(
                "Mooncake linker prefetch pages=%d objects=%d wait=%.0fms xfer=%.0fms helper=%.0fms",
                next((len(t.keys) for t in expanded if t.name == PoolName.KV), 0),
                info.get("objects", 0),
                (began - queued) * 1e3,
                (time.monotonic() - began) * 1e3,
                info.get("ms", 0.0),
            )
        return True

    def _prefetch_read(self, expanded: list[PoolTransfer], ready, queued: float) -> bool | str:
        """One request's read, as load_layer_wise reads a batch (per key or per
        layer, with its lease renewal and retry), with no layer counter."""
        began = time.monotonic()
        ready.synchronize()
        batches = self._read_batches(expanded)
        keys = [key for names, _ in batches.values() for key in names]
        result = self.sessions.start(keys)
        if result != [0] * len(keys):
            failed = Counter(v for v in result if v != 0)
            logger.warning(
                "Mooncake linker prefetch: session start failed for %d of %d objects "
                "by code %s",
                sum(failed.values()),
                len(keys),
                dict(failed),
            )
            self.stats["prefetch_failed"] = self.stats.get("prefetch_failed", 0) + 1
            # Not True, so the read fails; the string carries the return codes.
            return "session_start:" + ",".join(f"{k}x{v}" for k, v in sorted(failed.items()))
        try:
            if self.load_per_key:
                self._load_per_key(batches, [], None)
            else:
                for layer in range(self.num_layers):
                    for name, (names, locations) in batches.items():
                        meta = self.pools[name].get_prepared_layer_range_meta(
                            locations, layer
                        )
                        if meta is None:
                            continue
                        ptrs, sizes, offsets = meta
                        got = self._get_layer_ranges(names, ptrs, sizes, offsets)
                        if (
                            got is None
                            or isinstance(got, int)
                            or list(got) != [sum(item) for item in sizes]
                        ):
                            raise RuntimeError(
                                f"Mooncake range get failed for pool={name}, layer={layer}"
                            )
        except Exception as error:
            logger.warning("Mooncake linker prefetch read failed", exc_info=True)
            self.stats["prefetch_failed"] = self.stats.get("prefetch_failed", 0) + 1
            return f"read:{str(error)[:80]}"
        finally:
            self.sessions.end(keys)
        if self.page_checksum_rate:
            self._verify_page_checksums(batches)
        if self.log_load_timing:
            logger.info(
                "Mooncake linker prefetch pages=%d objects=%d wait=%.0fms xfer=%.0fms",
                next((len(t.keys) for t in expanded if t.name == PoolName.KV), 0),
                len(keys),
                (began - queued) * 1e3,
                (time.monotonic() - began) * 1e3,
            )
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

    @staticmethod
    def _overlap_enabled(overlap: bool, reserve_pages: int) -> bool:
        """Only CPU-known load slots (the reserve) may be read before this
        batch's own GPU work: other slot indices are written by allocation
        kernels queued after the previous launch (Hecate D4b: misaligned reads
        and a request restored into the wrong slots)."""
        if overlap and reserve_pages <= 0:
            logger.warning(
                "SGLANG_LINKER_LOAD_OVERLAP is ignored: loads still wait for all "
                "prior GPU work unless SGLANG_LINKER_LOAD_RESERVE_PAGES reserves "
                "CPU-known load slots."
            )
        return bool(overlap) and reserve_pages > 0

    def start_layer_wise_loading(self) -> int:
        previous = None
        if self.load_overlap:
            # Called for every prefill batch as it is built, before its forward.
            previous, self.prev_launch_event = (
                self.prev_launch_event,
                device_module.Event(),
            )
            self.prev_launch_event.record()
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
        now = time.monotonic()
        for rid in pending:
            others = self.pinned_others.pop(rid, None)
            if others:
                self.pin_expiry.append([now + self._PIN_GRACE_S, others, now])
        self._end_expired_pins(now)
        self._renew_held_pins(now)
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
        wait_event, late_event = ready_event, None
        if previous is not None and broadcaster is None:
            # Pools whose slots are CPU-known (the reserve) start after the work
            # queued before the previous launch; any other pool's indices are
            # written by this batch's GPU work, so it waits for ready_event.
            wait_event, late_event = previous, ready_event
            self.stats["load_overlapped"] = self.stats.get("load_overlapped", 0) + 1
        if broadcaster is None or broadcaster.is_src:
            self.load_queue.put(
                (counter_index, pending, wait_event, queued_at, late_event)
            )
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
                counter_index, pending, ready_event, queued_at, late_event = task
                try:
                    ready_event.synchronize()
                    self.load_layer_wise(
                        counter_index,
                        list(pending.values()),
                        [(rid, queued_at.get(rid)) for rid in pending],
                        pinned=self.load_pins.pop(counter_index, frozenset()),
                        late_wait=(
                            late_event.synchronize if late_event is not None else None
                        ),
                    )
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("Mooncake layer-wise load batch failed")
            finally:
                self.load_queue.task_done()

    def _store_arrays_get(self):
        """SGLANG_LINKER_STORE_ARRAYS: the store's array range get, or None (the
        flag is off, or the store lacks it: logged once, the list call is used)."""
        if not self.store_arrays:
            return None
        method = getattr(
            self.storage.store, "batch_get_into_multi_buffer_ranges_arrays", None
        )
        if method is None and not self._store_arrays_fallback_logged:
            self._store_arrays_fallback_logged = True
            logger.warning(
                "SGLANG_LINKER_STORE_ARRAYS is set but the Mooncake store has no "
                "batch_get_into_multi_buffer_ranges_arrays; using the list call"
            )
        return method

    def _range_get(self, keys, ranges):
        """One store range get; ``ranges`` is a RangeArrays or (ptrs, sizes, offsets)."""
        method = self._store_arrays_get()
        if method is not None:
            if not isinstance(ranges, RangeArrays):
                ranges = RangeArrays.from_lists(*ranges)
            return method(keys, *ranges)
        if isinstance(ranges, RangeArrays):
            ranges = ranges.to_lists()
        return self.storage.store.batch_get_into_multi_buffer_ranges(keys, *ranges)

    def _get_layer_ranges(self, keys, ptrs, sizes=None, offsets=None):
        """Range get with one retry of transient failures. ``ptrs`` is either the
        per-key ptr lists (with ``sizes`` and ``offsets``) or a RangeArrays."""
        ranges = ptrs if isinstance(ptrs, RangeArrays) else (ptrs, sizes, offsets)
        result = self._range_get(keys, ranges)
        if result is None or isinstance(result, int):
            return result
        result = list(result)
        if len(result) != len(keys):
            return result
        if isinstance(ranges, RangeArrays):
            expected = ranges.expected()
        else:
            expected = [sum(item) for item in sizes]
        # Retry this layer once for objects that failed transiently: an expired
        # session lease (Mooncake LEASE_EXPIRED, -707: leases can lapse between
        # layers of a long prefix; reacquire those immutable objects first), a
        # session another holder ended or that lapsed (-600, INVALID_PARAMS:
        # renewed too; Rhea pm-lk1 failed a per-key read on -707 and -600
        # together) or a failed transfer (-800). Anything else, like a missing
        # object (-704), still fails the load: under PP one stage's failure
        # splits the stages' queues (Hecate H112), so a transient one should
        # not get that far.
        retry = [i for i, (v, want) in enumerate(zip(result, expected)) if v != want]
        if not retry or any(result[i] not in self._RETRY_CODES for i in retry):
            return result
        expired_keys = [keys[i] for i in retry if result[i] in (-600, -707)]
        renewed = self.sessions.renew(expired_keys)
        if renewed != [0] * len(expired_keys):
            raise RuntimeError(
                f"Mooncake get session renewal failed: keys={len(expired_keys)}, "
                f"result_sample={renewed[:8]}"
            )
        expired = retry
        retried = self._range_get(
            [keys[i] for i in expired],
            ranges.select(expired)
            if isinstance(ranges, RangeArrays)
            else (
                [ptrs[i] for i in expired],
                [sizes[i] for i in expired],
                [offsets[i] for i in expired],
            ),
        )
        if retried is None or isinstance(retried, int):
            return retried
        retried = list(retried)
        if len(retried) != len(expired):
            raise RuntimeError("Mooncake renewed range get returned wrong result count")
        for index, value in zip(expired, retried):
            result[index] = value
        self.read_retries["retried"] += len(expired)
        self.read_retries["recovered"] += sum(result[i] == expected[i] for i in expired)
        logger.warning(
            "Mooncake range get retried %d objects (%d with renewed leases); "
            "this rank so far: %d retried, %d recovered",
            len(expired),
            len(expired_keys),
            self.read_retries["retried"],
            self.read_retries["recovered"],
        )
        return result

    def load_layer_wise(
        self,
        counter_index: int,
        request_transfers: list[list[PoolTransfer]],
        requests: list[tuple[str, float | None]] | None = None,
        pinned: frozenset = frozenset(),
        late_wait: Callable[[], object] | None = None,
    ) -> None:
        """``requests`` names each entry of ``request_transfers`` and when its
        load was queued; with ``log_load_timing`` each one is logged once all
        layers land. ``pinned``: rids whose sessions pin_hit already started;
        their sessions end here with the rest. ``late_wait`` (overlapped loads):
        called before the first read of a pool whose slot indices are not on the
        CPU; pools with CPU-known slots are read before it."""
        started = []
        began = time.monotonic()
        timed = self.log_load_timing and requests is not None
        # Per request: the key range each pool's batch holds for it.
        spans: list[list[tuple[PoolName, int, int]]] = []
        try:
            batches: dict[PoolName, tuple[list[str], list[int]]] = {}
            slot_sources: dict[PoolName, list] = {}
            unpinned: dict[PoolName, list[str]] = {}
            rids = [rid for rid, _ in requests] if requests else []
            pinned_keys: list[str] = []
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
                    if pinned_here:
                        pinned_keys.extend(keys[first:])
                    else:
                        unpinned.setdefault(transfer.name, []).extend(keys[first:])
                    span.append((transfer.name, first, len(keys)))
                    slot_sources.setdefault(transfer.name, []).append(
                        transfer.host_indices
                    )
                spans.append(span)
            # A pool is read early only when every slot index it uses is on
            # the CPU; the others wait for this batch's own GPU work.
            late = {
                name
                for name, sources in slot_sources.items()
                if late_wait is not None
                and any(getattr(x, "device", None) is not None and x.device.type != "cpu" for x in sources)
            }

            def locate(names):
                for name in names:
                    for host_indices in slot_sources[name]:
                        batches[name][1].extend(
                            self.pools[name].prepare_locations(host_indices)
                        )

            locate([name for name in batches if name not in late])
            loaded = [0] * len(request_transfers)
            # Ends every session of the batch, the pinned ones too (a failed
            # start keeps none).
            started.append(pinned_keys)
            for keys in unpinned.values():
                result = self.sessions.start(keys)
                if result != [0] * len(keys):
                    raise RuntimeError(
                        f"Mooncake get session start failed: keys={len(keys)}, "
                        f"result_sample={result[:8]}"
                    )
                started.append(keys)

            if self.load_per_key:
                early = {n: v for n, v in batches.items() if n not in late}
                if early:
                    self._load_per_key(early, spans, loaded if timed else None)
                if late:
                    late_wait()
                    self.stats["late_waits"] = self.stats.get("late_waits", 0) + 1
                    locate(late)
                    self._load_per_key(
                        {n: batches[n] for n in late}, spans, loaded if timed else None
                    )
                for layer in range(self.num_layers):
                    self.layer_done_counter.complete(counter_index, layer)
            elif late:
                # Per-layer reads interleave pools: wait before the first.
                late_wait()
                self.stats["late_waits"] = self.stats.get("late_waits", 0) + 1
                locate(late)
            # Per-key loads landed above; otherwise one read per layer and pool.
            for layer in () if self.load_per_key else range(self.num_layers):
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
                        failed = (
                            Counter(
                                v for v, want in zip(result, expected) if v != want
                            )
                            if isinstance(result, list)
                            else result
                        )
                        raise RuntimeError(
                            f"Mooncake range get failed for pool={name}, "
                            f"layer={layer}: objects={len(keys)}, failed by code "
                            f"{dict(failed) if isinstance(failed, Counter) else failed}"
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
                if not keys:
                    continue
                try:
                    self.sessions.end(keys)
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("Mooncake layer-wise load session cleanup failed")

    def _load_per_key(self, batches, spans, loaded) -> None:
        """SGLANG_LINKER_LOAD_PER_KEY: one range get for the batch, each key's
        entry holding every layer's ranges (Hecate lbw microbench: 2-3x the
        per-layer, per-pool reads). ``loaded`` (if given) gets each request's bytes."""
        keys_all, ptrs_all, sizes_all, offsets_all, base = [], [], [], [], {}
        arrays = self._store_arrays_get() is not None
        parts: list[RangeArrays] = []
        for name, (keys, locations) in batches.items():
            vector = (
                vector_key_range_meta(
                    self.pools[name],
                    locations,
                    self.num_layers,
                    self.coalesce_ranges,
                    arrays=arrays,
                )
                if self.vector_meta
                else None
            )
            if arrays and vector is not None:
                base[name] = len(keys_all)
                keys_all.extend(keys)
                parts.append(vector)
                continue
            if vector is not None:
                # Same lists as the per-layer loop below, built with numpy so the
                # read thread holds the GIL for a few array ops, not O(keys x layers).
                self.flag_hits["vector_meta_calls"] += 1
                if self.coalesce_ranges:
                    self.flag_hits["coalesced_keys"] += len(keys)
                base[name] = len(keys_all)
                keys_all.extend(keys)
                ptrs_all.extend(vector[0])
                sizes_all.extend(vector[1])
                offsets_all.extend(vector[2])
                continue
            entries = [([], [], []) for _ in keys]
            for layer in range(self.num_layers):
                meta = self.pools[name].get_prepared_layer_range_meta(locations, layer)
                if meta is None:
                    continue
                if len(meta[0]) != len(keys):
                    raise RuntimeError(
                        f"pool {name}: {len(meta[0])} range entries for {len(keys)} keys"
                    )
                for entry, ptrs, sizes, offsets in zip(entries, *meta):
                    entry[0].extend(ptrs)
                    entry[1].extend(sizes)
                    entry[2].extend(offsets)
            base[name] = len(keys_all)
            keys_all.extend(keys)
            if arrays:
                parts.append(
                    RangeArrays.from_lists(*([e[f] for e in entries] for f in range(3)))
                )
                continue
            for ptrs, sizes, offsets in entries:
                ptrs_all.append(ptrs)
                sizes_all.append(sizes)
                offsets_all.append(offsets)
        if arrays:
            ranges = RangeArrays.concat(parts) if parts else RangeArrays.from_lists([], [], [])
            result = self._get_layer_ranges(keys_all, ranges)
            expected = ranges.expected()
        else:
            result = self._get_layer_ranges(keys_all, ptrs_all, sizes_all, offsets_all)
            expected = [sum(item) for item in sizes_all]
        if result is None or isinstance(result, int) or list(result) != expected:
            failed = (
                Counter(v for v, want in zip(result, expected) if v != want)
                if isinstance(result, list)
                else result
            )
            raise RuntimeError(
                f"Mooncake range get failed for every pool and layer: objects="
                f"{len(keys_all)}, failed by code "
                f"{dict(failed) if isinstance(failed, Counter) else failed}"
            )
        if loaded is not None:
            for index, span in enumerate(spans):
                for pool, first, last in span:
                    if pool in base:
                        loaded[index] += sum(
                            expected[base[pool] + first : base[pool] + last]
                        )

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
        if self.cpu_indices:
            transfers = attach_cpu_indices(transfers)
            self.flag_hits["cpu_indices_attached"] += 1
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
        path = path_hashes() if path_hashes is not None else None
        if self.state_path_cap and state is not None and path is not None:
            victims = self._cap_state_path(list(state.keys), path)
        ready_events = self.record_offload_ready()
        if self.hole_trace is not None:
            self.hole_trace.note({k for t in expanded for k in t.keys}, "queued")
        self.offload_queue.put((expanded, tokens, ready_events, session_id, victims))
        tree = self.store_tree
        if tree is not None:
            for page in victims:
                tree.drop_state(page)
            pages = list(kv.keys)
            tree.add(
                pages,
                self._parent_page(pages, path),
                list(state.keys) if state is not None else (),
            )
            self._queue_tail_eviction(tree.evict())
        return True

    @staticmethod
    def _parent_page(pages: list[str], path: list[str] | None) -> str | None:
        """The stored page before ``pages`` on their path (root to node)."""
        if not path or not pages:
            return None
        try:
            index = path.index(pages[0])
        except ValueError:
            return None
        return path[index - 1] if index > 0 else None

    def _queue_tail_eviction(self, victims: list[tuple[str, bool]]) -> None:
        """This stage's objects of the evicted pages, removed on the offload
        thread after the puts queued before (one of them may write them)."""
        if not victims:
            return
        keys, pages = [], []
        for _, obj, suffix, trailing in self._pool_key_schema():
            for page, had_state in victims:
                if had_state or not trailing:
                    keys.append(f"{page}_{suffix}_{obj}")
                    pages.append(page)
        tick = self.store_tree.tick
        self.offload_queue.put(("remove", self.storage._tag_keys(keys), pages, tick))

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
                if task[0] == "remove":
                    self._remove_pages(task[1], task[2], task[3])
                    self._maybe_log_stats()
                    continue
                expanded, tokens, ready_events, session_id, victims = task
                for event in ready_events:
                    event.synchronize()
                extra_info = (
                    HiCacheStorageExtraInfo(extra_info={"session_id": session_id})
                    if session_id
                    else None
                )
                trace = self.hole_trace
                if trace is not None:
                    traced = {k for t in expanded for k in t.keys}
                    trace.note(traced, "submitted")
                if self.helper is not None and extra_info is None:
                    by_name, info = self.helper.submit(
                        "write", self._helper_pools(expanded)
                    ).result()
                    results = {t.name: by_name.get(str(t.name), []) for t in expanded}
                    put = {t.name: info["put_objects"].get(str(t.name), 0) for t in expanded}
                    exist_skipped = info["exist_skipped"]
                else:
                    results = self.storage.batch_set_v2(expanded, extra_info)
                    put = self.storage.last_put_objects
                    exist_skipped = self.storage.last_exist_skipped
                success = all(all(pool_results) for pool_results in results.values())
                if trace is not None:
                    trace.note(traced, "completed" if success else "failed")
                if success:
                    if self.page_checksum_rate:
                        self._record_page_checksums(expanded)
                    self.stats["offload"] += 1
                    if self.stats["offload"] == 1:
                        logger.info("Mooncake direct linker offload: tokens=%d", tokens)
                    # The store skips objects it already holds: count what it
                    # put, and what was offered.
                    self.write_bytes["exist_skipped"] += exist_skipped
                    for transfer in expanded:
                        size = self.object_bytes[transfer.name]
                        self.write_bytes[transfer.name] += put.get(transfer.name, 0) * size
                        self.write_bytes["offered"] += len(transfer.keys) * size
                    self.write_bytes["states"] += put.get(PoolName.MAMBA, 0)
                    if self.state_path_cap:
                        self._remove_states(victims)
                if self.store_tree is not None and self.page_remove_retries:
                    self._remove_pages([])
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
            if self.hole_trace is not None:
                self.hole_trace.note(page_hashes, "removed", "state cap")
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

    def _checksum_sampled(self, key: str) -> bool:
        return zlib.crc32(key.encode()) % 100000 < self.page_checksum_rate * 100000

    def _checksum_pages(self, name, keys, locations) -> list[tuple[str, int]]:
        """Checksums of the sampled pages of one pool, on a side stream so the
        forward stream is never synchronized (its writes were awaited already)."""
        picked = [(k, loc) for k, loc in zip(keys, locations) if self._checksum_sampled(k)]
        if not picked:
            return []
        entry = self.pools[name]
        stream = None
        if entry.kv_buffer[0].is_cuda:
            if self._checksum_stream is None:
                self._checksum_stream = torch.cuda.Stream(device=entry.kv_buffer[0].device)
            stream = self._checksum_stream
        if stream is None:
            values = page_checksums(entry, [loc for _, loc in picked])
        else:
            with torch.cuda.stream(stream):
                values = page_checksums(entry, [loc for _, loc in picked])
        return [(k, v) for (k, _), v in zip(picked, values)]

    def _record_page_checksums(self, expanded) -> None:
        """Offload thread, after the put: remember the sampled pages' checksums."""
        recorded = []
        for transfer in expanded:
            keys = self.storage._tag_keys(
                self.storage._get_hybrid_page_component_keys(list(transfer.keys), transfer)[0]
            )
            locations = self.pools[transfer.name].prepare_locations(transfer.host_indices)
            recorded += self._checksum_pages(transfer.name, keys, locations)
        with self.page_checksum_lock:
            for key, value in recorded:
                # First write wins: a later put of the same key may be skipped by
                # the store (already held), so its device bytes need not match.
                self.page_checksums.setdefault(key, value)
            while len(self.page_checksums) > 200_000:
                self.page_checksums.popitem(last=False)
            self.page_checksum_counts["recorded"] += len(recorded)

    def _verify_page_checksums(self, batches) -> None:
        """Prefetch thread, after the read landed and before publish."""
        for name, (keys, locations) in batches.items():
            for key, value in self._checksum_pages(name, keys, locations):
                with self.page_checksum_lock:
                    want = self.page_checksums.get(key)
                counts = self.page_checksum_counts
                if want is None:
                    counts["missing"] += 1
                    continue
                counts["checked"] += 1
                if value != want:
                    counts["mismatched"] += 1
                    if counts["mismatched"] <= 5:
                        logger.warning(
                            "Linker page checksum mismatch: key=%s pool=%s stored=%d read=%d",
                            key, name, want, value,
                        )

    def _remove_pages(
        self, keys: list[str], pages: Sequence[str] = (), decided: int = 0
    ) -> None:
        """Remove the objects of pages the store budget evicted (offload thread),
        and retry earlier ones a live lease refused (-706) once it can have
        expired. Until removed they still take store space; the budget counts
        them gone at once, since every stage has to decide alike and removals
        land at different times on each (the master's higher watermark covers
        the slack). A page loaded or stored again since its eviction keeps its
        objects: its removal is dropped right before the call."""
        now = time.monotonic()
        due, retries = [], self.page_remove_retries
        while retries and retries[0][0] <= now:
            due.append(retries.popleft())
        due += [
            (now, key, self._REMOVE_RETRIES, page, decided)
            for key, page in zip(keys, pages)
        ]
        retry_at = now + envs.SGLANG_LINKER_LEASE_TTL_SECS.get() + 1
        counts, tree = self.page_remove_counts, self.store_tree
        for _, key, attempts, page, decided in due:
            if tree.is_recalled(page, decided):
                counts["recalled"] += 1
                continue
            rc = self.storage.store.remove(key)
            if self.page_checksum_rate and rc in (0, -704):
                with self.page_checksum_lock:
                    self.page_checksums.pop(key, None)
            if self.hole_trace is not None and rc in (0, -704):
                self.hole_trace.note([page], "removed", "budget")
            if rc == 0:
                counts["removed"] += 1
            elif rc == -704:  # OBJECT_NOT_FOUND: already evicted
                counts["already_gone"] += 1
            elif rc == -706 and attempts > 0:  # OBJECT_HAS_LEASE
                counts["lease_retries"] += 1
                retries.append((retry_at, key, attempts - 1, page, decided))
            else:
                counts["failed"] += 1

    def _maybe_log_stats(self, every_s: float = 60.0) -> None:
        now = time.monotonic()
        if now - self.stats_logged_at < every_s:
            return
        if self.page_checksum_rate:
            c = self.page_checksum_counts
            logger.info(
                "Linker page checksum: checked %d, mismatched %d, missing %d (recorded %d)",
                c["checked"], c["mismatched"], c["missing"], c["recorded"],
            )
        window, self.stats_logged_at = now - self.stats_logged_at, now
        written, self.write_bytes = self.write_bytes, Counter()
        looked, self.lookup_counts = self.lookup_counts, Counter()
        lookups, self.lookup_samples = self.lookup_samples, []
        pins, self.pin_samples = self.pin_samples, []
        holes, self.pin_holes = self.pin_holes, 0
        engaged = dict(self.flag_hits)
        engaged["vector_put_batches"] = getattr(self.storage, "vector_put_batches", 0)
        logger.info(
            "Linker flag engagement since the last switch: %s; flags %s",
            engaged,
            self.flag_values(),
        )
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
        tree = self.store_tree
        if tree is not None:
            counts = self.page_remove_counts
            logger.info(
                "Linker store budget: %d of %d pages stored (%d pages with a state), "
                "%d evicted so far (%d put back, %d evictions stopped at a recent "
                "page); objects removed %d, kept for a reuse %d, already gone %d, "
                "lease retries %d (pending %d), failed %d",
                tree.units,
                tree.budget,
                tree.states,
                tree.counts["evicted"],
                tree.counts["recalled"],
                tree.counts["protected_stops"],
                counts["removed"],
                counts["recalled"],
                counts["already_gone"],
                counts["lease_retries"],
                len(self.page_remove_retries),
                counts["failed"],
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

    TOGGLE_FLAGS = (
        "SGLANG_LINKER_VECTOR_META",
        "SGLANG_LINKER_CPU_INDICES",
        "SGLANG_LINKER_COALESCE_RANGES",
        "SGLANG_LINKER_VECTOR_PUT",
        "SGLANG_LINKER_STORE_ARRAYS",
        "SGLANG_LINKER_PAGE_CHECKSUM",
        "SGLANG_LINKER_EARLY_COPY",
    )

    def refresh_flags(self) -> None:
        """Re-read the per-call switches from the environment."""
        self.page_checksum_rate = envs.SGLANG_LINKER_PAGE_CHECKSUM.get()
        self.vector_meta = envs.SGLANG_LINKER_VECTOR_META.get()
        self.coalesce_ranges = envs.SGLANG_LINKER_COALESCE_RANGES.get()
        self.cpu_indices = envs.SGLANG_LINKER_CPU_INDICES.get()
        store_arrays = getattr(envs, "SGLANG_LINKER_STORE_ARRAYS", None)
        if store_arrays is not None:
            self.store_arrays = store_arrays.get()
        storage = getattr(self, "storage", None)
        if storage is not None:
            storage._vector_put_flag = None

    def flag_values(self) -> dict:
        return {
            name: str(getattr(envs, name).get())
            for name in self.TOGGLE_FLAGS
            if hasattr(envs, name)
        }

    RESET_REMOVE_PASSES = 15

    def reset_cold(self) -> None:
        """After reset(): drop this stage's store bookkeeping and stats, move to a
        fresh store key namespace (late puts of the previous arm land in the old
        one and are never matched), and on PP0 remove every stored object."""
        self.reset()
        if self.store_tree is not None:
            tree = self.store_tree
            self.store_tree = _StoreTree(
                tree.budget,
                tree.state_units,
                protect=tree.protect,
                recall=tree.recall,
            )
        for table in (
            self.state_parent,
            self.state_children,
            self.state_remove_counts,
            self.page_remove_counts,
            self.read_retries,
            self.write_bytes,
            self.lookup_counts,
            self.flag_hits,
        ):
            table.clear()
        self.state_remove_retries.clear()
        self.page_remove_retries.clear()
        self.lookup_samples.clear()
        self.pin_samples.clear()
        self.pin_holes = self.state_removed = self.state_remove_failed = 0
        self.prepin_renewals = self.pin_renewals = 0
        self.stats = dict.fromkeys(self.stats, 0)
        self.storage.recently_loaded = _RecentKeys(500_000)
        if getattr(self.storage, "vector_put_batches", None) is not None:
            self.storage.vector_put_batches = 0
        if self.lease_refresher is not None and hasattr(self.lease_refresher, "clear"):
            self.lease_refresher.clear()
        if self.store_epoch == 0:
            self._base_config_prefix = self.storage.config_prefix or "linker"
        self.store_epoch += 1
        self.storage.config_prefix = f"{self._base_config_prefix}-e{self.store_epoch}"
        if self.helper is not None:
            self.helper.submit("prefix", self.storage.config_prefix).result()
        removed = None
        if self.pp_rank == 0 and self.offload_owner:
            remove_all = getattr(self.storage.store, "remove_all", None)
            if callable(remove_all):
                # One pass leaves objects under a lease or still being put (0.2-0.4 M
                # of ~8 M); repeat until a pass removes nothing (or returns an error).
                removed = []
                for _ in range(self.RESET_REMOVE_PASSES):
                    n = remove_all()
                    removed.append(n)
                    if not n or n < 0:
                        break
                    time.sleep(1.0)
        logger.info(
            "Linker cold reset: store key prefix %s, store objects removed %s",
            self.storage.config_prefix,
            removed,
        )

    def num_completed_offloads(self) -> int:
        return self.offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self.offload_results.get_nowait()

    def reset(self) -> None:
        for rid in list(self.prepins):
            self._drop_prepin(rid)
        for rid in list(self.pinned) + list(self.pinned_others):
            self.unpin(rid)
        self._end_expired_pins(float("inf"))
        self.pending_loads.clear()
        self.load_queued_at.clear()
        # Prefetch reads write device slots until they return.
        for future in list(self.prefetch_futures):
            future.cancel()
            if not future.cancelled():
                future.exception()
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
        for _ in self.load_threads:
            self.load_queue.put(None)
        self.offload_queue.put(None)
        for thread in self.load_threads:
            thread.join()
        self.offload_thread.join()
        if self.prefetch_executor is not None:
            self.prefetch_executor.shutdown(wait=True)
        if self.helper is not None:
            self.helper.close()
        logger.info("Mooncake direct linker stats: %s", self.stats)
        self.storage.close()
        if self.mla_broadcaster is not None:
            self.mla_broadcaster.destroy()
