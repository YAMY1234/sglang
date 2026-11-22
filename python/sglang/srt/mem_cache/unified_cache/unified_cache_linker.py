"""Direct L3 support for :class:`UnifiedRadixCache`.

Links the cache's device pools straight to an external KV store, with no host
tier in between. The transport contract and tree-side wrapper live here, while
each backend owns its device-pool layout and physical I/O.

* :class:`UnifiedCacheLinker` -- the transport interface a backend implements.
* :class:`UnifiedCacheLinkerWrapper` -- the tree-side flow that drives it. The
  cache owns one as a plain attribute, keeping the whole external-cache path out
  of the main tree file.

The tree only needs a handful of guarded hooks:

* ``match_prefix``      -> :meth:`UnifiedCacheLinkerWrapper.match`
* ``_prefetch_kvcache`` -> full-layer device prefetch before admission
* ``init_load_back``    -> :meth:`UnifiedCacheLinkerWrapper.load_back`
* ``check_hicache_events`` -> full-layer prefetch publication before admission
* ``BackupKV`` actions  -> :meth:`UnifiedCacheLinkerWrapper.offload_nodes`

A layer-wise load lands in device slots private to the loading request and joins the tree
only when that request inserts it like any computed KV. A failed load therefore
reaches nothing but its own requests, which the scheduler aborts.
Prefetch publishes complete KV before admission and recomputes failed reads.
"""

from __future__ import annotations

import json
import logging
import re
import os
import socket
import time
import urllib.parse
import urllib.request
import zlib
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    DecLockRefParams,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.near_term import load_fits
from sglang.srt.mem_cache.unified_cache.pp_linker_relay import PPLinkerRelay
from sglang.srt.mem_cache.unified_cache.components import (
    ComponentType,
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
    TreeComponent,
)
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.srt.session.streaming_session import is_virtual_node
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)
device_module = get_device_module()

_EXTERNAL_LINKER_SUPPORTED_COMPONENTS = frozenset(
    {
        ComponentType.FULL,
        ComponentType.SWA,
        ComponentType.MAMBA,
    }
)


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook.

    A failed layer does not raise into the forward that waits on it: the
    forward runs on whatever the slots hold, and :meth:`finish` reports the
    failure afterwards so the scheduler can abort the loading requests before
    their output is used.
    """

    def __init__(
        self,
        num_layers: int,
        on_layer_ready: Optional[Callable[[int, int], None]] = None,
    ):
        self.num_layers = num_layers
        self.on_layer_ready = on_layer_ready
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}

    def update_producer(self) -> int:
        self.producer_index += 1
        self.futures[self.producer_index] = [Future() for _ in range(self.num_layers)]
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        self.futures[index][layer].set_result(None)

    def fail(self, index: int, error: BaseException) -> None:
        for future in self.futures.get(index, ()):
            if not future.done():
                future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self.futures.get(index)
        if futures is None:
            return
        # Blocks until the layer resolves; a failure is reported by finish().
        futures[threshold].exception()
        # Runs on failure too: an MLA dedup source must still issue the layer
        # broadcast its peers are waiting for.
        if self.on_layer_ready is not None:
            self.on_layer_ready(index, threshold)

    def finish(self, index: int) -> bool:
        """Wait for every layer of batch ``index``; True when all of them landed."""
        futures = self.futures.get(index)
        if futures is None:
            return True
        # Wait before dropping the entry: the load thread still completes the
        # layers the forward never waited on through it.
        landed = all(future.exception() is None for future in futures)
        del self.futures[index]
        return landed

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self.futures.clear()


class TransferReady(NamedTuple):
    events: list[torch.cuda.Event]

    def synchronize(self) -> None:
        for event in self.events:
            event.synchronize()

    def wait(self, stream: torch.cuda.Stream) -> None:
        for event in self.events:
            stream.wait_event(event)


class UnifiedCacheLinker(ABC):
    """External KV store reached directly from the device pools.

    Out-of-tree linkers are constructed per worker as
    ``Linker(server_args, params, components=..., extra_config=..., forward_stream=...)``.
    Store ``forward_stream`` and use ``record_transfer_ready()`` to gate loads
    and offloads. ``layer_done_counter`` provides ``set_consumer(index)`` and
    ``wait_until(layer)``. Report completion only after device copies finish.
    """

    layer_done_counter: object
    pools: dict[PoolName, object]
    load_mode = "layer_wise"
    prefetch_policy = "timeout"
    # None when the current stream is the only one using the pages.
    forward_stream: Optional[torch.cuda.Stream] = None

    def record_transfer_ready(self) -> TransferReady:
        """Record what a transfer waits on before it touches device pages.

        The current stream computed the page indices; under read-done the
        forward in flight may still use those pages, including freed ones.
        """
        indices_ready = device_module.Event()
        indices_ready.record()
        events = [indices_ready]
        if self.forward_stream is not None:
            forward_done = device_module.Event()
            forward_done.record(self.forward_stream)
            events.append(forward_done)
        return TransferReady(events)

    def prefetch(self, transfers: list[PoolTransfer]) -> Future[bool] | None:
        """Read all layers; the future completes after all device writes stop."""
        return None

    @abstractmethod
    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        """Return every prefix length (in pages) that is fully restorable.

        A length is included only when *all* pools satisfy their hit policy at
        that exact boundary (contiguous prefix pools, plus each trailing-window
        pool's window ending there). Trailing-window state (SWA / compress
        state) only exists at offloaded node boundaries, so the set is sparse
        and generally non-contiguous -- returning just the local maximum would
        let the tree pick a length that is invalid on another rank.

        Local to this rank; the tree intersects the sets across ranks.
        """

    @abstractmethod
    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        """Queue a load into the given device indices.

        The transfer is executed by the next ``start_layer_wise_loading`` call,
        not here.
        """

    @abstractmethod
    def start_layer_wise_loading(self) -> int:
        """Start queued loads and return the layer-counter consumer index."""

    @abstractmethod
    def finish_layer_wise_loading(self, counter_index: int) -> bool:
        """Wait until the batch started as ``counter_index`` stops writing to
        device memory; True when every layer landed."""

    @abstractmethod
    def cancel_queued_load(self, rid: str) -> bool:
        """Cancel a load that has not started yet."""

    @abstractmethod
    def offload(
        self, transfers: list[PoolTransfer], session_id: Optional[str] = None
    ) -> bool:
        """Queue every transfer for atomic persistence.

        ``session_id`` names the session whose insert produced the transfers,
        when known; backends may attribute the stored objects to it.
        """

    @abstractmethod
    def num_completed_offloads(self) -> int:
        """Return the number of completed offloads waiting to be consumed."""

    @abstractmethod
    def pop_completed_offload(self) -> bool:
        """Consume the oldest completed offload and return its result."""

    @abstractmethod
    def reset(self) -> None:
        """Quiesce all transfers and reset backend state before returning."""

    @abstractmethod
    def close(self) -> None:
        """Quiesce all transfers and release backend resources."""


class ExternalCacheHitMarker(NamedTuple):
    """What ``match`` found in the external store, consumed by ``load_back``.

    ``tail_hashes`` are the per-page storage hashes of the restorable tail,
    starting at ``device_hit_len``.
    """

    tail_hashes: list[str]
    device_hit_len: int


@dataclass
class _ExternalCachePrefetch:
    key: RadixKey
    anchor_node_id: NodeId
    anchor_lock: DecLockRefParams
    component_transfers: list[tuple[TreeComponent, PoolTransfer]]
    future: Future[bool] | None = None
    timeout_s: float = 0.0
    wait_started: float | None = None
    discard: bool = False


@dataclass
class _PendingLookup:
    """A queued request's store lookup running off the scheduler thread."""

    key: RadixKey
    hashes: list[str]
    start: int
    anchor_node_id: NodeId
    anchor_lock: DecLockRefParams
    future: Future[list[int]]
    timeout_s: float = 0.0
    wait_started: float | None = None
    kind: str = "enq"
    issued_at: float = 0.0


_OBJECT_NOT_FOUND = -704  # Mooncake ErrorCode (mooncake-store/include/types.h)


@dataclass
class _LeaseChain:
    """A queued request's store chain whose read lease this stage keeps renewing."""

    rid: str
    hashes: list[str]
    pages: Optional[int]  # this stage's restorable pages at its lookup (None until known)
    last: float  # when this stage last touched the chain (lookup or refresh)


class _PrefetchHold(NamedTuple):
    node_id: NodeId
    lock: DecLockRefParams
    tokens: int


class _PendingOffload(NamedTuple):
    lock_node_id: NodeId
    lock_params: DecLockRefParams
    publish_node_ids: list[NodeId]


# requery_after_device_loss: the request waits for its relay cycle to settle. Truthy
# like a re-query, but the scheduler does not count it against the attempt cap.
REQUERY_SETTLING = 2


def near_term_attention_ranks(cache) -> int:
    """Ranks _all_reduce_attn_groups spans for ``cache`` (1 for the PP4 TP1 prefill)."""
    sizes = [
        torch.distributed.get_world_size(group=g)
        for g in (cache.attn_cp_group, cache.attn_tp_group)
        if g is not None
    ]
    return max(sizes + [cache.tp_world_size])


class UnifiedCacheLinkerWrapper:
    """Drives an external KV store on behalf of one :class:`UnifiedRadixCache`."""

    def __init__(
        self,
        cache: UnifiedRadixCache,
        cache_linker: UnifiedCacheLinker,
    ):
        unsupported = set(cache.tree_components) - _EXTERNAL_LINKER_SUPPORTED_COMPONENTS
        if unsupported:
            names = ", ".join(
                component.name for component in sorted(unsupported, key=int)
            )
            raise ValueError(
                "External cache linker supports only Full, SWA and Mamba tree "
                f"components; unsupported: {names}"
            )

        self.cache = cache
        self.cache_linker = cache_linker
        self.load_mode = cache_linker.load_mode
        self.pp_relay = None
        if self.load_mode == "prefetch" and cache.pp_size > 1:
            if not envs.SGLANG_EXTERNAL_LINKER_PP_RELAY.get():
                raise ValueError("External linker PP prefetch requires PP_RELAY.")
            if not envs.SGLANG_PREFETCH_VERDICT_RELAY.get():
                raise ValueError("External linker PP requires L's prefetch verdict relay.")
            if cache.tp_world_size != 1:
                raise ValueError("External linker PP relay is qualified for TP1 only.")
            if envs.SGLANG_EXTERNAL_LINKER_ASYNC_LOOKUP_THREADS.get() <= 0:
                raise ValueError("External linker PP requires asynchronous lookup.")
            if envs.SGLANG_EXTERNAL_LINKER_PREFETCH_HOLD_FRACTION.get() <= 0:
                raise ValueError("External linker PP requires a hold on every stage.")
            self.pp_relay = PPLinkerRelay(self)
        swa = cache.components.get(ComponentType.SWA)
        # A store without SWA pages tombstones restored tails instead.
        self._skip_swa = swa is not None and PoolName.SWA not in cache_linker.pools
        self._components = tuple(
            component
            for component in cache._components_tuple
            if not (self._skip_swa and component is swa)
        )
        # rid -> what match found, consumed by the next init_load_back.
        self.hit_markers: dict[str, ExternalCacheHitMarker] = {}
        self.prefetches: dict[CacheRequestHandle, _ExternalCachePrefetch] = {}
        # Store lookups in flight on worker threads, in submission order.
        self.lookups: dict[CacheRequestHandle, _PendingLookup] = {}
        self._draining_lookups = False
        lookup_threads = envs.SGLANG_EXTERNAL_LINKER_ASYNC_LOOKUP_THREADS.get()
        self._lookup_executor = (
            ThreadPoolExecutor(lookup_threads, thread_name_prefix="linker-lookup")
            if self.load_mode == "prefetch" and lookup_threads > 0
            else None
        )
        # Published prefetches still locked for their waiting request.
        self.holds: dict[CacheRequestHandle, _PrefetchHold] = {}
        self.held_tokens = 0
        hold_fraction = envs.SGLANG_EXTERNAL_LINKER_PREFETCH_HOLD_FRACTION.get()
        self.hold_cap_tokens = (
            int(hold_fraction * cache.token_to_kv_pool_allocator.size_full)
            if self.load_mode == "prefetch" and hold_fraction > 0
            else 0
        )
        load_fraction = envs.SGLANG_EXTERNAL_LINKER_LOAD_FRACTION.get()
        self.load_cap_tokens = (
            int(load_fraction * cache.token_to_kv_pool_allocator.size_full)
            if self.load_mode == "prefetch" and load_fraction > 0
            else 0
        )
        # Device tokens a new load must leave free (+ evictable): a chunked
        # continuation is admitted even with no budget, so loads must not take it.
        self.load_free_reserve_tokens = int(
            os.environ.get("SGLANG_EXTERNAL_LINKER_LOAD_FREE_RESERVE_TOKENS", "0")
        )
        # Tokens the in-flight chunked request still needs; the scheduler sets it.
        self.chunk_tail_tokens = 0
        # Near-term prefetch: a finished lookup's load starts only once its request
        # is in the window the scheduler plans for the next K prefill batches.
        near_term = int(os.environ.get("SGLANG_EXTERNAL_LINKER_NEAR_TERM_BATCHES", "0"))
        self.near_term_batches = near_term if self.load_mode == "prefetch" else 0
        # Window handle -> prefill work ahead of it in the queue (None: near-term off).
        self.near_term_window: Optional[dict] = {} if self.near_term_batches else None
        self.near_term_head: Optional[CacheRequestHandle] = None
        self.near_term_head_reserve = 0
        # Prefill tokens of lead per restored token (prefill tok/s x pool B/token / restore B/s).
        self.near_term_restore_ratio = float(
            os.environ.get("SGLANG_EXTERNAL_LINKER_NEAR_TERM_RESTORE_RATIO", "0")
        )
        # Device tokens out-of-window lookups may keep anchored; past it the newest
        # agreed hits are parked: anchor released, re-queried once in the window.
        self.near_term_anchor_cap = int(
            float(os.environ.get("SGLANG_EXTERNAL_LINKER_NEAR_TERM_ANCHOR_FRACTION", "0.25"))
            * cache.token_to_kv_pool_allocator.size_full
        )
        self.near_term_parked: dict[CacheRequestHandle, int] = {}
        # An out-of-window lookup whose RPC never returns still gets a (long) stop clock.
        self.near_term_hung_lookup_s = float(
            os.environ.get("SGLANG_EXTERNAL_LINKER_NEAR_TERM_HUNG_LOOKUP_S", "30")
        )
        # Liveness backstop: a queue head blocked this long (not on its own I/O)
        # releases every out-of-window anchor and every non-head hold.
        self.near_term_stall_s = float(
            os.environ.get("SGLANG_EXTERNAL_LINKER_NEAR_TERM_STALL_S", "10")
        )
        self._stall_head: Optional[CacheRequestHandle] = None
        self._stall_since = 0.0
        self._stall_backoff = 0
        self._no_token_head: Optional[CacheRequestHandle] = None
        self.near_term_stall_max_s = float(
            os.environ.get("SGLANG_EXTERNAL_LINKER_NEAR_TERM_STALL_MAX_S", "60")
        )
        # Admission waits on a settling relay cycle, outside the re-query attempt cap.
        self._settle_waits: dict[CacheRequestHandle, int] = {}
        self.near_term_settle_wait_max = int(
            os.environ.get("SGLANG_EXTERNAL_LINKER_SETTLE_WAIT_MAX", "64")
        )
        # F6c7e: past the re-query attempt cap, a request whose last load was refused for device
        # room keeps re-querying until it has waited this long since enqueue on PP0's clock, then
        # goes cold. 0 = F6c7c (the attempt cap admits it cold).
        self.wait_fit_s = float(os.environ.get("SGLANG_EXTERNAL_LINKER_WAIT_FIT_S", "0"))
        self._wait_fit_enq: dict[CacheRequestHandle, float] = {}
        self._room_refused: set[CacheRequestHandle] = set()
        self.requery_over_attempts = False  # the scheduler sets it before each re-query call
        if self.near_term_batches and near_term_attention_ranks(cache) > 1:
            # Window and lookup state are rank-local; collectives would diverge.
            raise ValueError(
                "External linker near-term prefetch needs one attention rank per prefill stage"
            )
        if self.near_term_batches:
            # Only window requests load or hold; the free reserve, not a fraction, bounds them.
            self.hold_cap_tokens = cache.token_to_kv_pool_allocator.size_full
            self.load_cap_tokens = self.hold_cap_tokens
            self.load_free_reserve_tokens = self.load_free_reserve_tokens or 32768
        # Published prefetches left unheld (over the cap or not resident): their end.
        self.unheld: dict[CacheRequestHandle, int] = {}
        self._hold_stats = dict.fromkeys(
            (
                "held",
                "over_cap",
                "not_resident",
                "admitted",
                "released",
                "unheld_kept",
                "unheld_evicted",
                "unheld_evicted_tokens",
            ),
            0,
        )
        self._hold_next_log = 0.0
        # Queued request -> the prefix its enqueue-time lookup left on device
        # (device match plus the accepted load); a shorter match at admission re-queries.
        self.requery_on_device_loss = (
            self.load_mode == "prefetch"
            and envs.SGLANG_EXTERNAL_LINKER_REQUERY_ON_DEVICE_LOSS.get()
        )
        self.match_targets: dict[CacheRequestHandle, int] = {}
        self.log_lookup_stats = envs.SGLANG_EXTERNAL_LINKER_LOOKUP_STATS.get()
        # One line per request at admission (PP0), joined to client exports by input length + time.
        self.request_log = (
            os.environ.get("SGLANG_EXTERNAL_LINKER_REQUEST_LOG") == "1" and cache.pp_rank == 0
        )
        self._request_records: dict[CacheRequestHandle, dict] = {}
        self._lookup_stats = dict.fromkeys(
            ("issued", "skipped_no_hash", "zero_restorable", "requeried", "load_capped", "load_reserve", "anchor_skip",
             "lookup_timeout", "parked", "stall_release", "settle_cap_hit",
             "refresh_rpcs", "refresh_keys", "refresh_found_short",
             "window_unhold", "load_waits", "head_room_park", "head_room_release", "admitted_reads", "reads_sum", "reads_2", "reads_3plus",
             "reads_max", "reread_tokens", "big_admits", "extra_beats", "extra_beats_big",
             "late_admits", "late_admits_big", "wait_fit_requery", "wait_fit_cap", "lease_budget_skip",
             "touch_dev_rpcs", "touch_dev_keys", "touch_put_rpcs", "touch_put_keys",
             "probe_n", "probe_pages", "probe_prefix_pages", "probe_fragment_pages", "probe_errors",
             "probe_selfcheck_fail", "probe_no_hit", "probe_dropped", "probe_ms_sum", "probe_ms_max",
             "evict_rounds", "evict_sessions", "evict_pages", "evict_objects", "evict_refused", "evict_errors",
             "evict_selfcheck_fail", "evict_selfcheck_none", "evict_shared_skip", "evict_device_skip", "evict_kept_pages",
             "evict_share_peers", "evict_fill_permille", "evict_ms_max"),
            0,
        )
        # Re-queries wait for the next event-loop boundary, so no load starts
        # inside the admission loop after the batch budget was taken.
        self._pending_requeries: dict[CacheRequestHandle, Req] = {}
        # Device tokens held by in-flight loads plus holds, at their peak since the last log line.
        self._occupied_peak = 0
        self._lookup_next_log = 0.0
        # Re-queried request -> when its re-query started; waits until publication (s).
        self._requery_started: dict[CacheRequestHandle, float] = {}
        self._requery_waits: list[float] = []
        # (rid, seconds waited, outcome) per re-query since the last log line, for joins with client exports.
        self._requery_records: list[tuple[str, float, str]] = []
        # Lookup kind ("enq" at enqueue, "rq" for a re-query) -> seconds per phase since the last log line.
        self._lookup_kind = "enq"
        self._phase_s = {k: dict.fromkeys(("n", "match", "keys", "rpc", "load"), 0.0) for k in ("enq", "rq")}
        # The store evicts the oldest read lease first (lookups and reads grant ~10 s), so a
        # queued chain nobody touches ages out while its request waits for the window.
        # Each stage renews its own chains every lease_refresh_s, oldest first, under a page
        # budget per second (0 = off). In a wave the budget binds (~120 chains x ~3.5k pages
        # at 25k pages/s is one renewal per chain every ~17 s), bounding the master's exist
        # load. Metadata only: the result feeds counters, never a decision.
        self.lease_refresh_s = float(os.environ.get("SGLANG_EXTERNAL_LINKER_LEASE_REFRESH_S", "5"))
        self.lease_refresh_max_keys_s = float(
            os.environ.get("SGLANG_EXTERNAL_LINKER_LEASE_REFRESH_MAX_KEYS_S", "25000")
        )
        self._lease_chains: dict[CacheRequestHandle, _LeaseChain] = {}
        self._lease_refresh_futures: list[tuple[Future, int]] = []
        self._lease_tick = 0.0
        # F9: renew only the queue's head, in enqueue order, up to this many tokens of chain; members past it
        # age like any unleased chain (fix 3's park / re-query at admission catches a shortened one). Pinning
        # every queued chain let a C320 burst hold 75% of the pool and evictions cut the sessions not queued.
        # 0 = every queued chain (fix 3).
        self.lease_queue_budget_tokens = int(
            os.environ.get("SGLANG_EXTERNAL_LINKER_LEASE_QUEUE_BUDGET_TOKENS", "0")
        )
        # (c1): the master evicts the oldest read lease first and a put sets none, so a chain's never-read pages (the
        # device-covered head, fresh writes) go first and cut it. Touch them (an ExistKey grants the 10 s read lease)
        # at each lookup and after each put, so eviction drops least recently used sessions. Metadata only, stage-local.
        self.touch_device_prefix = os.environ.get("SGLANG_EXTERNAL_LINKER_TOUCH_DEVICE_PREFIX") == "1"
        self.touch_after_put = os.environ.get("SGLANG_EXTERNAL_LINKER_TOUCH_AFTER_PUT") == "1"
        # Fragment probe (diagnostic, PP0): for a sampled fraction of lookups, read the chain's keys past the device
        # prefix through the master's admin endpoint, which grants no lease (BatchGetReplicaListForAdmin), and log
        # how much of the chain survives past its first missing page. Off unless both are set.
        self.fragment_probe_frac = float(os.environ.get("SGLANG_EXTERNAL_LINKER_FRAGMENT_PROBE", "0"))
        admin = os.environ.get("SGLANG_EXTERNAL_LINKER_ADMIN_URL", "auto")
        if admin in ("", "auto"):  # the master the store client uses (MOONCAKE_MASTER host:port), its admin HTTP port
            master = os.environ.get("MOONCAKE_MASTER", "")
            port = os.environ.get("SGLANG_EXTERNAL_LINKER_ADMIN_PORT", "8702")
            admin = f"http://{master.rsplit(':', 1)[0]}:{port}" if master else ""
        self.admin_url = admin.rstrip("/")
        # The probe never runs on the scheduler or the lookup executor: one dedicated worker, one probe at a time
        # (a probe sampled while one runs is dropped and counted); its cost is in probe_ms_sum / probe_ms_max.
        self._probe_executor: Optional[ThreadPoolExecutor] = None
        self._probe_busy = False
        # (c3) session eviction (PP0 of each P): keep pool fill at or below this fraction of capacity by removing whole
        # least recently used sessions, tail pages first, every stage's objects of a page in one batch, so the master's
        # BatchEvict (random victims among equal leases) need not run. Never forced: the master refuses objects under a
        # read lease (OBJECT_HAS_LEASE), which covers chains either P read in the last 10 s, queued chains (fix 3) and
        # loads. Removal only: no relay state, vote or decision changes. 0 = off.
        self.session_evict_target = float(os.environ.get("SGLANG_EXTERNAL_LINKER_SESSION_EVICT_TARGET", "0"))
        self.session_evict_period_s = float(os.environ.get("SGLANG_EXTERNAL_LINKER_SESSION_EVICT_PERIOD_S", "2"))
        self.session_evict_min_idle_s = float(os.environ.get("SGLANG_EXTERNAL_LINKER_SESSION_EVICT_MIN_IDLE_S", "30"))
        # This P's share of the excess: two P read the same fill and each removes its share of it.
        self.session_evict_share = float(os.environ.get("SGLANG_EXTERNAL_LINKER_SESSION_EVICT_SHARE", "0.5"))
        self._sessions: OrderedDict = OrderedDict()  # session -> [page hashes from page 0, last use]; LRU first
        self._request_session: dict[CacheRequestHandle, str] = {}
        self._evict_executor: Optional[ThreadPoolExecutor] = None
        self._evict_busy = False
        self._evict_next = 0.0
        self._evict_keys_ok: Optional[bool] = None  # None until the first round's key self-check
        self._evict_done: deque = deque()  # (session, pages left) from the worker, applied on the scheduler thread
        # Other P's use of a session (under LB a session's last turn is often on another P): every round each P's PP0
        # writes the ages of its sessions used in the last 300 s to its own file in this shared directory, and the worker
        # ranks candidates by the youngest age over all P. c3 needs it; unset keeps c3 off.
        self.session_share_dir = os.environ.get("SGLANG_EXTERNAL_LINKER_SESSION_SHARE_DIR", "")
        self._share_name = f"sessions-{socket.gethostname()}-{os.getpid()}.json"
        # Every stage writes its store key suffix (suffix-pp<k>) to the share dir once; PP0 builds another stage's keys by
        # swapping its own full suffix for that stage's (the NVFP4 linker appends a per-stage digest to it).
        self._suffix_written = False
        self._stage_suffixes: dict[int, str] = {}
        self._evict_logged = False
        # Last page of each recent pool hit every stage agreed on (read, so under a 10 s lease) or of a recent write:
        # the self-check's known-present pages until c3 has its verdict. Any one still present is enough.
        self._known_present: deque = deque(maxlen=16)
        # PP0 re-looks-up a held hit older than this before loading it (0 = off).
        self.stale_hit_s = float(os.environ.get("SGLANG_EXTERNAL_LINKER_STALE_HIT_S", "10"))
        # One batch ahead (needs NEAR_TERM_BATCHES=1): the adder keeps room for a request
        # skipped while its read runs, and a hold the next batch will not admit is released
        # at the loop hook (off = fix 3 behaviour).
        self.one_batch_ahead = (
            os.environ.get("SGLANG_EXTERNAL_LINKER_ONE_BATCH_AHEAD") == "1"
            and self.load_mode == "prefetch"
        )
        if self.one_batch_ahead and (self.near_term_batches != 1 or not self.requery_on_device_loss):
            # The committed number is read from the re-query targets.
            raise ValueError(
                "SGLANG_EXTERNAL_LINKER_ONE_BATCH_AHEAD=1 needs SGLANG_EXTERNAL_LINKER_NEAR_TERM_BATCHES=1 "
                "and SGLANG_EXTERNAL_LINKER_REQUERY_ON_DEVICE_LOSS=1"
            )
        self.big_prefix_tokens = int(
            os.environ.get("SGLANG_EXTERNAL_LINKER_BIG_PREFIX_TOKENS", "550000")
        )
        # F6c7b: the queue head's load skips the anchor allowance. The head is admitted next and
        # the adder keeps its room (committed number), so the load takes only memory its cold
        # compute would take; chunk tail, work ahead and the reserve still apply.
        self.head_load_skips_allowance = self.one_batch_ahead and (
            os.environ.get("SGLANG_EXTERNAL_LINKER_HEAD_LOAD_SKIPS_ALLOWANCE", "1") == "1"
        )
        # F6c7c: every one-batch window member's load skips the anchor allowance (work ahead,
        # chunk tail and reserve still apply); PP0 parks out-of-window anchors that would leave
        # the head less than its work (relayed, so every stage releases the same anchors).
        self.window_load_skips_allowance = self.one_batch_ahead and (
            os.environ.get("SGLANG_EXTERNAL_LINKER_WINDOW_LOAD_SKIPS_ALLOWANCE", "0") == "1"
        )
        self.near_term_head_work = 0
        # Per queued request: [reads started, tokens read, first read's tokens, first
        # read's end]; and admission passes skipped with its read in flight. Both feed
        # the churn / extra-beat stats when the request is admitted.
        self._reads: dict[CacheRequestHandle, list[int]] = {}
        self._late_beats: dict[CacheRequestHandle, int] = {}
        if self.near_term_batches:
            logger.info(
                "External linker near-term prefetch on: loads start for the next %d "
                "prefill batches, %d free tokens kept (fraction caps lifted)",
                self.near_term_batches,
                self.load_free_reserve_tokens,
            )
        elif self.hold_cap_tokens:
            logger.info(
                "External linker prefetch holds on: up to %d tokens (%.2f of the "
                "Full device pool) stay locked until admission",
                self.hold_cap_tokens,
                hold_fraction,
            )
        # Requests loading in the batch being built, then its counter index once
        # the batch starts; finish_loads consumes both after its forward.
        self.inflight_load_rids: list[str] = []
        # Loaded tails stay outside the tree until their forward completes.
        self.private_load_indices: dict[str, torch.Tensor] = {}
        self.inflight_load_index = -1
        # Offloads in flight, each holding a lock on its node until it lands.
        self.pending_offloads: list[_PendingOffload] = []
        # Page-aligned radix-key prefix excluded from offload during an insert.
        self.offload_skip_prefix_len = 0

        cache.tree_core.enable_external_cache_linker = True
        cache.write_through_threshold = 1

    @property
    def layer_done_counter(self) -> object:
        return self.cache_linker.layer_done_counter

    def has_hit(self, rid: str) -> bool:
        return rid in self.hit_markers

    # ---- match: probe the remote store and report host_hit_length ----

    def match(self, key: RadixKey, req: Req, result: MatchResult) -> MatchResult:
        cache = self.cache
        key, _ = key.maybe_to_bigram_view(cache.tree_core.is_eagle)
        page = cache.page_size
        device_hit_len = result.device_prefix_len
        if self.load_mode == "prefetch":
            # Prefetch is issued on enqueue; only completed GPU KV is matchable.
            return result
        if device_hit_len >= len(key):
            return result

        tail_hashes = self._tail_hashes(key, result, device_hit_len)
        if not tail_hashes:
            return result

        lookup_transfers = []
        for component in self._components:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOOKUP, None, tail_hashes
            )
            if transfer is None:
                return result
            lookup_transfers.append(transfer)
        by_pool = {transfer.name: transfer for transfer in lookup_transfers}

        # Tail-relative: page 0 of `tail_hashes` is the first uncached page.
        hit_pages = self._sync_restorable_prefix(
            self.cache_linker.lookup(req.rid, lookup_transfers),
            num_pages=len(tail_hashes),
            device_hit_pages=0,
        )
        if hit_pages == 0:
            return result
        hit_tokens = hit_pages * page

        swa_transfer = by_pool.get(PoolName.SWA)
        swa_host_hit_length = (
            min(len(swa_transfer.keys), hit_pages) * page
            if swa_transfer is not None
            else 0
        )
        # Mamba keeps a single state slot per node, so a hit is worth one slot.
        mamba_host_hit_length = 1 if PoolName.MAMBA in by_pool else 0

        self.hit_markers[req.rid] = ExternalCacheHitMarker(
            tail_hashes=list(tail_hashes[:hit_pages]),
            device_hit_len=device_hit_len,
        )
        return result._replace(
            last_host_node=result.best_match_node,
            host_hit_length=hit_tokens,
            swa_host_hit_length=max(result.swa_host_hit_length, swa_host_hit_length),
            mamba_host_hit_length=max(
                result.mamba_host_hit_length, mamba_host_hit_length
            ),
        )

    def prefetch(self, req: Req) -> None:
        """Stage a queued request; publish only after all ranks succeed."""
        if self.load_mode != "prefetch":
            return
        request = req.cache_request_handle
        self.near_term_parked.pop(request, None)
        if self.wait_fit_s:
            self._wait_fit_enq.setdefault(request, time.perf_counter())
        if self.pp_relay is not None:
            self.pp_relay.begin(req)
            entry = self.pp_relay.entries[request]
            if entry.phase != "lookup" or entry.stopped:
                return  # Its cycle is still settling; a lookup now would get no vote.
        # A superseded handle of this rid drops everything it holds, its relay entry too.
        relay_entries = self.pp_relay.entries if self.pp_relay is not None else ()
        superseded = {
            h
            for h in (*self.prefetches, *self.lookups, *self.holds, *self.near_term_parked,
                      *self.match_targets, *relay_entries)
            if h.rid == request.rid and h != request
        }
        for previous in superseded | {h for h in self._pending_requeries if h.rid == request.rid and h != request}:
            if previous in self.prefetches:
                self._cancel_prefetch(previous)
            self._cancel_lookup(previous)
            self._pending_requeries.pop(previous, None)  # a queued re-query would reopen it
            self._requery_started.pop(previous, None)
            self.release_hold(previous, admitted=False)
            self.cache.prefetch_loaded_tokens_by_reqid.pop(previous, None)
            self.cache.prefetch_loaded_storage_start_by_reqid.pop(previous, None)
        if self.cache.disable or request in self.prefetches or request in self.lookups:
            return
        # F6c7e: a new lookup cycle starts; only a room refusal in it marks the request again
        # (a chain gone from the pool, or agreed 0, must not keep it waiting).
        self._room_refused.discard(request)
        timed = self.log_lookup_stats
        kind = self._lookup_kind
        t0 = time.perf_counter() if timed else 0.0
        req.init_next_round_input(self.cache, cow_mamba=False)
        t1 = time.perf_counter() if timed else 0.0
        if is_virtual_node(req.last_node) or req.positional_embed_overrides is not None:
            return
        page = self.cache.page_size
        key = RadixKey(
            req.full_untruncated_fill_ids,
            extra_key=req.extra_key,
            cache_salt=req.cache_salt,
            is_bigram=self.cache.tree_core.is_eagle,
            limit=req._compute_max_prefix_len(len(req.full_untruncated_fill_ids)),
        ).page_aligned(page)
        start = req.prefix_len
        if self.request_log:
            self._note_enqueue(request, len(req.origin_input_ids), start, len(key) <= start)
        if self.requery_on_device_loss:
            self.match_targets[request] = start
        last_hash = self.cache.get_last_hash_value(req.last_node)
        if len(key) <= start:
            return
        if start and last_hash is None:
            self._lookup_stats["skipped_no_hash"] += 1
            self._note(request, "no_hash")
            return
        hashes = get_storage_hash_str(key[start:], last_hash, page_size=page)
        lookups = [
            c.build_external_linker_transfer(LinkerTransferPhase.LOOKUP, None, hashes)
            for c in self._components
        ]
        if any(t is None for t in lookups):
            return
        if timed:
            phase = self._phase_s[kind]
            phase["n"] += 1
            phase["match"] += t1 - t0
            phase["keys"] += time.perf_counter() - t1
        timeout_s = (
            self.cache.prefetch_timeout_base
            + len(hashes) * self.cache.prefetch_timeout_per_page
        )
        # The anchor turns evictable device KV into protected KV. A forced chunk
        # continuation counts on that KV, so keep free + evictable at the reserve.
        free = self.cache.token_to_kv_pool_allocator.available_size()
        before = free + self.cache.evictable_size()
        anchor_lock = self.cache.inc_lock_ref(req.last_node).to_dec_params()
        after = free + self.cache.evictable_size()
        if after < before and after < max(self.load_free_reserve_tokens, self.chunk_tail_tokens):
            self.cache.dec_lock_ref(req.last_node, anchor_lock)
            self._lookup_stats["anchor_skip"] += 1
            self._log_lookup_stats()
            return
        self._lookup_stats["issued"] += 1
        self._log_lookup_stats()
        if self._lookup_executor is not None:
            # The anchor stays locked while the lookup runs, as for a load.
            self.lookups[request] = _PendingLookup(
                key,
                hashes,
                start,
                req.last_node,
                anchor_lock,
                self._lookup_executor.submit(
                    self._timed_lookup, kind if timed else None, req.rid, lookups
                ),
                timeout_s=timeout_s,
                kind=kind,
                issued_at=time.monotonic(),
            )
            self._lease_chains[request] = _LeaseChain(req.rid, hashes, None, time.monotonic())
            if self.touch_device_prefix and start:
                own = self.cache.tree_core.get_hash_values(req.last_node)
                self._touch(req.rid, (self.cache.get_prefix_hash_values(req.last_node) + own)[: start // page], "touch_dev")
            if self.session_evict_target and (self.pp_relay is None or not self.pp_relay.rank) and req.session_id:
                head = []
                if start:
                    own = self.cache.tree_core.get_hash_values(req.last_node)
                    head = (self.cache.get_prefix_hash_values(req.last_node) + own)[: start // page]
                # [page hashes from page 0, last use, device-prefix pages, that prefix's device node, its last hash]
                self._sessions[req.session_id] = [
                    head + hashes, time.monotonic(), len(head), req.last_node if head else None, head[-1] if head else None
                ]
                self._sessions.move_to_end(req.session_id)
                self._request_session[request] = req.session_id
            if (
                self.fragment_probe_frac
                and self.admin_url
                and (self.pp_relay is None or not self.pp_relay.rank)
                and zlib.crc32(req.rid.encode()) % 10000 < self.fragment_probe_frac * 10000
            ):
                self._enqueue_probe(req.rid, lookups, self.lookups[request].future)
            return
        try:
            restorable = self._timed_lookup(kind if timed else None, req.rid, lookups)
        except Exception:
            logger.warning("External linker prefetch lookup failed", exc_info=True)
            restorable = []
        t2 = time.perf_counter() if timed else 0.0
        self._start_prefetch_load(
            request, key, hashes, start, req.last_node, anchor_lock, restorable, timeout_s
        )
        if timed:
            self._phase_s[kind]["load"] += time.perf_counter() - t2

    def _maybe_session_evict(self) -> None:
        """Scheduler thread (PP0): apply the worker's results, then every period hand the worker the oldest idle sessions
        that have no request queued, loading or held. The worker reads the fill and removes."""
        while self._evict_done:
            sid, left = self._evict_done.popleft()
            if sid in self._sessions:
                if left:
                    self._sessions[sid][0] = self._sessions[sid][0][:left]
                else:
                    del self._sessions[sid]
        now = time.monotonic()
        if self._evict_busy or now < self._evict_next or self._evict_keys_ok is False:
            return
        self._evict_next = now + self.session_evict_period_s
        busy = {self._request_session.get(h) for h in (*self.lookups, *self.prefetches, *self.holds, *self._lease_chains)}
        candidates, heads, checked = [], [], {}  # heads: (page hashes, resident pages) of prefixes still on this device
        for sid, entry in self._sessions.items():  # LRU first
            if now - entry[1] < self.session_evict_min_idle_s or len(candidates) >= 1024:  # room for ones skipped
                break
            if sid in busy or not entry[0]:
                continue
            resident = checked[sid] = self._head_resident(entry)
            if resident:  # its device prefix is still on this P: its pool pages are the only backup, never removed
                heads.append((entry[0], resident))
                self._lookup_stats["evict_device_skip"] += 1
                continue
            candidates.append((sid, entry[0], now - entry[1]))
        recent = {}  # for the other P: sessions used in the last 300 s, or (up to 900 s) still on this P's device
        for sid in reversed(self._sessions):
            entry = self._sessions[sid]
            age = now - entry[1]
            if age >= 900.0:
                break
            if sid in checked:
                resident = checked[sid]
            else:
                resident = self._head_resident(entry)
                if resident:
                    heads.append((entry[0], resident))
            if age < 300.0 or resident:
                recent[sid] = [round(age, 1), resident]
        # Self-check pages known to be in the pool: the last found page of chains this stage looked up or re-leased in
        # the last 5 s (their read lease runs 10 s), not the oldest candidates the master may already have dropped.
        probe = []
        if self._evict_keys_ok is None:
            probe = [c.hashes[c.pages - 1] for c in self._lease_chains.values()
                     if c.pages and now - c.last < 5.0 and len(c.hashes) >= c.pages][:8]
            probe += list(reversed(self._known_present))
        if self._evict_executor is None:
            self._evict_executor = ThreadPoolExecutor(1, thread_name_prefix="linker-evict")
        self._evict_busy = True
        self._evict_executor.submit(self._session_evict, candidates, recent, heads, probe)

    def _head_resident(self, entry: list) -> int:
        """Scheduler thread: pages of the session's lookup-time device prefix still on this P's device, 0 if that node
        was evicted, removed or reused (a reused id carries another last hash; a split keeps the id on the deep part)."""
        head_len, node, last_hash = entry[2], entry[3], entry[4]
        if not head_len or node is None:
            return 0
        try:
            if self.cache.get_last_hash_value(node) != last_hash or self.cache.tree_core.is_full_device_evicted(node):
                return 0
        except Exception:  # noqa: BLE001  (node gone from the tree)
            return 0
        return head_len

    def _pool_fill(self) -> tuple[float, float, float]:
        """The master's fill (used / capacity) and bytes per object, from the admin summary (read only)."""
        with urllib.request.urlopen(f"{self.admin_url}/metrics/summary", timeout=2.0) as resp:
            text = resp.read().decode()
        unit = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}
        m = re.search(r"Mem Storage: ([\d.]+) (\w+) / ([\d.]+) (\w+)", text)
        keys = re.search(r"Keys: (\d+)", text)
        used, cap = float(m[1]) * unit[m[2]], float(m[3]) * unit[m[4]]
        return used / cap, cap, used / max(1, int(keys[1]) if keys else 1)

    def _write_stage_suffix(self) -> None:
        """Every stage, once: its store key suffix (exact, digest included) for PP0's key building."""
        self._suffix_written = True
        rank = self.pp_relay.rank if self.pp_relay is not None else 0
        path = os.path.join(self.session_share_dir, f"suffix-pp{rank}")
        try:
            os.makedirs(self.session_share_dir, exist_ok=True)
            with open(f"{path}.{os.getpid()}.tmp", "w") as f:
                f.write(self.cache_linker.storage.mla_suffix)
            os.replace(f"{path}.{os.getpid()}.tmp", path)
        except OSError as e:
            logger.warning("External linker session evict: cannot write %s (%s); session eviction stays off", path, e)

    def _all_stage_keys(self, page_hashes: list[str]) -> Optional[list[str]]:
        """Every PP stage's master object keys for these pages, stage-major: this stage's through the store's own mapping
        (the probe's code), each other stage's by swapping this stage's full suffix for that stage's exact one (from its
        suffix file). None while a stage's suffix is not known yet."""
        transfers = [
            c.build_external_linker_transfer(LinkerTransferPhase.LOOKUP, None, page_hashes) for c in self._components
        ]
        own = [k for keys, _ in self._probe_object_keys(transfers) for k in keys]
        rank = self.pp_relay.rank if self.pp_relay is not None else 0
        size = self.pp_relay.size if self.pp_relay is not None else 1
        tag = f"_{self.cache_linker.storage.mla_suffix}_"
        if any(tag not in k for k in own):
            raise ValueError(f"store key without the stage suffix {tag!r}: {own[:1]}")
        out = []
        for s in range(size):
            if s == rank:
                out += own
                continue
            sfx = self._stage_suffixes.get(s)
            if sfx is None:
                try:
                    with open(os.path.join(self.session_share_dir, f"suffix-pp{s}")) as f:
                        sfx = f.read().strip()
                except OSError:
                    return None
                if not sfx:
                    return None
                self._stage_suffixes[s] = sfx
            out += [head + f"_{sfx}_" + tail for head, _, tail in (k.rpartition(tag) for k in own)]
        return out

    def _share_sessions(self, recent: dict) -> dict:
        """Write this P's recent sessions (session -> [age s, device-resident prefix pages]; tmp + rename) and read the
        other P's files: session -> (youngest age elsewhere, largest resident prefix elsewhere). A file being replaced or
        gone is skipped this round; a stale one only reports old ages (its resident counts are dropped after 30 s)."""
        d, wall = self.session_share_dir, time.time()
        os.makedirs(d, exist_ok=True)
        own = os.path.join(d, self._share_name)
        with open(own + ".tmp", "w") as f:
            json.dump({"t": wall, "ages": recent}, f)
        os.replace(own + ".tmp", own)
        seen: dict = {}
        peers = 0
        for name in os.listdir(d):
            if name == self._share_name or not (name.startswith("sessions-") and name.endswith(".json")):
                continue
            try:
                with open(os.path.join(d, name)) as f:
                    other = json.load(f)
            except (OSError, ValueError):
                continue
            lag = max(0.0, wall - other["t"])
            peers += lag < 30.0
            for sid, (age, resident) in other["ages"].items():
                a, r = seen.get(sid, (float("inf"), 0))
                seen[sid] = (min(a, age + lag), max(r, resident if lag < 30.0 else 0))
        self._lookup_stats["evict_share_peers"] = peers  # other P whose file is fresh: 0 with two P = no shared view
        return seen

    @staticmethod
    def _shared_prefix(hashes: list, heads: list) -> int:
        """Leading pages of ``hashes`` that a device-resident session's prefix on this P also holds (a shared system
        prompt, say). Hashes are chained, so page k matches only if every page before it does: binary search per head."""
        keep = 0
        for head, n in heads:
            m = min(len(hashes), n, len(head))
            if m <= keep or head[0] != hashes[0]:
                continue
            if head[m - 1] == hashes[m - 1]:
                keep = m
                continue
            lo, hi = 1, m
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if head[mid - 1] == hashes[mid - 1]:
                    lo = mid
                else:
                    hi = mid
            keep = max(keep, lo)
        return keep

    def _keys_selfcheck(self, pages: list) -> Optional[bool]:
        """Pages known to be in the pool must show every stage's objects for at least one of them that this stage holds;
        a stage with none is a key mismatch (False). Unknown suffixes or nothing held: inconclusive (None), retried.
        Logs the probed keys and the raw admin reply once per run."""
        keys = self._all_stage_keys(pages) if pages else None
        if keys is None:
            return None
        data = self._admin_query(keys)
        present = {k for k, v in data.items() if isinstance(v, dict) and v.get("ok")}
        size = self.pp_relay.size if self.pp_relay is not None else 1
        per_stage = len(keys) // size
        if not self._evict_logged:
            self._evict_logged = True
            first = [keys[s * per_stage] for s in range(size)]
            logger.info(
                "External linker session evict self-check: probed keys %s; one page on every stage %s; present per stage "
                "%s of %d; admin reply %s", keys[:8], first,
                [sum(keys[s * per_stage + i] in present for i in range(per_stage)) for s in range(size)], per_stage,
                json.dumps({k: data.get(k) for k in keys[:8] + first})[:3000],
            )
        held = [i for i in range(per_stage) if keys[i] in present]
        if not held:
            return None
        return all(any(keys[s * per_stage + i] in present for i in held) for s in range(size))

    def _session_evict(self, candidates: list, recent: dict, heads: list, probe: list) -> None:
        """Evict worker: remove this P's share of the fill above the target, least recently used session over all P
        first, tail pages first, all stages' objects of a page per batch, never forced. A session whose objects come
        back leased is left alone. Never removed: a session whose device prefix is still on either P's device (the
        planner skips this P's; the share files carry the other P's), nor the leading pages a candidate shares with a
        device-resident session's prefix here (``heads``)."""
        t0 = time.perf_counter()
        ls = self._lookup_stats
        try:
            elsewhere = self._share_sessions(recent)
            n, kept = len(candidates), []
            for sid, hashes, age in candidates:
                other_age, other_resident = elsewhere.get(sid, (age, 0))
                if other_resident:
                    ls["evict_device_skip"] += 1  # its prefix is on the other P's device
                elif other_age >= self.session_evict_min_idle_s:
                    kept.append((sid, hashes, min(age, other_age)))
            ls["evict_shared_skip"] += n - len(kept)  # used on, or still held by, another P
            candidates = kept
            candidates.sort(key=lambda c: -c[2])
            if not candidates:
                return
            fill, cap, obj_bytes = self._pool_fill()
            ls["evict_fill_permille"] = int(1000 * fill)
            need = (fill - self.session_evict_target) * cap * self.session_evict_share
            if need <= 0:
                return
            store = self.cache_linker.storage.store
            if self._evict_keys_ok is None:
                # Known-present pages first; then each candidate's newest and middle page (written last, so the store's
                # own eviction takes them last; a chain's head page goes first).
                fallback = [p for _, h, _ in candidates[:8] for p in (h[-1], h[len(h) // 2])]
                self._evict_keys_ok = self._keys_selfcheck(list(dict.fromkeys(probe + fallback)))
                if self._evict_keys_ok is None:
                    ls["evict_selfcheck_none"] += 1  # not engaged yet: never silent
                    return
                if not self._evict_keys_ok:
                    ls["evict_selfcheck_fail"] += 1
                    logger.warning(
                        "External linker session evict: key self-check failed (a stage holds none of the pages PP0 "
                        "holds); session eviction off for this run"
                    )
                    return
            ls["evict_rounds"] += 1
            freed = 0.0
            for sid, hashes, _ in candidates:
                keep = self._shared_prefix(hashes, heads)
                ls["evict_kept_pages"] += keep
                left = len(hashes)
                while left > keep and freed < need:
                    lo = max(keep, left - 64)
                    keys = self._all_stage_keys(hashes[lo:left])
                    codes = store.batch_remove(keys, False)
                    ok = sum(1 for c in codes if c == 0)
                    refused = sum(1 for c in codes if c not in (0, _OBJECT_NOT_FOUND))  # already gone is fine
                    ls["evict_objects"] += ok
                    ls["evict_refused"] += refused
                    freed += ok * obj_bytes
                    if refused:
                        break  # leased (in use somewhere): leave the rest of this session
                    ls["evict_pages"] += left - lo
                    left = lo
                self._evict_done.append((sid, left))
                ls["evict_sessions"] += left == 0
                if freed >= need:
                    break
        except Exception:
            ls["evict_errors"] += 1
        finally:
            ls["evict_ms_max"] = max(ls["evict_ms_max"], int(1e3 * (time.perf_counter() - t0)))
            self._evict_busy = False

    def _refresh_leases(self) -> None:
        """Renew the store's read lease on this stage's queued chains, oldest first.

        Stage-local and metadata only: it reads no relay state and changes none, nor any
        vote, agreed length, target or verdict input; the RPC result only feeds counters.
        A chain is renewed while its request is queued and no load of it is in flight
        (a load's reads grant the lease themselves).
        """
        keep = []
        for future, pages in self._lease_refresh_futures:
            if not future.done():
                keep.append((future, pages))
                continue
            try:
                got = future.result()
            except Exception:
                continue
            if pages and (got[-1] if got else 0) < pages:
                self._lookup_stats["refresh_found_short"] += 1
        self._lease_refresh_futures = keep
        if self.session_evict_target and self.session_share_dir and not self._suffix_written:
            self._write_stage_suffix()
        if (self.session_evict_target and self.admin_url and self.session_share_dir
                and (self.pp_relay is None or not self.pp_relay.rank)):
            self._maybe_session_evict()
        if not self.lease_refresh_s or self._lookup_executor is None or not self._lease_chains:
            return
        now = time.monotonic()
        budget = self.lease_refresh_max_keys_s * min(now - self._lease_tick, self.lease_refresh_s)
        if now - self._lease_tick < min(1.0, self.lease_refresh_s):
            return
        self._lease_tick = now
        renew = None
        if self.lease_queue_budget_tokens:
            # F9: the queue's head in enqueue order (dict order), whole chains up to the budget.
            renew, used = set(), 0
            for request, chain in self._lease_chains.items():
                used += (len(chain.hashes) if chain.pages is None else chain.pages) * self.cache.page_size
                if used > self.lease_queue_budget_tokens:
                    break
                renew.add(request)
        for request, chain in sorted(self._lease_chains.items(), key=lambda item: item[1].last):
            if now - chain.last < self.lease_refresh_s:
                break
            if request in self.prefetches:
                continue
            if renew is not None and request not in renew:
                self._lookup_stats["lease_budget_skip"] += 1
                continue
            if chain.pages is None:
                lookup = self.lookups.get(request)
                if lookup is not None and lookup.future.done() and not lookup.future.exception():
                    result = lookup.future.result()
                    chain.pages = result[-1] if result else 0
            hashes = chain.hashes[: chain.pages] if chain.pages else chain.hashes
            if chain.pages == 0 or (self.lease_refresh_max_keys_s and len(hashes) > budget):
                continue
            transfers = [
                c.build_external_linker_transfer(LinkerTransferPhase.LOOKUP, None, hashes)
                for c in self._components
            ]
            if any(t is None for t in transfers):
                continue
            budget -= len(hashes)
            chain.last = now
            rpc = getattr(self.cache_linker, "lease_refresh", None) or self.cache_linker.lookup
            self._lease_refresh_futures.append(
                (self._lookup_executor.submit(rpc, "lease:" + chain.rid, transfers), chain.pages)
            )
            self._lookup_stats["refresh_rpcs"] += 1
            self._lookup_stats["refresh_keys"] += len(hashes)

    def commit_load_wait(self, req, charge) -> int:
        """Admission skipped ``req`` while its read runs (one batch ahead): the tokens
        ``charge(target)`` it will need once the read lands, which the adder keeps free
        for it. Every stage computes this from relayed state (phase, stop, target) and
        fails closed unless it equals PP0's value: a divergence is a loud error, not
        different batches."""
        handle = req.cache_request_handle
        if self.pp_relay is not None:
            entry = self.pp_relay.entries.get(handle)
            loading = entry is not None and entry.phase == "load" and not entry.stopped
        else:
            loading = handle in self.prefetches
        tokens = 0
        if loading and handle in self.match_targets:
            tokens = charge(self.match_targets[handle])
            self._late_beats[handle] = self._late_beats.get(handle, 0) + 1
        if self.pp_relay is not None:
            pp0 = self._pp_verdict(tokens)
            assert pp0 == tokens, ("PP linker committed-load mismatch", tokens, pp0)
        if tokens:
            self._lookup_stats["load_waits"] += 1
        return tokens

    def release_out_of_window_holds(self) -> None:
        """Loop hook, after the plan (one batch ahead): a hold whose request the next
        batch will not admit is released, so held prefixes never exceed one batch. PP0
        decides; every stage unholds in action order, before its admission. The prefix
        stays matchable as evictable cache and the request is planned again."""
        if not self.one_batch_ahead or self.near_term_window is None:
            return
        out = [h for h in self.holds if h not in self.near_term_window]
        if self.pp_relay is None:
            for handle in out:
                self._unhold(handle)
            self._lookup_stats["window_unhold"] += len(out)
            return
        if self.pp_relay.rank:
            return
        for handle in out:
            entry = self.pp_relay.entries.get(handle)
            if entry is not None and entry.phase == "ready":
                self.pp_relay._issue(handle, entry.cycle, "unhold", 0)
                self._lookup_stats["window_unhold"] += 1

    def lookup_overdue(self, request: CacheRequestHandle) -> bool:
        """PP0: this lookup's RPC has not returned within its timeout (F6c7c head-room release)."""
        lookup = self.lookups.get(request)
        return lookup is not None and not lookup.future.done() and (
            time.monotonic() - lookup.issued_at > lookup.timeout_s
        )

    def stale_hit(self, request: CacheRequestHandle) -> bool:
        """PP0: this held hit's lookup predates the store's lease, so its agreed length
        may overstate what is left; the relay parks it for a fresh lookup instead."""
        lookup = self.lookups.get(request)
        return bool(self.stale_hit_s) and lookup is not None and (
            time.monotonic() - lookup.issued_at > self.stale_hit_s
        )

    def _timed_lookup(self, kind, rid, lookups):
        """The store lookup; with ``kind``, its time joins that kind's RPC phase (any thread)."""
        if kind is None:
            return self.cache_linker.lookup(rid, lookups)
        t0 = time.perf_counter()
        try:
            return self.cache_linker.lookup(rid, lookups)
        finally:
            self._phase_s[kind]["rpc"] += time.perf_counter() - t0

    def _start_prefetch_load(
        self,
        request: CacheRequestHandle,
        key: RadixKey,
        hashes: list[str],
        start: int,
        anchor_node_id: NodeId,
        anchor_lock: DecLockRefParams,
        restorable: list[int],
        timeout_s: float,
        ahead: Optional[int] = None,
    ) -> None:
        """Agree on the restorable prefix across ranks and queue its load; ``ahead``
        is PP0's work ahead of the request (every stage reserves the same)."""
        page = self.cache.page_size
        pages = self._sync_restorable_prefix(
            restorable, num_pages=len(hashes), device_hit_pages=0
        )
        self._note(request, "zero" if not pages else "found", pages * page)
        if not pages:
            self._lookup_stats["zero_restorable"] += 1
            self.cache.dec_lock_ref(anchor_node_id, anchor_lock)
            return
        if self.session_evict_target and self._evict_keys_ok is None:
            self._known_present.append(hashes[pages - 1])  # every stage just found it: present on all of them
        end = start + pages * page
        if request in self.match_targets:
            self.match_targets[request] = end
        self._room_refused.discard(request)
        cap = self._load_cap()
        if cap:
            # In-flight private slots (plus holds, under a shared cap) stay under the
            # cap, except that one load may always run; a refused load keeps its
            # target at `end`, so the admission re-query retries it.
            occupied = self.prefetch_size(PoolName.KV)
            if not self.load_cap_tokens:
                occupied += self.held_tokens
            if occupied and occupied + pages * page > cap:
                self._lookup_stats["load_capped"] += 1
                self._note(request, "capped")
                self.cache.dec_lock_ref(anchor_node_id, anchor_lock)
                return
            self._occupied_peak = max(self._occupied_peak, occupied + pages * page)
        need = self.load_free_reserve_tokens
        if request == self.near_term_head and self.near_term_head_reserve:
            # The head runs next: one chunk of headroom is enough for it.
            need = min(need, self.near_term_head_reserve)
        # Never take the tail an in-flight chunked request was promised at admission,
        # nor the work of requests the window admits before this one.
        if ahead is None:
            ahead = self.near_term_window.get(request, 0) if self.near_term_window else 0
            if not (
                (self.head_load_skips_allowance and request == self.near_term_head)
                or (self.window_load_skips_allowance and self.in_near_term_window(request))
            ):
                ahead += self.anchor_allowance()
        need = max(need, self.chunk_tail_tokens + ahead)
        if need:
            free = (
                self.cache.token_to_kv_pool_allocator.available_size()
                + self.cache.evictable_size()
            )
            if not load_fits(free, pages * page, need):
                # Same outcome as the cap: target kept, admission re-queries later.
                self._lookup_stats["load_reserve"] += 1
                self._note(request, "reserve")
                if self.wait_fit_s:
                    self._room_refused.add(request)
                if self.pp_relay is None or not self.pp_relay.rank:
                    # Which term refused it: ahead = window work ahead + anchor allowance (PP0's value).
                    window_ahead = self.near_term_window.get(request, 0) if self.near_term_window else 0
                    anchored = (
                        sum(lk.start for r, lk in self.lookups.items() if not self.in_near_term_window(r))
                        if self.near_term_window is not None
                        else 0
                    )
                    available = self.cache.token_to_kv_pool_allocator.available_size()
                    logger.info(
                        "External linker load refused: rid=%s load=%d available=%d evictable=%d "
                        "chunk_tail=%d work_ahead=%d allowance=%d anchored=%d reserve=%d need=%d",
                        request.rid, pages * page, available, free - available, self.chunk_tail_tokens,
                        window_ahead, ahead - window_ahead, anchored, self.load_free_reserve_tokens, need,
                    )
                self.cache.dec_lock_ref(anchor_node_id, anchor_lock)
                return
        state = _ExternalCachePrefetch(
            key[:end],
            anchor_node_id,
            anchor_lock,
            [],
            timeout_s=timeout_s,
        )
        self.prefetches[request] = state
        reads = self._reads.setdefault(request, [0, 0, 0, 0])
        reads[0] += 1
        reads[1] += pages * page
        if reads[0] == 1:
            reads[2], reads[3] = pages * page, end
        try:
            for component in self._components:
                transfer = component.build_external_linker_transfer(
                    LinkerTransferPhase.LOAD, None, hashes[:pages]
                )
                if transfer is None:
                    break
                state.component_transfers.append((component, transfer))
            if len(state.component_transfers) == len(self._components):
                state.future = self.cache_linker.prefetch(
                    [t for _, t in state.component_transfers]
                )
        except Exception:
            logger.warning("External linker prefetch submission failed", exc_info=True)
        accepted = torch.tensor([state.future is not None], dtype=torch.int)
        self.cache._all_reduce_attn_groups(accepted, torch.distributed.ReduceOp.MIN)
        if not accepted.item():
            self._cancel_prefetch(request)
            # A lookup drain is inside check_prefetch_events, which settles the
            # cancelled read right after it; re-entering would rerun the drain.
            if not self._draining_lookups:
                self.check_prefetch_events(drain_lookups=False)

    def _note_enqueue(
        self, request: CacheRequestHandle, isl: int, device_prefix: int, full_device: bool
    ) -> None:
        record = self._request_records.get(request)
        if record is None:
            if len(self._request_records) >= 20000:
                self._request_records.pop(next(iter(self._request_records)))
            record = self._request_records[request] = dict(
                isl=isl, dev=device_prefix, t=time.perf_counter(), found=0, rq=0
            )
        else:
            record["rq"] += 1
        record["outcome"] = "full_device" if full_device else "lookup"

    def _note(self, request: CacheRequestHandle, outcome: str, found: Optional[int] = None) -> None:
        record = self._request_records.get(request) if self.request_log else None
        if record is not None:
            record["outcome"] = outcome
            if found is not None:
                record["found"] = found

    def _load_cap(self) -> int:
        """Tokens loads may hold in flight: their own cap, else the hold cap they share."""
        return self.load_cap_tokens or self.hold_cap_tokens

    def set_near_term_window(
        self, handles, head: Optional[CacheRequestHandle] = None, head_reserve: int = 0,
        head_work: int = 0,
    ) -> None:
        """Requests the next prefill batches will admit (handle -> work ahead of it);
        only they start loads. ``head_work``: the head's own work (F6c7c's anchor floor)."""
        self.near_term_window = handles if isinstance(handles, dict) else dict.fromkeys(handles, 0)
        self.near_term_head, self.near_term_head_reserve = head, head_reserve
        self.near_term_head_work = head_work

    def _park_lookup(self, request: CacheRequestHandle, pages: int) -> None:
        """Release an out-of-window hit's anchor; admission re-queries it in the window."""
        lookup = self.lookups.pop(request, None)
        if lookup is None:
            return
        lookup.future.cancel()
        self.cache.dec_lock_ref(lookup.anchor_node_id, lookup.anchor_lock)
        tokens = pages * self.cache.page_size
        if not tokens:
            return
        if self.requery_on_device_loss:
            self.match_targets[request] = lookup.start + tokens
        self.near_term_parked[request] = tokens
        self._lookup_stats["parked"] += 1

    def _unhold(self, request: CacheRequestHandle) -> None:
        """Drop a published hold without admission; its prefix stays matchable until
        evicted, and admission re-queries it in the window (target kept)."""
        hold = self.holds.pop(request, None)
        if hold is None:
            return
        self.cache.dec_lock_ref(hold.node_id, hold.lock)
        self.held_tokens -= hold.tokens
        self._hold_stats["released"] += 1
        self.near_term_parked[request] = hold.tokens

    def anchor_allowance(self) -> int:
        """Evictable tokens out-of-window lookups may still pin (cap - anchored)."""
        if self.near_term_window is None:
            return 0
        anchored = sum(
            lk.start for r, lk in self.lookups.items() if not self.in_near_term_window(r)
        )
        return max(self.near_term_anchor_cap - anchored, 0)

    def note_head_no_token(self, request: CacheRequestHandle) -> None:
        """Scheduler: admission returned NO_TOKEN for the queue head with nothing admitted."""
        self._no_token_head = request

    def _head_waits_on_own_io(self, head: CacheRequestHandle) -> bool:
        if self.pp_relay is not None:
            entry = self.pp_relay.entries.get(head)
            return entry is not None and entry.phase in ("lookup", "load") and not entry.stopped
        return head in self.prefetches or head in self.lookups

    def check_stall(self, head: Optional[CacheRequestHandle]) -> None:
        """Scheduler, each loop: fire the backstop when the queue head keeps getting
        NO_TOKEN with nothing admitted, waits on no I/O of its own, and has done so
        for near_term_stall_s x 2^backoff."""
        now = time.monotonic()
        blocked, self._no_token_head = self._no_token_head == head, None
        if head != self._stall_head:
            self._stall_head, self._stall_since, self._stall_backoff = head, now, 0
            return
        if head is None or not blocked or self._head_waits_on_own_io(head):
            self._stall_since = now
            return
        delay = min(self.near_term_stall_s * (1 << self._stall_backoff), self.near_term_stall_max_s)
        if now - self._stall_since < delay:
            return
        self._stall_since = now
        self._stall_backoff = min(self._stall_backoff + 1, 6)
        if self.pp_relay is not None:
            if self.cache.pp_rank:
                return  # PP0 decides and relays the release to every stage.
            self.pp_relay.stall_release(head)
        else:
            for request, lookup in list(self.lookups.items()):
                if lookup.future.done() and not self.in_near_term_window(request):
                    try:
                        pages = min(lookup.future.result(), default=0)
                    except Exception:
                        pages = 0
                    self._park_lookup(request, pages)
            for request in [h for h in self.holds if h != head]:
                self._unhold(request)
        self._lookup_stats["stall_release"] += 1
        logger.warning(
            "External linker near-term stall release: head %s blocked %.1f s", head.rid,
            self.near_term_stall_s,
        )

    def parked_out_of_window(self, request: CacheRequestHandle) -> bool:
        """True while a parked hit waits outside the window. In the window the mark
        is dropped: admission re-queries it (or, if declined, recomputes it), and
        the planner must stop counting a restore that may not happen."""
        if request not in self.near_term_parked:
            return False
        if self.in_near_term_window(request):
            del self.near_term_parked[request]
            return False
        return True

    def in_near_term_window(self, request: CacheRequestHandle) -> bool:
        return self.near_term_window is None or request in self.near_term_window

    def restorable_tokens(self, request: CacheRequestHandle) -> int:
        """Pool tokens ``request`` will restore: its parked or agreed hit; 0 while
        unknown and once published (admission then counts it in prefix_len)."""
        if request in self.near_term_parked:
            return self.near_term_parked[request]
        if self.pp_relay is not None:
            entry = self.pp_relay.entries.get(request)
            if entry is None or entry.phase == "ready":
                return 0
            return entry.agreed * self.cache.page_size
        lookup = self.lookups.get(request)
        if lookup is None or not lookup.future.done() or lookup.future.cancelled():
            return 0
        if lookup.future.exception() is not None:
            return 0
        return min(lookup.future.result(), default=0) * self.cache.page_size

    def has_pending_prefetch(self) -> bool:
        return bool(self.prefetches or self.lookups)

    def check_prefetch_progress(self, request: CacheRequestHandle) -> bool:
        if self.pp_relay is not None:
            return self.pp_relay.progress(request, self._local_check_prefetch_progress)
        return self._local_check_prefetch_progress(request)

    def _pp_verdict(self, value: int) -> int:
        verdict = torch.tensor(value, dtype=torch.int)
        self.cache._exchange_pp0_verdict(verdict)
        return int(verdict.item())

    def _local_check_prefetch_progress(self, request: CacheRequestHandle) -> bool:
        if self.parked_out_of_window(request):
            return False  # Its re-query starts once the window reaches it.
        if request in self.lookups:
            return self._check_lookup_progress(request)
        state = self.prefetches.get(request)
        if state is None or state.discard:
            return True
        policy = self.cache.prefetch_stop_policy
        if policy == "wait_complete":
            return False
        if policy == "timeout":
            now = time.perf_counter()
            expired = torch.tensor(
                state.wait_started is not None
                and now - state.wait_started > state.timeout_s,
                dtype=torch.int,
            )
            self.cache._all_reduce_attn_groups(expired, torch.distributed.ReduceOp.MAX)
            if not expired.item():
                if state.wait_started is None:
                    state.wait_started = now
                return False
        # Consume any fully completed read before stopping. Running reads keep
        # their private slots until every rank has stopped writing to them.
        # Admission polls this: start no new load here.
        self.check_prefetch_events(drain_lookups=False)
        self._cancel_prefetch(request)
        return True

    def _check_lookup_progress(self, request: CacheRequestHandle) -> bool:
        """Admission poll while the request's lookup runs, under the stop
        policy of a running load."""
        lookup = self.lookups[request]
        in_window = self.in_near_term_window(request)
        if not in_window and lookup.future.done():
            # Held by the near-term window: no stop clock until it enters the window.
            lookup.wait_started = None
            return False
        limit = lookup.timeout_s if in_window else max(lookup.timeout_s, self.near_term_hung_lookup_s)
        policy = self.cache.prefetch_stop_policy
        if policy == "wait_complete":
            return False
        if policy == "timeout":
            now = time.perf_counter()
            expired = torch.tensor(
                lookup.wait_started is not None
                and now - lookup.wait_started > limit,
                dtype=torch.int,
            )
            self.cache._all_reduce_attn_groups(expired, torch.distributed.ReduceOp.MAX)
            if not expired.item():
                if lookup.wait_started is None:
                    lookup.wait_started = now
                return False
        self._lookup_stats["lookup_timeout"] += 1
        self._note(request, "lookup_timeout")
        self._cancel_lookup(request)
        return True

    def requery_after_device_loss(self, req: Req) -> bool:
        if self.pp_relay is not None:
            return self.pp_relay.requery(req, self._local_requery_after_device_loss)
        return self._local_requery_after_device_loss(req)

    def _local_requery_after_device_loss(self, req: Req) -> bool:
        """Look up again the prefix ``req`` lost on device since its last lookup.

        True while the re-query waits for the next event-loop boundary, where
        its lookup starts; the request then waits on that lookup and its load.
        """
        request = req.cache_request_handle
        target = self.match_targets.get(request)
        if target is not None and req.prefix_len < target and self.pp_relay is not None:
            entry = self.pp_relay.entries.get(request)
            if entry is not None and entry.phase != "ready":
                # Relayed state only: a load refused or a lookup skipped on one stage
                # must not change the answer. A stopped cycle settles by PP0's finish /
                # load 0, but a read that never completes holds it: cap it at 8x, then
                # admit cold on every stage (the late finish aborts: no hold, no records).
                waits = self._settle_waits.get(request, 0)
                cap = 8 * self.near_term_settle_wait_max if entry.stopped else self.near_term_settle_wait_max
                if waits < cap:
                    self._settle_waits[request] = waits + 1
                    return REQUERY_SETTLING  # re-query once its cycle has settled
                if entry.stopped and waits == cap:
                    self._settle_waits[request] = waits + 1  # count each request once
                    self._lookup_stats["settle_cap_hit"] += 1
                return False
        if (
            target is None
            or req.prefix_len >= target
            or request in self.prefetches
            or request in self.lookups
            or request in self.holds
        ):
            self.near_term_parked.pop(request, None)
            return False
        if request in self._pending_requeries:
            return True
        if self.requery_over_attempts:
            # F6c7e: past the attempt cap only a load refused for room keeps waiting. PP0's
            # clock and marks decide; stages > 0 follow its relayed verdict and read neither.
            if self.pp_relay is None or not self.pp_relay.rank:
                t0 = self._wait_fit_enq.get(request)
                if request not in self._room_refused or t0 is None:
                    return False
                if time.perf_counter() - t0 > self.wait_fit_s:
                    self._room_refused.discard(request)
                    self._lookup_stats["wait_fit_cap"] += 1
                    self._note(request, "wait_fit_cap")
                    return False
            self._lookup_stats["wait_fit_requery"] += 1
        self._lookup_stats["requeried"] += 1
        self._settle_waits.pop(request, None)
        self._pending_requeries[request] = req
        if self.log_lookup_stats:
            self._requery_started[request] = time.perf_counter()
        return True

    def _start_pending_requeries(self) -> None:
        """At the event-loop boundary: look up again the requests admission deferred."""
        pending, self._pending_requeries = self._pending_requeries, {}
        self._lookup_kind = "rq"
        try:
            for request, req in pending.items():
                self.prefetch(req)
                if request not in self.prefetches and request not in self.lookups:
                    started = self._requery_started.pop(request, None)
                    if started is not None:
                        self._requery_records.append(
                            (request.rid, time.perf_counter() - started, "none")
                        )
        finally:
            self._lookup_kind = "enq"

    def _log_lookup_stats(self) -> None:
        now = time.monotonic()
        if not self.log_lookup_stats or now < self._lookup_next_log:
            return
        self._lookup_next_log = now + 60.0
        waits = sorted(self._requery_waits)
        self._requery_waits.clear()
        p50, p90 = (
            (waits[len(waits) // 2], waits[len(waits) * 9 // 10])
            if waits
            else (0.0, 0.0)
        )
        phases = []
        for kind, phase in self._phase_s.items():
            n = phase["n"]
            phases.append(
                f"{kind} n={int(n)}"
                + "".join(
                    f" {name} {1e3 * phase[name] / n:.1f}"
                    for name in ("match", "keys", "rpc", "load")
                    if n
                )
            )
            self._phase_s[kind] = dict.fromkeys(phase, 0.0)
        logger.info(
            "External linker lookups: %s; re-query waits since last log: %d, "
            "p50 %.3f s, p90 %.3f s; ms per lookup: %s; loads plus holds peak %d "
            "of cap %d tokens",
            self._lookup_stats,
            len(waits),
            p50,
            p90,
            "; ".join(phases),
            self._occupied_peak,
            self._load_cap(),
        )
        self._occupied_peak = 0
        if self.near_term_batches:
            deferred = sum(
                1
                for request, lookup in self.lookups.items()
                if lookup.future.done() and not self.in_near_term_window(request)
            )
            logger.info(
                "External linker device tokens: available %d, evictable %d, protected %d; "
                "deferred lookups holding anchors %d",
                self.cache.token_to_kv_pool_allocator.available_size(),
                self.cache.evictable_size(),
                self.cache.protected_size(),
                deferred,
            )
        if self._requery_records:
            logger.info(
                "External linker re-queried rids (rid:wait_s:outcome): %s",
                " ".join(f"{rid}:{w:.3f}:{o}" for rid, w, o in self._requery_records),
            )
            self._requery_records.clear()

    def _cancel_lookup(self, request: CacheRequestHandle) -> None:
        lookup = self.lookups.pop(request, None)
        if lookup is not None:
            # A lookup reads store metadata only, so its result can be dropped.
            lookup.future.cancel()
            self.cache.dec_lock_ref(lookup.anchor_node_id, lookup.anchor_lock)

    def _drain_lookups(self) -> None:
        """Start the loads of lookups every rank has finished, in submission order."""
        if not self.lookups:
            return
        pending = list(self.lookups.items())
        done = torch.tensor([lk.future.done() for _, lk in pending], dtype=torch.int)
        self.cache._all_reduce_attn_groups(done, torch.distributed.ReduceOp.MIN)
        self._draining_lookups = True
        try:
            anchored = sum(
                lk.start for r, lk in self.lookups.items() if not self.in_near_term_window(r)
            )
            held = []
            for (request, _), finished in zip(pending, done.tolist()):
                if finished and not self.in_near_term_window(request):
                    held.append(request)
                    continue  # Keep the finished lookup; its load waits for the window.
                lookup = self.lookups.pop(request, None) if finished else None
                if lookup is None:
                    continue
                try:
                    restorable = lookup.future.result()
                except Exception:
                    logger.warning(
                        "External linker prefetch lookup failed", exc_info=True
                    )
                    restorable = []
                t0 = time.perf_counter() if self.log_lookup_stats else 0.0
                self._start_prefetch_load(
                    request,
                    lookup.key,
                    lookup.hashes,
                    lookup.start,
                    lookup.anchor_node_id,
                    lookup.anchor_lock,
                    restorable,
                    lookup.timeout_s,
                )
                if self.log_lookup_stats:
                    self._phase_s[lookup.kind]["load"] += time.perf_counter() - t0
            while held and anchored > self.near_term_anchor_cap:
                request = held.pop()  # newest first; the oldest are nearest the window
                lookup = self.lookups[request]
                anchored -= lookup.start
                try:
                    pages = min(lookup.future.result(), default=0)
                except Exception:
                    pages = 0
                self._park_lookup(request, pages)
        finally:
            self._draining_lookups = False

    def _cancel_prefetch(self, request: CacheRequestHandle) -> None:
        self._cancel_lookup(request)
        state = self.prefetches.get(request)
        if state is not None:
            state.discard = True
            if state.future is not None:
                state.future.cancel()

    def prefetch_size(self, pool: PoolName) -> int:
        """Device slots still owned by prefetch, including discarded writes."""
        return sum(
            len(transfer.device_indices)
            for state in self.prefetches.values()
            for _, transfer in state.component_transfers
            if transfer.name == pool and transfer.device_indices is not None
        )

    def check_prefetch_events(self, drain_lookups: bool = True) -> None:
        if drain_lookups:
            self._refresh_leases()
        if self.pp_relay is not None:
            if drain_lookups:
                self._start_pending_requeries()
                self.pp_relay.poll()
            return
        if drain_lookups:
            self._start_pending_requeries()
            self._drain_lookups()
        if not self.prefetches:
            return
        pending = list(self.prefetches.items())
        progress = []
        for _, state in pending:
            future = state.future
            done = future is None or future.done()
            landed = False
            try:
                if done and future is not None and not future.cancelled():
                    landed = future.result() is True
            except Exception:
                logger.warning("External linker prefetch failed", exc_info=True)
            progress.append([done, landed and not state.discard])
        # Keep the same operation order on every rank, including discarded
        # reads. Local cancellation must not change collective participation.
        verdicts = torch.tensor(progress, dtype=torch.int)
        self.cache._all_reduce_attn_groups(verdicts, torch.distributed.ReduceOp.MIN)
        for (request, _), (done, landed) in zip(pending, verdicts.tolist()):
            if done:
                self._finish_prefetch(request, publish=bool(landed))

    def _finish_prefetch(self, request: CacheRequestHandle, *, publish: bool) -> None:
        state = self.prefetches.pop(request, None)
        if state is None:
            return
        self._note(request, "loaded" if publish else "load_dropped")
        started = self._requery_started.pop(request, None)
        if started is not None:
            waited = time.perf_counter() - started
            self._requery_waits.append(waited)
            self._requery_records.append((request.rid, waited, "loaded" if publish else "aborted"))
        # Event polling or reset/close has already quiesced every device writer.
        if state.future is not None:
            assert state.future.done()
        end = len(state.key)
        try:
            if not publish:
                self._update_load(
                    ExternalLinkerLoadPhase.ABORT, None, state.component_transfers, end
                )
                return
            by_pool = {t.name: t for _, t in state.component_transfers}
            storage = self.cache.kv_storage()
            full = by_pool[PoolName.KV].device_indices
            start = end - len(full)
            evicted = {}
            if ComponentType.SWA in self.cache.components:
                swa = by_pool.get(PoolName.SWA)
                if swa is not None:
                    storage.bind_window(
                        storage.tree_sub_pool(ComponentType.SWA.name),
                        full[-len(swa.device_indices) :],
                        swa.device_indices,
                    )
                page = self.cache.page_size
                window = (self.cache.sliding_window_size + page - 1) // page * page
                evicted[ComponentType.SWA] = (
                    end if self._skip_swa else max(0, end - window)
                )
            prefix = self.cache.path_device_indices(state.anchor_node_id)
            assert len(prefix) == start
            # The normal insert protects the borrowed prefix and frees duplicate
            # fetched slots. Avoid writing the just-fetched data back to storage.
            mamba = by_pool.get(PoolName.MAMBA)
            if mamba is not None:
                sub_pool = storage.tree_sub_pool(ComponentType.MAMBA.name)
                # A restored checkpoint has no pending ReplaySSM updates.
                storage.restart_replay(sub_pool, mamba.device_indices)
            inserted = self.cache.insert(
                InsertParams(
                    key=state.key,
                    value=torch.cat([prefix, full]),
                    mamba_value=mamba.device_indices if mamba is not None else None,
                    prev_prefix_len=start,
                    inserted_len=end,
                    component_evicted_seqlens=evicted,
                )
            )
            if mamba is not None and inserted.mamba_exist:
                storage.free_state_slots(sub_pool, mamba.device_indices)
            if not state.discard:  # a cancelled request no longer waits for this prefix
                if self.hold_cap_tokens:
                    self._hold(request, state.key, end, len(full))
                self.cache.prefetch_loaded_tokens_by_reqid[request] = len(full)
                self.cache.prefetch_loaded_storage_start_by_reqid[request] = start
        finally:
            self.cache.dec_lock_ref(state.anchor_node_id, state.anchor_lock)

    def _hold(self, request, key: RadixKey, end: int, tokens: int) -> None:
        """Lock a just-published prefix for its waiting request, under the cap."""
        stats = self._hold_stats
        resident = self.cache.match_prefix(MatchPrefixParams(key=key))
        if resident.device_prefix_len < end:
            stats["not_resident"] += 1
            self.unheld[request] = end
        elif self.held_tokens + tokens > self.hold_cap_tokens:
            stats["over_cap"] += 1
            self.unheld[request] = end
        else:
            node = resident.last_device_node
            lock = self.cache.inc_lock_ref(node).to_dec_params()
            self.holds[request] = _PrefetchHold(node, lock, tokens)
            self.held_tokens += tokens
            stats["held"] += 1
        self._log_holds()

    def release_hold(
        self, request: CacheRequestHandle, *, admitted: bool, prefix_len: int = 0
    ) -> None:
        stats = self._hold_stats
        if self.request_log:
            record = self._request_records.pop(request, None)
            if record is not None and admitted:
                logger.info(
                    "External linker request: rid=%s isl=%d dev_at_enq=%d pool_found=%d "
                    "outcome=%s requeries=%d cached_at_admit=%d wait_ms=%.0f",
                    request.rid, record["isl"], record["dev"], record["found"],
                    record["outcome"], record["rq"], prefix_len,
                    1e3 * (time.perf_counter() - record["t"]),
                )
        self.near_term_parked.pop(request, None)
        self._settle_waits.pop(request, None)
        sid = self._request_session.pop(request, None)
        if admitted and sid in self._sessions:
            self._sessions[sid][1] = time.monotonic()
            self._sessions.move_to_end(sid)
        self._wait_fit_enq.pop(request, None)
        self._room_refused.discard(request)
        self._lease_chains.pop(request, None)
        reads = self._reads.pop(request, None)
        beats = self._late_beats.pop(request, 0)
        if admitted and reads:
            ls = self._lookup_stats
            big = reads[3] > self.big_prefix_tokens
            extra = max(beats - 1, 0)  # the first skip is the beat the read was planned for
            ls["admitted_reads"] += 1
            ls["reads_sum"] += reads[0]
            ls["reads_2"] += reads[0] == 2
            ls["reads_3plus"] += reads[0] >= 3
            ls["reads_max"] = max(ls["reads_max"], reads[0])
            ls["reread_tokens"] += reads[1] - reads[2]
            ls["big_admits"] += big
            ls["extra_beats"] += extra
            ls["late_admits"] += extra > 0
            if big:
                ls["extra_beats_big"] += extra
                ls["late_admits_big"] += extra > 0
        self.match_targets.pop(request, None)
        end = self.unheld.pop(request, None)
        if end is not None and admitted:
            # An unheld prefix the request no longer matches in full was evicted.
            if prefix_len < end:
                stats["unheld_evicted"] += 1
                stats["unheld_evicted_tokens"] += end - prefix_len
            else:
                stats["unheld_kept"] += 1
        hold = self.holds.pop(request, None)
        if hold is not None:
            self.cache.dec_lock_ref(hold.node_id, hold.lock)
            self.held_tokens -= hold.tokens
            stats["admitted" if admitted else "released"] += 1
        if self.pp_relay is not None:
            self.pp_relay.release(request)
        if admitted:
            self._log_holds()

    def _log_holds(self) -> None:
        now = time.monotonic()
        if now < self._hold_next_log:
            return
        self._hold_next_log = now + 60.0
        logger.info(
            "External linker prefetch holds: %d requests hold %d of %d tokens; %s",
            len(self.holds),
            self.held_tokens,
            self.hold_cap_tokens,
            self._hold_stats,
        )

    def _sync_restorable_prefix(
        self, restorable: list[int], *, num_pages: int, device_hit_pages: int
    ) -> int:
        """Intersect the per-rank sets of restorable prefix lengths and return the
        longest one, or 0 when the ranks share none beyond the device prefix.

        A rank's set is sparse, so reducing per-rank maxima could land on a
        length that only some ranks can restore. On a 0/1 mask MIN is AND, which
        makes the reduction an intersection.
        """
        mask = torch.zeros(num_pages + 1, dtype=torch.int)
        for pages in restorable:
            if device_hit_pages < pages <= num_pages:
                mask[pages] = 1
        self.cache._all_reduce_attn_groups(mask, torch.distributed.ReduceOp.MIN)
        common = mask.nonzero()
        if common.numel() == 0:
            return 0
        return int(common[-1].item())

    def _tail_hashes(
        self, key: RadixKey, result: MatchResult, device_hit_len: int
    ) -> list[str]:
        """Per-page storage hashes for the device-uncached tail of the prefix."""
        last_hash = None
        if device_hit_len > 0:
            last_hash = self.cache.get_last_hash_value(result.last_device_node)
            if last_hash is None:
                # Without the anchor the tail would hash as if it started at the
                # sequence head, yielding keys that can never match.
                return []
        page = self.cache.page_size
        tail_len = (len(key) - device_hit_len) // page * page
        if tail_len == 0:
            return []
        return get_storage_hash_str(
            key[device_hit_len : device_hit_len + tail_len],
            last_hash,
            page_size=page,
        )

    # ---- init_load_back: remote -> request-private device slots ----

    def load_back(self, req: Req) -> tuple[int, NodeId]:
        """Queue the external hit into device slots owned by ``req``.

        The loaded tail extends the request's prefix but stays out of the
        tree (``req.last_node`` is unchanged): the request inserts it like
        computed KV after its forward, once :meth:`finish_loads` has confirmed
        the load.
        """
        cache = self.cache
        hit = self.hit_markers.pop(req.rid, None)
        if hit is None:
            return 0, req.last_node

        device_hit_len = hit.device_hit_len
        tail_hashes = hit.tail_hashes
        prefix_len = device_hit_len + len(tail_hashes) * cache.page_size

        # Build per-component linker transfers.
        component_transfers: list[tuple[TreeComponent, PoolTransfer]] = []
        for component in self._components:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOAD, None, tail_hashes
            )
            if transfer is None:
                self._update_load(
                    ExternalLinkerLoadPhase.ABORT,
                    req,
                    component_transfers,
                    prefix_len,
                )
                return 0, req.last_node
            component_transfers.append((component, transfer))

        full_transfer = component_transfers[0][1]
        assert full_transfer.name == PoolName.KV
        # Queue before PREPARE touches the request, so a refused load only has
        # the allocations to undo before falling back to recompute. The result
        # depends on the transfers alone, so every rank takes the same branch.
        try:
            queued = self.cache_linker.load(
                req.rid, [transfer for _, transfer in component_transfers]
            )
        except BaseException:
            self._update_load(
                ExternalLinkerLoadPhase.ABORT, req, component_transfers, prefix_len
            )
            raise
        if not queued:
            self._update_load(
                ExternalLinkerLoadPhase.ABORT, req, component_transfers, prefix_len
            )
            return 0, req.last_node
        self._update_load(
            ExternalLinkerLoadPhase.PREPARE,
            req,
            component_transfers,
            prefix_len,
        )

        # Components omitted from the linker do not run their PREPARE hook.
        # Keep a non-restorable SWA range as tombstones instead of rebuilding
        # it from an uninitialized FULL-to-SWA mapping when the tail is inserted.
        if self._skip_swa:
            req.kv.set_evicted_seqlen(
                ComponentType.SWA,
                max(req.kv.get_evicted_seqlen(ComponentType.SWA), prefix_len),
            )

        # match placed the SWA branch point before this tail was known; a point
        # the tail covers is gone, since the request inserts the whole tail.
        if (
            req.swa_branching_seqlen is not None
            and req.swa_branching_seqlen <= prefix_len
        ):
            req.swa_branching_seqlen = None
        self.private_load_indices[req.rid] = full_transfer.device_indices
        self.inflight_load_rids.append(req.rid)
        return len(full_transfer.device_indices), req.last_node

    def _update_load(
        self,
        phase: ExternalLinkerLoadPhase,
        req: Req | None,
        component_transfers: list[tuple[TreeComponent, PoolTransfer]],
        prefix_len: int,
    ) -> list[PoolTransfer]:
        if not component_transfers:
            return []
        full = component_transfers[0][1]
        result = []
        transfers = (
            reversed(component_transfers)
            if phase == ExternalLinkerLoadPhase.ABORT
            else component_transfers
        )
        for component, transfer in transfers:
            transfer = component.update_external_linker_load(
                phase, req, full, transfer, prefix_len
            )
            if transfer is not None:
                result.append(transfer)
        return result

    # ---- offload: device -> remote, driven by the write-through chain ----

    def offload_nodes(self, node_ids: Sequence[NodeId]) -> None:
        """Persist a write-through chain, skipping nodes already in the store."""
        for node_id in node_ids:
            transfers = self.cache.tree_core.build_external_linker_offload_transfers(
                node_id
            )
            if transfers is None:
                continue
            if self._skip_swa:
                transfers = [t for t in transfers if t.name != PoolName.SWA]
            if self.offload_skip_prefix_len:
                transfers = self._drop_offload_skip_prefix(node_id, transfers)
                if transfers is None:
                    continue
            self._offload_node(node_id, transfers)

    @contextmanager
    def skip_offload_prefix(self, length: int):
        """Skip offloading the first ``length`` radix-key units during an insert."""
        self.offload_skip_prefix_len = length
        try:
            yield
        finally:
            self.offload_skip_prefix_len = 0

    def _drop_offload_skip_prefix(
        self, node_id: NodeId, transfers: list[PoolTransfer]
    ) -> Optional[list[PoolTransfer]]:
        info = self.cache.tree_core.collect_path_info([node_id])[0][-1]
        page = self.cache.page_size
        kept = []
        for transfer in transfers:
            first = info.path_end - len(transfer.keys) * page
            drop = min(
                len(transfer.keys),
                max(0, (self.offload_skip_prefix_len - first) // page),
            )
            if drop == len(transfer.keys):
                if transfer.name == PoolName.KV:
                    # The caller delegates storage of this prefix to another writer.
                    # Mark it stored locally to suppress uploads, without a persistence ACK.
                    self.cache.tree_core.mark_external_cache_stored_path(
                        node_id, info.parent_id
                    )
                    return None
                continue
            if drop:
                transfer = replace(
                    transfer,
                    keys=transfer.keys[drop:],
                    device_indices=transfer.device_indices[drop * page :],
                )
            kept.append(transfer)
        return kept

    def _offload_node(self, node_id: NodeId, transfers: list[PoolTransfer]) -> None:
        cache = self.cache
        lock_params = cache.inc_lock_ref(node_id).to_dec_params()
        try:
            queued = self.cache_linker.offload(
                transfers, session_id=getattr(cache, "insert_session_id", None)
            )
        except BaseException:
            cache.dec_lock_ref(node_id, lock_params)
            raise
        if not queued:
            cache.dec_lock_ref(node_id, lock_params)
            return

        cache.tree_core.mark_external_linker_offload_pending(node_id)
        self.pending_offloads.append(_PendingOffload(node_id, lock_params, [node_id]))

    def replace_pending_offload_node(
        self, ack_id: NodeId, old_node_id: NodeId, new_node_ids: list[NodeId]
    ) -> None:
        for index, pending in enumerate(self.pending_offloads):
            if pending.lock_node_id != ack_id:
                continue
            publish_node_ids = []
            for node_id in pending.publish_node_ids:
                if node_id == old_node_id:
                    publish_node_ids.extend(new_node_ids)
                else:
                    publish_node_ids.append(node_id)
            self.pending_offloads[index] = pending._replace(
                publish_node_ids=publish_node_ids
            )
            return

    def num_completed_offloads(self) -> int:
        return min(
            self.cache_linker.num_completed_offloads(), len(self.pending_offloads)
        )

    def take_completed_offloads(self, finish_count: int) -> list[bool]:
        assert finish_count <= len(self.pending_offloads)
        return [self.cache_linker.pop_completed_offload() for _ in range(finish_count)]

    def _probe_object_keys(self, transfers: list[PoolTransfer]) -> list[tuple[list[str], int]]:
        """This stage's master object keys for ``transfers``, per page: the store's own mapping (page hash ->
        prefixed, suffixed component keys), exactly as its loads build them."""
        cl = self.cache_linker
        group = getattr(cl, "pool_group", None)
        expanded = group.resolve_transfers(transfers) if group is not None else transfers
        out = []
        for t in expanded:
            keys, mult = cl.storage._get_hybrid_page_component_keys(list(t.keys), t)
            out.append((cl.storage._tag_keys(keys), mult))
        return out

    def _admin_query(self, keys: list[str]) -> dict:
        """The master's /batch_query_keys reply (no lease), merged over batches of 64: key -> {ok, values | error}."""
        data: dict = {}
        for i in range(0, len(keys), 64):
            query = urllib.parse.quote(",".join(keys[i : i + 64]), safe=",")
            with urllib.request.urlopen(f"{self.admin_url}/batch_query_keys?keys={query}", timeout=2.0) as resp:
                data.update(json.loads(resp.read()).get("data") or {})
        return data

    def _admin_present(self, keys: list[str]) -> set:
        return {k for k, v in self._admin_query(keys).items() if isinstance(v, dict) and v.get("ok")}

    def _enqueue_probe(self, rid: str, transfers: list[PoolTransfer], lookup_future: Future) -> None:
        """Scheduler thread: hand the probe to its own worker, or drop it while one runs."""
        if self._probe_busy:
            self._lookup_stats["probe_dropped"] += 1
            return
        if self._probe_executor is None:
            self._probe_executor = ThreadPoolExecutor(1, thread_name_prefix="linker-probe")
        self._probe_busy = True
        self._probe_executor.submit(self._timed_probe, rid, transfers, lookup_future)

    def _timed_probe(self, rid: str, transfers: list[PoolTransfer], lookup_future: Future) -> None:
        t0 = time.perf_counter()
        try:
            self._fragment_probe(rid, transfers, lookup_future)
        finally:
            ms = int(1e3 * (time.perf_counter() - t0))
            self._lookup_stats["probe_ms_sum"] += ms
            self._lookup_stats["probe_ms_max"] = max(self._lookup_stats["probe_ms_max"], ms)
            self._probe_busy = False

    def _fragment_probe(self, rid: str, transfers: list[PoolTransfer], lookup_future: Future) -> None:
        """Lookup-executor thread: which of this stage's objects of the chain exist, read through the admin endpoint (no
        lease). Waits for the request's own lookup and probes only a chain it found (>= 1 page). Positive control: the
        last found page must read present, else the probe counts probe_selfcheck_fail and turns itself off for the run.
        Logs the present prefix and the pages present past the first missing one."""
        if not self.fragment_probe_frac:
            return
        try:
            found = lookup_future.result(timeout=10.0)
        except Exception:
            return
        hit = found[-1] if found else 0
        if not hit:
            self._lookup_stats["probe_no_hit"] += 1
            return
        try:
            per = self._probe_object_keys(transfers)
            n = min(len(keys) // mult for keys, mult in per)
            page = lambda p, present: all(k in present for keys, m in per for k in keys[p * m : (p + 1) * m])
            control = self._admin_present([k for keys, m in per for k in keys[(hit - 1) * m : hit * m]])
            if not page(hit - 1, control):
                self._lookup_stats["probe_selfcheck_fail"] += 1
                self.fragment_probe_frac = 0.0
                logger.warning(
                    "External linker fragment probe: self-check failed (rid=%s page %d found by the lookup reads absent "
                    "on %s); probe off for this run", rid, hit - 1, self.admin_url,
                )
                return
            present = self._admin_present([k for keys, m in per for k in keys[: n * m]])
        except Exception:
            self._lookup_stats["probe_errors"] += 1
            return
        pages = [page(p, present) for p in range(n)]
        prefix = next((p for p, ok in enumerate(pages) if not ok), n)
        beyond = sum(pages[prefix:])
        ls = self._lookup_stats
        ls["probe_n"] += 1
        ls["probe_pages"] += n
        ls["probe_prefix_pages"] += prefix
        ls["probe_fragment_pages"] += beyond
        logger.info(
            "External linker fragment probe: rid=%s pages=%d prefix=%d beyond_present=%d beyond_objects=%d lookup_hit=%d",
            rid, n, prefix, beyond,
            sum(k in present for keys, m in per for k in keys[prefix * m : n * m]), hit,
        )

    def _touch(self, rid: str, hashes: list[str], kind: str) -> None:
        """(c1) Fire-and-forget ExistKey on ``hashes``: grants their read lease; the result is not used."""
        if not hashes or self._lookup_executor is None:
            return
        transfers = [
            c.build_external_linker_transfer(LinkerTransferPhase.LOOKUP, None, hashes)
            for c in self._components
        ]
        if any(t is None for t in transfers):
            return
        rpc = getattr(self.cache_linker, "lease_refresh", None) or self.cache_linker.lookup
        self._lookup_executor.submit(rpc, kind + ":" + rid, transfers)
        self._lookup_stats[kind + "_rpcs"] += 1
        self._lookup_stats[kind + "_keys"] += len(hashes)

    def commit_completed_offloads(self, successes: Sequence[bool]) -> None:
        assert len(successes) <= len(self.pending_offloads)
        for success in successes:
            pending = self.pending_offloads.pop(0)
            if success and self.touch_after_put:
                for node_id in pending.publish_node_ids:
                    self._touch("put", self.cache.tree_core.get_hash_values(node_id), "touch_put")
            if success and self.session_evict_target and self._evict_keys_ok is None and pending.publish_node_ids:
                written = self.cache.tree_core.get_hash_values(pending.publish_node_ids[-1])
                if written:  # just written by this stage: a self-check page until c3 has its verdict
                    self._known_present.append(written[-1])
            self.cache.tree_core.finish_external_linker_offload(
                pending.publish_node_ids, pending.lock_node_id, success
            )
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)

    def start_layer_wise_loading(self) -> int:
        self.inflight_load_index = self.cache_linker.start_layer_wise_loading()
        return self.inflight_load_index

    def finish_loads(self) -> list[str]:
        """Wait for the loads of the batch that just ran; return the rids whose
        KV did not land, identically on every rank.

        Runs after that batch's forward, so no slot it wrote to has been freed
        or inserted yet. Every rank holds the same ``inflight_load_rids``
        because ``match`` reduces the hit across ranks, so either all ranks or
        none enter the reduction. The verdict is reduced because an MLA dedup
        peer only receives the broadcast and never sees the source's error.
        """
        rids, self.inflight_load_rids = self.inflight_load_rids, []
        if not rids:
            return []
        index, self.inflight_load_index = self.inflight_load_index, -1
        landed = torch.tensor(
            [int(self.cache_linker.finish_layer_wise_loading(index))],
            dtype=torch.int,
        )
        self.cache._all_reduce_attn_groups(landed, torch.distributed.ReduceOp.MIN)
        for rid in rids:
            self.private_load_indices.pop(rid, None)
        if landed.item():
            return []
        logger.error(
            "External linker load failed; aborting %d request(s): %s", len(rids), rids
        )
        return rids

    # ---- lifecycle ----

    def reset(self) -> None:
        self.cache_linker.reset()
        self._clear_state()

    def _clear_state(self) -> None:
        for request in list(self.prefetches):
            self._finish_prefetch(request, publish=False)
        for request in list(self.lookups):
            self._cancel_lookup(request)
        for request in list(self.holds):
            self.release_hold(request, admitted=False)
        self.unheld.clear()
        self.match_targets.clear()
        self.near_term_parked.clear()
        self._settle_waits.clear()
        self._wait_fit_enq.clear()
        self._room_refused.clear()
        self._sessions.clear()
        self._request_session.clear()
        self._lease_chains.clear()
        self._lease_refresh_futures.clear()
        self._reads.clear()
        self._late_beats.clear()
        self._pending_requeries.clear()
        self._requery_started.clear()
        self._requery_records.clear()
        self.hit_markers.clear()
        self.inflight_load_rids.clear()
        self.private_load_indices.clear()
        self.inflight_load_index = -1
        for pending in self.pending_offloads:
            self.cache.tree_core.finish_external_linker_offload(
                pending.publish_node_ids, pending.lock_node_id, False
            )
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)
        self.pending_offloads.clear()
        if self.pp_relay is not None:
            self.pp_relay.reset()

    def release_request(self, request: CacheRequestHandle) -> None:
        rid = request.rid
        self.hit_markers.pop(rid, None)
        self._pending_requeries.pop(request, None)
        self._requery_started.pop(request, None)
        self._cancel_prefetch(request)
        self.release_hold(request, admitted=False)
        # A queued load targets slots the request owns and is about to free.
        # A started one is finished by finish_loads before anything is freed.
        if rid in self.inflight_load_rids and self.cache_linker.cancel_queued_load(rid):
            self.inflight_load_rids.remove(rid)
            self.private_load_indices.pop(rid, None)

    def close(self) -> None:
        if self._probe_executor is not None:
            self._probe_executor.shutdown(wait=False)
        if self._evict_executor is not None:
            self._evict_executor.shutdown(wait=False)
        self.cache_linker.close()
        self._clear_state()
        if self._lookup_executor is not None:
            self._lookup_executor.shutdown(wait=False, cancel_futures=True)
