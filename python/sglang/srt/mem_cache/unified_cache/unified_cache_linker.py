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

import logging
import time
from abc import ABC, abstractmethod
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

# The token and state components the linker stores, besides the tree's window
# slots (which it finds by family, whatever ids the tree gives them).
_EXTERNAL_LINKER_SUPPORTED_COMPONENTS = frozenset(
    {
        ComponentType.FULL,
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


class _PrefetchHold(NamedTuple):
    node_id: NodeId
    lock: DecLockRefParams
    tokens: int


class _PendingOffload(NamedTuple):
    lock_node_id: NodeId
    lock_params: DecLockRefParams
    publish_node_ids: list[NodeId]


class UnifiedCacheLinkerWrapper:
    """Drives an external KV store on behalf of one :class:`UnifiedRadixCache`."""

    def __init__(
        self,
        cache: UnifiedRadixCache,
        cache_linker: UnifiedCacheLinker,
    ):
        # The linker keeps every window slot of the tree, each as its own pool.
        windows = cache.window_slots()
        supported = _EXTERNAL_LINKER_SUPPORTED_COMPONENTS | set(windows)
        unsupported = set(cache.tree_components) - supported
        if unsupported:
            names = ", ".join(
                component.name for component in sorted(unsupported, key=int)
            )
            raise ValueError(
                "External cache linker supports only Full, window and Mamba tree "
                f"components; unsupported: {names}"
            )

        self.cache = cache
        self.cache_linker = cache_linker
        self.load_mode = cache_linker.load_mode
        if self.load_mode == "prefetch" and cache.pp_size > 1:
            raise ValueError("External linker prefetch currently requires PP=1.")
        self._windows = tuple(cache.components[slot] for slot in windows)
        # A window whose pages the store lacks tombstones restored tails instead.
        self._skipped_windows = tuple(
            window
            for window in self._windows
            if window.pool_name not in cache_linker.pools
        )
        self._components = tuple(
            component
            for component in cache._components_tuple
            if not any(component is window for window in self._skipped_windows)
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
        self._lookup_stats = dict.fromkeys(
            ("issued", "skipped_no_hash", "zero_restorable", "requeried"), 0
        )
        self._lookup_next_log = 0.0
        # Re-queried request -> when its re-query started; waits until publication (s).
        self._requery_started: dict[CacheRequestHandle, float] = {}
        self._requery_waits: list[float] = []
        # Lookup kind ("enq" at enqueue, "rq" for a re-query) -> seconds per phase since the last log line.
        self._lookup_kind = "enq"
        self._phase_s = {k: dict.fromkeys(("n", "match", "keys", "rpc", "load"), 0.0) for k in ("enq", "rq")}
        if self.hold_cap_tokens:
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

        # Mamba keeps a single state slot per node, so a hit is worth one slot.
        mamba_host_hit_length = 1 if PoolName.MAMBA in by_pool else 0

        self.hit_markers[req.rid] = ExternalCacheHitMarker(
            tail_hashes=list(tail_hashes[:hit_pages]),
            device_hit_len=device_hit_len,
        )
        window_hits = dict(result.window_host_hit_lengths)
        for window in self._windows:
            transfer = by_pool.get(window.pool_name)
            hit = min(len(transfer.keys), hit_pages) * page if transfer else 0
            name = window.window_sub_pool
            window_hits[name] = max(window_hits.get(name, 0), hit)
        return result._replace(
            last_host_node=result.best_match_node,
            host_hit_length=hit_tokens,
            window_host_hit_lengths=window_hits,
            mamba_host_hit_length=max(
                result.mamba_host_hit_length, mamba_host_hit_length
            ),
        )

    def prefetch(self, req: Req) -> None:
        """Stage a queued request; publish only after all ranks succeed."""
        if self.load_mode != "prefetch":
            return
        request = req.cache_request_handle
        for previous in self.prefetches:
            if previous.rid == request.rid and previous != request:
                self._cancel_prefetch(previous)
        for previous in [lk for lk in self.lookups if lk.rid == request.rid]:
            if previous != request:
                self._cancel_lookup(previous)
        for previous in [h for h in self.holds if h.rid == request.rid]:
            if previous != request:
                self.release_hold(previous, admitted=False)
        if self.cache.disable or request in self.prefetches or request in self.lookups:
            return
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
        if self.requery_on_device_loss:
            self.match_targets[request] = start
        last_hash = self.cache.get_last_hash_value(req.last_node)
        if len(key) <= start:
            return
        if start and last_hash is None:
            self._lookup_stats["skipped_no_hash"] += 1
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
        self._lookup_stats["issued"] += 1
        self._log_lookup_stats()
        if self._lookup_executor is not None:
            # The anchor stays locked while the lookup runs, as for a load.
            self.lookups[request] = _PendingLookup(
                key,
                hashes,
                start,
                req.last_node,
                self.cache.inc_lock_ref(req.last_node).to_dec_params(),
                self._lookup_executor.submit(
                    self._timed_lookup, kind if timed else None, req.rid, lookups
                ),
                timeout_s=timeout_s,
                kind=kind,
            )
            return
        try:
            restorable = self._timed_lookup(kind if timed else None, req.rid, lookups)
        except Exception:
            logger.warning("External linker prefetch lookup failed", exc_info=True)
            restorable = []
        anchor_lock = self.cache.inc_lock_ref(req.last_node).to_dec_params()
        t2 = time.perf_counter() if timed else 0.0
        self._start_prefetch_load(
            request, key, hashes, start, req.last_node, anchor_lock, restorable, timeout_s
        )
        if timed:
            self._phase_s[kind]["load"] += time.perf_counter() - t2

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
    ) -> None:
        """Agree on the restorable prefix across ranks and queue its load."""
        page = self.cache.page_size
        pages = self._sync_restorable_prefix(
            restorable, num_pages=len(hashes), device_hit_pages=0
        )
        if not pages:
            self._lookup_stats["zero_restorable"] += 1
            self.cache.dec_lock_ref(anchor_node_id, anchor_lock)
            return
        end = start + pages * page
        if request in self.match_targets:
            self.match_targets[request] = end
        state = _ExternalCachePrefetch(
            key[:end],
            anchor_node_id,
            anchor_lock,
            [],
            timeout_s=timeout_s,
        )
        self.prefetches[request] = state
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
                self.check_prefetch_events()

    def has_pending_prefetch(self) -> bool:
        return bool(self.prefetches or self.lookups)

    def check_prefetch_progress(self, request: CacheRequestHandle) -> bool:
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
        self.check_prefetch_events()
        self._cancel_prefetch(request)
        return True

    def _check_lookup_progress(self, request: CacheRequestHandle) -> bool:
        """Admission poll while the request's lookup runs, under the stop
        policy of a running load."""
        lookup = self.lookups[request]
        policy = self.cache.prefetch_stop_policy
        if policy == "wait_complete":
            return False
        if policy == "timeout":
            now = time.perf_counter()
            expired = torch.tensor(
                lookup.wait_started is not None
                and now - lookup.wait_started > lookup.timeout_s,
                dtype=torch.int,
            )
            self.cache._all_reduce_attn_groups(expired, torch.distributed.ReduceOp.MAX)
            if not expired.item():
                if lookup.wait_started is None:
                    lookup.wait_started = now
                return False
        self._cancel_lookup(request)
        return True

    def requery_after_device_loss(self, req: Req) -> bool:
        """Look up again the prefix ``req`` lost on device since its last lookup.

        True while the new lookup or load runs, so the request waits for it.
        """
        request = req.cache_request_handle
        target = self.match_targets.get(request)
        if (
            target is None
            or req.prefix_len >= target
            or request in self.prefetches
            or request in self.lookups
            or request in self.holds
        ):
            return False
        self._lookup_stats["requeried"] += 1
        started = time.perf_counter()
        self._lookup_kind = "rq"
        try:
            self.prefetch(req)
        finally:
            self._lookup_kind = "enq"
        pending = request in self.prefetches or request in self.lookups
        if pending and self.log_lookup_stats:
            self._requery_started[request] = started
        return pending

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
            "p50 %.3f s, p90 %.3f s; ms per lookup: %s",
            self._lookup_stats,
            len(waits),
            p50,
            p90,
            "; ".join(phases),
        )

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
            for (request, _), finished in zip(pending, done.tolist()):
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

    def check_prefetch_events(self) -> None:
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
        started = self._requery_started.pop(request, None)
        if started is not None:
            self._requery_waits.append(time.perf_counter() - started)
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
            page = self.cache.page_size
            for window in self._windows:
                transfer = by_pool.get(window.pool_name)
                if transfer is not None:
                    storage.bind_window(
                        window.window_sub_pool,
                        full[-len(transfer.device_indices) :],
                        transfer.device_indices,
                    )
                skipped = any(window is w for w in self._skipped_windows)
                kept = (window.sliding_window_size + page - 1) // page * page
                evicted[window.component_type] = end if skipped else max(0, end - kept)
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
        # Keep a non-restorable window range as tombstones instead of rebuilding
        # it from an uninitialized FULL-to-window mapping when the tail is inserted.
        for window in self._skipped_windows:
            name = window.window_sub_pool
            req.kv.set_evicted_seqlen(
                name, max(req.kv.get_evicted_seqlen(name), prefix_len)
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
            if self._skipped_windows:
                skipped = {window.pool_name for window in self._skipped_windows}
                transfers = [t for t in transfers if t.name not in skipped]
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

    def commit_completed_offloads(self, successes: Sequence[bool]) -> None:
        assert len(successes) <= len(self.pending_offloads)
        for success in successes:
            pending = self.pending_offloads.pop(0)
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
        self._requery_started.clear()
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

    def release_request(self, request: CacheRequestHandle) -> None:
        rid = request.rid
        self.hit_markers.pop(rid, None)
        self._requery_started.pop(request, None)
        self._cancel_prefetch(request)
        self.release_hold(request, admitted=False)
        # A queued load targets slots the request owns and is about to free.
        # A started one is finished by finish_loads before anything is freed.
        if rid in self.inflight_load_rids and self.cache_linker.cancel_queued_load(rid):
            self.inflight_load_rids.remove(rid)
            self.private_load_indices.pop(rid, None)

    def close(self) -> None:
        self.cache_linker.close()
        self._clear_state()
        if self._lookup_executor is not None:
            self._lookup_executor.shutdown(wait=False, cancel_futures=True)
