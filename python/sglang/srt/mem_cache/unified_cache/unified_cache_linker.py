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
* ``init_load_back``    -> :meth:`UnifiedCacheLinkerWrapper.load_back`
* ``BackupKV`` actions  -> :meth:`UnifiedCacheLinkerWrapper.offload_nodes`

A load lands in device slots private to the loading request and joins the tree
only when that request inserts it like any computed KV. A failed load therefore
reaches nothing but its own requests, which the scheduler aborts.
"""

from __future__ import annotations

import logging
import math
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
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
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)
device_module = get_device_module()

# init_load_back's result for a request skipped this pass (async admission):
# the scheduler keeps admitting the requests behind it.
LINKER_ADMISSION_SKIPPED = "linker_admission_skipped"

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
        # Do not short-circuit after an error: later layers may still be writing.
        errors = [future.exception() for future in futures]
        landed = all(error is None for error in errors)
        del self.futures[index]
        return landed

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self.futures.clear()


class UnifiedCacheLinker(ABC):
    """External KV store reached directly from the device pools.

    Out-of-tree linkers are constructed per worker as
    ``Linker(server_args, params, components=..., extra_config=..., forward_stream=...)``.
    Store ``forward_stream`` and use ``record_offload_ready()`` to gate offloads.
    ``layer_done_counter`` provides ``set_consumer(index)`` and
    ``wait_until(layer)``. Report completion only after device copies finish.
    """

    layer_done_counter: object
    pools: dict[PoolName, object]
    # The stream the forward writes the device pools on; None when the current
    # stream is the pages' only writer.
    forward_stream: Optional[torch.cuda.Stream] = None
    # Log each request's load time and bytes.
    log_load_timing: bool = False
    # Seconds the last pin_hit waited on the store.
    last_pin_rpc_s: float = 0.0
    # Under PP: PP0's pin_hit checks and holds every stage's objects of a hit.
    pins_hits: bool = False

    def record_offload_ready(self) -> list[torch.cuda.Event]:
        """Record the events an offload waits on before it reads device pages.

        The current stream has computed which pages to offload, and the forward
        stream may still be writing them: prefill read-done lets the scheduler's
        stream run ahead of the forward that fills a stashed chunk.
        """
        indices_ready = device_module.Event()
        indices_ready.record()
        events = [indices_ready]
        if self.forward_stream is not None:
            kv_written = device_module.Event()
            kv_written.record(self.forward_stream)
            events.append(kv_written)
        return events

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

    def lookup_stages(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        """lookup() over every pipeline stage's objects (SGLANG_LINKER_PP)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support pipeline parallelism."
        )

    def refresh_device_prefix(self, anchor: object, hashes: Callable) -> None:
        """Renew the store leases of a prefix the device serves; optional."""

    def pin_hit(self, rid: str, transfers: list[PoolTransfer]) -> int:
        """PP0 with ``pins_hits``, as a relayed hit is admitted: hold every
        stage's objects of it until their loads and return how many leading
        pages every stage can load."""
        raise NotImplementedError

    def unpin(self, rid: str) -> None:
        """Release what pin_hit holds for ``rid`` when it will not load."""

    def start_prepin(self, rid: str, transfers: list[PoolTransfer]) -> None:
        """PP0: begin pin_hit for a waiting request off the scheduler thread."""

    def take_prepin(self, rid: str, transfers: list[PoolTransfer]) -> Optional[int]:
        """PP0: the result of a finished start_prepin of this hit, else None."""
        return None

    def release_stale_prepins(self) -> None:
        """PP0: release pre-pins whose requests were not admitted in time."""

    def keep_prepins(self, rids: Sequence[str]) -> None:
        """PP0, async admission: end the pre-pins of requests not in ``rids``
        and renew the kept ones before their sessions lapse."""

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


class _PendingOffload(NamedTuple):
    lock_node_id: NodeId
    lock_params: DecLockRefParams
    publish_node_ids: list[NodeId]


class UnifiedCacheLinkerWrapper:
    """Drives an external KV store on behalf of one :class:`UnifiedRadixCache`."""

    # Async admission: the relayed verdict of a skipped hit, and the passes in a
    # row a hit may be skipped before PP0 pins it at admission (no starvation).
    _SKIPPED = -1
    _SKIP_LIMIT = 3

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
        # Requests loading in the batch being built, then its counter index once
        # the batch starts; finish_loads consumes both after its forward.
        self.inflight_load_rids: list[str] = []
        # Loaded tails stay outside the tree until their forward completes.
        self.private_load_indices: dict[str, torch.Tensor] = {}
        self.inflight_load_index = -1
        # Offloads in flight, each holding a lock on its node until it lands.
        self.pending_offloads: list[_PendingOffload] = []
        # Under PP every stage admits from PP0's lookup at enqueue
        # (Req.linker_pp_hit); otherwise SGLANG_LINKER_LOOKUP_ONCE reuses a
        # request's lookup across scheduling passes: rid -> (device prefix the
        # lookup started from, restorable prefix lengths in tokens).
        self.pp_size = getattr(cache, "pp_size", 1)
        self.pp_rank = getattr(cache, "pp_rank", 0)
        self.lookup_once = envs.SGLANG_LINKER_LOOKUP_ONCE.get()
        self.insert_at_release = (
            self.pp_size > 1 and envs.SGLANG_LINKER_PP_LOAD_CONSENSUS.get()
        )
        self.fail_load_prob = (
            envs.SGLANG_TEST_LINKER_FAIL_LOAD_PROB.get()
            if envs.SGLANG_TEST_LINKER_FAIL_LOAD_STAGE.get() == self.pp_rank
            else 0.0
        )
        self.lookup_cache: dict[str, tuple[int, list[int]]] = {}
        # rid -> (device prefix, tail length, tail page hashes) of its last match.
        self.tail_hash_cache: dict[str, tuple[int, int, list[str]]] = {}
        # PP0: rid -> when its relayed hit was last looked up. A hit older than
        # the store's lease is looked up again before its load.
        self.pp_lookup_time: dict[str, float] = {}
        self.stale_after_s = envs.SGLANG_LINKER_LEASE_TTL_SECS.get()
        self.pp_relookups = self.pp_relookups_shrunk = 0
        # rids whose relayed hit outlived its device prefix: PP0 looks their
        # whole tail up again before the load (every stage tracks them alike).
        self.pp_forced_relookup: set[str] = set()
        self.pp_counts: dict[str, int] = dict.fromkeys(
            (
                "anchor_shrunk",
                "shrunk_relooked",
                "recovered_pages",
                "lost_pages",
                "shrunk_unlooked",
                "relookups",
                "shrunk",
            ),
            0,
        )
        self.pp_counts_logged_at = time.monotonic()
        # PP admissions with a hit: (seconds in load_back, seconds of it waiting
        # on pin_hit's store call); and pins that found an object gone.
        self.admission_samples: list[dict[str, float]] = []
        # PP0: after each pass, pre-pin this many next waiting hits; admissions
        # that found a finished pre-pin, and those that pinned themselves.
        self.prepin_depth = envs.SGLANG_LINKER_PREPIN_DEPTH.get() if self.pp_size > 1 else 0
        self.prepin_counts = {"ready": 0, "missed": 0}
        # Async admission (PP0 decides, every stage follows): rid -> passes in a
        # row its pre-pin was not ready; after _SKIP_LIMIT PP0 pins it itself.
        self.async_admission = bool(
            self.prepin_depth and envs.SGLANG_LINKER_ASYNC_ADMISSION.get()
        )
        self.skip_counts: dict[str, int] = {}
        self.async_counts = {"skipped": 0, "forced_sync": 0}
        # Hits admitted in the current pass, and their moving average per pass.
        self.pass_hits = 0
        self.hits_per_pass = 0.0
        self.admission_rpc_s = 0.0
        self.admission_marks: dict[str, float] = {}
        self.pp_load_holes = 0

        cache.tree_core.enable_external_cache_linker = True
        cache.write_through_threshold = 1

    def _lookup_transfers(self, hashes: list[str]) -> list[PoolTransfer]:
        transfers = []
        for component in self._components:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOOKUP, None, hashes
            )
            if transfer is None:
                return []
            transfers.append(transfer)
        return transfers

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
        if device_hit_len > 0:
            self.cache_linker.refresh_device_prefix(
                result.last_device_node,
                lambda: cache.tree_core.get_prefix_hash_values(result.last_device_node),
            )
        if device_hit_len >= len(key):
            return result

        known = self._known_restorable(req, device_hit_len)
        if known is not None and not known:
            return self._no_hit(req, result)

        tail_hashes = self._cached_tail_hashes(key, req, result, device_hit_len)
        if not tail_hashes:
            return self._no_hit(req, result)

        lookup_transfers = self._lookup_transfers(tail_hashes)
        if not lookup_transfers:
            return self._no_hit(req, result)
        if known is None:
            known = self._lookup(req, device_hit_len, lookup_transfers)
        hit_pages = self._sync_restorable_prefix(
            known,
            num_pages=len(tail_hashes),
            device_hit_pages=0,
        )
        if hit_pages == 0:
            return self._no_hit(req, result)
        by_pool = {transfer.name: transfer for transfer in lookup_transfers}
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

    def _no_hit(self, req: Req, result: MatchResult) -> MatchResult:
        """A request whose device prefix shrank but leaves match with no hit
        skips load_back's re-lookup and recomputes those pages; counted, it
        should stay 0."""
        if req.rid in self.pp_forced_relookup:
            self.pp_forced_relookup.discard(req.rid)
            self.pp_counts["shrunk_unlooked"] += 1
        return result

    def _known_restorable(self, req: Req, device_hit_len: int) -> Optional[list[int]]:
        """Restorable prefix lengths in pages past ``device_hit_len`` from an
        earlier lookup, or None when this match must look the store up. Under PP
        only PP0's enqueue match looks up; every rank then uses its relayed
        result, so all stages admit the same hit without a store query."""
        if self.pp_size > 1:
            if req.linker_pp_lookup_pending:
                return None
            entry = req.linker_pp_hit
            if entry is None:
                return []
            if device_hit_len < entry[0]:
                # The device prefix shrank below the looked-up start while the
                # request queued, so the pages in between were never checked:
                # admit the relayed stops and let PP0 look the whole tail up again
                # before the load (Rhea 57360 dropped such hits instead).
                if req.rid not in self.pp_forced_relookup:
                    self.pp_forced_relookup.add(req.rid)
                    self.pp_counts["anchor_shrunk"] += 1
                # The start itself was on device, so written through: admit it
                # too, or a hit relayed with no stops recomputes the whole prefix
                # (Rhea 59439).
                entry = (entry[0], [entry[0], *entry[1]])
        elif self.lookup_once:
            entry = self.lookup_cache.get(req.rid)
            if entry is None or device_hit_len < entry[0]:
                return None
        else:
            return None
        page = self.cache.page_size
        return [(end - device_hit_len) // page for end in entry[1] if end > device_hit_len]

    def _lookup(
        self, req: Req, device_hit_len: int, transfers: list[PoolTransfer]
    ) -> list[int]:
        page = self.cache.page_size
        if self.pp_size > 1:
            pages = self.cache_linker.lookup_stages(req.rid, transfers)
            req.linker_pp_hit = (device_hit_len, [device_hit_len + p * page for p in pages])
            self.pp_lookup_time[req.rid] = time.monotonic()
            return pages
        pages = self.cache_linker.lookup(req.rid, transfers)
        if self.lookup_once:
            self.lookup_cache[req.rid] = (
                device_hit_len,
                [device_hit_len + p * page for p in pages],
            )
        return pages

    def _pp_fresh_hit_pages(
        self, req: Req, hit: ExternalCacheHitMarker, forced: bool = False
    ) -> int:
        """Pages of ``hit`` to load, the same on every stage. PP0 pins every
        stage's objects of the hit (pin_hit): the store's read-session start
        checks them and holds them until each stage's load. An object evicted
        since PP0's lookup shrinks the hit to the deepest stop PP0 finds below
        it, and the rest is recomputed rather than failed. Rhea 60076: looking
        every stale or shrunk hit up again held PP0 on the master's ExistKey
        ceiling."""
        admitted = len(hit.tail_hashes)
        transfers = self._lookup_transfers(hit.tail_hashes)
        if not (self.cache_linker.pins_hits and transfers):
            return self._pp_relookup_hit_pages(req, hit, forced)
        # Only PP0 pins: the relay below carries PP0's value, not a reduction.
        pinned = admitted
        if self.pp_rank == 0:
            taken = None
            if self.prepin_depth:
                taken = self.cache_linker.take_prepin(req.rid, transfers)
                self.prepin_counts["missed" if taken is None else "ready"] += 1
            if (
                taken is None
                and self.async_admission
                and self.skip_counts.get(req.rid, 0) < self._SKIP_LIMIT
            ):
                # Skip this pass rather than wait on the store; the relay
                # carries the skip, so every stage skips it too.
                pinned = self._SKIPPED
                self.cache_linker.start_prepin(req.rid, transfers)
            else:
                # Not ready: pin now. Without async admission, deferring would
                # end this pass's admission for every request behind it.
                if taken is None and self.async_admission:
                    self.async_counts["forced_sync"] += 1
                pinned = (
                    self.cache_linker.pin_hit(req.rid, transfers)
                    if taken is None
                    else taken
                )
                self.admission_rpc_s = self.cache_linker.last_pin_rpc_s
        keep = torch.tensor([min(pinned, admitted)], dtype=torch.int64)
        self.cache._all_reduce(keep, torch.distributed.ReduceOp.MIN)
        if int(keep[0]) == self._SKIPPED:
            # A TP peer of PP0 that took its pre-pin drops it with the rest.
            self.cache_linker.unpin(req.rid)
            self.skip_counts[req.rid] = self.skip_counts.get(req.rid, 0) + 1
            self.async_counts["skipped"] += 1
            self._maybe_log_pp_counts()
            return self._SKIPPED
        self.skip_counts.pop(req.rid, None)
        self.pass_hits += 1
        if int(keep[0]) < admitted:
            # Every stage drops its pins; the next pass pins the shorter hit.
            self.cache_linker.unpin(req.rid)
            self.pp_load_holes += 1
            best = torch.tensor([admitted], dtype=torch.int64)
            if self.pp_rank == 0:
                pages = self.cache_linker.lookup_stages(req.rid, transfers)
                best[0] = max((p for p in pages if p <= int(keep[0])), default=0)
            self.cache._all_reduce(best, torch.distributed.ReduceOp.MIN)
            logger.info(
                "Linker PP admission found a hole: rid=%s, objects up to page %d of "
                "%d; loading %d",
                req.rid,
                int(keep[0]),
                admitted,
                int(best[0]),
            )
            keep = best
        self._maybe_log_pp_counts()
        return int(keep[0])

    def prepin(self, waiting: Sequence[Req]) -> None:
        """PP0, after a scheduling pass: start the all-stage pins of the next
        ``prepin_depth`` waiting requests with a hit (in admission order), so
        their admission takes a finished pin instead of waiting on the store."""
        if not self.prepin_depth:
            return
        if self.async_admission:
            self._prepin_window(waiting)
            return
        self.cache_linker.release_stale_prepins()
        started = 0
        for req in waiting:
            hit = self.hit_markers.get(req.rid)
            if hit is None:
                continue
            transfers = self._lookup_transfers(hit.tail_hashes)
            if transfers:
                self.cache_linker.start_prepin(req.rid, transfers)
            started += 1
            if started >= self.prepin_depth:
                return

    def _prepin_window(self, waiting: Sequence[Req]) -> None:
        """Async admission: pre-pin the hits of the requests the next passes
        admit. ``waiting`` is the policy-sorted queue after this pass; the window
        is twice the recent hits admitted per pass plus one, up to
        prepin_depth. Pins outside it end; those inside are renewed."""
        self.hits_per_pass = 0.8 * self.hits_per_pass + 0.2 * self.pass_hits
        self.pass_hits = 0
        depth = min(self.prepin_depth, max(2, math.ceil(2 * self.hits_per_pass) + 1))
        window = []
        for req in waiting:
            if len(window) >= depth:
                break
            hit = self.hit_markers.get(req.rid)
            if hit is None:
                continue
            transfers = self._lookup_transfers(hit.tail_hashes)
            if transfers:
                self.cache_linker.start_prepin(req.rid, transfers)
                window.append(req.rid)
        self.cache_linker.keep_prepins(window)

    def _pp_relookup_hit_pages(
        self, req: Req, hit: ExternalCacheHitMarker, forced: bool
    ) -> int:
        """_pp_fresh_hit_pages for a linker that cannot pin. PP0 looked the hit
        up at enqueue; once that is older than the store's lease, its objects
        may have been evicted while the request queued (Rhea 57360: a 354 s wait
        failed its load), so PP0 looks them up again, as an unrelayed match
        does every pass, and relays the longest prefix still restorable."""
        admitted = len(hit.tail_hashes)
        keep = torch.tensor([admitted], dtype=torch.int64)
        looked = self.pp_lookup_time.get(req.rid)
        now = time.monotonic()
        if self.pp_rank == 0 and looked is not None and (
            forced or now - looked > self.stale_after_s
        ):
            transfers = self._lookup_transfers(hit.tail_hashes)
            pages = self.cache_linker.lookup_stages(req.rid, transfers) if transfers else []
            self.pp_lookup_time[req.rid] = time.monotonic()
            keep[0] = max((p for p in pages if p <= admitted), default=0)
            self.pp_relookups += 1
            self.pp_relookups_shrunk += int(keep[0]) < admitted
            self.pp_counts["relookups"] += 1
            self.pp_counts["shrunk"] += int(keep[0]) < admitted
            logger.info(
                "Linker PP stale hit looked up again: rid=%s age=%.1f s, %d of %d "
                "pages restorable; re-lookups %d, shrunk %d",
                req.rid,
                now - looked,
                int(keep[0]),
                admitted,
                self.pp_relookups,
                self.pp_relookups_shrunk,
            )
        self.cache._all_reduce(keep, torch.distributed.ReduceOp.MIN)
        self._maybe_log_pp_counts()
        return int(keep[0])

    def _maybe_log_pp_counts(self) -> None:
        now = time.monotonic()
        if self.pp_rank != 0 or now - self.pp_counts_logged_at < 60:
            return
        window = now - self.pp_counts_logged_at
        logger.info(
            "Linker PP admissions in %.0f s: anchor-shrunk hits %d (re-looked "
            "up %d: %d pages recovered, %d lost; %d without a re-lookup), stale "
            "re-lookups %d (shrunk %d)",
            window,
            *self.pp_counts.values(),
        )
        samples, self.admission_samples = self.admission_samples, []

        def ms(part):
            values = sorted(sample[part] * 1e3 for sample in samples)
            if not values:
                return "-"
            p50 = values[len(values) // 2]
            return f"{p50:.1f}/{values[min(len(values) - 1, int(0.9 * len(values)))]:.1f}"

        evicting = [s["evicted"] for s in samples if s["evicted"]]
        logger.info(
            "Linker PP admission timing in %.0f s: %d admissions with a hit, ms "
            "p50/p90 total %s = session start %s + relay %s + slot build %s "
            "(evict %s; %d hits evicted, mean %d max %d tokens) + load call %s + "
            "prepare %s; load holes %d; pre-pins ready %d, missed %d",
            window,
            len(samples),
            ms("total"),
            ms("pin"),
            ms("relay"),
            ms("build"),
            ms("evict"),
            len(evicting),
            sum(evicting) // max(len(evicting), 1),
            max(evicting, default=0),
            ms("load"),
            ms("prepare"),
            self.pp_load_holes,
            self.prepin_counts["ready"],
            self.prepin_counts["missed"],
        )
        if self.async_admission:
            logger.info(
                "Linker PP async admission in %.0f s: %s; pre-pin window %.1f hits "
                "per pass; renewals %d",
                window,
                self.async_counts,
                self.hits_per_pass,
                getattr(self.cache_linker, "prepin_renewals", 0),
            )
            self.async_counts = dict.fromkeys(self.async_counts, 0)
        self.prepin_counts = dict.fromkeys(self.prepin_counts, 0)
        self.pp_load_holes = 0
        self.pp_counts = dict.fromkeys(self.pp_counts, 0)
        self.pp_counts_logged_at = now

    def _cached_tail_hashes(
        self, key: RadixKey, req: Req, result: MatchResult, device_hit_len: int
    ) -> list[str]:
        """_tail_hashes, kept per request while its device prefix and length hold:
        SPF re-matches every waiting request on every pass."""
        if not (self.pp_size > 1 or self.lookup_once):
            return self._tail_hashes(key, result, device_hit_len)
        cached = self.tail_hash_cache.get(req.rid)
        if cached is not None and cached[:2] == (device_hit_len, len(key)):
            return cached[2]
        hashes = self._tail_hashes(key, result, device_hit_len)
        self.tail_hash_cache[req.rid] = (device_hit_len, len(key), hashes)
        return hashes

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
        # Timed on PP0, the rank that logs it.
        if self.pp_size == 1 or self.pp_rank != 0 or req.rid not in self.hit_markers:
            return self._load_back(req)
        began = time.monotonic()
        self.admission_rpc_s = 0.0
        self.admission_marks = {}
        self._components[0].load_evict = (0, 0.0)
        result = None
        try:
            result = self._load_back(req)
            return result
        finally:
            end = time.monotonic()
            marks = self.admission_marks
            fresh = marks.get("fresh", end)
            built = marks.get("built", fresh)
            queued = marks.get("queued", built)
            evicted, evict_s = self._components[0].load_evict
            if result is not LINKER_ADMISSION_SKIPPED:
                self.admission_samples.append(
                    {
                        "total": end - began,
                        "pin": self.admission_rpc_s,
                        "relay": fresh - began - self.admission_rpc_s,
                        "build": built - fresh,
                        "evict": evict_s,
                        "evicted": evicted,
                        "load": queued - built,
                        "prepare": end - queued,
                    }
                )

    def _load_back(self, req: Req) -> tuple[int, NodeId]:
        cache = self.cache
        hit = self.hit_markers.pop(req.rid, None)
        if hit is None:
            return 0, req.last_node
        if self.pp_size > 1:
            forced = req.rid in self.pp_forced_relookup
            keep = self._pp_fresh_hit_pages(req, hit, forced)
            if keep == self._SKIPPED:
                return LINKER_ADMISSION_SKIPPED
            self.admission_marks["fresh"] = time.monotonic()
            anchor, stops = req.linker_pp_hit
            end = hit.device_hit_len + keep * cache.page_size
            if forced:
                # Re-anchor at the device prefix this lookup started from.
                self.pp_forced_relookup.discard(req.rid)
                anchor = hit.device_hit_len
                stops = sorted({stop for stop in stops if anchor < stop <= end} | {end})
                req.linker_pp_hit = (anchor, stops)
                self.pp_counts["shrunk_relooked"] += 1
                self.pp_counts["recovered_pages"] += keep
                self.pp_counts["lost_pages"] += len(hit.tail_hashes) - keep
            if keep < len(hit.tail_hashes):
                # Part of the relayed hit was evicted while the request queued:
                # leave it waiting, and its next match admits what is left; the
                # missing pages are recomputed, not failed reads.
                req.linker_pp_hit = (anchor, [stop for stop in stops if stop <= end])
                return None

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
                self.cache_linker.unpin(req.rid)
                self._update_load(
                    ExternalLinkerLoadPhase.ABORT,
                    req,
                    component_transfers,
                    prefix_len,
                )
                return 0, req.last_node
            component_transfers.append((component, transfer))

        self.admission_marks["built"] = time.monotonic()
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
            self.cache_linker.unpin(req.rid)
            self._update_load(
                ExternalLinkerLoadPhase.ABORT, req, component_transfers, prefix_len
            )
            raise
        if not queued:
            self.cache_linker.unpin(req.rid)
            self._update_load(
                ExternalLinkerLoadPhase.ABORT, req, component_transfers, prefix_len
            )
            return 0, req.last_node
        self.admission_marks["queued"] = time.monotonic()
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
            if req.kv is None:
                from sglang.srt.managers.schedule_batch import ReqKvInfo

                req.kv = ReqKvInfo(kv_allocated_len=prefix_len)
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
        if self.insert_at_release:
            req.linker_insert_at_release = True
        return len(full_transfer.device_indices), req.last_node

    def _update_load(
        self,
        phase: ExternalLinkerLoadPhase,
        req: Req,
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
            if transfers is not None:
                if self._skip_swa:
                    transfers = [t for t in transfers if t.name != PoolName.SWA]
                self._offload_node(node_id, transfers)

    def _offload_node(self, node_id: NodeId, transfers: list[PoolTransfer]) -> None:
        cache = self.cache
        lock_params = cache.inc_lock_ref(node_id).to_dec_params()
        path = {}
        if getattr(self.cache_linker, "state_path_cap", 0):
            path["path_hashes"] = lambda: cache.tree_core.get_prefix_hash_values(node_id)
        try:
            queued = self.cache_linker.offload(
                transfers, session_id=getattr(cache, "insert_session_id", None), **path
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

    def wait_completed_offloads(self, count: int, timeout_s: float = 600.0) -> None:
        """Block until this rank has ``count`` finished offloads to take: a
        later PP stage commits PP0's count, and its own puts of the same nodes
        were issued at the same logical step, so they land within milliseconds."""
        deadline = time.monotonic() + timeout_s
        while self.cache_linker.num_completed_offloads() < count:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"External linker offloads stalled: {count} committed on PP0, "
                    f"{self.cache_linker.num_completed_offloads()} finished here."
                )
            time.sleep(0.0005)

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
        if self.fail_load_prob and random.random() < self.fail_load_prob:
            landed[0] = 0
        self.cache._all_reduce_attn_groups(landed, torch.distributed.ReduceOp.MIN)
        for rid in rids:
            self.private_load_indices.pop(rid, None)
        if landed.item():
            logger.info(
                "External linker stage load completed: requests=%d rids=%s",
                len(rids),
                rids,
            )
            return []
        logger.error(
            "External linker load failed; aborting %d request(s): %s", len(rids), rids
        )
        return rids

    # ---- lifecycle ----

    def reset(self) -> None:
        self.cache_linker.reset()
        self.hit_markers.clear()
        self.lookup_cache.clear()
        self.tail_hash_cache.clear()
        self.pp_lookup_time.clear()
        self.pp_forced_relookup.clear()
        self.inflight_load_rids.clear()
        self.private_load_indices.clear()
        self.inflight_load_index = -1
        self._release_pending_offloads()

    def _release_pending_offloads(self) -> None:
        for pending in self.pending_offloads:
            self.cache.tree_core.finish_external_linker_offload(
                pending.publish_node_ids, pending.lock_node_id, False
            )
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)
        self.pending_offloads.clear()

    def release_request(self, rid: str) -> None:
        self.hit_markers.pop(rid, None)
        self.skip_counts.pop(rid, None)
        self.lookup_cache.pop(rid, None)
        self.tail_hash_cache.pop(rid, None)
        self.pp_lookup_time.pop(rid, None)
        self.pp_forced_relookup.discard(rid)
        # A queued load targets slots the request owns and is about to free.
        # A started one is finished by finish_loads before anything is freed.
        if rid in self.inflight_load_rids and self.cache_linker.cancel_queued_load(rid):
            self.inflight_load_rids.remove(rid)
            self.private_load_indices.pop(rid, None)

    def close(self) -> None:
        self.cache_linker.close()
        self.inflight_load_rids.clear()
        self.private_load_indices.clear()
        self._release_pending_offloads()
