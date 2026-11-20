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
from dataclasses import dataclass
import random
import time
import zlib
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
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
from sglang.srt.mem_cache.unified_cache.pp_linker_relay import PPLinkerRelay
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.srt.session.streaming_session import is_virtual_node
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

    def prefetch(self, transfers: list[PoolTransfer]) -> Optional[Future]:
        """SGLANG_LINKER_LOAD_MODE=prefetch: read all layers of ``transfers``
        into their device slots. The future (True when every byte landed)
        completes after all device writes stop; None when nothing was queued."""
        return None

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

    def trace_hole(
        self, rid: str, transfers: list[PoolTransfer], keep: int, looked_up
    ) -> None:
        """SGLANG_LINKER_HOLE_TRACE: every stage, as admission finds objects
        of ``rid``'s hit gone past page ``keep``; ``looked_up``: PP0's lookup
        (wall clock), if known. Optional."""

    def start_prepin(
        self, rid: str, transfers: list[PoolTransfer], urgent: bool = False
    ) -> None:
        """PP0: begin pin_hit for a waiting request off the scheduler thread;
        ``urgent``: a hit skipped too often, pinned ahead of the window."""

    def peek_prepin(self, rid: str, transfers: list[PoolTransfer]) -> Optional[int]:
        """PP0: take_prepin's result without taking the pin."""
        return None

    def take_prepin(self, rid: str, transfers: list[PoolTransfer]) -> Optional[int]:
        """PP0: the result of a finished start_prepin of this hit, else None."""
        return None

    def release_stale_prepins(self) -> None:
        """PP0: release pre-pins whose requests were not admitted in time."""

    def keep_prepins(self, rids: Sequence[str], urgent: Sequence[str] = ()) -> None:
        """PP0, async admission: end the pre-pins of requests not in ``rids``
        and renew the kept ones before their sessions lapse; ``urgent`` ones
        (in ``rids``) are held as they are until admission takes them."""

    @abstractmethod
    def reset(self) -> None:
        """Quiesce all transfers and reset backend state before returning."""

    @abstractmethod
    def close(self) -> None:
        """Quiesce all transfers and release backend resources."""


@dataclass
class LinkerAdmissionVerdicts:
    """SGLANG_LINKER_ASYNC_BATCHED_RELAY: PP0's admission verdicts for the next
    pass, rid -> (hit pages, pages its pre-pin holds). Rides the request relay
    PP0 already forwards to every stage each round, so every stage applies the
    same map before its pass; a hit not in it is skipped everywhere."""

    ready: dict


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
    # Why this read was dropped locally (cancel reason or read failure).
    abort_reason: str | None = None


@dataclass
class _PendingLookup:
    """A queued request's store lookup running off the scheduler thread. Under
    PP only PP0 runs it (``future``); the other stages keep the key and wait
    for PP0's relayed result."""

    key: RadixKey
    hashes: list[str]
    start: int
    anchor_node_id: NodeId
    anchor_lock: DecLockRefParams
    future: Future[list[int]] | None
    timeout_s: float = 0.0
    wait_started: float | None = None
    kind: str = "enq"


class _PrefetchHold(NamedTuple):
    node_id: NodeId
    lock: DecLockRefParams
    tokens: int



class _PrefetchTimeline:
    """SGLANG_LINKER_PREFETCH_TIMELINE: per-request prefetch milestones on this rank
    (perf_counter), the interval percentiles between them, admission-pass skips of
    not-ready requests and the interval between passes that admitted a batch."""

    PAIRS = (
        ("enqueue", "lookup_done"),
        ("lookup_done", "load_issued"),
        ("load_issued", "landed"),
        ("landed", "finish_issued"),
        ("finish_issued", "ready"),
        ("enqueue", "ready"),
        ("ready", "admitted"),
        ("enqueue", "read_start"),
        ("read_start", "landed"),
        ("landed", "voted"),
    )
    # Milestones of one request in 32, chosen by rid (the same requests on every
    # stage), logged in wall-clock seconds so stages can be aligned by rid.
    WALL = ("enqueue", "read_start", "landed", "voted", "finish_issued", "ready", "admitted")

    def __init__(self):
        self.wall_offset = time.time() - time.perf_counter()
        self.marks: dict = {}
        self.intervals = {pair: [] for pair in self.PAIRS}
        self.pass_skips: list[int] = []
        self.skip_positions: list[int] = []
        self.batch_times: list[float] = []
        self._pass_t = None
        self._pass_pos = 0
        self._pass_skipped = 0
        self._pass_admitted = 0

    def mark(self, request, stage: str, t: Optional[float] = None) -> None:
        self.marks.setdefault(request, {}).setdefault(
            stage, time.perf_counter() if t is None else t
        )

    def admitted(self, request) -> None:
        self._pass_admitted += 1
        marks = self.marks.pop(request, None)
        if not marks:
            return
        marks.setdefault("admitted", time.perf_counter())
        for a, b in self.PAIRS:
            if a in marks and b in marks:
                self.intervals[(a, b)].append(1e3 * (marks[b] - marks[a]))
        rid = getattr(request, "rid", None)
        if rid is not None and zlib.crc32(str(rid).encode()) % 32 == 0:
            logger.info(
                "Linker prefetch wall rid=%s %s", rid,
                " ".join(f"{k}={marks[k] + self.wall_offset:.3f}" for k in self.WALL if k in marks),
            )

    def drop(self, request) -> None:
        self.marks.pop(request, None)

    def pass_begin(self) -> None:
        if self._pass_t is not None:
            self.pass_skips.append(self._pass_skipped)
            if self._pass_admitted:
                self.batch_times.append(self._pass_t)
        self._pass_t = time.perf_counter()
        self._pass_pos = self._pass_skipped = self._pass_admitted = 0

    def candidate(self, ready: bool) -> None:
        if not ready:
            self._pass_skipped += 1
            self.skip_positions.append(self._pass_pos)
        self._pass_pos += 1

    @staticmethod
    def _pct(values, q):
        return values[min(len(values) - 1, int(len(values) * q))] if values else 0.0

    def summary(self) -> str:
        parts = []
        for (a, b), values in self.intervals.items():
            values.sort()
            parts.append(
                f"{a}->{b} n={len(values)} p50 {self._pct(values, 0.5):.0f} "
                f"p90 {self._pct(values, 0.9):.0f} max {values[-1] if values else 0:.0f}"
            )
            values.clear()
        gaps = sorted(
            1e3 * (t1 - t0) for t0, t1 in zip(self.batch_times, self.batch_times[1:])
        )
        skips = sorted(self.pass_skips)
        positions = sorted(self.skip_positions)
        line = (
            "; ".join(parts)
            + f"; batch interval n={len(gaps)} mean "
            + f"{(sum(gaps) / len(gaps) if gaps else 0.0):.0f} ms"
            + f"; not-ready skips per pass p50 {self._pct(skips, 0.5)} p90 {self._pct(skips, 0.9)}"
            + f"; skipped queue position p50 {self._pct(positions, 0.5)}"
        )
        self.batch_times = self.batch_times[-1:]
        self.pass_skips.clear()
        self.skip_positions.clear()
        return line

class UnifiedCacheLinkerWrapper:
    """Drives an external KV store on behalf of one :class:`UnifiedRadixCache`."""
    # SGLANG_LINKER_PREFETCH_TIMELINE recorder (None: off); class default for test doubles.
    timeline: Optional["_PrefetchTimeline"] = None
    # SGLANG_LINKER_EARLY_READS (None: off): this stage's EarlyReads and its slab.
    early = None
    _early_slab = None
    # Finish-path waits (ms) for staged loads still being queued, since the last stats line.
    _finish_wait_ms: tuple = ()
    _staged_fallbacks: Counter | None = None

    # Async admission: the relayed verdict of a skipped hit, and the passes in a
    # row after which its pre-pin goes ahead of the window and is held until
    # admission takes it (no starvation; PP0 still never pins at admission).
    _SKIPPED = -1
    _SKIP_LIMIT = 3
    # Progress guarantee: the relayed verdict of a hit admitted as a miss of
    # its unready pages (SGLANG_LINKER_SKIP_FALLBACK_*).
    _FALLBACK = -2

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
        # row its pre-pin was not ready; from _SKIP_LIMIT on its pin is urgent.
        self.async_admission = bool(
            self.prepin_depth and envs.SGLANG_LINKER_ASYNC_ADMISSION.get()
        )
        self.skip_counts: dict[str, int] = {}
        self.async_counts = dict.fromkeys(
            (
                "skipped",
                "urgent",
                "fallback",
                "fallback_pages",
                "hole_fallback",
                "hole_fallback_pages",
            ),
            0,
        )
        self.hole_fallback = envs.SGLANG_LINKER_HOLE_FALLBACK.get()
        # PP0: rid -> when its current run of skips began; a run of
        # fallback_passes skips or fallback_s seconds recomputes the hit.
        self.skip_since: dict[str, float] = {}
        self.fallback_passes = max(0, envs.SGLANG_LINKER_SKIP_FALLBACK_PASSES.get())
        self.fallback_s = max(0.0, envs.SGLANG_LINKER_SKIP_FALLBACK_MS.get()) / 1e3
        # Batched relay: the verdict map every stage applies this pass, and the
        # one PP0 sends with the next round's requests.
        self.batched_relay = bool(
            self.async_admission and envs.SGLANG_LINKER_ASYNC_BATCHED_RELAY.get()
        )
        self.verdicts: dict = {}
        self.next_verdicts: Optional[dict] = None
        self.window_min = max(1, envs.SGLANG_LINKER_PREPIN_WINDOW_MIN.get())
        # Hits admitted in the current pass, and their moving average per pass.
        self.pass_hits = 0
        self.hits_per_pass = 0.0
        self.admission_rpc_s = 0.0
        self.admission_marks: dict[str, float] = {}
        self.pp_load_holes = 0
        self._init_prefetch_state()

        cache.tree_core.enable_external_cache_linker = True
        cache.write_through_threshold = 1

    def _init_prefetch_state(self) -> None:
        """SGLANG_LINKER_LOAD_MODE=prefetch, ported from GLM's linker: lookup at
        enqueue, a read into request-private slots, publication into the tree
        once every rank (under PP: every stage) landed it. Otherwise inert."""
        cache = self.cache
        prefetch = envs.SGLANG_LINKER_LOAD_MODE.get() == "prefetch"
        self.load_mode = "prefetch" if prefetch else "layer_wise"
        self.pp_relay = None
        # Reads in flight into slots private to their queued request.
        self.prefetches: dict[CacheRequestHandle, _ExternalCachePrefetch] = {}
        # Store lookups in flight on worker threads, in submission order.
        self.lookups: dict[CacheRequestHandle, _PendingLookup] = {}
        self._draining_lookups = False
        self._lookup_executor = None
        # Published prefetches still locked for their waiting request.
        self.holds: dict[CacheRequestHandle, _PrefetchHold] = {}
        self.held_tokens = 0
        self.hold_cap_tokens = 0
        self.load_cap_tokens = 0
        self.max_prefetch_states = 0
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
        self.requery_on_device_loss = False
        self.match_targets: dict[CacheRequestHandle, int] = {}
        self.log_lookup_stats = False
        self._lookup_stats = dict.fromkeys(
            (
                "issued",
                "skipped_no_hash",
                "zero_restorable",
                "requeried",
                "load_capped",
                "state_capped",
                "published",
                "aborted",
                "timeout_load",
                "timeout_lookup",
                "requery_capped",
            ),
            0,
        )
        # Aborted reads by reason (this rank's cancel reason, read failure, or
        # "remote_vote" when another stage failed it), logged every 60 s.
        self._abort_reasons: Counter[str] = Counter()
        # Re-queries per request rid, capped by SGLANG_LINKER_MAX_REQUERIES.
        self._requery_counts: Counter[str] = Counter()
        self.max_requeries = 0
        self._pending_requeries: dict[CacheRequestHandle, Req] = {}
        self._occupied_peak = 0
        self._lookup_next_log = 0.0
        self._requery_started: dict[CacheRequestHandle, float] = {}
        self._requery_waits: list[float] = []
        self._requery_records: list[tuple[str, float, str]] = []
        self._lookup_kind = "enq"
        self._phase_s = {
            k: dict.fromkeys(("n", "match", "keys", "rpc", "load"), 0.0)
            for k in ("enq", "rq")
        }
        if not prefetch:
            return
        if ComponentType.SWA in cache.components:
            # GLM binds the SWA window of a published read (bind_window); this
            # tree has no such API, and K3.1 has no SWA.
            raise ValueError("External linker prefetch mode does not support SWA.")
        if getattr(cache.req_to_token_pool, "mamba_ckpt_pool", None) is not None:
            raise ValueError(
                "External linker prefetch mode publishes recurrent states into the "
                "active state pool; int8 state checkpoints are not supported."
            )
        hold_fraction = envs.SGLANG_EXTERNAL_LINKER_PREFETCH_HOLD_FRACTION.get()
        lookup_threads = envs.SGLANG_EXTERNAL_LINKER_ASYNC_LOOKUP_THREADS.get()
        if self.pp_size > 1:
            if cache.tp_world_size != 1:
                raise ValueError("External linker PP prefetch is qualified for TP1 only.")
            if hold_fraction <= 0:
                raise ValueError("External linker PP prefetch requires a hold on every stage.")
            # PP0 looks up every stage's objects; it must not block its scheduler.
            lookup_threads = max(1, lookup_threads) if self.pp_rank == 0 else 0
            self.pp_relay = PPLinkerRelay(self)
        if lookup_threads > 0:
            self._lookup_executor = ThreadPoolExecutor(
                lookup_threads, thread_name_prefix="linker-lookup"
            )
        size_full = cache.token_to_kv_pool_allocator.size_full
        if hold_fraction > 0:
            self.hold_cap_tokens = int(hold_fraction * size_full)
        load_fraction = envs.SGLANG_EXTERNAL_LINKER_LOAD_FRACTION.get()
        if load_fraction > 0:
            self.load_cap_tokens = int(load_fraction * size_full)
        if ComponentType.MAMBA in cache.components:
            self.max_prefetch_states = max(
                0, envs.SGLANG_LINKER_PREFETCH_MAX_STATES.get()
            )
        self.requery_on_device_loss = (
            envs.SGLANG_EXTERNAL_LINKER_REQUERY_ON_DEVICE_LOSS.get()
        )
        self.log_lookup_stats = envs.SGLANG_EXTERNAL_LINKER_LOOKUP_STATS.get()
        self.timeline = (
            _PrefetchTimeline() if envs.SGLANG_LINKER_PREFETCH_TIMELINE.get() else None
        )
        self.max_requeries = envs.SGLANG_LINKER_MAX_REQUERIES.get()
        logger.info(
            "External linker prefetch mode: lookup threads %d, hold cap %d tokens, "
            "load cap %d tokens, state cap %d reads, PP relay %s",
            lookup_threads,
            self.hold_cap_tokens,
            self.load_cap_tokens,
            self.max_prefetch_states,
            self.pp_relay is not None,
        )

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
        if self.load_mode == "prefetch":
            # Prefetch is issued on enqueue; only published device KV is matchable.
            return result
        # A skipped hit keeps its marker until it is matched again.
        self.hit_markers.pop(req.rid, None)
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
        if self.batched_relay:
            return self._pp_verdict_hit_pages(req, hit, transfers, admitted)
        # Only PP0 pins: the relay below carries PP0's value, not a reduction.
        pinned = admitted
        if self.pp_rank == 0:
            taken = None
            if self.prepin_depth:
                taken = self.cache_linker.take_prepin(req.rid, transfers)
                self.prepin_counts["missed" if taken is None else "ready"] += 1
            if taken is None and self.async_admission and self._falls_back(req.rid):
                pinned = self._FALLBACK
            elif taken is None and self.async_admission:
                # Skip this pass rather than wait on the store; the relay
                # carries the skip, so every stage skips it too.
                pinned = self._SKIPPED
                urgent = self.skip_counts.get(req.rid, 0) + 1 >= self._SKIP_LIMIT
                self.async_counts["urgent"] += urgent
                self.cache_linker.start_prepin(req.rid, transfers, urgent=urgent)
            else:
                # Not ready: pin now. Without async admission, deferring would
                # end this pass's admission for every request behind it.
                pinned = (
                    self.cache_linker.pin_hit(req.rid, transfers)
                    if taken is None
                    else taken
                )
                self.admission_rpc_s = self.cache_linker.last_pin_rpc_s
        keep = torch.tensor([min(pinned, admitted)], dtype=torch.int64)
        self.cache._all_reduce(keep, torch.distributed.ReduceOp.MIN)
        if int(keep[0]) == self._FALLBACK:
            return self._fall_back(req, admitted)
        if int(keep[0]) == self._SKIPPED:
            # A TP peer of PP0 that took its pre-pin drops it with the rest.
            self.cache_linker.unpin(req.rid)
            self._count_skip(req.rid)
            return self._SKIPPED
        self.skip_counts.pop(req.rid, None)
        self.skip_since.pop(req.rid, None)
        self.pass_hits += 1
        keep = int(keep[0])
        if keep < admitted:
            # Every stage drops its pins; the next pass pins the shorter hit.
            keep = self._pp_hole(req, transfers, keep, admitted)
        self._maybe_log_pp_counts()
        return keep

    def _pp_hole(self, req: Req, transfers, keep: int, admitted: int) -> int:
        """Every stage: PP0's pin found objects of the hit gone past page
        ``keep``. Returns the pages still restorable (below ``admitted``)."""
        self.cache_linker.unpin(req.rid)
        self.pp_load_holes += 1
        looked = self.pp_lookup_time.get(req.rid)
        self.cache_linker.trace_hole(
            req.rid,
            transfers,
            keep,
            None if looked is None else time.time() - (time.monotonic() - looked),
        )
        if self.hole_fallback:
            # _load_back admits it as a miss; the shorter hit needs no lookup.
            logger.info(
                "Linker PP admission found a hole: rid=%s, objects up to page %d of "
                "%d; admitted as a miss",
                req.rid,
                keep,
                admitted,
            )
            return keep
        best = torch.tensor([admitted], dtype=torch.int64)
        if self.pp_rank == 0:
            pages = self.cache_linker.lookup_stages(req.rid, transfers)
            best[0] = max((p for p in pages if p <= keep), default=0)
        self.cache._all_reduce(best, torch.distributed.ReduceOp.MIN)
        logger.info(
            "Linker PP admission found a hole: rid=%s, objects up to page %d of "
            "%d; loading %d",
            req.rid,
            keep,
            admitted,
            int(best[0]),
        )
        return int(best[0])

    def _pp_verdict_hit_pages(self, req: Req, hit, transfers, admitted: int) -> int:
        """Batched relay: admit or skip from the verdict map PP0 relayed with
        this round's requests; no PP relay per hit. Only a hole (a pre-pin
        that found an object gone) still relays PP0's re-lookup."""
        verdict = self.verdicts.pop(req.rid, None)
        if verdict == (admitted, self._FALLBACK):
            return self._fall_back(req, admitted)
        if verdict is None or verdict[0] != admitted:
            if self.pp_rank == 0:
                urgent = self.skip_counts.get(req.rid, 0) + 1 >= self._SKIP_LIMIT
                self.async_counts["urgent"] += urgent
                self.cache_linker.start_prepin(req.rid, transfers, urgent=urgent)
            self._count_skip(req.rid)
            return self._SKIPPED
        self.skip_counts.pop(req.rid, None)
        self.skip_since.pop(req.rid, None)
        self.pass_hits += 1
        keep = min(verdict[1], admitted)
        if self.pp_rank == 0:
            taken = self.cache_linker.take_prepin(req.rid, transfers)
            self.prepin_counts["ready" if taken is not None else "missed"] += 1
            if taken is None:
                # The verdict stands on every stage; the loads start their own
                # sessions. Counted, not expected.
                self.async_counts["verdict_unpinned"] = (
                    self.async_counts.get("verdict_unpinned", 0) + 1
                )
        if keep < admitted:
            keep = self._pp_hole(req, transfers, keep, admitted)
        self._maybe_log_pp_counts()
        return keep

    def _count_skip(self, rid: str) -> None:
        self.skip_counts[rid] = self.skip_counts.get(rid, 0) + 1
        self.skip_since.setdefault(rid, time.monotonic())
        self.async_counts["skipped"] += 1
        self._maybe_log_pp_counts()

    def _falls_back(self, rid: str) -> bool:
        """PP0: whether a skipped hit whose pin is still not ready has waited
        out the progress bound and is admitted as a miss instead."""
        if rid not in self.skip_counts:
            return False
        if self.fallback_passes and self.skip_counts[rid] >= self.fallback_passes:
            return True
        return bool(
            self.fallback_s
            and time.monotonic() - self.skip_since[rid] >= self.fallback_s
        )

    def _fall_back(self, req: Req, admitted: int) -> int:
        """Every stage, from PP0's verdict: drop the pins and recompute the hit."""
        self.cache_linker.unpin(req.rid)
        self.skip_counts.pop(req.rid, None)
        self.skip_since.pop(req.rid, None)
        self.async_counts["fallback"] += 1
        self.async_counts["fallback_pages"] += admitted
        self._maybe_log_pp_counts()
        return self._FALLBACK

    def take_verdicts(self) -> Optional[LinkerAdmissionVerdicts]:
        """PP0, as the next round's requests are relayed: this pass's verdicts."""
        if not self.batched_relay:
            return None
        ready, self.next_verdicts = self.next_verdicts or {}, None
        return LinkerAdmissionVerdicts(ready=ready)

    def apply_verdicts(self, verdicts: LinkerAdmissionVerdicts) -> None:
        """Every stage, PP0 too, before its next pass."""
        self.verdicts = dict(verdicts.ready)

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
        depth = min(
            self.prepin_depth,
            max(self.window_min, math.ceil(2 * self.hits_per_pass) + 1),
        )
        # Hits skipped _SKIP_LIMIT passes in a row hold their pins outside the
        # window's count, so a deep queue cannot push them out.
        urgent = [
            rid for rid, n in self.skip_counts.items() if n >= self._SKIP_LIMIT
        ]
        window = list(urgent)
        taken = 0
        for req in waiting:
            if taken >= depth:
                break
            hit = self.hit_markers.get(req.rid)
            if hit is None or req.rid in urgent:
                continue
            transfers = self._lookup_transfers(hit.tail_hashes)
            if transfers:
                self.cache_linker.start_prepin(req.rid, transfers)
                window.append(req.rid)
                taken += 1
        self.cache_linker.keep_prepins(window, urgent)
        if self.batched_relay:
            # Hits whose pre-pin has landed are admitted next pass on every
            # stage; skipped hits past the progress bound are recomputed.
            ready = {}
            for req in waiting:
                hit = self.hit_markers.get(req.rid)
                if hit is None or (
                    req.rid not in window and req.rid not in self.skip_counts
                ):
                    continue
                transfers = self._lookup_transfers(hit.tail_hashes)
                pages = self.cache_linker.peek_prepin(req.rid, transfers) if transfers else None
                if pages is not None:
                    ready[req.rid] = (len(hit.tail_hashes), pages)
                elif self._falls_back(req.rid):
                    ready[req.rid] = (len(hit.tail_hashes), self._FALLBACK)
            self.next_verdicts = ready

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

    # ---- prefetch (SGLANG_LINKER_LOAD_MODE=prefetch, ported from GLM) ----

    def prefetch(self, req: Req) -> None:
        """Stage a queued request; publish only after all ranks succeed."""
        if self.load_mode != "prefetch":
            return
        request = req.cache_request_handle
        if self.pp_relay is not None:
            self.pp_relay.begin(req)
        for previous in [p for p in self.prefetches if p.rid == request.rid]:
            if previous != request:
                self._cancel_prefetch(previous, "superseded")
        for previous in [lk for lk in self.lookups if lk.rid == request.rid]:
            if previous != request:
                self._cancel_lookup(previous)
        for previous in [h for h in self.holds if h.rid == request.rid]:
            if previous != request:
                self.release_hold(previous, admitted=False)
        if self.cache.disable or request in self.prefetches or request in self.lookups:
            return
        if not self._stage_lookup(req, request) and self.pp_relay is not None:
            # Every stage made the same decision from the same tree: nothing to load.
            self.pp_relay.settle(request)

    def _stage_lookup(self, req: Req, request: CacheRequestHandle) -> bool:
        """GLM's prefetch body after its guards; True when a lookup (or under
        PP, on stages past PP0, its placeholder) was queued."""
        timed = self.log_lookup_stats
        kind = self._lookup_kind
        t0 = time.perf_counter() if timed else 0.0
        req.init_next_round_input(self.cache, cow_mamba=False)
        t1 = time.perf_counter() if timed else 0.0
        if is_virtual_node(req.last_node) or req.positional_embed_overrides is not None:
            return False
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
            return False
        if start and last_hash is None:
            self._lookup_stats["skipped_no_hash"] += 1
            return False
        hashes = get_storage_hash_str(key[start:], last_hash, page_size=page)
        lookups = [
            c.build_external_linker_transfer(LinkerTransferPhase.LOOKUP, None, hashes)
            for c in self._components
        ]
        if any(t is None for t in lookups):
            return False
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
        if self._lookup_executor is not None or self.pp_relay is not None:
            # The anchor stays locked while the lookup runs, as for a load.
            future = (
                self._lookup_executor.submit(
                    self._timed_lookup, kind if timed else None, req.rid, lookups
                )
                if self._lookup_executor is not None
                else None
            )
            if self.timeline is not None:
                self.timeline.mark(request, "enqueue")
                if future is not None:
                    timeline = self.timeline
                    future.add_done_callback(
                        lambda _f, r=request: timeline.mark(r, "lookup_done")
                    )
            self.lookups[request] = _PendingLookup(
                key,
                hashes,
                start,
                req.last_node,
                self.cache.inc_lock_ref(req.last_node).to_dec_params(),
                future,
                timeout_s=timeout_s,
                kind=kind,
            )
            return True
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
        return True

    def _store_lookup(self, rid: str, lookups: list[PoolTransfer]) -> list[int]:
        # Under PP, PP0 looks up every stage's objects at once (lookup_stages).
        if self.pp_size > 1:
            return self.cache_linker.lookup_stages(rid, lookups)
        return self.cache_linker.lookup(rid, lookups)

    def _timed_lookup(self, kind, rid, lookups):
        """The store lookup; with ``kind``, its time joins that kind's RPC phase (any thread)."""
        if kind is None:
            return self._store_lookup(rid, lookups)
        t0 = time.perf_counter()
        try:
            return self._store_lookup(rid, lookups)
        finally:
            self._phase_s[kind]["rpc"] += time.perf_counter() - t0

    def _prefetch_states(self) -> int:
        """In-flight reads holding a recurrent-state slot."""
        return sum(
            1
            for state in self.prefetches.values()
            for _, transfer in state.component_transfers
            if transfer.name == PoolName.MAMBA and transfer.device_indices is not None
        )

    def attach_early_reads(self) -> None:
        """SGLANG_LINKER_EARLY_READS: every PP stage, at its PP loop start (the
        channel's address goes out over the PP group, a collective)."""
        if not envs.SGLANG_LINKER_EARLY_READS.get() or self.pp_relay is None:
            return
        linker = self.cache_linker
        if not hasattr(linker, "prefetch_from_staging") or self.cache.tp_world_size != 1:
            logger.warning("SGLANG_LINKER_EARLY_READS needs the Mooncake direct linker and one TP rank per stage; off")
            return
        from sglang.srt.mem_cache.storage.mooncake_store.linker_staging import EarlyChannel, EarlyReads
        from sglang.srt.runtime_context import get_parallel

        slab = linker.staging_slab(
            envs.SGLANG_LINKER_EARLY_GB.get(),
            envs.SGLANG_LINKER_EARLY_STATES.get(),
            envs.SGLANG_LINKER_EARLY_PAGES.get(),
            envs.SGLANG_LINKER_EARLY_NODE_GB.get(),
        )
        early = EarlyReads(slab, self._early_plan, linker.early_read)
        channel = EarlyChannel(self.pp_rank, get_parallel().pp_group, on_message=early.start)
        if self.pp_rank == 0:
            early.send = channel.send
        self._early_slab, self._early_channel, self.early = slab, channel, early
        self._finish_wait_ms = []
        self._staged_fallbacks = Counter()

    def _staged_load_failed(self, request, state, publish: bool) -> bool:
        """A staged load failed after its early vote counted, so every stage
        publishes the request while this stage's slots were never filled: read them
        from the store now (the scheduler waits), or raise under
        SGLANG_LINKER_EARLY_STRICT. Returns whether this stage may publish."""
        error = state.future.exception()
        if envs.SGLANG_LINKER_EARLY_STRICT.get():
            raise RuntimeError(f"staged load of {request.rid} failed after its vote") from error
        logger.warning("Linker staged load of %s failed (%r); reading it from the store", request.rid, error)
        self._staged_fallbacks["fallback"] += 1
        transfers = [t for _, t in state.component_transfers]
        for _ in range(2):
            future = self.cache_linker.prefetch(transfers)
            if future is not None and future.result(timeout=120) is True:
                state.future = future
                return publish
        # Not publishing desyncs this stage's tree from the others; publishing would
        # serve unfilled slots. Neither is safe: counted and logged loudly.
        self._staged_fallbacks["fallback_failed"] += 1
        state.abort_reason = "staged_fallback_failed"
        logger.error("Linker staged load of %s: the store read failed too; not published on this stage", request.rid)
        return False

    def _early_plan(self, hashes: list[str]) -> dict:
        transfers = [
            c.build_external_linker_transfer(LinkerTransferPhase.LOOKUP, None, hashes)
            for c in self._components
        ]
        return self.cache_linker.early_objects(transfers)

    def _start_prefetch_from_staging(self, request, *args, staged) -> None:
        """The relay's load with a landed early read: as _start_prefetch_load, the
        private slots filled from staging. A load refused here (caps) is refused on
        every stage at this pass; its staging is freed."""
        self._start_prefetch_load(request, *args, staged=staged)
        state = self.prefetches.get(request)
        if state is None or state.future is None:
            self.early.release(staged)

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
        staged=None,
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
                self.cache.dec_lock_ref(anchor_node_id, anchor_lock)
                return
            self._occupied_peak = max(self._occupied_peak, occupied + pages * page)
        if self.max_prefetch_states and self._prefetch_states() >= self.max_prefetch_states:
            # Each read holds one recurrent-state slot per stage until it is
            # published; the pool has a few hundred. The request recomputes.
            self._lookup_stats["state_capped"] += 1
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
        if self.timeline is not None:
            self.timeline.mark(request, "read_start")
        try:
            for component in self._components:
                transfer = component.build_external_linker_transfer(
                    LinkerTransferPhase.LOAD, None, hashes[:pages]
                )
                if transfer is None:
                    break
                state.component_transfers.append((component, transfer))
            if len(state.component_transfers) == len(self._components):
                transfers = [t for _, t in state.component_transfers]
                if staged is not None:
                    state.future = self.cache_linker.prefetch_from_staging(
                        transfers, staged, self._early_slab, self.early.release
                    )
                else:
                    state.future = self.cache_linker.prefetch(transfers)
                if self.timeline is not None and state.future is not None:
                    timeline = self.timeline
                    state.future.add_done_callback(
                        lambda _f, r=request: timeline.mark(r, "landed")
                    )
        except Exception:
            logger.warning("External linker prefetch submission failed", exc_info=True)
        accepted = torch.tensor([state.future is not None], dtype=torch.int)
        self.cache._all_reduce_attn_groups(accepted, torch.distributed.ReduceOp.MIN)
        if not accepted.item():
            self._cancel_prefetch(request, "not_submitted")
            # A lookup drain is inside check_prefetch_events, which settles the
            # cancelled read right after it; re-entering would rerun the drain.
            if not self._draining_lookups:
                self.check_prefetch_events(drain_lookups=False)

    def _load_cap(self) -> int:
        """Tokens loads may hold in flight: their own cap, else the hold cap they share."""
        return self.load_cap_tokens or self.hold_cap_tokens

    def has_pending_prefetch(self) -> bool:
        return bool(self.prefetches or self.lookups)

    def check_prefetch_progress(self, request: CacheRequestHandle) -> bool:
        if self.pp_relay is not None:
            ready = self.pp_relay.progress(request, self._stop_due)
        else:
            ready = self._local_check_prefetch_progress(request)
        if self.timeline is not None:
            self.timeline.candidate(ready)
        return ready

    def admission_pass_begin(self) -> None:
        """SGLANG_LINKER_PREFETCH_TIMELINE: a prefill admission pass starts."""
        if self.timeline is not None:
            self.timeline.pass_begin()

    def _stop_due(self, request: CacheRequestHandle) -> bool:
        """PP0 under PP: whether the stop policy ends the wait for ``request``'s
        lookup or read now (GLM's timeout policy, without its local cancel: the
        relay orders the stop on every stage)."""
        policy = self.cache.prefetch_stop_policy
        if policy != "timeout":
            return False
        pending = self.lookups.get(request) or self.prefetches.get(request)
        if pending is None:
            return False
        now = time.perf_counter()
        if pending.wait_started is None:
            pending.wait_started = now
            return False
        return now - pending.wait_started > pending.timeout_s

    def _local_check_prefetch_progress(self, request: CacheRequestHandle) -> bool:
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
            self._lookup_stats["timeout_load"] += 1
        # Consume any fully completed read before stopping. Running reads keep
        # their private slots until every rank has stopped writing to them.
        # Admission polls this: start no new load here.
        self.check_prefetch_events(drain_lookups=False)
        self._cancel_prefetch(request, "timeout")
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
            self._lookup_stats["timeout_lookup"] += 1
        self._cancel_lookup(request)
        return True

    def requery_after_device_loss(self, req: Req) -> bool:
        if self.load_mode != "prefetch":
            return False
        # Under PP every stage reaches the same decision from the same tree and
        # targets, so no verdict is exchanged.
        return self._local_requery_after_device_loss(req)

    def _local_requery_after_device_loss(self, req: Req) -> bool:
        """Look up again the prefix ``req`` lost on device since its last lookup.

        True while the re-query waits for the next event-loop boundary, where
        its lookup starts; the request then waits on that lookup and its load.
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
        if request in self._pending_requeries:
            return True
        if self.max_requeries and self._requery_counts[request.rid] >= self.max_requeries:
            # Stop re-looking it up: admit with what is on device.
            self._lookup_stats["requery_capped"] += 1
            return False
        self._requery_counts[request.rid] += 1
        self._lookup_stats["requeried"] += 1
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
        if self.timeline is not None:
            logger.info("Linker prefetch timeline (ms): %s", self.timeline.summary())
            if self.early is not None:
                waits = sorted(self._finish_wait_ms)
                self._finish_wait_ms = []
                pick = lambda q: waits[min(len(waits) - 1, int(q * len(waits)))] if waits else 0.0
                logger.info("Linker early reads: %s, slab free %s; finish waits n=%d p50 %.1f p99 %.1f max %.1f ms; "
                            "staged loads read from the store %s",
                            dict(self.early.counts), {str(k): len(v) for k, v in self._early_slab.free.items()},
                            len(waits), pick(0.5), pick(0.99), waits[-1] if waits else 0.0, dict(self._staged_fallbacks))
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
            "of cap %d tokens; reads in flight %d (%d states)",
            self._lookup_stats,
            len(waits),
            p50,
            p90,
            "; ".join(phases),
            self._occupied_peak,
            self._load_cap(),
            len(self.prefetches),
            self._prefetch_states(),
        )
        self._occupied_peak = 0
        if self._requery_records:
            logger.info(
                "External linker re-queried rids (rid:wait_s:outcome): %s",
                " ".join(f"{rid}:{w:.3f}:{o}" for rid, w, o in self._requery_records),
            )
            self._requery_records.clear()
        if self._abort_reasons:
            logger.info(
                "Linker prefetch aborts by reason: %s", dict(self._abort_reasons)
            )

    def _cancel_lookup(self, request: CacheRequestHandle) -> None:
        lookup = self.lookups.pop(request, None)
        if lookup is not None:
            # A lookup reads store metadata only, so its result can be dropped.
            if lookup.future is not None:
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

    @staticmethod
    def local_abort_reason(state) -> str | None:
        """This rank's reason a finished read did not land, or None if it did."""
        if state is None:
            return "missing"
        if state.discard:
            return state.abort_reason or "discard"
        future = state.future
        if future is None:
            return "not_submitted"
        if not future.done():
            return None
        if future.cancelled():
            return "cancelled"
        error = future.exception()
        if error is not None:
            return f"error:{type(error).__name__}"
        result = future.result()
        if result is True:
            return None
        # The backend returns a reason string (read return codes) on failure.
        return result if isinstance(result, str) else "read_failed"

    def _cancel_prefetch(
        self, request: CacheRequestHandle, reason: str = "cancelled"
    ) -> None:
        self._cancel_lookup(request)
        state = self.prefetches.get(request)
        if state is not None:
            state.discard = True
            if state.abort_reason is None:
                state.abort_reason = reason
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
        if self.load_mode != "prefetch":
            return
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
            if done and not landed and state.abort_reason is None:
                state.abort_reason = self.local_abort_reason(state)
            progress.append([done, landed and not state.discard])
        # Keep the same operation order on every rank, including discarded
        # reads. Local cancellation must not change collective participation.
        verdicts = torch.tensor(progress, dtype=torch.int)
        self.cache._all_reduce_attn_groups(verdicts, torch.distributed.ReduceOp.MIN)
        for (request, _), (done, landed) in zip(pending, verdicts.tolist()):
            if done:
                self._finish_prefetch(request, publish=bool(landed))

    def _finish_prefetch(
        self, request: CacheRequestHandle, *, publish: bool, hold: bool = True
    ) -> None:
        """``hold``: False for a request already released (finished), whose
        published prefix nobody would unlock."""
        state = self.prefetches.pop(request, None)
        if state is None:
            return
        started = self._requery_started.pop(request, None)
        if started is not None:
            waited = time.perf_counter() - started
            self._requery_waits.append(waited)
            self._requery_records.append((request.rid, waited, "loaded" if publish else "aborted"))
        if state.future is not None and not state.future.done() and getattr(state.future, "staged", False):
            # SGLANG_LINKER_EARLY_READS: the relay's finish can apply before the
            # staged load's plan and scatter are queued (they wait for the slot
            # indices on the staging thread); wait for that, counted.
            waited = time.perf_counter()
            state.future.result(timeout=120)
            self._finish_wait_ms.append((time.perf_counter() - waited) * 1e3)
        # Event polling or reset/close has already quiesced every device writer.
        if state.future is not None:
            assert state.future.done()
            if getattr(state.future, "staged", False) and state.future.exception() is not None:
                publish = self._staged_load_failed(request, state, publish)
            copy_event = getattr(state.future, "copy_event", None)
            if copy_event is not None:
                # Staged load: the forward (or a reuse of freed slots) runs after the scatter.
                torch.cuda.current_stream().wait_event(copy_event)
        end = len(state.key)
        try:
            if not publish:
                self._lookup_stats["aborted"] += 1
                self._abort_reasons[state.abort_reason or self.local_abort_reason(state) or "remote_vote"] += 1
                self._update_load(
                    ExternalLinkerLoadPhase.ABORT, None, state.component_transfers, end
                )
                return
            by_pool = {t.name: t for _, t in state.component_transfers}
            full = by_pool[PoolName.KV].device_indices
            start = end - len(full)
            prefix = self.cache.path_device_indices(state.anchor_node_id)
            assert len(prefix) == start
            mamba = by_pool.get(PoolName.MAMBA)
            if mamba is not None:
                # A restored state has no pending replay updates (GLM: restart_replay).
                pool = self.cache.req_to_token_pool
                write_pos = pool.mamba_pool.replayssm_write_pos
                if write_pos is not None:
                    write_pos[pool.translate_mamba_indices(mamba.device_indices)] = 0
            # The normal insert protects the borrowed prefix and frees duplicate
            # fetched slots; inserted_len=end counts no hit, so the just-fetched
            # data is not written back to the store.
            inserted = self.cache.insert(
                InsertParams(
                    key=state.key,
                    value=torch.cat([prefix.to(full.dtype), full]),
                    mamba_value=mamba.device_indices if mamba is not None else None,
                    prev_prefix_len=start,
                    inserted_len=end,
                )
            )
            if mamba is not None and inserted.mamba_exist:
                self.cache.req_to_token_pool.mamba_allocator.free(mamba.device_indices)
            self._lookup_stats["published"] += 1
            if not hold:
                return
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
        if self.timeline is not None:
            if admitted:
                self.timeline.admitted(request)
            else:
                self.timeline.drop(request)
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
            "External linker prefetch holds: %d requests hold %d of %d tokens; %s; "
            "lookups %s",
            len(self.holds),
            self.held_tokens,
            self.hold_cap_tokens,
            self._hold_stats,
            self._lookup_stats,
        )

    def _release_prefetch_request(self, rid: str) -> None:
        """GLM's release_request for the prefetch state of ``rid``."""
        self._requery_counts.pop(rid, None)
        handles = {
            h
            for table in (self.prefetches, self.lookups, self.holds, self._pending_requeries)
            for h in table
            if h.rid == rid
        }
        if self.pp_relay is not None:
            handles.update(h for h in self.pp_relay.entries if h.rid == rid)
        for request in handles:
            self._pending_requeries.pop(request, None)
            self._requery_started.pop(request, None)
            self._cancel_prefetch(request, "released")
            self.release_hold(request, admitted=False)

    def _clear_prefetch_state(self) -> None:
        for request in list(self.prefetches):
            self._finish_prefetch(request, publish=False)
        for request in list(self.lookups):
            self._cancel_lookup(request)
        for request in list(self.holds):
            self.release_hold(request, admitted=False)
        self.unheld.clear()
        self.match_targets.clear()
        self._pending_requeries.clear()
        self._requery_started.clear()
        self._requery_records.clear()
        self._requery_counts.clear()
        if self.pp_relay is not None:
            self.pp_relay.reset()

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
                # Keep the marker: the next window gives verdicts only to hits
                # that hold one (pm-lk4: popped here, a skipped hit never got one).
                self.hit_markers[req.rid] = hit
                return LINKER_ADMISSION_SKIPPED
            if keep == self._FALLBACK:
                # Admitted as a miss on every stage: the adder re-selects this
                # request with no host hit and recomputes the unready pages.
                anchor, stops = req.linker_pp_hit
                stops = [stop for stop in stops if stop <= hit.device_hit_len]
                req.linker_pp_hit = (anchor, stops)
                return 0, req.last_node
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
                if self.hole_fallback:
                    # keep is reduced over every stage, so all admit it as a
                    # miss and the pass goes on to the requests behind it.
                    req.linker_pp_hit = (
                        anchor,
                        [stop for stop in stops if stop <= hit.device_hit_len],
                    )
                    self.async_counts["hole_fallback"] += 1
                    self.async_counts["hole_fallback_pages"] += len(hit.tail_hashes)
                    return 0, req.last_node
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
        if getattr(self.cache_linker, "state_path_cap", 0) or getattr(
            self.cache_linker, "store_tree", None
        ):
            path["path_hashes"] = lambda: cache.tree_core.get_prefix_hash_values(node_id)
        defer = getattr(cache, "ring_wait_work", None)
        if defer is not None:
            path["defer"] = defer
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
        if self.log_lookup_stats:
            tokens = self.__dict__.setdefault("pending_offload_tokens", {})
            if len(tokens) > 4 * len(self.pending_offloads) + 64:
                live = {p.lock_node_id for p in self.pending_offloads}
                for stale in [k for k in tokens if k not in live]:
                    del tokens[stale]
            tokens[node_id] = sum(
                len(t.device_indices)
                for t in transfers
                if t.name == PoolName.KV and t.device_indices is not None
            )

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

    def _run_deferred_offloads(self) -> None:
        """SGLANG_PP_RING_WAIT_WORK: queued offload tails go to the backend before it
        resets, so its queue and results stay in step with pending_offloads."""
        work = getattr(self.cache, "ring_wait_work", None)
        while work:
            work.popleft()()

    def reset(self) -> None:
        self._run_deferred_offloads()
        self.cache_linker.reset()
        if self.load_mode == "prefetch":
            self._clear_prefetch_state()
        self.hit_markers.clear()
        self.lookup_cache.clear()
        self.tail_hash_cache.clear()
        self.pp_lookup_time.clear()
        self.pp_forced_relookup.clear()
        self.inflight_load_rids.clear()
        self.private_load_indices.clear()
        self.inflight_load_index = -1
        self._release_pending_offloads()

    def refresh_flags(self) -> None:
        """Re-read the backend's per-call switches (POST /linker_flags)."""
        refresh = getattr(self.cache_linker, "refresh_flags", None)
        if refresh is not None:
            refresh()

    def flag_values(self) -> dict:
        values = getattr(self.cache_linker, "flag_values", None)
        return values() if values is not None else {}

    def reset_cold(self) -> None:
        """reset() plus the backend's store state and every stats counter, so
        the next request sees the linker as a fresh boot would."""
        self.reset()
        cold = getattr(self.cache_linker, "reset_cold", None)
        if cold is not None:
            cold()
        for table in (self._lookup_stats, self._hold_stats):
            for key in table:
                table[key] = 0
        for phase in self._phase_s.values():
            for key in phase:
                phase[key] = 0.0
        self._occupied_peak = 0
        self._requery_waits.clear()

    def _release_pending_offloads(self) -> None:
        for pending in self.pending_offloads:
            self.cache.tree_core.finish_external_linker_offload(
                pending.publish_node_ids, pending.lock_node_id, False
            )
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)
        self.pending_offloads.clear()

    def release_request(self, rid: str) -> None:
        if self.load_mode == "prefetch":
            self._release_prefetch_request(rid)
        self.hit_markers.pop(rid, None)
        self.skip_counts.pop(rid, None)
        self.skip_since.pop(rid, None)
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
        self._run_deferred_offloads()
        self.cache_linker.close()
        if self.load_mode == "prefetch":
            self._clear_prefetch_state()
            if self._lookup_executor is not None:
                self._lookup_executor.shutdown(wait=False, cancel_futures=True)
        self.inflight_load_rids.clear()
        self.private_load_indices.clear()
        self._release_pending_offloads()
