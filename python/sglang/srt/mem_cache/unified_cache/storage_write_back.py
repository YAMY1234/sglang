"""HiCache L2->L3 write-back (``SGLANG_OPT_HICACHE_STORAGE_WRITE_BACK``).

In cache mode the storage tier is normally written when a page enters the host
pool. Under write-back it is written when the page leaves it: each node a
device->host backup acknowledges is pinned with a host lock, so host eviction
passes over it. When an eviction falls short, the oldest pins are written to
storage and the eviction waits for enough of those acks before evicting again.
A page thus leaves the host pool only after storage acknowledged it, or after a
failed write that is counted as skipped.

Every decision reads rank-replicated state (the tree, the pin order, the issue
order of storage writes) and every ack the wait consumes is counted, so all TP
and PP ranks free the same pages at the same step.

At SIGTERM and at exit each process waits up to
``SGLANG_HICACHE_STORAGE_WRITE_BACK_DRAIN_TIMEOUT`` seconds for its storage
writes in flight, then logs one ``HiCache L3 write-back final stats`` line.

With ``SGLANG_HICACHE_STORAGE_WRITE_BACK_CLEANER`` (Rust TreeCore only) nothing
is pinned. Each acknowledged host copy is marked dirty in the tree core, and only
host-leaf eviction skips a dirty page: a dirty copy whose node is also on the
device stays reclaimable. A cleaner writes the coldest dirty pages, in the order
host eviction would take them, while unissued dirty pages exceed
``..._DIRTY_START_FRACTION`` of the host pool, within ``..._PREFETCH_BUDGET_FRACTION``
(written + in flight - host evicted) and ``..._MAX_INFLIGHT_FRACTION``. A page
becomes clean, so evictable, only at an ack all ranks agree succeeded; a failed
write leaves it dirty to be written again. Eviction takes clean pages first and
waits for acks only when none is left.
"""

from __future__ import annotations

import datetime
import logging
import os
import signal
import socket
import time
import weakref
from typing import TYPE_CHECKING, Optional

import msgspec

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import STORAGE_BATCH_SIZE, PoolHitPolicy
from sglang.srt.mem_cache.unified_cache.component_type import BASE_COMPONENT_TYPE
from sglang.srt.runtime_context import get_model

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        StorageOperation,
    )
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)

# Arbitrary; both only pace diagnostics.
_STATS_LOG_INTERVAL_S = 60.0
_ACK_WAIT_WARN_S = 30.0

# One SIGTERM handler per process finishes every live write-back.
_EXIT_INSTANCES: weakref.WeakSet[StorageWriteBack] = weakref.WeakSet()
_PREV_SIGTERM_HANDLER = None
_SIGTERM_HANDLER_INSTALLED = False


def _on_sigterm(signum, frame):
    for write_back in list(_EXIT_INSTANCES):
        try:
            write_back.finish(reason="SIGTERM")
        except Exception:
            logger.exception("HiCache L3 write-back: final stats failed.")
    prev = _PREV_SIGTERM_HANDLER
    if callable(prev):
        prev(signum, frame)
    elif prev != signal.SIG_IGN:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


def _install_sigterm_handler_once() -> None:
    global _PREV_SIGTERM_HANDLER, _SIGTERM_HANDLER_INSTALLED
    if _SIGTERM_HANDLER_INSTALLED:
        return
    try:
        _PREV_SIGTERM_HANDLER = signal.signal(signal.SIGTERM, _on_sigterm)
        _SIGTERM_HANDLER_INSTALLED = True
    except ValueError:
        logger.warning(
            "HiCache L3 write-back: created off the main thread, so no SIGTERM "
            "handler; the final stats line is logged at a normal exit only."
        )


class _CleanerLimits(msgspec.Struct, frozen=True):
    """Cleaner thresholds in host tokens, per rank."""

    host_tokens: int
    # Writing begins when dirty tokens not yet issued exceed this.
    start_tokens: int
    # Dirty tokens (in flight included) above this breach the clean target.
    floor_tokens: int
    # Cap on cleaner writes ahead of eviction: written + in flight - host evicted.
    budget_tokens: int
    max_inflight_tokens: int
    # The largest radix node, so the largest write: the context length in whole
    # pages. Contract: written <= host evicted + budget + this, at every step.
    max_node_tokens: int

    @classmethod
    def from_env(
        cls, host_tokens: int, batch_tokens: int, page_size: int, context_length: int
    ) -> _CleanerLimits:
        def tokens(field) -> int:
            fraction = field.get()
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"{field.name} must be in [0, 1]; got {fraction}.")
            return int(fraction * host_tokens) // page_size * page_size

        return cls(
            host_tokens=host_tokens,
            start_tokens=tokens(envs.SGLANG_HICACHE_STORAGE_WRITE_BACK_DIRTY_START_FRACTION),
            floor_tokens=host_tokens
            - tokens(envs.SGLANG_HICACHE_STORAGE_WRITE_BACK_MIN_CLEAN_FRACTION),
            budget_tokens=tokens(
                envs.SGLANG_HICACHE_STORAGE_WRITE_BACK_PREFETCH_BUDGET_FRACTION
            ),
            max_inflight_tokens=max(
                batch_tokens,
                tokens(envs.SGLANG_HICACHE_STORAGE_WRITE_BACK_MAX_INFLIGHT_FRACTION),
            ),
            max_node_tokens=-(-context_length // page_size) * page_size,
        )


class _Write(msgspec.Struct, frozen=True):
    """A cleaner write in flight; its ack cleans every fragment of its span."""

    node_id: NodeId
    path_start: int
    issued_at: float


class _Pin(msgspec.Struct, frozen=True):
    """A host-resident node not yet written to storage."""

    lock_params: DecLockRefParams
    # Half-open offsets from the root at pin time; a later split leaves the span
    # as a chain of fragments that ends at the pinned node.
    path_start: int
    path_end: int


class StorageWriteBack:
    """Pins acknowledged host copies and writes them to storage on host pressure."""

    def __init__(
        self,
        cache: UnifiedRadixCache,
        *,
        batch_tokens: int,
        bytes_per_token: int,
        cleaner: Optional[_CleanerLimits] = None,
    ):
        self._cache = cache
        self._batch_tokens = batch_tokens
        self._bytes_per_token = bytes_per_token
        self._cleaner = cleaner
        # Cleaner writes in flight by storage op id, in issue order.
        self._writes: dict[int, _Write] = {}
        # Cleaner: unissued dirty level below which a plan found nothing to write.
        self._clean_retry_at = 0
        # Writes acked at shutdown, with the tree left alone.
        self._clean_untracked_tokens = 0
        self._below_floor_since: Optional[float] = None
        # Inside a cleaner eviction, acks land before the evictions they serve:
        # no stats line is logged there.
        self._evicting = False
        # Oldest first: device eviction orders the host acks, and host eviction
        # takes leaves in nearly the same order.
        self._pins: dict[NodeId, _Pin] = {}
        self._pinned_tokens = 0
        # Storage op id -> tokens, in issue order, which is the backup ack order.
        self._inflight: dict[int, int] = {}
        self._inflight_tokens = 0
        # Backup acks the eviction wait consumed; a drain whose counts were agreed
        # before them takes that many fewer.
        self.acks_taken = 0
        self._stats: dict[str, float] = dict.fromkeys(
            (
                "written_tokens",
                "written_bytes",
                "skipped_tokens",
                "host_evicted_tokens",
                "admission_backups",
                "evict_waits",
                "evict_wait_s",
                "read_tokens",
                "read_bytes",
                "read_dropped_tokens",
            ),
            0,
        )
        if cleaner is not None:
            self._stats.update(
                dict.fromkeys(
                    (
                        "clean_victim_tokens",
                        "waited_victim_tokens",
                        "cleaner_issued_tokens",
                        "demand_issued_tokens",
                        "failed_write_tokens",
                        "acks",
                        "ack_latency_s",
                        "ack_latency_max_s",
                        "floor_breaches",
                        "time_below_floor_s",
                        "max_dirty_tokens",
                        "max_written_node_tokens",
                        "oversize_skipped_tokens",
                        "budget_skipped_tokens",
                    ),
                    0,
                )
            )
        self._logged_stats: dict[str, float] = {}
        self._window_stats: dict[str, float] = dict(self._stats)
        self._last_log = time.monotonic()
        self._warned_skip = False
        self._finished = False

    @classmethod
    def create(cls, cache: UnifiedRadixCache) -> Optional[StorageWriteBack]:
        if (
            not envs.SGLANG_OPT_HICACHE_STORAGE_WRITE_BACK.get()
            or cache.cache_controller is None
        ):
            return None
        if cache.host_memory_mode != "cache":
            raise ValueError(
                "SGLANG_OPT_HICACHE_STORAGE_WRITE_BACK needs "
                "--hicache-host-memory-mode cache."
            )
        if set(cache.components) != {BASE_COMPONENT_TYPE}:
            raise ValueError(
                "SGLANG_OPT_HICACHE_STORAGE_WRITE_BACK supports full-attention "
                f"trees only; got {sorted(ct.name for ct in cache.components)}."
            )
        host = cache.cache_controller.mem_pool_host
        page_size = cache.page_size
        fraction = envs.SGLANG_HICACHE_STORAGE_WRITE_BACK_BATCH_FRACTION.get()
        batch_tokens = max(
            page_size, int(fraction * host.size) // page_size * page_size
        )
        cleaner = None
        if envs.SGLANG_HICACHE_STORAGE_WRITE_BACK_CLEANER.get():
            from sglang.srt.mem_cache.rust_tree_core.adapter import (
                RustUnifiedTreeCore,
            )

            if not (
                isinstance(cache.tree_core, RustUnifiedTreeCore)
                and cache.tree_core.supports_host_dirty()
            ):
                raise ValueError(
                    "SGLANG_HICACHE_STORAGE_WRITE_BACK_CLEANER needs the Rust "
                    "TreeCore built with host dirty tracking (plan_host_eviction); "
                    f"got {type(cache.tree_core).__name__}."
                )
            context_length = get_model().context_length
            if not context_length:
                raise ValueError(
                    "SGLANG_HICACHE_STORAGE_WRITE_BACK_CLEANER needs a resolved "
                    "context length to bound the largest write."
                )
            cleaner = _CleanerLimits.from_env(
                host.size, batch_tokens, page_size, context_length
            )
            logger.info(
                "HiCache L3 write-back cleaner: host pool %d pages; writing begins "
                "when unissued dirty pages exceed %d, clean target dirty <= %d "
                "pages, prefetch budget %d pages, at most %d pages in flight, "
                "batch %d pages, %d pages per storage call; largest node %d pages "
                "(context_length %d / page_size %d), so written <= host evicted + "
                "%d pages.",
                host.size // page_size,
                cleaner.start_tokens // page_size,
                cleaner.floor_tokens // page_size,
                cleaner.budget_tokens // page_size,
                cleaner.max_inflight_tokens // page_size,
                batch_tokens // page_size,
                STORAGE_BATCH_SIZE,
                cleaner.max_node_tokens // page_size,
                context_length,
                page_size,
                (cleaner.budget_tokens + cleaner.max_node_tokens) // page_size,
            )
        else:
            logger.info(
                "HiCache L3 write-back: storage is written on host eviction, not "
                "at host admission (SGLANG_OPT_HICACHE_STORAGE_WRITE_BACK); "
                "write-ahead batch %d of %d host tokens.",
                batch_tokens,
                host.size,
            )
        write_back = cls(
            cache,
            batch_tokens=batch_tokens,
            bytes_per_token=host.size_per_token,
            cleaner=cleaner,
        )
        _EXIT_INSTANCES.add(write_back)
        _install_sigterm_handler_once()
        return write_back

    def reset(self) -> None:
        """Forget pins and writes; the tree and its queues reset alongside."""
        self._pins.clear()
        self._pinned_tokens = 0
        self._inflight.clear()
        self._inflight_tokens = 0
        self._writes.clear()
        self._clean_retry_at = 0

    @property
    def cleaner(self) -> bool:
        return self._cleaner is not None

    def release_all(self) -> None:
        """Storage is detaching: release every pin and forget writes still in flight."""
        for node_id, pin in self._pins.items():
            self._cache.dec_host_lock_ref(node_id, pin.lock_params)
        self._stats["skipped_tokens"] += self._pinned_tokens + self._inflight_tokens
        if self._cleaner is not None:
            # Dirty pages not in flight leave without a storage copy, counted.
            cleared = self._cache.tree_core.clear_host_dirty()
            self._stats["skipped_tokens"] += cleared - self._inflight_tokens
        self.reset()

    def pin(self, node_ids: list[NodeId]) -> None:
        """Hold each newly acknowledged host copy until storage has it."""
        if self._cleaner is not None:
            # Dirty, not pinned: the tree core skips nodes without a host copy.
            self._cache.tree_core.mark_host_dirty(node_ids)
            self._clean()
            return
        cache = self._cache
        node_ids = [
            node_id
            for node_id in node_ids
            if node_id not in self._pins and cache.tree_core.is_backuped(node_id)
        ]
        if not node_ids:
            return
        for node_id, path in zip(node_ids, cache.tree_core.collect_path_info(node_ids)):
            span = path[-1]
            lock_params = cache.inc_host_lock_ref(node_id).to_dec_params()
            self._pins[node_id] = _Pin(lock_params, span.path_start, span.path_end)
            self._pinned_tokens += span.path_end - span.path_start

    def on_demoted(self) -> None:
        """Dirty copies just became host-only, so writable: run the cleaner."""
        if self._cleaner is not None:
            self._clean_retry_at = 0
            self._clean()

    def evict_host(self, num_tokens: int, evicted: int) -> int:
        """Finish a host eviction that freed only ``evicted`` of ``num_tokens``:
        write the oldest pins to storage, wait for enough of those writes, and
        evict again. Returns the tokens freed in all."""
        self._stats["host_evicted_tokens"] += evicted
        if self._cleaner is not None:
            return self._evict_host_cleaner(num_tokens, evicted)
        while evicted < num_tokens:
            shortfall = num_tokens - evicted
            self._write_ahead(shortfall)
            if not self._inflight:
                break
            self._wait(shortfall)
            freed = self._cache._drive_host_eviction(shortfall, BASE_COMPONENT_TYPE)
            self._stats["host_evicted_tokens"] += freed
            evicted += freed
        self._maybe_log()
        return evicted

    def on_backup_acks(self, operations: list[StorageOperation], synced: bool) -> None:
        """Account drained storage acks. With the cleaner, and when every rank
        drained the same acks (``synced``), a write counts as stored only if it
        succeeded on all ranks, so all trees clean the same pages."""
        oks = None
        if self._cleaner is not None:
            oks = [
                op.id not in self._inflight
                or self._write_ok(op, self._inflight[op.id])
                for op in operations
            ]
            if synced:
                oks = self._cache.agree_on_all_ranks(oks)
        for i, operation in enumerate(operations):
            self.on_backup_ack(operation, None if oks is None else oks[i])

    def on_backup_ack(
        self,
        operation: StorageOperation,
        ok: Optional[bool] = None,
        *,
        update_tree: bool = True,
    ) -> None:
        """Account one drained storage ack; its host lock is already released."""
        tokens = self._inflight.pop(operation.id, None)
        if tokens is None:
            # Every storage write under this policy is a write-back.
            self._stats["admission_backups"] += 1
            return
        self._inflight_tokens -= tokens
        if self._cleaner is not None:
            self._on_cleaner_ack(operation, tokens, ok, update_tree)
            if not self._evicting:
                self._maybe_log()
            return
        written = operation.completed_tokens
        self._stats["written_tokens"] += written
        self._stats["written_bytes"] += self._written_bytes(operation)
        if written < tokens and not self._cache.cache_controller.backup_skip:
            self._stats["skipped_tokens"] += tokens - written
            if not self._warned_skip:
                self._warned_skip = True
                logger.warning(
                    "HiCache L3 write-back: storage kept %d of %d tokens; the rest "
                    "leave the host pool without a storage copy (counted as "
                    "skipped_pages).",
                    written,
                    tokens,
                )
        self._maybe_log()

    def on_storage_read(self, num_tokens: int) -> None:
        self._stats["read_tokens"] += num_tokens
        self._stats["read_bytes"] += num_tokens * self._bytes_per_token

    def on_storage_read_dropped(self, num_tokens: int) -> None:
        """Read tokens freed instead of inserted (a dropped insert, or a prefix
        already in the host pool): they never reach a host eviction."""
        self._stats["read_dropped_tokens"] += num_tokens

    # ---- Dirty-page cleaner ----

    def _sidecar_short_pages(self, operation: StorageOperation) -> int:
        """KV pages written whose page of an all-pages sidecar (DSA indexer,
        draft KV) was not: a read of them misses, as the hit is truncated."""
        kv_pages = operation.completed_tokens // self._cache.page_size
        policies = {t.name: t.hit_policy for t in operation.pool_transfers or ()}
        return sum(
            max(0, kv_pages - hits)
            for name, hits in operation.pool_storage_result.extra_pool_hit_pages.items()
            if policies.get(name) == PoolHitPolicy.ALL_PAGES
        )

    def _write_ok(self, operation: StorageOperation, tokens: int) -> bool:
        """Whether this rank stored all it had to: the KV (unless another TP rank
        writes it) and every all-pages sidecar page."""
        controller = self._cache.cache_controller
        owns_write = not controller.backup_skip or any(
            controller.should_backup(t) for t in operation.pool_transfers or ()
        )
        if owns_write and operation.completed_tokens < tokens:
            return False
        return not self._sidecar_short_pages(operation)

    def _on_cleaner_ack(
        self,
        operation: StorageOperation,
        tokens: int,
        ok: Optional[bool],
        update_tree: bool,
    ) -> None:
        stats = self._stats
        write = self._writes.pop(operation.id)
        latency = time.monotonic() - write.issued_at
        stats["acks"] += 1
        stats["ack_latency_s"] += latency
        stats["ack_latency_max_s"] = max(stats["ack_latency_max_s"], latency)
        if ok is None:
            ok = self._write_ok(operation, tokens)
        if not ok:
            # The page stays dirty, so unevictable, and is written again.
            stats["failed_write_tokens"] += tokens
            if not self._warned_skip:
                self._warned_skip = True
                logger.warning(
                    "HiCache L3 write-back cleaner: a storage write failed or fell "
                    "short (%d of %d tokens); its pages stay dirty in the host pool "
                    "and are written again (counted as failed_write_pages).",
                    operation.completed_tokens,
                    tokens,
                )
            return
        stats["written_tokens"] += tokens
        stats["written_bytes"] += self._written_bytes(operation)
        stats["max_written_node_tokens"] = max(stats["max_written_node_tokens"], tokens)
        if not update_tree:
            self._clean_untracked_tokens += tokens
            return
        core = self._cache.tree_core
        path = core.collect_path_info([write.node_id])[0]
        # A split in flight leaves the written span as fragments ending at the node.
        core.mark_host_clean(
            [span.node_id for span in path if span.path_start >= write.path_start]
        )
        self._sample_floor(self._dirty_tokens(), time.monotonic())

    def _prefetch_tokens(self) -> int:
        stats = self._stats
        return (
            stats["written_tokens"]
            + self._inflight_tokens
            - stats["host_evicted_tokens"]
        )

    def _dirty_tokens(self) -> int:
        dirty, _ = self._cache.tree_core.host_dirty_tokens()
        return dirty

    def _sample_floor(self, dirty: int, now: float) -> None:
        stats = self._stats
        stats["max_dirty_tokens"] = max(stats["max_dirty_tokens"], dirty)
        below = dirty > self._cleaner.floor_tokens
        if below and self._below_floor_since is None:
            self._below_floor_since = now
            stats["floor_breaches"] += 1
        elif not below and self._below_floor_since is not None:
            stats["time_below_floor_s"] += now - self._below_floor_since
            self._below_floor_since = None

    def _clean(self) -> None:
        """Write the coldest dirty pages while unissued dirty pages exceed the
        start threshold, within the prefetch budget and the in-flight cap."""
        limits = self._cleaner
        dirty = self._dirty_tokens()
        self._sample_floor(dirty, time.monotonic())
        unissued = dirty - self._inflight_tokens
        excess = unissued - limits.start_tokens
        if excess <= 0 or unissued < self._clean_retry_at:
            return
        prefetch = self._prefetch_tokens()
        # A budget smaller than a node still lets one node out at a time.
        if prefetch > 0 and prefetch >= limits.budget_tokens:
            return
        if self._inflight_tokens >= limits.max_inflight_tokens:
            return
        issued = self._issue_coldest(max(excess, self._batch_tokens))
        # Nothing writable (dirty pages locked or on the device): retry a batch later.
        self._clean_retry_at = unissued + self._batch_tokens if not issued else 0

    def _issue_coldest(self, target: int, *, shortfall: int = 0) -> int:
        """Issue storage writes for the dirty pages host eviction would take
        first, until ``target`` tokens; returns the tokens issued.

        Every write keeps written + in flight - host evicted within the budget
        plus one node (the contract). The cleaner (``shortfall`` 0) writes past
        the budget only with its first node; an eviction short of ``shortfall``
        is credited with what it will free of its own writes."""
        core = self._cache.tree_core
        writes = list(self._writes.values())
        released = []
        if writes:
            paths = core.collect_path_info([w.node_id for w in writes])
            for write, path in zip(writes, paths):
                released.extend(
                    span.node_id for span in path if span.path_start >= write.path_start
                )
        in_flight = set(released)
        picked = [
            node_id
            for node_id, _, dirty in core.plan_host_eviction(target, released)
            if dirty and node_id not in in_flight
        ]
        if not picked:
            return 0
        limits = self._cleaner
        stats = self._stats
        issued = 0
        now = time.monotonic()
        for node_id, path in zip(picked, core.collect_path_info(picked)):
            if issued >= target:
                break
            span = path[-1]
            tokens = span.path_end - span.path_start
            if tokens > limits.max_node_tokens:
                # Longer than the context: the contract cannot cover it.
                stats["oversize_skipped_tokens"] += tokens
                continue
            after = self._prefetch_tokens() + tokens
            if shortfall:
                credit = min(issued + tokens, shortfall)
                if after - credit > limits.budget_tokens + limits.max_node_tokens:
                    stats["budget_skipped_tokens"] += tokens
                    continue
            elif issued:
                if after > limits.budget_tokens:
                    break
                if self._inflight_tokens + tokens > limits.max_inflight_tokens:
                    break
            operation_id = self._cache.write_backup_storage(node_id)
            if operation_id is None:
                continue
            self._inflight[operation_id] = tokens
            self._inflight_tokens += tokens
            self._writes[operation_id] = _Write(node_id, span.path_start, now)
            issued += tokens
        stats["demand_issued_tokens" if shortfall else "cleaner_issued_tokens"] += issued
        return issued

    def _evict_host_cleaner(self, num_tokens: int, evicted: int) -> int:
        """Clean victims went first; with none left, wait for the writes in
        flight (issuing the coldest dirty pages if none is), then evict again."""
        stats = self._stats
        stats["clean_victim_tokens"] += evicted
        self._evicting = True
        try:
            evicted = self._evict_waiting(num_tokens, evicted)
        finally:
            self._evicting = False
        # Evictions turn parents into leaves: run the cleaner again.
        self._clean_retry_at = 0
        self._clean()
        self._maybe_log()
        return evicted

    def _evict_waiting(self, num_tokens: int, evicted: int) -> int:
        stats = self._stats
        while evicted < num_tokens:
            shortfall = num_tokens - evicted
            if self._inflight_tokens < shortfall:
                self._issue_coldest(
                    shortfall - self._inflight_tokens, shortfall=shortfall
                )
            if not self._inflight:
                # Nothing evictable or writable: the caller backs off.
                break
            failed = stats["failed_write_tokens"]
            self._wait(shortfall)
            freed = self._cache._drive_host_eviction(shortfall, BASE_COMPONENT_TYPE)
            stats["host_evicted_tokens"] += freed
            stats["waited_victim_tokens"] += freed
            evicted += freed
            if not freed and stats["failed_write_tokens"] > failed:
                # Storage is failing: the dirty pages stay, the caller backs off.
                break
        return evicted

    def _write_ahead(self, shortfall: int) -> None:
        """Issue storage writes for the oldest pins unless the writes in flight
        already cover ``shortfall``; at least one batch goes out at a time."""
        need = shortfall - self._inflight_tokens
        if need <= 0:
            return
        need = max(need, self._batch_tokens)
        while self._pins and need > 0:
            node_id = next(iter(self._pins))
            need -= self._write(node_id, self._pins.pop(node_id))

    def _write(self, node_id: NodeId, pin: _Pin) -> int:
        cache = self._cache
        self._pinned_tokens -= pin.path_end - pin.path_start
        path = cache.tree_core.collect_path_info([node_id])[0]
        assert path[-1].path_end == pin.path_end, (
            f"pinned node {node_id} ends at {path[-1].path_end}, "
            f"pinned at {pin.path_end}"
        )
        issued = 0
        for span in path:
            if span.path_end <= pin.path_start:
                continue
            tokens = span.path_end - span.path_start
            # Each write takes its own host lock until its ack.
            operation_id = cache.write_backup_storage(span.node_id)
            if operation_id is None:
                self._stats["skipped_tokens"] += tokens
                continue
            self._inflight[operation_id] = tokens
            self._inflight_tokens += tokens
            issued += tokens
        cache.dec_host_lock_ref(node_id, pin.lock_params)
        return issued

    def _wait(self, shortfall: int) -> None:
        """Consume backup acks in issue order until writes covering
        ``shortfall`` tokens have landed, or every write in flight has."""
        num_acks, landed = 0, 0
        for tokens in self._inflight.values():
            num_acks += 1
            landed += tokens
            if landed >= shortfall:
                break
        controller = self._cache.cache_controller
        start = time.perf_counter()
        warn_at = start + _ACK_WAIT_WARN_S
        while controller.ack_backup_queue.qsize() < num_acks:
            if controller.storage_stop_event.is_set():
                raise RuntimeError(
                    "HiCache L3 write-back is waiting on storage writes, but the "
                    "storage threads have stopped."
                )
            if time.perf_counter() > warn_at:
                logger.warning(
                    "HiCache L3 write-back: host eviction has waited %.0f s for %d "
                    "storage write acks.",
                    time.perf_counter() - start,
                    num_acks,
                )
                warn_at += _ACK_WAIT_WARN_S
            time.sleep(0.0005)
        self._cache._drain_storage_control_queues_impl(
            n_storage_hit=0,
            n_ack_prefetch=0,
            n_backup=num_acks,
            n_release=0,
            extra_release_counts=None,
            log_metrics=True,
        )
        self.acks_taken += num_acks
        self._stats["evict_waits"] += 1
        self._stats["evict_wait_s"] += time.perf_counter() - start

    def _written_bytes(self, operation: StorageOperation) -> int:
        cache = self._cache
        nbytes = operation.completed_tokens * self._bytes_per_token
        entries = cache.cache_controller.mem_pool_host.entry_map
        for name, pages in operation.pool_storage_result.extra_pool_hit_pages.items():
            entry = entries.get(name)
            if entry is not None:
                nbytes += pages * cache.page_size * entry.host_pool.size_per_token
        return nbytes

    def stats(self) -> dict[str, float]:
        page_size = self._cache.page_size
        stats = self._stats
        return {
            "written_pages": stats["written_tokens"] // page_size,
            "written_bytes": stats["written_bytes"],
            "inflight_pages": self._inflight_tokens // page_size,
            "pinned_pages": self._pinned_tokens // page_size,
            "host_evicted_pages": stats["host_evicted_tokens"] // page_size,
            "skipped_pages": stats["skipped_tokens"] // page_size,
            "admission_backups": stats["admission_backups"],
            "evict_waits": stats["evict_waits"],
            "evict_wait_s": round(stats["evict_wait_s"], 3),
            "read_pages": stats["read_tokens"] // page_size,
            "read_bytes": stats["read_bytes"],
            "read_dropped_pages": stats["read_dropped_tokens"] // page_size,
            **(self._cleaner_stats() if self._cleaner is not None else {}),
        }

    def _cleaner_stats(self) -> dict[str, float]:
        page_size = self._cache.page_size
        stats = self._stats
        limits = self._cleaner
        host = limits.host_tokens
        dirty, dirty_reclaimed = self._cache.tree_core.host_dirty_tokens()
        dirty -= self._clean_untracked_tokens
        acks = stats["acks"]
        below_floor_s = stats["time_below_floor_s"]
        if self._below_floor_since is not None:
            below_floor_s += time.monotonic() - self._below_floor_since
        return {
            "dirty_pages": dirty // page_size,
            "dirty_fraction": round(dirty / host, 4),
            "max_dirty_fraction": round(stats["max_dirty_tokens"] / host, 4),
            "pinned_fraction": round(self._inflight_tokens / host, 4),
            "prefetch_pages": self._prefetch_tokens() // page_size,
            "prefetch_budget_pages": limits.budget_tokens // page_size,
            "clean_victim_pages": stats["clean_victim_tokens"] // page_size,
            "waited_victim_pages": stats["waited_victim_tokens"] // page_size,
            "cleaner_issued_pages": stats["cleaner_issued_tokens"] // page_size,
            "demand_issued_pages": stats["demand_issued_tokens"] // page_size,
            "failed_write_pages": stats["failed_write_tokens"] // page_size,
            "dirty_reclaimed_pages": dirty_reclaimed // page_size,
            "acks": acks,
            "ack_latency_mean_s": round(stats["ack_latency_s"] / acks, 4) if acks else 0,
            "ack_latency_max_s": round(stats["ack_latency_max_s"], 4),
            "floor_breaches": stats["floor_breaches"],
            "time_below_floor_s": round(below_floor_s, 3),
            "max_written_node_pages": stats["max_written_node_tokens"] // page_size,
            "max_node_pages": limits.max_node_tokens // page_size,
            "prefetch_bound_pages": (limits.budget_tokens + limits.max_node_tokens)
            // page_size,
            "oversize_skipped_pages": stats["oversize_skipped_tokens"] // page_size,
            "budget_skipped_pages": stats["budget_skipped_tokens"] // page_size,
        }

    def finish(self, reason: str) -> None:
        """Wait for the storage writes in flight, then log the final counts once.

        Runs in the SIGTERM handler, so it takes acks off the queue without its
        lock (the handler may have interrupted this thread holding it) and leaves
        the tree alone: host locks die with the process."""
        if self._finished:
            return
        self._finished = True
        timeout = envs.SGLANG_HICACHE_STORAGE_WRITE_BACK_DRAIN_TIMEOUT.get()
        start = time.monotonic()
        error = self._drain_inflight(start + timeout)
        drain_s = time.monotonic() - start
        cache = self._cache
        page_size = cache.page_size
        if self._cleaner is not None:
            self._close_floor_interval(time.monotonic())
        stats = self.stats()
        final = {
            "time": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "reason": reason,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "pp_rank": cache.pp_rank,
            "drained": error is None,
            "drain_s": round(drain_s, 3),
            **stats,
        }
        if error is not None:
            final["drain_error"] = error
            final["drain_timeout_s"] = timeout
        # Under write-back every host page is a pin, a write in flight, a write
        # acked or skipped, or an L3 read inserted, so with nothing else holding
        # host pages: written + read + skipped
        #   = host_evicted + (host-resident - pinned - inflight) + read_dropped.
        controller = cache.cache_controller
        host = controller.mem_pool_host
        release_pending = sum(
            page.numel() for page in list(controller.host_mem_release_queue.queue)
        )
        resident = host.size - host.available_size() - release_pending
        final["l2_resident_pages"] = resident // page_size
        unwritten = self._pinned_tokens + self._inflight_tokens
        dirty_reclaimed = 0
        if self._cleaner is not None:
            # Dirty pages (in flight included) are the unwritten ones; a dirty
            # copy the device still held may leave without a write.
            dirty, dirty_reclaimed = cache.tree_core.host_dirty_tokens()
            unwritten = dirty - self._clean_untracked_tokens
        final["l2_unpinned_resident_pages"] = (resident - unwritten) // page_size
        busy = [
            name
            for name, ops in (
                ("prefetch", cache.ongoing_prefetch),
                ("write_through", cache.ongoing_write_through),
            )
            if ops
        ]
        if busy:
            final["identity"] = "unchecked:" + "+".join(busy) + "_in_flight"
        else:
            counts = self._stats
            lhs = (
                counts["written_tokens"]
                + counts["read_tokens"]
                + counts["skipped_tokens"]
                + dirty_reclaimed
            )
            rhs = (
                counts["host_evicted_tokens"]
                + resident
                - unwritten
                + counts["read_dropped_tokens"]
            )
            final["identity"] = (
                "exact" if lhs == rhs else f"off_by_{lhs - rhs}_tokens"
            )
        logger.info(
            "HiCache L3 write-back final stats: %s",
            " ".join(f"{key}={value}" for key, value in final.items()),
        )

    def _drain_inflight(self, deadline: float) -> Optional[str]:
        """Account acks until no write is in flight; returns why it stopped
        short, or None once drained."""
        controller = self._cache.cache_controller
        acks = controller.ack_backup_queue.queue
        while self._inflight:
            try:
                operation = acks.popleft()
            except IndexError:
                if controller.storage_stop_event.is_set():
                    return "storage_threads_stopped"
                if time.monotonic() > deadline:
                    return "timeout"
                time.sleep(0.0005)
                continue
            self.on_backup_ack(operation, update_tree=False)
        if self._inflight_tokens:
            # An ack this thread was handling when the signal landed.
            return "interrupted_inside_an_ack"
        return None

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log < _STATS_LOG_INTERVAL_S:
            return
        window_s = now - self._last_log
        self._last_log = now
        if self._cleaner is not None:
            self._close_floor_interval(now)
        stats = self.stats()
        if stats == self._logged_stats:
            return
        self._logged_stats = stats
        if self._cleaner is not None:
            stats.update(self._window_deltas(window_s))
        logger.info(
            "HiCache L3 write-back stats: %s",
            " ".join(f"{key}={value}" for key, value in stats.items()),
        )

    def _close_floor_interval(self, now: float) -> None:
        """Book the open below-floor interval up to ``now``, so each window
        holds its own elapsed time."""
        if self._below_floor_since is not None:
            self._stats["time_below_floor_s"] += now - self._below_floor_since
            self._below_floor_since = now

    def _window_deltas(self, window_s: float) -> dict[str, float]:
        page_size = self._cache.page_size
        stats, last = self._stats, self._window_stats
        delta = {key: stats[key] - last.get(key, 0) for key in stats}
        self._window_stats = dict(stats)
        issued = delta["cleaner_issued_tokens"] + delta["demand_issued_tokens"]
        return {
            "window_s": round(window_s, 3),
            "w_written_pages": delta["written_tokens"] // page_size,
            "w_host_evicted_pages": delta["host_evicted_tokens"] // page_size,
            "w_clean_victim_pages": delta["clean_victim_tokens"] // page_size,
            "w_waited_victim_pages": delta["waited_victim_tokens"] // page_size,
            "w_evict_wait_s": round(delta["evict_wait_s"], 3),
            "w_floor_breaches": delta["floor_breaches"],
            "w_time_below_floor_s": round(delta["time_below_floor_s"], 3),
            "writer_issue_pages_per_s": round(issued / page_size / window_s, 1)
            if window_s > 0
            else 0,
        }


class WriteOnHostEvictFinalStats:
    """SGLANG_HICACHE_L3_WRITE_ON_HOST_EVICT: at SIGTERM or exit, wait for the
    write-behind L3 writes in flight, then log one final stats line from the
    mode's own counters. Observability only; it changes no write or eviction."""

    def __init__(self, cache: UnifiedRadixCache):
        self._cache = cache
        self._finished = False
        self.written_tokens = 0
        self.read_tokens = 0
        self.host_evicted_tokens = 0

    @classmethod
    def register(cls, cache: UnifiedRadixCache) -> WriteOnHostEvictFinalStats:
        final = cls(cache)
        _EXIT_INSTANCES.add(final)
        _install_sigterm_handler_once()
        return final

    def finish(self, reason: str) -> None:
        if self._finished:
            return
        self._finished = True
        cache = self._cache
        timeout = envs.SGLANG_HICACHE_STORAGE_WRITE_BACK_DRAIN_TIMEOUT.get()
        start = time.monotonic()
        error = self._drain(start + timeout)
        page_size = cache.page_size
        stats = cache._l3_tier_stats
        final = {
            "time": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "reason": reason,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "pp_rank": cache.pp_rank,
            "drained": error is None,
            "drain_s": round(time.monotonic() - start, 3),
            "written_pages": self.written_tokens // page_size,
            "read_pages": self.read_tokens // page_size,
            "host_evicted_pages": self.host_evicted_tokens // page_size,
            "wb_walks": stats["wb_walks"],
            "wb_issued_tokens": stats["wb_issued_tokens"],
            "wb_clean_tokens": stats["wb_clean_tokens"],
            "wb_unbacked_tokens": stats["wb_unbacked_tokens"],
            "wb_inflight_tokens": sum(cache._write_behind_inflight.values()),
        }
        if error is not None:
            final["drain_error"] = error
            final["drain_timeout_s"] = timeout
        logger.info(
            "HiCache L3 write-on-host-evict final stats: %s",
            " ".join(f"{key}={value}" for key, value in final.items()),
        )

    def _drain(self, deadline: float) -> Optional[str]:
        """Account write-behind acks until none is in flight; returns why it
        stopped short, or None once drained. Lock-free and tree-free, as the
        SIGTERM handler may have interrupted this thread anywhere."""
        cache = self._cache
        controller = cache.cache_controller
        acks = controller.ack_backup_queue.queue
        inflight = cache._write_behind_inflight
        while inflight:
            try:
                operation = acks.popleft()
            except IndexError:
                if controller.storage_stop_event.is_set():
                    return "storage_threads_stopped"
                if time.monotonic() > deadline:
                    return "timeout"
                time.sleep(0.0005)
                continue
            if inflight.pop(operation.id, None) is not None:
                self.written_tokens += operation.completed_tokens
        return None
