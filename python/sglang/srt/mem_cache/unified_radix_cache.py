from __future__ import annotations

import atexit
import contextlib
import logging
import threading
import time
from array import array
from collections import deque
from dataclasses import replace
from http import HTTPStatus
from queue import Queue
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterator,
    NamedTuple,
    Optional,
    Sequence,
    TypeVar,
)

import msgspec
import torch

from sglang.srt.distributed.communication_tags import P2PTag
from sglang.srt.environ import envs
from sglang.srt.managers import pp_bubble_diag
from sglang.srt.managers.cache_controller import CacheOperation
from sglang.srt.managers.scheduler_components.shm_tag_exchange import (
    ShmTagExchange,
    fold_u64,
    get_shm_exchange,
)
from sglang.srt.mem_cache import pmicro_hoststats
from sglang.srt.mem_cache.allocator.page_interleave import (
    page_interleave_shard_size,
)
from sglang.srt.mem_cache.allocator.unified_hybrid_swa import UnifiedSWAAllocatorBase
from sglang.srt.mem_cache.allocator.unified_mamba import (
    UnifiedMambaTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    CacheRequestHandle,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
    TreeLock,
)
from sglang.srt.mem_cache.buffer_mode.pipeline import (
    BufferModePipeline,
    validate_buffer_only_stack,
)
from sglang.srt.mem_cache.buffer_mode.storage_existence_cache import (
    PoolBeliefPolicy,
    StorageExistenceCache,
    default_belief_policy,
)
from sglang.srt.mem_cache.common import RetractionBackup
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    MambaStateRestoreOperation,
    PPPrefetchDecision,
    PrefetchOperation,
)
from sglang.srt.mem_cache.l3_lifetime_ledger import L3LifetimeLedger
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.cache_action import (
    BackupKV,
    CacheAction,
    ComponentAction,
    FreeComponentDeviceSlot,
    FreeDeviceKV,
    FreeDeviceKVFullOnly,
    UpdatePendingBackupOnNodeSplit,
)

# UnifiedTreeNode / UnifiedLRUList live on the tree core; re-exported here
# because other modules and tests import them from this module.
from sglang.srt.mem_cache.unified_cache.components import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
    FullComponent,
    MambaComponent,
    PrepareLoadBackResult,
    SWAComponent,
    TreeComponent,
)
from sglang.srt.mem_cache.unified_cache.session_ref_tracker import (
    UnifiedSessionRefTracker,
)
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.srt.mem_cache.unified_cache.storage_write_back import (
    StorageWriteBack,
    WriteOnHostEvictFinalStats,
)
from sglang.srt.mem_cache.unified_cache.tree_core_registry import (
    create_tree_core,
    select_tree_core_backend,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinker,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (  # noqa: F401
    NodeId,
    UnifiedLRUList,
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.srt.observability.metrics_collector import (
    StorageMetrics,
    StorageMetricsCollector,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_memory,
    get_model,
    get_observability,
    get_parallel,
)
from sglang.srt.session.streaming_session import StreamingSession, is_virtual_node
from sglang.srt.utils.common import ceil_align, get_device_module, is_cuda, is_hip
from sglang.srt.utils.poll_timer import PollTimer

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import HiCacheAck
    from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        PrefetchOperation,
    )
    from sglang.srt.mem_cache.pool_host import PoolEntry
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
        BaseEvictionResult,
    )
    from sglang.srt.server_args import ServerArgs

from sglang.srt.utils.rank_consensus_checker import rank_consensus

T = TypeVar("T")


def _c128_transfer_num_pages(transfers: Sequence[PoolTransfer], page_size: int) -> int:
    num_pages = 0
    for transfer in transfers:
        if transfer.host_indices is None:
            continue
        num_slots = len(transfer.host_indices)
        assert num_slots % page_size == 0, (
            f"C128 load-back transfers must contain complete physical pages: "
            f"{num_slots=}, {page_size=}"
        )
        num_pages += num_slots // page_size
    return num_pages


COMPONENT_REGISTRY: dict[ComponentType, type[TreeComponent]] = {
    ComponentType.FULL: FullComponent,
    ComponentType.MAMBA: MambaComponent,
    ComponentType.SWA: SWAComponent,
}


logger = logging.getLogger(__name__)
device_module = get_device_module()


class _OngoingWriteThrough(NamedTuple):
    """Tracks an in-flight D→H write-through operation."""

    node_id: NodeId
    lock_params: Optional[DecLockRefParams]
    publish_node_ids: list[NodeId]


class _OngoingLoadBack(NamedTuple):
    """Tracks an in-flight H→D load-back operation."""

    node_id: NodeId
    lock_params: DecLockRefParams
    host_lock_params: DecLockRefParams


class _StoragePrefetchWait(msgspec.Struct):
    timer: PollTimer = msgspec.field(default_factory=PollTimer)
    timeout_s: float = 0.0
    expired: bool = False


class _OngoingPrefetch(NamedTuple):
    """Tracks an in-flight storage→host prefetch operation."""

    anchor_node_id: NodeId
    prefetch_key: RadixKey
    host_indices: torch.Tensor
    operation: PrefetchOperation
    anchor_lock_params: DecLockRefParams
    comp_xfers: dict[ComponentType, list[PoolTransfer]]


class UnifiedRadixCache(BasePrefixCache):
    # SGLANG_HICACHE_WB_EVICT_BATCH in scope (init_hicache); the walk below
    # takes the per-victim path unless it is set.
    _wb_evict_batch = False
    # SGLANG_MAMBA_PATH_CAP_COUNTS_HOST, set by _init_mamba_path_cap_counts_host.
    _mamba_path_cap_counts_host = False
    # SGLANG_HICACHE_MAMBA_BACKUP_BEFORE_DEMOTE / SGLANG_LOG_MAMBA_DEMOTE_DROPS,
    # set by _init_mamba_demote_watch.
    _mamba_demote_watch = False
    _mamba_backup_before_demote = False
    # Nodes whose device-only Mamba state was copied to host before their demote;
    # a later load-back of one counts in demote_copy_restores.
    _demote_copied_nodes: Optional[set[NodeId]] = None
    mamba_retention_counts: Optional[dict[str, int]] = None
    # L2->L3 write-back (SGLANG_OPT_HICACHE_STORAGE_WRITE_BACK); set in init_hicache.
    _storage_write_back: Optional[StorageWriteBack] = None
    # check_hicache_events reads this on stubs built with object.__new__ in unit tests;
    # init_hicache sets the real value.
    _l3_write_on_host_evict: bool = False
    _l3_write_state_on_tombstone: bool = False
    _l3_saved_token_threshold: bool = False
    _l3_belief_query_renew: bool = True
    _l3_lifetime_ledger: Optional[L3LifetimeLedger] = None
    _l3_tombstone_quarantined_ops: frozenset[int] = frozenset()
    # SGLANG_HICACHE_L3_TOMBSTONE_ASYNC: storage op id -> pinned victim, in issue order.
    _l3_tombstone_pending: Optional[dict[int, NodeId]] = None
    _l3_tombstone_reserve: int = 0
    # Final stats at SIGTERM / exit under SGLANG_HICACHE_L3_WRITE_ON_HOST_EVICT.
    _l3_tier_final: Optional[WriteOnHostEvictFinalStats] = None
    # SGLANG_HICACHE_L3_ACK_LAG_STEPS on a PP>1 cache-mode PP0 (else 0); see
    # _lag_backup_ack_count. Cumulative counts of this rank's ack_backup_queue.
    _l3_ack_lag_steps: int = 0
    _l3_ack_lag_hist: Optional[deque] = None
    # SGLANG_HICACHE_DEVICE_WRITE_AHEAD under write_back (init_hicache); see
    # _write_ahead_device_tail.
    _device_write_ahead: bool = False
    _l3_ack_removed: int = 0

    def __init__(
        self,
        params: CacheInitParams,
    ):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.disable = params.disable

        if params.enable_metrics:
            self.init_metrics_collector()
        self._enable_metrics_flag = params.enable_metrics
        self.enable_storage_metrics = False
        self.storage_metrics_collector: Optional[StorageMetricsCollector] = None
        self.extra_metric_labels = None

        assert params.tree_components is not None
        self.tree_components = tuple(params.tree_components)
        self.enable_session_radix_cache = params.enable_session_radix_cache
        component_registry = COMPONENT_REGISTRY
        if params.component_registry_override:
            component_registry = {
                **COMPONENT_REGISTRY,
                **params.component_registry_override,
            }
        self.components: dict[ComponentType, TreeComponent] = {
            ct: component_registry[ct](self, params) for ct in self.tree_components
        }
        self._components_tuple: tuple[TreeComponent, ...] = tuple(
            self.components.values()
        )
        # Whether SWA is enabled.
        self.is_swa_enabled = ComponentType.SWA in params.tree_components
        # Whether Mamba is enabled.
        self.is_mamba_enabled = ComponentType.MAMBA in params.tree_components
        # The tree holds no Mamba state but requests do: the cache only frees
        # each request's own state slot when it leaves (state-free PD decode).
        self.request_mamba_only = params.request_mamba_only
        assert not (self.request_mamba_only and self.is_mamba_enabled)
        # Whether the mamba extra (ping-pong) buffer is enabled.
        self.enable_mamba_extra_buffer = (
            params.enable_mamba_extra_buffer if self.is_mamba_enabled else False
        )
        # Required cache history, shared by retention, budgeting and transfers.
        self._sliding_window_size = (
            params.swa_retain_window if self.is_swa_enabled else None
        )
        self.decoder_swa_bounded_replay = params.decoder_swa_bounded_replay
        self.draft_swa_window = params.draft_swa_window
        # The TreeCore owns the tree member-var state (structure, LRUs, sizes,
        # evictable leaves) and drives the components' tree-level hooks.
        self._tree_core_backend = select_tree_core_backend(params)
        # A disabled tree holds nothing, so it reports no KV events and never
        # evicts; its eviction policy config is not built or validated.
        tree_params = (
            replace(
                params,
                enable_kv_cache_events=False,
                eviction_policy="lru",
                eviction_policy_config=None,
            )
            if self.disable
            else params
        )
        self.tree_core = create_tree_core(
            name=self._tree_core_backend,
            params=tree_params,
            components=self.components,
        )
        logger.info("Radix cache tree core: %s", self._tree_core_backend)
        # Components execute boundary actions through the tree core.
        for component in self.components.values():
            component.tree_core = self.tree_core

        if (
            page_interleave_shard_size(params.token_to_kv_pool_allocator) > 1
            and not self.tree_core.supports_rotation_base
        ):
            # A core that does not model rotation_base would never decline a
            # cross-base graft, and the resulting cached path's page owners are
            # not one cyclic run: later readers take a negative allgather pad or
            # silently read another rank's scratch rows. Fail at construction
            # rather than corrupt reads at serve time.
            raise ValueError(
                "logical-page KV sharding requires a tree core that tracks "
                "rotation bases; "
                f"SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND={self._tree_core_backend!r} "
                "does not."
            )

        # Session ref tracking (--enable-session-radix-cache).
        self.session_refs = UnifiedSessionRefTracker(
            components=self._components_tuple,
            tree_core=self.tree_core,
            enable_session_radix_cache=self.enable_session_radix_cache,
        )

        self.sidecar_pool_specs: list[SidecarPoolSpec] = []

        # Always on; the try_* entries short-circuit on non-streaming reqs.
        self.session = StreamingSession(self)

        self.tp_group = params.tp_cache_group
        self.attn_cp_group = params.attn_cp_cache_group
        self.attn_tp_group = params.attn_tp_cache_group
        self.pp_group = params.pp_cache_group
        self.tp_world_size = (
            1
            if self.tp_group is None
            else torch.distributed.get_world_size(group=self.tp_group)
        )
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.work_list: list[torch.distributed.Work] = []
        self.pp_sync_window = 1
        if envs.SGLANG_PP_SYNC_WINDOW.get() and self.pp_size > 1:
            self.pp_sync_window = self.pp_size + get_parallel().pp_async_batch_depth
            logger.info(
                "HiCache PP-sync window: %d rounds (SGLANG_PP_SYNC_WINDOW).",
                self.pp_sync_window,
            )
        self.pp_sync_rounds: deque[list[torch.distributed.Work]] = deque()

        # HiCache D↔H defaults (overridden by init_hicache)
        self.cache_controller: Optional[HybridCacheController] = None
        self.host_pool_group = None  # set by attach_hybrid_pool_to_unified_cache
        # Host backups of restores whose host-to-device copies may still be in
        # flight, each with its copy's finish event; their host slots are
        # returned once the copy has finished.
        self._pending_restore_backups: list[tuple[object, RetractionBackup]] = []
        # Owns the storage backend lifecycle; built by init_hicache.
        self._storage_attachment: Optional[StorageAttachment] = None
        self.linker: Optional[UnifiedCacheLinkerWrapper] = None
        self.prefetch_stop_policy = "best_effort"
        self.prefetch_threshold = 256
        self.prefetch_timeout_base = 1.0
        self.prefetch_timeout_per_page = 0.25
        self.hicache_storage_pass_prefix_keys = False
        # Buffer-only host memory mode (host RAM as transient GPU↔storage
        # staging, not an L2 tier); resolved in init_hicache, which also
        # constructs the pipeline collaborator (None = cache mode).
        self.host_memory_mode = "cache"
        self.buffer_pipeline: Optional[BufferModePipeline] = None
        # Session of the insert whose actions are being applied (see ``insert``).
        self.insert_session_id: Optional[str] = None
        # Write-back Full-leaf eviction fences (see _wb_evict_overlap_stream).
        self._wb_evict_pd_prefill = False  # Set by init_hicache.
        self._wb_evict_relaxed_fence_logged = False
        self._wb_overlap_in_scope = not self.is_swa_enabled and not isinstance(
            self.token_to_kv_pool_allocator,
            (UnifiedSWAAllocatorBase, UnifiedMambaTokenToKVPoolAllocator),
        )
        self._wb_overlap_stream: Optional[torch.Stream] = None
        self._wb_overlap_max_seen_id = -1
        self._wb_overlap_watermark = -1
        self._wb_overlap_barrier_seq = 0
        self._wb_overlap_counts = {"overlap": 0, "fresh": 0, "scope": 0}
        # Batched write_back eviction (_backup_full_leaf_batched): host locks of
        # victims whose queued D2H copies the walk has not waited for yet.
        self._wb_batch_locks: list[tuple[NodeId, DecLockRefParams]] = []
        self._wb_batch_stream: Optional[torch.Stream] = None
        self._wb_batch_stats = [0, 0, 0]  # waits, victims, victims waited unlocked
        # Write-side dedupe: beliefs about what storage already holds, so
        # re-inserts of hot prefixes skip the redundant backup.
        self.storage_existence_cache = StorageExistenceCache()
        # Cumulative prefetch-outcome counters, exported through the
        # log_storage_metrics flow.
        self._prefetch_outcome_stats: dict[str, float] = {
            "attempts": 0,
            "issued": 0,
            "issued_assumed_stored": 0,
            "declined_too_short": 0,
            "declined_rate_limited": 0,
            "declined_anchor_lost": 0,
            "declined_device_covered": 0,
            "declined_host_oversize": 0,
            "declined_unservable_prefix": 0,
            "revoked_insufficient": 0,
            "revoked_full_miss": 0,
            "l3_demand_requests": 0,
            "l3_miss_tokens": 0,
            "l1l2_miss_tokens": 0,
            "full_miss_kv_absent": 0,
            "full_miss_kv_present": 0,
            # Lookups anchored past the deepest state (SGLANG_HICACHE_L3_READ_PAST_TREE_KV),
            # their hits, and the tokens those hits make reusable (tree KV + L3).
            "issued_past_tree_kv": 0,
            "hits_past_tree_kv": 0,
            "hit_tokens_past_tree_kv": 0,
            "lookups_logged": 0,
        }
        # Exclusive L2->L3 tiering: storage is written from the coldest host pages ahead
        # of eviction, not at L2 admission. Resolved in init_hicache.
        self._l3_write_on_host_evict = False
        self._l3_evict_write_reserve_fraction = 0.0
        # op id -> tokens of write-behind backups not yet acked; they count as
        # covered reserve so a slow ack does not re-issue deeper into the tail.
        self._write_behind_inflight: dict[int, int] = {}
        # Hybrid trees: op id -> node of a Mamba host-leaf write-behind backup in flight.
        self._write_behind_mamba_inflight: dict[int, NodeId] = {}
        # Mamba host states kept free or behind written leaves; 0 disables the walk.
        self._l3_mamba_leaf_reserve = 0
        # Step counter (not wall clock: the walk must run on every rank in the
        # same steps to keep backup issue order identical).
        self._write_behind_step = 0
        # Tail tokens the last walk found already in storage; they count as
        # covered until host eviction consumes them, so a steady state with a
        # fully tiered tail does not re-walk the heap every step.
        self._write_behind_clean_tokens = 0
        self._l3_wb_ack_credit = False
        self._l3_wb_walk_every = 4
        self._l3_stats_logged_at = 0.0
        # Belief TTL (0: off): PP0's clock picks the rotation step for every rank.
        self._l3_belief_ttl_s = 0.0
        self._l3_beliefs_rotated_at = time.monotonic()
        self._l3_tier_stats: dict[str, int] = {
            "wb_walks": 0,
            "wb_walk_us": 0,
            "wb_issued_tokens": 0,
            "wb_clean_tokens": 0,
            "wb_unbacked_tokens": 0,
            # Host evictions past the tail the walk knows is in storage: pages that left L2 without an L3 copy.
            "wb_evict_uncovered_tokens": 0,
            "wb_mamba_walks": 0,
            "belief_rotations": 0,
            "wb_mamba_leaf_issued_tokens": 0,
            # Diagnostics: write-behind ops by whether they carry a KDA state.
            "wb_full_ops_state": 0,
            "wb_full_ops_nostate": 0,
            "wb_full_tokens_state": 0,
            "wb_full_tokens_nostate": 0,
            "wb_mamba_ops_state": 0,
            "wb_mamba_ops_nostate": 0,
            "trace_state_key_not_last_page": 0,
            # Advisory pre-peek classification, not actual eviction/write outcomes.
            # The lifetime ledger records actual tombstones and their storage ACKs.
            "trace_mamba_evict_leaf_written": 0,
            "trace_mamba_evict_leaf_inflight": 0,
            "trace_mamba_evict_leaf_unwritten": 0,
            "trace_mamba_evict_leaf_unwritten_tokens": 0,
            "trace_mamba_evict_tombstone": 0,
            # Diagnostics: pages and states PP0 issues again after an earlier write, by the
            # age of that earlier write (the store may still hold them, or have dropped them).
            "trace_rewrite_pages_lt10m": 0,
            "trace_rewrite_pages_ge10m": 0,
            "trace_rewrite_states_lt10m": 0,
            "trace_rewrite_states_ge10m": 0,
            "trace_written_pages": 0,
            "trace_written_states": 0,
            # Diagnostics: every lookup (>= 8192 tokens) after PP0's first L3 write, by when PP0
            # wrote the first queried page (KV absent) or a state inside the found span (KV present).
            "trace_absent_never": 0,
            "trace_absent_lt7m": 0,
            "trace_absent_7to15m": 0,
            "trace_absent_ge15m": 0,
            "trace_present_state_lt7m": 0,
            "trace_present_state_older": 0,
            "trace_present_no_state": 0,
            # KV-absent lookups whose first page PP0 never wrote but where it wrote a later page
            # and a state of the span less than 7 min ago: the ancestors stayed in the tree and
            # only the tail went to L3 (a read that starts at the tree's KV end would hit).
            "trace_absent_tail_in_l3": 0,
            "trace_absent_tail_in_l3_tokens": 0,
        }
        self._l3_last_write_has_state = False
        self._l3_trace: Optional[dict] = None

        self.reset()
        logger.info(
            f"Init Unified Radix Cache. Components: {self.tree_components}. "
            f"Tree Core: {type(self.tree_core).__name__}"
        )

    def _all_reduce_attn_groups(self, tensor: torch.Tensor, op):
        reduced = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.all_reduce(tensor, op=op, group=group)
                reduced = True
        if not reduced and self.tp_world_size > 1:
            torch.distributed.all_reduce(tensor, op=op, group=self.tp_group)

    def _drain_async_work(self):
        """
        Block until all outstanding async sends are consumed, then clear.

        Called at the start of each event round, so work_list holds the sends
        accumulated since the last round. This bounds it and applies
        backpressure when a downstream PP rank lags. Scheduler thread only.
        With SGLANG_PP_SYNC_WINDOW, only rounds older than pp_sync_window are waited.
        """
        with pp_bubble_diag.wait("drain"):
            if self.pp_sync_window > 1:
                self.pp_sync_rounds.append(self.work_list)
                self.work_list = []
                while len(self.pp_sync_rounds) >= self.pp_sync_window:
                    for work in self.pp_sync_rounds.popleft():
                        work.wait()
                return
            for work in self.work_list:
                work.wait()
            self.work_list.clear()

    def _all_reduce(self, data: torch.Tensor, tp_reduce_op: torch.distributed.ReduceOp):
        """
        Synchronize data across all TP and PP ranks.

        In particular, "tp_reduce_op" is performed on all TP ranks of the first PP rank,
        and then the result is propagated to all following PP ranks.

        Must be called in the scheduler thread.
        """
        if self.pp_rank == 0:
            self._all_reduce_attn_groups(data, tp_reduce_op)
        self._pp_sync(data)

    def agree_on_all_ranks(self, flags: list[bool]) -> list[bool]:
        """AND of per-rank flags with the same reduction as the drained counts
        (TP on PP0, then propagated along PP), so every rank decides alike."""
        data = torch.tensor(flags, dtype=torch.int)
        self._all_reduce(data, torch.distributed.ReduceOp.MIN)
        return [bool(flag) for flag in data.tolist()]

    def _pp_sync(self, data: torch.Tensor) -> None:
        """
        Synchronize data across the PP pipeline, where PPn (n>0) will receive PP0's data.
        """
        if self.pp_size <= 1 or self.pp_group is None:
            return
        if self.pp_rank > 0:
            with pp_bubble_diag.relay_wait():
                torch.distributed.recv(
                    data,
                    group_src=self.pp_rank - 1,
                    group=self.pp_group,
                    tag=P2PTag.HICACHE_PP_SYNC,
                )
        if self.pp_rank + 1 < self.pp_size:
            copy_of_data = data.clone()
            send_work = torch.distributed.isend(
                copy_of_data,
                group_dst=self.pp_rank + 1,
                group=self.pp_group,
                tag=P2PTag.HICACHE_PP_SYNC,
            )
            self.work_list.append(send_work)

    def init_cache_linker(self, cache_linker: UnifiedCacheLinker) -> None:
        """Attach an external KV store directly to the device pools."""
        self.linker = UnifiedCacheLinkerWrapper(self, cache_linker)

    def reset(self) -> None:
        if self.linker is not None:
            self.linker.reset()
        self._reset_full()

    def _reset_full(self) -> None:
        """Full reset: destroy entire tree and all state."""
        # In-flight restores still read their host slots; finish them first.
        self._discard_restore_backups()
        self.tree_core.reset()
        self.session_refs.reset()

        # Reset Controller.
        self.session.slots.clear()
        self.ongoing_write_through: dict[int, _OngoingWriteThrough] = {}
        # Device write-ahead batches: locked until their rank-agreed ack commits.
        self.ongoing_write_ahead: dict[int, _OngoingWriteThrough] = {}
        self._wa_issue_steps: deque[int] = deque()
        self._wa_step = 0
        self._wa_evicted_tokens = 0
        self._wa_peak_evicted = 0.0
        self._wa_ready_count = 0
        self.ongoing_load_back: dict[int, _OngoingLoadBack] = {}
        self.enable_storage = False
        self.prefetch_loaded_tokens_by_reqid: dict[CacheRequestHandle, int] = {}
        self.prefetch_loaded_storage_start_by_reqid: dict[CacheRequestHandle, int] = {}
        self.ongoing_prefetch: dict[CacheRequestHandle, _OngoingPrefetch] = {}
        self._mamba_state_restores: dict[
            CacheRequestHandle, MambaStateRestoreOperation
        ] = {}
        self._storage_prefetch_wait: dict[CacheRequestHandle, _StoragePrefetchWait] = {}
        # Rank-agreed L3-hit tokens not yet resolved as usable or unfulfilled.
        # Cache-mode entries survive L3->L2 until H2D succeeds or admission
        # fails; buffer-mode entries survive staging until the H2D ack.
        self._storage_prefetch_hit_remaining_by_reqid: dict[
            CacheRequestHandle, int
        ] = {}
        self.storage_prefetch_retries = StoragePrefetchRetries()
        self.ongoing_backup: dict[int, tuple[NodeId, DecLockRefParams]] = {}
        if self._storage_write_back is not None:
            self._storage_write_back.reset()
        self._write_behind_inflight.clear()
        self._write_behind_mamba_inflight.clear()
        self._write_behind_clean_tokens = 0
        self._reset_ack_lag()
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.reset()
            # The dropped writes' content no longer reaches a storage ack.
            self.storage_existence_cache.clear_inflight()

        if self.cache_controller is not None:
            self.cache_controller.reset()
            self.cache_controller.mem_pool_host.clear()
            self.enable_storage = self.cache_controller.enable_storage

        self.tree_core.kv_events.record_all_cleared()

    def init_hicache(self, server_args: ServerArgs, params: CacheInitParams) -> None:
        """Initialize HiCache infrastructure."""
        self.host_memory_mode = get_memory().hicache_host_memory_mode
        if self.host_memory_mode == "buffer_only":
            supported = {
                ComponentType.FULL,
                ComponentType.SWA,
                ComponentType.AUXILIARY_SWA,
                ComponentType.MAMBA,
            }
            if not set(self.tree_components) <= supported:
                raise ValueError(
                    "--hicache-host-memory-mode buffer_only supports only "
                    "FULL/SWA/AUXILIARY_SWA/MAMBA unified trees; got components "
                    f"{sorted(ct.name for ct in self.tree_components)}."
                )
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            attach_hybrid_pool_to_unified_cache,
        )

        self.load_cache_event = threading.Event()
        self.sidecar_pool_specs.clear()
        self.extra_metric_labels = get_observability().extra_metric_labels
        self._storage_attachment = StorageAttachment(self)

        # Parse storage config once, share with assembler and tree
        storage_backend = get_memory().hicache_storage_backend
        storage_extra_config = None
        storage_prefetch_threshold = 256
        prefetch_timeout_base = 1.0
        prefetch_timeout_per_ki_token = 0.25
        hicache_storage_pass_prefix_keys = False
        if storage_backend is not None:
            (
                storage_extra_config,
                storage_prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = HybridCacheController.parse_storage_backend_extra_config(
                get_memory().hicache_storage_backend_extra_config
            )
            if server_args.file_storage_path:
                storage_extra_config = dict(storage_extra_config or {})
                storage_extra_config.setdefault(
                    "file_storage_path", server_args.file_storage_path
                )
        storage_prefetch_threshold = (
            self._storage_attachment.resolve_prefetch_threshold(
                storage_prefetch_threshold
            )
        )

        attach_hybrid_pool_to_unified_cache(
            self,
            params,
            server_args,
            load_cache_event=self.load_cache_event,
            storage_backend=storage_backend,
            storage_extra_config=storage_extra_config,
            storage_prefetch_threshold=storage_prefetch_threshold,
        )
        # Tag HiCache enablement on the TreeCore.
        if self.cache_controller is not None:
            self.tree_core.set_hicache_enabled()
            if self.supports_swa():
                swa = self.components[ComponentType.SWA]
                self.tree_core.has_swa_host_pool = swa._swa_kv_pool_host is not None

        self._l3_write_on_host_evict = bool(
            envs.SGLANG_HICACHE_L3_WRITE_ON_HOST_EVICT.get()
            and self.cache_controller is not None
            and self.cache_controller.enable_storage
        )
        self._l3_saved_token_threshold = (
            envs.SGLANG_HICACHE_L3_SAVED_TOKEN_THRESHOLD.get()
        )
        self._inflight_admission_waits = max(
            0, envs.SGLANG_HICACHE_L3_INFLIGHT_ADMISSION_WAITS.get()
        )
        self._l3_wb_ack_credit = envs.SGLANG_HICACHE_L3_WB_ACK_CREDIT.get()
        self._l3_wb_walk_every = max(1, envs.SGLANG_HICACHE_L3_WB_WALK_EVERY.get())
        self._l3_belief_query_renew = envs.SGLANG_HICACHE_L3_BELIEF_QUERY_RENEW.get()
        self._init_ack_lag()
        self._l3_write_state_on_tombstone = (
            envs.SGLANG_HICACHE_L3_WRITE_STATE_ON_TOMBSTONE.get()
        )
        if self._l3_write_state_on_tombstone:
            if (
                self.host_memory_mode != "cache"
                or not self.is_mamba_enabled
                or self.cache_controller is None
                or not self.cache_controller.enable_storage
            ):
                raise ValueError(
                    "SGLANG_HICACHE_L3_WRITE_STATE_ON_TOMBSTONE requires "
                    "a Mamba host cache and storage backend"
                )
            # Fail at initialization with an extension missing the two-phase walk.
            self.tree_core.begin_mamba_host_eviction(0, min_depth=8192)
            if envs.SGLANG_HICACHE_L3_TOMBSTONE_ASYNC.get():
                self._l3_tombstone_pending = {}
                self._l3_tombstone_reserve = max(0, envs.SGLANG_HICACHE_L3_TOMBSTONE_RESERVE.get())
                logger.info(
                    "HiCache L3 state tombstones written asynchronously (reserve %d host states).",
                    self._l3_tombstone_reserve,
                )
        self._l3_lifetime_ledger = None
        if self.pp_rank == 0 and envs.SGLANG_HICACHE_L3_LIFETIME_LEDGER.get():
            self._l3_lifetime_ledger = L3LifetimeLedger.from_env(
                self.pp_rank, get_parallel().tp_rank
            )
            if self._l3_lifetime_ledger is not None:
                self.tree_core.set_l3_lifetime_ledger_enabled(True)
        if self._l3_write_on_host_evict:
            if self.host_memory_mode == "buffer_only":
                raise ValueError(
                    "SGLANG_HICACHE_L3_WRITE_ON_HOST_EVICT needs a host cache tier; "
                    "it does not apply to --hicache-host-memory-mode buffer_only"
                )
            # Fails here, not at the first walk, on a Rust extension without the peek.
            self.tree_core.peek_host_eviction_candidates(BASE_COMPONENT_TYPE, 0)
            self._l3_evict_write_reserve_fraction = (
                envs.SGLANG_HICACHE_L3_EVICT_WRITE_RESERVE_FRACTION.get()
            )
            self._l3_belief_ttl_s = envs.SGLANG_HICACHE_L3_BELIEF_TTL_S.get()
            logger.info(
                "HiCache L3 write-on-host-evict enabled: reserve fraction %.3f "
                "of host pool (%d tokens)",
                self._l3_evict_write_reserve_fraction,
                int(
                    self._l3_evict_write_reserve_fraction
                    * self.cache_controller.mem_pool_host.size
                ),
            )
            self._l3_tier_final = WriteOnHostEvictFinalStats.register(self)
            if envs.SGLANG_HICACHE_L3_TRACE.get() and self.pp_rank == 0:
                self._l3_trace = dict(kv={}, state={}, writes_logged=0, absent_logged=0,
                                      present_logged=0, queried_logged=0)
            if (
                envs.SGLANG_HICACHE_L3_WRITE_MAMBA_HOST_LEAVES.get()
                and self.is_mamba_enabled
            ):
                self.tree_core.peek_mamba_host_eviction_leaves(0)
                states = self.cache_controller.mem_pool_host.get_pool(
                    PoolName.MAMBA
                ).size
                self._l3_mamba_leaf_reserve = max(
                    1, int(self._l3_evict_write_reserve_fraction * states)
                )
                logger.info(
                    "HiCache L3 write-on-host-evict: also writing the host leaves "
                    "among the %d coldest of %d Mamba host states",
                    self._l3_mamba_leaf_reserve,
                    states,
                )

        if self.host_memory_mode == "buffer_only":
            self.tree_core.set_host_memory_buffer_only()
            swa = self.components.get(ComponentType.SWA)
            auxiliary_swa = self.components.get(ComponentType.AUXILIARY_SWA)
            validate_buffer_only_stack(
                sidecar_pool_specs=self.sidecar_pool_specs,
                host_pool_group=self.host_pool_group,
                swa_component=swa,
                auxiliary_swa_component=auxiliary_swa,
                storage_prefetch_threshold=storage_prefetch_threshold,
            )
            self.buffer_pipeline = self.components[
                ComponentType.FULL
            ].create_buffer_pipeline(
                max_context_len=get_model().context_length or 0,
                swa_component=swa,
                auxiliary_swa_component=auxiliary_swa,
                # Leak backstop only: the queue is charged once per unique node
                # span across pools; a cap binding on live content would
                # drop-newest and punch storage holes.
                write_backlog_cap=2 * self.token_to_kv_pool_allocator.size_full,
            )
        # State initialization
        if self.buffer_pipeline is not None:
            self.cache_controller.host_write_staged_tokens_fn = lambda: (
                self.buffer_pipeline.write_staged_tokens_
            )
            # Pools a component's backups stage through besides the stock
            # ones (an entry may alias the anchor's host pool).
            for component in self.components.values():
                for entry in component.buffer_mode_host_pool_entries(
                    self.host_pool_group
                ):
                    self.cache_controller.register_host_pool_entry(entry)
        self.write_through_threshold = (
            1 if get_memory().hicache_write_policy == "write_through" else 2
        )
        self.is_write_back = (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        )
        self._wb_evict_pd_prefill = (
            get_disagg().disaggregation_mode == "prefill"
            and envs.SGLANG_HICACHE_WB_EVICT_RELAXED_FENCE.get()
        )
        self._wb_evict_batch = (
            self.is_write_back
            and self.buffer_pipeline is None
            and self._wb_overlap_in_scope
            and envs.SGLANG_HICACHE_WB_EVICT_BATCH.get()
        )
        if self._wb_evict_batch:
            logger.info(
                "HiCache write_back eviction: one D2H wait per eviction walk "
                "(SGLANG_HICACHE_WB_EVICT_BATCH)."
            )
        self._device_write_ahead = (
            self.is_write_back
            and self.buffer_pipeline is None
            and envs.SGLANG_HICACHE_DEVICE_WRITE_AHEAD.get()
        )
        self._write_ahead_stream_ = None
        if self._device_write_ahead:
            logger.info(
                "HiCache write_back: device LRU tail written ahead of eviction "
                "(SGLANG_HICACHE_DEVICE_WRITE_AHEAD, K=%.1f, tail %d..%d tokens).",
                envs.SGLANG_HICACHE_DEVICE_WRITE_AHEAD_K.get(),
                envs.SGLANG_HICACHE_DEVICE_WRITE_AHEAD_MIN_TOKENS.get(),
                envs.SGLANG_HICACHE_DEVICE_WRITE_AHEAD_MAX_TOKENS.get(),
            )
        self._init_mamba_path_cap_counts_host()
        self._init_mamba_demote_watch()
        self._storage_write_back = StorageWriteBack.create(self)
        # Pre-seed the logical dropped-tokens series.
        if self.metrics_collector is not None and self.cache_controller is not None:
            reasons = ["host_pressure"]
            if self._tracks_write_through_unbacked_evictions():
                reasons.append("write_through_unbacked_eviction")
            for reason in reasons:
                self.metrics_collector.increment_dropped_tokens(
                    num_tokens=0,
                    reason=reason,
                    pool=PoolName.KV.value,
                )
        self.load_back_threshold = 10
        self.prefetch_stop_policy = get_memory().hicache_storage_prefetch_policy

        # Runtime attach/detach of the L3 backend (startup, admin API, atexit).
        atexit.register(self.shutdown)

        if storage_backend is not None:
            self._storage_attachment.apply_runtime_config(
                storage_backend=storage_backend,
                prefetch_threshold=storage_prefetch_threshold,
                prefetch_timeout_base=prefetch_timeout_base,
                prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
                enable_storage=self.cache_controller.enable_storage,
                enable_storage_metrics=self._enable_metrics_flag,
                extra_metric_labels=self.extra_metric_labels,
            )

    def _init_mamba_path_cap_counts_host(self) -> None:
        """SGLANG_MAMBA_PATH_CAP_COUNTS_HOST: --mamba-max-states-per-path also
        counts host copies, and a capped state loses its host copy. Off by default."""
        if not envs.SGLANG_MAMBA_PATH_CAP_COUNTS_HOST.get():
            return
        cap = (
            self.components[ComponentType.MAMBA].mamba_max_states_per_path
            if self.is_mamba_enabled
            else -1
        )
        if (
            cap < 0
            or self.cache_controller is None
            or self.host_memory_mode == "buffer_only"
        ):
            logger.info(
                "SGLANG_MAMBA_PATH_CAP_COUNTS_HOST ignored: it needs a Mamba tree with "
                "--mamba-max-states-per-path and a HiCache host tier in cache mode"
            )
            return
        self.tree_core.set_mamba_path_cap_counts_host(True)
        self._mamba_path_cap_counts_host = True
        self.mamba_retention_counts = {"path_cap_host_frees": 0}
        logger.info(
            "Mamba path cap counts host copies (cap %d, tree core %s)",
            cap,
            type(self.tree_core).__name__,
        )

    def _init_mamba_demote_watch(self) -> None:
        """A write_back demote of a leaf whose Full KV is already on host drops a
        device-only Mamba state: count those (SGLANG_LOG_MAMBA_DEMOTE_DROPS), or
        copy the state to host first (SGLANG_HICACHE_MAMBA_BACKUP_BEFORE_DEMOTE)."""
        backup = envs.SGLANG_HICACHE_MAMBA_BACKUP_BEFORE_DEMOTE.get()
        if not (backup or envs.SGLANG_LOG_MAMBA_DEMOTE_DROPS.get()):
            return
        if (
            not self.is_mamba_enabled
            or self.cache_controller is None
            or self.host_memory_mode == "buffer_only"
            or not self.is_write_back
        ):
            logger.info(
                "Mamba demote watch ignored: it needs a Mamba tree with a "
                "write_back HiCache host tier in cache mode"
            )
            return
        self._mamba_demote_watch = True
        self._mamba_backup_before_demote = backup
        self.mamba_retention_counts = {
            **(self.mamba_retention_counts or {}),
            "demote_state_drops": 0,
            "demote_state_copies": 0,
            "demote_copy_restores": 0,
            # Load-backs of a host-only Mamba state refused, by reason; and those
            # where the request then continues from a device anchor past the root.
            "lb_refused_held": 0,
            "lb_refused_quota": 0,
            "lb_refused_capacity": 0,
            "lb_refused_alloc": 0,
            "lb_refused_nonzero_prefix": 0,
        }
        self._demote_copied_nodes = set()
        logger.info("Mamba demote watch on, back up before demote %s", backup)

    def _count_mamba_load_back_refusal(self, req, reason: str) -> None:
        """A refused load-back of a request that matched a host-only Mamba state."""
        if (
            self._mamba_demote_watch
            and req is not None
            and req.mamba_host_hit_length > 0
        ):
            self.mamba_retention_counts[f"lb_refused_{reason}"] += 1

    def _watch_backed_leaf_state(self, node_id: NodeId) -> None:
        """Before a backed leaf's demote: copy its device-only Mamba state to host
        and land the copy, or count the state the demote drops."""
        if not self.tree_core.is_backuped(node_id):
            return
        _, comp_xfers = self.tree_core.build_backup_spec(node_id)
        if ComponentType.MAMBA not in comp_xfers:
            return
        counts = self.mamba_retention_counts
        if self._mamba_backup_before_demote:
            mamba_host_pool = self.host_pool_group.get_pool(PoolName.MAMBA)
            if mamba_host_pool is not None and mamba_host_pool.available_size() < 1:
                self.evict_host(1, ComponentType.MAMBA)
            if self.backup_node_for_write_back(node_id):
                counts["demote_state_copies"] += 1
                self._demote_copied_nodes.add(node_id)
                return
        counts["demote_state_drops"] += 1

    def register_sidecar_pool(
        self, spec: SidecarPoolSpec, entry: Optional[PoolEntry] = None
    ) -> None:
        if entry is not None:
            if self.cache_controller is None:
                raise RuntimeError("HiCache controller is not attached.")
            self.cache_controller.register_host_pool_entry(entry)
        self.sidecar_pool_specs.append(spec)

    def release_host_resources(self) -> None:
        if self.linker is not None:
            self.linker.close()
        if self.host_pool_group is not None:
            self.host_pool_group.destroy()

    @rank_consensus(
        same_params=["params"],
        same_results=["result.full_kv_hit_length", "result.swa_host_hit_length"],
    )
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        result = self.session.try_match_prefix(params)
        if result is not None:
            return result
        return self._match_tree(params)

    def _match_tree(self, params: MatchPrefixParams) -> MatchResult:
        if self.disable:
            return self.tree_core.empty_match_result
        if params.full_kv_only:
            assert self.supports_full_kv_only_match()
            return self._match_full_kv_only(params)
        result = self.tree_core.match_prefix(params)
        # Apply the walk's actions (e.g. a pending write-through relocation on
        # a split) before the finalizers, which can evict or raise.
        self._apply_cache_actions(result.cache_actions)
        return self._finalize_match_result(params, result)

    def _finalize_match_result(
        self, params: MatchPrefixParams, result: MatchResult
    ) -> MatchResult:
        for component in self._components_tuple:
            result = component.finalize_match_result_in_cache(params, result)
        # Finalizers must not emit actions; the walk's were applied above.
        assert not result.cache_actions
        # PD decode has no linker load-back path, so it only offloads.
        if (
            self.linker is not None
            and params.req is not None
            and get_disagg().disaggregation_mode != "decode"
        ):
            result = self.linker.match(params.key, params.req, result)
        return result

    def supports_full_kv_only_match(self) -> bool:
        # Host or external-store hits carry Mamba-gated load plans.
        return (
            (self.is_mamba_enabled or self.request_mamba_only)
            and not self.disable
            and self.cache_controller is None
            and self.linker is None
        )

    def _match_full_kv_only(self, params: MatchPrefixParams) -> MatchResult:
        """The deepest FULL-KV device hit, past the last cached Mamba state. The
        caller supplies the request's Mamba state, so nothing is copied from the
        tree. The regular walk still splits the matched node and refreshes LRU."""
        assert not params.cow_mamba, "a FULL-KV-only hit has no Mamba state to copy"
        result = self.tree_core.match_prefix(params)
        self._apply_cache_actions(result.cache_actions)
        matched_len, node, _ = self.tree_core.match_full_device_prefix(params.key)
        result = result._replace(cache_actions=())
        if matched_len <= result.device_prefix_len:
            return result
        return result._replace(
            device_prefix_len=matched_len,
            last_device_node=node,
            last_host_node=node,
            best_match_node=node,
            mamba_branching_seqlen=None,
        )

    def supports_fast_match_prefix(self) -> bool:
        return self.tree_core.supports_fast_match_prefix()

    def peek_prefix_len(self, key: RadixKey) -> Optional[int]:
        if self.disable:
            return 0
        return self.tree_core.peek_prefix_len(key)

    def touch_prefix(self, key: RadixKey) -> None:
        # Tree walk only; skips match_prefix's session shortcut, finalizers and linker.
        if self.disable:
            return
        result = self.tree_core.match_prefix(MatchPrefixParams(key=key))
        self._apply_cache_actions(result.cache_actions)

    def supports_prefix_sharing(self) -> bool:
        return not self.disable

    @rank_consensus(
        same_params=["len(params.key)"],
        same_results=["result.prefix_len"],
    )
    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)
        # Fail fast on re-entrancy without touching the in-flight walk.
        assert not self.tree_core.has_ongoing_insert(), "re-entrant insert"
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.begin_insert_walk(
                is_request_end=params.is_request_end, key_len=len(params.key)
            )
        # External linker offloads (BackupKV) issued by this insert carry its session.
        self.insert_session_id = params.session_id
        # Pump the resumable insert, applying each step's actions at its barrier.
        try:
            step = self.tree_core.begin_insert(params)
            while True:
                self._apply_cache_actions(step.actions)
                if step.result is not None:
                    # Walk actions flow through the steps; the result is action-free.
                    assert not step.result.cache_actions
                    if self.buffer_pipeline is not None:
                        self.buffer_pipeline.on_insert_complete(step.result)
                    self._wb_overlap_observe(step.result.last_device_node)
                    return step.result
                step = self.tree_core.resume_insert()
        finally:
            # Drain still-pending actions so frees reach the allocator on abort.
            try:
                self._apply_cache_actions(self.tree_core.end_insert())
            finally:
                self.insert_session_id = None
            if self.buffer_pipeline is not None:
                self.buffer_pipeline.end_insert_walk()

    @rank_consensus(same_params=True, same_results=True)
    def evict(self, params: EvictParams) -> EvictResult:
        return self._evict(params)

    @rank_consensus(
        same_params=["params", "allocation_reclaim_satisfied is not None"],
        same_results=True,
    )
    def evict_for_alloc(
        self,
        params: EvictParams,
        *,
        allocation_reclaim_satisfied: Optional[Callable[[], bool]] = None,
    ) -> EvictResult:
        """Evict using component shortfalls or cumulative joint-reclaim quotas.

        Without ``allocation_reclaim_satisfied``, ``params`` contains component
        shortfalls. These become availability targets, allowing collateral frees
        to satisfy an allocation before its requested eviction count is reached.

        With ``allocation_reclaim_satisfied``, ``params`` instead contains
        cumulative reclaim quotas and component-local targets are bypassed.
        The pure predicate may stop the walk before those quotas are reached;
        the allocator must prepare and recheck actual capacity after the walk.
        """
        if self.disable:
            return EvictResult()
        if allocation_reclaim_satisfied is not None:
            return self._evict(
                params, allocation_reclaim_satisfied=allocation_reclaim_satisfied
            )

        request_by_type = self._evict_request_by_type(params)
        available_size_targets = {
            ct: (ct, self._component_available_size(ct) + request_cnt)
            for ct, request_cnt in request_by_type.items()
            if request_cnt > 0
        }
        allocator = self.token_to_kv_pool_allocator
        mamba_full_donor = allocator.mamba_full_cache_donor()
        mamba_target = available_size_targets.get(ComponentType.MAMBA)
        initial_params = params
        if mamba_target is not None and mamba_full_donor is not None:
            # Full KV can supply bytes but cannot recycle Mamba virtual IDs.
            mamba_id_shortfall = max(
                0,
                mamba_target[1]
                - self.req_to_token_pool.mamba_allocator.available_size(),
            )
            initial_params = EvictParams(
                num_tokens=params.num_tokens,
                swa_num_tokens=params.swa_num_tokens,
                mamba_num=mamba_id_shortfall,
            )
        result = self._evict(initial_params, available_size_targets)

        if mamba_target is not None and mamba_full_donor is not None:
            mamba_full_donor.prepare_mamba_allocation(mamba_target[1])
            mamba_free_ids = self.req_to_token_pool.mamba_allocator.available_size()
            mamba_capacity = self._component_available_size(ComponentType.MAMBA)

            if mamba_free_ids >= mamba_target[1] and mamba_capacity < mamba_target[1]:
                full_evictable = self.full_evictable_size()
                if full_evictable > 0:
                    donor_result = self._evict(
                        EvictParams(num_tokens=full_evictable),
                        {ComponentType.FULL: mamba_target},
                    )
                    result.num_tokens_evicted += donor_result.num_tokens_evicted
                    result.swa_num_tokens_evicted += donor_result.swa_num_tokens_evicted
                    result.mamba_num_evicted += donor_result.mamba_num_evicted

                # Preserve Mamba-victim recovery if Full cannot fund the target.
                if (
                    self._component_available_size(ComponentType.MAMBA)
                    < mamba_target[1]
                ):
                    mamba_evictable = self.mamba_evictable_size()
                    if mamba_evictable > 0:
                        fallback_result = self._evict(
                            EvictParams(mamba_num=mamba_evictable),
                            {ComponentType.MAMBA: mamba_target},
                        )
                        result.num_tokens_evicted += fallback_result.num_tokens_evicted
                        result.swa_num_tokens_evicted += (
                            fallback_result.swa_num_tokens_evicted
                        )
                        result.mamba_num_evicted += fallback_result.mamba_num_evicted

        return result

    @staticmethod
    def _evict_request_by_type(params: EvictParams) -> dict[ComponentType, int]:
        return {
            ComponentType.FULL: params.num_tokens,
            ComponentType.SWA: params.swa_num_tokens,
            ComponentType.MAMBA: params.mamba_num,
            ComponentType.C128: 0,
        }

    def _component_available_size(self, component_type: ComponentType) -> int:
        """Return capacity usable by the component's next allocation.

        Shared allocators expose schedulable capacity, which includes peer holes
        that an urgent allocator flush can reclaim without further eviction.
        """
        if component_type == ComponentType.FULL:
            if self.supports_swa():
                return self.token_to_kv_pool_allocator.full_available_size()
            return self.token_to_kv_pool_allocator.available_size()
        if component_type == ComponentType.SWA:
            return self.token_to_kv_pool_allocator.swa_available_size()
        if component_type == ComponentType.MAMBA:
            return self.req_to_token_pool.mamba_allocator.schedulable_available_size()
        raise ValueError(f"Unsupported cache component: {component_type}")

    def _evict(self, params: EvictParams, *args, **kwargs) -> EvictResult:
        # Eviction frees unlocked nodes only, never an in-flight transfer's pages: apply them
        # now even under defer_device_frees, or an allocation that evicts to make room fails.
        sink = getattr(self, "_deferred_free_sink", None)
        if sink is None:
            return self._evict_impl(params, *args, **kwargs)
        self._deferred_free_sink = None
        try:
            return self._evict_impl(params, *args, **kwargs)
        finally:
            self._deferred_free_sink = sink

    def _evict_impl(
        self,
        params: EvictParams,
        available_size_targets: Optional[
            dict[ComponentType, tuple[ComponentType, int]]
        ] = None,
        *,
        allocation_reclaim_satisfied: Optional[Callable[[], bool]] = None,
    ) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        tracker = {ct: 0 for ct in self.tree_components}

        if self._wb_evict_relaxed_fence():
            self._flush_queued_writes_fenced()
        request_by_type = self._evict_request_by_type(params)
        self._evict_components(
            request_by_type,
            tracker,
            available_size_targets=available_size_targets,
            allocation_reclaim_satisfied=allocation_reclaim_satisfied,
        )

        if (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        ):
            self.writing_check(write_back=True)
        if self._device_write_ahead:
            self._wa_evicted_tokens += tracker[BASE_COMPONENT_TYPE]

        # Report full-layer tokens only
        self.update_eviction_metrics(tracker[BASE_COMPONENT_TYPE], start_time)
        return EvictResult(
            num_tokens_evicted=tracker[BASE_COMPONENT_TYPE],
            swa_num_tokens_evicted=tracker.get(ComponentType.SWA, 0),
            mamba_num_evicted=tracker.get(ComponentType.MAMBA, 0),
        )

    def _free_values(
        self,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Free a tree-side step's returned device and host values right away."""
        # Both drains must run even if one raises.
        try:
            self._drain_device_frees(device_frees)
        finally:
            self._drain_host_frees(host_frees)

    def _apply_eviction_result(self, result: BaseEvictionResult) -> None:
        """Drain a tree step before the next allocator-dependent decision."""
        try:
            self._free_values(result.device_frees, result.host_frees)
        finally:
            self._apply_cache_actions(result.cache_actions)
            self._drain_l3_lifetime_events()

    def _drain_l3_lifetime_events(self) -> None:
        if self._l3_lifetime_ledger is not None:
            self._l3_lifetime_ledger.drain_tree(self.tree_core)

    def _accumulate_tracker(
        self,
        tracker: dict[ComponentType, int],
        delta: dict[ComponentType, int],
    ) -> None:
        """Fold a step result's evicted delta into the running totals."""
        for ct, n in delta.items():
            tracker[ct] += n

    def _evict_device_next_node(
        self, component_type: ComponentType, tracker: dict[ComponentType, int]
    ) -> tuple[Optional[NodeId], bool]:
        """Advance the eviction walk one node, consuming its step result."""
        result = self.tree_core.evict_device_next_node(component_type, tracker)
        if result.mamba_backup_node_id is not None:
            assert component_type == ComponentType.MAMBA and result.node_id is None
            assert (
                not result.device_frees
                and not result.host_frees
                and not result.tracker
                and not result.cache_actions
            )
            # Reserve a host state slot and wait for the backup acknowledgment
            # before freeing device state. If allocation fails, eviction still
            # proceeds to make room on the device.
            node_id = result.mamba_backup_node_id
            mamba_host_pool = self.host_pool_group.get_pool(PoolName.MAMBA)
            if mamba_host_pool is not None and mamba_host_pool.available_size() < 1:
                self.evict_host(1, ComponentType.MAMBA)
            self.backup_node_for_write_back(node_id)
            result = self.tree_core.finish_mamba_state_eviction(node_id)
        elif result.swa_backup_node_id is not None:
            assert component_type == ComponentType.SWA and result.node_id is None
            assert (
                not result.device_frees
                and not result.host_frees
                and not result.tracker
                and not result.cache_actions
            )
            # The backup can cover several unbacked SWA segments. Reserve host
            # space for the whole window before copying it, then resume eviction
            # even if host allocation fails.
            node_id = result.swa_backup_node_id
            needed = result.swa_backup_num_tokens
            swa_host_pool = self.host_pool_group.get_pool(PoolName.SWA)
            if swa_host_pool is not None and swa_host_pool.available_size() < needed:
                self.evict_host(needed, ComponentType.SWA)
            self.backup_node_for_write_back(node_id)
            result = self.tree_core.finish_swa_state_eviction(node_id)
        self._apply_eviction_result(result)
        if self._tracks_write_through_unbacked_evictions():
            self._record_dropped_tokens(
                result.unbacked_tokens,
                reason="write_through_unbacked_eviction",
            )
        self._accumulate_tracker(tracker, result.tracker)
        return result.node_id, result.made_progress

    def _evict_device_leaf(
        self, node_id: NodeId, tracker: dict[ComponentType, int]
    ) -> Optional[BackupKV]:
        """Evict one device leaf, consuming its step result; returns the
        deferred write-back BackupKV when one must run before the demote."""
        result = self.tree_core.evict_device_leaf(node_id, self.is_write_back)
        self._apply_eviction_result(result)
        if self._tracks_write_through_unbacked_evictions():
            self._record_dropped_tokens(
                result.unbacked_tokens,
                reason="write_through_unbacked_eviction",
            )
        self._accumulate_tracker(tracker, result.tracker)
        return result.backup_kv

    def _demote(self, node_id: NodeId, tracker: dict[ComponentType, int]) -> None:
        """Demote a backed-up node, consuming its step result."""
        result = self.tree_core.demote(node_id)
        self._apply_eviction_result(result)
        self._accumulate_tracker(tracker, result.tracker)

    def _drop_subtree_no_host(
        self, node_id: NodeId, tracker: dict[ComponentType, int]
    ) -> bool:
        """Run the write-back drop fallback, consuming its step result."""
        result = self.tree_core.drop_subtree_no_host(node_id)
        self._apply_eviction_result(result)
        if result.is_dropped:
            self._record_dropped_tokens(
                result.tracker.get(BASE_COMPONENT_TYPE, 0), reason="host_pressure"
            )
        self._accumulate_tracker(tracker, result.tracker)
        return result.is_dropped

    def _evict_components(
        self,
        request_by_type: dict[ComponentType, int],
        tracker: dict[ComponentType, int],
        available_size_targets: Optional[
            dict[ComponentType, tuple[ComponentType, int]]
        ] = None,
        *,
        allocation_reclaim_satisfied: Optional[Callable[[], bool]] = None,
    ) -> None:
        # Buffer mode: eviction always wins over queued backup intents — a
        # destroyed victim's intent is stale-swept and the content rewrites
        # after its recompute.
        last_mamba_donor_check = 0
        mamba_donor_prepared = False

        def target_reached(component_type: ComponentType) -> bool:
            nonlocal last_mamba_donor_check, mamba_donor_prepared
            if allocation_reclaim_satisfied is not None:
                return allocation_reclaim_satisfied()
            if available_size_targets is None:
                return False
            target = available_size_targets.get(component_type)
            if target is None:
                return False
            target_component, target_size = target
            # A Full-leaf cascade can release Mamba or SWA state directly.
            if self._component_available_size(target_component) >= target_size:
                return True
            if (
                component_type == ComponentType.FULL
                and target_component == ComponentType.MAMBA
            ):
                donor = self.token_to_kv_pool_allocator.mamba_full_cache_donor()
                assert donor is not None, "Mamba target requires a Full donor"
                recheck_after = (
                    1
                    if mamba_donor_prepared
                    else donor.full_tokens_before_mamba_recheck(target_size)
                )
                if tracker[component_type] - last_mamba_donor_check < recheck_after:
                    return False
                donor.prepare_mamba_allocation(target_size)
                last_mamba_donor_check = tracker[component_type]
                mamba_donor_prepared = True
            # Schedulable capacity includes donor holes that allocation can compact.
            return self._component_available_size(target_component) >= target_size

        for ct in self.tree_components:
            request_cnt = request_by_type[ct]
            # An allocation-driven FULL request names the allocator's
            # shortfall, not what the walk must free: an allocator can hide
            # headroom behind a clamp (a reserve it refills before raw
            # allocation sees a page again) or page rounding, so freeing
            # exactly the shortfall can leave its target unmet. Such a walk
            # runs to the target; the check after every victim still stops it
            # at the first sufficient one.
            to_target = (
                ct == ComponentType.FULL
                and request_cnt > 0
                and available_size_targets is not None
                and ct in available_size_targets
            )
            # A preceding component may have cascade-evicted this component or,
            # on a shared pool, released enough bytes to satisfy its allocation.
            if target_reached(ct) or (not to_target and tracker[ct] >= request_cnt):
                continue
            walk_quota = request_cnt
            if to_target:
                walk_quota = max(
                    request_cnt, tracker[ct] + self.tree_core.full_evictable_size()
                )
            self.tree_core.evict_device_start(ct, walk_quota)
            try:
                while True:
                    node_id, made_progress = self._evict_device_next_node(ct, tracker)
                    if node_id is None:
                        if not made_progress:
                            break
                    else:
                        if self._mamba_demote_watch:
                            self._watch_backed_leaf_state(node_id)
                        backup_kv = self._evict_device_leaf(node_id, tracker)
                    if node_id is not None and backup_kv is not None:
                        # Deferred demote: run the D->H backup, demote only on success.
                        if self._wb_evict_batch:
                            written = self._backup_full_leaf_batched(backup_kv)
                        else:
                            written = self._backup_full_leaf_for_write_back(backup_kv)
                        if written > 0:
                            self._demote(node_id, tracker)
                        elif self._drop_subtree_no_host(node_id, tracker):
                            logger.warning(
                                "write_back: KV subtree dropped without backup "
                                "due to host memory pressure, root node %d",
                                node_id,
                            )
                        else:
                            logger.warning(
                                "write_back: backup failed under host memory "
                                "pressure but subtree drop declined (node "
                                "locked); root node %d stays device-resident "
                                "until host space frees",
                                node_id,
                            )
                    if target_reached(ct):
                        break
            finally:
                try:
                    if self._wb_evict_batch:
                        self._wb_batch_flush()
                finally:
                    self.tree_core.evict_device_end(ct)

    def _tracks_write_through_unbacked_evictions(self) -> bool:
        return (
            self.host_memory_mode == "cache"
            and self.cache_controller is not None
            and self.cache_controller.write_policy == "write_through"
        )

    def _record_dropped_tokens(self, dropped_tokens: int, reason: str) -> None:
        """Record logical KV tokens irreversibly dropped without a host backup."""
        if self.metrics_collector is None or dropped_tokens <= 0:
            return
        self.metrics_collector.increment_dropped_tokens(
            num_tokens=dropped_tokens,
            reason=reason,
            pool=PoolName.KV.value,
        )

    def inc_lock_ref(
        self, node_id: NodeId, skip_lock_components: Sequence[ComponentType] = ()
    ) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult()
        return self.tree_core.inc_lock_ref(node_id, skip_lock_components)

    def dec_lock_ref(
        self,
        node_id: NodeId,
        params: DecLockRefParams,
        skip_swa: bool = False,
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult()
        return self.tree_core.dec_lock_ref(node_id, params, skip_swa)

    def lock(
        self, node_id: NodeId, skip_lock_components: Sequence[ComponentType] = ()
    ) -> Optional[TreeLock]:
        # A session's virtual node stands for a lock its slot already holds.
        if is_virtual_node(node_id):
            return None
        result = self.inc_lock_ref(node_id, skip_lock_components)
        return TreeLock(node_id, result.to_dec_params())

    def unlock(self, lock: Optional[TreeLock]) -> None:
        if lock is None:
            return
        self.dec_lock_ref(lock.node, lock.receipt, skip_swa=lock.swa_released)

    def release_swa(self, lock: Optional[TreeLock]) -> None:
        """Release the SWA part of ``lock`` early, once."""
        if self.disable or lock is None or lock.swa_released:
            return
        result = self.tree_core.dec_swa_lock_only(lock.node, lock.receipt)
        self._apply_eviction_result(result)
        lock.swa_released = True

    def release_swa_prefix_lock(self, req: Req) -> None:
        """The request's window has moved past its prefix: leave the prefix's
        SWA evictable. A session turn releases its slot's lock, once per session."""
        lock = (self.session.borrowed_slot(req) or req).lock
        if (
            lock is not None
            and lock.receipt.component_lock_uuids.get(ComponentType.SWA) is not None
        ):
            self.release_swa(lock)

    def inc_host_lock_ref(self, node_id: NodeId) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult()
        return self.tree_core.inc_host_lock_ref(node_id)

    def dec_host_lock_ref(
        self, node_id: NodeId, params: DecLockRefParams
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult()
        return self.tree_core.dec_host_lock_ref(node_id, params)

    def maybe_hand_to_session(self, req: Req) -> None:
        self.session.take(req)

    def free_kv_row(self, kv: ReqKvInfo, ranges: list[tuple[int, int]]) -> None:
        # A restore's host-to-device copy may still be writing this row's
        # pages; they must not return to the allocator before it finishes.
        self._finish_restore(kv)
        super().free_kv_row(kv, ranges)

    def claim_kv_row(self, req: Req) -> bool:
        if self.linker is not None and req.finished():
            self.linker.release_request(req.rid)
        # Retraction also enters here: retain its ticket until actual finish.
        if (
            self.cache_controller is not None
            and self.cache_controller.pp_prefetch_command_group is not None
            and req.finished()
        ):
            self.cache_controller.release_pp_prefetch(req.rid)
        return self.session.try_cache_finished_req(req)

    @rank_consensus(same_params=["req.rid", "checkpointed"])
    def on_release(self, req: Req, *, checkpointed: bool) -> None:
        if checkpointed:
            return
        for comp in self._components_tuple:
            comp.cleanup_after_caching_req(req, is_finished=True)
        self._release_request_mamba(req)

    def _release_request_mamba(self, req: Req) -> None:
        """Free the request's own Mamba state slot when no tree component takes
        it over (``request_mamba_only``)."""
        if self.request_mamba_only and req.kv.holds_mamba:
            self.req_to_token_pool.free_mamba_cache(req)

    @rank_consensus(same_params=["req.rid", "up_to"])
    def checkpoint(self, req: Req, *, up_to: int, **kwargs) -> None:
        if self.session.try_checkpoint(req, up_to=up_to, **kwargs):
            return
        self.checkpoint_into_tree(req, up_to=up_to, **kwargs)

    def checkpoint_into_tree(self, req: Req, *, up_to: int, **kwargs) -> None:
        """Insert ``[cache_protected_len, up_to)`` of the request's record and
        move its tree lock onto the node the insert ended on."""
        is_finished = req.finished()
        self._checkpoint_into_tree(req, up_to=up_to, **kwargs)
        if is_finished:
            self._release_request_mamba(req)

    def _checkpoint_into_tree(self, req: Req, *, up_to: int, **kwargs) -> None:
        # A finished request hands its component state (mamba) to the tree
        # instead of forking it, and the tree frees what the request still held.
        is_finished = req.finished()
        token_ids = req.full_untruncated_fill_ids[:up_to]

        if self.disable:
            if is_finished:
                for comp in self._components_tuple:
                    comp.cleanup_after_caching_req(req, is_finished=True)
            return

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : len(token_ids)
        ]

        # components prepare insert data + return effective cache_len
        insert_params = InsertParams(
            prev_prefix_len=req.kv.cache_protected_len,
            inserted_len=req.kv.cache_inserted_len,
            priority=req.priority or 0,
            session_id=req.session_id,
            rotation_base=req.kv_rotation_base,
            is_request_end=(
                is_finished
                or (
                    get_disagg().disaggregation_mode == "prefill"
                    and up_to >= len(req.origin_input_ids)
                )
            ),
        )
        effective_cache_len = len(token_ids)
        for comp in self._components_tuple:
            cl = comp.prepare_for_caching_req(
                req=req,
                insert_params=insert_params,
                token_ids_len=len(token_ids),
                is_finished=is_finished,
            )
            if cl is not None:
                effective_cache_len = min(effective_cache_len, cl)

        for comp in self._components_tuple:
            effective_cache_len = comp.floor_cache_len(effective_cache_len)

        radix_key = RadixKey(
            token_ids[:effective_cache_len],
            req.extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=req.cache_salt,
        )

        if (
            not is_finished
            and envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.get()
        ):
            # The frontier lands a page below page_floor(pre_len + 1), which has to
            # be where the insert stops, or the leaf it creates keeps less than a
            # sliding window of live SWA and the match after the insert rejects it.
            # The insert stops at page_floor(len(radix_key)), and a bigram key is
            # one shorter than the tokens it spans, so measure the key.
            for comp in self._components_tuple:
                comp.free_out_of_window_slots(req, len(radix_key) - 1, insert_params)

        if effective_cache_len <= 0:
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(
                    req, is_finished=is_finished, insert_params=insert_params
                )
            return

        kv_indices = kv_indices_orig[:effective_cache_len]

        radix_key = radix_key.page_aligned(self.page_size)
        page_aligned_len = len(radix_key)
        values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)

        insert_params.key = radix_key
        insert_params.value = values
        result = self.insert(insert_params)

        if result.rotation_tail_declined:
            # Rotation-base discontinuity with the matched chain (pipelined
            # batches raced this request's insert against another chain over
            # the same prefix). Adopting the canonical locs would leave this
            # request's row mixing two rotation runs, which the cyclic-owner
            # gather contract forbids -- keep the request entirely on its own
            # pages: no dedup free, no rebind, no protection change. The insert
            # declined before its walk, so nothing was freed underneath us. The
            # release_kv_cache releases everything past the protected
            # prefix.
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(
                    req, is_finished=is_finished, insert_params=insert_params
                )
            return
        req.kv.cache_inserted_len = max(req.kv.cache_inserted_len, page_aligned_len)

        # Split the leaf at the prompt boundary so eviction can drop the output
        # KV without the prompt. prev_prefix_len keeps the overlapping indices
        # from being freed as duplicates.
        prompt_key = RadixKey(
            req.origin_input_ids,
            req.extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=req.cache_salt,
        ).page_aligned(self.page_size)
        if (
            len(self._components_tuple) == 1
            and self._components_tuple[0].component_type == BASE_COMPONENT_TYPE
            and 0 < len(prompt_key) < len(radix_key)
        ):
            self.insert(
                replace(
                    insert_params,
                    key=prompt_key,
                    value=values[: len(prompt_key)],
                    prev_prefix_len=len(prompt_key),
                    inserted_len=len(prompt_key),
                    priority=min(
                        insert_params.priority + 1, torch.iinfo(torch.int64).max
                    ),
                )
            )

        # With PD decode radix caching, SWA matching can stop short of the inserted path.
        # Repoint from that path to protect all tree-owned KV.
        new_last_node = result.last_device_node
        new_indices = self.path_device_indices(new_last_node)
        new_prefix_len = result.prefix_len
        assert req.kv.cache_protected_len <= len(new_indices) + self.page_size - 1, (
            f"{req.kv.cache_protected_len=}, {len(new_indices)=}, {page_aligned_len=}"
        )
        assert (
            new_prefix_len <= len(new_indices) and len(new_indices) == page_aligned_len
        ), f"{new_prefix_len=}, {len(new_indices)=}, {page_aligned_len=}"
        self.req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(req.kv.cache_protected_len, len(new_indices))),
            new_indices[req.kv.cache_protected_len :],
        )

        self.unlock(req.lock)
        # Opt-in: leave the matched-prefix mamba evictable during decode (it is
        # already COW'd to the request's own slot, never read from this node again).
        # Safe only because any future COW source is the COWing request's own
        # admission-locked last_node (recorded only if still present, locked before
        # the next alloc) -- not this evictable node. A scheduler that matched a
        # whole batch before locking would break that. Off = original full lock.
        req.lock = self.lock(
            new_last_node,
            skip_lock_components=(
                (ComponentType.MAMBA,)
                if envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.get()
                else ()
            ),
        )

        # Update req fields
        req.kv.cache_protected_len = len(new_indices)
        req.last_node = new_last_node

        # cleanup
        for comp in self._components_tuple:
            comp.cleanup_after_caching_req(
                req,
                is_finished=is_finished,
                insert_result=result,
                insert_params=insert_params,
            )

        if self.enable_session_radix_cache and is_finished:
            from sglang.srt.managers.schedule_batch import FINISH_ABORT

            if not isinstance(req.finished_reason, FINISH_ABORT):
                self.session_refs.register_session_ref(
                    req, leaf=result.last_device_node
                )

    @rank_consensus(same_params=["req.rid"])
    def advance_unpublished_req(self, req: Req) -> None:
        assert not self.supports_mamba()
        token_ids = req.get_fill_ids()
        insert_params = InsertParams(
            prev_prefix_len=req.kv.cache_protected_len,
            priority=req.priority or 0,
            rotation_base=req.kv_rotation_base,
        )
        effective_cache_len = len(token_ids)
        for comp in self._components_tuple:
            cache_len = comp.prepare_for_caching_req(
                req=req,
                insert_params=insert_params,
                token_ids_len=len(token_ids),
                is_finished=False,
            )
            if cache_len is not None:
                effective_cache_len = min(effective_cache_len, cache_len)

        radix_key = RadixKey(
            token_ids[:effective_cache_len],
            req.extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=req.cache_salt,
        )
        if envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.get():
            for comp in self._components_tuple:
                comp.free_out_of_window_slots(req, len(radix_key) - 1, insert_params)

        for comp in self._components_tuple:
            comp.cleanup_after_caching_req(
                req, is_finished=False, insert_params=insert_params
            )

    # ---- Internal Helpers ----

    def _apply_cache_actions(
        self, actions: list[CacheAction | ComponentAction]
    ) -> None:
        # Apply and consume one at a time: a spent list cannot be double-applied.
        actions.reverse()
        try:
            while actions:
                self._apply_cache_action(actions.pop())
        finally:
            actions.reverse()

    @contextlib.contextmanager
    def defer_device_frees(self, sink: list):
        """Collect device KV / slot frees into ``sink`` instead of freeing; the
        caller applies them with ``apply_deferred_frees`` once nothing reads them."""
        prev = getattr(self, "_deferred_free_sink", None)
        self._deferred_free_sink = sink
        try:
            yield sink
        finally:
            self._deferred_free_sink = prev

    def apply_deferred_frees(self, sink: list) -> None:
        self._apply_cache_actions(sink)

    def _apply_cache_action(self, action: CacheAction | ComponentAction) -> None:
        sink = getattr(self, "_deferred_free_sink", None)
        if sink is not None and isinstance(
            action, (FreeDeviceKV, FreeDeviceKVFullOnly, FreeComponentDeviceSlot)
        ):
            sink.append(action)
            return
        # Component actions route to their component class; the rest are
        # cache-owned and handled here by type.
        if isinstance(action, ComponentAction):
            if (
                isinstance(action, UpdatePendingBackupOnNodeSplit)
                and action.ack_id is not None
            ):
                node_ids = [action.new_node_id, action.new_child_node_id]
                self._replace_pending_write_through_node(
                    action.ack_id, action.old_node_id, node_ids
                )
                if self.linker is not None:
                    self.linker.replace_pending_offload_node(
                        action.ack_id, action.old_node_id, node_ids
                    )
            self.components[action.component_type].apply_component_action(action)
        elif isinstance(action, FreeDeviceKV):
            # tree values are page-aligned copies of a kv row: page-exact segments
            for indices in action.indices:
                self.token_to_kv_pool_allocator.free_segment(indices, start_pos=0)
        elif isinstance(action, FreeDeviceKVFullOnly):
            for indices in action.indices:
                self.token_to_kv_pool_allocator.free_full_segment(indices, start_pos=0)
        elif isinstance(action, BackupKV):
            if self.linker is not None:
                self.linker.offload_nodes(action.node_ids)
            else:
                self._execute_and_commit_kv_backup(action)
        else:
            raise AssertionError(f"unhandled CacheAction: {type(action).__name__}")

    def _drain_device_frees(
        self, device_frees: dict[ComponentType, list[torch.Tensor]]
    ) -> None:
        # Free per component device slots, consuming each entry as it frees.
        for ct in list(device_frees):
            self._apply_cache_action(
                FreeComponentDeviceSlot(device_frees.pop(ct), component_type=ct)
            )

    def _drain_host_frees(
        self, host_frees: dict[ComponentType, list[torch.Tensor]]
    ) -> None:
        # Free per component host-pool slots, consuming each entry as it frees.
        for ct in list(host_frees):
            self.components[ct].free_host_values(host_frees.pop(ct))

    def evict_host(
        self, num_tokens: int, component_type: ComponentType = BASE_COMPONENT_TYPE
    ) -> int:
        """Evict host resources for a specific component to free host pool space."""
        with pp_bubble_diag.sub("evict_host"):
            return self._evict_host_timed(num_tokens, component_type)

    def _evict_host_timed(self, num_tokens: int, component_type: ComponentType) -> int:
        if not pmicro_hoststats.ON:
            return self._evict_host(num_tokens, component_type)
        pmicro_hoststats.note_host_evict_call()
        t0 = time.perf_counter()
        evicted = self._evict_host(num_tokens, component_type)
        pmicro_hoststats.OPS["host_evict"].add(time.perf_counter() - t0)
        return evicted

    def _evict_host(self, num_tokens: int, component_type: ComponentType) -> int:
        if self.host_memory_mode == "buffer_only":
            # The tree never holds host values in buffer mode, and staging
            # is operation-owned (freed at each ack): nothing is evictable.
            return 0
        if self._l3_trace is not None and component_type == ComponentType.MAMBA:
            self._l3_trace_mamba_host_eviction(num_tokens)
        evicted = self._drive_host_eviction(num_tokens, component_type)
        if (
            self._storage_write_back is not None
            and component_type == BASE_COMPONENT_TYPE
        ):
            t0 = time.perf_counter() if pmicro_hoststats.ON else 0.0
            evicted = self._storage_write_back.evict_host(num_tokens, evicted)
            if pmicro_hoststats.ON:
                pmicro_hoststats.OPS["host_evict_l3"].add(time.perf_counter() - t0)
        if self._l3_tier_final is not None and component_type == BASE_COMPONENT_TYPE:
            self._l3_tier_final.host_evicted_tokens += evicted
        return evicted

    def _drive_host_eviction(
        self, num_tokens: int, component_type: ComponentType
    ) -> int:
        if self._l3_write_state_on_tombstone and component_type == ComponentType.MAMBA:
            if not pmicro_hoststats.ON:
                return self._drive_mamba_host_eviction_with_storage(num_tokens)
            t0 = time.perf_counter()
            evicted = self._drive_mamba_host_eviction_with_storage(num_tokens)
            pmicro_hoststats.OPS["host_evict_l3"].add(time.perf_counter() - t0)
            return evicted
        result = self.tree_core.drive_host_eviction(component_type, num_tokens)
        self._apply_eviction_result(result)
        evicted = result.tracker.get(component_type, 0)
        if component_type == BASE_COMPONENT_TYPE and self._l3_write_on_host_evict:
            self._l3_tier_stats["wb_evict_uncovered_tokens"] += max(
                0, evicted - self._write_behind_clean_tokens
            )
        if component_type == BASE_COMPONENT_TYPE and self._write_behind_clean_tokens:
            self._write_behind_clean_tokens = max(
                0, self._write_behind_clean_tokens - evicted
            )
        return evicted

    def _drive_mamba_host_eviction_with_storage(self, num_states: int) -> int:
        """Write an actual state-only victim before releasing its host slot.

        The Rust walk pauses before eligible non-leaf tombstones. The scheduler
        pins both source pools while the independent backup worker copies them;
        it never waits on a scheduler-drained acknowledgment or enters a rank
        collective from this allocator-dependent path.
        """
        if self._l3_tombstone_pending is not None:
            return self._drive_mamba_host_eviction_async(num_states)
        evicted = 0
        while evicted < num_states:
            result = self.tree_core.begin_mamba_host_eviction(
                num_states - evicted, min_depth=8192
            )
            self._apply_eviction_result(result)
            evicted += result.tracker.get(ComponentType.MAMBA, 0)
            node_id = result.mamba_storage_backup_node_id
            if node_id is None:
                break
            operation_id = self._write_mamba_tombstone(node_id)
            t0 = time.perf_counter() if pmicro_hoststats.ON else 0.0
            self._wait_mamba_storage_backup(operation_id)
            if pmicro_hoststats.ON:
                pmicro_hoststats.note_host_wait(time.perf_counter() - t0, True)
            evicted += self._finish_mamba_tombstone(node_id)
        return evicted

    def _write_mamba_tombstone(self, node_id: NodeId) -> int:
        # Advisory beliefs are not a transfer acknowledgment: force the
        # backend to confirm the checkpoint, including after TTL eviction.
        operation_id = self._write_backup_storage(
            node_id, force=True, ledger_reason="state_tombstone"
        )
        if operation_id is None:
            raise RuntimeError("Mamba tombstone victim has no writable host state")
        return operation_id

    def _finish_mamba_tombstone(self, node_id: NodeId, *, revived_ok: bool = False) -> int:
        result = self.tree_core.finish_mamba_host_eviction(node_id)
        self._apply_eviction_result(result)
        freed = result.tracker.get(ComponentType.MAMBA, 0)
        if freed == 0 and not revived_ok:
            # No other scheduler mutation can run while we wait. A failed
            # resume must not spin forever on the same allocator victim.
            raise RuntimeError("Mamba tombstone victim changed while writing storage")
        return freed

    def _issue_mamba_tombstones(self, num_states: int) -> int:
        """SGLANG_HICACHE_L3_TOMBSTONE_ASYNC: free plain victims now and start the
        tombstone write of at most num_states pinned ones (the walk passes pinned
        nodes); those slots free in _drain_backup. Returns the states freed now."""
        evicted = issued = 0
        while evicted + issued < num_states:
            result = self.tree_core.begin_mamba_host_eviction(
                num_states - evicted - issued, min_depth=8192
            )
            self._apply_eviction_result(result)
            evicted += result.tracker.get(ComponentType.MAMBA, 0)
            node_id = result.mamba_storage_backup_node_id
            if node_id is None:
                break
            self._l3_tombstone_pending[self._write_mamba_tombstone(node_id)] = node_id
            issued += 1
        return evicted

    def _drive_mamba_host_eviction_async(self, num_states: int) -> int:
        evicted = self._issue_mamba_tombstones(num_states)
        # The caller needs the slots now: finish the oldest writes synchronously,
        # as the synchronous mode would have (the same step on every rank).
        mamba_host_pool = self.host_pool_group.get_pool(PoolName.MAMBA)
        while self._l3_tombstone_pending and mamba_host_pool.available_size() < num_states:
            operation_id, node_id = next(iter(self._l3_tombstone_pending.items()))
            del self._l3_tombstone_pending[operation_id]
            self._wait_mamba_storage_backup(operation_id)
            # Scheduling ran since the write: a hit may have revived the state.
            evicted += self._finish_mamba_tombstone(node_id, revived_ok=True)
        return evicted

    def _refill_mamba_tombstone_reserve(self) -> None:
        # After the synced drain, at the same point of every rank's step.
        mamba_host_pool = self.host_pool_group.get_pool(PoolName.MAMBA)
        deficit = (
            self._l3_tombstone_reserve
            - mamba_host_pool.available_size()
            - len(self._l3_tombstone_pending)
        )
        if deficit > 0:
            self._issue_mamba_tombstones(deficit)

    def _wait_mamba_storage_backup(self, operation_id: int) -> None:
        with pp_bubble_diag.wait("mamba_l3_backup"):
            return self._wait_mamba_storage_backup_impl(operation_id)

    def _wait_mamba_storage_backup_impl(self, operation_id: int) -> None:
        cc = self.cache_controller
        ack_queue = cc.ack_backup_queue
        start = time.monotonic()
        warn_at = start + 30.0
        while True:
            # Inspect only this operation; earlier ordinary ACKs belong to the
            # synchronized drain. Failed receipts stay quarantined in the queue.
            with ack_queue.mutex:
                operation = next(
                    (op for op in reversed(ack_queue.queue) if op.id == operation_id),
                    None,
                )
            if operation is not None:
                break
            if cc.storage_stop_event.is_set() or not cc.backup_thread.is_alive():
                # Do not unlock: without an ACK, source DMA completion is unknown.
                self._l3_tombstone_quarantined_ops |= {operation_id}
                raise RuntimeError("Storage worker stopped before Mamba tombstone ACK")
            if time.monotonic() >= warn_at:
                logger.warning(
                    "Mamba tombstone waiting %.0f s for storage operation %d",
                    time.monotonic() - start,
                    operation_id,
                )
                warn_at += 30.0
            time.sleep(0.0005)

        self._check_mamba_tombstone_ack(operation, operation_id)
        # This synchronous local write was issued after any enclosing drain's
        # count snapshot. Remove only its own ACK, leaving that ordinary FIFO
        # and count intact even if ranks wrote different numbers of tombstones.
        # Do not seed rank-replicated write-behind beliefs from this local ACK.
        with ack_queue.mutex:
            ack_queue.queue.remove(operation)
            ack_queue.not_full.notify()
        if self._l3_ack_lag_steps:
            self._l3_ack_removed += 1
        node_id, lock_params = self.ongoing_backup.pop(operation_id)
        self.dec_host_lock_ref(node_id, lock_params)

    def _check_mamba_tombstone_ack(self, operation, operation_id: int) -> None:
        """Keep a tombstone source pinned unless its KV and state reached storage."""
        cc = self.cache_controller
        state_written = False
        succeeded = operation.completed_tokens >= len(operation.hash_value) * self.page_size
        for transfer in operation.pool_transfers or ():
            if not cc.should_backup(transfer):
                continue
            expected = len(transfer.keys or ())
            written = operation.pool_storage_result.extra_pool_hit_pages.get(
                transfer.name, 0
            )
            succeeded = succeeded and expected > 0 and written >= expected
            state_written |= transfer.name == PoolName.MAMBA and expected > 0 and written >= expected
        node_id, _ = self.ongoing_backup[operation_id]
        if not succeeded or not state_written:
            operation.l3_tombstone_quarantined = True
            self._l3_tombstone_quarantined_ops |= {operation_id}
        if self._l3_lifetime_ledger is not None:
            self._l3_lifetime_ledger.emit(
                "state_tombstone_write_ack", op_id=operation_id, node_id=node_id,
                succeeded=bool(succeeded and state_written),
                completed_tokens=operation.completed_tokens,
                pool_hit_pages=dict(operation.pool_storage_result.extra_pool_hit_pages),
            )
        if not succeeded or not state_written:
            # NIXL can ACK a failure after a transfer timeout while DMA is still
            # reading these slots. Retain ownership and stop the scheduler; a
            # later drain must not treat this receipt as permission to unlock.
            raise RuntimeError("Storage backup failed before Mamba host-state eviction")

    # ---- Decode retraction ----

    def supports_retraction_backup(self) -> bool:
        if self.cache_controller is None or self.host_pool_group is None:
            return False
        if self.supports_mamba():
            return False

        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kv_cache, SWAKVPool):
            return (
                self.supports_swa()
                and {
                    PoolName.KV,
                    PoolName.SWA,
                }
                <= self.host_pool_group.entry_map.keys()
            )
        return isinstance(kv_cache, MHATokenToKVPool) and (
            PoolName.KV in self.host_pool_group.entry_map
        )

    def validate_retraction_host_capacity(self) -> None:
        if not self.supports_retraction_backup():
            raise ValueError(
                "--disaggregation-decode-retraction-backup=host_pool requires "
                "an MHA or hybrid-SWA HiCache host stack."
            )

        for spec in self.sidecar_pool_specs:
            source_size = self.host_pool_group.entry_map[
                spec.indices_from_pool
            ].host_pool.logical_size
            sidecar_size = self.host_pool_group.entry_map[
                spec.pool_name
            ].host_pool.logical_size
            if sidecar_size < source_size:
                raise ValueError(
                    "Retraction sidecar host pool is smaller than its index source: "
                    f"pool={spec.pool_name}, host_slots={sidecar_size}, "
                    f"source={spec.indices_from_pool}, source_slots={source_size}."
                )

    @staticmethod
    def _pad_retraction_indices(indices: torch.Tensor, page_size: int) -> torch.Tensor:
        aligned_len = ceil_align(len(indices), page_size)
        if aligned_len == len(indices):
            return indices
        tail = indices[-1] + torch.arange(
            1,
            aligned_len - len(indices) + 1,
            dtype=torch.int64,
            device=indices.device,
        )
        return torch.cat([indices, tail])

    def _retraction_device_transfers(
        self, req: Req
    ) -> tuple[torch.Tensor, list[PoolTransfer]]:
        num_tokens = req.seqlen - 1
        full_indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :num_tokens
        ].to(torch.int64)
        full_indices = self._pad_retraction_indices(full_indices, self.page_size)

        component_transfers: dict[ComponentType, list[PoolTransfer]] = {}
        if self.supports_swa():
            assert self.sliding_window_size is not None
            window_start = max(0, num_tokens - self.sliding_window_size)
            window_start = window_start // self.page_size * self.page_size
            window_indices = self.req_to_token_pool.req_to_token[
                req.kv.req_pool_idx, window_start:num_tokens
            ].to(torch.int64)
            swa_indices = (
                self.token_to_kv_pool_allocator.translate_swa_indices_for_transfer(
                    window_indices
                )
            )
            # Checked on the device: a host-side check would wait for the GPU.
            torch._assert_async(
                (swa_indices > 0).all(),
                f"unmapped SWA window positions for request {req.rid}",
            )
            component_transfers[ComponentType.SWA] = [
                PoolTransfer(
                    name=PoolName.SWA,
                    device_indices=self._pad_retraction_indices(
                        swa_indices, self.page_size
                    ),
                )
            ]

        kv_transfer = PoolTransfer(name=PoolName.KV, device_indices=full_indices)
        extra_transfers = [
            transfer
            for transfers in component_transfers.values()
            for transfer in transfers
        ]
        sidecar_transfers = self._build_sidecar_transfers(
            CacheTransferPhase.BACKUP_HOST,
            kv_transfer,
            component_transfers,
        )
        extra_transfers.extend(sidecar_transfers)
        return full_indices, extra_transfers

    def _reclaim_retraction_host(self, num_tokens: int) -> int:
        if self.disable:
            return 0
        return self.evict_host(num_tokens)

    def _discard_restore_backups(self) -> None:
        """Wait for every pending restore copy and return its host slots."""
        for finish_event, backup in self._pending_restore_backups:
            finish_event.synchronize()
            self.discard_kv_cache_backup(backup)
        self._pending_restore_backups = []

    def _finish_restore(self, kv: ReqKvInfo) -> None:
        """Wait for the copy restoring ``kv``'s row, if any, and return its host
        slots. Host capacity changes only here and when the cache is reset, at
        points every rank reaches alike, never when one rank's copy happens to
        finish."""
        event = kv.restore_done_event
        if event is None:
            return
        event.synchronize()
        kv.restore_done_event = None
        for i, (finish_event, backup) in enumerate(self._pending_restore_backups):
            if finish_event is event:
                del self._pending_restore_backups[i]
                self.discard_kv_cache_backup(backup)
                return

    def backup_kv_cache(self, req: Req) -> Optional[RetractionBackup]:
        """Back up device KV to the host pool; None when it cannot fit after reclaim."""
        assert req.seqlen > 1

        device_indices, extra_transfers = self._retraction_device_transfers(req)
        allocation = self.cache_controller.allocate_host_transfers(
            device_indices,
            extra_transfers or None,
            reclaim=self._reclaim_retraction_host,
        )
        if allocation is None:
            return None
        host_indices, resolved = allocation

        backup = RetractionBackup(
            host_indices=host_indices,
            pool_transfers=[replace(x, device_indices=None) for x in resolved or []]
            or None,
        )
        operation = CacheOperation(
            host_indices,
            device_indices,
            node_id=-1,
            pool_transfers=resolved,
        )
        try:
            write_host, write_device, write_pools = (
                self.cache_controller._move_write_operation(operation)
            )
            completion = self.cache_controller.l2_transfer_engine.submit_device_to_host(
                self.cache_controller._l2_transfers(
                    write_host, write_device, write_pools
                )
            )
            completion.finish_event.synchronize()
        except Exception:
            self.discard_kv_cache_backup(backup)
            raise
        return backup

    def restore_kv_cache(self, req: Req, backup: RetractionBackup) -> None:
        self.start_restore_kv_cache(req, backup)
        # Order later work on this stream after the copy instead of blocking
        # the scheduler on it: the forward is ordered behind this stream.
        device_module.current_stream().wait_event(req.kv.restore_done_event)

    def start_restore_kv_cache(self, req: Req, backup: RetractionBackup) -> None:
        """Start copying ``backup`` into ``req``'s row without ordering later
        work after the copy; the row may be read once ``req.kv.restore_done_event``
        has completed. Page moves and frees wait for the copy as for a HiCache
        load; the host slots stay reserved until finish_restore_kv_cache, or
        freeing the row (free_kv_row), returns them after the copy."""
        device_indices, current_transfers = self._retraction_device_transfers(req)
        assert len(backup.host_indices) == len(device_indices), (
            f"Host backup has {len(backup.host_indices)} slots, but restore has "
            f"{len(device_indices)}"
        )

        current_by_name = {transfer.name: transfer for transfer in current_transfers}
        saved_by_name = {
            transfer.name: transfer for transfer in backup.pool_transfers or []
        }
        assert current_by_name.keys() == saved_by_name.keys(), (
            f"Host backup pools {set(saved_by_name)} do not match restore pools "
            f"{set(current_by_name)}"
        )
        restored_transfers = [
            replace(
                saved,
                device_indices=current_by_name[name].device_indices,
            )
            for name, saved in saved_by_name.items()
        ]
        resolved = self.cache_controller._resolve_device_transfers(
            restored_transfers or None,
            kv_device_indices=device_indices,
            kv_host_indices=backup.host_indices,
        )
        assert resolved is not None or not restored_transfers

        operation = CacheOperation(
            backup.host_indices,
            device_indices,
            node_id=-1,
            pool_transfers=resolved,
        )
        load_host, load_device, load_pools = self.cache_controller._move_op_indices(
            operation
        )
        completion = self.cache_controller.l2_transfer_engine.submit_host_to_device(
            self.cache_controller._l2_load_transfers(
                load_host, load_device, load_pools
            ),
            transfer_layer_id_max=self.cache_controller.transfer_layer_id_max,
        )
        self.token_to_kv_pool_allocator.set_hicache_transfer_done_event(
            (id(self), "restore"), completion.finish_event
        )
        req.kv.restore_done_event = completion.finish_event
        self._pending_restore_backups.append((completion.finish_event, backup))

    def finish_restore_kv_cache(self, req: Req) -> None:
        """Return the host slots of ``req``'s restore once its copy has finished."""
        self._finish_restore(req.kv)

    def discard_kv_cache_backup(self, backup: RetractionBackup) -> None:
        self.host_pool_group.free(backup.host_indices)
        self.host_pool_group.release_transfers(backup.pool_transfers)

    # ---- HiCache: Backup / LoadBack ----

    def backup_node_for_write_back(self, node_id: NodeId) -> bool:
        """Synchronously back up one node (Full KV plus any unbacked component
        state) and drain the ack -- the deferred-demote shape the eviction
        loop runs for leaves, reusable by component evictors ahead of an
        internal-state tombstone. Returns True once the backup is committed.
        """
        # An auxiliary backup may already cover this node under another ack.
        # Finish it before building a new transfer and claiming ack ownership.
        if self.ongoing_write_through:
            self.writing_check(write_back=True)
        written = self._execute_and_commit_kv_backup(
            BackupKV(node_ids=[node_id]), write_back=True
        )
        # A node whose Full KV is already on host copies component state only
        # (written == 0); that copy must land before the device source is freed.
        if written == 0 and node_id not in self.ongoing_write_through:
            return False
        self.writing_check(write_back=True)
        return True

    def buffer_backup_pending(self, node_id: NodeId) -> bool:
        """Whether a buffer-mode write of ``node_id`` is queued, staged or
        awaiting its storage ack (see ``BufferModePipeline.backup_pending``)."""
        return self.buffer_pipeline is not None and self.buffer_pipeline.backup_pending(
            node_id
        )

    def request_buffer_backup(self, node_id: NodeId) -> None:
        """Queue a buffer-mode backup of ``node_id`` (and its unstored
        ancestors) outside the insert trigger; no-op without buffer storage."""
        if self.buffer_pipeline is None or not self.enable_storage:
            return
        self._execute_and_commit_kv_backup(
            self.tree_core.build_backup_kv_action(node_id)
        )

    def _execute_and_commit_kv_backup(
        self, action: BackupKV, write_back: bool = False
    ) -> int:
        """Run a backup action top-down, stopping at the first failed backup."""
        if self.buffer_pipeline is not None:
            # Buffer mode bypasses the host-backup contiguity below: nothing
            # is ever host-backuped here. Contiguity comes from end-to-end
            # FIFO ordering instead (BackupKV chains are parent-before-child
            # and every pipeline stage drains in order).
            for node_id in action.node_ids:
                self.buffer_pipeline.enqueue_backup_intent(node_id)
            return 0
        written = 0
        for node_id in action.node_ids:
            device_value, comp_xfers = self.tree_core.build_backup_spec(node_id)
            # Overlapping chain actions may revisit nodes with Full KV already
            # backed up. Skip only when no transfer remains.
            if device_value.numel() == 0 and not comp_xfers:
                continue
            sidecar_xfers = self._build_backup_sidecar(device_value, comp_xfers)
            if pmicro_hoststats.ON and write_back:
                host_free = self.cache_controller.mem_pool_host.available_size()
                host_evicts = pmicro_hoststats.HOST_EVICT_CALLS[0]
            host_indices = self._execute_kv_backup(
                node_id, device_value, comp_xfers, sidecar_xfers
            )
            if pmicro_hoststats.ON and write_back:
                pmicro_hoststats.note_evict_backup(
                    host_free,
                    self._pmicro_l3_pending(),
                    pmicro_hoststats.HOST_EVICT_CALLS[0] != host_evicts,
                )
            if host_indices is None:
                return 0
            if pmicro_hoststats.ON:
                pmicro_hoststats.note_backup(node_id, len(host_indices))
            self.tree_core.commit_backup(node_id, host_indices, comp_xfers)
            lock_params = None
            if not write_back:
                lock_params = self.inc_lock_ref(node_id).to_dec_params()
            publish_node_ids = self._backup_publish_node_ids(node_id, comp_xfers)
            self._track_write_through_node(
                node_id, lock_params, publish_node_ids=publish_node_ids
            )
            written = len(host_indices)
        return written

    def _pmicro_l3_pending(self) -> int:
        """Host KV tokens host-locked by L3 writes not yet drained."""
        tokens = sum(self._write_behind_inflight.values())
        if self._storage_write_back is not None:
            tokens += self._storage_write_back._inflight_tokens
        return tokens

    @staticmethod
    def _backup_publish_node_ids(
        node_id: NodeId, comp_xfers: dict[ComponentType, list[PoolTransfer]]
    ) -> list[NodeId]:
        """The acked node plus every node a component backup transfer covers."""
        publish_node_ids: list[NodeId] = []
        for transfers in comp_xfers.values():
            for transfer in transfers:
                publish_node_ids.extend(transfer.nodes_to_load or ())
        if node_id not in publish_node_ids:
            publish_node_ids.append(node_id)
        return list(dict.fromkeys(publish_node_ids))

    def _build_backup_sidecar(self, device_value, comp_xfers):
        """Gather sidecar transfer spec."""
        kv_xfer = PoolTransfer(name=PoolName.KV, device_indices=device_value)
        return self._build_sidecar_transfers(
            CacheTransferPhase.BACKUP_HOST, kv_xfer, comp_xfers
        )

    def _execute_kv_backup(self, node_id, device_value, comp_xfers, sidecar_xfers):
        with pp_bubble_diag.sub("kv_backup"):
            return self._execute_kv_backup_impl(
                node_id, device_value, comp_xfers, sidecar_xfers
            )

    def _execute_kv_backup_impl(self, node_id, device_value, comp_xfers, sidecar_xfers):
        """Execute Backup action."""
        kv_tokens = len(device_value)
        anchor_entry = self.cache_controller.mem_pool_host.anchor_entry
        if (
            anchor_entry.host_pool.shared_allocation_domain is None
            or anchor_entry.host_evict_fn is None
        ):
            host_avail = self.cache_controller.mem_pool_host.available_size()
            if host_avail < kv_tokens:
                needed = kv_tokens - host_avail
                if self.evict_host(needed) < needed:
                    return None
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        # Preserve the ordinary controller API when no shard owner is present.
        start_owner = self.tree_core.rotation_phase_of(node_id)
        shard_kwargs = {} if start_owner is None else {"start_owner": start_owner}
        # Defer submission so the next flush can merge pending node backups.
        return self.cache_controller.write(
            device_value,
            node_id=node_id,
            extra_pools=aux_xfers or None,
            flush=False,
            **shard_kwargs,
        )

    def _track_write_through_node(
        self,
        node_id: NodeId,
        lock_params: Optional[DecLockRefParams],
        publish_node_ids: list[NodeId],
    ) -> None:
        publish_node_ids = self.tree_core.mark_write_through_pending(
            publish_node_ids, ack_id=node_id
        )
        self.ongoing_write_through[node_id] = _OngoingWriteThrough(
            node_id, lock_params, publish_node_ids
        )

    def _replace_pending_write_through_node(
        self, ack_id: int, old_node_id: NodeId, new_node_ids: list[NodeId]
    ) -> None:
        ongoing = self.ongoing_write_through
        pending = ongoing.get(ack_id)
        if pending is None:
            ongoing = self.ongoing_write_ahead
            pending = ongoing.get(ack_id)
        if pending is None:
            return

        lock_node_id, lock_params, publish_node_ids = pending
        updated_node_ids = []
        replaced = False
        for node_id in publish_node_ids:
            if node_id == old_node_id:
                updated_node_ids.extend(new_node_ids)
                replaced = True
            else:
                updated_node_ids.append(node_id)

        if not replaced:
            return

        ongoing[ack_id] = _OngoingWriteThrough(
            lock_node_id,
            lock_params,
            updated_node_ids,
        )

    def _finish_write_through_ack(
        self, ack_id: int, ongoing: Optional[dict[int, _OngoingWriteThrough]] = None
    ) -> None:
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.finish_backup_ack(ack_id)
            return

        if ongoing is None:
            ongoing = self.ongoing_write_through
        lock_node_id, lock_params, publish_node_ids = ongoing.pop(ack_id)
        self.tree_core.finish_write_through(publish_node_ids, ack_id)
        if lock_params is not None:
            self.dec_lock_ref(lock_node_id, lock_params)
        if self.enable_storage and self._storage_write_back is not None:
            # Storage is written when these host copies are evicted.
            self._storage_write_back.pin(publish_node_ids)
        elif self.enable_storage and not self._l3_write_on_host_evict:
            # Back up each fragment: after a split, lock_node only holds the
            # suffix; the prefix fragment must be persisted as well.
            for node_id in publish_node_ids:
                self.write_backup_storage(node_id)

    def load_back(
        self,
        node_id: NodeId,
        mem_quota: Optional[int] = None,
        req=None,
    ) -> bool:
        """Load evicted KV data from host back to device (H→D)."""
        if self.cache_controller is None:
            return False

        host_anchor_params = self.inc_host_lock_ref(node_id).to_dec_params()

        # Lock the path before building transfers (the aux build can evict).
        result = self.inc_lock_ref(node_id)
        ancestor_lock_params = result.to_dec_params()

        # Let each component pre-allocate per-request state for the load-back;
        # the finally below lets components recover it unless the load succeeds.
        preps: dict[ComponentType, PrepareLoadBackResult] = {
            comp.component_type: comp.prepare_load_back(node_id, req=req)
            for comp in self._components_tuple
        }
        success = False
        try:
            success = self._load_back_transfers(
                node_id=node_id,
                mem_quota=mem_quota,
                req=req,
                result=result,
                ancestor_lock_params=ancestor_lock_params,
                host_anchor_params=host_anchor_params,
            )
            return success
        finally:
            for comp in self._components_tuple:
                comp.finalize_load_back(req, preps[comp.component_type], success)

    def _load_back_transfers(
        self,
        *,
        node_id: NodeId,
        mem_quota: Optional[int],
        req,
        result: IncLockRefResult,
        ancestor_lock_params: DecLockRefParams,
        host_anchor_params: DecLockRefParams,
    ) -> bool:
        # Build the KV + per-component aux transfers.
        kv_xfer, comp_xfers = self.tree_core.build_load_back_spec(node_id, req=req)
        kv_tokens = len(kv_xfer.host_indices)
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.LOAD_BACK, kv_xfer, comp_xfers
        )

        # Skip if there is nothing to load, or if the Full-KV transfer is too
        # small / exceeds memory quota. Aux transfers should still run even
        # when the Full-KV load is skipped by thresholding. max(1, ...): an
        # entirely empty spec (e.g. foreign-pin rejection) must never report
        # success, even at load_back_threshold <= 0.
        if (kv_tokens < max(1, self.load_back_threshold) and not comp_xfers) or (
            mem_quota is not None and kv_tokens + result.delta > mem_quota
        ):
            # An empty spec: the node is held by another load-back (or pinned).
            empty = kv_tokens == 0 and not comp_xfers
            self._count_mamba_load_back_refusal(req, "held" if empty else "quota")
            self.dec_lock_ref(node_id, ancestor_lock_params)
            self.dec_host_lock_ref(node_id, host_anchor_params)
            return False

        c128_allocator = getattr(
            self.token_to_kv_pool_allocator, "c128_attn_allocator", None
        )
        if c128_allocator is not None:
            c128_num_pages = _c128_transfer_num_pages(
                comp_xfers.get(ComponentType.C128, ()),
                c128_allocator.page_size,
            )
            if not self.token_to_kv_pool_allocator.ensure_c128_capacity(
                self, c128_num_pages
            ):
                self._count_mamba_load_back_refusal(req, "capacity")
                self.dec_lock_ref(node_id, ancestor_lock_params)
                self.dec_host_lock_ref(node_id, host_anchor_params)
                return False

        avail = self._component_available_size(ComponentType.FULL)
        if avail < kv_tokens:
            needed = kv_tokens - avail
            self.evict_for_alloc(EvictParams(num_tokens=needed))
            if self._component_available_size(ComponentType.FULL) < kv_tokens:
                self._count_mamba_load_back_refusal(req, "capacity")
                self.dec_lock_ref(node_id, ancestor_lock_params)
                self.dec_host_lock_ref(node_id, host_anchor_params)
                return False

        # Load H→D
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        device_indices = self.cache_controller.load(
            host_indices=kv_xfer.host_indices,
            node_id=node_id,
            extra_pools=aux_xfers or None,
        )

        self.dec_lock_ref(node_id, ancestor_lock_params)
        if device_indices is None:
            self._count_mamba_load_back_refusal(req, "alloc")
            self.dec_host_lock_ref(node_id, host_anchor_params)
            return False
        if self._demote_copied_nodes and comp_xfers.get(ComponentType.MAMBA):
            if node_id in self._demote_copied_nodes:
                self._demote_copied_nodes.discard(node_id)
                self.mamba_retention_counts["demote_copy_restores"] += 1

        # Commit the loaded KV back onto the node + apply its emitted actions.
        self._apply_cache_actions(
            self.tree_core.commit_load_back(
                node_id, device_indices, kv_xfer, comp_xfers
            )
        )

        self.ongoing_load_back[node_id] = _OngoingLoadBack(
            node_id,
            self.inc_lock_ref(node_id).to_dec_params(),
            host_anchor_params,
        )

        return True

    def _build_sidecar_transfers(
        self,
        phase: CacheTransferPhase,
        kv_xfer: PoolTransfer,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> list[PoolTransfer]:
        transfers: list[PoolTransfer] = []
        for spec in self.sidecar_pool_specs:
            if spec.indices_from_pool == PoolName.KV:
                indices_source = kv_xfer
            else:
                source_component = {
                    PoolName.SWA: ComponentType.SWA,
                    PoolName.MAMBA: ComponentType.MAMBA,
                }.get(spec.indices_from_pool)
                if source_component is None:
                    raise AssertionError(
                        f"Unsupported sidecar indices source pool "
                        f"{spec.indices_from_pool}."
                    )
                matching_sources = comp_xfers.get(source_component, ())
                if not matching_sources:
                    continue
                indices_source = matching_sources[0]
                if indices_source.name != spec.indices_from_pool:
                    raise AssertionError(
                        f"Sidecar indices source pool {spec.indices_from_pool} "
                        f"resolved to {indices_source.name} during {phase}."
                    )

            indices = (
                indices_source.device_indices
                if phase == CacheTransferPhase.BACKUP_HOST
                else indices_source.host_indices
            )
            # Prefetch staging is allocated at hit time, so a prefetch sidecar
            # resolves its indices from its source then, whatever the source.
            deferred = phase == CacheTransferPhase.PREFETCH
            if (indices is None or len(indices) == 0) and not deferred:
                continue
            transfers.append(
                PoolTransfer(
                    name=spec.pool_name,
                    keys=indices_source.keys,
                    hit_policy=spec.hit_policy,
                    indices_from_pool=spec.indices_from_pool,
                )
            )
        return transfers

    def _storage_backup_is_covered(self, spec) -> bool:
        beliefs = self.storage_existence_cache
        if not beliefs.pool(PoolName.KV).covered(spec.hash_value):
            return False
        kv_xfer = PoolTransfer(
            name=PoolName.KV, host_indices=spec.host_value, keys=spec.hash_value
        )
        aux_xfers = [x for xfers in spec.comp_xfers.values() for x in xfers]
        aux_xfers.extend(
            self._build_sidecar_transfers(
                CacheTransferPhase.BACKUP_STORAGE, kv_xfer, spec.comp_xfers
            )
        )
        # KV coverage alone does not prove that a checkpoint or sidecar was written.
        return all(
            transfer.keys is not None
            and beliefs.pool(transfer.name).covered(transfer.keys)
            for transfer in aux_xfers
        )

    @rank_consensus
    def write_backup_storage(self, node_id: NodeId) -> Optional[int]:
        """Issue the node's host->storage backup; returns the operation id, or
        None when nothing was issued (no host copy or all pools covered)."""
        return self._write_backup_storage(node_id)

    def _write_backup_storage(
        self, node_id: NodeId, *, force: bool = False, ledger_reason: str = "write_behind"
    ) -> Optional[int]:
        if not self.enable_storage or self.cache_controller is None:
            return None
        spec = self.tree_core.build_storage_backup_spec(
            node_id, self.hicache_storage_pass_prefix_keys
        )
        if spec is None:
            return None
        # A recompute after L2 eviction re-creates the node with backuped=False, so the
        # flag alone would re-write content L3 holds; stale positives heal on prefetch miss.
        if not force and self._storage_backup_is_covered(spec):
            return None

        kv_xfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=spec.host_value,
            keys=spec.hash_value,
        )
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.BACKUP_STORAGE, kv_xfer, spec.comp_xfers
        )
        aux_xfers = [x for xfers in spec.comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)

        lock_params = self.inc_host_lock_ref(node_id).to_dec_params()
        try:
            operation_id = self.cache_controller.write_storage(
                spec.host_value,
                spec.token_ids,
                spec.hash_value,
                spec.prefix_keys,
                extra_pools=aux_xfers or None,
                **(
                    {"ledger_reason": ledger_reason}
                    if force or self._l3_lifetime_ledger is not None
                    else {}
                ),
            )
        except Exception:
            self.dec_host_lock_ref(node_id, lock_params)
            raise
        self._l3_last_write_has_state = any(x.name == PoolName.MAMBA for x in aux_xfers)
        if self._l3_trace is not None:
            self._l3_trace_write(spec.hash_value, aux_xfers)
        self.ongoing_backup[operation_id] = (node_id, lock_params)
        if self._l3_lifetime_ledger is not None:
            self._l3_lifetime_ledger.emit(
                "storage_write_issue", reason=ledger_reason, node_id=node_id,
                op_id=operation_id, kv_keys=list(spec.hash_value),
                pools={x.name.value: list(x.keys or ()) for x in aux_xfers},
                depth=(
                    len(self.tree_core.get_prefix_hash_values(node_id))
                    + len(spec.hash_value)
                ) * self.page_size,
            )
        return operation_id

    def _write_behind_mamba_leaves(self) -> None:
        """A Mamba host eviction takes a host leaf whole (Full pages and state)
        without passing the Full walk below, so write the host leaves among the
        coldest Mamba host states ahead of it. Rank-replicated inputs, as there."""
        pool = self.cache_controller.mem_pool_host
        deficit = self._l3_mamba_leaf_reserve - pool.available_size(PoolName.MAMBA)
        if deficit <= 0:
            return
        stats = self._l3_tier_stats
        stats["wb_mamba_walks"] += 1
        inflight = set(self._write_behind_mamba_inflight.values())
        for node_id, num_tokens, _ in self.tree_core.peek_mamba_host_eviction_leaves(
            deficit
        ):
            if node_id in inflight:
                continue
            # None when all pools are covered already (or no host copy).
            operation_id = self.write_backup_storage(node_id)
            if operation_id is None:
                continue
            self._write_behind_mamba_inflight[operation_id] = node_id
            stats["wb_mamba_leaf_issued_tokens"] += num_tokens
            stats[f"wb_mamba_ops_{'state' if self._l3_last_write_has_state else 'nostate'}"] += 1

    def _maybe_rotate_beliefs(self) -> None:
        """Rotate the storage beliefs every half TTL, by PP0's clock agreed on all ranks
        (one small collective every 256 steps)."""
        now = time.monotonic()
        due = now - self._l3_beliefs_rotated_at >= self._l3_belief_ttl_s / 2
        if self.agree_on_all_ranks([self.pp_rank == 0 and due])[0]:
            self.storage_existence_cache.rotate()
            self._l3_beliefs_rotated_at = now
            self._l3_tier_stats["belief_rotations"] += 1

    def _write_ahead_device_tail(self) -> None:
        with pp_bubble_diag.sub("write_ahead_tail"):
            self._write_ahead_device_tail_impl()

    def _write_ahead_device_tail_impl(self) -> None:
        """Keep the device LRU tail backed on host so device eviction demotes
        without a D2H wait (SGLANG_HICACHE_DEVICE_WRITE_AHEAD).

        The tail is K x the decayed peak of per-step device evictions. Host
        eviction never reclaims its copies as duplicates; the unbacked ones are
        copied in one batch per step on a private stream behind the forward,
        locked until the rank-agreed ack commits them (_write_ahead_commit). A
        node enters the tail's hot end about K steps before eviction reaches it,
        so its copy has K steps to commit. Every input is rank-replicated: tree
        state, eviction counts and committed acks, never a local event query, so
        all ranks issue the same batch at the same step.
        """
        t0 = time.perf_counter() if pmicro_hoststats.ON else 0.0
        self._wa_step += 1
        self._wa_peak_evicted = max(
            self._wa_peak_evicted * 0.99, float(self._wa_evicted_tokens)
        )
        self._wa_evicted_tokens = 0
        target = int(
            min(
                max(
                    envs.SGLANG_HICACHE_DEVICE_WRITE_AHEAD_K.get()
                    * self._wa_peak_evicted,
                    envs.SGLANG_HICACHE_DEVICE_WRITE_AHEAD_MIN_TOKENS.get(),
                ),
                envs.SGLANG_HICACHE_DEVICE_WRITE_AHEAD_MAX_TOKENS.get(),
            )
        )
        tail = self.tree_core.peek_device_eviction_candidates(
            BASE_COMPONENT_TYPE, target
        )
        self.tree_core.set_write_ahead_spare([node_id for node_id, *_ in tail])
        if (
            len(self._wa_issue_steps)
            >= envs.SGLANG_HICACHE_DEVICE_WRITE_AHEAD_MAX_INFLIGHT.get()
        ):
            return
        cc = self.cache_controller
        # Writes queued by others keep their own ack queue and fence.
        cc.start_writing()
        budget = envs.SGLANG_HICACHE_DEVICE_WRITE_AHEAD_BATCH_MAX_TOKENS.get()
        batch = []
        for node_id, num_tokens, unbacked, pending in tail:
            if not unbacked or pending:
                continue
            if num_tokens > budget:
                break
            batch.append((node_id, num_tokens))
            budget -= num_tokens
        # One host eviction for the whole batch, not one per node.
        anchor_entry = cc.mem_pool_host.anchor_entry
        if batch and (
            anchor_entry.host_pool.shared_allocation_domain is None
            or anchor_entry.host_evict_fn is None
        ):
            needed = sum(n for _, n in batch) - cc.mem_pool_host.available_size()
            if needed > 0:
                self.evict_host(needed)
        issued = 0
        for node_id, num_tokens in batch:
            device_value, comp_xfers = self.tree_core.build_backup_spec(node_id)
            sidecar_xfers = self._build_backup_sidecar(device_value, comp_xfers)
            host_indices = self._execute_kv_backup(
                node_id, device_value, comp_xfers, sidecar_xfers
            )
            if host_indices is None:
                break
            self.tree_core.commit_backup(node_id, host_indices, comp_xfers)
            lock_params = self.inc_lock_ref(node_id).to_dec_params()
            publish_node_ids = self.tree_core.mark_write_through_pending(
                self._backup_publish_node_ids(node_id, comp_xfers), ack_id=node_id
            )
            self.ongoing_write_ahead[node_id] = _OngoingWriteThrough(
                node_id, lock_params, publish_node_ids
            )
            issued += num_tokens
        if issued:
            cc.start_writing(
                ack_queue=cc.ack_write_ahead_queue, stream=self._write_ahead_stream()
            )
            self._wa_issue_steps.append(self._wa_step)
        if pmicro_hoststats.ON:
            pmicro_hoststats.OPS["write_ahead"].add(time.perf_counter() - t0, issued)

    def _write_ahead_stream(self) -> torch.Stream:
        """A private D2H stream: an eviction's fallback copy never queues behind
        a write-ahead batch on the shared one."""
        if self._write_ahead_stream_ is None:
            engine = self.cache_controller.l2_transfer_engine
            if is_cuda() or is_hip():
                from sglang.srt.managers.utils import allocate_distinct_stream

                avoid = [
                    device_module.current_stream(),
                    engine.device_to_host_stream,
                    engine.host_to_device_stream,
                    engine.index_stream,
                ]
                if self.cache_controller.write_fence_stream is not None:
                    avoid.append(self.cache_controller.write_fence_stream)
                self._write_ahead_stream_ = allocate_distinct_stream(
                    device_module, avoid
                )
            else:
                self._write_ahead_stream_ = device_module.Stream()
        return self._write_ahead_stream_

    def _write_ahead_publish_count(self) -> int:
        """PP0's ready write-ahead acks issued at least ACK_LAG walks ago: later
        PP stages issued the same batch at the same walk, so their own copies
        have had as long to land before they commit this count."""
        lag = envs.SGLANG_HICACHE_DEVICE_WRITE_AHEAD_ACK_LAG.get()
        count = 0
        for ack, step in zip(
            self.cache_controller.ack_write_ahead_queue, self._wa_issue_steps
        ):
            if step > self._wa_step - lag or not ack.finish_event.query():
                break
            count += 1
        return count

    def _write_ahead_commit(self, finish_count: int) -> None:
        """Commit the first finish_count write-ahead batches (rank-agreed, FIFO):
        their nodes become backed host duplicates and unlock for eviction."""
        t0 = time.perf_counter() if pmicro_hoststats.ON and finish_count else None
        cc = self.cache_controller
        while finish_count > 0:
            ack = cc.ack_write_ahead_queue.pop(0)
            self._wa_issue_steps.popleft()
            with pp_bubble_diag.wait("ack_sync.write_ahead"):
                ack.finish_event.synchronize()
            for ack_id in ack.node_ids:
                self._finish_write_through_ack(ack_id, self.ongoing_write_ahead)
            self._log_write_ack_metrics(ack)
            finish_count -= 1
        if t0 is not None:
            pmicro_hoststats.OPS["write_ahead_commit"].add(time.perf_counter() - t0)

    def _write_behind_host_tail(self) -> None:
        """Keep the reserve at the LRU host tail either free or already in
        storage, so drive_host_eviction only drops pages storage holds. Every
        input (pool state, tree, beliefs, in-flight set) is rank-replicated."""
        self._write_behind_step += 1
        if self._l3_belief_ttl_s > 0 and self._write_behind_step % 256 == 0:
            self._maybe_rotate_beliefs()
        if self._write_behind_step % self._l3_wb_walk_every:
            return
        walk_start = time.perf_counter()
        if self._l3_mamba_leaf_reserve:
            self._write_behind_mamba_leaves()
        pool = self.cache_controller.mem_pool_host
        target = int(self._l3_evict_write_reserve_fraction * pool.size)
        covered = (
            pool.available_size()
            + sum(self._write_behind_inflight.values())
            + self._write_behind_clean_tokens
        )
        deficit = target - covered
        # Refill in quarter-reserve quanta: the candidate walk heapifies every
        # host leaf, so running it per freed page would eat the scheduler thread.
        if deficit < max(1, target // 4):
            self._l3_tier_stats["wb_walk_us"] += int((time.perf_counter() - walk_start) * 1e6)
            return
        stats = self._l3_tier_stats
        stats["wb_walks"] += 1
        # write_backup_storage repeats this check for its write-through callers;
        # doing it here first lets clean tail tokens count toward the reserve.
        beliefs = self.storage_existence_cache
        clean = 0
        for (
            node_id,
            num_tokens,
            hash_value,
        ) in self.tree_core.peek_host_eviction_candidates(
            BASE_COMPONENT_TYPE, target - pool.available_size()
        ):
            if hash_value and beliefs.pool(PoolName.KV).covered(hash_value):
                spec = self.tree_core.build_storage_backup_spec(
                    node_id, self.hicache_storage_pass_prefix_keys
                )
                if spec is not None and self._storage_backup_is_covered(spec):
                    clean += num_tokens
                    continue
            operation_id = self.write_backup_storage(node_id)
            if operation_id is None:
                # No host copy to write from; the page leaves L2 without an L3 copy.
                stats["wb_unbacked_tokens"] += num_tokens
                continue
            self._write_behind_inflight[operation_id] = num_tokens
            stats["wb_issued_tokens"] += num_tokens
            kind = "state" if self._l3_last_write_has_state else "nostate"
            stats[f"wb_full_ops_{kind}"] += 1
            stats[f"wb_full_tokens_{kind}"] += num_tokens
        self._write_behind_clean_tokens = clean
        stats["wb_clean_tokens"] += clean
        stats["wb_walk_us"] += int((time.perf_counter() - walk_start) * 1e6)

    def is_backuped(self, node_id: NodeId) -> bool:
        return self.tree_core.is_backuped(node_id)

    def is_root(self, node_id: NodeId) -> bool:
        return self.tree_core.is_root(node_id)

    def get_last_hash_value(self, node_id: NodeId) -> Optional[str]:
        return self.tree_core.get_last_hash_value(node_id)

    def get_prefix_hash_values(self, node_id: NodeId) -> list[str]:
        return self.tree_core.get_prefix_hash_values(node_id)

    def rotation_base_of(self, node_id: Optional[NodeId]) -> Optional[int]:
        if node_id is None:
            return None
        return self.tree_core.rotation_base_of(node_id)

    def mamba_storage_match_length(
        self, token_ids, extra_key=None, cache_salt=None
    ) -> int:
        """Usable joint prefix, including device KV behind a host-only state.

        Such KV is exposed to the request only after the existing H2D state
        load; prefix_len + host_hit_length intentionally undercounts it before
        admission and is therefore not the saved-token baseline for another L3 read.
        """
        return self.tree_core.peek_prefix_len(
            RadixKey(
                array("q", token_ids),
                extra_key=extra_key,
                cache_salt=cache_salt,
                is_bigram=self.tree_core.is_eagle,
            )
        )

    def prefetch_mamba_state_within_tree(
        self,
        request: CacheRequestHandle,
        token_ids: list[int],
        matched_tokens: int,
        full_tokens: int,
        full_node_id: Optional[NodeId],
        matched_node_id: Optional[NodeId],
        on_complete: Callable[[], None],
        extra_key: Optional[str] = None,
        cache_salt: Optional[str] = None,
    ) -> bool:
        """Queue state-only recovery before the normal v4 suffix prefetch.

        Enqueue even a local miss: worker collectives must see identical request
        order while PP intake is still relaying requests to later stages.
        """
        if (
            not envs.SGLANG_HICACHE_L3_READ_WITHIN_TREE_KV.get()
            or not self.enable_storage
            or self.host_memory_mode == "buffer_only"
            or not self.supports_mamba()
        ):
            return False
        pending = getattr(self, "_mamba_state_restores", None)
        if pending is None:
            pending = self._mamba_state_restores = {}
        if request in pending:
            return True
        key = RadixKey(
            array("q", token_ids),
            extra_key=extra_key,
            cache_salt=cache_salt,
            is_bigram=self.tree_core.is_eagle,
        ).page_aligned(self.page_size)
        operation = MambaStateRestoreOperation(
            request,
            key,
            matched_tokens,
            full_tokens,
            None,
            self.prefetch_threshold
            if envs.SGLANG_HICACHE_L3_SAVED_TOKEN_THRESHOLD.get()
            else 1,
        )
        operation.on_complete = on_complete
        # Pin both the Full endpoint and the previously usable state. An
        # intermediate state may otherwise be tombstoned during slot allocation.
        if full_node_id is not None and full_tokens > matched_tokens:
            for node_id in dict.fromkeys((full_node_id, matched_node_id)):
                if node_id is None:
                    continue
                device = self.inc_lock_ref(node_id).to_dec_params()
                host = self.inc_host_lock_ref(node_id).to_dec_params()
                operation.restore_locks.append((node_id, device, host))
            operation.state_host_indices = self.components[
                ComponentType.MAMBA
            ].alloc_prefetch_staging(1)
        stats = self._prefetch_outcome_stats
        stats["within_tree_queries"] = stats.get("within_tree_queries", 0) + 1
        if operation.restore_locks and operation.state_host_indices is None:
            stats["within_tree_alloc_failed"] = (
                stats.get("within_tree_alloc_failed", 0) + 1
            )
        pending[request] = operation
        self.cache_controller.prefetch_queue.put(operation)
        return True

    def _cancel_mamba_state_restore(self, request: CacheRequestHandle) -> None:
        operation = getattr(self, "_mamba_state_restores", {}).get(request)
        if operation is not None:
            # The worker owns the destination until its result reaches the
            # scheduler. Never free a slot while a remote get may write to it.
            operation.cancelled = True
            operation.on_complete = None

    def _finish_mamba_state_restore(self, operation) -> None:
        pending = getattr(self, "_mamba_state_restores", {})
        pending.pop(operation.handle, None)
        indices = operation.state_host_indices
        attached = False
        try:
            if not operation.cancelled and operation.state_end_tokens:
                node_id, actions = self.tree_core.prepare_mamba_host_restore(
                    operation.token_ids, operation.state_end_tokens
                )
                self._apply_cache_actions(actions)
                if node_id is not None:
                    attached = self.tree_core.commit_mamba_host_restore(
                        node_id, indices
                    )
            stats = self._prefetch_outcome_stats
            if attached:
                stats["within_tree_hits"] = stats.get("within_tree_hits", 0) + 1
                stats["within_tree_saved_tokens"] = (
                    stats.get("within_tree_saved_tokens", 0)
                    + operation.state_saved_tokens
                )
            elif operation.state_outcome in ("threshold_rejected", "transfer_failed"):
                name = "within_tree_" + operation.state_outcome
                stats[name] = stats.get(name, 0) + 1
            ledger = getattr(self, "_l3_lifetime_ledger", None)
            if ledger is not None:
                end = operation.state_end_tokens
                hashes = operation.state_hashes
                ledger.emit(
                    "state_lookup",
                    wall_ns=operation.query_wall_ns or time.time_ns(),
                    rid=operation.request_id,
                    matched_tokens=operation.matched_tokens,
                    tree_kv_tokens=operation.full_tokens,
                    checkpoint_tokens=end,
                    checkpoint_key=hashes[end // self.page_size - 1] if end else None,
                    request_end_key=hashes[-1] if hashes else None,
                    local_state_keys=[
                        key
                        for key, hit in zip(hashes, operation.state_exists)
                        if hit
                    ],
                    outcome=operation.state_outcome,
                    attached=attached,
                    cancelled=operation.cancelled,
                    saved_tokens=operation.state_saved_tokens if attached else 0,
                    loaded_kv_tokens=0,
                )
        finally:
            if indices is not None and not attached:
                self.components[ComponentType.MAMBA].free_host_values([indices])
            operation.state_host_indices = None
            for node_id, device, host in reversed(operation.restore_locks):
                self.dec_host_lock_ref(node_id, host)
                self.dec_lock_ref(node_id, device)
            operation.restore_locks.clear()
        # State-only reads do not credit Full KV storage tokens. Rematch on the
        # scheduler, then let the unchanged v4 path fetch beyond the tree's KV.
        if operation.on_complete is not None:
            operation.on_complete()

    def query_storage_hit_length(
        self,
        last_host_node_id: NodeId,
        new_input_tokens: list[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[list[str]] = None,
        extra_key: Optional[str] = None,
        cache_salt: Optional[str] = None,
    ) -> int:
        """Probe L3 with the request namespace."""
        if (
            not self.enable_storage
            or self.cache_controller is None
            or self.cache_controller.prefetch_rate_limited()
        ):
            return 0

        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=cache_salt,
        ).page_aligned(self.page_size)
        if len(prefetch_key) < self.prefetch_threshold:
            return 0

        operation = PrefetchOperation(
            CacheRequestHandle("__storage_hit_query__", 0),
            prefetch_key,
            last_hash,
            prefix_keys,
        )
        _, storage_hit_count = self.cache_controller._storage_hit_query(operation)
        storage_hit_count_tensor = torch.tensor(storage_hit_count, dtype=torch.int)
        self._all_reduce_attn_groups(
            storage_hit_count_tensor, torch.distributed.ReduceOp.MIN
        )
        storage_hit_count = storage_hit_count_tensor.item()
        storage_hit_count -= storage_hit_count % self.page_size
        return storage_hit_count

    @rank_consensus(same_params=["request.rid", "len(new_input_tokens)"])
    def prefetch_from_storage(
        self,
        request: CacheRequestHandle,
        last_host_node_id: NodeId,
        new_input_tokens: list[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[list[str]] = None,
        matched_prefix_tokens: Optional[list[int]] = None,
        extra_key: Optional[str] = None,
        cache_salt: Optional[str] = None,
        storage_hit_end: Optional[int] = None,
        tree_kv_tokens: int = 0,
    ) -> Optional[bool]:
        """tree_kv_tokens: stateless tree KV between the deepest state and this anchor; a
        hit resumes the request through it, so it counts toward the threshold."""
        if not self.enable_storage or self.cache_controller is None:
            return

        req_id = request.rid
        self.storage_prefetch_retries.cancel(req_id)
        wait = self._storage_prefetch_wait.get(request)
        if wait is not None and wait.expired:
            return
        submission = self.cache_controller.get_prefetch_submission(req_id)
        if submission is not None:
            return submission.decision

        buffer_mode = self.host_memory_mode == "buffer_only"
        if request in self.ongoing_prefetch or (
            self.buffer_pipeline is not None
            and self.buffer_pipeline.has_staged(request)
        ):
            return
        # Prefix-covered writes justify extending a known hit to the left,
        # never beyond its confirmed right endpoint (including the SWA window).
        storage_start = len(matched_prefix_tokens or [])
        boundary_tokens = int(self.tree_core.is_eagle)
        assume_stored = (
            storage_hit_end is not None
            and storage_start
            < storage_hit_end
            <= storage_start + len(new_input_tokens) - boundary_tokens
        )
        if assume_stored:
            new_input_tokens = new_input_tokens[
                : storage_hit_end - storage_start + boundary_tokens
            ]
        # Key the span by the request's namespace, not the anchor's (a root
        # anchor has none): a span published under the wrong namespace gets
        # re-owned by the request's own insert (double free).
        anchor_extra_key, anchor_cache_salt = self.tree_core.prefetch_anchor_info(
            last_host_node_id
        )
        assert (anchor_extra_key is None or anchor_extra_key == extra_key) and (
            anchor_cache_salt is None or anchor_cache_salt == cache_salt
        ), (
            f"prefetch anchor namespace {(anchor_extra_key, anchor_cache_salt)} "
            f"!= request namespace {(extra_key, cache_salt)}"
        )
        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=cache_salt,
        ).page_aligned(self.page_size)
        assume_stored = (
            assume_stored and storage_start + len(prefetch_key) == storage_hit_end
        )
        prefetch_length = len(prefetch_key)
        stats = self._prefetch_outcome_stats
        if prefetch_length > 0:
            stats["attempts"] += 1
        if prefetch_length + tree_kv_tokens < self.prefetch_threshold:
            if prefetch_length > 0:
                stats["declined_too_short"] += 1
            # No lookup was issued, so no retry is armed: polling here would
            # spend the re-issue budget the admission-time device-hit-loss
            # re-query needs once the device match evicts while queued.
            return
        if not buffer_mode and self.cache_controller.prefetch_rate_limited():
            stats["declined_rate_limited"] += 1
            # Paced: the limiter clears as transfers finish, not on the next pass.
            self.storage_prefetch_retries.poll_miss(req_id, storage_hit_end)
            return

        # Buffer mode holds no tree state during the fetch: buffers are
        # operation-owned, so the anchor needs no pin.
        anchor_lock_params = (
            None
            if buffer_mode
            else self.inc_host_lock_ref(last_host_node_id).to_dec_params()
        )
        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue
            # Size the component's staging now; it is allocated at hit time,
            # next to the KV staging, so the query holds no host memory.
            prep = self.components[ct].prepare_prefetch(
                last_host_node_id, prefetch_tokens=len(prefetch_key)
            )
            if prep.staging_tokens == 0:
                continue
            transfers = self.tree_core.build_hicache_transfers(
                ct,
                last_host_node_id,
                CacheTransferPhase.PREFETCH,
                token_ids=prefetch_key.token_ids,
                prefetch_tokens=len(prefetch_key),
                staging_tokens=prep.staging_tokens,
                last_hash=last_hash,
            )
            if transfers:
                comp_xfers[ct] = transfers
        if buffer_mode and ComponentType.FULL in self.tree_components:
            transfers = self.components[ComponentType.FULL].build_hicache_transfers(
                None,
                CacheTransferPhase.PREFETCH,
                prefix_key=RadixKey(
                    array("q", matched_prefix_tokens or ()),
                    extra_key=extra_key,
                    is_bigram=self.tree_core.is_eagle,
                    cache_salt=cache_salt,
                ),
            )
            if transfers:
                comp_xfers[ComponentType.FULL] = transfers
        kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=None)
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.PREFETCH, kv_xfer, comp_xfers
        )

        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        submission = self.cache_controller.submit_prefetch(
            request,
            prefetch_key,
            last_hash,
            prefix_keys,
            matched_prefix_tokens,
            aux_xfers or None,
            assume_stored=assume_stored,
        )
        operation = submission.operation
        if operation is None:
            assert submission.decision is not None
            self.cache_controller.append_host_mem_release(extra_pools=aux_xfers or None)
            return submission.decision

        stats["issued"] += 1
        if operation.assume_stored:
            stats["issued_assumed_stored"] += 1
        if tree_kv_tokens:
            stats["issued_past_tree_kv"] += 1
        operation.stats_tree_kv_tokens = tree_kv_tokens
        # Keep the requested span stable for the wait budget and miss accounting;
        # the hit hash chain can shrink as device coverage changes.
        operation.stats_requested_tokens = prefetch_length
        operation.storage_start = storage_start
        if submission.decision is not None:
            return submission.decision

        self.ongoing_prefetch[request] = _OngoingPrefetch(
            last_host_node_id,
            prefetch_key,
            None,
            operation,
            anchor_lock_params,
            comp_xfers,
        )
        if buffer_mode:
            # The query reads no tree state; pinning across it would hold the
            # anchor cap for a storage round trip, so the pin waits for IO commit.
            self.buffer_pipeline.set_prefix_ctx(
                request,
                matched_prefix_tokens,
                extra_key=extra_key,
                cache_salt=cache_salt,
            )
        else:
            # Cache mode reserves the requested span up front; buffer mode
            # grants occupancy later at hit-alloc time, sized to the hit.
            self.cache_controller.prefetch_tokens_occupied += len(prefetch_key)

    def _start_storage_prefetch_wait(self, operation: PrefetchOperation) -> None:
        if (
            self.prefetch_stop_policy == "timeout"
            and operation.handle not in self._storage_prefetch_wait
        ):
            # Start at the first blocked admission poll; refetches keep this budget.
            pages = operation.stats_requested_tokens // self.page_size
            wait = _StoragePrefetchWait(
                timeout_s=self.prefetch_timeout_base
                + pages * self.prefetch_timeout_per_page,
            )
            wait.timer.observe(ready=False)
            self._storage_prefetch_wait[operation.handle] = wait

    def _prefetch_deadline_expired(self, operation: PrefetchOperation) -> bool:
        if operation.host_indices is not None and (
            operation.completed_tokens >= len(operation.hash_value) * self.page_size
        ):
            # KV landed: cutting the pending sidecar read would salvage nothing.
            return False
        wait = self._storage_prefetch_wait.get(operation.handle)
        return wait is not None and (
            wait.expired
            or (wait.timer.running and wait.timer.elapsed() > wait.timeout_s)
        )

    def _hold_inflight_hybrid_prefetch(self, operation: PrefetchOperation) -> bool:
        """SGLANG_HICACHE_L3_INFLIGHT_ADMISSION_WAITS: under best_effort, a read that is
        all-or-nothing (it carries a Mamba / SWA transfer) and already has its host
        staging (IO in flight) gets that many admission polls to finish; cutting it at
        the first poll discards the whole read. Rank-replicated: the staging is allocated
        and the completion committed in the synced drains, and admission polls run in
        lockstep. The cut after the last poll is a policy deadline, not absent data."""
        waits = getattr(self, "_inflight_admission_waits", 0)
        if (
            not waits
            or self.prefetch_stop_policy != "best_effort"
            or operation.host_indices is None
            or not any(
                t.indices_from_pool != PoolName.KV for t in operation.pool_transfers or ()
            )
        ):
            return False
        held = getattr(operation, "admission_waits", 0)
        if held >= waits:
            self._mark_prefetch_deadline_expired(operation)
            return False
        operation.admission_waits = held + 1
        stats = self._prefetch_outcome_stats
        stats["inflight_admission_waits"] = stats.get("inflight_admission_waits", 0) + 1
        if held == 0:
            stats["inflight_admission_held"] = stats.get("inflight_admission_held", 0) + 1
        return True

    def _mark_prefetch_deadline_expired(self, operation: PrefetchOperation) -> None:
        operation.deadline_expired = True
        self.storage_prefetch_retries.cancel(operation.handle.rid)
        # Only a rank-agreed verdict can suppress retries, including on peers
        # that have not reached their conditional admission poll yet.
        self._storage_prefetch_wait.setdefault(
            operation.handle, _StoragePrefetchWait()
        ).expired = True

    @rank_consensus(same_results=True)
    def _can_terminate_prefetch(self, operation: PrefetchOperation) -> bool:
        if self.prefetch_stop_policy == "best_effort":
            return True
        if self.prefetch_stop_policy == "wait_complete":
            # Completion is committed by the ACK drain (sync-thread MIN-reduced
            # progress, rank-min queue drain); a vote here only adds collectives.
            return False
        if self.prefetch_stop_policy == "timeout":
            # Wall clocks differ across ranks: PP0 decides, TP takes MAX (any
            # rank timed out) and _all_reduce broadcasts the verdict along PP.
            should_terminate = False
            if self.pp_rank == 0:
                should_terminate = self._prefetch_deadline_expired(operation)
            should_terminate_tensor = torch.tensor(
                int(should_terminate), dtype=torch.int, device="cpu"
            )
            self._all_reduce(should_terminate_tensor, torch.distributed.ReduceOp.MAX)
            return should_terminate_tensor.item() == 1
        return True

    def has_ongoing_prefetch(self, handle: CacheRequestHandle) -> bool:
        return handle in self.ongoing_prefetch

    def bind_prefetch_ticket(self, req_id: str, decision: bool = True) -> None:
        # PP0 also records misses/skips; only hits travel with the request relay.
        with self.cache_controller.pp_prefetch_state_lock:
            self.cache_controller.pp_prefetch_decisions[req_id] = (
                PPPrefetchDecision.TICKETED if decision else PPPrefetchDecision.SKIPPED
            )

    def _check_pp_prefetch_progress(self, request: CacheRequestHandle) -> bool:
        req_id = request.rid
        ready = self.pp_rank == 0 and self.cache_controller.is_pp_prefetch_ready(req_id)
        ready_tensor = torch.tensor(int(ready), dtype=torch.int, device="cpu")
        self._all_reduce(ready_tensor, torch.distributed.ReduceOp.MAX)
        if ready_tensor.item() == 0:
            return False

        state = self.cache_controller.take_ready_pp_prefetch(req_id)
        if state is None:
            raise RuntimeError(
                f"PP prefetch became ready before local state existed: {req_id}"
            )

        operation = state.operation
        if operation.host_indices is None or operation.completed_tokens == 0:
            self.prefetch_loaded_tokens_by_reqid[request] = 0
            return True

        ticket = state.ticket
        prefetch_key = ticket.prefetch_key
        self.ongoing_prefetch[request] = _OngoingPrefetch(
            self.root_node_handle(prefetch_key.extra_key),
            prefetch_key,
            operation.host_indices,
            operation,
            None,
            {
                BASE_COMPONENT_TYPE: [
                    transfer
                    for transfer in operation.pool_transfers or []
                    if transfer.indices_from_pool is None
                ]
            },
        )
        self.buffer_pipeline.set_prefix_ctx(
            request,
            ticket.matched_prefix_tokens,
            extra_key=prefetch_key.extra_key,
            cache_salt=prefetch_key.cache_salt,
        )
        self.buffer_pipeline.try_lock_anchor(request, operation.completed_tokens)
        self.cache_controller.append_host_mem_release(
            operation.host_indices[operation.completed_tokens :]
        )
        self._handle_prefetch_result(operation)
        return True

    def waiting_storage_span(
        self, request: CacheRequestHandle
    ) -> Optional[tuple[int, int]]:
        if self.buffer_pipeline is None:
            return None
        info = self.ongoing_prefetch.get(request)
        if info is not None:
            if info.host_indices is None:
                return None
            # Scheduler-accepted logical span; worker progress can count pool rows.
            op = info.operation
            return op.storage_start, op.storage_start + op.storage_hit_count
        staged = self.buffer_pipeline.staged_prefetches.get(request)
        if staged is not None:
            return staged.matched_len, staged.matched_len + staged.num_tokens
        return None

    def prefetch_skip_reason(self, request: CacheRequestHandle) -> str:
        """Why check_prefetch_progress just returned False (diagnostics only)."""
        if request in getattr(self, "_mamba_state_restores", {}):
            return "kda_restore_pending"
        if (
            self.cache_controller is not None
            and self.cache_controller.pp_prefetch_decisions.get(request.rid)
            is PPPrefetchDecision.TICKETED
        ):
            return "ticket_not_ready"
        if request in self.ongoing_prefetch:
            return "prefetch_running"
        return "other"

    @rank_consensus(same_params=True, same_results=True)
    def check_prefetch_progress(self, request: CacheRequestHandle) -> bool:
        if request in getattr(self, "_mamba_state_restores", {}):
            return False
        if (
            self.cache_controller is not None
            and self.cache_controller.pp_prefetch_decisions.get(request.rid)
            is PPPrefetchDecision.TICKETED
        ):
            return self._check_pp_prefetch_progress(request)

        if request not in self.ongoing_prefetch:
            return True

        operation = self.ongoing_prefetch[request].operation

        if self._hold_inflight_hybrid_prefetch(operation):
            return False
        if not operation.is_terminated():
            if not self._can_terminate_prefetch(operation):
                self._start_storage_prefetch_wait(operation)
                return False
            if self.prefetch_stop_policy == "timeout":
                self._mark_prefetch_deadline_expired(operation)

        self.cache_controller.terminate_prefetch(operation)
        if operation.host_indices is None:
            if operation.deadline_expired:
                self._finish_storage_prefetch(request, 0, reason="deadline")
            else:
                self.storage_prefetch_retries.poll_miss(request.rid)
            self.revoke_pending_prefetch(request)
        else:
            self._handle_prefetch_result(operation)
        return True

    def _handle_prefetch_result(self, operation: PrefetchOperation) -> None:
        # This function **owns**:
        # - host_indices[0 : completed_tokens]
        # - sidecar pool hits if operation.pool_transfers_done is true
        #
        # That is, when this function returns the host memory referenced must be inserted
        # into the radix tree or released to pool.

        request = operation.handle
        completed_tokens = operation.completed_tokens
        hash_value = operation.hash_value

        (
            last_host_node_id,
            prefetch_key,
            host_indices,
            _,
            anchor_lock_params,
            comp_xfers,
        ) = self.ongoing_prefetch[request]

        # All PP/TP ranks will get the same `min_completed_tokens`, because `completed_tokens`
        # and `pool_hits` in their operations are same.  No need to sync cross-rank here.
        if not self._check_hybrid_prefetch_result(
            request, operation, hash_value, host_indices
        ):
            # Hybrid all-or-nothing check failed; result already discarded.
            return

        allocated_tokens = len(host_indices)
        if completed_tokens < allocated_tokens:
            self._resolve_storage_prefetch_tokens(
                request,
                allocated_tokens - completed_tokens,
                reason="deadline" if operation.deadline_expired else "storage_transfer",
            )
        if (
            completed_tokens > 0
            and self.enable_storage_metrics
            and self.storage_metrics_collector is not None
        ):
            self.storage_metrics_collector.log_prefetched_tokens(completed_tokens)
        if self._storage_write_back is not None:
            self._storage_write_back.on_storage_read(completed_tokens)
        if self._l3_tier_final is not None:
            self._l3_tier_final.read_tokens += completed_tokens

        if self.buffer_pipeline is not None:
            # No graft: release the rank-local tail beyond the synced usable
            # length, then park the bounce for admission-time consumption.
            return self.buffer_pipeline.stage_completed_prefetch(
                request, completed_tokens, hash_value
            )

        fetched_key = prefetch_key[:completed_tokens]
        insert_result = self.tree_core.insert_host(
            last_host_node_id,
            fetched_key,
            host_indices[:completed_tokens],
            hash_value[: completed_tokens // self.page_size],
        )

        # Apply the host-insert walk's actions before the transfer commit.
        self._apply_cache_actions(insert_result.cache_actions)

        # The rank-synced fetch proves L3 presence even if the host insert is dropped.
        self.storage_existence_cache.pool(PoolName.KV).add(
            hash_value[: completed_tokens // self.page_size]
        )

        if insert_result.host_insert_dropped:
            if self._storage_write_back is not None:
                self._storage_write_back.on_storage_read_dropped(completed_tokens)
            self._resolve_storage_prefetch_tokens(request, insert_result.prefix_len)
            self._finish_storage_prefetch(request, fulfilled_tokens=0, reason="dropped")
            self.cache_controller.append_host_mem_release(
                host_indices=host_indices[:completed_tokens],
                extra_pools=[x for xfers in comp_xfers.values() for x in xfers],
            )
            loaded_from_storage = 0
        else:
            commit_actions: list[CacheAction | ComponentAction] = []
            self.tree_core.commit_hicache_transfers(
                last_host_node_id,
                CacheTransferPhase.PREFETCH,
                comp_xfers,
                cache_actions=commit_actions,
                insert_result=insert_result,
                pool_storage_result=operation.pool_storage_result,
            )
            self._apply_cache_actions(commit_actions)
            # The commit emits via commit_actions only; the walk's were applied above.
            assert not insert_result.cache_actions

            self.cache_controller.mem_pool_host.free(
                host_indices[: insert_result.prefix_len]
            )
            loaded_from_storage = completed_tokens - insert_result.prefix_len
            if self._storage_write_back is not None and insert_result.prefix_len:
                self._storage_write_back.on_storage_read_dropped(
                    insert_result.prefix_len
                )
            # Cache mode has only completed L3 -> L2 here. Keep the usable
            # storage span unresolved until admission proves that L2 -> L1
            # load-back actually materialized it for this request.
            self._resolve_storage_prefetch_tokens(request, insert_result.prefix_len)

        self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
        del self.ongoing_prefetch[request]
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)

        self.prefetch_loaded_tokens_by_reqid[request] = loaded_from_storage
        if loaded_from_storage > 0:
            self.prefetch_loaded_storage_start_by_reqid[request] = (
                operation.storage_start + insert_result.prefix_len
            )
        else:
            self.prefetch_loaded_storage_start_by_reqid.pop(request, None)
        logger.info(
            "HiCache prefetch %s req=%s completed=%d matched=%d loaded=%d occupied=%d",
            "dropped" if insert_result.host_insert_dropped else "success",
            request.rid,
            completed_tokens,
            insert_result.prefix_len,
            loaded_from_storage,
            self.cache_controller.prefetch_tokens_occupied,
        )
        return

    def _check_hybrid_prefetch_result(
        self,
        request: CacheRequestHandle,
        operation: PrefetchOperation,
        hash_value: list[str],
        host_indices: torch.Tensor,
    ) -> bool:
        """Decide the length of usable prefix.

        Two strategies depending on the hybrid layout:

        * DSA-style (Full attention + KV-derived ALL_PAGES sidecar such as the
          DSA / MiniMax indexer): *clamp* to the minimum fetched prefix shared by
          the Full KV pool and every sidecar. A partial prefix is still usable
          because the sidecar is page-aligned with KV and required for every page.
        * Everything else (SWA / Mamba components, mixed DeepSeekV4 stacks):
          *all-or-nothing*. Their pools only cover a window / tail and cannot be
          truncated page by page, so any shortfall discards the whole prefetch.

        Returns true if prefetch success, or false when an all-or-nothing prefetch
        was discarded (the caller should then treat the prefetch as finished).
        """
        # Sync completed tokens and per-pool hit pages across ATTN groups, taking
        # the minimum so every rank agrees on the same usable prefix length.
        #
        # Skip KV-derived pools, which do not report hits in operation.pool_storage_result.
        # Their hit lengths are stored in completed_tokens.
        pool_transfers = [
            transfer
            for transfer in operation.pool_transfers or []
            if transfer.indices_from_pool != PoolName.KV
        ]
        hit_pages = (
            operation.pool_storage_result.extra_pool_hit_pages if pool_transfers else {}
        )
        pool_hit_pages = [hit_pages.get(t.name, 0) for t in pool_transfers]
        completed_tokens = operation.completed_tokens
        # Hybrid cache state is all-or-nothing: every extra pool (SWA / Mamba / ...)
        # must cover the same fetched prefix. If any pool falls short the whole
        # prefetch result is unusable, so discard it and release everything.
        expected_tokens = len(hash_value) * self.page_size
        all_succeeded = completed_tokens == expected_tokens and all(
            transfer.keys is not None and count == len(transfer.keys)
            for transfer, count in zip(pool_transfers, pool_hit_pages)
        )
        if pool_transfers and not all_succeeded:
            # Drop each pool's beliefs beyond what it served (KV and its
            # sidecars beyond the completed span); the next insert re-writes
            # only the missing pools. A deadline is not evidence of absent data.
            invalidation_hashes = (
                operation.sidecar_hash_values
                if operation.sidecar_hash_values is not None
                else hash_value
            )
            trimmed_pages = len(invalidation_hashes) - len(hash_value)
            keep_pages = trimmed_pages + completed_tokens // self.page_size
            beliefs = self.storage_existence_cache
            if not operation.deadline_expired and keep_pages < len(invalidation_hashes):
                self._belief_policy(
                    PoolName.KV, PoolHitPolicy.ALL_PAGES
                ).heal_from_transfer(
                    beliefs.pool(PoolName.KV), invalidation_hashes, keep_pages
                )
                for transfer in operation.pool_transfers or ():
                    if transfer.indices_from_pool == PoolName.KV:
                        self._belief_policy(
                            transfer.name, transfer.hit_policy
                        ).heal_from_transfer(
                            beliefs.pool(transfer.name),
                            invalidation_hashes,
                            keep_pages,
                        )
            for transfer, count in zip(pool_transfers, pool_hit_pages):
                if (
                    not operation.deadline_expired
                    and transfer.keys is not None
                    and count < len(transfer.keys)
                ):
                    self._belief_policy(
                        transfer.name, transfer.hit_policy
                    ).heal_from_transfer(
                        beliefs.pool(transfer.name), transfer.keys, count
                    )
            reason = "deadline" if operation.deadline_expired else "storage_transfer"
            self._finish_storage_prefetch(request, fulfilled_tokens=0, reason=reason)
            # The controller's prefetch IO thread already releases the untransferred
            # tail (host_indices[completed_tokens:])
            self._retire_ongoing_prefetch(
                request,
                host_indices[:completed_tokens],
                pool_transfers if operation.pool_transfers_done else None,
            )
            self.prefetch_loaded_tokens_by_reqid[request] = 0
            self.prefetch_loaded_storage_start_by_reqid.pop(request, None)
            logger.warning(
                "HiCache hybrid prefetch discarded req=%s reason=%s completed=%d "
                "requested=%d kv_beliefs_kept_pages=%s aux_pool_hit_pages=%s",
                request.rid,
                reason,
                completed_tokens,
                expected_tokens,
                None if operation.deadline_expired else keep_pages,
                {
                    getattr(transfer.name, "value", transfer.name): count
                    for transfer, count in zip(pool_transfers, pool_hit_pages)
                },
            )
            return False
        return True

    def _record_storage_prefetch_hit(
        self, request: CacheRequestHandle, num_tokens: int
    ) -> None:
        """Start accounting one rank-agreed positive L3 query result."""
        if self._l3_lifetime_ledger is not None and num_tokens > 0:
            pending = getattr(self, "_l3_ledger_prefetch", None)
            if pending is None:
                pending = self._l3_ledger_prefetch = {}
            previous = pending.pop(request, None)
            if previous is not None:
                self._l3_lifetime_ledger.emit(
                    "storage_prefetch_discard", rid=request.rid,
                    op_id=previous["op_id"], remaining_tokens=previous["remaining"],
                    reason="replaced",
                )
            info = self.ongoing_prefetch.get(request)
            operation = info.operation if info is not None else None
            pending[request] = {
                "remaining": num_tokens,
                "storage_start": operation.storage_start if operation else None,
                "op_id": operation.id if operation else None,
            }
        if (
            num_tokens <= 0
            or not self.enable_storage_metrics
            or self.storage_metrics_collector is None
        ):
            return
        if request in self._storage_prefetch_hit_remaining_by_reqid:
            logger.warning(
                "Replacing unresolved storage-hit accounting req=%s old=%d new=%d",
                request.rid,
                self._storage_prefetch_hit_remaining_by_reqid[request],
                num_tokens,
            )
            self._storage_prefetch_hit_remaining_by_reqid.pop(request, None)
        self._storage_prefetch_hit_remaining_by_reqid[request] = num_tokens
        self.storage_metrics_collector.log_storage_prefetch_hit_tokens(num_tokens)

    def _resolve_storage_prefetch_tokens(
        self, request: CacheRequestHandle, num_tokens: int, reason: Optional[str] = None
    ) -> None:
        if num_tokens <= 0:
            return
        if self._l3_lifetime_ledger is not None:
            entry = getattr(self, "_l3_ledger_prefetch", {}).get(request)
            if entry is not None:
                resolved = min(num_tokens, entry["remaining"])
                entry["remaining"] -= resolved
                self._l3_lifetime_ledger.emit(
                    "storage_prefetch_resolved", rid=request.rid,
                    op_id=entry["op_id"], storage_start=entry["storage_start"],
                    resolved_tokens=resolved, reason=reason or "tree_overlap",
                )
        remaining = self._storage_prefetch_hit_remaining_by_reqid.get(request)
        if remaining is None:
            return
        dropped = min(num_tokens, remaining)
        if num_tokens > remaining:
            logger.warning(
                "Storage-prefetch accounting exceeded remaining "
                "tokens req=%s requested=%d remaining=%d reason=%s",
                request.rid,
                num_tokens,
                remaining,
                reason,
            )
        if reason is not None:
            self.storage_metrics_collector.log_storage_prefetch_unfulfilled_tokens(
                dropped, reason
            )
        remaining -= dropped
        if remaining:
            self._storage_prefetch_hit_remaining_by_reqid[request] = remaining
        else:
            self._storage_prefetch_hit_remaining_by_reqid.pop(request, None)

    def _finish_storage_prefetch(
        self, request: CacheRequestHandle, fulfilled_tokens: int, reason: Optional[str]
    ) -> None:
        if self._l3_lifetime_ledger is not None:
            entry = getattr(self, "_l3_ledger_prefetch", {}).pop(request, None)
            if entry is not None:
                fulfilled = min(fulfilled_tokens, entry["remaining"])
                self._l3_lifetime_ledger.emit(
                    "storage_prefetch_finish", rid=request.rid,
                    op_id=entry["op_id"], storage_start=entry["storage_start"],
                    fulfilled_tokens=fulfilled,
                    unfulfilled_tokens=entry["remaining"] - fulfilled, reason=reason,
                )
        remaining = self._storage_prefetch_hit_remaining_by_reqid.pop(request, None)
        if remaining is None:
            return
        if fulfilled_tokens > remaining:
            logger.warning(
                "Storage-prefetch fulfilled accounting exceeded remaining "
                "tokens req=%s fulfilled=%d remaining=%d",
                request.rid,
                fulfilled_tokens,
                remaining,
            )
        unfulfilled = max(0, remaining - fulfilled_tokens)
        if reason is not None:
            self.storage_metrics_collector.log_storage_prefetch_unfulfilled_tokens(
                unfulfilled, reason
            )

    def _settle_storage_prefetch_hit(
        self, request: CacheRequestHandle, credited_tokens: int
    ) -> None:
        """At buffer-mode admission, hit tokens not credited to storage were
        covered by the device's joint match; the rest resolve at the fill ack."""
        remaining = self._storage_prefetch_hit_remaining_by_reqid.get(request)
        if remaining is not None:
            self._resolve_storage_prefetch_tokens(
                request, remaining - credited_tokens, reason="device_covered"
            )

    def finish_storage_prefetch_admission(
        self, request: CacheRequestHandle, fulfilled_tokens: int, reason: Optional[str]
    ) -> None:
        """Close the admission wait budget and resolve a cache-mode L3 hit.

        Buffer mode resolves at the H2D completion ack instead, because its
        host allocation is transport staging rather than a resident L2 hit.
        """
        self._storage_prefetch_wait.pop(request, None)
        if self.host_memory_mode == "cache":
            self._finish_storage_prefetch(request, fulfilled_tokens, reason)

    def discard_storage_prefetch_accounting(self, request: CacheRequestHandle) -> None:
        """Drop lifecycle state for cases intentionally excluded from metrics."""
        if self._l3_lifetime_ledger is not None:
            entry = getattr(self, "_l3_ledger_prefetch", {}).pop(request, None)
            if entry is not None:
                self._l3_lifetime_ledger.emit(
                    "storage_prefetch_discard", rid=request.rid,
                    op_id=entry["op_id"], remaining_tokens=entry["remaining"],
                )
        self._storage_prefetch_hit_remaining_by_reqid.pop(request, None)

    def _handle_storage_prefetch_anchor_loss(self, request: CacheRequestHandle) -> None:
        operation = self.ongoing_prefetch[request].operation
        storage_hit_end = operation.storage_start + operation.storage_hit_count
        self._finish_storage_prefetch(request, fulfilled_tokens=0, reason="anchor_lost")
        self.revoke_pending_prefetch(request)
        self.storage_prefetch_retries.refetch(request.rid, storage_hit_end)

    def _log_storage_prefetch_deferred(self, num_tokens: int, reason: str) -> None:
        if self.enable_storage_metrics and self.storage_metrics_collector is not None:
            self.storage_metrics_collector.log_storage_prefetch_deferred_tokens(
                num_tokens, reason
            )

    def pop_prefetch_loaded_tokens(self, request: CacheRequestHandle) -> int:
        self.prefetch_loaded_storage_start_by_reqid.pop(request, None)
        return self.prefetch_loaded_tokens_by_reqid.pop(request, 0)

    def pop_prefetch_loaded_span(
        self, request: CacheRequestHandle
    ) -> tuple[int, Optional[int]]:
        """Pop the loaded L3 token count and its absolute prefix start."""
        loaded = self.prefetch_loaded_tokens_by_reqid.pop(request, 0)
        if loaded and self._storage_write_back is not None:
            self._storage_write_back.on_storage_hit(loaded)
        return (
            loaded,
            self.prefetch_loaded_storage_start_by_reqid.pop(request, None),
        )

    @rank_consensus(same_params=True)
    def release_aborted_request(self, request: CacheRequestHandle) -> None:
        self._cancel_mamba_state_restore(request)
        rid = request.rid
        self._storage_prefetch_wait.pop(request, None)
        if self.linker is not None:
            self.linker.release_request(rid)
        self.prefetch_loaded_tokens_by_reqid.pop(request, None)
        self.prefetch_loaded_storage_start_by_reqid.pop(request, None)
        self.storage_prefetch_retries.cancel(rid)
        if (
            self.buffer_pipeline is not None
            and self.cache_controller.pp_prefetch_command_group is not None
            and self.cache_controller.release_pp_prefetch(rid)
        ):
            return
        if (
            self.buffer_pipeline is not None
            and self.buffer_pipeline.release_staged_hold(request)
        ):
            return
        self.discard_storage_prefetch_accounting(request)
        if request not in self.ongoing_prefetch:
            return

        info = self.ongoing_prefetch[request]
        if info.operation.host_indices is None:
            self.cache_controller.terminate_prefetch(info.operation)
            self.revoke_pending_prefetch(request)
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(info.operation)
        self._retire_ongoing_prefetch(
            request,
            info.host_indices[:completed_tokens],
            (
                [x for xfers in info.comp_xfers.values() for x in xfers]
                if info.operation.pool_transfers_done
                else None
            ),
        )

    def _retire_ongoing_prefetch(
        self,
        request: CacheRequestHandle,
        host_indices: Optional[torch.Tensor],
        extra_pools: Optional[list[PoolTransfer]],
    ) -> None:
        """Drop an in-flight prefetch, releasing ``host_indices`` and ``extra_pools``."""
        info = self.ongoing_prefetch.pop(request)
        if info.anchor_lock_params is not None:
            self.dec_host_lock_ref(info.anchor_node_id, info.anchor_lock_params)
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.pop_prefix_ctx(request)
            self.buffer_pipeline.release_anchor_lock(request)
        cc = self.cache_controller
        cc.append_host_mem_release(host_indices=host_indices, extra_pools=extra_pools)
        cc.prefetch_tokens_occupied = max(
            0,
            cc.prefetch_tokens_occupied
            - self._prefetch_occupied_span(
                info.prefetch_key, info.host_indices, operation=info.operation
            ),
        )

    def _belief_policy(
        self, pool: PoolName, hit_policy: PoolHitPolicy
    ) -> PoolBeliefPolicy:
        """The policy healing ``pool``'s existence beliefs: the owning
        component's, else the built-in one for the transfer's hit policy."""
        for component in self.components.values():
            policy = component.buffer_belief_policy(pool)
            if policy is not None:
                return policy
        return default_belief_policy(hit_policy)

    def _invalidate_absent_from_hit_query(self, operation) -> None:
        """Heal each pool's beliefs beyond its rank-agreed boundary (a
        chain-page pool also learns presence up to it, unless
        SGLANG_HICACHE_L3_BELIEF_QUERY_RENEW=0); a pool without a verdict
        heals from the folded cut."""
        if operation.deadline_expired:
            # A cut query's agreed hit may be a MIN over skipped lookups, not a verdict.
            return
        chain = operation.all_hash_values
        if chain is None:
            return
        beliefs = self.storage_existence_cache
        verdicts = operation.query_pool_hit_pages
        # No rank-agreed KV query evidence; leave beliefs unchanged.
        if PoolName.KV not in verdicts:
            return
        folded_pages = operation.storage_hit_count // self.page_size

        def verdict(pool: PoolName) -> tuple[int, bool]:
            pages = verdicts.get(pool)
            return (folded_pages, False) if pages is None else (pages, True)

        kv_pages, kv_evidence = verdict(PoolName.KV)
        renew = self._l3_belief_query_renew
        self._belief_policy(PoolName.KV, PoolHitPolicy.ALL_PAGES).heal_from_query(
            beliefs.pool(PoolName.KV),
            chain,
            chain,
            kv_pages,
            evidence=kv_evidence,
            renew=renew,
        )
        for transfer in operation.pool_transfers or ():
            hit_pages, evidence = verdict(transfer.name)
            self._belief_policy(transfer.name, transfer.hit_policy).heal_from_query(
                beliefs.pool(transfer.name),
                chain,
                list(transfer.keys or ()),
                hit_pages,
                evidence=evidence,
                renew=renew,
            )

    def _account_prefetch_outcome(self, operation, revoked: bool) -> None:
        """Feed the cumulative prefetch-outcome counters at the (rank-synced)
        query outcome: T = prompt tokens, L = requested, m = L3-miss."""
        requested = operation.stats_requested_tokens
        if requested <= 0 or operation.deadline_expired:
            return
        stats = self._prefetch_outcome_stats
        hit = max(0, min(operation.storage_hit_count, requested))
        if self._l3_lifetime_ledger is not None:
            chain = operation.all_hash_values or operation.hash_value
            hit_pages = hit // self.page_size
            self._l3_lifetime_ledger.emit(
                "storage_lookup", rid=operation.request_id, op_id=operation.id,
                M=operation.storage_start - operation.stats_tree_kv_tokens,
                K=operation.storage_start,
                D=operation.storage_start + hit if hit else None,
                requested_tokens=requested, hit_tokens=hit, revoked=revoked,
                anchor_hash=operation.last_hash,
                query_end_hash=chain[-1] if chain else None,
                checkpoint_hash=chain[hit_pages - 1] if hit_pages else None,
                pool_hit_pages=dict(operation.query_pool_hit_pages or {}),
            )
        pools = (operation.query_pool_hit_pages or {}) if requested >= 8192 else {}
        kv_pages = pools.get(PoolName.KV, 0)
        if requested >= 8192 and hit == 0:
            # Diagnostics: a full miss with KV pages present lacks a state or sidecar.
            stats["full_miss_kv_present" if kv_pages else "full_miss_kv_absent"] += 1
        if (
            requested >= 8192
            and kv_pages > 0
            and stats["lookups_logged"] < envs.SGLANG_HICACHE_L3_LOOKUP_LOG_N.get()
            and self.pp_rank == 0
        ):
            stats["lookups_logged"] += 1
            # Diagnostics: per-pool boundaries tell a missing page from a missing state.
            logger.info(
                "HiCache L3 lookup: rid=%s anchor=%s storage_start=%d requested=%d "
                "hit=%d pool_hit_pages=%s restorable=%s",
                operation.request_id,
                "root" if operation.last_hash is None else "node",
                operation.storage_start,
                requested,
                hit,
                {str(k): v for k, v in (operation.query_pool_hit_pages or {}).items()},
                (operation.pool_storage_result.restorable_prefix_pages or [])[-3:],
            )
        if self._l3_trace is not None and requested >= 8192:
            self._l3_trace_lookup(operation, kv_pages)
        if hit > 0 and not revoked and operation.stats_tree_kv_tokens:
            stats["hits_past_tree_kv"] += 1
            stats["hit_tokens_past_tree_kv"] += hit + operation.stats_tree_kv_tokens
        if revoked:
            if hit > 0:
                stats["revoked_insufficient"] += 1
            else:
                stats["revoked_full_miss"] += 1
        miss = requested - hit
        stats["l3_demand_requests"] += 1
        stats["l1l2_miss_tokens"] += requested
        stats["l3_miss_tokens"] += miss

    def _l3_trace_object_keys(self, page_hashes: list[str]):
        """Diagnostics: the tagged KDA-state object keys the backend forms for these page hashes."""
        backend = self.cache_controller.storage_backend
        backend = getattr(backend, "_backend", backend)
        try:
            keys, _ = backend._get_hybrid_page_component_keys(
                list(page_hashes), PoolTransfer(name=PoolName.MAMBA)
            )
            return backend._tag_keys(keys)
        except Exception as e:  # diagnostics must not fail a write
            return f"<{e!r}>"

    def _l3_trace_write(self, hash_value: list[str], aux_xfers: list[PoolTransfer]) -> None:
        """Diagnostics (PP0): remember when each page and state hash was issued to L3."""
        trace, now = self._l3_trace, time.monotonic()
        kv, stats = trace["kv"], self._l3_tier_stats
        for h in hash_value:
            then = kv.get(h)
            if then is not None:
                stats["trace_rewrite_pages_lt10m" if now - then < 600 else "trace_rewrite_pages_ge10m"] += 1
            kv[h] = now
        stats["trace_written_pages"] += len(hash_value)
        for xfer in aux_xfers:
            if xfer.name != PoolName.MAMBA or not xfer.keys:
                continue
            state_key = xfer.keys[-1]
            then = trace["state"].get(state_key)
            if then is not None:
                stats["trace_rewrite_states_lt10m" if now - then[0] < 600 else "trace_rewrite_states_ge10m"] += 1
            stats["trace_written_states"] += 1
            trace["state"][state_key] = (now, len(hash_value))
            if state_key != hash_value[-1]:
                stats["trace_state_key_not_last_page"] += 1
            if trace["writes_logged"] < 5:
                trace["writes_logged"] += 1
                logger.info(
                    "HiCache L3 trace write: pages=%d last_page=%s state_key=%s object_keys=%s",
                    len(hash_value), hash_value[-1], state_key,
                    self._l3_trace_object_keys([state_key]),
                )

    def _l3_trace_mamba_host_eviction(self, num_states: int) -> None:
        """Diagnostics (PP0): classify the coldest host states a Mamba host eviction is about
        to take (read-only peek, same LRU order as the eviction)."""
        stats, written = self._l3_tier_stats, self._l3_trace["state"]
        inflight = set(self._write_behind_mamba_inflight.values())
        leaves = self.tree_core.peek_mamba_host_eviction_leaves(num_states)
        for node_id, num_tokens, hashes in leaves:
            if hashes and hashes[-1] in written:
                stats["trace_mamba_evict_leaf_written"] += 1
            elif node_id in inflight:
                stats["trace_mamba_evict_leaf_inflight"] += 1
            else:
                stats["trace_mamba_evict_leaf_unwritten"] += 1
                stats["trace_mamba_evict_leaf_unwritten_tokens"] += num_tokens
        stats["trace_mamba_evict_tombstone"] += max(0, num_states - len(leaves))

    def _l3_trace_lookup(self, operation, kv_pages: int) -> None:
        """Diagnostics (PP0): for one lookup, how much of the queried span PP0 ever wrote and
        where (page, age) it wrote states, against the rank-agreed KV boundary."""
        trace, now = self._l3_trace, time.monotonic()
        chain = operation.all_hash_values or []
        kv, states, stats = trace["kv"], trace["state"], self._l3_tier_stats
        if not kv:  # before PP0's first write every lookup misses trivially
            return
        if kv_pages:
            ages = [now - states[h][0] for h in chain[:kv_pages] if h in states]
            stats["trace_present_no_state" if not ages else "trace_present_state_lt7m"
                  if min(ages) < 420 else "trace_present_state_older"] += 1
        else:
            then = kv.get(chain[0]) if chain else None
            age = None if then is None else now - then
            stats["trace_absent_never" if age is None else "trace_absent_lt7m" if age < 420
                  else "trace_absent_7to15m" if age < 900 else "trace_absent_ge15m"] += 1
            if age is None:
                fresh = [i for i, h in enumerate(chain) if h in states and now - states[h][0] < 420]
                if fresh:
                    stats["trace_absent_tail_in_l3"] += 1
                    stats["trace_absent_tail_in_l3_tokens"] += (fresh[-1] + 1) * self.page_size
        counter = "present_logged" if kv_pages else "absent_logged"
        if not kv_pages and trace[counter] >= envs.SGLANG_HICACHE_L3_TRACE_ABSENT_N.get():
            return
        trace[counter] += 1
        prefix = 0
        for h in chain:
            if h not in kv:
                break
            prefix += 1
        ever = sum(h in kv for h in chain)
        first_age = round(now - kv[chain[0]]) if chain and chain[0] in kv else None
        at = [
            (i + 1, round(now - states[h][0]), states[h][1])
            for i, h in enumerate(chain)
            if h in states
        ]
        logger.info(
            "HiCache L3 trace lookup: rid=%s storage_start=%d pages=%d kv_found=%d hit_pages=%d "
            "kv_written_prefix=%d kv_written_any=%d first_page_age_s=%s "
            "states_written(page,age_s,node_pages)=%s",
            operation.request_id, operation.storage_start, len(chain), kv_pages,
            operation.storage_hit_count // self.page_size, prefix, ever, first_age, at[-8:],
        )
        if kv_pages and trace["queried_logged"] < 5:
            trace["queried_logged"] += 1
            pages = [p for p, _, _ in at[-2:]] + [min(kv_pages, len(chain))]
            logger.info(
                "HiCache L3 trace queried: rid=%s pages=%s object_keys=%s",
                operation.request_id, pages,
                self._l3_trace_object_keys([chain[p - 1] for p in pages]),
            )

    def prefetch_outcome_stats_snapshot(self) -> dict:
        stats = self._prefetch_outcome_stats.copy()
        if self._l3_write_on_host_evict:
            stats.update(self._l3_tier_stats)
            stats["wb_inflight_tokens"] = sum(self._write_behind_inflight.values())
            stats["wb_mamba_inflight_ops"] = len(self._write_behind_mamba_inflight)
        return stats

    def _prefetch_occupied_span(
        self, prefetch_key, host_indices, *, operation=None
    ) -> int:
        """Occupancy units held by a prefetch: cache mode reserves the
        requested span at enqueue; buffer mode grants at hit-alloc, sized
        to the allocation (0 while still querying / parked)."""
        if self.host_memory_mode == "buffer_only":
            if (
                operation is not None
                and operation.buffer_host_occupied_units is not None
            ):
                return operation.buffer_host_occupied_units
            return len(host_indices) if host_indices is not None else 0
        return len(prefetch_key)

    def _storage_prefetch_min_hit_tokens(self, operation) -> int:
        """A positive L3 checkpoint remains mandatory even with tree KV credit."""
        if not self._l3_saved_token_threshold:
            return self.prefetch_threshold
        tree_tokens = max(0, operation.stats_tree_kv_tokens)
        return max(
            self.page_size,
            ceil_align(max(0, self.prefetch_threshold - tree_tokens), self.page_size),
        )

    def _storage_prefetch_meets_threshold(self, operation) -> bool:
        return (
            operation.storage_hit_count > 0
            and operation.storage_hit_count >= self._storage_prefetch_min_hit_tokens(operation)
        )

    def _alloc_prefetch_aux_staging(
        self, info: _OngoingPrefetch, hit_tokens: int
    ) -> bool:
        """Bind the aux staging (SWA window, mamba state) next to the KV staging;
        all pools or none, so a fetch never launches half-staged."""
        cc = self.cache_controller
        hit_pages = hit_tokens // self.page_size
        taken: list[PoolTransfer] = []
        for ct, transfers in info.comp_xfers.items():
            for transfer in transfers:
                if transfer.host_indices is not None or transfer.indices_from_pool:
                    continue
                entry = cc.mem_pool_host.entry_map.get(transfer.name)
                pool_page_size = (
                    entry.host_pool.page_size if entry is not None else self.page_size
                )
                num_tokens = min(len(transfer.keys or ()), hit_pages) * pool_page_size
                if num_tokens == 0:
                    continue
                host_indices = self.components[ct].alloc_prefetch_staging(num_tokens)
                if host_indices is None:
                    if (
                        self.enable_storage_metrics
                        and self.storage_metrics_collector is not None
                    ):
                        self.storage_metrics_collector.log_prefetch_aux_alloc_failed_tokens(
                            num_tokens
                        )
                    cc.append_host_mem_release(extra_pools=taken)
                    for t in taken:
                        t.host_indices = None
                    return False
                transfer.host_indices = host_indices
                taken.append(transfer)
        return True

    def _trim_buffer_prefetch_full_head(
        self,
        request: CacheRequestHandle,
        info: _OngoingPrefetch,
        operation,
        full_match_len: int,
        hit_tokens: int,
    ) -> tuple[_OngoingPrefetch, int, int]:
        """Drop FULL pages the hit-time rematch found on device; trailing aux
        transfers keep the query's endpoint so SWA still fetches its window."""
        # A capacity-deferred hit comes back already trimmed; size its sidecars
        # from the pre-trim boundary or the SWA staging comes up short.
        original_hit_tokens = (
            operation.sidecar_hit_pages * self.page_size
            if operation.sidecar_hash_values is not None
            else hit_tokens
        )
        trim_tokens = min(hit_tokens, max(0, full_match_len - operation.storage_start))
        if trim_tokens == 0:
            return info, hit_tokens, original_hit_tokens
        assert trim_tokens % self.page_size == 0

        old_key = info.prefetch_key
        prefix_ctx = self.buffer_pipeline._prefetch_prefix_ctx[request]
        prefix_tokens, extra_key, cache_salt = prefix_ctx
        # Prefix context excludes the boundary token; the suffix owns it even
        # when all FULL pages are trimmed and only sidecars remain to fetch.
        new_prefix_tokens = prefix_tokens + list(old_key.token_ids[:trim_tokens])
        self.buffer_pipeline._prefetch_prefix_ctx[request] = (
            new_prefix_tokens,
            extra_key,
            cache_salt,
        )
        info = info._replace(
            prefetch_key=RadixKey(
                old_key.raw_token_ids()[trim_tokens:],
                extra_key=old_key.extra_key,
                is_bigram=old_key.is_bigram,
                cache_salt=old_key.cache_salt,
            )
        )
        self.ongoing_prefetch[request] = info
        self.cache_controller.trim_prefetch_full_head(operation, trim_tokens)
        # Labeled at admission: FULL reusable only with the fetched aux tail is a
        # storage hit there, not device_covered.
        return info, hit_tokens - trim_tokens, original_hit_tokens

    def revoke_pending_prefetch(self, request: CacheRequestHandle) -> None:
        self._finish_storage_prefetch(request, fulfilled_tokens=0, reason="dropped")
        info = self.ongoing_prefetch.get(request)
        if info is None:
            return
        # Every revoke path runs before the bounce alloc, so buffer mode
        # holds no occupancy here; post-alloc aborts go through
        # release_aborted_request instead.
        assert info.host_indices is None or self.host_memory_mode != "buffer_only"
        self._retire_ongoing_prefetch(
            request, None, [x for xfers in info.comp_xfers.values() for x in xfers]
        )

    def _drain_storage_control_queues_impl(
        self,
        n_storage_hit: Optional[int],
        n_ack_prefetch: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        extra_release_counts: Optional[dict[PoolName, int]],
        log_metrics: bool,
    ) -> None:
        cc = self.cache_controller
        write_back = self._storage_write_back
        acks_taken_at_entry = 0 if write_back is None else write_back.acks_taken

        def _drain_queue(
            q: Queue[T], n: Optional[int], timed: bool = False
        ) -> Iterator[T]:
            """If n is None, consume all items from the queue.
            Otherwise, consume n items from the queue.  Blocking if there are no enough n items.

            In TP, each rank consumes the a minimal number of items of all ranks.
            In PP, each rank consumes the exact number of items of PP0.  Refer to _pp_sync for more details.

            This prevents TP/PP divergence.
            """
            if n is None:
                while not q.empty():
                    item = q.get()
                    yield item
            else:
                for _ in range(n):
                    # Block when there are not enough elements.
                    # All TP/PP ranks must consume the same number of elements.
                    if timed:
                        blocked = q.empty()
                        t0 = time.perf_counter()
                        item = q.get()
                        pmicro_hoststats.note_ack_get(time.perf_counter() - t0, blocked)
                    else:
                        item = q.get()
                    yield item

        buffer_mode = self.host_memory_mode == "buffer_only"

        def _try_alloc_storage_hit(operation) -> bool:
            """Allocate the hit-sized bounce and launch the transfer.
            Returns False when staging pressure defers the allocation
            (buffer mode parks and retries; cache mode revokes)."""
            request = operation.handle
            info = self.ongoing_prefetch.get(request)
            hit_tokens = operation.storage_hit_count
            if info is None:
                return True  # aborted/cleaned; nothing to retry
            if operation.is_terminated():
                self.revoke_pending_prefetch(request)
                return True

            if buffer_mode and cc.prefetch_rate_limited():
                # Pool is load-saturated: hold the KNOWN hit until staged
                # prefetches ahead of us are consumed. The op stays in
                # ongoing_prefetch, so wait_complete keeps gating admission.
                return False
            if buffer_mode:
                # IO commit pins the splice base; a parked op held no pin, so a
                # moved or vanished anchor is detected (and recovered) here.
                anchor, full_match_len = self.buffer_pipeline.try_lock_anchor(
                    request, operation.storage_hit_count
                )
                if anchor == "anchor_lost":
                    self._prefetch_outcome_stats["declined_anchor_lost"] += 1
                    self._handle_storage_prefetch_anchor_loss(request)
                    return True
                if anchor == "cap_busy":
                    # Other prefetches hold the budget; ours fits once they
                    # drain, so park rather than launch this one unprotected.
                    return False
                if anchor == "unservable":
                    # The device prefix is resident but unreadable; the
                    # request recomputes and heals it instead of splicing.
                    self._prefetch_outcome_stats["declined_unservable_prefix"] += 1
                    logger.info(
                        "HiCache prefetch declined req=%s reason=unservable_prefix "
                        "device_prefix=%d hit=%d",
                        request.rid,
                        full_match_len,
                        hit_tokens,
                    )
                    self._finish_storage_prefetch(
                        request, fulfilled_tokens=0, reason="unservable_prefix"
                    )
                    self.revoke_pending_prefetch(request)
                    return True
                if self.buffer_pipeline.staged_span_covered(
                    request, operation.storage_hit_count
                ):
                    # Live tree already covers the span: nothing left to
                    # splice, so skip the storage read.
                    self._prefetch_outcome_stats["declined_device_covered"] += 1
                    self._finish_storage_prefetch(
                        request, fulfilled_tokens=0, reason="device_covered"
                    )
                    self.revoke_pending_prefetch(request)
                    return True
                info, hit_tokens, aux_hit_tokens = self._trim_buffer_prefetch_full_head(
                    request, info, operation, full_match_len, hit_tokens
                )
                return self.buffer_pipeline.allocate_and_queue_prefetch(
                    operation, info, hit_tokens, aux_hit_tokens
                )
            aux_hit_tokens = hit_tokens
            host_indices, alloc_len = cc.allocate_storage_hit(
                operation,
                hit_tokens,
                # A confirmed Mamba checkpoint belongs to this exact endpoint.
                # A shorter KV allocation does not prove a state at its new end.
                allow_partial=not (
                    self._l3_saved_token_threshold
                    and ComponentType.MAMBA in info.comp_xfers
                ),
                min_tokens=self._storage_prefetch_min_hit_tokens(operation),
                evict_host=self.evict_host,
            )
            if host_indices is None:
                self._finish_storage_prefetch(
                    request, fulfilled_tokens=0, reason="host_capacity"
                )
                self.revoke_pending_prefetch(request)
                self.storage_prefetch_retries.poll_miss(
                    request.rid, operation.storage_start + operation.storage_hit_count
                )
                self._log_storage_prefetch_deferred(
                    max(alloc_len, aux_hit_tokens), "host_capacity"
                )
                return True
            if not self._alloc_prefetch_aux_staging(info, aux_hit_tokens):
                # Same outcome as a KV shortfall: nothing stays staged.
                cc.append_host_mem_release(host_indices=host_indices)
                self._finish_storage_prefetch(
                    request, fulfilled_tokens=0, reason="host_capacity"
                )
                self.revoke_pending_prefetch(request)
                self.storage_prefetch_retries.poll_miss(
                    request.rid, operation.storage_start + operation.storage_hit_count
                )
                self._log_storage_prefetch_deferred(
                    max(alloc_len, aux_hit_tokens), "host_capacity"
                )
                return True

            self._resolve_storage_prefetch_tokens(
                request, hit_tokens - alloc_len, reason="host_capacity"
            )
            operation.storage_hit_count = alloc_len
            operation.hash_value = operation.hash_value[: alloc_len // self.page_size]
            operation.host_indices = host_indices
            self.ongoing_prefetch[request] = info._replace(host_indices=host_indices)
            cc.prefetch_buffer.put(operation)
            return True

        def _drain_and_alloc_storage_hit():
            # Parked hits first (FIFO fairness with retries; buffer only).
            if buffer_mode:
                parked = self.buffer_pipeline.pending_hit_allocs
                while parked:
                    if not _try_alloc_storage_hit(parked[0]):
                        break
                    parked.popleft()
            for operation in _drain_queue(cc.prefetch_hit_queue, n_storage_hit):
                if isinstance(operation, MambaStateRestoreOperation):
                    self._finish_mamba_state_restore(operation)
                    continue
                self._invalidate_absent_from_hit_query(operation)
                request = operation.handle
                hit_tokens = operation.storage_hit_count
                info = self.ongoing_prefetch.get(request)
                if info is None:
                    if hit_tokens > 0:
                        self.discard_storage_prefetch_accounting(request)
                    continue
                if hit_tokens > 0:
                    self._record_storage_prefetch_hit(request, hit_tokens)
                if operation.is_terminated():
                    # A storage miss can retry; a policy deadline cannot.
                    if hit_tokens > 0:
                        reason = None
                        if operation.deadline_expired:
                            reason = "deadline"
                        elif not self._storage_prefetch_meets_threshold(operation):
                            reason = "below_threshold"
                        self._finish_storage_prefetch(
                            request, fulfilled_tokens=0, reason=reason
                        )
                    # The published (rank-agreed) hit count is the L3 verdict;
                    # without this, store misses never reach l3_miss_tokens.
                    self._account_prefetch_outcome(operation, revoked=True)
                    if not operation.deadline_expired:
                        self.storage_prefetch_retries.poll_miss(request.rid)
                    self.revoke_pending_prefetch(request)
                    continue
                if not self._storage_prefetch_meets_threshold(operation):
                    # Below-threshold hits are not worth the transfer.
                    self._account_prefetch_outcome(operation, revoked=True)
                    self._finish_storage_prefetch(
                        request, fulfilled_tokens=0, reason="below_threshold"
                    )
                    self.storage_prefetch_retries.poll_miss(request.rid)
                    self.revoke_pending_prefetch(request)
                    continue
                self._account_prefetch_outcome(operation, revoked=False)
                # A parked hit keeps its turn: a newer hit must not take the
                # staging or anchor budget the parked head is waiting for.
                parked_ahead = buffer_mode and bool(
                    self.buffer_pipeline.pending_hit_allocs
                )
                if parked_ahead or not _try_alloc_storage_hit(operation):
                    # Counted once at first parking, not per retry tick.
                    self._prefetch_outcome_stats["declined_rate_limited"] += 1
                    self.buffer_pipeline.pending_hit_allocs.append(operation)

        def _drain_ack_prefetch():
            for ack in _drain_queue(cc.ack_prefetch_queue, n_ack_prefetch):
                operation = ack.operation
                info = self.ongoing_prefetch.get(operation.handle)
                is_current = info is not None and info.operation is operation
                if ack.completed_tokens is not None:
                    if is_current:
                        assert operation.completed_tokens <= ack.completed_tokens
                        operation.completed_tokens = ack.completed_tokens
                if ack.pool_hits is not None:
                    if is_current:
                        operation.pool_storage_result.update_extra_pool_hit_pages(
                            ack.pool_hits
                        )
                        operation.pool_transfers_done = True
                if ack.completed_req:
                    if is_current:
                        # check_prefetch_progress() is not called for this rid yet.
                        # Let us insert the prefetch result into the radix tree.
                        self._handle_prefetch_result(operation)
                    if operation.ack_releases_incomplete_host_indices:
                        cc.append_host_mem_release(
                            operation.host_indices[operation.completed_tokens :],
                            (
                                operation.pool_transfers
                                if not operation.pool_transfers_done
                                else None
                            ),
                        )

        def _drain_backup():
            drained = 0
            n = n_backup
            if write_back is not None and n:
                # A write-back wait inside this drain took acks off the queue
                # front; they count against the n agreed before it.
                n = max(0, n - (write_back.acks_taken - acks_taken_at_entry))
            write_back_acks = []
            for operation in _drain_queue(
                cc.ack_backup_queue, n, timed=pmicro_hoststats.ON
            ):
                if (
                    operation.id in self._l3_tombstone_quarantined_ops
                    or getattr(operation, "l3_tombstone_quarantined", False)
                ):
                    raise RuntimeError(
                        "Refusing to release a quarantined Mamba storage source"
                    )
                drained += 1
                tombstone = (
                    self._l3_tombstone_pending.pop(operation.id, None)
                    if self._l3_tombstone_pending
                    else None
                )
                if tombstone is not None:
                    # Every rank takes this ACK at the same step: unpin and free now.
                    self._check_mamba_tombstone_ack(operation, operation.id)
                    node_id, lock_params = self.ongoing_backup.pop(operation.id)
                    self.dec_host_lock_ref(node_id, lock_params)
                    # A hit since the write may have revived it: then it stays.
                    self._finish_mamba_tombstone(tombstone, revived_ok=True)
                    continue
                if buffer_mode:
                    # Storage write acked: free the staging.
                    self.buffer_pipeline.finish_storage_write_ack(operation.id)
                else:
                    entry = self.ongoing_backup.pop(operation.id, None)
                    if entry is not None:
                        node_id, lock_params = entry
                        self.dec_host_lock_ref(node_id, lock_params)
                    if write_back is not None:
                        write_back_acks.append(operation)
                    if (
                        self._l3_tier_final is not None
                        and (
                            operation.id in self._write_behind_inflight
                            or operation.id in self._write_behind_mamba_inflight
                        )
                    ):
                        self._l3_tier_final.written_tokens += operation.completed_tokens
                    acked_tokens = self._write_behind_inflight.pop(operation.id, None)
                    if acked_tokens and self._l3_wb_ack_credit:
                        # Rank-replicated (issued) count, not completed_tokens.
                        self._write_behind_clean_tokens += acked_tokens
                    self._write_behind_mamba_inflight.pop(operation.id, None)
                    # Added unconditionally: completed_tokens can diverge across ranks on
                    # backend failure and a divergent belief desyncs the rank collectives.
                    if operation.hash_value:
                        self.storage_existence_cache.pool(PoolName.KV).add(
                            operation.hash_value
                        )
                    for transfer in operation.pool_transfers or ():
                        if transfer.keys:
                            self.storage_existence_cache.pool(transfer.name).add(
                                transfer.keys
                            )
                if (
                    log_metrics
                    and self.enable_storage_metrics
                    and self.storage_metrics_collector is not None
                ):
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )
            if self._l3_ack_lag_steps:
                self._l3_ack_removed += drained
            if write_back_acks:
                # n is None only on the unsynchronized detach/shutdown drain.
                write_back.on_backup_acks(write_back_acks, synced=n is not None)
            return drained

        def _drain_release():
            host_indices_list = []
            released_tokens = 0
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
                released_tokens += len(host_indices)
            if host_indices_list:
                cc.mem_pool_host.free(torch.cat(host_indices_list, dim=0))
            return len(host_indices_list), released_tokens

        def _drain_extra_release():
            drained: dict[PoolName, tuple[int, int]] = {}
            if not extra_release_counts:
                return drained
            for pool_name, limit in extra_release_counts.items():
                release_queue = cc.extra_host_mem_release_queues.get(pool_name)
                if release_queue is None:
                    continue
                host_indices_list = []
                released_tokens = 0
                for host_indices in _drain_queue(release_queue, limit):
                    host_indices_list.append(host_indices)
                    released_tokens += len(host_indices)
                if host_indices_list:
                    cc.mem_pool_host.free(
                        torch.cat(host_indices_list, dim=0), pool=pool_name
                    )
                drained[pool_name] = (len(host_indices_list), released_tokens)
            return drained

        _drain_and_alloc_storage_hit()
        _drain_ack_prefetch()
        _drain_backup()
        _drain_release()
        _drain_extra_release()

    def drain_storage_control_queues(self) -> None:
        cc = self.cache_controller
        extra_release_queues = getattr(cc, "extra_host_mem_release_queues", {})
        extra_pool_names = list(extra_release_queues)
        local_qsize_list = [
            cc.prefetch_hit_queue.qsize(),
            cc.ack_prefetch_queue.qsize(),
            cc.ack_backup_queue.qsize(),
            cc.host_mem_release_queue.qsize(),
            *[
                extra_release_queues[pool_name].qsize()
                for pool_name in extra_pool_names
            ],
        ]
        qsizes = torch.tensor(
            local_qsize_list,
            dtype=torch.int,
        )
        self._all_reduce(qsizes, torch.distributed.ReduceOp.MIN)
        qsize_list = list(map(int, qsizes.tolist()))
        n_storage_hit, n_ack_prefetch, n_backup, n_release = qsize_list[:4]
        extra_release_counts = {
            pool_name: count
            for pool_name, count in zip(extra_pool_names, qsize_list[4:])
        }
        self._drain_storage_control_queues_impl(
            n_storage_hit=n_storage_hit,
            n_ack_prefetch=n_ack_prefetch,
            n_backup=n_backup,
            n_release=n_release,
            extra_release_counts=extra_release_counts,
            log_metrics=True,
        )

    def drain_storage_control_queues_local(self) -> None:
        """Drain the storage control queues without cross-rank synchronization.

        For the detach / shutdown path, where best-effort cleanup matters more than
        keeping the drained counts identical across ranks. The prefetch-hit queue is
        deliberately skipped: servicing it would allocate host pages for a prefetch
        that can no longer complete.
        """
        cc = self.cache_controller
        # The storage queues are created by the controller when the storage threads
        # start, so they are still None when a backend was never attached.
        if cc is None or cc.prefetch_hit_queue is None:
            return
        self._drain_storage_control_queues_impl(
            n_storage_hit=0,
            n_ack_prefetch=0,
            n_backup=None,
            n_release=None,
            extra_release_counts={
                name: None for name in cc.extra_host_mem_release_queues
            },
            log_metrics=False,
        )
        self._reset_ack_lag()

    # ---- HiCache: Storage backend lifecycle (delegated) ----

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Attach (enable) the HiCache storage backend at runtime."""
        if self._storage_attachment is None:
            return (
                False,
                "HiCache is not initialized; launch with "
                "--enable-hierarchical-cache to attach a storage backend.",
            )
        return self._storage_attachment.attach(
            storage_backend=storage_backend,
            storage_backend_extra_config_json=storage_backend_extra_config_json,
            served_model_name=served_model_name,
            hicache_storage_prefetch_policy=hicache_storage_prefetch_policy,
            hicache_write_policy=hicache_write_policy,
        )

    def detach_storage_backend(self) -> tuple[bool, str]:
        """Detach (disable) the HiCache storage backend at runtime."""
        if self._storage_attachment is None:
            return False, "HiCache storage backend is not initialized."
        return self._storage_attachment.detach()

    def shutdown(self) -> None:
        """Best-effort auto-detach of the storage backend on process shutdown."""
        if self._storage_write_back is not None:
            try:
                self._storage_write_back.finish(reason="exit")
            except Exception:
                logger.exception("HiCache L3 write-back: final stats failed.")
        if self._l3_tier_final is not None:
            try:
                self._l3_tier_final.finish(reason="exit")
            except Exception:
                logger.exception("HiCache L3 write-on-host-evict: final stats failed.")
        if self._storage_attachment is not None:
            self._storage_attachment.shutdown()

    def clear_storage_backend(self) -> bool:
        if self._storage_attachment is None:
            return False
        ok = self._storage_attachment.clear()
        if ok:
            # L3 is empty now: every storage-presence belief is stale, and a
            # retained positive would skip that page's backup forever.
            self.storage_existence_cache.clear()
        return ok

    # ---- HiCache: Async Event Management ----

    def _init_ack_lag(self) -> None:
        # Only PP0 of a cache-mode pipeline: its counts are the ones relayed down.
        ack_lag = max(0, envs.SGLANG_HICACHE_L3_ACK_LAG_STEPS.get())
        if (
            ack_lag
            and self.pp_size > 1
            and self.pp_rank == 0
            and self.host_memory_mode != "buffer_only"
        ):
            self._l3_ack_lag_steps = ack_lag
            self._l3_ack_lag_hist = deque(maxlen=ack_lag + 1)
            logger.info(
                "HiCache L3: PP0 publishes storage write acks %d steps after they "
                "are ready (SGLANG_HICACHE_L3_ACK_LAG_STEPS).",
                ack_lag,
            )

    def _reset_ack_lag(self) -> None:
        # The queue was emptied or replaced: restart the cumulative counts at it.
        if self._l3_ack_lag_hist is not None:
            self._l3_ack_lag_hist.clear()
            self._l3_ack_removed = 0

    def _lag_backup_ack_count(self, ready: int) -> int:
        """SGLANG_HICACHE_L3_ACK_LAG_STEPS: the ack_backup_queue count to publish.

        The queue is FIFO and every removal is counted, so ready + removed is the
        cumulative number of acks that ever became ready here. Publishing that
        cumulative from D steps ago minus removed releases only acks ready for at
        least D steps; later PP stages issued the same puts at the same step and
        drain this same relayed count, so their q.get() rarely blocks.
        """
        hist = self._l3_ack_lag_hist
        hist.append(ready + self._l3_ack_removed)
        if len(hist) < hist.maxlen:
            return 0
        return max(0, min(hist[0] - self._l3_ack_removed, ready))

    def _count_ready_acks(self, ack_queue) -> int:
        ready_count = 0
        for ack in ack_queue:
            if not ack.finish_event.query():
                break
            ready_count += 1
        return ready_count

    def _sync_hicache_ready_counts(
        self,
    ) -> tuple[int, int, tuple[int, ...], tuple[PoolName, ...]]:
        cc = self.cache_controller
        extra_release_queues = getattr(cc, "extra_host_mem_release_queues", {})
        extra_pool_names = tuple(extra_release_queues) if self.enable_storage else ()
        if cc is None or (self.pp_rank > 0 and self.host_memory_mode != "buffer_only"):
            write_acks = 0
            load_acks = 0
            wa_acks = 0
            # Zero placeholders shaped like PP0's slots: _pp_sync hands the
            # received tensor back in place, so all ranks must build the same
            # length or PP1+ would recv into a mismatched buffer.
            storage_queue_sizes = (
                (0,) * (4 + len(extra_pool_names)) if self.enable_storage else ()
            )
        else:
            write_acks = self._count_ready_acks(cc.ack_write_queue)
            load_acks = self._count_ready_acks(cc.ack_load_queue)
            wa_acks = (
                self._write_ahead_publish_count() if self._device_write_ahead else 0
            )
            storage_queue_sizes = (
                (
                    cc.prefetch_hit_queue.qsize(),
                    cc.ack_prefetch_queue.qsize(),
                    cc.ack_backup_queue.qsize(),
                    cc.host_mem_release_queue.qsize(),
                    *(extra_release_queues[name].qsize() for name in extra_pool_names),
                )
                if self.enable_storage
                else ()
            )
            if self._l3_ack_lag_steps and storage_queue_sizes:
                lagged = self._lag_backup_ack_count(storage_queue_sizes[2])
                storage_queue_sizes = (
                    storage_queue_sizes[:2] + (lagged,) + storage_queue_sizes[3:]
                )

        # Piggybacked TP check: [digest, -digest] MIN-reduces to [min, -max],
        # equal iff reclaim victim order matched on every rank.
        digest = self.tree_core.write_back_duplicate_reclaim_digest
        # The write-ahead slot exists only when the flag is on (on every rank).
        head = [write_acks, load_acks] + ([wa_acks] if self._device_write_ahead else [])
        n_head = len(head)
        exchange = self._shm_count_exchange()
        if exchange is not None:
            values = [*head, *storage_queue_sizes, *fold_u64(digest)]
            # Storage adds a count per extra host pool; a row wider than the table
            # takes the gloo reduce below.
            if len(values) <= exchange.width:
                rows = exchange.allgather(values)
                assert all(row[-2:] == rows[0][-2:] for row in rows), (
                    "write_back duplicate-reclaim victims diverged across TP ranks"
                )
                mins = [min(col) for col in zip(*rows)]
                self._wa_ready_count = mins[2] if self._device_write_ahead else 0
                return mins[0], mins[1], tuple(mins[n_head:-2]), extra_pool_names
        ready_counts = torch.tensor(
            [
                *head,
                *storage_queue_sizes,
                digest,
                -digest,
            ],
            dtype=torch.int64,
            device="cpu",
        )
        if self.host_memory_mode == "buffer_only" and self.pp_size > 1:
            self._all_reduce_attn_groups(ready_counts, torch.distributed.ReduceOp.MIN)
        else:
            self._all_reduce(ready_counts, torch.distributed.ReduceOp.MIN)

        count_values = list(map(int, ready_counts.tolist()))
        assert digest == count_values[-2] and digest == -count_values[-1], (
            "write_back duplicate-reclaim victims diverged across PP/TP ranks"
        )
        self._wa_ready_count = count_values[2] if self._device_write_ahead else 0
        return (
            count_values[0],
            count_values[1],
            tuple(count_values[n_head:-2]),
            extra_pool_names,
        )

    def _shm_count_exchange(self) -> Optional[ShmTagExchange]:
        # Only the single-group, single-stage layout; anything else keeps the gloo reduce.
        if self.pp_size > 1 or self.cache_controller is None:
            return None
        groups = [
            g
            for g in (self.attn_cp_group, self.attn_tp_group)
            if g is not None and torch.distributed.get_world_size(group=g) > 1
        ]
        if len(groups) > 1:
            return None
        group = (
            groups[0] if groups else (self.tp_group if self.tp_world_size > 1 else None)
        )
        # Across hosts the leaders' all_gather beats an all-rank gloo reduce (measured at TP8);
        # the request broadcast stays on gloo there, a one-source broadcast does not gain.
        return get_shm_exchange(group, "hicache", multihost=True)

    def _is_pd_prefill_write_back(self) -> bool:
        cc = self.cache_controller
        return (
            cc is not None
            and cc.write_policy == "write_back"
            and self._wb_evict_pd_prefill
        )

    def _wb_evict_relaxed_fence(self) -> bool:
        """Whether a Full-leaf write-back eviction may fence its D2H on the
        in-flight forward's start event instead of on the whole forward.

        On a PD prefill server the in-flight forward writes only its own
        requests' pages and state slots; the only tree nodes among them belong
        to a chunked request's checkpoint, which holds the Full lock on its path.
        A Full-leaf victim has no Full lock, and its backup chain (the leaf, its
        unbacked ancestors and their SWA window) cannot descend from that
        checkpoint, so all its KV and state were written by earlier forwards.
        Internal SWA/Mamba state backups keep the full fence: an older SWA
        window of the in-flight checkpoint is evictable.
        """
        return (
            self._is_pd_prefill_write_back()
            and envs.SGLANG_HICACHE_WB_EVICT_RELAXED_FENCE.get()
            and self.cache_controller.forward_start_event is not None
        )

    def _backup_full_leaf_for_write_back(self, backup_kv: BackupKV) -> int:
        """Back up a Full-leaf victim's chain and wait for the copy."""
        stream = self._wb_evict_overlap_stream(backup_kv)
        with (
            contextlib.nullcontext() if stream is None else device_module.stream(stream)
        ):
            written = self._execute_and_commit_kv_backup(backup_kv, write_back=True)
            if written > 0:
                self.writing_check(write_back=True, relaxed_fence=True)
        return written

    def _backup_full_leaf_batched(self, backup_kv: BackupKV) -> int:
        """Queue a Full-leaf victim's chain backup without waiting for the copy;
        the walk waits once in _wb_batch_flush before it returns.

        The caller demotes the victim right away, so the walk sees the same
        victims and frees as the per-victim path. Its device slots go back to
        the allocator before the copy lands, which is safe because the walk
        writes no device memory and nothing allocates before the flush. The
        chain stays host-locked until the flush: host eviction for a later
        victim cannot reclaim these host slots and hand them to a second copy
        in the same merged transfer. A chain holding host-only aux state is not
        locked (the lock would reorder that state's host LRU); its backup waits
        at once, as on the per-victim path, and lands the queued copies too.
        """
        stream = self._wb_evict_overlap_stream(backup_kv)
        if self._wb_batch_locks and stream is not self._wb_batch_stream:
            self._wb_batch_flush()
        while True:
            with (
                contextlib.nullcontext()
                if stream is None
                else device_module.stream(stream)
            ):
                written = self._execute_and_commit_kv_backup(
                    backup_kv, write_back=True
                )
            if written > 0 or not self._wb_batch_locks:
                break
            # Queued victims may hold the host space: land them, retry once.
            if not pmicro_hoststats.ON:
                self._wb_batch_flush()
                continue
            l3_blocked = self._pmicro_l3_pending() > 0
            t0 = time.perf_counter()
            self._wb_batch_flush()
            pmicro_hoststats.note_host_wait(time.perf_counter() - t0, l3_blocked)
        if written > 0:
            self._wb_batch_stream = stream
            self._wb_batch_stats[1] += 1
            # Every node whose host slots the queued copy writes: the chain and
            # the nodes its component (Mamba) transfers cover.
            lock_ids = list(backup_kv.node_ids)
            for chain_node_id in backup_kv.node_ids:
                pending = self.ongoing_write_through.get(chain_node_id)
                if pending is not None:
                    lock_ids.extend(pending.publish_node_ids)
            lock_ids = list(dict.fromkeys(lock_ids))
            if any(map(self._has_host_only_aux_state, lock_ids)):
                # A host lock takes host-only aux state out of its host LRU and
                # the release puts it back as most recent, so a stale ancestor
                # state would outlive its descendants; wait now, unlocked.
                self._wb_batch_stats[2] += 1
                self._wb_batch_flush(force=True)
                return written
            for lock_id in lock_ids:
                params = self.inc_host_lock_ref(lock_id).to_dec_params()
                self._wb_batch_locks.append((lock_id, params))
        return written

    def _has_host_only_aux_state(self, node_id: NodeId) -> bool:
        return any(
            ct != BASE_COMPONENT_TYPE
            and self.tree_core.component_has_host_value_only(node_id, ct)
            for ct in self.tree_components
        )

    def _wb_batch_flush(self, force: bool = False) -> None:
        """Submit the walk's queued backups as one transfer, wait for it, then
        release the victims' host locks."""
        locks = self._wb_batch_locks
        if not locks and not force:
            return
        stream = self._wb_batch_stream
        self._wb_batch_locks = []
        self._wb_batch_stream = None
        try:
            with (
                contextlib.nullcontext()
                if stream is None
                else device_module.stream(stream)
            ):
                self.writing_check(write_back=True, relaxed_fence=True)
        finally:
            for node_id, params in locks:
                self.dec_host_lock_ref(node_id, params)
        stats = self._wb_batch_stats
        stats[0] += 1
        if stats[0] % 1024 == 0:
            logger.info(
                "HiCache write_back eviction batch: %d waits, %.1f victims per wait, "
                "%d victims waited unlocked (host-only aux state).",
                stats[0],
                stats[1] / stats[0],
                stats[2],
            )

    def _wb_overlap_sync_watermark(self) -> None:
        # Node ids only grow: at the first look after a WAR barrier, every id
        # seen so far belongs to a node created before that barrier.
        seq = self.cache_controller.war_barrier_seq
        if seq != self._wb_overlap_barrier_seq:
            self._wb_overlap_barrier_seq = seq
            self._wb_overlap_watermark = self._wb_overlap_max_seen_id

    def _wb_overlap_observe(self, node_id: Optional[NodeId]) -> None:
        if not isinstance(node_id, int) or not self._is_pd_prefill_write_back():
            return
        self._wb_overlap_sync_watermark()
        self._wb_overlap_max_seen_id = max(self._wb_overlap_max_seen_id, node_id)

    def _wb_overlap_chain_is_fresh(self, node_ids: Sequence[NodeId]) -> bool:
        """Whether a chain node may hold a device value written after the latest
        WAR barrier: a node created since (insert, split head) has an id above
        the watermark, and a split tail gets a new parent. Nodes not seen before
        the barrier count as fresh."""
        node_ids = list(node_ids)
        if not node_ids:
            return True
        self._wb_overlap_sync_watermark()
        watermark = self._wb_overlap_watermark
        paths = self.tree_core.collect_path_info(node_ids)
        for node_id, path in zip(node_ids, paths):
            if not path or path[-1].node_id != node_id:
                return True
            if node_id > watermark or path[-1].parent_id > watermark:
                return True
        return False

    def _wb_evict_overlap_stream(self, backup_kv: BackupKV) -> Optional[torch.Stream]:
        """The private stream a relaxed Full-leaf backup runs on after a WAR
        barrier, or None to stay on the current stream.

        After the barrier the schedule stream waits on the in-flight forward, and
        an L2 transfer started there (it records its start event on the current
        stream) would wait for that forward too. The private stream waits only on
        ``forward_start_event`` and ``war_barrier_event``, then builds the
        transfer indices, so every device tensor the backup reads must have been
        produced before those events. Inserts and splits after the barrier copy
        node values on the schedule stream, behind the forward, so a chain with
        such a node (_wb_overlap_chain_is_fresh) stays on the current stream.
        SWA trees and page-relocating allocators are out of scope.
        """
        if not self._wb_evict_relaxed_fence():
            return None
        cc = self.cache_controller
        if not cc.war_barrier_after_forward or cc.war_barrier_event is None:
            return None
        counts = self._wb_overlap_counts
        if not self._wb_overlap_in_scope:
            counts["scope"] += 1
            return None
        if self._wb_overlap_chain_is_fresh(backup_kv.node_ids):
            counts["fresh"] += 1
            stream = None
        else:
            counts["overlap"] += 1
            stream = self._wb_overlap_get_stream(cc)
            stream.wait_event(cc.forward_start_event)
            stream.wait_event(cc.war_barrier_event)
        if sum(counts.values()) % 1024 == 0:
            logger.info(
                "HiCache write_back eviction overlap: %d overlapped, "
                "%d fenced (fresh chain), %d out of scope.",
                counts["overlap"],
                counts["fresh"],
                counts["scope"],
            )
        return stream

    def _wb_overlap_get_stream(self, cc: HybridCacheController) -> torch.Stream:
        if self._wb_overlap_stream is not None:
            return self._wb_overlap_stream
        engine = cc.l2_transfer_engine
        redrawn = []
        if is_cuda() or is_hip():
            from sglang.srt.managers.utils import allocate_distinct_stream

            # Pooled streams can alias: an L2 stream that is the schedule or
            # forward stream would queue the copy behind the forward.
            scheduler_streams = [device_module.current_stream()]
            if cc.write_fence_stream is not None:
                scheduler_streams.append(cc.write_fence_stream)
            scheduler_handles = {s.cuda_stream for s in scheduler_streams}

            def avoid(*extra):
                l2 = [engine.device_to_host_stream, engine.index_stream]
                return scheduler_streams + l2 + list(extra)

            if engine.device_to_host_stream.cuda_stream in scheduler_handles:
                engine.device_to_host_stream = allocate_distinct_stream(
                    device_module, avoid()
                )
                redrawn.append("device_to_host_stream")
            if engine.index_stream.cuda_stream in scheduler_handles:
                engine.index_stream = allocate_distinct_stream(device_module, avoid())
                redrawn.append("index_stream")
            stream = allocate_distinct_stream(
                device_module, avoid(engine.host_to_device_stream)
            )
        else:
            stream = device_module.Stream()
        self._wb_overlap_stream = stream
        logger.info(
            "HiCache write_back eviction: Full-leaf backups after a WAR barrier "
            "use a private stream (redrawn L2 streams: %s).",
            ", ".join(redrawn) or "none",
        )
        return stream

    def _flush_queued_writes_fenced(self) -> None:
        # Writes queued before eviction (insert-time backups) keep the full
        # fence; the relaxed submit must not carry them.
        cc = self.cache_controller
        if cc is not None and cc.write_queue:
            cc.start_writing()

    def writing_check(
        self,
        write_back: bool = False,
        finish_count: Optional[int] = None,
        relaxed_fence: bool = False,
    ) -> None:
        """Poll write-through completions. Only Full-leaf eviction passes
        ``relaxed_fence`` (see _wb_evict_relaxed_fence)."""
        cc = self.cache_controller
        if cc is None:
            return

        if write_back:
            # Blocking: submit what is still queued, then wait for every ack.
            fence_event = None
            if relaxed_fence and self._wb_evict_relaxed_fence():
                fence_event = cc.forward_start_event
                if not self._wb_evict_relaxed_fence_logged:
                    self._wb_evict_relaxed_fence_logged = True
                    logger.info(
                        "HiCache write_back eviction: relaxed D2H fence for "
                        "Full-leaf victims (SGLANG_HICACHE_WB_EVICT_RELAXED_FENCE)."
                    )
            cc.start_writing(fence_event=fence_event)
            t0 = (
                time.perf_counter()
                if pmicro_hoststats.ON and self.ongoing_write_through
                else None
            )
            while self.ongoing_write_through:
                for ack in cc.ack_write_queue:
                    with pp_bubble_diag.wait("ack_sync.write_flush"):
                        ack.finish_event.synchronize()
                    for ack_id in ack.node_ids:
                        if ack_id in self.ongoing_write_through:
                            self._finish_write_through_ack(ack_id)
                    self._log_write_ack_metrics(ack)
                cc.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            if t0 is not None:
                pmicro_hoststats.OPS["evict_wait"].add(time.perf_counter() - t0)
            return

        if finish_count is None:
            # Every rank must enter the all_reduce below; ongoing_write_through can
            # diverge across ranks (e.g. write_backup returning 0 on a subset).
            finish_count = 0
            if self.pp_rank == 0:
                finish_count = self._count_ready_acks(cc.ack_write_queue)
            finish_count_tensor = torch.tensor(
                finish_count, dtype=torch.int, device="cpu"
            )
            self._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
            finish_count = finish_count_tensor.item()

        # Process completed acks
        while finish_count > 0:
            ack = cc.ack_write_queue.pop(0)
            with pp_bubble_diag.wait("ack_sync.write"):
                ack.finish_event.synchronize()
            for ack_id in ack.node_ids:
                self._finish_write_through_ack(ack_id)
            self._log_write_ack_metrics(ack)
            finish_count -= 1

    def _log_write_ack_metrics(self, ack: HiCacheAck) -> None:
        """Record D->H backup volume and duration for a completed write ack."""
        if self.metrics_collector is None:
            return
        for pool, num_tokens in (ack.num_tokens_by_pool or {}).items():
            if num_tokens > 0:
                self.metrics_collector.increment_backup_num_tokens(
                    num_tokens=num_tokens, pool=pool
                )
        if ack.num_bytes > 0:
            self.metrics_collector.increment_backup_num_bytes(ack.num_bytes)
        if ack.timing_enabled:
            duration_ms = ack.start_event.elapsed_time(ack.finish_event)
            self.metrics_collector.observe_backup_duration(duration_ms / 1000.0)

    def loading_check(self, finish_count: Optional[int] = None) -> None:
        """Poll load-back completions."""
        cc = self.cache_controller
        if cc is None:
            return
        if finish_count is None:
            # Every rank must enter the all_reduce below; ongoing_load_back can
            # diverge across ranks.
            finish_count = 0
            if self.pp_rank == 0:
                finish_count = self._count_ready_acks(cc.ack_load_queue)
            # Piggybacked TP check: [digest, -digest] MIN-reduces to [min, -max],
            # equal iff reclaim victim order matched on every rank.
            digest = self.tree_core.write_back_duplicate_reclaim_digest
            sync_tensor = torch.tensor(
                [finish_count, digest, -digest], dtype=torch.int64, device="cpu"
            )
            self._all_reduce(sync_tensor, torch.distributed.ReduceOp.MIN)
            finish_count = int(sync_tensor[0].item())
            assert sync_tensor[1].item() == -sync_tensor[2].item(), (
                "write_back duplicate-reclaim victims diverged across TP ranks"
            )

        while finish_count > 0:
            ack = cc.ack_load_queue.pop(0)
            with pp_bubble_diag.wait("ack_sync.load"):
                ack.finish_event.synchronize()
            for ack_id in ack.node_ids:
                if (
                    self.buffer_pipeline is not None
                    and self.buffer_pipeline.try_finish_load_back(ack_id)
                ):
                    continue
                node, lock_params, host_lock_params = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(node, lock_params)
                self.dec_host_lock_ref(node, host_lock_params)
                # Unpin the loaded nodes; host copies stay as reclaimable duplicates.
                self.tree_core.finish_load_back(node)

            if self.metrics_collector is not None:
                for pool, num_tokens in (ack.num_tokens_by_pool or {}).items():
                    if num_tokens > 0:
                        self.metrics_collector.increment_load_back_num_tokens(
                            num_tokens=num_tokens, pool=pool
                        )
                if ack.num_bytes > 0:
                    self.metrics_collector.increment_load_back_num_bytes(ack.num_bytes)
                if ack.timing_enabled:
                    duration_ms = ack.start_event.elapsed_time(ack.finish_event)
                    self.metrics_collector.observe_load_back_duration(
                        duration_ms / 1000.0
                    )
            finish_count -= 1

    # ---- HiCache: Scheduler Entry Points ----

    @rank_consensus(same_params=["params.host_hit_length"])
    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> Optional[tuple[int, NodeId]]:
        """Only buffer mode returns None: its admission retries without
        committing a load."""
        if self.buffer_pipeline is not None:
            return self.buffer_pipeline.init_load_back(params)
        best_match_node_id = params.best_match_node
        mem_quota = params.mem_quota
        req = params.req
        assert req is not None
        if self.linker is not None and self.linker.has_hit(req.rid):
            return self.linker.load_back(req)
        last_best_match_device_node_id = req.last_node

        if (
            self.tree_core.is_full_device_evicted(best_match_node_id)
            or params.host_hit_length > 0
            or (
                req is not None
                and (req.swa_host_hit_length > 0 or req.mamba_host_hit_length > 0)
            )
        ):
            if self.load_back(best_match_node_id, mem_quota, req=req):
                new_indices = self.tree_core.collect_full_device_indices(
                    best_match_node_id, last_best_match_device_node_id
                )
                if new_indices.numel() == 0:
                    return 0, last_best_match_device_node_id

                logger.debug(
                    "init_load_back success: loaded %d tokens for node %d",
                    len(new_indices),
                    best_match_node_id,
                )
                return len(new_indices), best_match_node_id

        return 0, last_best_match_device_node_id

    def load_back_is_cache_owned(self) -> bool:
        # Linker-loaded slots stay request-owned until the load succeeds.
        # Keep cache_protected_len unchanged so failed loads can be freed.
        return self.linker is None

    def finish_external_linker_loads(self, reqs: list[Req]) -> list[Req]:
        """Wait for the external-linker loads of the batch that just ran and
        return the requests whose KV did not land, marked so their output is
        dropped and their KV never reaches the tree. Their slots are private
        (see :meth:`UnifiedCacheLinkerWrapper.load_back`), so nothing else read
        them."""
        if self.linker is None:
            return []
        failed_rids = set(self.linker.finish_loads())
        failed = [req for req in reqs if req.rid in failed_rids]
        if failed:
            from sglang.srt.managers.schedule_batch import FINISH_ABORT

            for req in failed:
                req.skip_radix_cache_insert = True
                req.discard_output_reason = FINISH_ABORT(
                    "External KV cache load failed", HTTPStatus.SERVICE_UNAVAILABLE
                )
        return failed

    def check_hicache_events(self) -> None:
        """Called per scheduler step to poll async HiCache events."""
        if self.linker is not None:
            finish_counts = torch.tensor(
                [self.linker.num_completed_offloads()],
                dtype=torch.int,
                device="cpu",
            )
            self._all_reduce_attn_groups(finish_counts, torch.distributed.ReduceOp.MIN)
            offload_count = int(finish_counts.item())
            local_successes = self.linker.take_completed_offloads(offload_count)
            if local_successes:
                successes = torch.tensor(local_successes, dtype=torch.int, device="cpu")
                self._all_reduce_attn_groups(successes, torch.distributed.ReduceOp.MIN)
                self.linker.commit_completed_offloads(
                    [bool(success) for success in successes.tolist()]
                )
            return

        # Reap the previous round's PP-sync sends before issuing new ones.
        self._drain_async_work()
        # Backups queued outside process_batch_result: the chunked-prefill stash
        # in get_next_batch_to_run, abort_request, and the PD prefill release.
        self.flush_pending_backups()

        (
            write_finish_count,
            load_finish_count,
            storage_queue_sizes,
            extra_pool_names,
        ) = self._sync_hicache_ready_counts()
        self.writing_check(finish_count=write_finish_count)
        self.loading_check(finish_count=load_finish_count)
        if self._device_write_ahead:
            self._write_ahead_commit(self._wa_ready_count)

        if self.enable_storage and storage_queue_sizes:
            n_storage_hit, n_ack_prefetch, n_backup, n_release = storage_queue_sizes[:4]
            extra_release_counts = {
                pool_name: count
                for pool_name, count in zip(extra_pool_names, storage_queue_sizes[4:])
            }
            self._drain_storage_control_queues_impl(
                n_storage_hit=n_storage_hit,
                n_ack_prefetch=n_ack_prefetch,
                n_backup=n_backup,
                n_release=n_release,
                extra_release_counts=extra_release_counts,
                log_metrics=True,
            )
        if self._l3_tombstone_pending is not None:
            self._refill_mamba_tombstone_reserve()
        if self._device_write_ahead:
            self._write_ahead_device_tail()
        if self._l3_write_on_host_evict:
            t0 = time.perf_counter() if pmicro_hoststats.ON else 0.0
            self._write_behind_host_tail()
            if pmicro_hoststats.ON:
                pmicro_hoststats.OPS["write_behind"].add(time.perf_counter() - t0)
        if pmicro_hoststats.ON and self.enable_storage:
            pmicro_hoststats.sample_l3_pending(
                self._pmicro_l3_pending(), len(self._write_behind_mamba_inflight)
            )
        if self.enable_storage and self.pp_rank == 0:
            # Diagnostics only: wall clock picks the log step, no decision reads it.
            now = time.monotonic()
            if now - self._l3_stats_logged_at >= 60.0:
                self._l3_stats_logged_at = now
                logger.info(
                    "HiCache L3 prefetch stats: %s", self.prefetch_outcome_stats_snapshot()
                )
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.flush_pending_writes()
        self._drain_l3_lifetime_events()
        if self.enable_storage_metrics and self.storage_metrics_collector is not None:
            storage_metrics = self.cache_controller.storage_backend.get_stats()
            if storage_metrics is None:
                storage_metrics = StorageMetrics()
            if not hasattr(storage_metrics, "prefetch_stats"):
                storage_metrics.prefetch_stats = self.prefetch_outcome_stats_snapshot()
            self.storage_metrics_collector.log_storage_metrics(storage_metrics)

    def flush_pending_backups(self) -> None:
        """Submit pending D2H backups as a merged operation."""
        if self.linker is not None or self.cache_controller is None:
            return
        self.cache_controller.start_writing()

    def ready_to_load_host_cache(self) -> int:
        """Notify the cache controller to start the KV cache loading."""
        if self.linker is not None:
            return self.linker.start_layer_wise_loading()
        if self.cache_controller is not None:
            return self.cache_controller.start_loading()
        return 0

    def is_load_back_event_done(self, consumer_index: int) -> bool:
        """Return True after this rank's load-back event is complete."""
        if consumer_index < 0 or self.cache_controller is None:
            return True

        finish_event = self.cache_controller.layer_done_counter.events[
            consumer_index
        ].finish_event
        return finish_event.query()

    def has_free_load_back_slot(self) -> bool:
        """Acks are reaped in lockstep, so every rank agrees on this."""
        cc = self.cache_controller
        return len(cc.ack_load_queue) < cc.layer_done_counter.num_counters

    # ---- Query / Inspection APIs ----
    # These APIs exist for compatibility with other RadixTree implementations.
    # TODO: simplify and consolidate in a future refactor.

    @property
    def sliding_window_size(self):
        return self._sliding_window_size

    def swa_reprefill_tail_tokens(self, req=None) -> int:
        """Tokens at the end of a matched prefix to re-prefill because their SWA
        cannot be reused as is.

        unified_kv keeps SWA in a per-request ring (state_slot/pos) the tree never
        stores, so a reused window would read another request's stale slots; a
        host pool restores exact contents instead. Decoder SWA bounded replay
        writes late-layer window KV only for an extend's tail rows, so a matched
        prefix may lack it; a streaming turn borrowing its own record has it.
        """
        from sglang.srt.layers.attention.dsv4.unified_kv_gate import (
            is_unified_kv_triton,
        )

        swa = self.components.get(ComponentType.SWA)
        if swa is None or not swa.sliding_window_size:
            return 0
        if self.decoder_swa_bounded_replay and not self.session.will_lend_record(req):
            return swa.sliding_window_size
        if not is_unified_kv_triton():
            return 0
        if self.tree_core.has_swa_host_pool:
            return 0
        return swa.sliding_window_size

    def swa_retain_floor(self, req) -> int | None:
        if not self.is_mamba_enabled or self._sliding_window_size is None:
            return None
        checkpoint = req.kv.mamba_last_track_seqlen
        if checkpoint is None:
            return None
        return checkpoint - self._sliding_window_size

    def supports_swa(self) -> bool:
        return self.is_swa_enabled

    def supports_mamba(self) -> bool:
        # A request-only cache still owns the requests' Mamba slot lifecycle.
        return self.is_mamba_enabled or self.request_mamba_only

    # ---- Session radix cache API (delegates to composed UnifiedSessionRefTracker) ----

    def open_radix_session(self, session_id: str) -> Optional[int]:
        return self.session_refs.open_radix_session(session_id)

    def ensure_session_generation(self, session_id: str) -> int:
        return self.session_refs.ensure_session_generation(session_id)

    def release_radix_session(self, session_id: str) -> int:
        return self.session_refs.release_radix_session(session_id)

    # ---- Streaming session API (delegates to self.session) ----

    def release_session(self, session_id: str) -> None:
        self.session.release_session(session_id)

    def session_records(self) -> dict[str, ReqKvInfo]:
        return {sid: slot.kv for sid, slot in self.session.slots.items()}

    def evictable_size(self) -> int:
        return self.tree_core.evictable_size()

    def protected_size(self) -> int:
        return self.tree_core.protected_size()

    def full_evictable_size(self) -> int:
        return self.tree_core.full_evictable_size()

    def full_protected_size(self) -> int:
        return self.tree_core.full_protected_size()

    def swa_evictable_size(self) -> int:
        return self.tree_core.swa_evictable_size()

    def mamba_evictable_size(self) -> int:
        return self.tree_core.mamba_evictable_size()

    def swa_protected_size(self) -> int:
        return self.tree_core.swa_protected_size()

    def mamba_protected_size(self) -> int:
        return self.tree_core.mamba_protected_size()

    def total_size(self) -> tuple[int, int]:
        return self.tree_core.total_size()

    def all_values_flatten(self) -> torch.Tensor:
        return self.tree_core.all_values_flatten()

    def all_mamba_values_flatten(self) -> torch.Tensor:
        if not self.is_mamba_enabled:
            return torch.empty((0,), dtype=torch.int64)
        return self.tree_core.all_mamba_values_flatten()

    def available_and_evictable_str(self) -> str:
        # TODO(zhangmj): need more detailed log info for session reference.
        if self.supports_swa():
            full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        else:
            full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable = self.tree_core.component_evictable_size(BASE_COMPONENT_TYPE)
        lines = [
            f"Available full tokens: {full_available_size + full_evictable} "
            f"(full_available_size={full_available_size} + full_evictable_size_={full_evictable})"
        ]
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue
            if ct.is_swa:
                available_size = self.token_to_kv_pool_allocator.swa_available_size()
            elif ct.is_mamba:
                available_size = self.req_to_token_pool.mamba_allocator.available_size()
            else:
                continue

            lines.append(
                f"Available {ct}: {available_size + self.tree_core.component_evictable_size(ct)} "
                f"(available_size={available_size} + component_evictable_size_={self.tree_core.component_evictable_size(ct)})"
            )
        return "\n".join(lines) + "\n"

    def sanity_check(self):
        """Verify tree invariants.

        TODO(hzh): This method has relatively high latency; simplify the
        check logic once the tree implementation stabilizes.
        """
        # Skip when streaming sessions hold tree locks: the check asserts
        # all nodes are unlocked during idle, which streaming sessions break
        # by design (each slot holds a lock on its prefix across turns).
        if self.session.any_holding_kv():
            return

        # Pass ongoing ops as lightweight (id, node_id) pairs so the tree core
        # can resolve + validate them without reaching into Controller state.
        if self.buffer_pipeline is not None:
            ongoing_write_through = [
                (nid, entry.intent.snapshot.node_id)
                for nid, entry in self.buffer_pipeline.ongoing_write_through.items()
            ]
        else:
            ongoing_write_through = [
                (nid, wt.node_id)
                for ongoing in (self.ongoing_write_through, self.ongoing_write_ahead)
                for nid, wt in ongoing.items()
            ]
        ongoing_load_back = [
            (nid, lb.node_id) for nid, lb in self.ongoing_load_back.items()
        ]
        self.tree_core.sanity_check(ongoing_write_through, ongoing_load_back)

    def pretty_print(self) -> None:
        self.tree_core.pretty_print()

    # ---- TreeCore state delegation ----
    # The facade re-exposes tree-owned config (page_size, enable_storage, ...) so its
    # own coordination methods and external callers read them off the cache.

    # ``page_size`` keeps a setter: StreamingSession forwards assignment onto its
    # inner cache (the PrefixCacheTrait surface).
    @property
    def page_size(self):
        return self.tree_core.page_size

    @page_size.setter
    def page_size(self, value) -> None:
        self.tree_core.page_size = value

    @property
    def enable_storage(self):
        return self.tree_core.enable_storage

    @enable_storage.setter
    def enable_storage(self, value) -> None:
        self.tree_core.enable_storage = value

    @property
    def write_through_threshold(self):
        return self.tree_core.write_through_threshold

    @write_through_threshold.setter
    def write_through_threshold(self, value) -> None:
        self.tree_core.write_through_threshold = value

    @property
    def is_write_back(self):
        return self.tree_core.is_write_back

    @is_write_back.setter
    def is_write_back(self, value) -> None:
        self.tree_core.is_write_back = value

    @property
    def device(self):
        return self.tree_core.device

    @property
    def root_node(self):
        return self.tree_core.root_node

    def take_events(self):
        # Drain the KV event queue from the TreeCore.
        return self.tree_core.take_events()

    def resolve_node_handle(self, node_handle):
        """Look up the node object from its NodeId.

        TODO(Jialin): Remove after the Unified Radix Cache split.
        """
        if isinstance(node_handle, int):
            return self.tree_core.node_by_id(node_handle)
        # Internal callers (and the session sentinel / None) pass a non-int through.
        return node_handle

    def prefix_device_indices(self, req: Req) -> torch.Tensor:
        root = self.root_node_handle(req.extra_key)
        path = self.tree_core.collect_full_device_indices(req.last_node, root)
        if self.linker is not None:
            tail = self.linker.private_load_indices.get(req.rid)
            if tail is not None:
                path = torch.cat([path, tail])
        assert len(path) >= req.prefix_len, (req.rid, len(path), req.prefix_len)
        return path[: req.prefix_len]

    def path_device_indices(self, node: NodeId) -> torch.Tensor:
        return self.tree_core.collect_full_device_indices(node, self.root_node_handle())

    def root_node_handle(self, extra_key: Optional[str] = None) -> NodeId:
        """The root's NodeId -- URC match results carry NodeIds."""
        return self.tree_core.root_node_handle(extra_key)

    def dfs_weight_order(self, node_handles: Sequence[NodeId]) -> list[int]:
        return self.tree_core.dfs_weight_order(node_handles)
