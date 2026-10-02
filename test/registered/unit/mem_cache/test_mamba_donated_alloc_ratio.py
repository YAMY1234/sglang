"""CPU-only unit tests for the mamba pool ratio vs the prefill->decode peak.

Pins the sizing invariant behind MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO:
at the first cache_unfinished_req, a request still holds its admission-locked
matched-prefix mamba (protected) plus its own COW slot, and then allocates a
donated slot. With N distinct-prefix requests that peak is N own + N locked +
1 donated. An effective ratio of 2 (pool = 2N) leaves no evictable victim and
the donated alloc asserts; ratio 3 (pool = 3N) has headroom. Once decode's
skip_mamba leaves the matched prefix evictable, even ratio 2 recovers via
eviction -- which is why the peak, not the decode steady state, sets the floor.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
)
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

N = 4  # concurrent distinct-prefix requests


class _BoundedMambaAllocator:
    """Fixed-capacity slot allocator; alloc returns None once exhausted."""

    def __init__(self, size: int):
        self.free_ids = list(range(size))

    def alloc(self, n: int):
        if len(self.free_ids) < n:
            return None
        return torch.tensor([self.free_ids.pop() for _ in range(n)], dtype=torch.int64)

    def free(self, value: torch.Tensor):
        self.free_ids.extend(int(v) for v in value.tolist())


class _RatioCache:
    tree_components = (ComponentType.FULL, ComponentType.MAMBA)

    def __init__(self, pool_size: int):
        self.root_node = UnifiedTreeNode(self.tree_components)
        self.allocator = _BoundedMambaAllocator(pool_size)
        self.req_to_token_pool = SimpleNamespace(mamba_allocator=self.allocator)
        self.component_evictable_size_ = {ComponentType.MAMBA: 0}
        self.component_protected_size_ = {ComponentType.MAMBA: 0}
        self.prefix_nodes = []
        self.alloc_evict_params = []

    def evict_for_alloc(self, params: EvictParams):
        self.alloc_evict_params.append(params)
        need = params.mamba_num
        for node in list(self.prefix_nodes):
            if need <= 0:
                break
            cd = node.component_data[ComponentType.MAMBA]
            if cd.lock_ref == 0 and cd.value is not None:
                self.allocator.free(cd.value)
                self.component_evictable_size_[ComponentType.MAMBA] -= len(cd.value)
                cd.value = None
                self.prefix_nodes.remove(node)
                need -= 1


def _build_peak(pool_size: int, lock_prefixes: bool):
    """N own slots + N matched-prefix snapshots, then return the component ready
    to allocate one donated slot. Prefix snapshots are locked (protected,
    prefill peak) or left evictable (decode steady state after skip_mamba)."""
    cache = _RatioCache(pool_size)
    component = object.__new__(MambaComponent)
    component.cache = cache
    # The TreeCore owns the tree member-var state the component reads through.
    component.tree_core = cache
    component.component_type = ComponentType.MAMBA

    owned = [cache.allocator.alloc(1) for _ in range(N)]
    assert all(s is not None for s in owned)

    for _ in range(N):
        node = UnifiedTreeNode(cache.tree_components)
        slot = cache.allocator.alloc(1)
        assert slot is not None
        node.component_data[ComponentType.MAMBA].value = slot
        cache.component_evictable_size_[ComponentType.MAMBA] += len(slot)
        cache.prefix_nodes.append(node)
        if lock_prefixes:
            component.acquire_component_lock(node, IncLockRefResult())

    return component, cache, owned


class TestMambaRatioEnvGate(unittest.TestCase):
    """SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK gates the pool ratio: off restores the
    original base 3 (overlap 5, lazy 4, no_buffer 3), on drops the base to 2
    (overlap 4, lazy 3) while no_buffer stays 3. Guards the flag wiring so the
    ratio can never drift out of sync with whether the decode lock is skipped."""

    @staticmethod
    def _ratio(*, extra_buffer, lazy, disable_overlap, skip):
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        fake = SimpleNamespace(server_args=SimpleNamespace())
        # Every input is a published leaf now: the extra-buffer predicates read
        # the radix-cache strategy off the bags, so the fixture publishes the
        # strategy that produces the combination under test.
        strategy = (
            "extra_buffer_lazy"
            if lazy
            else "extra_buffer"
            if extra_buffer
            else "no_buffer"
        )
        from sglang.srt import runtime_context as rc

        with envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.override(skip):
            with rc.get_context().override_server_args(
                disable_radix_cache=False,
                disable_overlap_schedule=disable_overlap,
                mamba_radix_cache_strategy=strategy,
            ):
                return KVCacheConfigurator._calculate_mamba_ratio(fake)

    def test_prefill_only_capacity_preserves_transient_headroom(self):
        from sglang.srt import runtime_context as rc
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
        from sglang.srt.managers.schedule_policy import PrefillAdder

        fake = SimpleNamespace(model_config=SimpleNamespace(context_len=262144),
                               attn_dp_size=1, mambaish_config=object())
        fake._calculate_mamba_ratio = lambda: KVCacheConfigurator._calculate_mamba_ratio(fake)
        pool = SimpleNamespace(mamba_allocator=SimpleNamespace(schedulable_available_size=lambda: 3),
                               enable_mamba_extra_buffer=True, mamba_ping_pong_track_buffer_size=2)
        adder = SimpleNamespace(is_hybrid_ssm_cache=True, can_run_list=[],
                                tree_cache=SimpleNamespace(req_to_token_pool=pool, mamba_evictable_size=lambda: 0))
        for enabled, requested, expected in [(False, 144, 96), (True, 144, 144), (True, 192, 190)]:
            adder.tree_cache.mamba_evictable_size = lambda: 0
            with rc.get_context().override_server_args(
                radix_cache_skip_decode_insert=enabled, disable_radix_cache=False,
                disable_overlap_schedule=False, mamba_radix_cache_strategy="extra_buffer_lazy",
                max_mamba_cache_size=384, max_running_requests=requested,
            ):
                self.assertEqual(KVCacheConfigurator.resolve_max_num_reqs(fake, 10_000_000), expected)
                self.assertEqual(PrefillAdder._has_mamba_prefill_headroom(adder), not enabled)
                if enabled:
                    adder.tree_cache.mamba_evictable_size = lambda: 1
                    self.assertTrue(PrefillAdder._has_mamba_prefill_headroom(adder))

    def test_flag_off_restores_original_ratios(self):
        def r(**kwargs):
            return self._ratio(skip=False, **kwargs)

        self.assertEqual(
            r(extra_buffer=False, lazy=False, disable_overlap=True), 3
        )  # no_buffer
        self.assertEqual(
            r(extra_buffer=True, lazy=True, disable_overlap=False), 4
        )  # lazy
        self.assertEqual(
            r(extra_buffer=True, lazy=False, disable_overlap=False), 5
        )  # overlap

    def test_flag_on_drops_base_but_keeps_no_buffer(self):
        def r(**kwargs):
            return self._ratio(skip=True, **kwargs)

        self.assertEqual(
            r(extra_buffer=False, lazy=False, disable_overlap=True), 3
        )  # no_buffer
        self.assertEqual(
            r(extra_buffer=True, lazy=True, disable_overlap=False), 3
        )  # lazy
        self.assertEqual(
            r(extra_buffer=True, lazy=False, disable_overlap=False), 4
        )  # overlap


class _RecordingComp:
    """Fake tree component: records the dec params it is asked to release with."""

    def __init__(self, component_type, priority):
        self.component_type = component_type
        self._priority = priority
        self.released = []

    def eviction_priority(self, is_leaf):
        return self._priority

    def release_component_lock(self, node, params):
        self.released.append(params)

    def release_window_lock(  # SWA only
        self, node, swa_uuid_for_lock, device_frees, host_frees
    ):
        pass


class TestDecSwaLockSkip(unittest.TestCase):
    """dec_swa_lock_only early-releases SWA plus co-located lower-tier (Mamba)
    locks. On a node whose acquire skipped Mamba (decode hold), the release
    must skip it too, else it drops a mamba lock it never took -- another
    request's, on a shared FULL+SWA+MAMBA node (Inkling). Guards the contract
    without booting a 3-component model."""

    def _run(self, skipped_lock_components):
        # internal-node priority: full=2 > swa=1 > mamba=0
        full = _RecordingComp(ComponentType.FULL, 2)
        swa = _RecordingComp(ComponentType.SWA, 1)
        mamba = _RecordingComp(ComponentType.MAMBA, 0)
        node = SimpleNamespace(id=7)
        tree_core = SimpleNamespace(
            root_node=object(),
            components=(full, swa, mamba),
            components_by_type={ComponentType.SWA: swa},
            node_by_id=lambda node_id: node,
            _assert_receipt_anchor=UnifiedTreeCore._assert_receipt_anchor,
        )
        UnifiedTreeCore.dec_swa_lock_only(
            tree_core,
            node.id,
            DecLockRefParams(
                node_id=node.id,
                skipped_lock_components=skipped_lock_components,
                component_lock_uuids={ComponentType.SWA: None},
            ),
        )
        return full, mamba

    def test_unlocked_mamba_is_not_released(self):
        full, mamba = self._run(skipped_lock_components=(ComponentType.MAMBA,))
        # mamba took no lock at acquire, so the early release skips it too
        self.assertEqual(mamba.released, [])
        # full (above swa) is never touched
        self.assertEqual(full.released, [])

    def test_lower_tier_released_when_locked(self):
        full, mamba = self._run(skipped_lock_components=())
        self.assertEqual(len(mamba.released), 1)
        self.assertEqual(full.released, [])


class TestMambaDonatedAllocRatio(unittest.TestCase):
    def test_prefill_peak_ratio2_exhausts_pool(self):
        # pool = 2N, all N prefixes admission-locked: no evictable victim.
        component, cache, _ = _build_peak(pool_size=2 * N, lock_prefixes=True)
        with self.assertRaisesRegex(AssertionError, "Can not alloc mamba cache"):
            component._alloc_mamba_slot()
        self.assertEqual(
            cache.alloc_evict_params, [EvictParams(num_tokens=0, mamba_num=1)]
        )

    def test_prefill_peak_ratio3_has_headroom(self):
        # pool = 3N: N free slots remain after own + locked prefix.
        component, cache, _ = _build_peak(pool_size=3 * N, lock_prefixes=True)
        slot = component._alloc_mamba_slot()
        self.assertIsNotNone(slot)
        self.assertEqual(cache.component_protected_size_[ComponentType.MAMBA], N)

    def test_decode_steady_evictable_prefix_ratio2_ok(self):
        # pool = 2N but the matched prefixes are evictable (skip_mamba on decode):
        # eviction reclaims a victim, so even ratio 2 serves the donated alloc.
        component, cache, _ = _build_peak(pool_size=2 * N, lock_prefixes=False)
        slot = component._alloc_mamba_slot()
        self.assertIsNotNone(slot)
        self.assertEqual(len(cache.prefix_nodes), N - 1)
        self.assertEqual(
            cache.alloc_evict_params, [EvictParams(num_tokens=0, mamba_num=1)]
        )


class TestPPMambaPoolSizing(unittest.TestCase):
    """A PP rank only allocates mamba state for its own [start_layer, end_layer)
    slice, so charging it for the whole model's layers starves the pool. Sizing
    uses the largest per-stage share, which also keeps every rank on the same
    pool size (and hence the same max_running_requests / pp_max_micro_batch_size)
    without a collective."""

    # Kimi-K3 shaped: 93 layers, linear attention everywhere except every 4th and
    # the last, so the 69 mamba layers split unevenly over 8 stages (9 or 8 each).
    TOTAL_LAYERS = 93
    MAMBA_LAYERS = [i for i in range(93) if (i + 1) % 4 != 0 and i <= 90]
    BUDGET_GB = 8.0

    @classmethod
    def _pool_size(cls, pp_rank, pp_size):
        from sglang.srt import runtime_context as rc
        from sglang.srt.configs.mamba_utils import (
            Mamba2CacheParams,
            Mamba2StateDType,
            Mamba2StateShape,
        )
        from sglang.srt.distributed.utils import get_pp_indices
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
        from sglang.srt.runtime_context import get_schedule

        shape = Mamba2StateShape(
            conv=[(4096, 3)],
            temporal=(64, 128, 128),
            intermediate_size=0,
            conv_dim=0,
            ssm_state_size=0,
            num_heads=0,
            head_dim=0,
            state_size=0,
            conv_kernel=0,
            num_k_heads_per_tp=8,
        )
        params = Mamba2CacheParams(
            shape=shape,
            dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
            layers=list(cls.MAMBA_LAYERS),
        )
        start, end = get_pp_indices(cls.TOTAL_LAYERS, pp_rank, pp_size)
        fake = SimpleNamespace(
            mambaish_config=SimpleNamespace(mamba2_cache_params=params),
            extra_mamba_cache_bytes_per_req=0,
            server_args=SimpleNamespace(),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            layer_info=SimpleNamespace(start_layer=start, end_layer=end),
            attn_dp_size=1,
            pp_size=pp_size,
            hybrid_gdn_config=None,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(), num_hidden_layers=cls.TOTAL_LAYERS
            ),
        )
        with rc.get_context().override_server_args(
            disable_radix_cache=False,
            max_mamba_cache_size=None,
            max_running_requests=None,
            mamba_full_memory_ratio=0.5,
            enable_linear_replayssm_spec=False,
        ):
            KVCacheConfigurator._handle_max_mamba_cache(fake, cls.BUDGET_GB)
            return get_schedule().max_mamba_cache_size

    def test_stage_is_not_charged_for_the_whole_model(self):
        solo = self._pool_size(0, 1)
        staged = self._pool_size(0, 8)
        # The busiest stage holds 9 of the 69 mamba layers, so it should fit
        # roughly 69/9 more slots than a rank holding all of them. pp_size=1 is
        # unchanged: that rank does hold every layer.
        self.assertGreater(staged, solo * 5)

    def test_every_stage_agrees_on_the_pool_size(self):
        sizes = {self._pool_size(r, 8) for r in range(8)}
        self.assertEqual(
            len(sizes), 1, f"per-rank pool sizes diverged: {sorted(sizes)}"
        )


class TestExtraMambaCacheSizing(unittest.TestCase):
    @staticmethod
    def _size(extra_bytes, *, pp_size=1, dp_size=1, draft_tokens=None, **schedule):
        from sglang.srt import runtime_context as rc
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
        from sglang.srt.runtime_context import get_schedule

        fake = SimpleNamespace(
            mambaish_config=SimpleNamespace(
                mamba2_cache_params=SimpleNamespace(
                    layers=[0, 1, 2, 3], mamba_cache_per_req=4 << 20
                )
            ),
            extra_mamba_cache_bytes_per_req=extra_bytes,
            spec_algorithm=SimpleNamespace(is_none=lambda: draft_tokens is None),
            attn_dp_size=dp_size,
            pp_size=pp_size,
            model_config=SimpleNamespace(num_hidden_layers=4),
            _calculate_mamba_ratio=lambda: 2,
        )
        args = dict(
            disable_radix_cache=False,
            max_mamba_cache_size=None,
            max_running_requests=256 if draft_tokens is not None else None,
            mamba_full_memory_ratio=0.5,
            enable_linear_replayssm_spec=False,
            speculative_num_draft_tokens=draft_tokens,
        )
        args.update(schedule)
        with rc.get_context().override_server_args(**args):
            remaining = KVCacheConfigurator._handle_max_mamba_cache(fake, 120 / 1024)
            return get_schedule().max_mamba_cache_size, remaining * (1 << 30)

    def test_auto_capacity_reserves_backend_state(self):
        self.assertEqual(self._size(0), (9, 80 << 20))
        self.assertEqual(self._size(4 << 20), (4, 80 << 20))

    def test_fixed_capacity_charges_padding_and_partitioned_state(self):
        for pp_size in (1, 2):
            for dp_size in (1, 2):
                for schedule in (
                    dict(max_mamba_cache_size=8),
                    dict(disable_radix_cache=True, max_running_requests=8),
                ):
                    with self.subTest(pp=pp_size, dp=dp_size, **schedule):
                        slots, remaining = self._size(
                            8 << 20, pp_size=pp_size, dp_size=dp_size, **schedule
                        )
                        self.assertEqual(slots, 8 // dp_size)
                        self.assertEqual(
                            remaining,
                            (120 << 20) - (slots + 1) * (12 << 20) // pp_size,
                        )

    def test_speculative_auto_capacity_keeps_scratch_separate(self):
        self.assertEqual(self._size(0, draft_tokens=2), (3, 88 << 20))
        self.assertEqual(self._size(4 << 20, draft_tokens=2), (2, 80 << 20))


if __name__ == "__main__":
    unittest.main()
from unittest.mock import Mock
import pytest
from sglang.srt.arg_groups.fields.memory import Memory
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache import common
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.unified_cache.components import mamba
from sglang.srt.speculative import spec_utils
@pytest.fixture
def skip_decode(monkeypatch):
    ns, pool = SimpleNamespace, object.__new__(HybridReqToTokenPool)
    pool.mamba_allocator, pool.free = _BoundedMambaAllocator(10), Mock()
    live, track = pool.mamba_allocator.alloc(1)[0], pool.mamba_allocator.alloc(1)
    pool.enable_mamba_extra_buffer = pool.enable_mamba_extra_buffer_lazy = True
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping = track.reshape(1, 1)
    req = ns(kv=ReqKvInfo(req_pool_idx=0, kv_committed_len=4, kv_allocated_len=4,
             mamba_pool_idx=live, mamba_ping_pong_track_buffer=track,
             mamba_next_track_idx=0, mamba_last_track_idx=0, mamba_last_track_seqlen=4),
             origin_input_ids=[1, 2, 3, 4], output_ids=[5, 6], skip_radix_cache_insert=False, owned_kv_len=lambda: 4)
    tree = ns(enable_mamba_extra_buffer=True, req_to_token_pool=pool, supports_mamba=lambda: True,
              claim_kv_row=lambda r: False, token_to_kv_pool_allocator=ns(page_size=1),
              free_kv_row=Mock(), unpin=Mock(), insert_req=Mock(), prefixes={})
    component = object.__new__(MambaComponent)
    component.cache, component._alloc_mamba_slot = tree, lambda: pool.mamba_allocator.alloc(1)
    def publish(r, **kw):
        params = InsertParams()
        length = component.prepare_for_caching_req(r, params, r.owned_kv_len(), False)
        tree.prefixes[length], r.kv.cache_protected_len = params.mamba_value.item(), length
        component.cleanup_after_caching_req(r, False, insert_result=ns(mamba_exist=False), insert_params=params)
    tree.cache_unfinished_req, tree.on_release = Mock(side_effect=publish), lambda r, inserted: component.cleanup_after_caching_req(r, True)
    memory = ns(radix_cache_skip_decode_insert=True)
    for module in (common, mamba, spec_utils): monkeypatch.setattr(module, "get_memory", lambda: memory)
    monkeypatch.setattr(common, "get_spec", lambda: ns(speculative_algorithm="EAGLE3"))
    return req, tree, pool, memory
def test_skip_decode_default_unchanged(skip_decode):
    req, tree, _, memory = skip_decode
    memory.radix_cache_skip_decode_insert = Memory().radix_cache_skip_decode_insert
    common.release_kv_cache(req, tree)
    tree.insert_req.assert_called_once()
def test_skip_decode_prefill_checkpoint_retained(skip_decode):
    req, tree, pool, _ = skip_decode
    common.maybe_cache_unfinished_req(req, tree)
    assert tree.prefixes == {4: 8} and 8 not in pool.mamba_allocator.free_ids and req.kv.mamba_ping_pong_track_buffer is None
def test_skip_decode_finish_releases_once(skip_decode):
    req, tree, pool, _ = skip_decode
    common.maybe_cache_unfinished_req(req, tree)
    common.release_kv_cache(req, tree)
    tree.insert_req.assert_not_called()
    assert sorted(pool.mamba_allocator.free_ids) == [0, 1, 2, 3, 4, 5, 6, 7, 9]
def test_skip_decode_verify_updates_live_state(skip_decode, monkeypatch):
    ns, live = SimpleNamespace, torch.zeros(1)
    batch = ns(mamba_track_indices=torch.tensor([9]), mamba_track_mask=torch.tensor([True]), forward_mode=ns(is_idle=lambda: False), req_pool_indices=torch.tensor([0]))
    spec_utils.prepare_mamba_track_for_verify(batch)
    update = Mock(side_effect=lambda **kw: live.copy_(torch.tensor([11., 22., 33., 44.])[kw["last_correct_step_indices"]]))
    worker = ns(model_runner=ns(model_config=ns(), req_to_token_pool=ns(), model=object(), attn_backend=ns(update_mamba_state_after_mtp_verify=update)))
    monkeypatch.setattr(spec_utils, "mambaish_config", lambda _: object())
    monkeypatch.setattr(spec_utils, "pp_spec_stable_rows_enabled", lambda: False)
    spec_utils.commit_mamba_states_after_verify(worker, batch, torch.tensor([3]), torch.tensor([[0, 1, 2, 3]]), 4)
    assert live.tolist() == [33.] and update.call_args.kwargs["mamba_steps_to_track"] is None
