"""CPU-only unit tests for the per-path Mamba checkpoint cap."""

from sglang.srt.arg_groups.mamba_hook import handle_mamba_backend
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=13, stage="base-b", runner_config="1-gpu-small")

import unittest
from array import array
from collections import defaultdict
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import MambaEvictExcessPathStates
from sglang.srt.mem_cache.unified_cache.components.base import (
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.components.mamba import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedLRUList,
    UnifiedRadixCache,
    UnifiedTreeNode,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.kits.unified_radix_cache.config import (
    CacheConfig,
)
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures
from sglang.test.test_utils import CustomTestCase


class _FakeTreeCore:
    tree_components = (ComponentType.FULL, ComponentType.MAMBA)

    def __init__(self):
        self.root_node = UnifiedTreeNode(self.tree_components)
        self.evictable_device_leaves = set()
        self.mamba_path_cap_counts_host = False
        self.mamba_path_cap_anchor = False
        self.mamba_path_cap_count_evictable_only = False
        self.mamba_path_cap_audit = None
        self.component_evictable_size_ = {ComponentType.MAMBA: 0}
        self.component_protected_size_ = {ComponentType.MAMBA: 0}
        self.lru_lists = {
            ComponentType.MAMBA: UnifiedLRUList(
                ComponentType.MAMBA, self.tree_components
            )
        }
        self.host_lru_lists = {
            ComponentType.MAMBA: UnifiedLRUList(
                ComponentType.MAMBA, self.tree_components, use_host_ptr=True
            )
        }
        self.evicted = []
        self.cascaded = []
        # The real eviction helper reports unbacked FULL evictions to the
        # write-through drop counter; this fake never tracks a walk.
        self._is_tracking_unbacked_tokens = False
        self._tracked_unbacked_tokens = 0

    def _evict_component_and_detach_lru(self, node, component, *args, **kwargs):
        self.evicted.append(node)
        return UnifiedTreeCore._evict_component_and_detach_lru(
            self, node, component, *args, **kwargs
        )

    def _cascade_evict(
        self, node, component, tracker, device_frees, host_frees, cache_actions
    ):
        self.cascaded.append(node)


class _FakeUnifiedCache:
    tree_components = _FakeTreeCore.tree_components


def _build_unified_chain(cap, length=3):
    cache = _FakeUnifiedCache()
    core = _FakeTreeCore()
    component = object.__new__(MambaComponent)
    component.cache = cache
    component.tree_core = core
    component.mamba_max_states_per_path = cap

    nodes = []
    parent = core.root_node
    for index in range(length):
        node = UnifiedTreeNode(core.tree_components)
        node.parent = parent
        node.key = RadixKey(array("q", [index]))
        node.component_data[ComponentType.FULL].value = torch.tensor([100 + index])
        node.component_data[ComponentType.MAMBA].value = torch.tensor([index])
        parent.children[index] = node
        core.component_evictable_size_[ComponentType.MAMBA] += 1
        core.lru_lists[ComponentType.MAMBA].insert_mru(node)
        nodes.append(node)
        parent = node
    return component, nodes, core, cache


class TestMambaPathStateCap(unittest.TestCase):
    def test_server_arg_rejects_zero_and_values_below_negative_one(self):
        for value in (0, -2):
            args = ServerArgs(model_path="dummy", mamba_max_states_per_path=value)
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    ValueError,
                    "must be -1 \\(unlimited\\) or a positive integer",
                ),
            ):
                handle_mamba_backend(args)

    def test_unified_cache_removes_only_shallow_mamba_state(self):
        component, nodes, core, cache = _build_unified_chain(cap=2)

        device_frees = defaultdict(list)
        host_frees = defaultdict(list)
        cache_actions = []
        component._evict_excess_path_states(
            nodes[-1], device_frees, host_frees, cache_actions
        )
        self.assertEqual(cache_actions, [])

        self.assertEqual(core.evicted, [nodes[0]])
        self.assertEqual(core.cascaded, [nodes[0]])
        self.assertIsNone(nodes[0].component_data[ComponentType.MAMBA].value)
        self.assertIsNotNone(nodes[-1].component_data[ComponentType.MAMBA].value)
        self.assertEqual(
            [v.item() for v in device_frees[ComponentType.MAMBA]],
            [0],
        )
        self.assertEqual(core.component_evictable_size_[ComponentType.MAMBA], 2)
        self.assertFalse(core.lru_lists[ComponentType.MAMBA].in_list(nodes[0]))
        self.assertTrue(
            all(
                node.component_data[ComponentType.FULL].value is not None
                for node in nodes
            )
        )

    def test_unified_cache_cap_is_soft_for_fork_and_locked_nodes(self):
        component, nodes, core, cache = _build_unified_chain(cap=1, length=4)
        fork_child = UnifiedTreeNode(core.tree_components)
        fork_child.parent = nodes[0]
        nodes[0].children["fork"] = fork_child
        nodes[1].component_data[ComponentType.MAMBA].lock_ref = 1

        device_frees = defaultdict(list)
        host_frees = defaultdict(list)
        cache_actions = []
        component._evict_excess_path_states(
            nodes[-1], device_frees, host_frees, cache_actions
        )
        self.assertEqual(cache_actions, [])

        self.assertEqual(core.evicted, [nodes[2]])
        self.assertIsNotNone(nodes[0].component_data[ComponentType.MAMBA].value)
        self.assertIsNotNone(nodes[1].component_data[ComponentType.MAMBA].value)
        self.assertIsNone(nodes[2].component_data[ComponentType.MAMBA].value)
        self.assertIsNotNone(nodes[3].component_data[ComponentType.MAMBA].value)

    def test_count_evictable_only_spares_the_budget_of_forks_and_locked_nodes(self):
        # Six states (slot = depth - 1), cap 3: depth 1 is a fork shared with another
        # path and depth 3 is locked. Counting them leaves the path's own states two
        # slots; counting only evictable states (plus the tail) frees one, the first
        # victim of the same shallow-first order.
        for evictable_only, freed in ((False, [1, 3, 4]), (True, [1])):
            with self.subTest(evictable_only=evictable_only):
                component, nodes, core, cache = _build_unified_chain(cap=3, length=6)
                core.mamba_path_cap_count_evictable_only = evictable_only
                fork_child = UnifiedTreeNode(core.tree_components)
                fork_child.parent = nodes[0]
                nodes[0].children["fork"] = fork_child
                nodes[2].component_data[ComponentType.MAMBA].lock_ref = 1
                device_frees = defaultdict(list)
                component._evict_excess_path_states(
                    nodes[-1], device_frees, defaultdict(list), []
                )
                self.assertEqual(
                    [v.item() for v in device_frees[ComponentType.MAMBA]], freed
                )
        # Without forks or locks both rules free the same shallow states.
        for evictable_only in (False, True):
            component, nodes, core, cache = _build_unified_chain(cap=2, length=4)
            core.mamba_path_cap_count_evictable_only = evictable_only
            device_frees = defaultdict(list)
            component._evict_excess_path_states(
                nodes[-1], device_frees, defaultdict(list), []
            )
            self.assertEqual(
                [v.item() for v in device_frees[ComponentType.MAMBA]], [0, 1]
            )

    def test_unified_cache_preserves_existing_host_backup(self):
        component, nodes, core, cache = _build_unified_chain(cap=2)
        mamba_data = nodes[0].component_data[ComponentType.MAMBA]
        mamba_data.host_value = torch.tensor([10])

        device_frees = defaultdict(list)
        host_frees = defaultdict(list)
        cache_actions = []
        component._evict_excess_path_states(
            nodes[-1], device_frees, host_frees, cache_actions
        )
        self.assertEqual(cache_actions, [])

        self.assertIsNone(mamba_data.value)
        self.assertIsNotNone(mamba_data.host_value)
        self.assertTrue(core.host_lru_lists[ComponentType.MAMBA].in_list(nodes[0]))

    def _anchored_frees(self, cap, anchor):
        # Six one-token nodes, a state at each depth 1..6 (slot = depth - 1).
        component, nodes, core, cache = _build_unified_chain(cap=cap, length=6)
        core.mamba_path_cap_anchor = anchor
        device_frees = defaultdict(list)
        component._evict_excess_path_states(
            nodes[-1], device_frees, defaultdict(list), []
        )
        return sorted(v.item() for v in device_frees[ComponentType.MAMBA])

    def test_unified_cache_anchor_keeps_the_deepest_state_at_half_depth(self):
        # Tail depth 6: the newest cap-1 (depths 5, 6) plus the anchor at depth 3.
        self.assertEqual(self._anchored_frees(cap=3, anchor=True), [0, 1, 3])
        self.assertEqual(self._anchored_frees(cap=3, anchor=False), [0, 1, 2])

    def test_unified_cache_anchor_needs_cap_two(self):
        self.assertEqual(
            self._anchored_frees(cap=1, anchor=True),
            self._anchored_frees(cap=1, anchor=False),
        )

    def test_unified_cache_audit_records_holder_and_kept_depths(self):
        component, nodes, core, cache = _build_unified_chain(cap=3, length=6)
        core.mamba_path_cap_anchor = True
        core.mamba_path_cap_audit = []
        component._evict_excess_path_states(
            nodes[-1], defaultdict(list), defaultdict(list), []
        )
        self.assertEqual(
            core.mamba_path_cap_audit, [(6, [6, 5, 4, 3, 2, 1], [6, 5, 3])]
        )

    def test_audit_counts_walks_that_follow_the_anchor_rule(self):
        before = [6, 5, 4, 3, 2, 1]
        # The Rust adapter has no mamba_path_cap_anchor attribute: spec it out.
        core = mock.Mock(spec=["take_mamba_path_cap_audit"])
        core.take_mamba_path_cap_audit.return_value = [
            (6, before, [6, 5, 3]),
            (6, before, [6, 5, 4]),
        ]
        cache = SimpleNamespace(
            components={ComponentType.MAMBA: mock.Mock(mamba_max_states_per_path=3)},
            tree_core=core,
            mamba_path_cap_audit_counts=dict(walks=0, exact=0, other=0, moved=0),
            _mamba_path_cap_audit_every=100,
            _mamba_path_cap_anchor=True,
        )
        UnifiedRadixCache._audit_mamba_path_cap(cache)
        self.assertEqual(
            cache.mamba_path_cap_audit_counts,
            dict(walks=2, exact=1, other=1, moved=2),
        )

    def test_chunk_checkpoint_switch_keeps_all_none_or_the_last_chunk(self):
        chunk = 16384
        # remaining input after this chunk -> kept under all / none / last
        cases = {
            3 * chunk: (True, False, False),
            chunk: (True, False, True),
            0: (True, True, True),
            -5: (True, True, True),
        }
        for remaining, expected in cases.items():
            req = SimpleNamespace(
                origin_input_ids=[0] * 100000,
                extend_end=100000 - remaining,
                mamba_branching_seqlen=None,
            )
            for mode, keep in zip(("all", "none", "last"), expected):
                with (
                    self.subTest(remaining=remaining, mode=mode),
                    envs.SGLANG_MAMBA_CHUNK_CKPT.override(mode),
                    mock.patch(
                        "sglang.srt.mem_cache.unified_cache.components.mamba.get_schedule",
                        return_value=SimpleNamespace(chunked_prefill_size=chunk),
                    ),
                ):
                    self.assertEqual(
                        MambaComponent._skip_chunk_checkpoint(req, req.extend_end),
                        not keep,
                    )

    def test_unified_cache_anchor_falls_back_without_a_shallow_state(self):
        component, nodes, core, cache = _build_unified_chain(cap=2, length=6)
        core.mamba_path_cap_anchor = True
        for node in nodes[:3]:  # states only at depths 4, 5, 6
            node.component_data[ComponentType.MAMBA].value = None
        device_frees = defaultdict(list)
        component._evict_excess_path_states(
            nodes[-1], device_frees, defaultdict(list), []
        )
        self.assertEqual([v.item() for v in device_frees[ComponentType.MAMBA]], [3])

    def test_unified_cache_negative_one_disables_cap(self):
        component, nodes, core, cache = _build_unified_chain(cap=-1)

        device_frees = defaultdict(list)
        host_frees = defaultdict(list)
        cache_actions = []
        component._evict_excess_path_states(
            nodes[-1], device_frees, host_frees, cache_actions
        )
        self.assertEqual(cache_actions, [])

        self.assertEqual(dict(device_frees), {})
        self.assertEqual(core.evicted, [])
        self.assertTrue(
            all(
                node.component_data[ComponentType.MAMBA].value is not None
                for node in nodes
            )
        )


@unittest.skipUnless(torch.cuda.is_available(), "mamba pool fixtures need CUDA")
class TestMambaPathCapWriteThroughOrdering(UnifiedRadixCacheFixtures, CustomTestCase):
    """Write-through backup ordering regressions under the Mamba path-state cap."""

    cfg = CacheConfig(components=(ComponentType.FULL, ComponentType.MAMBA))

    def test_write_through_backup_survives_mamba_path_cap(self):
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        cache.write_through_threshold = 2
        cache.components[ComponentType.MAMBA].mamba_max_states_per_path = 1

        # The first insert creates the ancestor with a mamba state (hit_count 1).
        self._insert(cache, allocator, req_to_token_pool, [1, 2])
        ancestor = next(iter(cache.root_node.children.values()))
        self.assertIsNotNone(ancestor.component_data[ComponentType.MAMBA].value)

        # The extending insert crosses the ancestor's write-through threshold in
        # the same walk whose commit runs the path-cap eviction; the cap must
        # leave the pending-backup node's device state for the deferred BackupKV.
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        cache.writing_check(write_back=True)

        self.assertTrue(ancestor.backuped)
        self.assertIsNotNone(ancestor.component_data[ComponentType.MAMBA].host_value)

    def test_write_through_backup_chain_survives_mamba_path_cap(self):
        """A failed backup leaves an unbacked ancestor inside a later deferred
        backup chain; the cap walk must spare the whole chain, not just its tip."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        cache.write_through_threshold = 3
        mamba_comp = cache.components[ComponentType.MAMBA]

        self._insert(cache, allocator, req_to_token_pool, [1, 2])
        ancestor = next(iter(cache.root_node.children.values()))
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        middle = next(iter(ancestor.children.values()))

        # The ancestor crosses the threshold here; a host-exhaustion failure
        # leaves it unbacked with hit_count past the bar and its state intact.
        with mock.patch.object(cache, "_execute_kv_backup", return_value=None):
            self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4, 5, 6])
        self.assertFalse(ancestor.backuped)
        self.assertIsNotNone(ancestor.component_data[ComponentType.MAMBA].value)

        # The middle node crosses next, so the deferred chain is
        # [ancestor, middle]; the extending insert adopts a new leaf state,
        # firing the now-enabled cap walk before the chain executes — it must
        # not evict either chain node's device state.
        mamba_comp.mamba_max_states_per_path = 1
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4, 5, 6, 7, 8])
        cache.writing_check(write_back=True)

        self.assertTrue(ancestor.backuped)
        self.assertIsNotNone(ancestor.component_data[ComponentType.MAMBA].host_value)
        self.assertTrue(middle.backuped)
        self.assertIsNotNone(middle.component_data[ComponentType.MAMBA].host_value)

    def test_backup_retry_after_mamba_cap_skips_tombstoned_state(self):
        """A backup that fails before the cap and retries via the leaf action
        rebuilds its spec post-cap: KV backs up, the tombstoned mamba arm stays gone."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        cache.write_through_threshold = 1
        mamba_comp = cache.components[ComponentType.MAMBA]

        # A failed write-through leaves the ancestor unbacked with device state.
        with mock.patch.object(cache, "_execute_kv_backup", return_value=None):
            self._insert(cache, allocator, req_to_token_pool, [1, 2])
        ancestor = next(iter(cache.root_node.children.values()))
        self.assertFalse(ancestor.backuped)
        self.assertIsNotNone(ancestor.component_data[ComponentType.MAMBA].value)

        # The walk backup fails again, the cap tombstones the unlocked
        # ancestor's mamba state, then the leaf-action retry succeeds.
        mamba_comp.mamba_max_states_per_path = 1
        real_backup = cache._execute_kv_backup
        attempts = []

        def fail_once(*args, **kwargs):
            attempts.append(args)
            if len(attempts) == 1:
                return None
            return real_backup(*args, **kwargs)

        with mock.patch.object(cache, "_execute_kv_backup", side_effect=fail_once):
            self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        leaf = next(iter(ancestor.children.values()))
        cache.writing_check(write_back=True)

        # Post-cap spec rebuild: no resurrection of the tombstoned mamba state.
        self.assertTrue(ancestor.backuped)
        ancestor_cd = ancestor.component_data[ComponentType.MAMBA]
        self.assertIsNone(ancestor_cd.value)
        self.assertIsNone(ancestor_cd.host_value)
        self.assertTrue(leaf.backuped)
        self.assertIsNotNone(leaf.component_data[ComponentType.MAMBA].host_value)
        cache.sanity_check()

    def test_walk_backup_excludes_same_insert_restamped_mamba(self):
        """The walked target's backup executes before commit hooks, so a mamba
        value re-stamped by the same insert stays out of the host backup."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        mamba_comp = cache.components[ComponentType.MAMBA]

        self._insert(cache, allocator, req_to_token_pool, [1, 2])
        ancestor = next(iter(cache.root_node.children.values()))
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])

        # The cap walk tombstones the ancestor's mamba state.
        mamba_comp.mamba_max_states_per_path = 1
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4, 5, 6])
        self.assertIsNone(ancestor.component_data[ComponentType.MAMBA].value)

        # Re-inserting [1, 2] crosses the threshold and re-stamps the tombstone
        # in the same insert; the backup must not carry the fresh mamba state.
        cache.write_through_threshold = ancestor.hit_count + 1
        self._insert(cache, allocator, req_to_token_pool, [1, 2])
        cache.writing_check(write_back=True)

        self.assertTrue(ancestor.backuped)
        ancestor_cd = ancestor.component_data[ComponentType.MAMBA]
        self.assertIsNotNone(ancestor_cd.value)
        self.assertIsNone(ancestor_cd.host_value)

    def test_cap_walk_failure_still_drains_frees_and_actions(self):
        """A failed cap walk drains collected frees before its actions."""
        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        mamba_comp = cache.components[ComponentType.MAMBA]

        self._insert(cache, allocator, req_to_token_pool, [1, 2])
        ancestor = next(iter(cache.root_node.children.values()))
        self._insert(cache, allocator, req_to_token_pool, [1, 2, 3, 4])
        leaf = next(iter(ancestor.children.values()))
        self.assertIsNotNone(ancestor.component_data[ComponentType.MAMBA].value)

        mamba_comp.mamba_max_states_per_path = 1
        available = req_to_token_pool.mamba_allocator.available_size()
        action = mock.sentinel.eviction_action

        def fail_after_action(*args, **kwargs):
            kwargs["cache_actions"].append(action)
            raise RuntimeError("boom")

        applied_actions = []
        apply_cache_action = cache._apply_cache_action

        def check_action_after_free(received):
            if received is action:
                self.assertEqual(
                    req_to_token_pool.mamba_allocator.available_size(), available + 1
                )
                applied_actions.append(received)
            else:
                apply_cache_action(received)

        with (
            mock.patch.object(
                cache.tree_core, "_cascade_evict", side_effect=fail_after_action
            ),
            mock.patch.object(
                cache, "_apply_cache_action", side_effect=check_action_after_free
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                mamba_comp.apply_component_action(MambaEvictExcessPathStates(leaf.id))

        self.assertEqual(applied_actions, [action])
        self.assertIsNone(ancestor.component_data[ComponentType.MAMBA].value)
        self.assertEqual(
            req_to_token_pool.mamba_allocator.available_size(), available + 1
        )


if __name__ == "__main__":
    unittest.main()
