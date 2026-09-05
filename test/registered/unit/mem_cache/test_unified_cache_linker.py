from types import SimpleNamespace

import pytest
import torch

from array import array

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, InsertResult
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import (
    ReplaceWriteThroughOnNodeSplit,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinker,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeLinker(UnifiedCacheLinker):
    def __init__(self):
        self.layer_done_counter = object()
        self.restorable = []
        self.queued_loads = {}
        self.queued_offloads = []
        self.completed_loads = []
        self.completed_offloads = []
        self.reset_count = 0
        self.closed = False

    def lookup(self, rid, transfers):
        return list(self.restorable)

    def load(self, rid, transfers):
        self.queued_loads[rid] = list(transfers)
        return True

    def start_layer_wise_loading(self):
        return 3

    def cancel_queued_load(self, rid):
        if rid not in self.queued_loads:
            return False
        del self.queued_loads[rid]
        return True

    def num_completed_loads(self):
        return len(self.completed_loads)

    def pop_completed_load(self):
        return self.completed_loads.pop(0)

    def offload(self, transfers):
        self.queued_offloads.append(list(transfers))
        return True

    def num_completed_offloads(self):
        return len(self.completed_offloads)

    def pop_completed_offload(self):
        return self.completed_offloads.pop(0)

    def reset(self):
        self.reset_count += 1

    def close(self):
        self.closed = True


class _MappingRecorder:
    def __init__(self):
        self.mapping = []

    def set_full_to_swa_mapping(self, full, swa):
        self.mapping.append((full.clone(), swa.clone()))


def _cache_for_wrapper(**kwargs):
    defaults = {
        "tree_core": SimpleNamespace(enable_external_cache_linker=False),
        "write_through_threshold": 256,
        "pp_size": 1,
        "pp_group": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_cache_linker_attachment_is_backend_independent():
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.tree_core = SimpleNamespace(
        enable_external_cache_linker=False,
        write_through_threshold=256,
    )
    cache.linker = None
    linker = _FakeLinker()

    cache.init_cache_linker(linker)

    assert cache.linker.cache_linker is linker
    assert cache.tree_core.enable_external_cache_linker
    assert cache.write_through_threshold == 1
    assert cache.linker.layer_done_counter is linker.layer_done_counter


def test_restorable_prefix_intersects_sparse_rank_results():
    remote_mask = torch.tensor([0, 0, 1, 0, 0], dtype=torch.int)

    def intersect_remote_mask(mask, op):
        assert op == torch.distributed.ReduceOp.MIN
        mask.copy_(torch.minimum(mask, remote_mask))

    cache = _cache_for_wrapper(_all_reduce_attn_groups=intersect_remote_mask)
    wrapper = UnifiedCacheLinkerWrapper(cache, _FakeLinker())

    hit_pages = wrapper._sync_restorable_prefix([2, 4], num_pages=4, device_hit_pages=0)

    assert hit_pages == 2


def test_async_offload_pins_node_until_completion():
    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            assert phase == LinkerTransferPhase.OFFLOAD
            return PoolTransfer(name=PoolName.KV, keys=["page"])

    linker = _FakeLinker()
    lock_params = object()
    locks = []
    unlocks = []

    def inc_lock_ref(node):
        locks.append(node)
        return SimpleNamespace(to_dec_params=lambda: lock_params)

    node_id = 7
    node = SimpleNamespace(
        id=node_id,
        external_cache_stored=False,
        write_through_pending_id=None,
    )
    cache = _cache_for_wrapper(
        tree_core=SimpleNamespace(
            enable_external_cache_linker=False,
            mark_write_through_pending=lambda node_ids, ack_id: (
                setattr(node, "write_through_pending_id", ack_id) or list(node_ids)
            ),
        ),
        _components_tuple=(_Component(),),
        inc_lock_ref=inc_lock_ref,
        dec_lock_ref=lambda node, params: unlocks.append((node, params)),
        resolve_node_handle=lambda value: node if value == node_id else None,
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)

    wrapper.offload_nodes([node_id])

    assert locks == [node_id]
    assert node.external_cache_stored
    assert not unlocks

    linker.completed_offloads.append(False)
    completed = wrapper.take_completed_offloads(finish_count=1)
    wrapper.commit_completed_offloads(completed)

    assert not node.external_cache_stored
    assert unlocks == [(node_id, lock_params)]


def test_async_load_pins_node_until_completion():
    linker = _FakeLinker()
    lock_params = object()
    locks = []
    unlocks = []

    def inc_lock_ref(node):
        locks.append(node)
        return SimpleNamespace(to_dec_params=lambda: lock_params)

    node_id = 7
    cache = _cache_for_wrapper(
        inc_lock_ref=inc_lock_ref,
        dec_lock_ref=lambda node, params: unlocks.append((node, params)),
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)

    wrapper._queue_load("rid", node_id, [object()])

    assert locks == [node_id]
    assert not unlocks

    linker.completed_loads.append(["rid"])
    wrapper.drain_loads(finish_count=1)

    assert unlocks == [(node_id, lock_params)]


def test_release_request_cancels_queued_load():
    linker = _FakeLinker()
    lock_params = object()
    unlocks = []
    cache = _cache_for_wrapper(
        dec_lock_ref=lambda node, params: unlocks.append((node, params))
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.hit_markers["rid"] = object()
    wrapper.pending_loads["rid"] = (7, lock_params)
    linker.queued_loads["rid"] = [object()]

    wrapper.release_request("rid")

    assert wrapper.hit_markers == {}
    assert wrapper.pending_loads == {}
    assert "rid" not in linker.queued_loads
    assert unlocks == [(7, lock_params)]


def test_failed_offload_rolls_back_split_fragments():
    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            return PoolTransfer(name=PoolName.KV, keys=["page"])

    linker = _FakeLinker()
    lock_params = object()
    unlocks = []
    child = SimpleNamespace(
        id=7,
        external_cache_stored=False,
        write_through_pending_id=None,
    )
    parent = SimpleNamespace(
        id=8,
        external_cache_stored=False,
        write_through_pending_id=None,
    )
    nodes = {child.id: child, parent.id: parent}

    def mark_pending(node_ids, ack_id):
        for node_id in node_ids:
            nodes[node_id].write_through_pending_id = ack_id
        return list(node_ids)

    cache = _cache_for_wrapper(
        tree_core=SimpleNamespace(
            enable_external_cache_linker=False,
            mark_write_through_pending=mark_pending,
        ),
        _components_tuple=(_Component(),),
        inc_lock_ref=lambda node_id: SimpleNamespace(to_dec_params=lambda: lock_params),
        dec_lock_ref=lambda node_id, params: unlocks.append((node_id, params)),
        resolve_node_handle=nodes.__getitem__,
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.offload_nodes([child.id])

    parent.external_cache_stored = child.external_cache_stored
    parent.write_through_pending_id = child.write_through_pending_id
    wrapper.replace_pending_offload_node(child.id, child.id, [parent.id, child.id])
    linker.completed_offloads.append(False)
    wrapper.commit_completed_offloads(wrapper.take_completed_offloads(finish_count=1))

    assert not parent.external_cache_stored
    assert not child.external_cache_stored
    assert parent.write_through_pending_id is None
    assert child.write_through_pending_id is None
    assert unlocks == [(child.id, lock_params)]


def test_split_action_retargets_pending_external_offload():
    calls = []
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.linker = SimpleNamespace(
        replace_pending_offload_node=lambda *args: calls.append(("linker", *args))
    )
    cache._replace_pending_write_through_node = lambda *args: calls.append(
        ("hicache", *args)
    )
    action = ReplaceWriteThroughOnNodeSplit(
        ack_id=7,
        old_node_id=7,
        new_node_id=8,
        new_child_node_id=7,
    )

    cache._apply_cache_action(action)

    assert calls == [
        ("hicache", 7, 7, [8, 7]),
        ("linker", 7, 7, [8, 7]),
    ]


def test_reset_quiesces_backend_before_releasing_pending_locks():
    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            return PoolTransfer(name=PoolName.KV, keys=["page"])

    events = []

    class _QuiescentFakeLinker(_FakeLinker):
        def reset(self):
            events.append("backend")
            super().reset()

    linker = _QuiescentFakeLinker()
    node = SimpleNamespace(
        id=7,
        external_cache_stored=False,
        write_through_pending_id=None,
    )
    cache = _cache_for_wrapper(
        tree_core=SimpleNamespace(
            enable_external_cache_linker=False,
            mark_write_through_pending=lambda node_ids, ack_id: (
                setattr(node, "write_through_pending_id", ack_id) or list(node_ids)
            ),
        ),
        _components_tuple=(_Component(),),
        inc_lock_ref=lambda node_id: SimpleNamespace(to_dec_params=object),
        dec_lock_ref=lambda node_id, params: events.append(("unlock", node_id)),
        resolve_node_handle=lambda node_id: node,
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper._queue_load("rid", node.id, [object()])
    wrapper.offload_nodes([node.id])

    wrapper.reset()

    assert events == ["backend", ("unlock", node.id), ("unlock", node.id)]
    assert wrapper.pending_loads == {}
    assert wrapper.pending_offloads == []
    assert not node.external_cache_stored
    assert node.write_through_pending_id is None


def test_close_quiesces_backend_before_releasing_pending_loads():
    events = []

    class _ClosingFakeLinker(_FakeLinker):
        def close(self):
            events.append("backend")
            super().close()

    linker = _ClosingFakeLinker()
    cache = _cache_for_wrapper(
        dec_lock_ref=lambda node_id, params: events.append(("unlock", node_id))
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.pending_loads["rid"] = (7, object())

    wrapper.close()

    assert events == ["backend", ("unlock", 7)]
    assert linker.closed
    assert wrapper.pending_loads == {}


def test_check_hicache_events_commits_common_rank_results():
    committed = []
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.linker = SimpleNamespace(
        num_completed_loads=lambda: 1,
        drain_loads=lambda count: committed.append(("load", count)),
        num_completed_offloads=lambda: 3,
        take_completed_offloads=lambda count: [True] * count,
        commit_completed_offloads=committed.append,
    )

    reduce_calls = 0

    def reduce_to_common_state(value, op):
        nonlocal reduce_calls
        assert op == torch.distributed.ReduceOp.MIN
        reduce_calls += 1
        if reduce_calls == 1:
            value.copy_(torch.tensor([1, 1]))
        else:
            value.fill_(0)

    cache._all_reduce_attn_groups = reduce_to_common_state

    cache.check_hicache_events()

    assert committed == [("load", 1), [False]]


def test_component_commit_keeps_only_adopted_pages():
    mapping = _MappingRecorder()
    cache = _cache_for_wrapper(
        page_size=2,
        token_to_kv_pool_allocator=SimpleNamespace(
            set_full_to_swa_mapping=mapping.set_full_to_swa_mapping
        ),
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, _FakeLinker())
    full_component = FullComponent.__new__(FullComponent)
    full_component.cache = cache
    full_component.component_type = ComponentType.FULL
    swa_component = SWAComponent.__new__(SWAComponent)
    swa_component.cache = cache
    swa_component.component_type = ComponentType.SWA
    full = PoolTransfer(
        name=PoolName.KV,
        keys=["a", "b", "c", "d"],
        device_indices=torch.tensor([100, 101, 102, 103, 104, 105, 106, 107]),
    )
    canonical_tail = torch.tensor([10, 11, 102, 103, 14, 15, 106, 107])
    swa = PoolTransfer(
        name=PoolName.SWA,
        keys=["a", "b", "c", "d"],
        device_indices=torch.tensor([200, 201, 202, 203, 204, 205, 206, 207]),
    )
    insert_result = InsertResult(
        prefix_len=0,
        adopted_ranges={
            ComponentType.FULL: [(2, 4), (6, 8)],
            ComponentType.SWA: [(2, 4), (6, 8)],
        },
    )

    filtered = wrapper._update_load(
        ExternalLinkerLoadPhase.COMMIT,
        SimpleNamespace(),
        [(full_component, full), (swa_component, swa)],
        prefix_len=8,
        insert_result=insert_result,
        canonical_full=canonical_tail,
    )

    assert filtered == [full, swa]
    assert full.keys == ["b", "d"]
    assert full.device_indices.tolist() == [102, 103, 106, 107]
    assert swa.keys == ["b", "d"]
    assert swa.device_indices.tolist() == [202, 203, 206, 207]
    mapped_full, mapped_swa = mapping.mapping[0]
    assert mapped_full.tolist() == [102, 103, 106, 107]
    assert mapped_swa.tolist() == [202, 203, 206, 207]


class _MambaSlotAllocator:
    def __init__(self, slots):
        self.slots = slots
        self.freed = []

    def alloc(self, size):
        value = self.slots[:size].clone()
        self.slots = self.slots[size:]
        return value

    def free(self, value):
        self.freed.append(value.clone())


def test_mamba_component_external_linker_request_owned_load():
    class LRU:
        def __init__(self):
            self.nodes = set()

        def insert_mru(self, node):
            self.nodes.add(node.id)

        def in_list(self, node):
            return node.id in self.nodes

        def remove_node(self, node):
            self.nodes.remove(node.id)

    component = MambaComponent.__new__(MambaComponent)
    allocator = _MambaSlotAllocator(torch.tensor([7, 8, 9]))
    req_pool = SimpleNamespace(mamba_allocator=allocator, mamba_ckpt_pool=None)
    cd = SimpleNamespace(value=None, host_value=None)
    node = SimpleNamespace(
        id=11,
        hash_value=["a", "b"],
        component_data={ComponentType.MAMBA: cd},
    )
    device_lru = LRU()
    host_lru = LRU()
    tree_core = SimpleNamespace(
        lru_lists={ComponentType.MAMBA: device_lru},
        host_lru_lists={ComponentType.MAMBA: host_lru},
        component_evictable_size_={ComponentType.MAMBA: 0},
        _update_evictable_leaf_sets=lambda node: None,
        get_component_device_value=lambda node_id, component_type: cd.value,
    )
    component.cache = SimpleNamespace(
        req_to_token_pool=req_pool,
        evict=lambda params: None,
        resolve_node_handle=lambda node_id: node,
    )
    component.tree_core = tree_core
    component.mamba_max_states_per_path = -1

    lookup = component.build_external_linker_transfer(
        LinkerTransferPhase.LOOKUP, None, ["a", "b"]
    )
    assert lookup.keys == ["b"]
    assert lookup.hit_policy == PoolHitPolicy.TRAILING_PAGES

    transfer = component.build_external_linker_transfer(
        LinkerTransferPhase.LOAD, None, ["a", "b"]
    )
    req = SimpleNamespace(
        mamba_pool_idx=None,
        mamba_cow_src_index=None,
        mamba_needs_clear=True,
    )
    component.update_external_linker_load(
        ExternalLinkerLoadPhase.PREPARE,
        req,
        PoolTransfer(name=PoolName.KV),
        transfer,
        prefix_len=2,
    )
    assert req.mamba_pool_idx.item() == 7
    assert not req.mamba_needs_clear

    insert_result = InsertResult(
        prefix_len=0,
        last_device_node=node.id,
        adopted_ranges={},
    )
    component.commit_insert_component_data(
        node,
        True,
        InsertParams(
            key=RadixKey(array("q", [1, 2])),
            mamba_value=transfer.device_indices,
        ),
        insert_result,
        [],
    )
    assert insert_result.adopted_ranges == {ComponentType.MAMBA: [(1, 2)]}
    committed = component.update_external_linker_load(
        ExternalLinkerLoadPhase.COMMIT,
        req,
        PoolTransfer(name=PoolName.KV),
        transfer,
        prefix_len=2,
        insert_result=insert_result,
    )
    assert committed is transfer
    assert cd.value is None
    assert not device_lru.nodes
    assert tree_core.component_evictable_size_[ComponentType.MAMBA] == 0

    cd.value = torch.tensor([42])
    conflict = component.build_external_linker_transfer(
        LinkerTransferPhase.LOAD, None, ["c"]
    )
    component.update_external_linker_load(
        ExternalLinkerLoadPhase.PREPARE,
        req,
        PoolTransfer(name=PoolName.KV),
        conflict,
        prefix_len=3,
    )
    component.finalize_external_linker_insert(
        req,
        conflict,
        InsertResult(prefix_len=0, last_device_node=node.id, mamba_exist=True),
    )
    assert req.mamba_pool_idx.item() == 7
    assert req.mamba_cow_src_index.tolist() == [42]


def test_component_commit_handles_single_slot_mamba_with_large_tree_pages():
    class Component:
        def __init__(self, component_type):
            self.component_type = component_type
            self.canonical_full = "unset"

        def update_external_linker_load(
            self,
            phase,
            req,
            full_transfer,
            transfer,
            prefix_len,
            *,
            insert_result=None,
            canonical_full=None,
        ):
            self.canonical_full = canonical_full
            return transfer

    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = SimpleNamespace(page_size=64)
    full_component = Component(ComponentType.FULL)
    mamba_component = Component(ComponentType.MAMBA)
    full = PoolTransfer(
        name=PoolName.KV,
        keys=["page"],
        device_indices=torch.arange(64),
    )
    mamba = PoolTransfer(
        name=PoolName.MAMBA,
        keys=["page"],
        device_indices=torch.tensor([7]),
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )
    canonical = torch.arange(100, 164)
    insert_result = InsertResult(
        prefix_len=0,
        adopted_ranges={
            ComponentType.FULL: [(0, 64)],
            ComponentType.MAMBA: [(63, 64)],
        },
    )

    filtered = wrapper._update_load(
        ExternalLinkerLoadPhase.COMMIT,
        SimpleNamespace(),
        [(full_component, full), (mamba_component, mamba)],
        prefix_len=64,
        insert_result=insert_result,
        canonical_full=canonical,
    )

    assert filtered == [full, mamba]
    assert full.device_indices.tolist() == list(range(64))
    assert mamba.device_indices.tolist() == [7]
    assert torch.equal(full_component.canonical_full, canonical)
    assert mamba_component.canonical_full is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
