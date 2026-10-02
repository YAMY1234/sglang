"""CPU ownership regression for prefill-only hybrid cache retention."""

import argparse
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.arg_groups.fields.memory import Memory
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache import common
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.unified_cache.components import mamba
from sglang.srt.speculative import spec_utils
from sglang.srt.server_args import ServerArgs


class Slots:
    def __init__(self):
        self.live = {1, 2}
        self.next_id = 3

    def alloc(self, n):
        ids = torch.arange(self.next_id, self.next_id + n)
        self.next_id += n
        self.live.update(ids.tolist())
        return ids

    def free(self, ids):
        for i in ids.tolist():
            assert i in self.live, f"double free or invalid slot {i}"
            self.live.remove(i)


def fixture():
    pool = object.__new__(HybridReqToTokenPool)
    pool.mamba_allocator = Slots()
    pool.enable_mamba_extra_buffer = pool.enable_mamba_extra_buffer_lazy = True
    pool.mamba_ping_pong_track_buffer_size = 2
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.tensor([[2, -1]])
    pool.free = Mock()
    req = NS(
        kv=ReqKvInfo(
            req_pool_idx=0, kv_committed_len=4, kv_allocated_len=4,
            mamba_pool_idx=torch.tensor(1),
            mamba_ping_pong_track_buffer=torch.tensor([2, -1]),
            mamba_next_track_idx=0, mamba_last_track_idx=0,
            mamba_last_track_seqlen=4,
        ),
        origin_input_ids=[10, 11, 12, 13], output_ids=[14],
        skip_radix_cache_insert=False,
    )
    req.owned_kv_len = lambda: req.kv.kv_committed_len
    tree = NS(
        enable_mamba_extra_buffer=True, req_to_token_pool=pool,
        supports_mamba=lambda: True, claim_kv_row=lambda req: False,
        token_to_kv_pool_allocator=NS(page_size=1),
        free_kv_row=Mock(), unpin=Mock(), prefixes={},
    )
    component = object.__new__(mamba.MambaComponent)
    component.cache = tree
    component._alloc_mamba_slot = lambda: pool.mamba_allocator.alloc(1)

    def insert(req, up_to, finished):
        params = InsertParams()
        length = component.prepare_for_caching_req(req, params, up_to, finished)
        if length:
            tree.prefixes[length] = params.mamba_value.item()
            req.kv.cache_protected_len = length
        component.cleanup_after_caching_req(
            req, finished,
            insert_result=NS(mamba_exist=False) if length else None,
            insert_params=params,
        )

    tree.cache_unfinished_req = Mock(
        side_effect=lambda req, **kw: insert(req, req.owned_kv_len(), False)
    )
    tree.insert_req = Mock(side_effect=lambda req, up_to: insert(req, up_to, True))
    tree.on_release = lambda req, inserted: (
        None if inserted else component.cleanup_after_caching_req(req, True)
    )
    return req, tree, pool, component


@pytest.mark.parametrize("enabled", [False, True])
def test_prefill_retained_decode_released_without_double_free(enabled):
    req, tree, pool, component = fixture()
    req.origin_input_ids = list(range(8))
    memory = NS(radix_cache_skip_decode_insert=enabled)
    with patch.object(common, "get_memory", return_value=memory), \
         patch.object(mamba, "get_memory", return_value=memory), \
         patch.object(common, "get_spec", return_value=NS(speculative_algorithm="EAGLE3")):
        common.maybe_cache_unfinished_req(req, tree, chunked=True)
        assert tree.prefixes == {4: 2}
        assert pool.mamba_allocator.live == {1, 2, 3}
        # The final prefill checkpoint is donated separately from the live state.
        req.kv.mamba_last_track_seqlen = 8
        req.kv.kv_committed_len = req.kv.kv_allocated_len = 8
        common.maybe_cache_unfinished_req(req, tree)
        retained = set(tree.prefixes.values())
        if enabled:
            assert pool.mamba_allocator.live == {1} | retained
            assert req.kv.mamba_ping_pong_track_buffer is None
        else:
            req.kv.mamba_last_track_seqlen = 10
        req.output_ids.extend([15, 16])
        req.kv.kv_committed_len, req.kv.kv_allocated_len = 10, 12
        common.release_kv_cache(req, tree)
        assert sorted(tree.prefixes) == ([4, 8] if enabled else [4, 8, 10])
        assert pool.mamba_allocator.live == set(tree.prefixes.values())
        assert not req.kv.holds_mamba and req.kv.is_kv_released
        tree.insert_req.assert_not_called() if enabled else tree.insert_req.assert_called_once()
        tree.unpin.assert_called_once_with(req)
        tree.free_kv_row.assert_any_call(req.kv, [(8 if enabled else 10, 10)])
        tree.free_kv_row.assert_any_call(req.kv, [(10, 12)])


def test_prefill_eos_still_publishes_prefix_and_retraction_does_not_publish_output():
    memory = NS(radix_cache_skip_decode_insert=True)
    with patch.object(common, "get_memory", return_value=memory), \
         patch.object(mamba, "get_memory", return_value=memory), \
         patch.object(common, "get_spec", return_value=NS(speculative_algorithm="EAGLE3")):
        req, tree, pool, component = fixture()
        common.release_kv_cache(req, tree)
        assert tree.prefixes == {4: 2}
        assert pool.mamba_allocator.live == {2}
        req, tree, pool, component = fixture()
        req.kv.mamba_last_track_seqlen = 8
        params = InsertParams()
        assert component.prepare_for_caching_req(req, params, 8, False) == 0
        assert params.mamba_value is None
        assert pool.mamba_allocator.live == {1, 2}


def test_default_off_and_verify_keeps_active_commit_without_checkpoint():
    assert Memory().radix_cache_skip_decode_insert is False
    batch = NS(mamba_track_indices=torch.tensor([9]), mamba_track_mask=torch.tensor([True]))
    with patch.object(spec_utils, "get_memory", return_value=NS(radix_cache_skip_decode_insert=True)):
        spec_utils.prepare_mamba_track_for_verify(batch)
    assert batch.mamba_track_indices is batch.mamba_track_mask is None
    active, checkpoint = spec_utils._verify_commit_step_indices(
        batch=batch, accept_index=torch.tensor([[0, 1, 2, 3]]),
        accept_lens=torch.tensor([3]), draft_token_num=4,
    )
    assert active.tolist() == [2] and checkpoint is None


def test_flag_is_exposed_by_native_cli():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "dummy", "--radix-cache-skip-decode-insert"])
    assert args.radix_cache_skip_decode_insert is True
