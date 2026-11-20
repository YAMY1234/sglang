"""The donated mamba checkpoint depth must land on the tree page.

DCP widens the tree page past the mamba chunk grid. A checkpoint picked on the
finer grid names a depth no radix node can carry, so it gets attached to the
preceding node and a later request resumes from a state that already covers
tokens past that node. The same happens when the depth is measured from a
prefix that a chunked prefill left on the scheduler page but off the widened
tree page: 64 + 512 = 576 is no 512-page boundary, and the finished request
then frees two kv-row segments that share the page holding 576.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
    _MambaRadixCacheV2TrackEntry,
)
from sglang.srt.managers.schedule_policy import (
    PrefillAdder,
    _PrefillAdmission,
    mamba_prefill_chunk_grid,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context, mamba_track_grid
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CHUNK = 64


def _prepare_extend(
    *,
    tree_page: int,
    prefix_len: int,
    extend_len: int,
    dcp_enabled: bool = False,
    next_idx: int = 0,
    branching: int | None = None,
    input_len: int | None = None,
) -> tuple[_MambaRadixCacheV2TrackEntry, Req]:
    """Run one extend through the tracker; return its track entry and the request."""
    server_args = ServerArgs(model_path="dummy", page_size=CHUNK)
    # The property would otherwise load the HF config for the dummy model.
    server_args._mamba_cache_chunk_size = CHUNK
    set_global_server_args_for_scheduler(server_args)

    sampling_params = SamplingParams(max_new_tokens=1)
    sampling_params.normalize(None)
    req = Req(
        rid="req",
        origin_input_text="",
        origin_input_ids=array("q", [1] * (input_len or prefix_len + extend_len)),
        sampling_params=sampling_params,
        vocab_size=128,
    )
    req.prefix_len = prefix_len
    req.extend_end = prefix_len + extend_len
    req.kv.mamba_next_track_idx = next_idx
    req.mamba_branching_seqlen = branching

    batch = ScheduleBatch(reqs=[req])
    batch.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(mamba_chunk_size=CHUNK)
    )
    batch.tree_cache = SimpleNamespace(page_size=tree_page)
    batch.req_to_token_pool = MagicMock()
    batch.req_to_token_pool.get_mamba_ping_pong_other_idx.return_value = 1 - next_idx

    with patch(
        "sglang.srt.managers.schedule_batch.get_parallel",
        return_value=SimpleNamespace(dcp_enabled=dcp_enabled),
    ):
        entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(req)
    return entry, req


def _track_seqlen(**kwargs) -> int | None:
    """The donated depth of one extend, or None when it donates no checkpoint."""
    entry, req = _prepare_extend(**kwargs)
    if not entry.track_mask:
        assert req.kv.mamba_last_track_seqlen is None
        return None
    return req.kv.mamba_last_track_seqlen


class TestMambaCheckpointDepth(unittest.TestCase):
    def test_widened_tree_page_moves_the_donated_depth_onto_it(self):
        # 4066 tokens past a 16384 prefix: the chunk grid would stop at 20416,
        # which a 256-token page cannot name.
        depth = _track_seqlen(
            tree_page=256,
            prefix_len=16384,
            extend_len=4066,
            dcp_enabled=True,
        )
        self.assertEqual(depth % 256, 0)
        self.assertEqual(depth, 20224)

    def test_unwidened_tree_page_keeps_the_chunk_grid(self):
        depth = _track_seqlen(tree_page=CHUNK, prefix_len=16384, extend_len=4066)
        self.assertEqual(depth, 20416)

    def test_dcp_depth_satisfies_tree_page_and_snapshot_grids(self):
        # (tree_page, prefix_len, extend_len) -> donated depth, or None when no
        # prefix-relative kernel snapshot lands on an absolute tree page.
        for tree_page, prefix_len, extend_len, expected in (
            (512, 64, 540, 512),
            (512, 448, 128, 512),
            (512, 64, 400, None),
            (512, 37, 540, None),
        ):
            with self.subTest(prefix_len=prefix_len, extend_len=extend_len):
                self.assertEqual(
                    _track_seqlen(
                        tree_page=tree_page,
                        prefix_len=prefix_len,
                        extend_len=extend_len,
                        dcp_enabled=True,
                    ),
                    expected,
                )

    def test_non_dcp_depth_remains_relative_to_chunked_prefill_prefix(self):
        # Active requests can stop off the absolute grid between prefill chunks.
        # Keep donating their prefix-relative snapshots when DCP is disabled.
        self.assertEqual(
            _track_seqlen(tree_page=16, prefix_len=4253, extend_len=16384),
            20637,
        )


class TestBranchingTrackInTheLastForward(unittest.TestCase):
    # A KV hit past the deepest state (branching point 6144) inside the forward
    # that ends the input: today the branching point is tracked and the insert
    # stops there; SGLANG_MAMBA_BRANCH_TRACK_PREFER_END tracks the input end.
    ARGS = dict(tree_page=CHUNK, prefix_len=4096, extend_len=4000)

    def test_the_branching_point_takes_the_last_forward_by_default(self):
        self.assertEqual(_track_seqlen(**self.ARGS, branching=6144), 6144)

    def test_prefer_end_tracks_the_input_end_in_the_last_forward(self):
        end = _track_seqlen(**self.ARGS)
        with envs.SGLANG_MAMBA_BRANCH_TRACK_PREFER_END.override(True):
            self.assertEqual(_track_seqlen(**self.ARGS, branching=6144), end)
        self.assertGreater(end, 6144)

    def test_prefer_end_keeps_the_branching_point_in_an_earlier_forward(self):
        with envs.SGLANG_MAMBA_BRANCH_TRACK_PREFER_END.override(True):
            self.assertEqual(
                _track_seqlen(**self.ARGS, branching=6144, input_len=40000), 6144
            )


class TestBranchTrackSplit(unittest.TestCase):
    # SGLANG_MAMBA_BRANCH_TRACK_SPLIT: a forward that would end the input with the
    # branching point (6144) inside is cut there, so both states get tracked.
    def _adder(self, new_chunked_req=None) -> PrefillAdder:
        adder = object.__new__(PrefillAdder)
        adder.page_size = CHUNK
        adder.new_chunked_req = new_chunked_req
        return adder

    def _req(self, branching, input_len=8096):
        return SimpleNamespace(
            mamba_branching_seqlen=branching, full_untruncated_fill_ids=[1] * input_len
        )

    def _split(self, adder, req, prefix_len=4096, extend_len=4000, has_chunked=False):
        with get_context().override_server_args(_mamba_cache_chunk_size=CHUNK):
            return adder._branch_split_len(req, prefix_len, extend_len, has_chunked)

    def test_off_by_default(self):
        self.assertIsNone(self._split(self._adder(), self._req(6144)))

    def test_the_last_forward_is_cut_at_the_branching_point(self):
        with envs.SGLANG_MAMBA_BRANCH_TRACK_SPLIT.override(True):
            self.assertEqual(self._split(self._adder(), self._req(6144)), 2048)

    def test_no_cut_when_not_needed_or_not_possible(self):
        with envs.SGLANG_MAMBA_BRANCH_TRACK_SPLIT.override(True):
            # an earlier forward (the input continues past this one) already tracks it
            self.assertIsNone(self._split(self._adder(), self._req(6144, 40000)))
            self.assertIsNone(self._split(self._adder(), self._req(None)))
            self.assertIsNone(self._split(self._adder(), self._req(6100)))  # off-grid
            self.assertIsNone(self._split(self._adder(), self._req(4096 + 32)))
            # another request already holds this batch's one chunked slot
            self.assertIsNone(self._split(self._adder(object()), self._req(6144)))
            # or the scheduler's chunked request is still unfinished
            self.assertIsNone(
                self._split(self._adder(), self._req(6144), has_chunked=True)
            )

    def test_the_cut_forward_tracks_k_and_the_next_one_the_input_end(self):
        args = dict(tree_page=CHUNK, input_len=8096)
        self.assertEqual(
            _track_seqlen(prefix_len=4096, extend_len=2048, branching=6144, **args),
            6144,
        )
        end = _track_seqlen(prefix_len=6144, extend_len=1952, branching=6144, **args)
        self.assertEqual(end, _track_seqlen(prefix_len=6144, extend_len=1952, **args))
        self.assertGreater(end, 6144)

    def test_a_split_admission_holds_the_chunked_slot_and_closes_the_batch(self):
        adder = self._adder()
        adder.rem_chunk_tokens = 16384
        adder.can_run_list = []
        adder.tree_cache = MagicMock()
        for name in (
            "_update_prefill_budget",
            "_reserve_chunked_tail",
            "_account_prefill_cache_admission",
            "_request_slot_reserve_for_req",
        ):
            setattr(adder, name, MagicMock(return_value=0))
        req = SimpleNamespace(prefix_len=4096, last_node=None, retracted_stain=False)
        admission = _PrefillAdmission(4096, 2048, 0, True, 0, branch_split=True)
        adder._commit_prefill_admission(req, admission, mamba_gap_reserve=0)
        self.assertEqual(req.extend_end, 6144)
        self.assertIs(adder.new_chunked_req, req)
        self.assertEqual(adder.rem_chunk_tokens, 0)

    def test_chunk_skip_keeps_the_chunk_cut_at_the_branching_point(self):
        from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent

        req = SimpleNamespace(
            origin_input_ids=[1] * 8096, extend_end=6144, mamba_branching_seqlen=6144
        )
        with envs.SGLANG_MAMBA_CHUNK_CKPT.override("none"):
            self.assertFalse(MambaComponent._skip_chunk_checkpoint(req))
            req.mamba_branching_seqlen = None
            self.assertTrue(MambaComponent._skip_chunk_checkpoint(req))


class TestChunkedPrefillStaysOnTheChunkGrid(unittest.TestCase):
    """The scheduler side of the same constraint.

    `test_dcp_depth_satisfies_tree_page_and_snapshot_grids` shows a prefix of
    37 donates nothing. A prefix only reaches 37 because a prefill chunk was
    cut there, and every later chunk of that request inherits the offset, so
    one unaligned cut costs the request every checkpoint it would ever donate.
    """

    def _floor(self, trunc_len: int, remaining: int, grid: int = CHUNK) -> int:
        adder = object.__new__(PrefillAdder)
        adder.mamba_chunk_grid = grid
        return adder.floor_chunked_tokens(trunc_len, remaining)

    def test_a_cut_is_rounded_down_onto_the_grid(self):
        self.assertEqual(self._floor(100, 8192), 64)
        self.assertEqual(self._floor(8010, 100000), 8000)

    def test_a_cut_already_on_the_grid_is_untouched(self):
        self.assertEqual(self._floor(8192, 100000), 8192)

    def test_the_chunk_that_finishes_the_request_may_stop_anywhere(self):
        self.assertEqual(self._floor(37, 37), 37)

    def test_a_cut_shorter_than_the_grid_still_makes_progress(self):
        # Rounding to zero would stall the request forever; the donation is
        # the cheaper thing to lose.
        self.assertEqual(self._floor(40, 8192), 40)

    def test_no_rounding_without_the_mamba_extra_buffer(self):
        self.assertEqual(self._floor(100, 8192, grid=1), 100)

    def _published_grid(self, *, dcp_size: int, strategy: str = "extra_buffer") -> int:
        with get_context().override_server_args(
            disable_radix_cache=False,
            mamba_radix_cache_strategy=strategy,
            dcp_size=dcp_size,
            _mamba_cache_chunk_size=CHUNK,
        ):
            return mamba_prefill_chunk_grid()

    def test_dcp_extra_buffer_uses_the_chunk_grid(self):
        self.assertEqual(self._published_grid(dcp_size=2), CHUNK)
        self.assertEqual(
            self._floor(100, 8192, grid=self._published_grid(dcp_size=2)), 64
        )

    def test_non_dcp_extra_buffer_leaves_the_cut(self):
        # #40907 already donates from a prefix-relative snapshot here, so the
        # cut must stay where the prefill budget put it.
        self.assertEqual(self._published_grid(dcp_size=1), 1)
        self.assertEqual(
            self._floor(100, 8192, grid=self._published_grid(dcp_size=1)), 100
        )

    def test_non_dcp_does_not_change_truncation_alignment(self):
        # The old gate would write the chunk size into a still-empty alignment.
        scheduler = object.__new__(Scheduler)
        scheduler.truncation_align_size = None
        with get_context().override_server_args(
            disable_radix_cache=False,
            mamba_radix_cache_strategy="extra_buffer",
            dcp_size=1,
            _mamba_cache_chunk_size=CHUNK,
        ):
            scheduler.init_mamba_truncation_align()
        self.assertIsNone(scheduler.truncation_align_size)

    def test_dcp_folds_the_chunk_into_truncation_alignment(self):
        scheduler = object.__new__(Scheduler)
        scheduler.truncation_align_size = 48
        with get_context().override_server_args(
            disable_radix_cache=False,
            mamba_radix_cache_strategy="extra_buffer",
            dcp_size=2,
            _mamba_cache_chunk_size=CHUNK,
        ):
            scheduler.init_mamba_truncation_align()
        self.assertEqual(scheduler.truncation_align_size, 192)

    def test_every_chunk_of_a_cut_up_request_can_still_donate(self):
        # Walk a 100k prompt through chunks the scheduler cut to whatever the
        # shared prefill budget had left, and check each one donates.
        prefix_len, total = 0, 100_000
        for budget in (8192, 5000, 8192, 3333, 8192, 777):
            trunc_len = self._floor(budget, total - prefix_len)
            self.assertIsNotNone(
                _track_seqlen(
                    tree_page=256,
                    prefix_len=prefix_len,
                    extend_len=trunc_len,
                    dcp_enabled=True,
                ),
                f"no donation at {prefix_len=} {trunc_len=}",
            )
            prefix_len += trunc_len


class TestMambaTrackPosition(unittest.TestCase):
    def test_track_position_is_taken_before_the_ping_pong_swap(self):
        # prepare_for_extend gathers the slot id at this position; the read it
        # replaced took buffer[mamba_next_track_idx] before the boundary swap.
        for next_idx in (0, 1):
            with self.subTest(next_idx=next_idx):
                entry, req = _prepare_extend(
                    tree_page=CHUNK, prefix_len=0, extend_len=96, next_idx=next_idx
                )
                self.assertTrue(entry.track_mask)
                self.assertEqual(entry.track_position, next_idx)
                self.assertEqual(req.kv.mamba_next_track_idx, 1 - next_idx)


class TestMambaTrackGrid(unittest.TestCase):
    def _grid(self, *, interval: int, tree_page: int, chunk: int = CHUNK) -> int:
        with get_context().override_server_args(
            mamba_track_interval=interval, _mamba_cache_chunk_size=chunk
        ):
            return mamba_track_grid(tree_page)

    def test_widened_tree_page_rounds_the_interval_up(self):
        self.assertEqual(self._grid(interval=256, tree_page=512), 512)

    def test_grid_stays_on_the_chunk_size(self):
        self.assertEqual(self._grid(interval=192, tree_page=64, chunk=128) % 128, 0)


if __name__ == "__main__":
    unittest.main()
