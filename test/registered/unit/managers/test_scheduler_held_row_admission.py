"""A request that already holds a req_to_token row is not counted twice at admission (#38118, #35537)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocation import paged_extend_num_tokens
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

NUM_ROWS = 4


class _Req:
    def __init__(self, holds_row):
        self.kv = SimpleNamespace(holds_kv=holds_row)
        self.extend_end = None
        self.beam_group = None
        self.token_indices_to_pool = None
        self.inflight_middle_chunks = 0

    def init_next_round_input(self, tree_cache=None):
        pass


class _Adder:
    """Admits whatever the gate lets through; the final chunk joins first."""

    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.rem_chunk_tokens = 0

    def add_chunked_req(self, req):
        self.can_run_list.append(req)
        return None

    def budget_state(self):
        return AddReqResult.CONTINUE

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE


class _BatchBuilt(Exception):
    pass


class TestHeldRowAdmission(CustomTestCase):
    def _scheduler(self, disaggregation_mode, running, final_chunk, waiting):
        s = Scheduler.__new__(Scheduler)
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.waiting_queue = waiting
        s.chunked_req = final_chunk
        s.min_free_slots_delayer = None
        s.policy = Mock()
        s.policy.shortest_prefill_chunk_limit.return_value = None
        s.processed_tokens_counter = 0
        s.chunked_prefill_size = 8192
        s.dynamic_chunk_sizer = None
        s.pp_min_chunked_prefill_size = None
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
        )
        s.get_prefill_adder_class = lambda: _Adder
        s.page_size = 1
        s.tree_cache = SimpleNamespace(
            buffer_pipeline=None, storage_prefetch_retries=None
        )
        s.token_to_kv_pool_allocator = None
        s.new_token_ratio_tracker = SimpleNamespace(current=0.0)
        s.max_prefill_tokens = 1 << 20
        s.is_mixed_chunk = False
        s.priority_scheduling_preemption_threshold = 0
        s.max_prefill_bs = 0
        s.max_running_requests = NUM_ROWS
        s.dllm_config = None
        s.enable_lora = False
        s.disaggregation_mode = disaggregation_mode
        s.disagg_prefill_bootstrap_queue = None
        s.enable_hicache_storage = False
        s.enable_lmcache = False
        s.truncation_align_size = None
        s.model_config = None
        s.enable_overlap = False
        s.spec_algorithm = None
        # The final chunk and the running requests hold rows; one row is free.
        s.req_to_token_pool = SimpleNamespace(
            available_size=lambda: NUM_ROWS - len(running) - 1
        )
        s.beam_coordinator = SimpleNamespace(pending_member_rows=lambda batch: 0)
        return s

    def _admitted(self, disaggregation_mode):
        running = [_Req(holds_row=True), _Req(holds_row=True)]
        final_chunk = _Req(holds_row=True)
        waiting = _Req(holds_row=False)
        running_batch = ScheduleBatch(reqs=running)
        s = self._scheduler(disaggregation_mode, running, final_chunk, [waiting])

        def capture(can_run_list, *args, **kwargs):
            raise _BatchBuilt(list(can_run_list))

        with (
            get_context().override_server_args(pp_max_micro_batch_size=NUM_ROWS),
            patch.object(ScheduleBatch, "init_new", side_effect=capture),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
        ):
            with self.assertRaises(_BatchBuilt) as built:
                s._get_new_batch_prefill_raw(
                    prefill_delayer_single_pass=None, running_batch=running_batch
                )
        return (
            built.exception.args[0],
            running_batch.batch_is_full,
            final_chunk,
            waiting,
        )

    def test_final_chunk_leaves_the_free_row_to_a_waiting_request(self):
        for mode in (DisaggregationMode.NULL, DisaggregationMode.PREFILL):
            with self.subTest(mode=mode):
                can_run_list, batch_is_full, final_chunk, waiting = self._admitted(mode)
                self.assertEqual(can_run_list, [final_chunk, waiting])
                self.assertFalse(batch_is_full)


PAGE = 512  # DCP8-widened allocator page (64 x 8)


class _PdReq(_Req):
    def __init__(self, prefix_len, fill_len, holds_row):
        super().__init__(holds_row)
        self.rid = f"pd-{prefix_len}"
        self.origin_input_ids = range(fill_len)
        self.prefix_len = prefix_len
        self.extend_end = prefix_len if holds_row else None
        self.fill_len = fill_len
        self.lock = None
        self.host_loaded_length = 0


class _PdAdder(_Adder):
    """Admits the chunk (or parks it) and one new request with a host load."""

    park = False
    memory_budget = SimpleNamespace(remaining_total=0)

    def add_chunked_req(self, req):
        if self.park:
            return req
        req.extend_end = req.prefix_len + 36224
        self.can_run_list.append(req)
        return req

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        req.prefix_len = 8192
        req.host_loaded_length = 8192
        req.extend_end = req.fill_len
        req.lock = "lock-" + str(id(req))
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE


class TestPdPrefillKvShortfall(CustomTestCase):
    """A PD prefill batch whose paged KV cannot be allocated is undone and
    retried, not raised as "Prefill out of memory" (K3.1 c640 jobs)."""

    CHUNK_DEMAND = 35840  # 36224 tokens; the first 384 fill the page at 39040
    NEW_DEMAND = 1024  # 1000 tokens after an aligned 8192-token loaded prefix

    def _pd_scheduler(self, *, available, park=False):
        chunked = _PdReq(39040, 39040 + 65536, holds_row=True)
        new = _PdReq(0, 8192 + 1000, holds_row=False)
        s = TestHeldRowAdmission._scheduler(
            self, DisaggregationMode.PREFILL, [], chunked, [new]
        )
        s.get_prefill_adder_class = lambda: type("Adder", (_PdAdder,), {"park": park})
        s.token_to_kv_pool_allocator = MagicMock(spec=PagedTokenToKVPoolAllocator)
        s.token_to_kv_pool_allocator.page_size = PAGE
        s.token_to_kv_pool_allocator.available_size.return_value = available
        s.tree_cache = MagicMock()
        s.tree_cache.buffer_pipeline = None
        s.tree_cache.storage_prefetch_retries = None
        s.tree_cache.supports_mamba.return_value = False
        s.tree_cache.evictable_size.return_value = 0
        s.tree_cache.ready_to_load_host_cache.return_value = 1
        s.enable_hierarchical_cache = True
        s.enable_unified_cache_external_linker = False
        s.prefill_kv_deferrals = 0
        s._revert_unadmitted_mamba = Mock()
        return s, chunked, new

    def _build(self, s):
        """The admitted list if a batch is built, else the method's result."""

        def capture(can_run_list, *args, **kwargs):
            raise _BatchBuilt(list(can_run_list))

        with (
            get_context().override_server_args(pp_max_micro_batch_size=NUM_ROWS),
            patch.object(ScheduleBatch, "init_new", side_effect=capture),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
        ):
            try:
                return s._get_new_batch_prefill_raw(
                    prefill_delayer_single_pass=None,
                    running_batch=ScheduleBatch(reqs=[]),
                )[0]
            except _BatchBuilt as built:
                return built.args[0]

    def test_demand_is_alloc_extend_page_count(self):
        self.assertEqual(
            paged_extend_num_tokens([39040], [39040 + 36224], PAGE), self.CHUNK_DEMAND
        )
        self.assertEqual(paged_extend_num_tokens([8192], [9192], PAGE), self.NEW_DEMAND)
        self.assertEqual(paged_extend_num_tokens([100], [300], PAGE), 0)

    def test_fitting_batch_is_built(self):
        s, chunked, new = self._pd_scheduler(
            available=self.CHUNK_DEMAND + self.NEW_DEMAND
        )
        self.assertEqual(self._build(s), [chunked, new])
        self.assertEqual(chunked.inflight_middle_chunks, 1)
        self.assertEqual(s.prefill_kv_deferrals, 0)

    def test_shortfall_defers_and_keeps_requests_queued(self):
        s, chunked, new = self._pd_scheduler(
            available=self.CHUNK_DEMAND + self.NEW_DEMAND - PAGE
        )
        self.assertIsNone(self._build(s))
        # The chunk resumes from the same boundary; the new request stays queued.
        self.assertIs(s.chunked_req, chunked)
        self.assertEqual(chunked.extend_end, 39040)
        self.assertEqual(chunked.inflight_middle_chunks, 0)
        self.assertEqual(s.waiting_queue, [new])
        self.assertIsNone(new.lock)
        self.assertIsNone(new.extend_end)
        self.assertEqual(new.host_loaded_length, 0)
        s.tree_cache.unlock.assert_called_once_with("lock-" + str(id(new)))
        s._revert_unadmitted_mamba.assert_called_once_with(new)
        # The queued load-back is run to completion before its rows are reused.
        s.tree_cache.ready_to_load_host_cache.assert_called_once()
        s.tree_cache.cache_controller.layer_done_counter.events[
            1
        ].finish_event.synchronize.assert_called_once()
        self.assertEqual(s.prefill_kv_deferrals, 1)

    def test_shortfall_raises_through_when_disabled(self):
        s, chunked, new = self._pd_scheduler(available=0)
        with envs.SGLANG_PREFILL_DEFER_ON_KV_SHORTFALL.override(False):
            self.assertEqual(self._build(s), [chunked, new])

    def test_evictable_tokens_count_as_free(self):
        s, chunked, new = self._pd_scheduler(available=0)
        s.tree_cache.evictable_size.return_value = self.CHUNK_DEMAND + self.NEW_DEMAND
        self.assertEqual(self._build(s), [chunked, new])

    def test_parked_chunk_admits_no_one(self):
        # #44823: no batch may carry a chunked request it does not run.
        s, chunked, new = self._pd_scheduler(available=1 << 20, park=True)
        with self.assertLogs("sglang.srt.managers.scheduler", "INFO") as logs:
            self.assertIsNone(self._build(s))
        self.assertIn("K31 park-chunked-prefill: parked", logs.output[0])
        self.assertIs(s.chunked_req, chunked)
        self.assertEqual(chunked.inflight_middle_chunks, 0)
        self.assertIsNone(new.extend_end)
        self.assertEqual(s.waiting_queue, [new])


if __name__ == "__main__":
    unittest.main()
