from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _Req:
    def __init__(self, *, inflight_middle_chunks: int, allocated: bool = True):
        self.rid = "aborted-prefill"
        self.cache_request_handle = CacheRequestHandle(self.rid, 0)
        self.inflight_middle_chunks = inflight_middle_chunks
        self.kv = ReqKvInfo(
            req_pool_idx=1 if allocated else None,
            kv_allocated_len=1 if allocated else 0,
            mamba_pool_idx=object() if allocated else None,
        )
        self.metadata_buffer_index = 7 if allocated else -1
        self.pending_bootstrap = allocated
        self.disagg_kv_sender = Mock()
        self.to_finish = FINISH_ABORT() if allocated else None
        self.finished_reason = None if allocated else FINISH_ABORT()
        self.discard_output_reason = None
        self.return_logprob = False
        self.return_sampling_mask = False
        self.grammar = None
        self.output_ids = []
        self.origin_input_ids = list(range(100))
        self.extend_end = None
        self.time_stats = SimpleNamespace(
            set_prefill_finished_time=Mock(),
            set_last_chunked_prefill_finish_time=Mock(),
            set_completion_time=Mock(),
        )

    def finished(self):
        return self.finished_reason is not None

    def update_finish_state(self):
        self.finished_reason = self.to_finish
        self.to_finish = None


class _Scheduler(SchedulerDisaggregationPrefillMixin):
    def __init__(self):
        self.batch_result_processor = SimpleNamespace(
            snapshot_auxiliary_output_starts=Mock(return_value=[]),
            move_logprobs_to_cpu=Mock(),
            consume_auxiliary_output=Mock(),
        )
        self.spec_algorithm = SimpleNamespace(is_eagle=lambda: False)
        self.tree_cache = Mock()
        self.disagg_prefill_inflight_queue = []
        self.disagg_prefill_pending_chunk_rids = {"aborted-prefill"}
        self.send_kv_chunk = Mock()
        self.output_streamer = Mock()
        self.metrics_reporter = SimpleNamespace(report_prefill_stats=Mock())
        self.maybe_send_health_check_signal = Mock()
        self.req_to_metadata_buffer_idx_allocator = Mock()
        self.enable_hicache_storage = True
        self.chunked_req = None


def _batch(req):
    return SimpleNamespace(
        reqs=[req],
        spec_info=None,
        prefill_stats=None,
        dp_cooperation_info=None,
    )


def _result():
    return GenerationBatchResult(next_token_ids=torch.tensor([11]))


def _free_req(req, _tree_cache, *, checkpoint):
    assert checkpoint is False
    req.kv.req_pool_idx = None
    req.kv.mark_kv_released()
    req.kv.mamba_pool_idx = None


class TestPrefillCompleteResult(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            disaggregation_decode_allocation_policy="prefill_complete"
        )
        override.install()
        self.addCleanup(override.restore)

    @patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache")
    def test_ready_observes_accepted_token_and_speculative_sidecars(self, cache):
        scheduler = _Scheduler()
        scheduler.spec_algorithm = SimpleNamespace(is_eagle=lambda: True)
        req = _Req(inflight_middle_chunks=0)
        req.to_finish = None
        req.time_stats.set_prefill_transfer_queue_entry_time = Mock()
        draft = SimpleNamespace(
            hidden_states=torch.tensor([[1.0, 2.0]]),
            topk_p=torch.tensor([[0.75, 0.25]]),
            topk_index=torch.tensor([[3, 4]]),
            dsa_topk_indices=torch.tensor([[5, 6]]),
        )
        batch = _batch(req)
        batch.spec_info = draft
        result = GenerationBatchResult(
            next_token_ids=torch.tensor([11]), next_draft_input=draft
        )
        observed = []

        def ready():
            observed.append(
                (
                    list(req.output_ids),
                    req.hidden_states_tensor.tolist(),
                    req.output_dsa_topk_indices.tolist(),
                    req in scheduler.disagg_prefill_inflight_queue,
                )
            )

        req.disagg_kv_sender.mark_prefill_complete.side_effect = ready
        scheduler.process_batch_result_disagg_prefill(batch, result)
        self.assertEqual(observed, [([11], [1.0, 2.0], [5, 6], True)])
        self.assertTrue(req.pending_bootstrap)
        scheduler.send_kv_chunk.assert_not_called()

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
    def test_aborted_final_result_never_publishes_ready(self, release):
        scheduler = _Scheduler()
        req = _Req(inflight_middle_chunks=0)
        scheduler.process_batch_result_disagg_prefill(_batch(req), _result())
        self.assertTrue(req.finished())
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        req.disagg_kv_sender.mark_prefill_complete.assert_not_called()

    @patch("sglang.srt.disaggregation.prefill.should_force_retry", return_value=True)
    def test_discarded_attempt_never_publishes_ready(self, force_retry):
        scheduler = _Scheduler()
        req = _Req(inflight_middle_chunks=0)
        req.to_finish = None
        retried = []
        scheduler.optimistic_release_and_requeue = lambda r: retried.append(r)
        scheduler.process_batch_result_disagg_prefill(_batch(req), _result())
        self.assertEqual(retried, [req])
        self.assertEqual(req.output_ids, [])
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        req.disagg_kv_sender.mark_prefill_complete.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


class TestEarlySentLastChunkResult(CustomTestCase):
    """SGLANG_PP_PD_PREFILL_EARLY_SEND: a PP stage already sent the last chunk when
    its forward completed; the result still records the request but sends nothing."""

    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            disaggregation_decode_allocation_policy="early"
        )
        override.install()
        self.addCleanup(override.restore)

    @patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache")
    def test_result_skips_the_early_sent_send(self, checkpoint):
        for early_sent in (True, False):
            with self.subTest(early_sent=early_sent):
                scheduler = _Scheduler()
                del scheduler.send_kv_chunk  # the real one, over a stub transfer
                scheduler._send_kv_chunk = Mock()
                scheduler.kv_checksum_computer = None
                req = _Req(inflight_middle_chunks=0)
                req.to_finish = None
                req.pending_bootstrap = False
                req.bootstrap_room = 5
                req.time_stats.set_prefill_transfer_queue_entry_time = Mock()
                order = []
                status = SimpleNamespace(
                    get=lambda room: order.append("transfer ended") or KVPoll.Success
                )
                req.disagg_kv_sender.kv_mgr = SimpleNamespace(
                    request_status=status, _staging_outstanding={}
                )
                scheduler.disagg_prefill_early_sent_rids = (
                    {req.rid} if early_sent else set()
                )
                # Its transfer must have ended before the checkpoint can dedupe:
                # the PP loop settles the batch's early sends first.
                checkpoint.side_effect = lambda *a: order.append("checkpoint")
                batch = _batch(req)
                scheduler._pp_pd_settle_early_sends(batch)
                scheduler.process_batch_result_disagg_prefill(batch, _result())
                self.assertEqual(req.output_ids, [11])
                self.assertEqual(scheduler.disagg_prefill_inflight_queue, [req])
                self.assertEqual(scheduler.disagg_prefill_early_sent_rids, set())
                if early_sent:
                    self.assertEqual(order[:2], ["transfer ended", "checkpoint"])
                    scheduler._send_kv_chunk.assert_not_called()
                else:
                    self.assertEqual(order, ["checkpoint"])
                    scheduler._send_kv_chunk.assert_called_once_with(
                        req, last_chunk=True, end_idx=None
                    )
