import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_batch() -> tuple[Req, ScheduleBatch]:
    sampling_params = SamplingParams(max_new_tokens=32)
    sampling_params.normalize(None)
    req = Req(
        rid="req",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2]),
        sampling_params=sampling_params,
        vocab_size=128,
    )
    req.output_ids.append(3)
    req.kv_committed_len = 2

    batch = ScheduleBatch(reqs=[req])
    batch.device = "cpu"
    batch.model_config = SimpleNamespace(is_encoder_decoder=False)
    batch.enable_overlap = True
    batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    batch.sampling_info = SimpleNamespace(
        penalizer_orchestrator=SimpleNamespace(is_required=False)
    )
    batch.hisparse_coordinator = None
    batch.seq_lens = torch.tensor([2], dtype=torch.int64)
    batch.seq_lens_cpu = torch.tensor([2], dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor([2], dtype=torch.int32)
    return req, batch


def _make_processor() -> SchedulerBatchResultProcessor:
    metrics_reporter = MagicMock()
    metrics_reporter.num_generated_tokens = 0
    metrics_reporter.forward_ct_decode = 0
    tree_cache = MagicMock()
    tree_cache.is_chunk_cache.return_value = False
    allocator = MagicMock()
    allocator.device = torch.device("cpu")
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=True,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(),
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=allocator,
        tree_cache=tree_cache,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=metrics_reporter,
        draft_worker=None,
        model_worker=MagicMock(),
        logprob_result_processor=None,
        output_streamer=MagicMock(),
        abort_request=lambda *args, **kwargs: None,
    )


def _make_result():
    return SimpleNamespace(
        copy_done=None,
        routed_experts_output=None,
        indexer_topk_output=None,
        logits_output=SimpleNamespace(hidden_states=None, customized_info=None),
        next_token_ids=[4],
        can_run_cuda_graph=False,
        num_correct_drafts=0,
        num_block_accept_tokens=0,
        num_cap_tokens=0,
        speculative_num_draft_tokens=0,
    )


class TestMambaBoundaryMaskReuse(unittest.TestCase):
    def test_overlap_scheduler_handles_zero_and_one_batch_lookahead(self):
        for schedule_next_decode, expected_lookahead in ((False, 0), (True, 1)):
            with self.subTest(schedule_next_decode=schedule_next_decode):
                req, batch = _make_batch()
                processor = _make_processor()
                result = _make_result()

                scheduler = Scheduler.__new__(Scheduler)
                scheduler.gracefully_exit = False
                scheduler.request_receiver = MagicMock()
                scheduler.request_receiver.recv_requests.side_effect = [
                    [],
                    [],
                    StopIteration,
                ]
                scheduler.process_input_requests = MagicMock()
                scheduler._engine_paused = False
                scheduler.running_batch = batch
                scheduler.is_disable_overlap_for_batch = MagicMock(return_value=False)
                scheduler.run_batch = MagicMock(return_value=result)
                scheduler._apply_war_barrier = MagicMock()
                scheduler.is_generation = False
                scheduler.last_batch = None

                plan_count = 0

                def get_next_batch_to_run(*, running_batch, last_batch):
                    nonlocal plan_count
                    del running_batch, last_batch
                    plan_count += 1
                    if plan_count == 1:
                        batch.prepare_for_decode()
                        return SimpleNamespace(
                            running_batch=batch,
                            batch_to_run=batch,
                        )
                    if plan_count == 2 and schedule_next_decode:
                        batch.prepare_for_decode()
                        return SimpleNamespace(
                            running_batch=batch,
                            batch_to_run=batch,
                        )
                    return SimpleNamespace(
                        running_batch=batch,
                        batch_to_run=None,
                    )

                scheduler.get_next_batch_to_run = get_next_batch_to_run
                observed_lookahead = []

                def process_batch_result(result_batch, batch_result):
                    observed_lookahead.append(
                        req.decode_batch_idx
                        - result_batch.mamba_decode_batch_idx_cpu[0]
                    )
                    processor.process_batch_result_decode(result_batch, batch_result)

                scheduler.process_batch_result = process_batch_result

                with (
                    # The mamba predicates and the track interval read the
                    # published bags, so publish the configuration under test
                    # (non-lazy extra buffer, interval 4); observability and
                    # disagg reads are served by the same publish at their
                    # defaults.
                    get_context().override_server_args(
                        mamba_radix_cache_strategy="extra_buffer",
                        mamba_track_interval=4,
                    ),
                    patch(
                        "sglang.srt.managers.schedule_batch.alloc_for_decode",
                        return_value=torch.tensor([3], dtype=torch.int64),
                    ),
                    patch(
                        "sglang.srt.managers.schedule_batch.set_mamba_track_indices_from_reqs"
                    ),
                    patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor),
                    patch.object(
                        SchedulerBatchResultProcessor,
                        "_mamba_prefix_cache_update",
                    ) as cache_update,
                ):
                    with self.assertRaises(StopIteration):
                        scheduler.event_loop_overlap()

                self.assertEqual(observed_lookahead, [expected_lookahead])
                if expected_lookahead == 0:
                    cache_update.assert_not_called()
                else:
                    self.assertTrue(cache_update.call_args.kwargs["known_boundary"])

    def test_pending_kv_retirement_waits_for_event_and_lookahead(self):
        req, lookahead = _make_batch()
        processor = _make_processor()
        last_use_done = MagicMock()
        last_use_done.query.side_effect = [False, True, True]
        processor._pending_kv_retirements[req] = (True, last_use_done)

        with patch.object(
            SchedulerBatchResultProcessor,
            "_release_finished_req_kv_cache",
            autospec=True,
        ) as release:
            self.assertEqual(processor.retire_ready_kv_cache(None), 0)
            self.assertEqual(processor.retire_ready_kv_cache(lookahead), 0)
            self.assertEqual(processor.retire_ready_kv_cache(None), 1)

        release.assert_called_once_with(processor, req, is_insert=True)
        processor.token_to_kv_pool_allocator.free_group_begin.assert_called_once()
        processor.token_to_kv_pool_allocator.free_group_end_cpu_async.assert_called_once()
        self.assertNotIn(req, processor._pending_kv_retirements)

    def test_pending_kv_retirement_avoids_device_fence(self):
        req, _ = _make_batch()
        processor = _make_processor()
        processor.token_to_kv_pool_allocator.device = torch.device("cuda")
        last_use_done = MagicMock()
        last_use_done.query.return_value = True
        processor._pending_kv_retirements[req] = (True, last_use_done)

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(
                SchedulerBatchResultProcessor,
                "_synchronize_kv_retirement_device",
            ) as synchronize,
            patch.object(
                SchedulerBatchResultProcessor,
                "_release_finished_req_kv_cache",
                autospec=True,
            ) as release,
        ):
            self.assertEqual(processor.retire_ready_kv_cache(None), 1)

        synchronize.assert_not_called()
        release.assert_called_once_with(processor, req, is_insert=True)


if __name__ == "__main__":
    unittest.main()
