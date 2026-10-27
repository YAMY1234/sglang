import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers import overlap_utils
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_len_per_decode,
    get_alloc_reserve_per_decode,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative import eagle_utils
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.prepared_next_draft import (
    NextDraftInputStager,
    PreparedNextDraftInputs,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _req(slot: int, *, allocated: int, committed: int = 100) -> Req:
    sampling_params = SamplingParams(max_new_tokens=16, temperature=0)
    sampling_params.normalize(None)
    req = Req(
        rid=f"r{slot}",
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=sampling_params,
    )
    req.kv.req_pool_idx = slot
    req.kv.kv_committed_len = committed
    req.kv.kv_allocated_len = allocated
    return req


def _batch(slots, allocated, *, mode=ForwardMode.DECODE):
    return SimpleNamespace(
        reqs=[
            _req(slot, allocated=alloc)
            for slot, alloc in zip(slots, allocated, strict=True)
        ],
        forward_mode=mode,
    )


def _prepared(num_requests: int = 2) -> PreparedNextDraftInputs:
    return PreparedNextDraftInputs(
        num_requests=num_requests,
        seq_lens=torch.zeros(num_requests, dtype=torch.int64),
        positions=torch.zeros(num_requests, dtype=torch.int64),
        mrope_positions=None,
        draft_out_cache_loc=torch.zeros(num_requests * 6, dtype=torch.int64),
        write_window=torch.zeros(num_requests * 7, dtype=torch.int64),
    )


def _decode_prepare_batch(reqs, *, penalized=False):
    return SimpleNamespace(
        reqs=reqs,
        maybe_evict_swa=lambda: None,
        sampling_info=SimpleNamespace(
            penalizer_orchestrator=SimpleNamespace(is_required=penalized)
        ),
        cumulate_penalty_output_tokens=lambda: None,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
        tree_cache=SimpleNamespace(supports_swa=lambda: False),
        req_to_token_pool=SimpleNamespace(),
        req_pool_indices=torch.tensor([req.kv.req_pool_idx for req in reqs]),
        seq_lens_cpu=torch.tensor([req.kv.kv_committed_len for req in reqs]),
        device="cpu",
        spec_decode_schedule_stream_idle=False,
    )


class TestPreparedNextDraft(CustomTestCase):
    def test_inputs_survive_the_length_read_and_are_consumed_once(self):
        """The scheduler may read prepared lengths before the worker consumes the
        matching buffers; reordered, padded, or under-reserved batches must decline."""
        stager = NextDraftInputStager(window_size=7, graph_sizes=[2, 4])
        inputs = _prepared()
        seq_lens = torch.tensor([104, 105])
        batch = _batch([5, 9], [121, 121])

        stager.note_decode(batch, prepared=inputs, new_seq_lens=seq_lens)
        self.assertIs(stager.take_seq_lens(batch), seq_lens)
        self.assertIs(stager.take_seq_lens(batch), seq_lens)
        self.assertIs(stager.take_inputs(batch), inputs)
        self.assertIsNone(stager.take_inputs(batch))

        for previous, current in (
            (batch, _batch([9, 5], [121, 121])),
            (_batch([5, 9, 3], [121] * 3), _batch([5, 9, 3], [121] * 3)),
            (_batch([5, 9], [121, 120]), batch),
        ):
            with self.subTest(slots=[req.kv.req_pool_idx for req in current.reqs]):
                stager.note_decode(previous, prepared=inputs, new_seq_lens=seq_lens)
                self.assertIsNone(stager.take_seq_lens(current))
                self.assertIsNone(stager.take_inputs(current))

    def test_only_decode_can_take_prepared_lengths(self):
        """A prefill reusing the same request slots cannot consume decode lengths."""
        worker = EAGLEWorkerV2.__new__(EAGLEWorkerV2)
        worker._next_draft_stager = NextDraftInputStager(window_size=7, graph_sizes=[2])
        seq_lens = torch.tensor([104, 105])
        decode = _batch([5, 9], [121, 121])
        worker._next_draft_stager.note_decode(
            decode, prepared=_prepared(), new_seq_lens=seq_lens
        )
        self.assertIs(worker.take_prepared_seq_lens(decode), seq_lens)
        decode.forward_mode = ForwardMode.EXTEND
        self.assertIsNone(worker.take_prepared_seq_lens(decode))

    def test_fast_path_reserves_three_decode_windows(self):
        """Prepared inputs remain valid through the two overlap buffers plus the
        subsequent draft window; every other configuration retains two windows."""
        with (
            get_context().override_server_args(
                speculative_algorithm="EAGLE",
                speculative_num_steps=6,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=7,
                page_size=1,
            ),
            get_parallel().override(attn_dcp_size=1),
            envs.SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH.override(True),
            patch(
                "sglang.srt.speculative.prepared_next_draft."
                "nvidia_eagle_decode_fast_path_enabled",
                return_value=True,
            ),
        ):
            window = get_alloc_len_per_decode()
            self.assertEqual(get_alloc_reserve_per_decode(), 3 * window)
        with (
            get_context().override_server_args(
                speculative_algorithm="EAGLE",
                speculative_num_steps=6,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=7,
                page_size=1,
            ),
            get_parallel().override(attn_dcp_size=1),
            envs.SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH.override(False),
            patch(
                "sglang.srt.speculative.prepared_next_draft."
                "nvidia_eagle_decode_fast_path_enabled",
                return_value=True,
            ),
        ):
            self.assertEqual(get_alloc_reserve_per_decode(), 2 * window)

    def test_future_map_uses_prepared_lengths_without_waiting_for_the_relay(self):
        """An unchanged batch must not wait for or gather the relay that the
        preceding post-verify graph has already materialized."""
        pool = SimpleNamespace(req_to_token=torch.zeros((8, 16), dtype=torch.int32))
        with (
            patch.object(overlap_utils, "_is_cuda", False),
            patch.object(
                overlap_utils, "spec_relay_bypass_enabled", return_value=False
            ),
            patch(
                "sglang.srt.speculative.prepared_next_draft."
                "nvidia_eagle_decode_fast_path_enabled",
                return_value=True,
            ),
        ):
            future_map = FutureMap(
                torch.device("cpu"), SpeculativeAlgorithm.EAGLE, pool
            )
        future_map.publish_ready = Mock()
        future_indices = torch.tensor([3, 1])
        batch = SimpleNamespace(
            spec_info=SimpleNamespace(future_indices=future_indices),
            req_pool_indices_cpu=future_indices,
        )
        prepared = torch.tensor([30, 10])

        future_map.resolve_seq_lens_cpu(batch, prepared_seq_lens=prepared)

        self.assertIs(batch.seq_lens, prepared)
        self.assertEqual(batch.seq_lens_cpu.tolist(), [30, 10])
        future_map.publish_ready.wait.assert_not_called()

    def test_schedule_stream_join_is_skipped_only_after_idle_preparation(self):
        """Prepared lengths alone do not waive a schedule-stream join when KV
        allocation or penalty bookkeeping launched work on that stream."""
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.forward_stream = "forward"
        scheduler.schedule_stream = "schedule"
        prepared = torch.tensor([10])
        cases = (
            (prepared, True, False),
            (prepared, False, True),
            (None, True, True),
        )
        for seq_lens, idle, joins in cases:
            batch = SimpleNamespace(spec_decode_schedule_stream_idle=idle)
            with (
                self.subTest(prepared=seq_lens is not None, idle=idle),
                patch.object(
                    scheduler_module.spec_decode_resources, "wait_stream"
                ) as wait,
            ):
                scheduler._join_schedule_stream(batch, seq_lens)
                self.assertEqual(wait.call_count, int(joins))
                if seq_lens is not None:
                    self.assertFalse(batch.spec_decode_schedule_stream_idle)

    def test_decode_without_new_pages_updates_only_host_watermarks(self):
        """A fully reserved round avoids allocation tensors and marks the schedule
        stream idle only when penalty and cache bookkeeping also launch no work."""
        reserve = 21
        rounds = (
            (121, False, True),
            (121, True, False),
            (120, False, False),
        )
        for allocated, penalized, idle in rounds:
            req = _req(5, allocated=allocated)
            batch = _decode_prepare_batch([req], penalized=penalized)
            with (
                self.subTest(allocated=allocated, penalized=penalized),
                patch.object(
                    eagle_utils,
                    "get_alloc_reserve_per_decode",
                    return_value=reserve,
                ),
                patch.object(
                    eagle_utils,
                    "nvidia_eagle_decode_fast_path_enabled",
                    return_value=True,
                ),
                patch.object(
                    eagle_utils,
                    "get_exec",
                    return_value=SimpleNamespace(
                        mamba=SimpleNamespace(enable_mamba_extra_buffer_lazy=False)
                    ),
                ),
                patch.object(
                    eagle_utils, "update_spec_decode_allocated_lens"
                ) as update,
                patch.object(eagle_utils, "alloc_for_spec_decode") as allocate,
            ):
                eagle_utils.eagle_prepare_for_decode(batch)
                self.assertEqual(batch.spec_decode_schedule_stream_idle, idle)
                if allocated == 121:
                    update.assert_called_once_with([req], [121])
                    allocate.assert_not_called()
                else:
                    update.assert_not_called()
                    allocate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
