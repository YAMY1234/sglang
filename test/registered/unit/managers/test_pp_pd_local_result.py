"""PP prefill (PD) without the output ring, and HiCache counts on the request relay.

SGLANG_PP_PD_LOCAL_RESULT: the last stage processes each microbatch from its own
output and the earlier stages from a host placeholder, so no stage sends or
receives the output dict. SGLANG_PP_HICACHE_COUNTS_PIGGYBACK: PP0's HiCache
ready counts reach the later stages on the request relay of the same loop
iteration instead of a blocking send/recv chain.
"""

import unittest
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_pp_mixin import (  # noqa: E402
    PPBatchMetadata,
    PPHiCacheCounts,
    SchedulerPPMixin,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors  # noqa: E402
from sglang.srt.model_executor.forward_mode import ForwardMode  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _req(**flags):
    return SimpleNamespace(
        grammar=flags.get("grammar"),
        return_sampling_mask=flags.get("return_sampling_mask", False),
        return_routed_experts=flags.get("return_routed_experts", False),
    )


def _batch(name, reqs=None, forward_mode=ForwardMode.EXTEND, return_logprob=False):
    return SimpleNamespace(
        name=name,
        forward_mode=forward_mode,
        reqs=reqs if reqs is not None else [_req()],
        return_logprob=return_logprob,
        contains_last_prefill_chunk=True,
        spec_info=None,
    )


def _device_module():
    return SimpleNamespace(
        Event=lambda: SimpleNamespace(record=lambda stream: None),
        current_stream=lambda: None,
    )


def _scheduler(is_last_rank, enabled=True):
    scheduler = SchedulerPPMixin()
    scheduler._pp_pd_local_result_enabled = enabled
    scheduler.pp_group = SimpleNamespace(
        is_last_rank=is_last_rank, is_first_rank=not is_last_rank
    )
    scheduler.device_module = _device_module()
    scheduler.copy_stream = SimpleNamespace(
        wait_stream=lambda stream: None, wait_event=lambda event: None
    )
    scheduler.copy_stream_ctx = nullcontext()
    scheduler.schedule_stream = None
    scheduler.pp_comm_stream_ctx = nullcontext()
    scheduler.launch_event = None
    scheduler._pp_local_outputs = {}
    return scheduler


class TestPDLocalResultEligibility(CustomTestCase):
    def test_plain_prefill_batch_skips_ring(self):
        self.assertTrue(_scheduler(True)._pp_pd_local_result(_batch("a")))

    def test_batches_with_a_per_stage_result_consumer_keep_ring(self):
        scheduler = _scheduler(True)
        for name, batch in (
            ("logprob", _batch("b", return_logprob=True)),
            ("grammar", _batch("b", reqs=[_req(), _req(grammar=object())])),
            ("sampling_mask", _batch("b", reqs=[_req(return_sampling_mask=True)])),
            ("routed_experts", _batch("b", reqs=[_req(return_routed_experts=True)])),
            ("decode", _batch("b", forward_mode=ForwardMode.DECODE)),
        ):
            with self.subTest(name):
                self.assertFalse(scheduler._pp_pd_local_result(batch))

    def test_off_by_default(self):
        self.assertFalse(SchedulerPPMixin._pp_pd_local_result_enabled)
        self.assertFalse(_scheduler(True, enabled=False)._pp_pd_local_result(_batch("a")))


class TestPDLocalResultProcessing(CustomTestCase):
    def test_earlier_stage_processes_host_placeholder(self):
        scheduler = _scheduler(is_last_rank=False)
        batch = _batch("a", reqs=[_req(), _req()])
        batch.spec_info = object()
        forward_event = object()
        md = PPBatchMetadata(
            can_run_cuda_graph=False, pd_local_result=True, forward_event=forward_event
        )

        outputs, result, d2h_event = scheduler._pp_make_local_result(batch, md, 0)

        self.assertIsNone(outputs)
        # The loop synchronizes on this before processing: it must be this mb's
        # own forward completion, not an event on the schedule stream.
        self.assertIs(d2h_event, forward_event)
        self.assertEqual(result.next_token_ids.device.type, "cpu")
        self.assertEqual(result.next_token_ids.tolist(), [0, 0])
        # process_batch_result_disagg_prefill asserts spec_info is next_draft_input.
        self.assertIsNone(batch.spec_info)
        self.assertIsNone(result.next_draft_input)

    def test_earlier_stage_without_forward_event_fails_loudly(self):
        scheduler = _scheduler(is_last_rank=False)
        md = PPBatchMetadata(can_run_cuda_graph=False, pd_local_result=True)
        with self.assertRaises(AssertionError):
            scheduler._pp_make_local_result(_batch("a"), md, 0)

    def _run_last_stage(self, eligible):
        """Drive the last stage of PP4 (depth 0) through 8 loop iterations."""
        pp_size = 4
        scheduler = _scheduler(is_last_rank=True)
        mbs = [None] * pp_size
        mb_metadata = [None] * pp_size
        scheduler.mb_metadata = mb_metadata
        queue = deque()
        sends, recvs, processed = [], [], []
        scheduler._pp_send_dict_to_next_stage = (
            lambda tensors, **kwargs: sends.append(tensors["id"].item()) or []
        )
        scheduler._pp_recv_dict_from_prev_stage = lambda: (
            recvs.append(True) or ({"id": torch.tensor(-1)}, None)
        )
        scheduler._pp_prep_batch_result = lambda batch, md, outputs: SimpleNamespace(
            logits_output=None, source=outputs.tensors["id"].item(), batch=batch
        )

        with published_topology(
            role="scheduler", ranks={"world_rank": pp_size - 1}, pp_size=pp_size
        ):
            for step in range(8):
                mb_id = step % pp_size
                batch = _batch(f"b{step}")
                mbs[mb_id] = batch
                mb_metadata[mb_id] = PPBatchMetadata(
                    can_run_cuda_graph=False, pd_local_result=eligible(step)
                )
                queue.append((None, PPProxyTensors({"id": torch.tensor(step)})))
                _, result, _, _ = scheduler._pp_send_recv_and_preprocess_output_tensors(
                    mb_id, (mb_id + 1) % pp_size, mbs, mb_metadata, queue, None
                )
                if result is not None:
                    processed.append((result.batch.name, result.source))
        return sends, recvs, processed, scheduler

    def test_last_stage_uses_own_output_and_skips_ring(self):
        sends, recvs, processed, scheduler = self._run_last_stage(lambda step: True)

        self.assertEqual(sends, [])
        self.assertEqual(recvs, [])
        # Each processed microbatch is matched with the output it produced.
        self.assertTrue(processed)
        for name, source in processed:
            self.assertEqual(name, f"b{source}")
        # Outputs of the three microbatches still in flight stay held.
        self.assertEqual(len(scheduler._pp_local_outputs), 3)

    def test_ineligible_batch_keeps_ring_among_local_ones(self):
        sends, recvs, processed, _ = self._run_last_stage(lambda step: step != 2)

        self.assertEqual(sends, [2])
        self.assertEqual(len(recvs), 1)
        for name, source in processed:
            if source != -1:
                self.assertEqual(name, f"b{source}")
        self.assertIn(("b2", -1), processed)

    def test_last_stage_copies_held_outputs_at_launch(self):
        scheduler = _scheduler(is_last_rank=True)
        scheduler._pp_spec_relay = False
        scheduler.forward_stream_ctx = nullcontext()
        scheduler.forward_stream = Mock()
        scheduler.pp_send_done_event = None
        scheduler.pp_proxy_recv_event = None
        token_ids = torch.tensor([7, 8])
        scheduler.run_batch = lambda batch, proxy: SimpleNamespace(
            can_run_cuda_graph=False,
            spec_verify_out_cache_loc=None,
            fpm_timing=None,
            next_token_ids=token_ids,
            next_draft_input=None,
            logits_output=None,
            accept_lens=None,
        )
        batch = _batch("a", reqs=[])
        batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        mb_metadata = [None]
        queue = deque()

        _, launch_event = scheduler._pp_launch_batch(0, batch, None, mb_metadata, queue)

        self.assertTrue(mb_metadata[0].pd_local_result)
        self.assertIs(mb_metadata[0].forward_event, launch_event)
        held = queue[0][1].tensors["next_token_ids"]
        self.assertEqual(held.tolist(), [7, 8])
        self.assertNotEqual(held.data_ptr(), token_ids.data_ptr())


def _cache(pp_rank, pp_size):
    cache = object.__new__(UnifiedRadixCache)
    cache.pp_rank = pp_rank
    cache.pp_size = pp_size
    cache.pp_counts_piggyback = True
    cache.pp_counts_in = deque()
    cache.pp_counts_out = []
    cache.attn_reduced = []
    cache._all_reduce_attn_groups = lambda tensor, op: cache.attn_reduced.append(
        tensor.tolist()
    )
    return cache


class TestHiCacheCountsPiggyback(CustomTestCase):
    def test_counts_relay_from_pp0_to_every_stage(self):
        pp_size = 4
        caches = [_cache(rank, pp_size) for rank in range(pp_size)]
        schedulers = []
        for cache in caches:
            scheduler = SchedulerPPMixin()
            scheduler._pp_hicache_piggyback = True
            scheduler.tree_cache = cache
            scheduler.relayed = []
            scheduler._pp_send_pyobj_to_next_stage = (
                lambda data, async_send, s=scheduler: s.relayed.append(data) or []
            )
            schedulers.append(scheduler)

        pp0_counts = [3, 1, 5, -5]
        for rank in range(pp_size):
            local = torch.tensor(pp0_counts if rank == 0 else [0, 0, 5, -5])
            if rank > 0:
                received = schedulers[rank - 1].relayed[-1]
                kept = schedulers[rank]._pp_take_relay_extras(received)
                self.assertEqual(kept, [])
            reduced = caches[rank]._piggyback_ready_counts(local)
            self.assertEqual(reduced.tolist(), pp0_counts)
            if rank + 1 < pp_size:
                schedulers[rank]._pp_send_reqs_to_next_stage([])
                payload = schedulers[rank].relayed[-1]
                self.assertIsInstance(payload[-1], PPHiCacheCounts)
                self.assertEqual(payload[-1].counts, [pp0_counts])
            self.assertEqual(caches[rank].pp_counts_out, [])
            self.assertFalse(caches[rank].pp_counts_in)

        # Only PP0 reduces over its attention ranks; the others adopt its result.
        self.assertEqual(caches[0].attn_reduced, [pp0_counts])
        self.assertTrue(all(not cache.attn_reduced for cache in caches[1:]))

    def test_missing_relay_counts_fail_loudly(self):
        with self.assertRaises(AssertionError):
            _cache(1, 2)._piggyback_ready_counts(torch.tensor([0, 0]))


if __name__ == "__main__":
    unittest.main()
