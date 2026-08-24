import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.dp_attn import (
    _local_prefill_phase,
    select_dp_prefill_phase_ranks,
)


def _select(tokens, kinds, prefixes, prompts, pending):
    return select_dp_prefill_phase_ranks(
        tokens, kinds, prefixes, prompts, prompts, pending
    )


class TestDPPrefillPhaseAdmission(unittest.TestCase):
    def test_deferred_batch_is_retried_without_advancing_chunk_state(self):
        candidate = object()
        running_batch = SimpleNamespace(batch_is_full=True, is_empty=lambda: True)
        calls = {"process": 0, "select": 0, "coordinate": 0}
        scheduler = SimpleNamespace(
            _dp_deferred_prefill_batch=None,
            waiting_queue=[],
            chunked_req=None,
            disagg_prefill_bootstrap_queue=SimpleNamespace(queue=[]),
            process_pending_chunked_abort=lambda: None,
            resolve_waiting_queue_bootstrap=lambda: None,
        )

        def process_chunk(**_kwargs):
            calls["process"] += 1

        def get_new(_running_batch):
            calls["select"] += 1
            return SimpleNamespace(batch_to_run=candidate, running_batch=running_batch)

        def coordinate(batch):
            calls["coordinate"] += 1
            if calls["coordinate"] == 1:
                scheduler._dp_deferred_prefill_batch = batch
                return None
            return batch

        scheduler.process_prefill_chunk = process_chunk
        scheduler.get_new_batch_prefill = get_new
        scheduler.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch, **_kwargs: batch
        )
        scheduler._coordinate_dp_prefill_phase_admission = coordinate

        run = SchedulerDisaggregationPrefillMixin.get_next_disagg_prefill_batch_to_run
        self.assertIsNone(run(scheduler, running_batch, None).batch_to_run)
        self.assertIs(run(scheduler, running_batch, None).batch_to_run, candidate)
        self.assertEqual(calls, {"process": 1, "select": 1, "coordinate": 2})

    def test_deferred_batch_keeps_scheduler_non_idle(self):
        scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(is_empty=lambda: True),
            chunked_req=None,
            _dp_deferred_prefill_batch=object(),
            dllm_manager=SimpleNamespace(any_staging_reqs=lambda: False),
            last_batch=None,
            enable_overlap=True,
            result_queue=[],
            _pp_microbatches_drained=lambda: True,
            waiting_queue=[],
            grammar_manager=SimpleNamespace(grammar_queue=[]),
            disaggregation_mode=DisaggregationMode.PREFILL,
            disagg_prefill_inflight_queue=[],
            disagg_prefill_bootstrap_queue=SimpleNamespace(queue=[]),
            enable_hisparse=False,
            enable_hierarchical_cache=False,
        )
        self.assertFalse(Scheduler.is_fully_idle(scheduler))

    def test_local_phase_uses_prepared_prefixes(self):
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            reqs=[SimpleNamespace(origin_input_ids=range(131072))],
            prefix_lens=[8192],
            batch_size=lambda: 1,
        )
        self.assertEqual(
            _local_prefill_phase(batch), (2, 8192, 8192, 1, 131072, 131072)
        )

    def test_phase_selection(self):
        cases = (
            (
                [32768] * 4,
                [2] * 4,
                [65536] * 4,
                [131072] * 4,
                [0] * 4,
                [True] * 4,
            ),
            (
                [32768, 0, 32768, 0],
                [1, 0, 1, 0],
                [0, -1, 0, -1],
                [131072, -1, 131072, -1],
                [0, 1, 0, 1],
                [False] * 4,
            ),
            (
                [32768] * 4,
                [2] * 4,
                [65536, 32768, 65536, 32768],
                [131072] * 4,
                [0] * 4,
                [False, True, False, True],
            ),
            (
                [8192] * 4,
                [2] * 4,
                [196608, 65536, 229376, 32768],
                [232030, 235891, 249691, 240644],
                [0] * 4,
                [True] * 4,
            ),
        )
        for tokens, kinds, prefixes, prompts, pending, expected in cases:
            with self.subTest(prefixes=prefixes, pending=pending):
                self.assertEqual(
                    _select(tokens, kinds, prefixes, prompts, pending), expected
                )


if __name__ == "__main__":
    unittest.main()
