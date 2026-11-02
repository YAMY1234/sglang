"""CPU coverage of post-verify key admission, fallback counts and decode logs."""

import os
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components import metrics_reporter
from sglang.srt.speculative import eagle_post_verify_graphs as post_verify
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextmanager
def _capture_context(stream):
    yield SimpleNamespace(stream=stream)


class TestPostVerifyCapacity(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(post_verify, "_ONE_LAUNCH_LOAD", False))

    def _graphs(self, capacity):
        with envs.SGLANG_POST_VERIFY_MAX_STATES.override(capacity):
            graphs = post_verify._PostVerifyGraphs(
                body=Mock(return_value=tuple(range(7))),
                commit=Mock(),
                sim_rng=None,
            )
        self.enterContext(
            patch.object(graphs, "_draw_sim_len", return_value=(None, None))
        )
        return graphs

    def _run(self, graphs, key):
        return graphs.run(
            key=(key,),
            inputs={},
            make_slot=lambda staged: SimpleNamespace(num_requests=1),
            sim_max_len=7,
        )

    def test_unset_capacity_keeps_64_keys(self):
        with patch.dict(os.environ):
            envs.SGLANG_POST_VERIFY_MAX_STATES.clear()
            self.assertEqual(envs.SGLANG_POST_VERIFY_MAX_STATES.get(), 64)
            graphs = self._graphs(envs.SGLANG_POST_VERIFY_MAX_STATES.get())
        for key in range(64):
            self.assertEqual(self._run(graphs, key), tuple(range(6)))
        self.assertIsNone(self._run(graphs, 64))
        self.assertEqual(len(graphs._states), 64)
        self.assertEqual(graphs._eager_at_bound, 1)

    def test_typed_override_admits_256_distinct_keys(self):
        graphs = self._graphs(256)
        # Resolve at construction: restoring the env does not shrink this worker.
        for key in range(256):
            self.assertEqual(self._run(graphs, key), tuple(range(6)))
        for _ in range(3):
            self.assertIsNone(self._run(graphs, 256))
        self.assertEqual(len(graphs._states), 256)
        self.assertEqual(graphs._body.call_count, 256)
        self.assertEqual(graphs._eager_at_bound, 3)
        graphs._draw_sim_len.assert_called_with(7)
        self.assertEqual(graphs._draw_sim_len.call_count, 256)

    def test_full_cache_keeps_existing_keys_and_counts_completed_slot_captures(self):
        graphs = self._graphs(1)
        for _ in range(post_verify._NUM_WARMUP * 2):
            self.assertIsNotNone(self._run(graphs, "existing"))
        self.assertEqual(graphs._num_captured, 0)
        with (
            patch.object(torch.cuda, "graph_pool_handle", return_value=object()),
            patch.object(torch.cuda, "Stream", return_value=MagicMock()),
            patch.object(torch.cuda, "Event", side_effect=MagicMock),
            patch.object(torch.cuda, "CUDAGraph", side_effect=MagicMock),
            patch.object(torch.cuda, "current_stream", return_value=MagicMock()),
            patch.object(
                torch.cuda, "graph", side_effect=lambda *a, **k: nullcontext()
            ),
            patch.object(torch.cuda, "stream", side_effect=lambda *a: nullcontext()),
            patch(
                "sglang.srt.distributed.parallel_state.graph_capture",
                side_effect=_capture_context,
            ),
        ):
            # Two alternating slots capture independently, then only replay.
            for _ in range(6):
                self.assertEqual(self._run(graphs, "existing"), tuple(range(6)))
            self.assertEqual(graphs._num_captured, 2)
            for _ in range(4):
                self.assertIsNone(self._run(graphs, "new"))
            self.assertEqual(self._run(graphs, "existing"), tuple(range(6)))
        facade = post_verify.EaglePostVerifyGraphs.__new__(
            post_verify.EaglePostVerifyGraphs
        )
        facade._graphs = graphs
        self.assertEqual(
            facade.log_stats(),
            "post-verify graphs: keys=1, captured=2, eager-at-bound=4",
        )


class TestPostVerifyDecodeLog(unittest.TestCase):
    def test_periodic_decode_line_includes_stats_only_when_graphs_are_enabled(self):
        for graphs in (
            None,
            SimpleNamespace(
                log_stats=Mock(
                    return_value="post-verify graphs: keys=64, captured=128, eager-at-bound=9"
                )
            ),
        ):
            with self.subTest(enabled=graphs is not None):
                reporter = metrics_reporter.SchedulerMetricsReporter.__new__(
                    metrics_reporter.SchedulerMetricsReporter
                )
                batch = SimpleNamespace(reqs=[], forward_iter=1)
                reporter.scheduler = SimpleNamespace(
                    running_batch=batch,
                    waiting_queue=[],
                    spec_algorithm=SimpleNamespace(
                        is_none=lambda: False, is_dspark=lambda: False
                    ),
                    draft_worker=SimpleNamespace(_post_verify_graphs=graphs),
                    disaggregation_mode=DisaggregationMode.NULL,
                    pool_stats_observer=SimpleNamespace(
                        get_pool_stats=lambda: SimpleNamespace(
                            get_decode_usage_msg_parts=list
                        )
                    ),
                    kv_events_publisher=MagicMock(),
                )
                reporter.current_scheduler_metrics_enabled = False
                reporter.is_stats_logging_rank = True
                reporter.enable_mfu_metrics = False
                reporter.forward_ct_decode = 1
                reporter.decode_log_interval = 2
                reporter.last_decode_stats_tic = 0.0
                reporter.num_generated_tokens = 7
                reporter.spec_num_accept_tokens = 5
                reporter.spec_num_correct_drafts = 4
                reporter.spec_num_forward_ct = 1
                reporter.spec_num_cap_tokens = 0
                reporter.spec_num_block_accept_tokens = 0
                reporter.spec_total_num_accept_tokens = 0
                reporter.spec_total_num_forward_ct = 0
                reporter._graph_backend_label = "cuda graph"
                with (
                    patch.object(metrics_reporter, "logger") as logger,
                    patch.object(metrics_reporter, "RECORD_STEP_TIME", False),
                    patch.object(
                        metrics_reporter, "ENABLE_METRICS_DEVICE_TIMER", False
                    ),
                    patch.object(
                        metrics_reporter,
                        "get_spec",
                        return_value=SimpleNamespace(speculative_num_draft_tokens=7),
                    ),
                    patch.object(
                        metrics_reporter,
                        "get_disagg",
                        return_value=SimpleNamespace(language_only=False),
                    ),
                ):
                    reporter.report_decode_stats(True)
                    logger.info.assert_not_called()
                    if graphs is not None:
                        graphs.log_stats.assert_not_called()
                    reporter.forward_ct_decode = 2
                    reporter.report_decode_stats(True)
                    logger.info.assert_called_once()
                    message = logger.info.call_args.args[0]
                    self.assertIn("Decode batch", message)
                    self.assertIn("cuda graph: True", message)
                    if graphs is None:
                        self.assertNotIn("post-verify graphs:", message)
                    else:
                        graphs.log_stats.assert_called_once_with()
                        self.assertIn(
                            "post-verify graphs: keys=64, captured=128, eager-at-bound=9",
                            message,
                        )


if __name__ == "__main__":
    unittest.main()
