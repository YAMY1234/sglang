import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req():
    return SimpleNamespace(
        req_pool_idx=1,
        pending_bootstrap=False,
        extend_range=SimpleNamespace(end=128),
        origin_input_ids=list(range(128)),
        start_send_idx=0,
        disagg_decode_prefix_len=0,
        disagg_mamba_state_index_cpu=None,
        disagg_mamba_state_index_copy_done=None,
        disagg_page_indices_staging=None,
    )


class TestPrefillMetadataStaging(unittest.TestCase):
    def test_stages_mamba_index_and_waits_before_host_read(self):
        scheduler = SchedulerDisaggregationPrefillMixin()
        scheduler.enable_overlap = True
        scheduler.schedule_stream = object()
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
            kv_manager=SimpleNamespace(
                kv_args=SimpleNamespace(state_types=[StateType.MAMBA])
            )
        )
        device_index = torch.tensor([7], dtype=torch.int32)
        scheduler.req_to_token_pool = SimpleNamespace(
            req_index_to_mamba_index_mapping={1: torch.tensor([3])},
            translate_mamba_indices=MagicMock(return_value=device_index),
        )
        req = _req()
        host_index = MagicMock()
        host_value = object()
        host_index.numpy.return_value = host_value
        event = MagicMock()

        with (
            patch(
                "sglang.srt.disaggregation.prefill.torch.empty_like",
                return_value=host_index,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.torch.cuda.Event",
                return_value=event,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.torch.cuda.stream",
                return_value=nullcontext(),
            ),
        ):
            scheduler.stage_mamba_state_index(req)

        host_index.copy_.assert_called_once_with(device_index, non_blocking=True)
        event.record.assert_called_once_with()
        self.assertEqual(scheduler.get_mamba_state_index_payload(req), [host_value])
        event.synchronize.assert_called_once_with()

    def test_stages_final_page_ids_and_waits_for_exact_range(self):
        scheduler = SchedulerDisaggregationPrefillMixin()
        scheduler.enable_overlap = True
        scheduler.enable_staging = False
        scheduler.schedule_stream = object()
        scheduler.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(256, dtype=torch.int64).reshape(2, 128)
        )
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=64,
            translate_kv_indices_for_transfer=lambda indices: indices,
        )
        req = _req()
        host_indices = MagicMock()
        host_value = object()
        host_indices.numpy.return_value = host_value
        event = MagicMock()

        with (
            patch(
                "sglang.srt.disaggregation.prefill.torch.empty_like",
                return_value=host_indices,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.torch.cuda.Event",
                return_value=event,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.torch.cuda.stream",
                return_value=nullcontext(),
            ),
        ):
            scheduler.stage_final_kv_page_indices(req)

        torch.testing.assert_close(
            host_indices.copy_.call_args.args[0], torch.tensor([2, 3])
        )
        self.assertTrue(host_indices.copy_.call_args.kwargs["non_blocking"])
        self.assertEqual(req.disagg_page_indices_staging[0][:2], (0, 128))
        self.assertIs(
            scheduler.get_staged_final_kv_page_indices(req, 0, 128), host_value
        )
        self.assertIsNone(scheduler.get_staged_final_kv_page_indices(req, 64, 128))
        event.synchronize.assert_called_once_with()

    @patch("sglang.srt.managers.schedule_batch.torch.tensor")
    def test_mamba_track_metadata_uses_pinned_nonblocking_copy(self, tensor_mock):
        host_tensor = MagicMock()
        tensor_mock.return_value = host_tensor

        ScheduleBatch._stage_mamba_track_metadata(
            [True, False],
            dtype=torch.bool,
            device="cuda",
            pin_memory=True,
        )

        tensor_mock.assert_called_once_with(
            [True, False], dtype=torch.bool, pin_memory=True
        )
        host_tensor.to.assert_called_once_with("cuda", non_blocking=True)


if __name__ == "__main__":
    unittest.main()
