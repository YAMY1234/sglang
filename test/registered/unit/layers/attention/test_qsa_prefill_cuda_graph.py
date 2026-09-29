"""CPU regression for the QSA breakable-prefill output bridge."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.layers.attention.qsa.prefill_cuda_graph import (
    _qsa_indexer_prefill_with_output,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQSAPrefillCudaGraphBridge(unittest.TestCase):
    def test_live_metadata_padding_and_bucket_boundary(self):
        live_metadata = object()
        capture_metadata = object()
        forward_batch = SimpleNamespace(
            extend_num_tokens=3,
            capture_metadata=capture_metadata,
        )
        context = SimpleNamespace(forward_batch=forward_batch, raw_num_tokens=3)
        backend = SimpleNamespace(
            get_indexer_metadata=Mock(return_value=live_metadata),
            should_capture_mtp_sparse_indices=Mock(return_value=True),
            capture_mtp_sparse_indices=Mock(),
        )
        hidden_states = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        positions = torch.arange(5, dtype=torch.int64)
        output = torch.empty((5, 4), dtype=torch.int32)
        first_result = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32)

        def forward_impl(hidden, pos, batch, metadata):
            self.assertEqual(tuple(hidden.shape), (3, 3))
            self.assertEqual(tuple(pos.shape), (3,))
            self.assertIs(batch, forward_batch)
            self.assertIs(metadata, live_metadata)
            self.assertIsNot(metadata, capture_metadata)
            return first_result

        indexer = SimpleNamespace(_forward_impl=Mock(side_effect=forward_impl))
        target = "sglang.srt.layers.attention.qsa.prefill_cuda_graph"
        with (
            patch(f"{target}.get_tc_piecewise_forward_context", return_value=context),
            patch(f"{target}.get_attn_backend", return_value=backend),
        ):
            _qsa_indexer_prefill_with_output(
                indexer, hidden_states, positions, output, layer_id=7
            )
            self.assertTrue(
                torch.equal(output[:2], torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]))
            )
            self.assertTrue(torch.equal(output[2:], torch.full((3, 4), -1)))
            backend.capture_mtp_sparse_indices.assert_called_once_with(
                first_result,
                forward_batch,
                7,
                metadata=live_metadata,
            )

            # The exact bucket boundary is valid; one token beyond it fails loudly.
            context.raw_num_tokens = 5
            indexer._forward_impl.return_value = torch.zeros((5, 4), dtype=torch.int32)
            indexer._forward_impl.side_effect = None
            _qsa_indexer_prefill_with_output(
                indexer, hidden_states, positions, output, layer_id=7
            )
            self.assertTrue(torch.equal(output, torch.zeros((5, 4), dtype=torch.int32)))

            context.raw_num_tokens = 6
            with self.assertRaisesRegex(ValueError, "Invalid QSA prefill token count"):
                _qsa_indexer_prefill_with_output(
                    indexer, hidden_states, positions, output, layer_id=7
                )
