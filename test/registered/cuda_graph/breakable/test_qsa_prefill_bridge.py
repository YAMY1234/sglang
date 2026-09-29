"""Compute-only unit coverage for live QSA BCG metadata and stable bridge rows."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.qsa.metadata import QSAIndexerMetadata
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
)
from sglang.srt.models import qwen4_exp


class TestQSAPrefillBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("QSA BCG bridge tests require an allocated CUDA device")

    def test_live_batch_and_padded_bridge_replay(self):
        # One graph bucket, changing batch identities, prefix lengths and token
        # counts. The eager callback must see the replacement live batch.
        context = SimpleNamespace(forward_batch=SimpleNamespace(rows=8, prefix=0))
        seen = []

        class Layer:
            def _compute_qsa_topk_indices_eager(
                self, hidden, positions, batch, *, use_host_prefill_lengths=False
            ):
                if not use_host_prefill_lengths:
                    raise AssertionError("BCG callback did not request CPU lengths")
                seen.append((id(batch), batch.rows, batch.prefix))
                return (hidden[:batch.rows, :2] + batch.prefix).to(torch.int32)

        x = torch.zeros((8, 2), device="cuda")
        positions = torch.arange(8, device="cuda")
        result = torch.empty((8, 2), device="cuda", dtype=torch.int32)
        layer = Layer()
        graph = BreakableCUDAGraph()
        stream = torch.cuda.Stream()
        with patch.object(qwen4_exp, "get_tc_piecewise_forward_context", return_value=context):
            # Warm the exact callback before capture.
            qwen4_exp._breakable_qsa_indexer(layer, x, positions)
            torch.cuda.synchronize()
            with BreakableCUDAGraphCapture(graph, stream=stream):
                bridge = qwen4_exp._breakable_qsa_indexer(layer, x + 1, positions)
                result.copy_(bridge)
            for rows, prefix in ((3, 17), (7, 33), (8, 4), (1, 99)):
                batch = SimpleNamespace(rows=rows, prefix=prefix)
                context.forward_batch = batch
                x.fill_(2)
                graph.replay()
                torch.cuda.synchronize()
                expected = torch.full_like(result, -1)
                expected[:rows].fill_(3 + prefix)
                self.assertTrue(torch.equal(result, expected), (rows, prefix, result))
                self.assertEqual(seen[-1], (id(batch), rows, prefix))

    def test_missing_runtime_context_rejected(self):
        with patch.object(qwen4_exp, "get_tc_piecewise_forward_context", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "live prefill context"):
                qwen4_exp._breakable_qsa_indexer(None, None, None)

    def test_host_lengths_match_legacy_gather(self):
        device = "cuda"
        buffer = torch.arange(64, device=device, dtype=torch.float32).reshape(16, 1, 4)
        pool = SimpleNamespace(
            get_qsa_compressed_k_buffer=lambda layer: buffer,
            qsa_index_kv_heads=1,
            qsa_index_head_dim=4,
        )
        kwargs = dict(
            sequence_lengths=torch.tensor([8, 4, 3], device=device),
            token_to_batch_idx=torch.tensor([0, 0, 1, 2], device=device),
            token_slot_table=torch.arange(24, device=device).reshape(3, 8),
            out_cache_loc=torch.arange(4, device=device),
            token_to_kv_pool=pool, compress_ratio=4, block_topk=2,
        )
        pos = torch.tensor([6, 7, 3, 2], device=device)
        legacy = QSAIndexerMetadata(**kwargs).get_prefill_mqa_inputs(0, pos)
        host = QSAIndexerMetadata(
            **kwargs, prefill_sequence_lengths_cpu=(8, 4, 3)
        ).get_prefill_mqa_inputs(0, pos)
        for actual, expected in zip(host, legacy):
            self.assertTrue(torch.equal(actual, expected), "host-length gather changed values")
        with self.assertRaisesRegex(ValueError, "sequence counts differ"):
            QSAIndexerMetadata(
                **kwargs, prefill_sequence_lengths_cpu=(8,)
            ).get_prefill_mqa_inputs(0, pos)


if __name__ == "__main__":
    unittest.main(verbosity=2)
