"""Compute-only unit coverage for live QSA BCG metadata and stable bridge rows."""
import unittest
from contextlib import nullcontext
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

    def test_descending_buckets_share_bridge_storage(self):
        context = SimpleNamespace(forward_batch=SimpleNamespace(rows=16))
        class Layer:
            def _compute_qsa_topk_indices_eager(self, hidden, positions, batch, **kwargs):
                return hidden[:batch.rows, :2].to(torch.int32)
        layer = Layer()
        large = torch.ones((16, 2), device="cuda")
        positions = torch.arange(16, device="cuda")
        with patch.object(qwen4_exp, "get_tc_piecewise_forward_context", return_value=context):
            a = qwen4_exp._breakable_qsa_indexer(layer, large, positions)
            context.forward_batch = SimpleNamespace(rows=3)
            b = qwen4_exp._breakable_qsa_indexer(layer, large[:8], positions[:8])
            self.assertEqual(a.data_ptr(), b.data_ptr(), "each bucket retained a separate bridge")
            self.assertEqual(layer._qsa_prefill_topk_bridge.shape, (16, 2))
            self.assertTrue(torch.equal(b[:3], torch.ones_like(b[:3])))
            self.assertTrue(torch.equal(b[3:], torch.full_like(b[3:], -1)))

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

    def test_mtp_live_embeddings_pad_without_mutating_target(self):
        from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner
        runner = SimpleNamespace(capture_num_tokens=[8, 128, 256])
        pad = PrefillCudaGraphRunner._pad_qwen_bcg_mtp_embeddings
        live = torch.arange(122 * 4, device="cuda", dtype=torch.float32).reshape(122, 4)
        original = live.clone()
        padded = pad(runner, live, 122, 128)
        self.assertEqual(padded.shape, (128, 4))
        self.assertTrue(torch.equal(padded[:122], live), "live MTP embeddings changed")
        self.assertTrue(torch.equal(padded[122:], torch.zeros_like(padded[122:])))
        ptr = padded.data_ptr()
        small = pad(runner, live[:3], 3, 8)
        self.assertEqual(small.data_ptr(), ptr, "MTP padding allocated per bucket")
        self.assertTrue(torch.equal(small[3:], torch.zeros_like(small[3:])))
        self.assertTrue(torch.equal(live, original), "target side channel was mutated")
        self.assertIs(pad(runner, live, 122, 122), live)
        with self.assertRaisesRegex(ValueError, "live token rows"):
            pad(runner, live, 121, 128)
        with self.assertRaisesRegex(ValueError, "exceed"):
            pad(runner, live, 122, 8)

    def test_mtp_replay_dispatch_uses_rewritten_draft_architecture(self):
        from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner
        from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import BreakableCudaGraphBackend
        runner = object.__new__(PrefillCudaGraphRunner)
        runner.backend = object.__new__(BreakableCudaGraphBackend)
        runner._is_full_backend = False
        runner._input_embeds_arg_idx = None
        runner.capture_num_tokens = [128]
        runner.layer_model = SimpleNamespace(forward=lambda: None)
        original = runner.layer_model.forward
        runner.buffer_registry = SimpleNamespace(has_slot=lambda name: False)
        runner._prefill_forward_context = lambda *a, **k: nullcontext()
        def forward(ids, positions, batch, **kwargs):
            self.assertEqual(batch.mm_input_embeds.shape[0], 128,
                             "actual MTP architecture did not route through padding")
            return batch.mm_input_embeds.clone()
        runner.model_runner = SimpleNamespace(is_draft_worker=True,
            model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=['Qwen4ExpForCausalLMMTP'])),
            model=SimpleNamespace(forward=forward))
        live = torch.ones((122, 4), device='cuda')
        batch = SimpleNamespace(mm_input_embeds=live)
        static = SimpleNamespace(input_ids=torch.zeros(128, device='cuda', dtype=torch.int64),
                                 positions=torch.arange(128, device='cuda'))
        result = runner._execute_body_capture(batch, static, 128, 122, None)
        self.assertTrue(torch.equal(result[:122], live))
        self.assertTrue(torch.equal(result[122:], torch.zeros_like(result[122:])))
        self.assertIs(batch.mm_input_embeds, live)
        self.assertIs(runner.layer_model.forward, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
