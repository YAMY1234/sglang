"""DSV4.1 indexer inputs on side streams (SGLANG_OPT_DSV41_LOW_LATENCY_INDEXER).

CPU: wk switches to ll_bf16 at 1-8 rows (rounded to bf16), the DeepGEMM decode data
takes a prepared index query, the split projection keeps the stock order, and the
side stream is used only with the flag and a DeepGEMM decode scorer. GPU (SM100 with
CuTe DSL): wk parity at N = 128, K = 512 (muLLM b5a4d16f
tests/models/test_deepseek_v41_low_latency_aux_projections.py::test_kernel_parity at
the wk shape), and the index query built on a side stream, eagerly and replayed from
a CUDA graph, equals the stock serial one bit for bit.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.gemm import ll_bf16
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer
from sglang.srt.layers.attention.dsv4.v41_indexer import scoring
from sglang.srt.layers.attention.dsv4.v41_indexer.types import DecodeInputs
from sglang.srt.models.deepseek_v4 import MQALayer
from sglang.test.test_utils import CustomTestCase

HIDDEN = 5120
Q_LORA = 1024
HEAD_DIM = 512
INDEX_HEADS = 64


def _indexer(*, hidden=64, q_lora=32, owns_k=True) -> DeepseekV41Indexer:
    config = SimpleNamespace(
        index_n_heads=INDEX_HEADS,
        index_head_dim=128,
        qk_rope_head_dim=64,
        index_topk=512,
        kv_source_layer_ids=[2] if owns_k else [],
        candidate_source_layer_id=-1,
        candidate_topk_blocks=0,
        candidate_block_size=0,
        q_lora_rank=q_lora,
        hidden_size=hidden,
        rms_norm_eps=1e-6,
    )
    return DeepseekV41Indexer(
        config, layer_id=2, head_dim=HEAD_DIM, quant_config=None, prefix="indexer"
    )


class TestWkRouting(CustomTestCase):
    def test_flag_off_keeps_the_stock_gemm(self):
        with envs.SGLANG_OPT_DSV41_LOW_LATENCY_INDEXER.override(False):
            indexer = _indexer()
        self.assertEqual(indexer.wk_low_latency_max_m, 0)
        with patch.object(ll_bf16, "ll_bf16_gemm") as ll:
            k = indexer.forward_wk(torch.randn(6, HEAD_DIM, dtype=torch.bfloat16))
        ll.assert_not_called()
        self.assertEqual((k.dtype, k.shape), (torch.bfloat16, (6, 128)))

    def test_flag_on_only_for_index_key_owners_with_the_kernel(self):
        with (
            envs.SGLANG_OPT_DSV41_LOW_LATENCY_INDEXER.override(True),
            patch.object(ll_bf16, "is_available", return_value=True),
        ):
            self.assertEqual(_indexer().wk_low_latency_max_m, 8)
            self.assertEqual(_indexer(owns_k=False).wk_low_latency_max_m, 0)
        with (
            envs.SGLANG_OPT_DSV41_LOW_LATENCY_INDEXER.override(True),
            patch.object(ll_bf16, "is_available", return_value=False),
        ):
            self.assertEqual(_indexer().wk_low_latency_max_m, 0)

    def test_switches_at_1_to_8_rows_and_rounds_to_bf16(self):
        indexer = _indexer()
        indexer.wk_low_latency_max_m = 8

        def fake(x, w):
            return torch.full((x.shape[0], w.shape[0]), 1.0 + 2**-12)

        with patch.object(ll_bf16, "ll_bf16_gemm", side_effect=fake) as ll:
            k = indexer.forward_wk(torch.zeros(8, HEAD_DIM, dtype=torch.bfloat16))
            self.assertEqual(k.dtype, torch.bfloat16)
            self.assertTrue(torch.all(k == 1.0))
            indexer.forward_wk(torch.zeros(9, HEAD_DIM, dtype=torch.bfloat16))
            indexer.forward_wk(torch.zeros(HEAD_DIM, 4, dtype=torch.bfloat16).T)
        self.assertEqual(ll.call_count, 1)


def _decode_inputs(indexer, prepared_q=None) -> DecodeInputs:
    return DecodeInputs(
        indexer=indexer,
        layer_id=2,
        compress_ratio=2,
        freqs_cis=torch.zeros(16, 32, dtype=torch.complex64),
        x=torch.zeros(6, 64, dtype=torch.bfloat16),
        q_lora=torch.zeros(6, 32, dtype=torch.bfloat16),
        positions=torch.arange(6),
        req_rows=torch.zeros(6, dtype=torch.int64),
        paged_metadata=SimpleNamespace(compressed_page_size=64),
        is_verify=True,
        out_page_indices=torch.empty(6, 512, dtype=torch.int32),
        prepared_q=prepared_q,
    )


class TestPreparedIndexQuery(CustomTestCase):
    def _pool(self):
        pool = MagicMock()
        pool.get_index_k_with_scale_buffer.return_value = torch.zeros(4, 64 * 68)
        return pool

    def test_decode_data_takes_the_prepared_query(self):
        indexer = _indexer()
        prepared = (
            torch.ones(6, INDEX_HEADS, 64, dtype=torch.int8),
            torch.ones(6, INDEX_HEADS, dtype=torch.int32),
            torch.ones(6, INDEX_HEADS),
        )
        with patch.object(scoring, "_index_q_and_weights") as stock:
            data = scoring.get_deep_gemm_decode_data(
                _decode_inputs(indexer, prepared), self._pool()
            )
        stock.assert_not_called()
        self.assertEqual(data.q_fp4.data_ptr(), prepared[0].data_ptr())
        self.assertEqual(data.q_fp4.shape, (6, 1, INDEX_HEADS, 64))
        self.assertIs(data.weights, prepared[2])

    def test_without_it_the_scorer_projects_itself(self):
        indexer = _indexer()
        out = (
            torch.zeros(6, INDEX_HEADS, 64, dtype=torch.int8),
            torch.zeros(6, INDEX_HEADS, dtype=torch.int32),
            torch.zeros(6, INDEX_HEADS),
        )
        with patch.object(scoring, "_index_q_and_weights", return_value=out) as stock:
            scoring.get_deep_gemm_decode_data(_decode_inputs(indexer), self._pool())
        stock.assert_called_once()

    def test_split_projection_keeps_the_stock_order(self):
        indexer = _indexer()
        calls = []

        class _WqB(torch.nn.Module):
            def forward(self, q):
                calls.append("wq_b")
                return torch.zeros(6, 1), None

        indexer.wq_b = _WqB()
        with (
            patch.object(
                DeepseekV41Indexer,
                "head_weights_raw",
                side_effect=lambda x: calls.append("weights_proj") or "w",
            ),
            patch.object(scoring, "index_q_rope_pack", return_value="packed") as pack,
        ):
            got = scoring._index_q_and_weights(
                indexer=indexer,
                x="x",
                q_lora="q_lora",
                freqs_cis="f",
                positions="p",
            )
        self.assertEqual(got, "packed")
        self.assertEqual(calls, ["wq_b", "weights_proj"])
        self.assertEqual(pack.call_args.kwargs["head_weights_raw"], "w")


class TestSideStreamGate(CustomTestCase):
    def _layer(self, *, flag=True, indexer=True):
        return SimpleNamespace(
            _low_latency_indexer=flag,
            indexer=object() if indexer else None,
            alt_streams=["kv", "index_q", "sources"],
        )

    def test_gate(self):
        deep_gemm = SimpleNamespace(low_ratio_decode_reads_prepared_q=True)
        torch_scorer = SimpleNamespace(low_ratio_decode_reads_prepared_q=False)
        gate = MQALayer._index_q_side_stream
        self.assertEqual(gate(self._layer(), deep_gemm), "index_q")
        self.assertIsNone(gate(self._layer(flag=False), deep_gemm))
        self.assertIsNone(gate(self._layer(indexer=False), deep_gemm))
        self.assertIsNone(gate(self._layer(), torch_scorer))


def _gpu_ok() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (10, 0)
        and ll_bf16.is_available()
    )


@unittest.skipUnless(_gpu_ok(), "needs an SM100+ GPU with CuTe DSL")
class TestLowLatencyIndexerGpu(CustomTestCase):
    def test_wk_kernel_parity(self):
        for m in range(1, 9):
            torch.manual_seed(128 + m)
            x = torch.randn(m, HEAD_DIM, device="cuda").bfloat16()
            w = (torch.randn(128, HEAD_DIM, device="cuda") / HEAD_DIM**0.5).bfloat16()
            out = ll_bf16.ll_bf16_gemm(x, w)
            self.assertEqual((out.dtype, out.shape), (torch.float32, (m, 128)))
            torch.testing.assert_close(
                out, x.float() @ w.float().T, atol=1e-4, rtol=1e-4
            )
            cublas = torch.mm(x, w.T, out_dtype=torch.float32)
            torch.testing.assert_close(out, cublas, atol=1e-4, rtol=1e-4)

    def test_side_stream_index_query_matches_the_serial_one(self):
        torch.manual_seed(0)
        indexer = _indexer(hidden=HIDDEN, q_lora=Q_LORA).cuda()
        for p in indexer.parameters():
            torch.nn.init.normal_(p, std=0.02)
        m = 6
        x = torch.randn(m, HIDDEN, device="cuda").bfloat16()
        q_lora = torch.randn(m, Q_LORA, device="cuda").bfloat16()
        freqs = torch.polar(
            torch.ones(64, 32, device="cuda"), torch.rand(64, 32, device="cuda")
        )
        pos = torch.arange(m, device="cuda") + 40
        side = torch.cuda.Stream()

        def serial():
            return scoring._index_q_and_weights(
                indexer=indexer, x=x, q_lora=q_lora, freqs_cis=freqs, positions=pos
            )

        def split():
            main = torch.cuda.current_stream()
            side.wait_stream(main)
            with torch.cuda.stream(side):
                torch.cuda._sleep(1_000_000)  # exposes a missing join
                raw = indexer.head_weights_raw(x)
                q, _ = indexer.wq_b(q_lora)
                out = scoring.index_q_rope_pack(
                    indexer=indexer,
                    q=q,
                    head_weights_raw=raw,
                    freqs_cis=freqs,
                    positions=pos,
                )
            main.wait_stream(side)
            return out

        expect = [t.clone() for t in serial()]
        got = split()
        for a, b in zip(got, expect):
            self.assertTrue(torch.equal(a, b))

        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            split()
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = split()
        for _ in range(3):
            x.normal_()
            q_lora.normal_()
            expect = [t.clone() for t in serial()]
            graph.replay()
            torch.cuda.synchronize()
            for a, b in zip(graph_out, expect):
                self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
