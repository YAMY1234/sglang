"""Bucket selection must capture the same decode kernels as each fixed policy."""

import json
import math
import os
import pathlib
import types
import unittest
from functools import partial
from unittest.mock import patch

import flashinfer
import torch
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestAdaptiveSplits(unittest.TestCase):
    def test_fixed_policy_graph_equivalence(self):
        torch.manual_seed(238)
        records = []
        device = "cuda"
        workspace = torch.zeros(512 << 20, dtype=torch.uint8, device=device)
        counter = torch.zeros(1 << 20, dtype=torch.uint8, device=device)
        real_decode = flashinfer.decode.trtllm_batch_decode_with_kv_cache
        calls = []

        def observe(**kwargs):
            calls.append(kwargs["seq_lens"].shape[0])
            return real_decode(**kwargs)

        def random_fp8(shape):
            return (torch.randn(shape, device=device) * 0.1).to(torch.float8_e4m3fn)

        with patch.object(
            flashinfer.decode, "trtllm_batch_decode_with_kv_cache", observe
        ):
            for batch in (8, 16):
                for width in (1, 8):
                    with self.subTest(batch=batch, width=width):
                        pages_per_req = 128
                        pages = batch * pages_per_req
                        query = random_fp8((batch * width, 32, 256))
                        # Keep the serving HND view over physically NHD-contiguous KV.
                        cache = tuple(
                            random_fp8((pages, 64, 2, 256)).permute(0, 2, 1, 3)
                            for _ in range(2)
                        )
                        table = torch.zeros(
                            batch, 4100, dtype=torch.int32, device=device
                        )
                        table[:, :pages_per_req] = torch.arange(
                            pages, dtype=torch.int32, device=device
                        ).view(batch, pages_per_req)
                        lens = torch.tensor(
                            [8192 - (i % 4) * 64 for i in range(batch)],
                            dtype=torch.int32,
                            device=device,
                        )
                        fixed_split = 1 if batch == 8 else 4
                        states = []
                        outs = []
                        graphs = []
                        capture_calls = []
                        for adaptive in (False, True):
                            state = types.SimpleNamespace(
                                workspace_buffer=workspace,
                                max_context_len=262144,
                                q_data_type=torch.bfloat16,
                                _multi_ctas_kv_counter_buffer=counter,
                                decode_seq_len_splits=4 if adaptive else fixed_split,
                                adaptive_decode_splits=adaptive,
                                decode_splits_small_bs=8,
                                fuse_split_gather=True,
                                _split_gather_used_logged=True,
                            )
                            out = torch.zeros(
                                batch * width,
                                32,
                                256,
                                dtype=torch.bfloat16,
                                device=device,
                            )

                            run = partial(
                                TRTLLMHAAttnBackend._run_fixed_q_len_decode,
                                state,
                                query=query,
                                kv_cache=cache,
                                block_tables=table,
                                seq_lens=lens,
                                bmm1_scale=1 / math.sqrt(256),
                                bmm2_scale=1.0,
                                window_left=-1,
                                sinks=None,
                                q_len_per_req=width,
                                out=out,
                            )

                            warm_stream = torch.cuda.Stream()
                            warm_stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(warm_stream):
                                for _ in range(3):
                                    run()
                            torch.cuda.current_stream().wait_stream(warm_stream)
                            torch.cuda.synchronize()
                            graph = torch.cuda.CUDAGraph()
                            before = len(calls)
                            with torch.cuda.graph(graph):
                                run()
                            capture_calls.append(calls[before:])
                            states.append(state)
                            outs.append(out)
                            graphs.append(graph)
                        self.assertEqual(capture_calls[0], capture_calls[1])
                        self.assertEqual(len(capture_calls[0]), fixed_split)
                        checks = []
                        # Reuse each graph with changed inputs and with padded rows.
                        for padding in (0, 2):
                            query.copy_(random_fp8(query.shape))
                            lens[-2:] = 8192 if padding == 0 else 0
                            for repetition in range(3):
                                for out in outs:
                                    out.fill_(repetition + 0.25)
                                before = len(calls)
                                for graph in graphs:
                                    graph.replay()
                                torch.cuda.synchronize()
                                self.assertEqual(
                                    len(calls), before, "Python ran on replay"
                                )
                                self.assertTrue(torch.equal(outs[0], outs[1]))
                                valid = outs[0][: (batch - padding) * width]
                                self.assertTrue(torch.isfinite(valid).all().item())
                                checks.append(
                                    {
                                        "padding": padding,
                                        "replay": repetition,
                                        "bitwise": True,
                                    }
                                )
                        records.append(
                            {
                                "batch": batch,
                                "q_len": width,
                                "selected_split": fixed_split,
                                "capture_group_rows": capture_calls,
                                "checks": checks,
                            }
                        )
                        del graphs, outs, states, cache, table, query, lens
        result = {
            "pass_gate": True,
            "cases": records,
            "torch": torch.__version__,
            "flashinfer": flashinfer.__version__,
            "device": torch.cuda.get_device_name(),
            "semantics": "Same helper, same buffers/inputs; capture fixed1/B8 or fixed4/B16 versus adaptive. Synthetic finite inputs, not full model accuracy.",
        }
        if os.environ.get("Q35_SPLITS_GATE_OUTPUT"):
            pathlib.Path(os.environ["Q35_SPLITS_GATE_OUTPUT"]).write_text(
                json.dumps(result, indent=2) + "\n"
            )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    unittest.main()
