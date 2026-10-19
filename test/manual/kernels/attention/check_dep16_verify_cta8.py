# SPDX-License-Identifier: Apache-2.0
"""Manual q7 native-launcher qualification, including the full frozen cohort.

The FP64 oracle uses the quantized Q/K/V values, the native causal query window,
and the same scalar scales. Each row/head and the global RMS must be within
twice the matched stock error; no absolute epsilon is added.
"""

import argparse
import json
import sys
from pathlib import Path

import flashinfer.decode as fd
import torch


def banks_for(lengths, banks=1, layout="hnd"):
    counts = [(x + 63) // 64 for x in lengths]
    pages = sum(counts)
    permutation = torch.randperm(pages, dtype=torch.int32, device="cuda")
    table = torch.zeros((len(lengths), max(counts)), dtype=torch.int32, device="cuda")
    offset = 0
    for i, n in enumerate(counts):
        table[i, :n] = permutation[offset : offset + n]
        offset += n
    kv = [
        tuple(
            (
                torch.randn(pages, 2, 64, 256, dtype=torch.bfloat16, device="cuda")
                * 0.5
            ).to(torch.float8_e4m3fn)
            for _ in range(2)
        )
        for _ in range(banks)
    ]
    if layout == "nhd":
        physical = []
        for bank in kv:
            views = []
            for value in bank:
                storage = torch.empty(
                    (pages, 64, 2, 256), dtype=value.dtype, device=value.device
                )
                view = storage.permute(0, 2, 1, 3)
                view.copy_(value)
                views.append(view)
            physical.append(tuple(views))
        kv = physical
    q = (torch.randn(len(lengths) * 7, 32, 256, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    return q, kv, table, torch.tensor(lengths, dtype=torch.int32, device="cuda")


def oracle(q, kv, table, lengths, qk_scale, v_scale):
    outputs = []
    q = q.view(len(lengths), 7, 32, 256)
    for row, length in enumerate(lengths):
        ids = table[row, : (length + 63) // 64].long()
        k, v = [
            x.index_select(0, ids)
            .permute(1, 0, 2, 3)
            .reshape(2, -1, 256)[:, :length]
            .double()
            for x in kv
        ]
        qq = q[row].permute(1, 0, 2).reshape(2, 16 * 7, 256).double()
        scores = torch.bmm(qq, k.transpose(1, 2)) * qk_scale
        positions = torch.arange(length, device=q.device)
        ends = length - 7 + 1 + torch.arange(7, device=q.device).repeat(16)
        scores.masked_fill_(
            positions[None, None, :] >= ends[None, :, None], float("-inf")
        )
        ref = torch.bmm(torch.softmax(scores, dim=-1), v) * v_scale
        outputs.append(ref.view(2, 16, 7, 256).permute(2, 0, 1, 3).reshape(7, 32, 256))
    return torch.cat(outputs)


def judge(stock, candidate, reference):
    a = (stock.double() - reference).square().mean(-1).sqrt()
    b = (candidate.double() - reference).square().mean(-1).sqrt()
    ag = (stock.double() - reference).square().mean().sqrt()
    bg = (candidate.double() - reference).square().mean().sqrt()
    passed = bool(
        torch.isfinite(candidate).all()
        and torch.isfinite(reference).all()
        and (b <= 2 * a).all()
        and bg <= 2 * ag
    )
    bad = (b > 2 * a).nonzero()
    return dict(
        passed=passed,
        stock_global_rms=float(ag),
        candidate_global_rms=float(bg),
        worst_row_ratio=float(
            torch.where(a > 0, b / a, torch.where(b == 0, 0.0, float("inf"))).max()
        ),
        bad_rows=int(bad.shape[0]),
        first_bad=bad[0].tolist() if bad.numel() else None,
    )


def run(args):
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    workspace = torch.zeros(512 * 1024**2, dtype=torch.uint8, device="cuda")
    counter = torch.zeros(8 * 1024**2, dtype=torch.uint8, device="cuda")
    module_name = "sglang.kernels.ops.attention.dep16_verify_cta8"
    original_entry = fd.trtllm_batch_decode_with_kv_cache
    original_getter = fd.get_trtllm_gen_fmha_module
    if args.mode == "off":
        # Exercise the actual backend method, not an independently rewritten OFF wrapper.
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        assert not envs.SGLANG_Q35MF_K4_DEP16_VERIFY_CTA8.get()
        backend = object.__new__(TRTLLMHAAttnBackend)
        backend.q_data_type = torch.bfloat16
        backend.workspace_buffer = workspace
        backend._multi_ctas_kv_counter_buffer = counter
        backend.max_context_len = 262144
        backend.decode_seq_len_splits = 1
        backend.decode_split_hoist = False
        backend._dep16_verify_cta8 = None
        q, kv, table, seq = banks_for([257] * 8, layout=args.layout)
        kw = dict(
            query=q,
            kv_cache=kv[0],
            block_tables=table,
            seq_lens=seq,
            bmm1_scale=1 / 16,
            bmm2_scale=1.0,
            window_left=-1,
            sinks=None,
            q_len_per_req=7,
            out_dtype=torch.bfloat16,
        )
        stock = original_entry(
            **kw,
            workspace_buffer=workspace,
            max_seq_len=262144,
            multi_ctas_kv_counter_buffer=counter,
        )
        observed = backend._run_fixed_q_len_decode(**kw, use_dep16_verify_cta8=True)
        assert torch.equal(stock.view(torch.uint8), observed.view(torch.uint8))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replay_out = backend._run_fixed_q_len_decode(
                **kw, use_dep16_verify_cta8=True
            )
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(stock.view(torch.uint8), replay_out.view(torch.uint8))
        mutated = replay_out.clone()
        mutated.view(torch.uint8).flatten()[0] ^= 1
        assert not torch.equal(stock.view(torch.uint8), mutated.view(torch.uint8))
        assert module_name not in sys.modules, "OFF imported the extension"
        assert (
            fd.trtllm_batch_decode_with_kv_cache is original_entry
            and fd.get_trtllm_gen_fmha_module is original_getter
        )
        print(
            json.dumps(dict(mode="off", passed=True, extension_imported=False)),
            flush=True,
        )
        return

    if args.stock_only:
        trtllm_dep16_verify_cta8 = original_entry
    else:
        from sglang.kernels.ops.attention.dep16_verify_cta8 import (
            prepare_dep16_verify_cta8,
            trtllm_dep16_verify_cta8,
        )

        module = prepare_dep16_verify_cta8(torch.device("cuda"))
        assert module is prepare_dep16_verify_cta8(torch.device("cuda", 0)), (
            "device aliases must share prewarm"
        )
    entries = []
    if args.mode == "cohort":
        fixture = json.loads(args.fixture.read_text())
        entries = [
            dict(snapshot=s["time_utc"], rank=rank, lengths=lens)
            for s in fixture["snapshots"]
            for rank, lens in s["ranks"].items()
        ]
        assert len(entries) == 48
    else:
        entries = [
            dict(lengths=[n] * batch)
            for batch, n in [(3, 17), (8, 257), (12, 2049), (13, 8193)]
        ]
    for index, row in enumerate(entries):
        lengths = row["lengths"]
        q, banks, table, seq = banks_for(
            lengths, 3 if args.mode == "cohort" else 1, layout=args.layout
        )
        for bank, kv in enumerate(banks):
            scalar_qk, scalar_v = 1 / 16, 1.0
            variants = [False, True] if args.mode == "controls" else [False]
            for tensor_scales in variants:
                qk = (
                    torch.tensor(scalar_qk, device="cuda", dtype=torch.float32)
                    if tensor_scales
                    else scalar_qk
                )
                vs = (
                    torch.tensor(scalar_v, device="cuda", dtype=torch.float32)
                    if tensor_scales
                    else scalar_v
                )
                query = q
                cache = kv
                if tensor_scales:
                    q_storage = torch.full(
                        (q.shape[0], 2, 32, 256),
                        float("nan"),
                        dtype=q.dtype,
                        device=q.device,
                    )
                    query = q_storage[:, 0]
                    query.copy_(q)
                    padded = []
                    for value in kv:
                        physical_shape = (
                            (value.shape[0], 2, 64, 2, 256)
                            if args.layout == "nhd"
                            else (value.shape[0], 2, 2, 64, 256)
                        )
                        storage = torch.full(
                            physical_shape,
                            float("nan"),
                            dtype=value.dtype,
                            device=value.device,
                        )
                        view = storage[:, 0]
                        if args.layout == "nhd":
                            view = view.permute(0, 2, 1, 3)
                        view.copy_(value)
                        padded.append(view)
                    cache = tuple(padded)
                kw = dict(
                    query=query,
                    kv_cache=cache,
                    workspace_buffer=workspace,
                    block_tables=table,
                    seq_lens=seq,
                    max_seq_len=262144,
                    bmm1_scale=qk,
                    bmm2_scale=vs,
                    out_dtype=torch.bfloat16,
                    q_len_per_req=7,
                    multi_ctas_kv_counter_buffer=counter,
                )
                # Sanitizers isolate the selected arm; numerical modes also run stock.
                stock = original_entry(**kw) if args.mode != "sanitize" else None
                candidate = trtllm_dep16_verify_cta8(**kw)
                torch.cuda.synchronize()
                saved = candidate.clone()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    replay_out = trtllm_dep16_verify_cta8(**kw)
                for _ in range(3 if args.mode == "cohort" else 16):
                    graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(
                    saved.view(torch.uint8), replay_out.view(torch.uint8)
                ), "eager/graph mismatch"
                if args.mode != "sanitize":
                    ref = oracle(query, cache, table, lengths, scalar_qk, scalar_v)
                    result = judge(stock, candidate, ref)
                    if args.mode == "controls":
                        corrupted = candidate.clone()
                        corrupted[0, 0, 0] = 1024.0
                        assert not judge(stock, corrupted, ref)["passed"], (
                            "numeric negative control escaped"
                        )
                    record = dict(
                        index=index,
                        bank=bank,
                        batch=len(lengths),
                        tensor_scales=tensor_scales,
                        layout=args.layout,
                        kv_strides=[list(v.stride()) for v in cache],
                        **row,
                        **result,
                    )
                    print(json.dumps(record), flush=True)
                    if args.out:
                        with open(args.out, "a") as f:
                            f.write(json.dumps(record) + "\n")
                    assert result["passed"], "same-floor numerical gate failed"
                    del ref
                else:
                    assert torch.isfinite(candidate).all()
                    print(
                        json.dumps(
                            dict(
                                index=index,
                                bank=bank,
                                batch=len(lengths),
                                layout=args.layout,
                                stock_only=args.stock_only,
                                passed=True,
                            )
                        ),
                        flush=True,
                    )
                del graph, replay_out, saved, stock, candidate
        del banks, q
        torch.cuda.empty_cache()
    assert (
        fd.trtllm_batch_decode_with_kv_cache is original_entry
        and fd.get_trtllm_gen_fmha_module is original_getter
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", required=True, choices=["off", "controls", "cohort", "sanitize"]
    )
    ap.add_argument("--fixture", type=Path)
    ap.add_argument("--layout", choices=["hnd", "nhd"], default="hnd")
    ap.add_argument("--stock-only", action="store_true")
    ap.add_argument("--out", type=Path)
    run(ap.parse_args())
