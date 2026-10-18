# SPDX-License-Identifier: Apache-2.0
"""Check production-layout paged FP8 attention, including scaled partial pages."""

import argparse
import json
import math
from pathlib import Path

import flashinfer.decode as fd
import torch
import triton
import triton.testing

from sglang.kernels.ops.attention.paged_fp8_tcgen import PagedFP8DecodePlan, paged_fp8_decode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--qlen", type=int, default=7)
    p.add_argument("--lengths", default="131072,2049")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--qk-scale", type=float, default=0.0625)
    p.add_argument("--v-scale", type=float, default=1.0)
    p.add_argument("--k-descaling", type=float, default=1.0)
    p.add_argument("--v-descaling", type=float, default=1.0)
    p.add_argument("--outliers", action="store_true")
    p.add_argument("--timing", action="store_true")
    p.add_argument("--setup-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    workspace = torch.empty(512 * 1024**2, dtype=torch.uint8, device="cuda")
    counter = torch.zeros(8 * 1024**2, dtype=torch.uint8, device="cuda")
    records = []
    for length in map(int, a.lengths.split(",")):
        for seed in map(int, a.seeds.split(",")):
            torch.manual_seed(seed)
            pages = triton.cdiv(length, 64)
            # Poison the unused page-table tail; the producer must not read it.
            table = torch.full((1, triton.cdiv(pages, 2) * 2 + 2), 2147483647,
                               dtype=torch.int32, device="cuda")
            table[0, :pages] = torch.randperm(pages, dtype=torch.int32, device="cuda")
            qf = torch.randn(a.qlen, a.heads, 256, device="cuda") * .5
            kf = torch.randn(pages, 1, 64, 256, device="cuda") * .5
            vf = torch.randn_like(kf) * .5
            if a.outliers:
                qf[..., ::31] *= 4
                kf[..., ::31] *= 4
                vf[..., ::29] *= 4
            kf /= a.k_descaling
            vf /= a.v_descaling
            q, k, v = [x.clamp(-448,448).to(torch.float8_e4m3fn) for x in (qf, kf, vf)]
            lens = torch.tensor([length], dtype=torch.int32, device="cuda")
            stock_out = torch.empty_like(q, dtype=torch.bfloat16)
            def stock():
                return fd.trtllm_batch_decode_with_kv_cache(
                    query=q, kv_cache=(k, v), workspace_buffer=workspace,
                    block_tables=table, seq_lens=lens, max_seq_len=262144,
                    bmm1_scale=a.qk_scale, bmm2_scale=a.v_scale,
                    window_left=-1, out=stock_out, out_dtype=torch.bfloat16,
                    q_len_per_req=a.qlen, multi_ctas_kv_counter_buffer=counter)
            stock()
            if a.setup_only:
                torch.cuda.synchronize()
                continue
            plan = PagedFP8DecodePlan(q, (k, v), table, lens)
            print("COMPILED", a.heads, a.qlen, length, seed, flush=True)
            def candidate():
                return paged_fp8_decode(q, (k, v), table, lens, plan,
                                        qk_scale=a.qk_scale, v_scale=a.v_scale)
            candidate()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                candidate()
            graph.replay()
            torch.cuda.synchronize()
            keys = k[table[0, :pages].long(), 0].reshape(-1, 256)[:length].double()
            vals = v[table[0, :pages].long(), 0].reshape(-1, 256)[:length].double()
            scores = q.reshape(-1, 256).double() @ keys.T * a.qk_scale
            qpos = torch.arange(a.qlen * a.heads, device="cuda") // a.heads
            kvpos = torch.arange(length, device="cuda")
            scores.masked_fill_(kvpos[None] > length - a.qlen + qpos[:, None], -float("inf"))
            ref = scores.softmax(-1) @ vals * a.v_scale
            c = plan.output.reshape(-1, 256).double() - ref
            s = stock_out.reshape(-1, 256).double() - ref
            cr, sr = c.square().mean(-1).sqrt(), s.square().mean(-1).sqrt()
            rms_c, rms_s = float(c.square().mean().sqrt()), float(s.square().mean().sqrt())
            finite = bool(torch.isfinite(plan.output).all() and torch.isfinite(stock_out).all())
            row = dict(heads=a.heads, qlen=a.qlen, length=length, seed=seed,
                       qk_scale=a.qk_scale, v_scale=a.v_scale, outliers=a.outliers,
                       nan_rows=torch.isnan(plan.output.reshape(-1,256)).any(-1).nonzero().flatten().tolist(),
                       nan_partial_splits=torch.isnan(plan.partial).any(-1).any(-1).nonzero().flatten().tolist(),
                       finite=finite, candidate_rms=rms_c, stock_rms=rms_s,
                       rms_ratio=rms_c/rms_s, worst_row_ratio=float((cr/sr).max()),
                       max_abs_ratio=float(c.abs().max()/s.abs().max()),
                       stage1_floor=bool(finite and rms_c <= 2*rms_s and (cr <= 2*sr).all()),
                       acc_t0=bool(finite and rms_c <= 1.05*rms_s and c.abs().max() <= 2*s.abs().max()))
            records.append(row)
            a.output.write_text(json.dumps(records, indent=2))
            print(json.dumps(row), flush=True)
            if not row["stage1_floor"] or not row["acc_t0"]:
                raise RuntimeError("Registered numerical floor failed; no timing")
            if a.timing:
                times = {}
                for name, fn in (("native_null_a", stock), ("candidate", candidate), ("native_null_b", stock)):
                    times[name] = [triton.testing.do_bench_cudagraph(fn, rep=2000) * 1000 for _ in range(3)]
                row["timing_us"] = times
                a.output.write_text(json.dumps(records, indent=2))
                print(json.dumps(times), flush=True)
    print("PASS", len(records), flush=True)


if __name__ == "__main__":
    main()
