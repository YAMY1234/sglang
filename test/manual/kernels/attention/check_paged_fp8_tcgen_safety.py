# SPDX-License-Identifier: Apache-2.0
"""Exercise every narrow paged-FP8 instantiation through eager and graph replay."""

import argparse
import json

import flashinfer.decode as fd
import torch

from sglang.kernels.ops.attention.paged_fp8_tcgen import PagedFP8DecodePlan, paged_fp8_decode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setup-only", action="store_true")
    p.add_argument("--heads",type=int,nargs="+",default=[4,8])
    p.add_argument("--qlens",type=int,nargs="+",default=[1,7])
    a = p.parse_args()
    torch.manual_seed(42)
    workspace = torch.empty(512 * 1024**2, dtype=torch.uint8, device="cuda")
    counter = torch.zeros(8 * 1024**2, dtype=torch.uint8, device="cuda")
    k, v = [(torch.randn(2048, 1, 64, 256, device="cuda") * 8).to(torch.float8_e4m3fn) for _ in range(2)]
    table = torch.full((1,2050),2147483647,dtype=torch.int32,device="cuda")
    table[0,:2048] = torch.randperm(2048,dtype=torch.int32,device="cuda")
    lengths = torch.tensor([131072], dtype=torch.int32,device="cuda")
    for heads in a.heads:
        for qlen in a.qlens:
            q = torch.randn(qlen,heads,256,device="cuda").to(torch.float8_e4m3fn)
            out = torch.empty_like(q,dtype=torch.bfloat16)
            print("PLAN_BEGIN",heads,qlen,flush=True)
            plan = PagedFP8DecodePlan(q,(k,v),table,lengths)
            print("PLAN_COMPILED",heads,qlen,flush=True)
            def run():
                if a.setup_only:
                    return fd.trtllm_batch_decode_with_kv_cache(
                        query=q,kv_cache=(k,v),workspace_buffer=workspace,
                        block_tables=table,seq_lens=lengths,max_seq_len=262144,
                        bmm1_scale=.002,bmm2_scale=.03,window_left=-1,
                        out=out,out_dtype=torch.bfloat16,q_len_per_req=qlen,
                        multi_ctas_kv_counter_buffer=counter)
                return paged_fp8_decode(q,(k,v),table,lengths,plan,
                                         qk_scale=.002,v_scale=.03,out=out)
            for _ in range(3): run()
            torch.cuda.synchronize()
            print("EAGER_COMPLETE",heads,qlen,flush=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): run()
            for length in (2049,2050,2111,2112,65536,131072):
                lengths.fill_(length)
                expected = run().clone()
                for _ in range(3):graph.replay()
                torch.cuda.synchronize()
                assert torch.isfinite(out).all()
                if not a.setup_only:
                    assert torch.equal(expected,out), (heads,qlen,length)
                print(json.dumps(dict(heads=heads,qlen=qlen,length=length,
                                      setup_only=a.setup_only,finite=True,
                                      eager_graph_bitwise=bool(torch.equal(expected,out)))),flush=True)
            graph.reset()
    print("SAFETY_CASES_PASS",flush=True)


if __name__ == "__main__":
    main()
