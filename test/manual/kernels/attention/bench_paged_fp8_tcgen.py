# SPDX-License-Identifier: Apache-2.0
"""Cold-bank, paired-event timing of paged FP8 attention with FP64 guards."""

import argparse
import json
import math
import statistics
from pathlib import Path

import flashinfer.decode as fd
import torch

from sglang.kernels.ops.attention.paged_fp8_tcgen import PagedFP8DecodePlan, paged_fp8_decode
from sglang.kernels.ops.attention.split_kv_verify import allocate_scratch, split_kv_verify


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heads",type=int,required=True)
    p.add_argument("--qlen",type=int,required=True)
    p.add_argument("--length",type=int,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    assert a.length % 64 == 0
    torch.manual_seed(a.seed)
    q = (torch.randn(a.qlen,a.heads,256,device="cuda")*.5).to(torch.float8_e4m3fn)
    pages = a.length // 64
    banks = [tuple((torch.randn(pages,1,64,256,device="cuda")*.5).to(torch.float8_e4m3fn) for _ in range(2)) for _ in range(16)]
    table = torch.randperm(pages,dtype=torch.int32,device="cuda")[None]
    lengths = torch.tensor([a.length],dtype=torch.int32,device="cuda")
    workspace = torch.empty(512*1024**2,dtype=torch.uint8,device="cuda")
    counter = torch.zeros(8*1024**2,dtype=torch.uint8,device="cuda")
    plan = PagedFP8DecodePlan(q,banks[0],table,lengths)
    cand = [torch.empty_like(q,dtype=torch.bfloat16) for _ in banks]
    native = [torch.empty_like(q,dtype=torch.bfloat16) for _ in banks]
    def run_native(i):
        return fd.trtllm_batch_decode_with_kv_cache(query=q,kv_cache=banks[i],workspace_buffer=workspace,
            block_tables=table,seq_lens=lengths,max_seq_len=262144,bmm1_scale=1/16,bmm2_scale=1.,
            window_left=-1,out=native[i],out_dtype=torch.bfloat16,q_len_per_req=a.qlen,
            multi_ctas_kv_counter_buffer=counter)
    def run_candidate(i):
        return paged_fp8_decode(q,banks[i],table,lengths,plan,qk_scale=1/16,v_scale=1.,out=cand[i])
    def capture(fn):
        for i in range(16):fn(i)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(48):fn(i%16)
        for _ in range(3):g.replay()
        return g
    graphs = {"native":capture(run_native),"candidate":capture(run_candidate),"null":capture(run_native)}
    if a.qlen == 7:
        split_scratch = allocate_scratch(q,1,148,a.qlen)
        split_outputs = [torch.empty_like(q,dtype=torch.bfloat16) for _ in banks]
        def run_split(i):
            return split_kv_verify(q,banks[i],table,lengths,split_outputs[i],split_scratch,
                splits=148,qlen=a.qlen,block_m=64,block_n=128,qk_scale=1/16,v_scale=1.,
                num_stages=2,skip_empty=True,short_merge=True,short_tiles=True,merge_tile=64)
        graphs["served_split"] = capture(run_split)
    graphs["candidate"].replay();graphs["native"].replay();torch.cuda.synchronize()
    checks = []
    rows = a.qlen*a.heads
    for i,(k,v) in enumerate(banks):
        keys = k[table[0].long(),0].reshape(-1,256).double()
        vals = v[table[0].long(),0].reshape(-1,256).double()
        scores = q.reshape(rows,256).double() @ keys.T /16
        qp = torch.arange(rows,device="cuda")//a.heads
        kp = torch.arange(a.length,device="cuda")
        scores.masked_fill_(kp[None]>a.length-a.qlen+qp[:,None],-float("inf"))
        ref = scores.softmax(-1) @ vals
        c = cand[i].reshape(rows,256).double()-ref
        s = native[i].reshape(rows,256).double()-ref
        cr,sr = c.square().mean(-1).sqrt(),s.square().mean(-1).sqrt()
        rc,rs = float(c.square().mean().sqrt()),float(s.square().mean().sqrt())
        finite = bool(torch.isfinite(c).all() and torch.isfinite(s).all())
        checks.append(dict(bank=i,rms_ratio=rc/rs,worst_row_ratio=float((cr/sr).max()),
                           maxabs_ratio=float(c.abs().max()/s.abs().max()),
                           passed=bool(finite and rc<=1.05*rs and (cr<=2*sr).all() and c.abs().max()<=2*s.abs().max())))
    info = dict(args=vars(a)|{"output":str(a.output)},bank_checks=checks,
                numeric_pass=all(x["passed"] for x in checks),samples_us={})
    a.output.write_text(json.dumps(info,indent=2))
    assert info["numeric_pass"], "Cold-bank numeric floor; no timing"
    samples = {k:[] for k in graphs}
    for trial in range(9):
        order = list(graphs) if trial%2==0 else list(graphs)[::-1]
        for name in order:
            begin,end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            begin.record()
            for _ in range(4):graphs[name].replay()
            end.record();end.synchronize()
            samples[name].append(begin.elapsed_time(end)*1000/192)
    info.update(samples_us=samples,median_us={k:statistics.median(v) for k,v in samples.items()})
    info["null_us"] = info["median_us"]["null"]-info["median_us"]["native"]
    info["saving_us"] = info["median_us"]["native"]-info["median_us"]["candidate"]
    a.output.write_text(json.dumps(info,indent=2))
    print(json.dumps({k:info[k] for k in ["median_us","null_us","saving_us","numeric_pass"]}),flush=True)


if __name__ == "__main__":
    main()
