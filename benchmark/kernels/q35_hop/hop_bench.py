"""Zero-skew latency of one decode hop (all-reduce + residual + RMSNorm [+ quant]) per backend.

Each backend is timed with no profiler: N back-to-back calls are captured in one CUDA graph,
and the per-call time is the slope between graphs of N1 and N2 calls, which cancels the
launch skew between ranks. One process per GPU; rank and world come from SLURM, the
rendezvous from MASTER_ADDR / MASTER_PORT. Every rank prints JSON lines.

Backends:
  fi_ar        FlashInfer MNNVL CuTe DSL all-reduce + residual + RMSNorm (verify hop B).
  fi_ar_q      fi_ar followed by the static per-tensor FP8 quant the next GEMM runs.
  fi_fin       FlashInfer MNNVL CuTe DSL MoE finalize + all-reduce + residual + RMSNorm,
               with a gated shared-expert output (verify hop A).
  fi_fin_q     fi_fin followed by the static FP8 quant (hop A as the qkvz input sees it).
  fi_mnnvl     FlashInfer legacy MNNVL one-shot all-reduce + residual + RMSNorm (draft hop).
  quant        the static FP8 quant alone.
  mkh:<name>   one hop from mkh_hop.cu: one (+FP8), one_nq (no FP8), fin / fin_fma (finalize +
               FP8), tu (tagged words + FP8), ldr1 / ldr2 (NVLS ld_reduce + FP8).
  chain:<name> producer + hop pairs from mkh_hop.cu: push (M2), plain_one (M1), plain_fi,
               plain_fiq (today's hop B, without / with quant), prod (producer only).
  check        bitwise comparison of the mkh kernels against FlashInfer and the Triton quant.
"""

import argparse
import json
import os
import socket
import statistics
import sys

import torch
import torch.distributed as dist


def _graph_time(fn, calls, replays):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(replays):
        dist.barrier()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) * 1000.0)
    del graph
    return statistics.median(out), out


def _bench(name, fn, args, ctx, m):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    slopes = []
    for rep in range(args.reps):
        t1, _ = _graph_time(fn, args.n1, args.replays)
        t2, raw2 = _graph_time(fn, args.n2, args.replays)
        per_call = (t2 - t1) / (args.n2 - args.n1)
        slopes.append(per_call)
        print(json.dumps(dict(event="hop", backend=name, rank=ctx["rank"], host=ctx["host"],
                              world=ctx["world"], m=m, hidden=args.hidden, rep=rep, n1=args.n1,
                              n2=args.n2, replays=args.replays, per_call_us=round(per_call, 4),
                              t_n1_us=round(t1, 2), t_n2_us=round(t2, 2),
                              spread_n2_us=round(max(raw2) - min(raw2), 2), tag=args.tag)),
              flush=True)
    return statistics.median(slopes)


def _make_inputs(args, m, dev, rank, seed=0):
    gen = torch.Generator(device=dev).manual_seed(1234 + rank + 1000 * seed)
    x = torch.randn(m, args.hidden, device=dev, dtype=torch.bfloat16, generator=gen) * 0.1
    # The residual is replicated across ranks, as in the model.
    rgen = torch.Generator(device=dev).manual_seed(99 + seed)
    residual = torch.randn(m, args.hidden, device=dev, dtype=torch.bfloat16, generator=rgen)
    wgen = torch.Generator(device=dev).manual_seed(7 + seed)
    # The model hands the fusion the applied multiplier (Gemma 1 + w) with weight_bias 0.
    gamma = (torch.randn(args.hidden, device=dev, dtype=torch.bfloat16, generator=wgen) * 0.1 + 1.0)
    k = args.top_k
    routed = torch.randn(m * k, args.hidden, device=dev, dtype=torch.bfloat16, generator=gen) * 0.01
    weights = torch.rand(m, k, device=dev, dtype=torch.float32, generator=gen).to(torch.bfloat16)
    permuted = torch.randperm(m * k, device=dev, generator=gen).to(torch.int32).view(m, k)
    shared = torch.randn(m, args.hidden, device=dev, dtype=torch.bfloat16, generator=gen) * 0.05
    scale = torch.full((1,), 0.0234375, device=dev, dtype=torch.float32)
    return dict(x=x, residual=residual, gamma=gamma, routed=routed, weights=weights,
                permuted=permuted, shared=shared, scale=scale)


def _diff(a, b):
    a8, b8 = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    if a.dtype == torch.bfloat16:
        a16, b16 = a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)
        n = int((a16 != b16).sum().item())
        mx = float((a.float() - b.float()).abs().max().item())
        return dict(mismatch=n, total=a.numel(), max_abs=mx)
    n = int((a8 != b8).sum().item())
    return dict(mismatch=n, total=a.numel())


def _check(args, ctx, m, t, fusion, mkh, static_quant_fp8, seed=0):
    """Bitwise: mkh one / fin under every reduction-order variant vs FlashInfer + Triton quant."""
    ref_n, ref_r = fusion.all_reduce_residual_rms_norm(
        local_contribution=t["x"], residual=t["residual"], gamma=t["gamma"])
    ref_q, _ = static_quant_fp8(ref_n, t["scale"])
    fin = dict(routed=t["routed"], weights=t["weights"], permuted=t["permuted"], shared=t["shared"])
    fref_n, fref_r = fusion.moe_finalize_all_reduce_rms_norm(
        routed_output=t["routed"], expert_weights=t["weights"], permuted_indices=t["permuted"],
        gated_shared_output=t["shared"], residual=t["residual"], gamma=t["gamma"])
    fref_q, _ = static_quant_fp8(fref_n, t["scale"])
    torch.cuda.synchronize()
    rows = []
    for name, rn, rr, rq in (("one", ref_n, ref_r, ref_q), ("fin", fref_n, fref_r, fref_q),
                             ("split", ref_n, ref_r, ref_q), ("split_fin", fref_n, fref_r, fref_q),
                             ("split_finq", fref_n, fref_r, fref_q)):
        for order in (1,):
            maker = mkh.make_split if name.startswith("split") else mkh.make_call
            call = maker(name, m=m, x=t["x"], residual=t["residual"], gamma=t["gamma"],
                         scale=t["scale"], fin=fin, order=order)
            call()
            torch.cuda.synchronize()
            o = mkh.outputs_of(name, m, order)
            rows.append(dict(event="check", backend=name, order=order, m=m, rank=ctx["rank"], seed=seed,
                             residual=_diff(o["residual"], rr), norm=_diff(o["norm"], rn),
                             fp8=_diff(o["fp8"].view(torch.float8_e4m3fn), rq)))
    for r in rows:
        print(json.dumps(r), flush=True)
    print(json.dumps(dict(event="errors", rank=ctx["rank"], **mkh.errors())), flush=True)


def _report_stamps(backend, m, ctx, stamps):
    torch.cuda.synchronize()
    calls_n, ctas = stamps.shape[0], m * 4
    st = stamps.view(-1)[: calls_n * ctas * 16].view(calls_n, ctas, 16).cpu().tolist()
    calls = [c for c in range(calls_n) if all(x[0] > 0 and x[5] > 0 and x[4] > 0 for x in st[c])]
    if len(calls) < 4:
        return
    order = sorted(calls, key=lambda c: min(x[0] for x in st[c]))
    seg = {"wait_after_entry": (1, 0), "push_after_wait": (2, 1), "pull_after_push": (3, 2),
           "end_after_pull": (5, 3), "rms_after_pull": (4, 3), "end_after_rms": (5, 4)}
    out = {}
    for k, (a, b) in seg.items():
        vals = sorted(x[a] - x[b] for c in order for x in st[c])
        out[k + "_ns_p50"] = vals[len(vals) // 2]
        out[k + "_ns_p90"] = vals[int(len(vals) * 0.9)]
    gaps, spans = [], []
    for c0, c1 in zip(order, order[1:]):
        # last CTA of call c0 ends -> first CTA of call c1 passes its PDL wait
        gaps.append(min(x[1] for x in st[c1]) - max(x[5] for x in st[c0]))
        spans.append(max(x[5] for x in st[c1]) - max(x[5] for x in st[c0]))
    gaps.sort(); spans.sort()
    out["release_gap_ns_p50"] = gaps[len(gaps) // 2]
    out["period_ns_p50"] = spans[len(spans) // 2]
    # per-source arrival after this CTA's push (thread 0 of every CTA), p50 over calls and CTAs
    arr = []
    for r in range(ctx["world"]):
        vals = sorted(x[8 + r] - x[2] for c in order for x in st[c] if x[8 + r] > 0)
        arr.append(vals[len(vals) // 2] if vals else -1)
    out["arrival_after_push_ns_p50"] = arr
    # raw first-CTA stamps (push time, seq) for cross-rank skew analysis
    out["cta0"] = [[st[c][0][2], st[c][0][6], st[c][0][1]] for c in order]
    print(json.dumps(dict(event="stamps", backend=backend, m=m, rank=ctx["rank"], calls=len(order),
                          **out)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="7,1")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-m", type=int, default=16)
    ap.add_argument("--n1", type=int, default=100)
    ap.add_argument("--n2", type=int, default=500)
    ap.add_argument("--replays", type=int, default=10)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--order", type=int, default=0)
    ap.add_argument("--check-seeds", type=int, default=20)
    ap.add_argument("--stamps", type=int, default=0, help="record per-phase globaltimer stamps")
    ap.add_argument("--backends", default="fi_ar,fi_ar_q,fi_fin,fi_fin_q,fi_mnnvl,quant")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    local = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)
    ctx = dict(rank=rank, world=world, host=socket.gethostname())
    backends = args.backends.split(",")
    ms = [int(v) for v in args.m.split(",")]
    print(json.dumps(dict(event="start", torch=torch.__version__, dev=local, backends=backends,
                          ms=ms, argv=sys.argv[1:], **ctx)), flush=True)

    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    need_fusion = any(b.startswith(("fi_ar", "fi_fin", "chain:plain_fi", "q35check")) or b.startswith("check")
                      for b in backends)
    fusion = None
    if need_fusion:
        import sglang.srt.layers.flashinfer_mnnvl_cutedsl as fmc

        class _SpecActive:  # the served stack runs MTP, which selects the safe finalize ordering
            speculative_algorithm = "NEXTN"

        fmc.get_spec = lambda: _SpecActive()
        fusion = fmc.FlashInferMNNVLCuteDSLARFusion(
            hidden_size=args.hidden, top_k=args.top_k, max_m=args.max_m, rms_epsilon=1e-6,
            weight_bias=0.0, process_group=dist.group.WORLD, device=dev,
            include_shared_expert=True,
        )
    ws = None
    if "fi_mnnvl" in backends:
        from flashinfer.comm import (AllReduceFusionPattern, allreduce_fusion,
                                     create_allreduce_fusion_workspace)
        from flashinfer.comm.mnnvl import TorchDistBackend

        ws = create_allreduce_fusion_workspace(
            backend="mnnvl", world_size=world, rank=rank, max_token_num=2048,
            hidden_dim=args.hidden, dtype=torch.bfloat16, gpus_per_node=torch.cuda.device_count(),
            comm_backend=TorchDistBackend(group=dist.group.WORLD), group=dist.group.WORLD,
        )
    mkh = None
    if any(b.startswith(("mkh:", "chain:", "check", "pingpong:", "split:")) for b in backends):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import mkh_hop

        mkh = mkh_hop.MkhHop(hidden=args.hidden, max_m=args.max_m, rank=rank, world=world,
                             device=dev, group=dist.group.WORLD)
        print(json.dumps(dict(event="mkh_build", rank=rank, digest=mkh.digest)), flush=True)
        if args.stamps:
            mkh.stamps = torch.zeros(64, args.max_m * mkh_hop.CTAS, 16, dtype=torch.int64, device=dev)

    q35 = None
    if any(b.startswith("q35:") for b in backends):
        from sglang.srt.layers.q35_hop import Q35Hop

        q35 = Q35Hop(max_m=args.max_m, rms_epsilon=1e-6, process_group=dist.group.WORLD, device=dev)

    results = {}
    for m in ms:
        t = _make_inputs(args, m, dev, rank)
        fin = dict(routed=t["routed"], weights=t["weights"], permuted=t["permuted"],
                   shared=t["shared"])
        box = {}

        def fi_ar():
            box["o"] = fusion.all_reduce_residual_rms_norm(
                local_contribution=t["x"], residual=t["residual"], gamma=t["gamma"])

        def fi_ar_q():
            n, _ = fusion.all_reduce_residual_rms_norm(
                local_contribution=t["x"], residual=t["residual"], gamma=t["gamma"])
            box["q"] = static_quant_fp8(n, t["scale"])

        def fi_fin():
            box["o"] = fusion.moe_finalize_all_reduce_rms_norm(
                routed_output=t["routed"], expert_weights=t["weights"],
                permuted_indices=t["permuted"], gated_shared_output=t["shared"],
                residual=t["residual"], gamma=t["gamma"])

        def fi_fin_q():
            n, _ = fusion.moe_finalize_all_reduce_rms_norm(
                routed_output=t["routed"], expert_weights=t["weights"],
                permuted_indices=t["permuted"], gated_shared_output=t["shared"],
                residual=t["residual"], gamma=t["gamma"])
            box["q"] = static_quant_fp8(n, t["scale"])

        norm_out = torch.empty_like(t["x"])
        res_out = torch.empty_like(t["x"])

        def fi_mnnvl():
            allreduce_fusion(
                input=t["x"], workspace=ws, pattern=AllReduceFusionPattern.kARResidualRMSNorm,
                launch_with_pdl=True, trigger_completion_at_end=False, residual_out=res_out,
                norm_out=norm_out, residual_in=t["residual"], rms_gamma=t["gamma"], rms_eps=1e-6,
                use_oneshot=None,
            )

        def quant():
            box["q"] = static_quant_fp8(t["x"], t["scale"])

        def fi_on(partial, quantize):
            n, _ = fusion.all_reduce_residual_rms_norm(
                local_contribution=partial, residual=t["residual"], gamma=t["gamma"])
            if quantize:
                box["q"] = static_quant_fp8(n, t["scale"])

        table = dict(fi_ar=fi_ar, fi_ar_q=fi_ar_q, fi_fin=fi_fin, fi_fin_q=fi_fin_q,
                     fi_mnnvl=fi_mnnvl, quant=quant)
        for b in backends:
            name = f"{b}@m{m}"
            try:
                if b == "check":
                    _check(args, ctx, m, t, fusion, mkh, static_quant_fp8)
                    continue
                if b.startswith("q35check"):
                    # production module (srt/layers/q35_hop.py) vs FlashInfer + Triton quant, bitwise
                    ref_n, ref_r = fusion.all_reduce_residual_rms_norm(
                        local_contribution=t["x"], residual=t["residual"], gamma=t["gamma"])
                    ref_q, _ = static_quant_fp8(ref_n, t["scale"])
                    fref_n, fref_r = fusion.moe_finalize_all_reduce_rms_norm(
                        routed_output=t["routed"], expert_weights=t["weights"],
                        permuted_indices=t["permuted"], gated_shared_output=t["shared"],
                        residual=t["residual"], gamma=t["gamma"])
                    fref_q, _ = static_quant_fp8(fref_n, t["scale"])
                    a_n, a_r, a_q = q35.all_reduce_residual_rms_norm(
                        local_contribution=t["x"], residual=t["residual"], gamma=t["gamma"],
                        fp8_scale=t["scale"])
                    f_n, f_r, f_q = q35.moe_finalize_all_reduce_rms_norm(
                        routed_output=t["routed"], expert_weights=t["weights"],
                        permuted_indices=t["permuted"], gated_shared_output=t["shared"],
                        residual=t["residual"], gamma=t["gamma"], fp8_scale=t["scale"])
                    torch.cuda.synchronize()
                    for nm, got, ref in (("q35_ar", (a_n, a_r, a_q), (ref_n, ref_r, ref_q)),
                                         ("q35_fin", (f_n, f_r, f_q), (fref_n, fref_r, fref_q))):
                        print(json.dumps(dict(event="check", backend=nm, order=1, m=m, rank=rank,
                                              norm=_diff(got[0], ref[0]), residual=_diff(got[1], ref[1]),
                                              fp8=_diff(got[2], ref[2]))), flush=True)
                    print(json.dumps(dict(event="errors", rank=rank, q35=q35.error_bits())), flush=True)
                    continue
                if b.startswith("q35:"):
                    kind = b[4:]
                    if kind == "ar":
                        fn = lambda: q35.all_reduce_residual_rms_norm(
                            local_contribution=t["x"], residual=t["residual"], gamma=t["gamma"],
                            fp8_scale=t["scale"])
                    elif kind == "ar_nq":
                        fn = lambda: q35.all_reduce_residual_rms_norm(
                            local_contribution=t["x"], residual=t["residual"], gamma=t["gamma"])
                    else:
                        fn = lambda: q35.moe_finalize_all_reduce_rms_norm(
                            routed_output=t["routed"], expert_weights=t["weights"],
                            permuted_indices=t["permuted"], gated_shared_output=t["shared"],
                            residual=t["residual"], gamma=t["gamma"],
                            fp8_scale=None if kind == "fin_nq" else t["scale"])
                    results[name] = _bench(b, fn, args, ctx, m)
                    continue
                if b.startswith("pingpong:"):
                    parts = b.split(":")
                    peer, uni = int(parts[1]), parts[2] == "uni"
                    gap = int(parts[3]) if len(parts) > 3 else 0
                    for rep in range(args.reps):
                        ns = mkh.pingpong(peer=peer, iters=1000, unicast=uni, x=t["x"], gap_ns=gap,
                                          residual=t["residual"], gamma=t["gamma"], scale=t["scale"])
                        if rank == 0:
                            print(json.dumps(dict(event="pingpong", peer=peer, unicast=uni, gap_ns=gap,
                                                  rep=rep, one_way_ns=round(ns, 1), **ctx)), flush=True)
                    continue
                if b == "checkmany":
                    for seed in range(1, args.check_seeds + 1):
                        ts = _make_inputs(args, m, dev, rank, seed=seed)
                        _check(args, ctx, m, ts, fusion, mkh, static_quant_fp8, seed=seed)
                    continue
                if b.startswith("mkh:"):
                    fn = mkh.make_call(b[4:], m=m, x=t["x"], residual=t["residual"],
                                       gamma=t["gamma"], scale=t["scale"], fin=fin,
                                       order=args.order)
                elif b.startswith("split:"):
                    fn = mkh.make_split(b[6:], m=m, x=t["x"], residual=t["residual"],
                                        gamma=t["gamma"], scale=t["scale"], fin=fin,
                                        order=args.order)
                elif b.startswith("chain:"):
                    kind = b[6:]
                    fi_call = None
                    if kind in ("plain_fi", "plain_fiq"):
                        q = kind == "plain_fiq"
                        fi_call = (lambda partial, _out, q=q: fi_on(partial, q))
                        kind = "plain_fi"
                    fn = mkh.make_chain(kind, m=m, x=t["x"], residual=t["residual"],
                                        gamma=t["gamma"], scale=t["scale"], fi_call=fi_call,
                                        order=args.order)
                else:
                    fn = table[b]
                if mkh is not None and mkh.stamps is not None:
                    mkh.stamps.zero_()
                results[name] = _bench(b, fn, args, ctx, m)
                if mkh is not None and mkh.stamps is not None and b.startswith(("mkh:", "chain:")):
                    _report_stamps(b, m, ctx, mkh.stamps)
            except Exception as e:  # report and continue with the other backends
                print(json.dumps(dict(event="error", backend=b, m=m, rank=rank,
                                      error=repr(e)[:600])), flush=True)
    if mkh is not None:
        torch.cuda.synchronize()
        print(json.dumps(dict(event="errors", rank=rank, **mkh.errors())), flush=True)
    dist.barrier()
    print(json.dumps(dict(event="done", rank=rank, **{k: round(v, 4) for k, v in results.items()})),
          flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
