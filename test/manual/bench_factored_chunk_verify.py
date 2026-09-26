"""Diagnostic only: 36-layer block verify + one-launch commit, production shapes, CUDA graphs.

Reference points from the served torch20 traces (docs/120 iteration 1): stock verify kernel
gdn_replayssm_spec_circular_kernel 7.7 / 8.3 us per layer at B1 / B8; frozen v1 fused window 26.8 / 40.3 us.
Entry counts cycle 8..15 over requests; accepted input index 3 for every row (the most cut-heavy commit).
"""
import argparse
import importlib
import json
from pathlib import Path
import statistics
import sys
from types import ModuleType, SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
_pkg = ModuleType('opus_chunk_kernels')
_pkg.__path__ = [str(ROOT / 'python/sglang/srt/layers/attention/linear/kernels')]
sys.modules['opus_chunk_kernels'] = _pkg
chunk = importlib.import_module('opus_chunk_kernels.gdn_factored_chunk')


def timed(graph, reps=50, rounds=7):
    values = []
    for _ in range(rounds):
        b, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        b.record()
        for _ in range(reps):
            graph.replay()
        e.record(); e.synchronize()
        values.append(b.elapsed_time(e) / reps)
    return statistics.median(values), values


def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    return g


def run(batch, step, warps=None, commit_warps=None, bv=None, inspan=False, cold=False, dbv=None, dwarps=None):
    if dbv:
        chunk.DENSE_BV = dbv
    if dwarps:
        chunk.DENSE_WARPS = dwarps
    chunk.CHUNK_BV = bv or 0
    if warps:
        chunk.CHUNK_WARPS = warps
    if commit_warps:
        chunk.COMMIT_WARPS = commit_warps
    torch.manual_seed(120)
    L, HV, H, K, V, T, S = 36, 24, 8, 128, 128, 4, 128
    dev = 'cuda'
    width = 2 * H * K + HV * V
    pool = SimpleNamespace()
    pool.a = torch.randn(L, S, HV, K, device=dev) * .05
    q, _ = torch.linalg.qr(torch.randn(L * S * HV, K, 16, device=dev))
    pool.U = q.transpose(-1, -2).reshape(L, S, HV, 16, K).to(torch.float16).contiguous()
    pool.W = (torch.randn(L, S, HV, 16, V, device=dev) * .05).to(torch.float16)
    counts = (8 + torch.arange(S, device=dev) % 8).to(torch.int32)
    pool.count = counts[None, :, None].expand(L, S, HV).contiguous()
    pool.stale = torch.zeros(S, dtype=torch.int32, device=dev)
    pool.dense_of = torch.full((S,), -1, dtype=torch.int32, device=dev)
    pool.dense_required = torch.zeros(S, dtype=torch.int32, device=dev)
    pool.prefix_valid = torch.zeros(S, dtype=torch.int32, device=dev)
    pool.vbar = torch.randn(L, HV, V, device=dev) * .05
    entry = {n: getattr(pool, n).clone() for n in ('a', 'U', 'W', 'count')}
    mixed = torch.randn(L, batch, T, width, device=dev, dtype=torch.bfloat16)
    slots_ = torch.arange(batch, dtype=torch.int64, device=dev) * 3 % S
    if inspan:
        # served keys lie mostly in the recurrent key basis: k = U^T r + 0.05 noise (second GS pass taken)
        hv_of_k = torch.arange(H, device=dev) * (HV // H)
        U_sel = pool.U[:, slots_][:, :, hv_of_k].float()  # (L, B, H, 16, K)
        r = torch.randn(L, batch, T, H, 16, device=dev)
        keys = torch.einsum('lbthr,lbhrk->lbthk', r, U_sel) + 0.05 * torch.randn(L, batch, T, H, K, device=dev)
        mixed[:, :, :, H * K:2 * H * K] = keys.reshape(L, batch, T, H * K).to(torch.bfloat16)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev) if cold else None
    ga = torch.randn(L, batch, T, HV, device=dev, dtype=torch.bfloat16)
    gb = torch.randn_like(ga)
    A_log = torch.randn(L, HV, device=dev) * .5
    dt_bias = torch.randn(L, HV, device=dev) * .5
    vbar = pool.vbar
    records = chunk.allocate_records(L, 96, T, HV, K, V, dev)
    slots = slots_
    indices = slots.clone()
    steps = torch.full((batch,), step, dtype=torch.int32, device=dev)
    compiled = {}

    def verify():
        for l in range(L):
            if flush is not None:
                flush.zero_()  # evict L2 between layers (served layers are cold)
            chunk.verify(mixed[l], ga[l], gb[l], A_log=A_log[l], dt_bias=dt_bias[l], vbar=vbar[l],
                               pa=pool.a[l], pu=pool.U[l], pw=pool.W[l], pcount=pool.count[l], indices=indices,
                               records=records, layer=l, scale=K ** -.5, num_q_heads=H)

    def reset():
        for n, t in entry.items():
            getattr(pool, n).copy_(t)

    def commit():
        chunk.commit_select(pool, records, indices, steps, r=8, rfull=16)

    gv = capture(verify)
    gf = capture(lambda: [flush.zero_() for _ in range(L)]) if flush is not None else None
    gr = capture(reset)
    gc_ = capture(lambda: (reset(), commit()))
    tv, sv = timed(gv)
    if gf is not None:
        tv -= timed(gf)[0]
    tr, _ = timed(gr)
    tc, sc = timed(gc_)
    k1 = chunk._factored_chunk_verify_kernel
    k2 = chunk._factored_commit_select_kernel
    res = {}
    for name, kern in (('verify', k1), ('commit', k2)):
        cache = getattr(kern, 'device_caches', None) or getattr(kern, 'cache', None)
        try:
            entries = list(cache[0][0].values()) if isinstance(cache, dict) and isinstance(cache.get(0), tuple) else []
        except Exception:
            entries = []
        res[name] = [dict(regs=getattr(c, 'n_regs', None), spills=getattr(c, 'n_spills', None)) for c in entries]
    cut_rows = int(((counts[slots] + step + 1) >= 16).sum())
    return dict(batch=batch, step=step, layers=L, verify_ms=tv, verify_us_per_layer=tv * 1000 / L,
                commit_ms=tc - tr, reset_ms=tr, cut_rows=cut_rows, verify_samples=sv, commit_samples=sc,
                resources=res, warps=dict(verify=chunk.CHUNK_WARPS, commit=chunk.COMMIT_WARPS), bv=chunk.CHUNK_BV,
                inspan=inspan, cold=cold, dense=dict(bv=chunk.DENSE_BV, warps=chunk.DENSE_WARPS, mode=chunk.MODE))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batches', type=int, nargs='+', default=[1, 8, 16, 32])
    p.add_argument('--warps', type=int, nargs='+', default=[chunk.CHUNK_WARPS])
    p.add_argument('--commit-warps', type=int, nargs='+', default=[chunk.COMMIT_WARPS])
    p.add_argument('--bv', type=int, nargs='+', default=[0])
    p.add_argument('--inspan', type=int, nargs='+', default=[0])
    p.add_argument('--cold', type=int, nargs='+', default=[0])
    p.add_argument('--dense-bv', type=int, nargs='+', default=[0])
    p.add_argument('--dense-warps', type=int, nargs='+', default=[0])
    a = p.parse_args()
    rows = []
    for b in a.batches:
        combos = [(w, a.commit_warps[0], bv) for bv in a.bv for w in a.warps] + \
                 [(a.warps[0], cw, a.bv[0]) for cw in a.commit_warps[1:]]
        for w, cw, bv in combos:
          for ins in a.inspan:
           for cold in a.cold:
            for dbv in a.dense_bv:
             for dw in a.dense_warps:
              for step in (0, 3):
                rows.append(run(b, step, w, cw, bv, bool(ins), bool(cold), dbv, dw))
            print(json.dumps({k: v for k, v in rows[-1].items() if not k.endswith('samples')}), flush=True)
            a.out.write_text(json.dumps(dict(complete=False, rows=rows), indent=1) + '\n')
    a.out.write_text(json.dumps(dict(complete=True, rows=rows, diagnostic_only=True), indent=1) + '\n')
