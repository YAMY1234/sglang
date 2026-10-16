"""docs/222 K6: grouped side-stream expiry (SGLANG_GDN_FACTORED_TRUNC_GROUPS) vs the served per-layer side-stream expiry.

From one cloned factored pool (36 layers, staggered counts, so every slot is due every m steps), run STEPS decode steps:
  served   per layer factored_packed_decode(post order, async side stream); join at the last layer; track copy on the
           forward stream (gdn_backend default path)
  grouped  per layer the step kernel only; at each group end GDNAttnBackend._factored_fork_group (the backend method:
           one side-stream launch over the layer slice, the track copy after the last group on that stream); a tail of
           forward-stream work; GDNAttnBackend.factored_forward_end (the one join)
eager and as one CUDA graph per step (replayed STEPS times with fresh inputs copied in), B rows incl. padded rows
(slot -1), radix track copies on some steps.  Every per-step output and the final a / U / W / count / stale must be
bitwise equal.
  python test_gdn_factored_grouped_parity.py [--tp 2] [--batches 1 4 12] [--groups 2 4 6] [--steps 64] [--json out]
"""
import argparse
import json
import sys
import types
import unittest

import torch

L, K, V, S = 36, 128, 128, 64
DEV = "cuda"


def heads(tp):
    return 16 // tp, 48 // tp


def make_pool(HV, r, gen):
    from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNPool  # noqa: F401  (import check)

    rfull = 2 * r
    rmax = max(16, 1 << (rfull - 1).bit_length())
    q = torch.linalg.qr(torch.randn(HV, K, rmax, generator=gen, device=DEV))[0]
    cnt = torch.randint(r, rfull, (S,), generator=gen, device=DEV, dtype=torch.int32)
    return dict(
        a=(torch.randn(L, S, HV, K, generator=gen, device=DEV) * 0.05).float(),
        U=q.transpose(-1, -2).expand(L, S, HV, rmax, K).to(torch.float16).contiguous(),
        W=(torch.randn(L, S, HV, rmax, V, generator=gen, device=DEV) * 0.1).to(torch.float16),
        count=cnt.view(1, S, 1).expand(L, S, HV).contiguous(),
        stale=torch.zeros(S, dtype=torch.int32, device=DEV),
        vbar=(torch.randn(L, HV, V, generator=gen, device=DEV) * 0.1).float(),
        prefix_valid=torch.ones(S, dtype=torch.int32, device=DEV),
        r=r, rfull=rfull)


def clone(p):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in p.items()}


class FakePool:
    """The FactoredGDNPool surface the backend's decode/grouped path touches (tensors stacked over local layers)."""

    def __init__(self, p):
        self.__dict__.update(p)
        self.warm_v = None
        self.cfg = types.SimpleNamespace(r=p["r"], rfull=p["rfull"], trunc_warps=None, trunc_iters=None)

    def pside_join(self, *a, **k):
        pass

    def track_copy(self, src, mask, dst, device_only=False):
        from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNPool

        FactoredGDNPool.track_copy(self, src, mask, dst, device_only=device_only)


def run_steps(p, mode, groups, B, H, HV, steps, graph, seed):
    """mode served | grouped; returns per-step outputs [steps, L, B, HV, V] and the pool."""
    from sglang.srt.layers.attention.linear import gdn_backend
    from sglang.srt.layers.attention.linear.kernels import gdn_factored as gf

    GB = gdn_backend.GDNAttnBackend
    pool = FakePool(p)
    side = torch.cuda.Stream()
    n = L
    ends = sorted({-(-(k + 1) * n // groups) - 1 for k in range(min(groups, n))}) if mode == "grouped" else []
    be = types.SimpleNamespace(
        _factored_group_end={e: (ends[i - 1] + 1 if i else 0) for i, e in enumerate(ends)},
        _factored_side_stream=side, _factored_join_pending=False, _factored_join_logged=set(),
        forward_metadata=types.SimpleNamespace(mamba_track_indices=torch.full((B,), -1, dtype=torch.int64, device=DEV)))
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    A_log = (torch.rand(L, HV, generator=g, device=DEV) * 2 - 4).float()
    dt_bias = torch.randn(L, HV, generator=g, device=DEV).float()
    mixed = torch.empty(B, 2 * H * K + HV * V, dtype=torch.bfloat16, device=DEV)
    a_in = torch.empty(B, HV, dtype=torch.bfloat16, device=DEV)
    b_in = torch.empty(B, HV, dtype=torch.bfloat16, device=DEV)
    live = max(1, B - (B // 4))  # last B // 4 rows are padding (slot -1), as in a padded decode-graph bucket
    perm = torch.randperm(S - 1, generator=g, device=DEV)
    idx = torch.full((B,), -1, dtype=torch.int32, device=DEV)
    idx[:live] = (perm[:live] + 1).to(torch.int32)
    track_dst = (perm[live:2 * live] + 1).to(torch.int64)  # free slots
    mask = torch.zeros(B, dtype=torch.bool, device=DEV)
    fb = types.SimpleNamespace(mamba_track_mask=mask)
    out = torch.empty(L, B, 1, HV, V, dtype=torch.bfloat16, device=DEV)
    tail = torch.empty(4096, 4096, dtype=torch.bfloat16, device=DEV)

    def one_step():
        for li in range(L):
            gf.factored_packed_decode(
                mixed, a_in, b_in, A_log=A_log[li], dt_bias=dt_bias[li], scale=K ** -0.5, vbar=pool.vbar[li],
                fa=pool.a[li], fu=pool.U[li], fw=pool.W[li], fcount=pool.count[li], stale=pool.stale,
                ssm_state_indices=idx, num_q_heads=H, num_v_heads=HV, head_k_dim=K, head_v_dim=V, r=pool.r,
                rfull=pool.rfull, out=out[li], truncate=(mode == "served"), kernel="split",
                async_stream=side if mode == "served" else None, post_order=True)
            if mode == "grouped" and li in be._factored_group_end:
                GB._factored_fork_group(be, pool, li, idx, fb, li == L - 1)
        if mode == "served":
            torch.cuda.current_stream().wait_stream(side)
            pool.track_copy(idx, mask, be.forward_metadata.mamba_track_indices)
        torch.mm(tail, tail)  # forward-stream work after the last GDN layer (layer 47, lm_head)
        if mode == "grouped":
            GB.factored_forward_end(be)

    def feed(s):
        gi = torch.Generator(device=DEV)
        gi.manual_seed(seed * 1000 + s)
        mixed.copy_((torch.randn(mixed.shape, generator=gi, device=DEV) * 0.5).to(torch.bfloat16))
        a_in.copy_(torch.randn(a_in.shape, generator=gi, device=DEV).to(torch.bfloat16))
        b_in.copy_(torch.randn(b_in.shape, generator=gi, device=DEV).to(torch.bfloat16))
        on = s % 8 == 5  # radix track copy on some steps, of the live rows, into free slots
        mask.zero_()
        be.forward_metadata.mamba_track_indices.fill_(-1)
        if on:
            mask[:live] = True
            be.forward_metadata.mamba_track_indices[:live] = track_dst

    outs = torch.empty(steps, L, B, 1, HV, V, dtype=torch.bfloat16, device=DEV)
    due = torch.zeros(S, dtype=torch.int32, device=DEV)
    if graph:
        feed(0)
        gr = torch.cuda.CUDAGraph()
        warm = clone({k: getattr(pool, k) for k in ("a", "U", "W", "count", "stale", "prefix_valid")})
        s0 = torch.cuda.Stream()
        s0.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s0):
            one_step()  # warm-up (compiles); state restored below
        torch.cuda.current_stream().wait_stream(s0)
        torch.cuda.synchronize()
        with torch.cuda.graph(gr):
            one_step()
        for k, v in warm.items():
            getattr(pool, k).copy_(v)
        torch.cuda.synchronize()
    for s in range(steps):
        feed(s)
        if graph:
            gr.replay()
        else:
            one_step()
        outs[s].copy_(out)
        due += (pool.count[0, :, 0] == pool.r).to(torch.int32)  # post order: a due slot is cut back to r this step
    torch.cuda.synchronize()
    return outs, pool, due, idx[:live]


def compare(B, groups, tp=2, r=16, steps=64, graph=False, seed=222):
    H, HV = heads(tp)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)
    base = make_pool(HV, r, gen)
    o1, p1, due, live = run_steps(clone(base), "served", groups, B, H, HV, steps, graph, seed)
    o2, p2, _, _ = run_steps(clone(base), "grouped", groups, B, H, HV, steps, graph, seed)
    names = ("a", "U", "W", "count", "stale", "prefix_valid")
    eq = {k: torch.equal(getattr(p1, k), getattr(p2, k)) for k in names}
    eq["out"] = torch.equal(o1, o2)
    return dict(tp=tp, r=r, B=B, groups=groups, graph=graph, steps=steps, bitwise=all(eq.values()), equal=eq,
                out_max=(o1.float() - o2.float()).abs().max().item(),
                due_events_min=int(due[live.long()].min().item()), live_rows=int(live.numel()),
                tracked_dst_stale=int(p1.stale.sum().item()))


class TestGroupedParity(unittest.TestCase):
    def test_grouped_bitwise(self):
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")
        for graph in (False, True):
            for B in (1, 4, 12):
                for groups in (2, 4, 6):
                    res = compare(B, groups, graph=graph)
                    self.assertGreaterEqual(res["due_events_min"], 3, res)
                    self.assertTrue(res["bitwise"], res)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "-v":
        ap = argparse.ArgumentParser()
        ap.add_argument("--tp", type=int, nargs="+", default=[2])
        ap.add_argument("--ranks", type=int, nargs="+", default=[16])
        ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 12])
        ap.add_argument("--groups", type=int, nargs="+", default=[2, 4, 6])
        ap.add_argument("--steps", type=int, default=64)
        ap.add_argument("--json", default=None)
        args = ap.parse_args()
        rows = []
        for tp in args.tp:
            for r in args.ranks:
                for graph in (False, True):
                    for B in args.batches:
                        for groups in args.groups:
                            row = compare(B, groups, tp=tp, r=r, steps=args.steps, graph=graph)
                            rows.append(row)
                            print(json.dumps(row), flush=True)
                            torch.cuda.empty_cache()
        if args.json:
            json.dump(rows, open(args.json, "w"), indent=1)
        print("ALL_BITWISE", all(x["bitwise"] for x in rows), "MIN_DUE", min(x["due_events_min"] for x in rows))
    else:
        unittest.main()
