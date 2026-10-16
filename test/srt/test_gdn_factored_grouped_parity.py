"""docs/222 K6: grouped side-stream expiry (SGLANG_GDN_FACTORED_TRUNC_GROUPS) and the bookkeeping fold
(SGLANG_GDN_FACTORED_BOOKKEEPING_FOLD) vs the served per-layer side-stream expiry with the torch prefix_valid ops.

From one cloned factored pool (36 layers, staggered counts, so every slot is due every m steps), run STEPS decode steps:
  served   per layer factored_packed_decode(post order, async side stream); join at the last layer; track copy on the
           forward stream (gdn_backend default path)
  grouped  per layer the step kernel only; at each group end GDNAttnBackend._factored_fork_group (the backend method:
           one side-stream launch over the layer slice, the track copy after the last group on that stream); a tail of
           forward-stream work; GDNAttnBackend.factored_forward_end (the one join)
  fold     layer 0's prefix_valid[max(idx, 0)] = 0 in its step kernel (INVALIDATE) and the track copy's
           prefix_valid clear in the copy kernel, instead of invalidate_prefix_dense + the pool's gather / where / put
  cond<k>  docs/222 §12 R8: the backend's per-layer expiry inside CUDA-graph IF conditional nodes
           (GDNAttnBackend._factored_cond_active / _factored_cond_expiry; eager = today's launch), the IF node of layer l
           on side stream l mod k (cond = cond2), with the step-entry count >= RMAX trap on; `--neg` runs the negative
           control (no setter: the trap must fire) in a subprocess.  Why k streams stay bitwise: the expiry of layer l
           reads and writes only layer l's U / W / count (the step of layer l+1 touches only layer l+1, as today), so the
           expiries of different layers commute; each still follows its own layer's step (fork after the step) and the
           setter, and every side stream joins the forward stream before the track copy and the next forward.
eager and as one CUDA graph per step (replayed STEPS times with fresh inputs copied in), B rows incl. padded rows
(slot -1), radix track copies on some steps (incl. one masked padded row: dst -1 -> slot 0 quirk), prefix_valid
re-set on a step as an extend would (incl. slot 0, outside the batch: a padded row clears it), and a host offload
round trip of the live slots' factored state on some steps.  Every per-step output
and the final a / U / W / count / stale / prefix_valid must be bitwise equal.
  python test_gdn_factored_grouped_parity.py [--cand grouped fold grouped+fold] [--tp 2] [--batches 1 4 12]
                                             [--groups 2 4 6] [--steps 64] [--json out]
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


def run_steps(p, mode, groups, B, H, HV, steps, graph, seed, fold=False, prio="normal", no_setter=False,
              on_step=None):
    """mode served | grouped | cond; fold = K6(c); returns per-step outputs [steps, L, B, HV, V] and the pool."""
    import os

    from sglang.srt.layers.attention.linear import gdn_backend
    from sglang.srt.layers.attention.linear.kernels import gdn_factored as gf
    from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNPool

    os.environ["SGLANG_GDN_FACTORED_BOOKKEEPING_FOLD"] = "1" if fold else "0"
    os.environ["SGLANG_GDN_FACTORED_TRUNC_COND"] = "1" if mode == "cond" else "0"
    trap = mode == "cond"
    if mode == "cond":
        from sglang.srt.layers.attention.linear.kernels import gdn_factored_expiry_cuda

        gdn_factored_expiry_cuda.cond_expiry()  # extension + IF-body stream, before any capture

    GB = gdn_backend.GDNAttnBackend
    pool = FakePool(p)
    side = torch.cuda.Stream()
    n = L
    ends = sorted({-(-(k + 1) * n // groups) - 1 for k in range(min(groups, n))}) if mode == "grouped" else []
    plan = {e: (ends[i - 1] + 1 if i else 0) for i, e in enumerate(ends)}
    if prio == "high":
        side = torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[1])
    be = types.SimpleNamespace(
        _factored_side_stream=side, _factored_cur_side=side, _factored_join_pending=None, _factored_join_logged=set(),
        forward_metadata=types.SimpleNamespace(mamba_track_indices=torch.full((B,), -1, dtype=torch.int64, device=DEV)),
        factored=types.SimpleNamespace(cfg=types.SimpleNamespace(decode_method="split"), layer_ids=list(range(L))),
        _factored_cond_no_setter=no_setter)
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

    async_mode = mode in ("served", "cond")

    def one_step():
        for li in range(L):
            if li == 0 and not fold:
                FactoredGDNPool.invalidate_prefix_dense(pool, idx)
            cond = mode == "cond" and GB._factored_cond_active(be, idx, False)
            if trap:
                GB._factored_count_trap(be, pool.count[li], idx, pool.U.shape[-2])
            gf.factored_packed_decode(
                mixed, a_in, b_in, A_log=A_log[li], dt_bias=dt_bias[li], scale=K ** -0.5, vbar=pool.vbar[li],
                fa=pool.a[li], fu=pool.U[li], fw=pool.W[li], fcount=pool.count[li], stale=pool.stale,
                ssm_state_indices=idx, num_q_heads=H, num_v_heads=HV, head_k_dim=K, head_v_dim=V, r=pool.r,
                rfull=pool.rfull, out=out[li], truncate=async_mode and not cond, kernel="split",
                async_stream=side if async_mode and not cond else None, post_order=True,
                **(dict(prefix_valid=pool.prefix_valid, invalidate=True) if fold and li == 0 else {}))
            if cond:
                GB._factored_cond_expiry(be, pool, li, idx, pool.U[li], pool.W[li], pool.count[li])
            if mode == "grouped" and li in plan:
                GB._factored_fork_group(be, pool, li, idx, fb, li == L - 1, plan)
        if async_mode:
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
            if live < B:
                mask[B - 1] = True  # a masked padded row (dst -1): clears prefix_valid[0] in both paths
        if s % 16 == 3:
            # extends publish prefix_valid = 1 (save_prefix_dense), incl. slot 0 outside this batch
            gp = torch.Generator(device=DEV)
            gp.manual_seed(seed * 7 + s)
            pool.prefix_valid.copy_(torch.randint(0, 2, (S,), generator=gp, device=DEV, dtype=torch.int32))
            pool.prefix_valid[0] = 1
        if s % 16 == 11:
            # host offload round trip of the live slots' factored state (as get_cpu_slots / load_cpu_slots)
            sl = idx[:live].long()
            for k in ("a", "U", "W", "count"):
                t = getattr(pool, k)
                host = t[:, sl].to("cpu")
                t[:, sl] = host.to(DEV)

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
        if on_step is not None:
            on_step(s)
        due += (pool.count[0, :, 0] == pool.r).to(torch.int32)  # post order: a due slot is cut back to r this step
    torch.cuda.synchronize()
    return outs, pool, due, idx[:live]


def compare(B, groups, tp=2, r=16, steps=64, graph=False, seed=222, cand="grouped"):
    H, HV = heads(tp)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)
    base = make_pool(HV, r, gen)
    o1, p1, due, live = run_steps(clone(base), "served", groups, B, H, HV, steps, graph, seed)
    mode = "grouped" if "grouped" in cand else "cond" if "cond" in cand else "served"
    if mode == "cond":
        import os

        os.environ["SGLANG_GDN_FACTORED_TRUNC_COND_STREAMS"] = cand.split("cond")[1][:1] or "2"
    o2, p2, _, _ = run_steps(clone(base), mode, groups, B, H, HV, steps,
                             graph, seed, fold="fold" in cand, prio="high" if "prio" in cand else "normal")
    names = ("a", "U", "W", "count", "stale", "prefix_valid")
    eq = {k: torch.equal(getattr(p1, k), getattr(p2, k)) for k in names}
    eq["out"] = torch.equal(o1, o2)
    return dict(cand=cand, tp=tp, r=r, B=B, groups=groups if "grouped" in cand else 0, graph=graph, steps=steps, bitwise=all(eq.values()), equal=eq,
                out_max=(o1.float() - o2.float()).abs().max().item(),
                due_events_min=int(due[live.long()].min().item()), live_rows=int(live.numel()),
                tracked_dst_stale=int(p1.stale.sum().item()), pv0=int(p1.prefix_valid[0].item()),
                pv_sum=int(p1.prefix_valid.sum().item()))


class TestGroupedParity(unittest.TestCase):
    def test_grouped_bitwise(self):
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")
        for cand, gs in (("grouped", (2, 4, 6)), ("fold", (0,)), ("grouped+fold", (4,))):
            for graph in (False, True):
                for B in (1, 4, 12):
                    for groups in gs:
                        res = compare(B, groups, graph=graph, cand=cand)
                        self.assertGreaterEqual(res["due_events_min"], 3, res)
                        self.assertTrue(res["bitwise"], res)


def negative_child(B=4, tp=2, steps=40, seed=223, k=2):
    """R8 negative control (run in its own process: the trap kills the CUDA context).  Phase 1: the cond graph with the
    setter, trap on -> must survive.  Phase 2: the same without the setter -> no expiry ever runs, so a slot enters a
    step with count == RMAX and the trap must fire before the steps end."""
    import os

    os.environ["SGLANG_GDN_FACTORED_TRUNC_COND_STREAMS"] = str(k)
    H, HV = heads(tp)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)
    base = make_pool(HV, 16, gen)
    run_steps(clone(base), "cond", 4, B, H, HV, steps, True, seed)
    torch.cuda.synchronize()
    print("SETTER_ON_OK", flush=True)

    def sync(s):
        torch.cuda.synchronize()  # surface the trap at the step that raises it
        print("NEG_STEP", s, flush=True)

    run_steps(clone(base), "cond", 4, B, H, HV, steps, True, seed, no_setter=True, on_step=sync)
    torch.cuda.synchronize()
    print("SETTER_OFF_SURVIVED", flush=True)


def negative_control(B=4, tp=2, k=2):
    import subprocess

    r = subprocess.run([sys.executable, __file__, "--neg-child", str(B), str(tp), str(k)], capture_output=True,
                       text=True, timeout=900)
    out = r.stdout + r.stderr
    steps = [int(x.split()[1]) for x in r.stdout.splitlines() if x.startswith("NEG_STEP")]
    return dict(cand="cond-negative", k=k, B=B, tp=tp, returncode=r.returncode, setter_on_ok="SETTER_ON_OK" in r.stdout,
                setter_off_survived="SETTER_OFF_SURVIVED" in r.stdout, trap_step=(max(steps) + 1) if steps else 0,
                trap_seen=("GDN_FACTORED_COUNT_TRAP" in out) or ("illegal instruction" in out.lower())
                or ("unspecified launch failure" in out.lower()),
                fired=r.returncode != 0 and "SETTER_ON_OK" in r.stdout and "SETTER_OFF_SURVIVED" not in r.stdout,
                tail=out[-600:])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--neg-child":
        negative_child(B=int(sys.argv[2]), tp=int(sys.argv[3]), k=int(sys.argv[4]) if len(sys.argv) > 4 else 2)
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] != "-v":
        ap = argparse.ArgumentParser()
        ap.add_argument("--cand", nargs="+", default=["grouped", "fold", "grouped+fold", "prio", "grouped+fold+prio"])
        ap.add_argument("--tp", type=int, nargs="+", default=[2])
        ap.add_argument("--ranks", type=int, nargs="+", default=[16])
        ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 12])
        ap.add_argument("--groups", type=int, nargs="+", default=[2, 4, 6])
        ap.add_argument("--steps", type=int, default=64)
        ap.add_argument("--json", default=None)
        ap.add_argument("--neg", action="store_true", help="R8: also run the trap negative control (subprocess)")
        args = ap.parse_args()
        rows = []
        if args.neg:
            ks = sorted({int(c.split("cond")[1][:1] or 2) for c in args.cand if "cond" in c}) or [2]
            for tp in args.tp:
              for k in ks:
                row = negative_control(B=4, tp=tp, k=k)
                print(json.dumps(row), flush=True)
                rows.append(row)
        for cand in args.cand:
            for tp in args.tp:
                for r in args.ranks:
                    for graph in (False, True):
                        for B in args.batches:
                            for groups in (args.groups if "grouped" in cand else [4]):
                                row = compare(B, groups, tp=tp, r=r, steps=args.steps, graph=graph, cand=cand)
                                rows.append(row)
                                print(json.dumps(row), flush=True)
                                torch.cuda.empty_cache()
        if args.json:
            json.dump(rows, open(args.json, "w"), indent=1)
        par = [x for x in rows if "bitwise" in x]
        neg = [x for x in rows if x.get("cand") == "cond-negative"]
        print("ALL_BITWISE", all(x["bitwise"] for x in par), "MIN_DUE", min(x["due_events_min"] for x in par),
              *(["NEG_FIRED", all(x["fired"] for x in neg)] if neg else []))
    else:
        unittest.main()
