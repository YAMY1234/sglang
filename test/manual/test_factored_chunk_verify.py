"""#ssmon-opus admission: block verify + one-launch commit vs an fp64 sequential reference.

Reference = the frozen step primitive's maths (docs/60 K0 step, GS twice-is-enough, sink recurrence) applied input
by input WITHOUT a cut inside the verify window (B-arm rule), then the accepted state s is cut r+m.. -> r iff its
count >= r+m, with the frozen truncation algorithm (G = W W^T, diagonal warm start, ITERS x (G Z, MGS2)).
Checks: verify outputs, committed a/count exactly-shaped, densified committed state S = vbar a^T + W^T U (request
slot and radix track slot), untouched non-destination slots, padded rows.  CPU runs under TRITON_INTERPRET=1 with
small K/V; GPU runs the production shapes (K = V = 128, HV = 24, H = 8).
"""
import argparse
import json
import os
from pathlib import Path
import importlib
import sys
import time
from types import ModuleType

import torch

ROOT = Path(__file__).resolve().parents[2]
_pkg = ModuleType('opus_chunk_kernels')
_pkg.__path__ = [str(ROOT / 'python/sglang/srt/layers/attention/linear/kernels')]
sys.modules['opus_chunk_kernels'] = _pkg
chunk = importlib.import_module('opus_chunk_kernels.gdn_factored_chunk')
_frozen = importlib.import_module('opus_chunk_kernels.gdn_factored')
GS_EPS, MGS_REL_TOL, TRUNC_ITERS = _frozen.GS_EPS, _frozen.MGS_REL_TOL, _frozen.TRUNC_ITERS


def mgs_ref(Y, rkeep, passes, rel_tol):
    Q = Y.clone()
    n0 = Y.norm(dim=0)
    for p in range(passes):
        for j in range(rkeep):
            y = Q[:, j].clone()
            proj = Q[:, :j].T @ y
            y = y - Q[:, :j] @ proj
            n = y.norm()
            ok = n > 1e-12
            if p == 0:
                ok = ok and n > rel_tol * n0[j]
            Q[:, j] = y / max(n, 1e-30) if ok else 0.0
    return Q


def cut_ref(U, W, n, r, iters):
    rows = W.shape[0]
    Wm = W.clone(); Wm[n:] = 0
    G = Wm @ Wm.T
    d = torch.where(torch.arange(rows) < n, G.diag(), torch.full((rows,), -1.0, dtype=G.dtype))
    order = sorted(range(rows), key=lambda i: (-d[i].item(), i))
    Z = torch.zeros(rows, rows, dtype=G.dtype)
    for rank, i in enumerate(order[:r]):
        if i < n:
            Z[i, rank] = 1.0
    for _ in range(iters):
        Z = G @ Z
        Z = mgs_ref(Z, r, 2, MGS_REL_TOL)
    Un = Z.T @ U; Wn = Z.T @ Wm
    Un[r:] = 0; Wn[r:] = 0
    return Un, Wn


def reference(state, inputs, consts, steps, track_steps, r, rfull):
    """Per (row, head): verify outputs and the committed states at steps[row] / track_steps[row]."""
    a0, U0, W0, c0 = state
    q, k, v, ga, gb = inputs
    A_log, dt_bias, vbar, scale = consts
    B, T, HV, K = k.shape[0], k.shape[1], vbar.shape[0], U0.shape[-1]
    V = W0.shape[-1]
    out = torch.zeros(B, T, HV, V, dtype=torch.float64)
    committed, tracked = {}, {}
    for n in range(B):
        for h in range(HV):
            cnt = int(c0[n, h])
            U = torch.zeros(32, K, dtype=torch.float64); W = torch.zeros(32, V, dtype=torch.float64)
            U[:cnt] = U0[n, h, :cnt]; W[:cnt] = W0[n, h, :cnt]
            a = a0[n, h].clone()
            vb = vbar[h]
            states = []
            for t in range(T):
                x = ga[n, t, h] + dt_bias[h]
                sp = torch.log1p(torch.exp(x)) if x <= 20 else x
                gt = torch.exp(-torch.exp(A_log[h]) * sp)
                beta = gb[n, t, h]
                qn = q[n, t, h] / torch.sqrt((q[n, t, h] ** 2).sum() + 1e-6) * scale
                kn = k[n, t, h] / torch.sqrt((k[n, t, h] ** 2).sum() + 1e-6)
                a = gt * (a - beta * kn * (kn @ a)) + beta * kn
                o = vb * (a @ qn)
                c = U[:cnt] @ kn
                kp = kn - U[:cnt].T @ c
                nrm2 = kp @ kp
                if nrm2 < 0.25:
                    c2 = U[:cnt] @ kp
                    kp = kp - U[:cnt].T @ c2
                    c = c + c2
                    nrm2 = kp @ kp
                nrm = torch.sqrt(nrm2)
                keep = nrm > GS_EPS
                khat = kp / max(nrm, GS_EPS) if keep else torch.zeros_like(kp)
                clast = nrm if keep else torch.zeros(())
                mvec = W[:cnt].T @ c
                delta = beta * ((v[n, t, h] - vb) - gt * mvec)
                cfull = torch.cat([c, clast.reshape(1)])
                cq = torch.cat([U[:cnt] @ qn, (khat @ qn).reshape(1)])
                o = o + gt * (W[:cnt].T @ cq[:cnt]) + delta * (cfull @ cq)
                out[n, t, h] = o
                W[:cnt + 1] = gt * W[:cnt + 1] + cfull[:, None] * delta[None, :]
                U[cnt] = khat.to(torch.float16).to(torch.float64)
                cnt += 1
                states.append((a.clone(), U.clone(), W.clone(), cnt))
            for store, s in ((committed, steps[n]), (tracked, track_steps[n])):
                if s < 0:
                    continue
                a_s, U_s, W_s, n_s = states[s]
                if n_s >= rfull:
                    U_s, W_s = cut_ref(U_s, W_s, n_s, r, TRUNC_ITERS)
                    n_s = r
                store[(n, h)] = (a_s, U_s[:16], W_s[:16], n_s)
    return out, committed, tracked


def dense(a, U, W, n, vb):
    return vb[:, None] * a[None, :] + W[:n].T @ U[:n]


def run(device, K, V, HV, H, B, seed=0):
    torch.manual_seed(seed)
    T, R, RF, S, L = 4, 8, 16, 11, 2
    dev = torch.device(device)
    slots = torch.tensor([5, 2, 9, 0, 7, 3][:B], dtype=torch.int64)
    counts = torch.tensor([8, 12, 13, 15, 14, 11][:B], dtype=torch.int32)
    steps = torch.tensor([3, 2, 3, 1, 0, 3][:B], dtype=torch.int32)
    track_slots = torch.tensor([10, -1, 6, 4, -1, 1][:B], dtype=torch.int64)
    track_steps = torch.tensor([1, -1, 3, 0, -1, 2][:B], dtype=torch.int32)
    pool_a = torch.randn(L, S, HV, K, dtype=torch.float64) * 0.1
    pool_U = torch.zeros(L, S, HV, 16, K, dtype=torch.float64)
    pool_W = torch.randn(L, S, HV, 16, V, dtype=torch.float64) * 0.1
    pool_c = torch.full((L, S, HV), R, dtype=torch.int32)
    for l in range(L):
        for s in range(S):
            for h in range(HV):
                Qm, _ = torch.linalg.qr(torch.randn(K, 16, dtype=torch.float64))
                pool_U[l, s, h] = Qm.T
    for i, s in enumerate(slots.tolist()):
        pool_c[:, s] = counts[i]
    for l in range(L):
        for s in range(S):
            for h in range(HV):
                pool_W[l, s, h, int(pool_c[l, s, h]):] = 0
    pool_U = pool_U.to(torch.float16).to(torch.float64)
    pool_W = pool_W.to(torch.float16).to(torch.float64)
    width = 2 * H * K + HV * V
    # keys partly inside the entry span so the second GS pass and zero-column paths are exercised
    mixed = torch.randn(L, B, T, width, dtype=torch.float64)
    for l in range(L):
        for n in range(B):
            for t in range(T):
                if (n + t) % 3 == 0:
                    for h in range(H):
                        hv = h * (HV // H)
                        base = pool_U[l, slots[n], hv, :int(counts[n])]
                        mixed[l, n, t, H * K + h * K:H * K + (h + 1) * K] = base.T @ torch.randn(int(counts[n]), dtype=torch.float64) + 0.05 * torch.randn(K, dtype=torch.float64)
    mixed = mixed.to(torch.bfloat16)
    ga = torch.randn(L, B, T, HV).to(torch.bfloat16)
    gb = torch.randn(L, B, T, HV).to(torch.bfloat16)
    A_log = torch.randn(L, HV) * 0.5
    dt_bias = torch.randn(L, HV) * 0.5
    vbar = torch.randn(L, HV, V) * 0.1
    scale = K ** -0.5

    class Pool:
        pass
    pool = Pool()
    pool.a = pool_a.to(torch.float32).to(dev)
    pool.U = pool_U.to(torch.float16).to(dev)
    pool.W = pool_W.to(torch.float16).to(dev)
    pool.count = pool_c.clone().to(dev)  # .to(cpu) would alias the reference counts
    pool.stale = torch.zeros(S, dtype=torch.int32, device=dev)
    pool.dense_of = torch.full((S,), 5, dtype=torch.int32, device=dev)
    pool.dense_required = torch.ones(S, dtype=torch.int32, device=dev)
    pool.prefix_valid = torch.ones(S, dtype=torch.int32, device=dev)
    before = {n: getattr(pool, n).clone() for n in ('a', 'U', 'W', 'count')}
    cap = B + 2  # padded verify rows
    records = chunk.allocate_records(L, cap, T, HV, K, V, dev)
    indices = torch.full((cap,), -1, dtype=torch.int64, device=dev)
    indices[:B] = slots.to(dev)
    outs = []
    for l in range(L):
        m = torch.zeros(cap, T, width, dtype=torch.bfloat16, device=dev); m[:B] = mixed[l].to(dev)
        xa = torch.zeros(cap, T, HV, dtype=torch.bfloat16, device=dev); xa[:B] = ga[l].to(dev)
        xb = torch.zeros(cap, T, HV, dtype=torch.bfloat16, device=dev); xb[:B] = gb[l].to(dev)
        o = chunk.chunk_verify(m, xa, xb, A_log=A_log[l].to(dev), dt_bias=dt_bias[l].to(dev), vbar=vbar[l].to(dev),
                               pa=pool.a[l], pu=pool.U[l], pw=pool.W[l], pcount=pool.count[l], indices=indices,
                               records=records, layer=l, scale=scale, num_q_heads=H)
        outs.append(o.float().cpu())
    for n in ('a', 'U', 'W', 'count'):
        assert torch.equal(before[n], getattr(pool, n)), 'verify must not write the pool: ' + n
    for l in range(L):
        assert torch.all(outs[l][B:] == 0), 'padded verify rows must output zeros'
    chunk.commit_select(pool, records, indices[:B], steps.to(dev), track_slots.to(dev), track_steps.to(dev), r=R, rfull=RF)
    report = dict(device=device, K=K, V=V, HV=HV, H=H, B=B, layers=[])
    worst = dict(out=0.0, state=0.0)
    for l in range(L):
        q = mixed[l, :, :, :H * K].double().reshape(B, T, H, K).repeat_interleave(HV // H, dim=2)
        k = mixed[l, :, :, H * K:2 * H * K].double().reshape(B, T, H, K).repeat_interleave(HV // H, dim=2)
        v = mixed[l, :, :, 2 * H * K:].double().reshape(B, T, HV, V)
        # beta = bf16(sigmoid(b)) exactly as the device converts: the Triton CPU interpreter truncates fp32 -> bf16,
        # the GPU rounds to nearest (checked against the kernel records, 2026-09-26).
        sig = torch.sigmoid(gb[l].float())
        gbeta = ((sig.view(torch.int32) & -65536).view(torch.float32) if device == 'cpu'
                 else sig.to(torch.bfloat16).float()).double()
        state = (pool_a[l, slots].to(torch.float32).double(), pool_U[l, slots], pool_W[l, slots], pool_c[l, slots])
        ref_out, committed, tracked = reference(state, (q, k, v, ga[l].double(), gbeta),
            (A_log[l].double(), dt_bias[l].double(), vbar[l].double(), scale), steps.tolist(), track_steps.tolist(), R, RF)
        err = ((outs[l][:B].double() - ref_out).abs().max() / ref_out.abs().max().clamp_min(1e-6)).item()
        worst['out'] = max(worst['out'], err)
        assert err < 2e-2, f'verify output error {err} (layer {l})'
        lay = dict(layer=l, out_rel=err, states=[])
        for dst_slots, store in ((slots, committed), (track_slots, tracked)):
            for n in range(B):
                dst = int(dst_slots[n])
                if dst < 0 or (n, 0) not in store:
                    continue
                for h in range(HV):
                    a_s, U_s, W_s, n_s = store[(n, h)]
                    got_n = int(pool.count[l, dst, h])
                    assert got_n == n_s, f'count {got_n} != {n_s} (layer {l} row {n} head {h} dst {dst})'
                    got_a = pool.a[l, dst, h].double().cpu()
                    assert torch.allclose(got_a, a_s, rtol=1e-4, atol=1e-5), 'sink a differs'
                    S_ref = dense(a_s, U_s, W_s, n_s, vbar[l, h].double())
                    S_got = dense(got_a, pool.U[l, dst, h].double().cpu(), pool.W[l, dst, h].double().cpu(), got_n,
                                  vbar[l, h].double())
                    e = ((S_got - S_ref).norm() / S_ref.norm()).item()
                    worst['state'] = max(worst['state'], e)
                    assert e < 2e-2, f'committed state error {e} (layer {l} row {n} head {h} n={n_s})'
                    lay['states'].append(dict(row=n, head=h, dst=dst, count=n_s, rel=e))
        report['layers'].append(dict(layer=l, out_rel=err, states=len(lay['states']),
                                     max_state_rel=max((x['rel'] for x in lay['states']), default=0.0)))
    touched = set(slots.tolist()) | {int(x) for x in track_slots.tolist() if x >= 0}
    for s in range(S):
        if s not in touched:
            for n in ('a', 'U', 'W', 'count'):
                assert torch.equal(before[n][:, s], getattr(pool, n)[:, s]), f'untouched slot {s} changed ({n})'
    for s in touched:
        assert int(pool.stale[s]) == 1 and int(pool.dense_of[s]) == -1 and int(pool.dense_required[s]) == 0 \
            and int(pool.prefix_valid[s]) == 0, 'metadata not published'
    report['worst'] = worst
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    t0 = time.time()
    if a.device == 'cpu':
        cases = [dict(K=32, V=32, HV=4, H=2, B=6, seed=1)]
    else:
        cases = [dict(K=128, V=128, HV=24, H=8, B=6, seed=s) for s in (1, 2, 3)]
    reports = [run(a.device, **c) for c in cases]
    result = dict(complete=True, passed=True, seconds=time.time() - t0, reports=reports)
    text = json.dumps(result, indent=1)
    if a.out:
        a.out.write_text(text + '\n')
    print(json.dumps(dict(passed=True, worst=[r['worst'] for r in reports], seconds=result['seconds'])))
