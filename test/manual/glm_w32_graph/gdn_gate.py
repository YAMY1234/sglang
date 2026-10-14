"""CPU-interpreter / CUDA gate of the GDN-class GLM W32 cut (glm_w32_gdn).

Run CPU: TRITON_INTERPRET=1 PYTHONPATH=python python test/manual/glm_w32_graph/gdn_gate.py --device cpu
Checks, per synthetic case: cut error against the optimal rank-R cut (SVD) and against the unmodified service
reference (FP64 QR2 + eigh, _project_shared_omega) on the same content; sink preserved; non-due and padding rows
bitwise untouched; a missing basis column restarts; the per-step due flag. Pre-registered bounds (CPU):
cold/warm cut error <= 1.10 x optimal where the optimal error is >= 1e-4 of the content norm, otherwise
error <= 1e-5 of the content norm (both at the rounding floor); sink rel <= 1e-5; basis orthonormality <= 1e-4.
"""

import argparse
import json

import torch

from sglang.srt.layers.attention.linear.kernels import glm_w32_gdn as gdn
from sglang.srt.models.glm_duet import _state_factor as reference

D, R = 128, 32


def content(state, vb):
    """C (V, K) = S - vb a^T, a = S^T vb / |vb|^2 (the GLM / GDN right sink)."""
    a = torch.einsum("...vk,...v->...k", state.double(), vb.double()) / (vb.double() ** 2).sum(-1, keepdim=True).clamp_min(1e-12)
    return state.double() - vb.double()[..., :, None] * a[..., None, :], a


def best_error(c, rank=R):
    s = torch.linalg.svdvals(c.double())
    return float(s[..., rank:].square().sum().sqrt()), float(s.square().sum().sqrt())


def make(case, gen):
    if case == "random":
        s = torch.randn(D, D, generator=gen)
    elif case == "decay":
        s = (torch.randn(D, D, generator=gen) * torch.exp(-torch.arange(D) / 10.0)) @ torch.randn(D, D, generator=gen) / D
    elif case == "low_rank":
        s = torch.randn(D, 12, generator=gen) @ torch.randn(12, D, generator=gen)
    elif case == "near_rank_deficient":
        s = torch.randn(D, 12, generator=gen) @ torch.randn(12, D, generator=gen) + 1e-5 * torch.randn(D, D, generator=gen)
    else:
        s = torch.zeros(D, D)
    return s


def evolve(state, gen, steps=32):
    """KDA-like: per-key-channel decay on K and a rank-1 write per step (rank grows by <= 1 per step)."""
    for _ in range(steps):
        alpha = torch.exp(-0.05 * torch.rand(D, generator=gen))
        k = torch.nn.functional.normalize(torch.randn(D, generator=gen), dim=0)
        v = torch.randn(D, generator=gen)
        state = state * alpha[None, :] + 0.5 * v[:, None] * k[None, :]
    return state


def run(device):
    torch.manual_seed(210)
    gen = torch.Generator().manual_seed(210)
    L, SLOTS, H = 2, 4, 2
    records, worst = [], {}
    omega = torch.randn(1, H, D, R + 8, generator=torch.Generator().manual_seed(reference.OMEGA_SEED))[0].to(device)
    for case in ("random", "decay", "low_rank", "near_rank_deficient", "zero"):
        states = torch.stack([torch.stack([torch.stack([make(case, gen) for _ in range(H)]) for _ in range(SLOTS)])
                              for _ in range(L)]).to(device).contiguous()
        vbar = torch.randn(L, H, D, generator=gen).to(device)
        warm = torch.zeros(L, SLOTS, H, D, R, device=device)
        before = states.clone()
        # cold on slots 1, 2 (slot 0 and 3 untouched)
        gdn.cold_cut(states, vbar, omega, warm, torch.tensor([1, 2], device=device))
        assert torch.equal(states[:, 0], before[:, 0]) and torch.equal(states[:, 3], before[:, 3])
        for l in range(L):
            for s in (1, 2):
                for h in range(H):
                    c0, a0 = content(before[l, s, h], vbar[l, h])
                    c1, a1 = content(states[l, s, h], vbar[l, h])
                    err = float((c0 - c1).norm())
                    opt, tot = best_error(c0)
                    ref, _ = reference._project_shared_omega(before[l, s, h][None, None].transpose(-1, -2).float(),
                                                             vbar[l, h][None].float(), R, None, True, "right")
                    ref_err = float((content(ref[0, 0].transpose(-1, -2), vbar[l, h])[0] - c0).norm())
                    z = warm[l, s, h].double()
                    gram = z.T @ z
                    nz = gram.diagonal() > 0.5
                    orth = float((gram[nz][:, nz] - torch.eye(int(nz.sum()), dtype=torch.float64)).abs().max()) if nz.any() else 0.0
                    sink = float((a1 - a0).norm() / a0.norm().clamp_min(1e-30))
                    records.append(dict(case=case, kind="cold", err=err, opt=opt, total=tot, ref_err=ref_err,
                                        ratio=(err / opt) if opt >= 1e-4 * tot and tot > 0 else None, sink=sink, orth=orth))
        # 32 KDA-like steps on every slot, then a warm cut where only slots 1 and 2 are due
        for l in range(L):
            for s in range(SLOTS):
                for h in range(H):
                    if case != "zero":
                        states[l, s, h] = evolve(states[l, s, h].cpu(), gen).to(device)
        if case == "decay":
            warm[:, 2, :, :, 5] = 0  # a missing basis column restarts from the diag rule
        before = states.clone()
        warm_before = warm.clone()
        phase = torch.tensor([31, 31, 31, 7], dtype=torch.int32, device=device)
        pending = torch.tensor([False, False, False, False], device=device)
        active = torch.tensor([False, True, True, True], device=device)
        indices = torch.tensor([0, 1, 2, 3, -1], dtype=torch.int32, device=device)
        flag = torch.zeros((), dtype=torch.bool, device=device)
        gdn.due_flag(indices, phase, pending, active, flag, 32)
        assert bool(flag)
        gdn.due_flag(indices[3:], phase, pending, active, flag, 32)
        assert not bool(flag)
        for first, n in ((0, 1), (1, 1)):
            gdn.warm_cut(states, vbar, warm, indices, phase, pending, active, first=first, layers=n, window=32)
        for s in (0, 3):  # inactive / not due: bitwise untouched
            assert torch.equal(states[:, s], before[:, s]) and torch.equal(warm[:, s], warm_before[:, s])
        for l in range(L):
            for s in (1, 2):
                for h in range(H):
                    c0, a0 = content(before[l, s, h], vbar[l, h])
                    c1, a1 = content(states[l, s, h], vbar[l, h])
                    err = float((c0 - c1).norm())
                    opt, tot = best_error(c0)
                    prev = warm_before[l, s, h]
                    # unmodified service reference warm cut: its warm basis is V-side, so feed it its own cold basis
                    _, basis = reference._project_shared_omega(before[l, s, h][None, None].transpose(-1, -2).float(),
                                                               vbar[l, h][None].float(), R, None, True, "right")
                    ref, _ = reference._project_shared_omega(before[l, s, h][None, None].transpose(-1, -2).float(),
                                                             vbar[l, h][None].float(), R, basis, True, "right")
                    ref_err = float((content(ref[0, 0].transpose(-1, -2), vbar[l, h])[0] - c0).norm())
                    z = warm[l, s, h].double()
                    gram = z.T @ z
                    nz = gram.diagonal() > 0.5
                    orth = float((gram[nz][:, nz] - torch.eye(int(nz.sum()), dtype=torch.float64)).abs().max()) if nz.any() else 0.0
                    sink = float((a1 - a0).norm() / a0.norm().clamp_min(1e-30))
                    records.append(dict(case=case, kind="warm", err=err, opt=opt, total=tot, ref_err=ref_err,
                                        ratio=(err / opt) if opt >= 1e-4 * tot and tot > 0 else None, sink=sink, orth=orth,
                                        restarted=bool(case == "decay" and s == 2 and float(prev[:, 5].norm()) == 0)))
    for kind in ("cold", "warm"):
        rows = [r for r in records if r["kind"] == kind]
        ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
        worst[kind] = dict(max_ratio_vs_svd=max(ratios), max_ref_ratio_vs_svd=max(r["ref_err"] / r["opt"] for r in rows if r["ratio"] is not None),
                           max_sink=max(r["sink"] for r in rows), max_orth=max(r["orth"] for r in rows),
                           max_abs_err_when_optimal_zero=max([r["err"] / max(r["total"], 1e-30) for r in rows if r["ratio"] is None] or [0.0]),
                           max_ref_abs_err_when_optimal_zero=max([r["ref_err"] / max(r["total"], 1e-30) for r in rows if r["ratio"] is None] or [0.0]))
    passed = all(w["max_ratio_vs_svd"] <= 1.10 and w["max_sink"] <= 1e-5 and w["max_orth"] <= 1e-4
                 and w["max_abs_err_when_optimal_zero"] <= 1e-5 for w in worst.values())
    return dict(device=device, passed=passed, worst=worst, records=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out")
    args = parser.parse_args()
    result = run(args.device)
    text = json.dumps(result, indent=1)
    if args.out:
        open(args.out, "w").write(text)
    print(json.dumps(dict(passed=result["passed"], worst=result["worst"]), indent=1))
    assert result["passed"]
