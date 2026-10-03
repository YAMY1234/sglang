"""R3-a admission: default scope, explicit control and numerical eigenspaces.

TRITON_INTERPRET=1 PYTHONPATH=python python -B test/registered/unit/mem_cache/nvfp4_k31_mixed_admission.py
CPU tests complement service job 949635; changed arithmetic is not byte-exact.
"""
import argparse
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch


def scope():
    from sglang.srt.mem_cache import gdn_factored_pool as pool

    rows = []
    base = dict(layers=36, b=1, h=24, v=128, k=128, r=16,
                oversample=8, dtype=torch.float32, method="k31", decode="iter")
    cases = [(None, {}, True), ("0", {}, False), ("1", {}, True)]
    # Production factored pools store fp16 factors; the gate must not drop them.
    cases += [(None, {"dtype": torch.float16}, True), ("0", {"dtype": torch.float16}, False)]
    cases += [(None, {key: value}, False) for key, value in (
        ("layers", 35), ("b", 2), ("h", 12), ("r", 8),
        ("v", 64), ("k", 64), ("oversample", 4),
        ("dtype", torch.bfloat16), ("method", "iter"), ("decode", "warm"))]
    for enabled, changes, expected in cases:
        g = dict(base, **changes)
        cfg = SimpleNamespace(r=g['r'], rmax=32, dtype=g['dtype'], init_iters=1,
                              init_oversample=g['oversample'], init_method=g['method'], decode_method=g['decode'])
        seen = []
        def fake(s, vbar, r, rmax, dtype, **kwargs):
            seen.append(kwargs['mixed_eigh'])
            b, h, v, k = s.shape
            return torch.zeros(b, h, k), torch.zeros(b, h, rmax, k), torch.zeros(b, h, rmax, v)
        env = dict(os.environ)
        env.pop('SGLANG_GDN_K31_MIXED_EIGH', None)
        if enabled is not None:
            env['SGLANG_GDN_K31_MIXED_EIGH'] = enabled
        with patch.dict(os.environ, env, clear=True), patch.object(pool, 'factorize_dense', fake):
            pool.factorize_layers([torch.zeros(g['b'], g['h'], g['v'], g['k'])] * g['layers'],
                                  torch.zeros(g['layers'], g['h'], g['v']), cfg)
        assert seen == [expected], (enabled, changes, seen)
        rows.append(dict(enabled=enabled, changes={k: str(v) for k, v in changes.items()}, mixed=expected))
    return rows


def numerics():
    from sglang.srt.duet.state_factor import pad_below_spectrum, small_eigh
    from sglang.srt.layers.attention.linear.kernels import gdn_k31_eigh_mixed as mixed
    from sglang.srt.layers.attention.linear.kernels import gdn_prefill_reference as ref

    torch.manual_seed(1589)
    x = torch.randn(3, 24, 24, dtype=torch.float64)
    g = x @ x.transpose(-1, -2)
    g[1] = x[1, :, :8] @ x[1, :, :8].T
    g[2] = torch.eye(24, dtype=torch.float64) * 1e-25
    p = pad_below_spectrum(g)
    q, counts = mixed.fp32_vectors(p)
    d, z = mixed.correct_once(p, q)
    norm = p.norm(dim=(-2, -1)).clamp_min(1e-300)
    residual = (p @ z - z * d[..., None, :]).norm(dim=(-2, -1)) / norm
    orth = (z.transpose(-1, -2) @ z - torch.eye(32)).norm(dim=(-2, -1))
    assert torch.isfinite(d).all() and torch.isfinite(z).all()
    assert residual.max() < 1e-5 and orth.max() < 1e-5
    assert (d[:, :8].amax(-1) < torch.linalg.eigvalsh(g).amin(-1)).all()
    assert counts.min() >= 0 and counts.max() <= 12
    dd, zz = small_eigh(g[:1], override='jacobi', _solver=mixed.eigh)
    expected = torch.linalg.eigh(g[:1])[1][..., -16:]
    kept = zz[..., -16:]
    error = (kept @ kept.transpose(-1, -2) - expected @ expected.transpose(-1, -2)).norm() / 4
    assert error < 1e-4, error
    # Exercise adapter selection; default generic and explicit torch overrides
    # retain FP64/torch semantics, and the mixed adapter actually calls R3-a.
    with patch.object(ref, 'K31_EIGH', 'jacobi'), patch.object(mixed, 'eigh', wraps=mixed.eigh) as solver:
        ref._small_eigh_fp64(g[:1], mixed_eigh=True)
        assert solver.call_count == 1
        ref._small_eigh_fp64(g[:1], mixed_eigh=False)
        assert solver.call_count == 1
    return dict(residual_max=float(residual.max()), orthogonality_max=float(orth.max()),
                retained_projector_relative=float(error), sweeps=counts.tolist(), padding_checked=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(dict(passed=True, scope=scope(), numerics=None if args.scope_only else numerics()), indent=2))
