"""P dump module gates; serving KL/GSM is an independent mandatory gate.

P5 compares today's unfused/non-mixed CholQR against fused+mixed, on identical
P-role states. P3 additionally reports same-S_N fp64 k31 W-scale ulps. It
never compares the r and r+1 routes by residual against fp64.
"""
import argparse
import glob
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'test/srt'))


def fp64_k31(s, vbar, omega, r, power):
    import torch
    s, vb, omega = s.double(), vbar.double(), omega.double()
    a = torch.einsum('bhvk,hv->bhk', s, vb) / vb.square().sum(-1).clamp_min(1e-12)[None, :, None]
    x = (s - vb[None, :, :, None] * a[:, :, None, :]).transpose(-1, -2)

    def orth(y):
        for _ in range(2):
            gram = y.transpose(-1, -2) @ y
            jitter = 1e-7 * gram.diagonal(dim1=-2, dim2=-1).mean(-1) + 1e-30
            gram = gram + jitter[..., None, None] * torch.eye(gram.shape[-1], device=s.device)
            chol = torch.linalg.cholesky_ex(gram)[0]
            y = torch.linalg.solve_triangular(chol, y.transpose(-1, -2), upper=False).transpose(-1, -2)
        return y

    y = x @ omega
    for _ in range(power):
        y = x @ orth(x.transpose(-1, -2) @ orth(y))
    q = orth(y)
    bm = q.transpose(-1, -2) @ x
    gram = bm @ bm.transpose(-1, -2)
    jitter = 1e-7 * gram.diagonal(dim1=-2, dim2=-1).mean(-1) + 1e-30
    vectors = torch.linalg.eigh(gram + jitter[..., None, None] * torch.eye(gram.shape[-1], device=s.device))[1]
    u = q @ vectors[..., -r:]
    return a, u.transpose(-1, -2), u.transpose(-1, -2) @ x


def fulln_audit(checker, paths):
    import torch
    records = []
    for path in paths:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if not payload['meta'].get('shallow_fulln'):
            raise ValueError(f'P3 requires selected shallow full-N dumps: {path}')
        with __import__('tempfile').TemporaryDirectory() as temporary:
            cfg = checker.cfg_for(temporary, payload['vbar'])
            vb = payload['vbar'].to(checker.DEV)
            omega = payload['omega'].to(checker.DEV)
            # The shallow states alone are S_N; deep emitter states are S_(N-1).
            for row, s in enumerate(payload['S']):
                states = [s[None, layer].to(checker.DEV) for layer in range(36)]
                actual = checker.run(states, vb, cfg, omega, True)
                for layer in range(24):
                    _, u64, w64 = fp64_k31(states[layer], vb[layer], omega,
                                           cfg.r, checker.ref.K31_POWER)
                    _, u, w = actual[layer]
                    # Eigensolver signs are arbitrary. Align by key-side basis
                    # before reporting W error in ulps at each head's RMS scale.
                    sign = (u[:, :, :cfg.r].double() * u64).sum(-1).sign()
                    aligned = w[:, :, :cfg.r].double() * sign[..., None]
                    scale = w64.square().mean((-1, -2)).sqrt().to(w.dtype)
                    ulp = (torch.nextafter(scale, torch.full_like(scale, float('inf'))) - scale).double()
                    error = (aligned - w64).abs() / ulp[..., None, None].clamp_min(1e-300)
                    records.append(dict(dump=str(path), row=row, layer=payload['meta']['layer_ids'][layer],
                                        w_scale_ulp_max=float(error.max()),
                                        w_scale_ulp_p999=float(torch.quantile(error.flatten(), .999)),
                                        finite=bool(torch.isfinite(error).all())))
    return dict(scope='same shallow S_N, full-fp64 k31; no r-vs-r+1 comparison',
                # docs/229 specifies reporting this audit, without a numeric
                # W-ulp cutoff. Keep that evidence separate from the R1 gate.
                records=records, numeric_cutoff_in_docs229=None)


def publication_counts(checker, paths):
    """Exercise real batch-buffer factor stores; count is not a fallback counter."""
    import torch
    from sglang.srt.mem_cache.gdn_prefill_batch_graph import BatchBuffers
    records = []
    for path in paths:
        data = torch.load(path, map_location='cpu', weights_only=False)
        vb, omega = data['vbar'].to(checker.DEV), data['omega'].to(checker.DEV)
        with __import__('tempfile').TemporaryDirectory() as temporary:
            cfg = checker.cfg_for(temporary, data['vbar'])
            layers, heads, width = vb.shape
            shape = (layers, 2, heads)
            pool = SimpleNamespace(
                cfg=cfg, layer_ids=list(range(layers)), hv=heads, v=width, k=width,
                vbar=vb, init_omega=lambda batch: omega[:1].expand(batch, -1, -1, -1),
                a=torch.zeros(*shape, width, device=checker.DEV),
                U=torch.zeros(*shape, cfg.rmax, width, device=checker.DEV, dtype=cfg.dtype),
                W=torch.zeros(*shape, cfg.rmax, width, device=checker.DEV, dtype=cfg.dtype),
                count=torch.zeros(*shape, device=checker.DEV, dtype=torch.int32),
                stale=torch.ones(2, device=checker.DEV, dtype=torch.int32),
                dense_of=torch.full((2,), -1, device=checker.DEV, dtype=torch.int32),
                dense_required=torch.zeros(2, device=checker.DEV, dtype=torch.int32),
                prefix_valid=torch.zeros(2, device=checker.DEV, dtype=torch.int32),
                dense_ring=torch.zeros(layers, 1, heads, width, width, device=checker.DEV))
            plan = SimpleNamespace(slots=torch.tensor([1], device=checker.DEV),
                                   ring_dst=torch.tensor([-1], device=checker.DEV),
                                   dense_required_after_commit=torch.zeros(1, device=checker.DEV, dtype=torch.int32))
            buffers = BatchBuffers(pool, 1, None, include_tail=False)
            for row, s in enumerate(data['S']):
                states = [(s[None, layer].to(checker.DEV), None) for layer in range(layers)]
                buffers.bind(plan, states, None, None, None)
                checker.arm(False)
                buffers.evaluate(checker.gp.factorize_layers)
                a, count = pool.a.clone(), pool.count.clone()
                checker.arm(True)
                buffers.evaluate(checker.gp.factorize_layers)
                torch.cuda.synchronize()
                records.append(dict(dump=str(path), row=row,
                                    a_bitwise=torch.equal(a, pool.a),
                                    count_bitwise=torch.equal(count, pool.count),
                                    all_layers_r=bool((pool.count[:, 1] == cfg.r).all())))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('p5', 'p3'), required=True)
    parser.add_argument('--dumps', action='append', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    import torch
    paths = sorted({path for pattern in args.dumps for path in glob.glob(pattern)})
    if not paths:
        parser.error('no P dump files')
    for path in paths:
        data = torch.load(path, map_location='cpu', weights_only=False)
        if data['meta'].get('role') != 'prefill':
            parser.error(f'not a P-role dump: {path}')
        if data['meta'].get('r') != 16 or len(data['meta'].get('layer_ids', [])) != 36:
            parser.error(f'expected the pre-registered R16/36-layer P recipe: {path}')
    import k31_fused_check as checker
    if checker.DEV.type != 'cuda':
        parser.error('module admission requires the target CUDA runtime')
    original_arm = checker.arm

    def arm(fused, *, fp64=False):
        original_arm(fused, fp64=fp64)
        if not fused and not fp64:
            os.environ['SGLANG_GDN_K31_CHOLQR_MIXED'] = '0'

    checker.arm = arm
    result = checker.gate(SimpleNamespace(dumps=paths, max_events=0, fixture=False))
    result['publication_counts'] = publication_counts(checker, paths)
    result['summary']['gate']['a_and_count_bitwise'] = result['summary']['a_bitwise'] and all(
        row['a_bitwise'] and row['count_bitwise'] and row['all_layers_r']
        for row in result['publication_counts'])
    result['summary']['gate']['fallback_delta'] = abs(
        result['summary']['fallbacks']['delta_fraction']) <= 1e-3
    result['summary']['pass'] = all(result['summary']['gate'].values())
    result['stage'] = args.stage
    result['baseline'] = 'same P dump; FUSED=0 CHOLQR_MIXED=0'
    result['candidate'] = 'same P dump; FUSED=1 CHOLQR_MIXED=1'
    result['count_scope'] = 'real BatchBuffers.evaluate factor stores; mixed fallback counter reported separately'
    if args.stage == 'p3':
        result['fulln_fp64'] = fulln_audit(checker, paths)
        result['summary']['gate']['fulln_fp64_finite'] = all(
            row['finite'] for row in result['fulln_fp64']['records'])
        result['summary']['pass'] = all(result['summary']['gate'].values())
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary']))
    raise SystemExit(0 if result['summary']['pass'] else 1)


if __name__ == '__main__':
    main()
