"""Replay saved same-S checkpoints through singleton/joint k31 geometries.

Diagnostic only: no model, no changes to the admitted thresholds or service
recipe. Solver and branch changes are reported explicitly for attribution.
"""
import argparse
import glob
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    import torch
    from checkpoint_gate import factor_diagnostic
    from sglang.srt.mem_cache.gdn_factored_pool import factorize_layers

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    for path in sorted(glob.glob(args.evidence)):
        data = torch.load(path, map_location='cpu', weights_only=False)
        if data['row']['route'] != '0' or data['row']['route_index'] != 2:
            continue
        repeated = torch.load(str(Path(path).with_name(Path(path).name.replace('-2.pt', '-3.pt'))),
                              map_location='cpu', weights_only=False)
        audit = dict(fulln_self_raw_equal=torch.equal(data['raw'], repeated['raw']),
            fulln_self_factor_equal={name: torch.equal(value, repeated['factors'][name])
                                     for name, value in data['factors'].items()})
        if data['row']['length'] == 129:
            matched = torch.load(path.replace('-129-1-2.pt', '-128-0-2.pt'),
                                 map_location='cpu', weights_only=False)
            audit['matched_prefix_checkpoint_equal'] = torch.equal(data['raw'], matched['raw'])
            audit['matched_prefix_unequal_layers'] = [int(lid) for lid, a, b in
                zip(data['layer_ids'], data['raw'], matched['raw']) if not torch.equal(a, b)]
        raw, normal = data['raw'].cuda(), data['publication']['normal_raw'].cuda()
        layers, heads = raw.shape[:2]
        pool = SimpleNamespace(vbar=data['vbar'].cuda(), k31_omega=data['omega'].cuda(),
                               cfg=data['cfg'], layer_ids=data['layer_ids'])
        omega = pool.k31_omega
        if omega.ndim == 3:
            omega = omega.unsqueeze(0)
        for mode, batch, mixed, any_batch in (
            ('checkpoint_b1_mixed', 1, 1, 0), ('joint_b2_fp64', 2, 1, 0),
            ('joint_b2_mixed', 2, 1, 1), ('checkpoint_b1_fp64', 1, 0, 0)):
            os.environ.update(SGLANG_GDN_K31_FUSED='0', SGLANG_GDN_K31_CHOLQR_MIXED='0',
                SGLANG_GDN_K31_MIXED_EIGH=str(mixed),
                SGLANG_GDN_K31_MIXED_EIGH_ANY_BATCH=str(any_batch))
            states = [raw[i:i+1] if batch == 1 else torch.cat((normal[i], raw[i:i+1]))
                      for i in range(layers)]
            values = factorize_layers(states, pool.vbar, pool.cfg,
                                       omega=omega if batch == 1 else torch.cat((omega, omega)))
            factors = {name: torch.stack([value[j][-1] for value in values]).cpu()
                       for j, name in enumerate(('a', 'U', 'W'))}
            report = factor_diagnostic(raw.cpu(), data['old_factors'], factors, pool)
            report.update(mode=mode, evidence=path, rank=data['row']['rank'],
                          length=data['row']['length'], final=data['row']['final'],
                          repeat_and_prefix_audit=audit,
                          saved_candidate_equal={name: torch.equal(factors[name], data['factors'][name])
                                                 for name in factors})
            records.append(report)
            print('P3_N128_REPLAY', json.dumps(report), flush=True)
        args.out.write_text(json.dumps(dict(diagnostic_only=True, records=records,
            device=torch.cuda.get_device_name(), tf32=False), indent=2) + '\n')
    assert records, 'no candidate evidence found'


if __name__ == '__main__':
    main()
