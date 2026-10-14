"""Read-only B32 attribution of an existing paired graph profiler round.

Cycle means include one expiry. CUDA duration sums are not wall time and do
not remove cross-stream overlap. Occupancy is a resource-derived estimate.
"""
import argparse
import collections
import gzip
import json
import statistics
from pathlib import Path

STATE = {'_step', '_truncate', '_selective_scan_update_kernel'}


def category(name):
    if name == '_step':
        return 'factor_step'
    if name == '_truncate':
        return 'truncate'
    if 'selective_scan_update' in name or 'selective_state_update' in name:
        return 'native_ssu'
    low = name.lower()
    if 'causal_conv' in low:
        return 'conv'
    if name == '_layer_norm_fwd_1pass_kernel':
        return 'gated_norm'
    if 'norm' in low:
        return 'other_norm'
    if 'copy' in low or 'cast' in low:
        return 'copy_cast'
    if any(x in low for x in ('softplus', 'silu', 'exp_kernel')):
        return 'standalone_dt_gate'
    return 'remaining_gemm_attention_moe_misc'


def run(root, batch):
    result = dict(batch=batch, units='microseconds', cycle_steps=16, arms={})
    for arm in ('original-off', 'factored'):
        path = root / 'micro' / arm
        rows, metadata, totals = [], {}, collections.Counter()
        resources = {}
        for f in sorted((path/'triton_cache_dir').glob('**/*.json')):
            obj = json.loads(f.read_text())
            if obj.get('name') in STATE:
                keys = ('hash', 'name', 'target', 'triton_version', 'num_warps',
                        'num_ctas', 'shared', 'num_stages', 'enable_fp_fusion', 'launch_pdl')
                metadata[str(f.relative_to(root))] = {k: obj.get(k) for k in keys}
        for i in range(16):
            file = path / 'probe' / f'B{batch}-r3-trace-step{i:03d}.json.gz'
            with gzip.open(file, 'rt') as f:
                trace = json.load(f)
            kernels = [x for x in trace['traceEvents'] if x.get('cat') == 'kernel']
            bycat = collections.Counter()
            layers = []
            for k in kernels:
                bycat[category(k['name'])] += k['dur']
                totals[k['name']] += k['dur']
                if k['name'] in STATE:
                    a = k['args']
                    keys = ('registers per thread', 'shared memory', 'grid', 'block',
                            'est. achieved occupancy %', 'occupancy')
                    resource = {key: a.get(key) for key in keys}
                    resources.setdefault(k['name'], {})[json.dumps(resource, sort_keys=True)] = resource
                if category(k['name']) in ('native_ssu', 'factor_step'):
                    layers.append(k)
            layers.sort(key=lambda k: k['ts'])
            assert len(layers) == 23, (arm, i, len(layers))
            assert abs(sum(bycat.values()) - sum(k['dur'] for k in kernels)) < 1e-6
            rows.append(dict(step=i, category_us=dict(bycat), kernel_count=len(kernels),
                             layer_us=[x['dur'] for x in layers]))
        cats = set().union(*(row['category_us'] for row in rows))
        cycle = {c: statistics.mean(row['category_us'].get(c, 0.) for row in rows) for c in sorted(cats)}
        expiry = max(range(16), key=lambda i: rows[i]['category_us'].get('truncate', 0.)) if arm == 'factored' else None
        # Synthetic micro begins at a known phase; retain explicit observed
        # duration evidence, rather than silently excluding the expensive step.
        if expiry is not None:
            assert expiry == 14, expiry
        ordinary = [x for x in rows if x['step'] != expiry]
        result['arms'][arm] = dict(
            cycle_mean_us=cycle, ordinary_mean_us={c: statistics.mean(x['category_us'].get(c, 0.) for x in ordinary) for c in sorted(cats)},
            expiry_step=expiry, expiry_us=rows[expiry]['category_us'] if expiry is not None else None,
            layer_cycle_mean_us=[statistics.mean(x['layer_us'][i] for x in rows) for i in range(23)],
            resources={k: list(v.values()) for k, v in resources.items()},
            triton_compile_metadata=metadata,
            by_name_cycle_mean_us={k: v/16 for k, v in totals.most_common()}, steps=rows,
        )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()
    result = run(args.root, args.batch)
    out = args.root / f'factored-attribution-B{args.batch}.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    for arm, row in result['arms'].items():
        print(arm, json.dumps(row['cycle_mean_us']))
        print('per_layer_us', statistics.mean(row['layer_cycle_mean_us']),
              'range', min(row['layer_cycle_mean_us']), max(row['layer_cycle_mean_us']))
        print('ordinary', json.dumps(row['ordinary_mean_us']))
        print('expiry', json.dumps(row['expiry_us']))
        print('resources', json.dumps(row['resources']))


if __name__ == '__main__':
    main()
