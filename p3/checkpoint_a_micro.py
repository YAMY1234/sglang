"""Actual P states: diagnose per-layer versus all-layer sink projection.

Same S and vbar, no rank comparison. This is a numerical/price diagnostic,
not an admission gate and not a replacement for stored-pool comparisons.
"""
import argparse
import glob
import json
from pathlib import Path
import statistics


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dumps', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(glob.glob(args.dumps))
    if not paths:
        parser.error('no actual P dumps')
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []

    def project(s, vb):
        return torch.einsum('bhvk,hv->bhk', s, vb) / vb.square().sum(-1).clamp_min(1e-12)[None, :, None]

    for path in paths:
        data = torch.load(path, map_location='cpu', weights_only=False)
        assert data['meta']['role'] == 'prefill' and data['meta']['shallow_fulln']
        vb = data['vbar'].cuda().float()
        for name in ('S', 'S_tracked'):
            if data.get(name) is None:
                continue
            s = data[name].cuda().float().contiguous()
            b, layers, heads, width, key = s.shape

            def layerwise():
                return torch.stack([project(torch.stack([s[:, i]], dim=1).reshape(b, heads, width, key),
                                            vb[i]) for i in range(layers)], dim=1)

            def batched():
                return project(s.reshape(b, layers * heads, width, key),
                               vb.reshape(layers * heads, width)).reshape(b, layers, heads, key)

            old, new = layerwise(), batched()
            reference = project(s.double().reshape(b, layers * heads, width, key),
                                vb.double().reshape(layers * heads, width)).float().reshape_as(new)
            graphs = {}
            for label, operation in (('layer', layerwise), ('batch', batched)):
                operation()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = operation()
                graphs[label] = graph, result
            deltas = []
            for _ in range(30):
                times = {'layer': [], 'batch': []}
                for label in ('layer', 'batch', 'batch', 'layer'):
                    begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    begin.record(); graphs[label][0].replay(); end.record(); end.synchronize()
                    times[label].append(begin.elapsed_time(end))
                deltas.append(statistics.mean(times['layer']) - statistics.mean(times['batch']))
            equal = (old == new)
            row = dict(dump=path, state=name, shape=list(s.shape),
                       bitwise_equal=torch.equal(old, new), unequal=int((~equal).sum()),
                       layers=[data['meta']['layer_ids'][i] for i in range(layers)
                               if not bool(equal[:, i].all())],
                       max_abs=float((old - new).abs().max()),
                       layer_fp64_max=float((old - reference).abs().max()),
                       batch_fp64_max=float((new - reference).abs().max()),
                       canonical_projection_extra_ms=dict(blocks=30, median=statistics.median(deltas),
                           mean=statistics.mean(deltas), se=statistics.stdev(deltas) / 30 ** .5))
            records.append(row)
            print('P3_A_PROJECTION_DIAGNOSTIC', json.dumps(row), flush=True)
    args.out.write_text(json.dumps(dict(scope='same actual P S, diagnostic only',
        device=torch.cuda.get_device_name(), torch=torch.__version__, tf32=False, records=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
