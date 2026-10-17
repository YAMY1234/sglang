"""docs/229 §3 P3 / P5 reader for route_micro samples-r<rank>.jsonl: per shape and pair (a, b), per ABBA block
d = mean(route b orders) - mean(route a orders) of gpu_ms / host_ms; median / mean / s.e. over the 30 blocks per rank;
fallback deltas; P5 pre-registration verdict on route 1 - 0n (pass <= -3 ms, kill > -1.5 ms, else n/j, per shape, both ranks).
pub_graph_ms = the publication graph span of each measured forward (p3_factory counters 'last_pub_graph_ms', recorded in each
row's 'after'): route b - a per block like the others, plus absolute medians per route; P5 pre-registration: route 1's
publication graph <= 2.4 ms.
  python3 p3_reduce.py <out dir> [--stage p5|p3]"""
import argparse
import collections
import json
import statistics as st
from pathlib import Path

PASS, KILL = -3.0, -1.5           # P5 C+: route 1 - 0n per b = 1 forward (ms)
PASS_P3, KILL_P3 = -8.0, -3.0     # P3 PC+: 1 - 0n pass; full-N alone (0 - 0n) kill above
PUB_MAX = 2.4                     # P5: route 1 publication graph (ms)
FIELDS = ('gpu_ms', 'host_ms', 'pub_graph_ms')


def value(row, field):
    return row['after'].get('last_pub_graph_ms', 0.0) if field == 'pub_graph_ms' else row[field]


def blocks(rows):
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r['prefix'], r['extend'], tuple(r['pair']), r['block'])][r['order']] = r
    out = collections.defaultdict(list)
    for (p, e, pair, blk), o in by.items():
        if len(o) != 4:
            continue
        a, b = pair
        for f in FIELDS:
            ma = st.mean(value(o[i], f) for i in range(4) if o[i]['route'] == a)
            mb = st.mean(value(o[i], f) for i in range(4) if o[i]['route'] == b)
            out[(p, e, pair, f)].append(mb - ma)
            for i in range(4):
                out[(p, e, ('abs', o[i]['route']), f)].append(value(o[i], f))
        out[(p, e, pair, 'fallback_delta')].append(sum(o[i]['after']['fallbacks'] - o[i]['before']['fallbacks'] for i in range(4)))
    return out


def summarize(d):
    n = len(d)
    return dict(n=n, median=round(st.median(d), 3), mean=round(st.mean(d), 3),
                se=round(st.stdev(d) / n ** .5, 3) if n > 1 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', type=Path)
    ap.add_argument('--stage', choices=('p5', 'p3'), default='p5')
    a = ap.parse_args()
    res = {}
    for f in sorted(a.out.glob('samples-r*.jsonl')):
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        for (p, e, pair, field), d in blocks(rows).items():
            label = f'abs:{pair[1]}' if pair[0] == 'abs' else f'{pair[1]}-{pair[0]}'
            key = f'{f.stem}|{p}+{e}|{label}|{field}'
            res[key] = summarize(d) if field != 'fallback_delta' else dict(total=sum(d))
    verdict = {}
    for k, v in res.items():
        rank, shape, pair, field = k.split('|')
        if field == 'pub_graph_ms' and pair == 'abs:1':
            verdict[f'{rank}|{shape}|pub'] = 'ok' if v['median'] <= PUB_MAX else 'over'
        if field != 'gpu_ms':
            continue
        m = v['median']
        if a.stage == 'p5' and pair == '1-0n':
            verdict[f'{rank}|{shape}'] = 'pass' if m <= PASS else 'kill' if m > KILL else 'n/j'
        if a.stage == 'p3' and pair == '1-0n':
            verdict[f'{rank}|{shape}|1-0n'] = 'pass' if m <= PASS_P3 else 'n/j'
        if a.stage == 'p3' and pair == '0-0n':
            verdict[f'{rank}|{shape}|fullN'] = 'kill' if m > KILL_P3 else 'ok'
    falls = sum(v.get('total', 0) for k, v in res.items() if k.endswith('fallback_delta'))
    for k, v in sorted(res.items()):
        print(f'{k:60s} {v}')
    print('VERDICT', json.dumps(verdict), 'fallbacks', falls)
    (a.out / 'reduce.json').write_text(json.dumps(dict(rows=res, verdict=verdict, fallbacks=falls,
                                                       publish_graph_ms='read from the CUDA timeline'), indent=1) + '\n')


if __name__ == '__main__':
    main()
