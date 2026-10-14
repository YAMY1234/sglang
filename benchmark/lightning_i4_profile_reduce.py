"""CPU-only phase reduction; launch-correlation attribution, no elapsed/sum mixing."""
import argparse
import bisect
import collections
import gzip
import hashlib
import json
from pathlib import Path


def union(intervals):
    end, total = -float('inf'), 0.
    for a, b in sorted(intervals):
        total += max(0., b-max(a, end))
        end = max(b, end)
    return total


def trace_reduce(path):
    events = json.loads(gzip.open(path, 'rt').read())['traceEvents']
    ranges = [e for e in events if e.get('name','').startswith('I4_STAGE/') and e.get('ph') == 'X']
    spans = sorted(ranges, key=lambda e: e['ts'])
    starts = [e['ts'] for e in spans]
    def phase(e):
        t = e['ts']; i = bisect.bisect_right(starts, t)-1
        # Innermost active range on this CPU thread. A CUDA activity's timestamp
        # cannot identify its CPU owner; call this only on correlated CPU events.
        while i >= 0:
            r = spans[i]
            if r['tid'] == e['tid'] and r['ts'] <= t < r['ts']+r['dur']:
                return r['name'].split('/',1)[1]
            i -= 1
        return 'unattributed'
    def corr(e):
        a=e.get('args',{});return a.get('correlation',a.get('Correlation ID'))
    runtime = [e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and e.get('ph') == 'X']
    correlations = {corr(e):phase(e) for e in runtime if corr(e) is not None}
    externals = {e.get('args',{}).get('External id'):phase(e) for e in events
        if e.get('cat') == 'cpu_op' and e.get('ph') == 'X' and e.get('args',{}).get('External id') is not None}
    totals = collections.defaultdict(lambda: dict(kernel_ms=0., kernel_count=0,
        memcpy_memset_ms=0., cpu_range_exclusive_ms=0., cuda_api_ms=0., cuda_wait_api_ms=0.))
    names = collections.defaultdict(collections.Counter)
    gpu = [e for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and e.get('ph') == 'X']
    intervals = []
    for e in gpu:
        stage = correlations.get(corr(e), externals.get(e.get('args',{}).get('External id'),'unattributed'))
        ms=e['dur']/1000
        if e['cat'] == 'kernel':
            totals[stage]['kernel_ms'] += ms; totals[stage]['kernel_count'] += 1
            names[stage][e['name']] += ms
        else:
            totals[stage]['memcpy_memset_ms'] += ms
        intervals.append((e['ts'],e['ts']+e['dur']))
    for r in ranges:
        children=[(e['ts'],e['ts']+e['dur']) for e in ranges if e is not r and e['tid']==r['tid']
            and e['ts']>=r['ts'] and e['ts']+e['dur']<=r['ts']+r['dur']]
        totals[r['name'].split('/',1)[1]]['cpu_range_exclusive_ms'] += (r['dur']-union(children))/1000
    for e in runtime:
        target=totals[phase(e)];target['cuda_api_ms'] += e['dur']/1000
        if 'Synchronize' in e['name'] or e['name'] in ('cudaMemcpy','cuMemcpyDtoH_v2'):
            target['cuda_wait_api_ms'] += e['dur']/1000
    first=min(a for a,b in intervals);last=max(b for a,b in intervals)
    busy=union(intervals)
    return dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),stages=dict(totals),
        gpu_span_ms=(last-first)/1000,gpu_busy_union_ms=busy/1000,gpu_idle_inside_span_ms=(last-first-busy)/1000,
        top_kernels={s: v.most_common(12) for s,v in names.items()},
        cuda_categories=dict(collections.Counter(e.get('cat') for e in events)),
        note='Kernel sums may overlap; CPU ranges and API waits overlap GPU and must not be added. total stage is residual outside labelled stages.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args();out={}
    for arm in ('original','batched'):
        root=args.root/arm/'probe'
        rows=[json.loads(s) for s in (root/'i4-stages.jsonl').read_text().splitlines()]
        measured=[r for r in rows if r['mode']=='i4-stages']
        assert len(measured)==25 and all(r['batch_size']==8 for r in rows)
        controls=[json.loads(s) for s in (root/'steps.jsonl').read_text().splitlines()]
        controls=[r for r in controls if r['forward_kind']=='prefill']
        assert len(controls)==50 and all(r['batch_size']==8 for r in controls)
        stage_means={s:{k:sum(r['stages'].get(s,{}).get(k,0.) for r in measured)/200
                       for k in ('host_exclusive_ms','cuda_span_exclusive_ms','calls')}
                     for s in set().union(*(r['stages'] for r in measured))}
        traces=list(root.glob('i4-trace*.gz'));assert len(traces)==1
        out[arm]=dict(stage_means_per_request=stage_means,
            instrumented_wall_ms_per_request=sum(r['completed_wall_ms'] for r in measured)/200,
            controls_wall_ms_per_request=sum(r['prefill_completed_wall_ms'] for r in controls)/400,
            controls_rounds_ms_per_request=[sum(r['prefill_completed_wall_ms'] for r in controls[i:i+25])/200 for i in (0,25)],
            trace=trace_reduce(traces[0]),trace_wall=next(r for r in rows if r['mode']=='i4-trace'))
    (args.root/'profile-reduced.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({a:{k:v for k,v in r.items() if k not in ('trace',)} for a,r in out.items()},indent=2))

if __name__=='__main__':main()
