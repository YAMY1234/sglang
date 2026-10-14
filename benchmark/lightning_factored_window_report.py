"""Read-only reconstruction of sealed 900s cohorts; incomplete exports stay invalid."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def reduce_arm(root, concurrency, snapshots):
    from twinstar.bench.duet_agentx_window import analyze
    from twinstar.bench.duet_agentx_user_rate import analyze as user_rates
    from twinstar.bench.flashnext_pd_cohort_cache import analyze as cache
    from twinstar.bench.flashnext_pd_ratio_cell import warmup_lane_receipt
    from accounting_diagnostic import latency_for
    point=root/f'C{concurrency}';run=point/'rep1'
    records=run/'aiperf_artifacts/profile_export.jsonl'
    metric_path=run/'aiperf_artifacts/server_metrics_export.json'
    metrics=json.loads(metric_path.read_text())
    phases=metrics['summary']['phase_time_ranges'];phase=phases['profiling']
    start=phase['start_ns'];end=start+900*10**9
    assert phase['end_ns']>=end and 'warmup' in phases, 'incomplete phase boundaries'
    summary=analyze(records,gpus=1,duration=900,window_start_ns=start)
    latency=latency_for(run,summary,900);rates=user_rates(records,start,end)
    cached=cache(records,json.loads((point/'server-info.json').read_text()),start,end)
    assert rates['valid_requests']==cached['requests']==summary['completed_in_window']>0
    assert rates['invalid_requests']==0
    drain=latency['client_phase_including_drain']
    assert drain['errors']==0 and 900<=drain['elapsed_seconds']<=935
    logs=(run/'client.log').read_text()
    warm_phase=re.findall(r'Phase warmup \(warmup\) complete \| completed=([\d,]+), cancelled=([\d,]+), errors=([\d,]+)',logs)
    assert len(warm_phase)==1, 'missing unique warmup completion receipt'
    all_rows=[json.loads(line) for line in records.open()]
    warm_rows=[r for r in all_rows if r['metadata'].get('benchmark_phase')=='warmup']
    assert len(warm_rows)==int(warm_phase[0][0].replace(',',''))
    assert int(warm_phase[0][1])==int(warm_phase[0][2])==0
    assert all(not r.get('error') and not r['metadata'].get('was_cancelled') and
               r['metadata']['request_end_ns']<=start for r in warm_rows)
    warm=warmup_lane_receipt(run,snapshots,concurrency,2,diagnostic=True)
    return dict(reduced_from_raw=True,complete_900s_cohort=True,
        formal_valid=bool(warm['complete']),warmup_protocol_review=not warm['complete'],
        tps_gpu=summary['tps_per_gpu'],ttft_p50_s=latency['latency_seconds']['time_to_first_token']['p50'],
        hit=cached['hit'],user_p90=rates['output_tps_user']['p90'],
        summary=summary,latency=latency,rates=rates,cache=cached,warmup=warm,phase_ranges=phases,
        raw_record_sha256=hashlib.sha256(records.read_bytes()).hexdigest(),
        raw_metrics_sha256=hashlib.sha256(metric_path.read_bytes()).hexdigest(),
        reducer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--concurrency',type=int,required=True)
    ap.add_argument('--helpers',required=True)
    ap.add_argument('--bundle',required=True)
    ap.add_argument('--snapshots',type=Path,required=True)
    args=ap.parse_args();sys.path[:0]=[args.bundle,args.helpers]
    snapshots=json.loads(args.snapshots.read_text());out={}
    for arm in ('original-off','factored'):
        try:out[arm]=reduce_arm(args.root/arm,args.concurrency,snapshots)
        except Exception as error:out[arm]=dict(reduced_from_raw=False,complete_900s_cohort=False,
            formal_valid=False,error=repr(error))
    out['paired_complete_900s_cohort']=all(out[a]['complete_900s_cohort'] for a in ('original-off','factored'))
    if out['paired_complete_900s_cohort']:
        from request_timeseries import pair_warmup
        try:out['warmup_pair']=pair_warmup(args.root/f'original-off/C{args.concurrency}',args.root/f'factored/C{args.concurrency}')
        except Exception as error:out['warmup_pair']=dict(passed=False,error=repr(error))
        a,b=(out[x] for x in ('original-off','factored'))
        out['ratios']=dict(tps=b['tps_gpu']/a['tps_gpu'],ttft_p50=b['ttft_p50_s']/a['ttft_p50_s'],
            user_p90=b['user_p90']/a['user_p90'],hit_delta_pp=100*(b['hit']-a['hit']))
    path=args.root/'reduced-from-raw.json'
    path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({a:{k:v for k,v in out[a].items() if k not in ('summary','latency','rates','cache','warmup','phase_ranges')} for a in ('original-off','factored')},indent=2))


if __name__=='__main__':main()
