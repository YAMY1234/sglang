"""CPU audit of the qualified I4 C8 pair; preserve protocol REVIEW flags."""
import argparse
import ast
import collections
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--qualification', type=Path, required=True)
    ap.add_argument('--bundle', type=Path, required=True)
    args = ap.parse_args()
    hashes = {}

    def read(path):
        raw = path.read_bytes(); hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    plan = read(args.plan); qualification = read(args.qualification/'result.json')
    assert hashes[str(args.qualification/'result.json')] == plan['qualification_result_sha256']
    selected = plan['selected_variant']
    assert qualification['complete'] and qualification['variant_passed'][selected]
    arms = {}
    for arm in ('original-off', 'factored'):
        root = args.root/arm
        row = read(root/'reduced-from-raw.json')
        execution = read(root/'execution.json')
        assert row['complete_900s_cohort'] and row['reduced_from_raw']
        assert '--disable-cuda-graph' not in execution['argv']
        assert '--disable-overlap-schedule' in execution['argv']
        assert '--disable-radix-cache' in execution['argv']
        assert execution['source_sha'] == plan['factored_sha' if arm == 'factored' else 'published_sha']
        env = execution['env']
        assert env['SGLANG_LTN_PREFILL_COMMIT'] == (plan['prefill_commit_mode'] if arm == 'factored' else 'reference')
        assert env['SGLANG_LTN_NATIVE_BATCHED_SHALLOW'] == ('1' if arm == 'factored' else '0')
        assert not env['LTNBRIDGE_PROBE_DIR']
        arms[arm] = row
    a, b = arms['original-off'], arms['factored']
    assert a['reducer_sha256'] == b['reducer_sha256']
    # Same r2 reducer owns all boundaries, including successful warmup and drain.
    sys.path.insert(0, str(args.bundle))
    from request_timeseries import pair_warmup
    try:
        warm = pair_warmup(args.root/'original-off/C8', args.root/'factored/C8')
    except Exception as error:
        warm = dict(passed=False, error=repr(error))
    ratios = dict(tps=b['tps_gpu']/a['tps_gpu'], user_p90=b['user_p90']/a['user_p90'],
                  ttft_p50=b['ttft_p50_s']/a['ttft_p50_s'], hit_delta_pp=100*(b['hit']-a['hit']))
    gates = dict(user_rate=ratios['user_p90'] >= .95, tps=ratios['tps'] >= .97)
    old_gates = dict(tps=gates['tps'], ttft=ratios['ttft_p50'] <= 1.05,
                     hit=ratios['hit_delta_pp'] >= -1)
    # Log routing is production metadata, not a timing probe. Includes warmup.
    handoffs = []
    interior_handoffs = []
    start_s, end_s = b['summary']['window_start_ns']/1e9, b['summary']['window_end_ns']/1e9
    for line in (args.root/'factored/server.log').read_text().splitlines():
        marker = 'Lightning DUET batched extend: '
        if marker in line:
            row = ast.literal_eval(line.split(marker, 1)[1])
            assert row['prefill_commit_mode'] == plan['prefill_commit_mode']
            assert not row['shallow_request_local'] and not row['emitter_native_batched']
            handoffs.append(row)
            stamp = datetime.strptime(line[1:20], '%Y-%m-%d %H:%M:%S').replace(
                tzinfo=ZoneInfo('America/Los_Angeles')).timestamp()
            # Log timestamps have second precision. Admit only bins entirely
            # inside the scored window; this is routing, not GPU stage timing.
            if start_s <= stamp and stamp+1 <= end_s:
                interior_handoffs.append(row)
    assert handoffs
    starts = [r['summary']['window_start_ns'] for r in arms.values()]
    ends = [r['summary']['window_end_ns'] for r in arms.values()]
    formal = bool(a['formal_valid'] and b['formal_valid'] and warm['passed'])
    result = dict(selected_variant=selected, arms=arms, ratios=ratios,
        preregistered_numeric_gates=gates, numerical_support=all(gates.values()),
        docs197_numeric_gates=old_gates, warmup_pair=warm, formal_valid=formal,
        formal_support=formal and all(gates.values()),
        profiling_overlap_seconds=max(0, min(ends)-max(starts))/1e9,
        routing=dict(verified_prefills_including_warmup=len(handoffs),
            batch_histogram=dict(collections.Counter(r['requests'] for r in handoffs)),
            interior_window_batch_histogram=dict(collections.Counter(r['requests'] for r in interior_handoffs)),
            interior_window_prefills=len(interior_handoffs),
            interior_scope='host routing log; complete second bins inside scored window; boundary bins excluded; timezone America/Los_Angeles',
            max_pending_dense_bytes=max(r['peak_pending_state_bytes'] for r in handoffs),
            scope='all logged prefills including startup/model/workload warmup; not a scored-window batch histogram'),
        source_sha256=hashes, report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.root/'i4-window-report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('arms','source_sha256')}, indent=2))


if __name__ == '__main__':
    main()
