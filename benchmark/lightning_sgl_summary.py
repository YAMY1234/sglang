"""Collect the line's Slurm accounting and completed cell evidence (read only)."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo


def collect(root):
    records = {}
    for path in root.glob('job-*.json'):
        job = path.stem.removeprefix('job-')
        if job.isdigit():
            records[job] = json.loads(path.read_text())
    output = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ','.join(sorted(records)),
        '-o', 'JobIDRaw,State,Submit,Start,End,ElapsedRaw,AllocTRES'], text=True)
    jobs = []
    for line in output.splitlines():
        values = line.split('|')
        if len(values) < 7 or values[0] not in records:
            continue
        job, state, submit, start, end, elapsed, tres = values[:7]
        now = datetime.datetime.now(ZoneInfo('America/Los_Angeles'))
        def parse(value):
            return datetime.datetime.fromisoformat(value).replace(tzinfo=now.tzinfo) if 'T' in value else None
        submitted, started = parse(submit), parse(start)
        wait = ((started or now) - submitted).total_seconds()/60 if submitted else None
        allocated = re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)', tres)
        gpus = int(allocated.group(1)) if allocated else 0
        observations = records[job].get('observations', [])
        jobs.append(dict(job_id=job, state=state, submit=submit, start=start, end=end,
                         wait_minutes=wait, waiting_reasonable=wait is not None and wait <= 30,
                         elapsed_seconds=int(elapsed), allocated_gpus=gpus, gpu_hours=int(elapsed)*gpus/3600,
                         phase=1 if job in ('934018','934036') else 2,
                         last_observation={k: observations[-1].get(k) for k in
                             ('observed_at','Reason','LastSchedEval','Priority','idle_nodes')} if observations else None))
    cells = []
    for path in sorted(root.glob('*/result.json')):
        if not path.parent.name.startswith(('stock-', 'load-', 'guard-', 'gsm200-')):
            continue
        data = json.loads(path.read_text())
        cells.append(dict(path=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          result={k:v for k,v in data.items() if k in
                              ('cell','status','job_id','wall_seconds','windows','continuation_tokens','means',
                               'duet_delta_nll','threshold','reference_repeat_noise','flags_off_bitwise',
                               'flags_off_max_token_delta','accuracy_first_options','reference_correct',
                               'engine_correct','delta_percentage_points','error')}))
    return dict(recorded_at=datetime.datetime.now().astimezone().isoformat(), jobs=jobs, cells=cells,
                phase2_gpu_hours_including_running=sum(j['gpu_hours'] for j in jobs if j['phase']==2),
                total_gpu_hours_including_running=sum(j['gpu_hours'] for j in jobs))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    result=collect(args.root)
    path=args.root/'phase2-summary.json'
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(path)
    print(json.dumps({k:v for k,v in result.items() if k not in ('jobs','cells')}))
