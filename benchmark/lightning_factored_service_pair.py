"""Same-rack two-node paired GSM200 / frozen AgentX 900s windows.

Only server orchestration is new. Frozen GSM, AgentX client and reducers remain
unchanged. Failed warmup/control/reducer evidence is retained as review required.
"""
import argparse
import ast
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from lightning_dense_bridge_qualify import atomic, http, worker


def barrier(root, stage, arm, timeout=1500):
    path = root / 'barriers' / stage
    path.mkdir(parents=True, exist_ok=True)
    atomic(path / (arm + '.json'), dict(time=time.time(), node=socket.gethostname()))
    deadline = time.monotonic() + timeout
    while not all((path / (x + '.json')).exists() for x in ('original-off', 'factored')):
        if list(root.glob('*/failure.json')):
            raise RuntimeError('paired arm failed before barrier')
        if time.monotonic() > deadline:
            raise TimeoutError('paired barrier ' + stage)
        time.sleep(.25)
    # Each side derives the SAME release time from immutable readiness receipts.
    release = max(json.loads((path / (x + '.json')).read_text())['time'] for x in ('original-off', 'factored')) + 3
    time.sleep(max(0, release - time.time()))
    return dict(release=release, actual=time.time())


def frozen_reducer(bundle):
    path = bundle / 'perf_runner.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    cut = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Try))
    ns = dict(__name__='frozen_window_definitions', __file__=str(path))
    exec(compile(ast.Module(body=tree.body[:cut], type_ignores=[]), str(path), 'exec'), ns)
    return ns['reduce']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--mode', choices=('gsm', 'perf'), required=True)
    ap.add_argument('--concurrency', type=int, default=32)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    arm = ('original-off', 'factored')[int(os.environ['SLURM_PROCID'])]
    root = args.out / arm
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in plan['source_file_sha256'].items():
        actual = hashlib.sha256((Path(plan['factored_source']) / name).read_bytes()).hexdigest()
        assert actual == expected, name
    os.environ.update(HF_HOME='/work/hf-cache', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      HF_DATASETS_CACHE=plan['datasets_cache'], HF_MODULES_CACHE=str(root / 'client-hf-modules'))
    Path(os.environ['HF_MODULES_CACHE']).mkdir(exist_ok=True)
    sys.path[:0] = plan['helper_paths']
    atomic(root / 'placement.json', dict(node=socket.gethostname(), rank=os.environ['SLURM_PROCID'],
        nodelist=os.environ['SLURM_JOB_NODELIST'], mode=args.mode, C=args.concurrency,
        plan=plan, graph=True, production_mgs_iterations=3))

    def evaluate(url, plan, root, env):
        if args.mode == 'gsm':
            import gsm_endpoint
            atomic(root / 'start-barrier.json', barrier(args.out, 'gsm', arm))
            gsm_endpoint.run(url, plan['model'], root / 'gsm200.json')
            return
        bundle = Path(plan['perf_bundle'])
        env = dict(env, PYTHONPATH=':'.join([str(bundle), *plan['helper_paths'], env['PYTHONPATH']]),
                   HF_DATASETS_CACHE='/work/datasets/hf-cache/datasets',
                   TWINSTAR_PD_WARMUP_FAILURE_AUDIT='1', TMPDIR='/tmp/ltnf-' + os.environ['SLURM_JOB_ID'])
        Path(env['TMPDIR']).mkdir(exist_ok=True)
        start = time.monotonic()
        sequence = []
        for i in range(8):
            payload = dict(input_ids=[1000]*8192,
                sampling_params=dict(temperature=0, max_new_tokens=32, ignore_eos=True),
                rid=f'remeasure-prewarm-{i}')
            sequence.append(payload)
            http(url, '/generate', payload)
        time.sleep(max(0, 60 - (time.monotonic()-start)))
        atomic(root / 'model-warmup.json', dict(seconds=time.monotonic()-start,
            requests=8, request_sequence=sequence, minimum_seconds=60))
        http(url, '/flush_cache', {})
        c = args.concurrency
        point = root / f'C{c}'
        command = [sys.executable, str(bundle / 'guarded_observed_client.py'),
            '--url', url, '--model', plan['model'], '--tokenizer', 'twinstar/lightning-tokenizer',
            '--out', str(point), '--server-log', str(root / 'server.log'), '--concurrency', str(c),
            '--warmup-requests-per-lane', '2', '--repetitions', '1', '--benchmark-duration', '900',
            '--expected-mamba-cache-size', '40']
        atomic(root / 'configuration.json', json.loads((root / 'execution.json').read_text()))
        atomic(root / f'C{c}-client-command.json', dict(command=command, frozen_client=True))
        atomic(root / 'start-barrier.json', barrier(args.out, 'perf', arm))
        with (root / f'C{c}-client.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=2400)
        os.environ.update(KDA_OUT=str(args.out), KDA_BUNDLE=str(bundle), KDA_RELEASE=plan['release'])
        sys.path.insert(0, str(bundle))
        try:
            row = frozen_reducer(bundle)(root, c, arm)
            # Preserve historical reducer fields, but never mislabel executed source.
            row['frozen_reducer_build_sha'] = row.pop('build_sha')
            row['build_sha'] = plan['factored_sha'] if arm == 'factored' else plan['published_sha']
            row['state_storage'] = 'factored FP32 R16 W16 no dense resident' if arm == 'factored' else 'native dense FP32'
            atomic(root / 'cell.json', row)
        except Exception as error:
            atomic(root / 'review-required.json', dict(error=repr(error), formal=False, raw=str(point)))
        from request_timeseries import export
        export(point)

    try:
        worker(plan, arm, 0, root, micro=False, gsm=False, evaluate=evaluate)
    except BaseException as error:
        atomic(root / 'failure.json', dict(error=repr(error), time=time.time()))
        raise


if __name__ == '__main__':
    main()
