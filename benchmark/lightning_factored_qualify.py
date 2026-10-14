"""M1 then matched GPU0 three-arm M2, three rounds in each server process."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from lightning_dense_bridge_qualify import atomic, worker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    source = Path(plan['factored_source'])
    hashes = {p: hashlib.sha256((source/p).read_bytes()).hexdigest() for p in plan['source_file_sha256']}
    assert hashes == plan['source_file_sha256'], 'source hash mismatch'
    atomic(root/'source-execution.json', dict(sha=plan['factored_sha'], hashes=hashes))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='3')
    with (root/'m1.log').open('w') as log:
        code = subprocess.call([sys.executable, str(source/'test/manual/duet/test_mamba2_factored.py'),
                                '--out', str(root/'m1.json')], env=env, stdout=log, stderr=subprocess.STDOUT)
    atomic(root/'m1-exit.json', dict(code=code))
    # Preserve failed numerical receipts; measurement is diagnostic, never
    # converted into a numerical PASS. A launch/runtime failure stops M2.
    if not (root/'m1.json').exists():
        raise RuntimeError('M1 did not produce numerical receipts')
    failures = {}
    for arm in ('original-off', 'bridge', 'factored'):
        print('STEP arm='+arm+' starting', flush=True)
        try:
            worker(plan, arm, 0, root/'micro'/arm, micro=True, gsm=False)
        except Exception as error:
            failures[arm] = repr(error)
            atomic(root/'failures.json', failures)
    atomic(root/'complete.json', dict(failures=failures, m1_exit=code))
    if failures:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
