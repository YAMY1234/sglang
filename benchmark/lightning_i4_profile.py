"""Matched B8 stage diagnosis on frozen E2b GSM prompts, one trace per arm."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
from lightning_dense_bridge_qualify import atomic, http, worker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--prompts', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    source = Path(plan['factored_source'])
    for name, digest in plan['source_file_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest, name
    prompts = json.loads(args.prompts.read_text())['input_ids']
    assert len(prompts) == 200
    batches = [prompts[i:i+8] for i in range(0, 200, 8)]
    tokens = [sum(map(len, b)) for b in batches]
    representative = min(range(25), key=lambda i: abs(tokens[i]-sum(tokens)/25))
    args.out.mkdir(parents=True, exist_ok=True)
    atomic(args.out/'inputs.json', dict(prompts_sha256=hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        representative=representative, batch_tokens=tokens, trace_lengths=list(map(len,batches[representative]))))

    def run_arm(label, arm, gpu, batched):
        def evaluate(url, _, root, env):
            probe = root/'probe'; probe.mkdir(exist_ok=True)
            responses = []
            def generate(inputs):
                return http(url, '/generate', dict(input_ids=inputs, sampling_params=dict(
                    temperature=0, max_new_tokens=1, ignore_eos=True)))
            for b in batches:
                generate(b)
            # Uninstrumented-stage events control before and after stage probes.
            for mode, order in [('events', range(25)), ('i4-stages', range(25)),
                                ('i4-trace', [representative]), ('events', reversed(range(25)))]:
                for i in order:
                    tag = f'{mode}-p{i}-n{len(responses)}'
                    atomic(probe/'trigger.json', dict(tag=tag, mode=mode, all_forward_modes=True,
                        synchronize_prefill=True, steps=1))
                    response = generate(batches[i])
                    (probe/'trigger.json').unlink()
                    responses.append(dict(mode=mode, index=i, responses=response))
                atomic(root/'responses.json', dict(complete=False, rows=responses))
            atomic(root/'responses.json', dict(complete=True, rows=responses))
        worker(dict(plan, instrument_forward=True, batched_extend=batched), arm, gpu,
               args.out/label, micro=False, gsm=False, evaluate=evaluate)

    failures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run_arm, *item): item[0] for item in
                   [('original', 'original-off', 0, False), ('batched', 'factored', 1, True)]}
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                failures[futures[f]] = repr(e)
            atomic(args.out/'collection.json', dict(complete=False, failures=failures))
    atomic(args.out/'collection.json', dict(complete=not failures, failures=failures))
    if failures:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
