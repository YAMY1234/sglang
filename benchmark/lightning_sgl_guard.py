"""One complete 32 x 256 Lightning numerical guard, without sample shortcuts.

Four workers share one four-GPU allocation: repeated pinned reference,
stock/repeat/flag-off, native SGLang DUET, and an accuracy-first option smoke. All cells must finish before
any numerical PASS is possible. Workers preserve per-token losses and IDs.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from lightning_sgl_decision import assess
from lightning_sgl_stage2 import load_cell, now, request, save, server

REFERENCE_SHA = 'dd9c7bdbd9550a5d86781ebfa3d965d01fc1e78a'


def validate_windows(data):
    if len(data['windows']) != 32:
        raise ValueError('the numerical guard requires all 32 windows')
    for row in data['windows']:
        if len(row['prompt']) != 3840 or len(row['target']) != 256:
            raise ValueError('the numerical guard requires 3840 prompt + 256 target tokens')
    return data['windows']


def engine_losses(endpoint, window):
    ids = window['prompt'] + window['target']
    response = request(endpoint, '/generate', {
        'input_ids': ids, 'sampling_params': {'temperature': 0, 'max_new_tokens': 1, 'ignore_eos': True},
        'return_logprob': True, 'logprob_start_len': len(window['prompt']) - 1,
    }, timeout=1800)
    rows = response['meta_info']['input_token_logprobs']
    if len(rows) != 257 or rows[0][0] is not None:
        raise AssertionError(f'unexpected SGLang logprob alignment: {len(rows)} rows, first={rows[:1]}')
    rows = rows[1:]
    if [row[1] for row in rows] != window['target']:
        raise AssertionError('SGLang logprob token IDs differ from all 256 teacher targets')
    losses = [-float(row[0]) for row in rows]
    if not all(math.isfinite(x) for x in losses):
        raise AssertionError('non-finite engine NLL')
    return losses


def engine_pass(args, endpoint, windows, name):
    rows = []
    start = time.monotonic()
    for i, window in enumerate(windows):
        losses = engine_losses(endpoint, window)
        rows.append(dict(id=window['id'], losses=losses, nll=sum(losses) / 256,
                         prompt_sha256=window['prompt_sha256'], target=window['target']))
        save(args.out / (name + '.json'), dict(name=name, complete=len(rows) == 32, windows=rows,
                                              seconds=time.monotonic() - start))
        print(f'{name} {i+1}/32 nll={rows[-1]["nll"]:.8f} seconds={time.monotonic()-start:.2f}', flush=True)


def reference_worker(args, windows):
    import torch
    audit = json.loads(Path('/task/reference-closure-audit.json').read_text())
    for name, expected in audit['files'].items():
        if hashlib.sha256((Path('/reference') / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'reference source changed since pin audit: {name}')
    from twinstar.models.ckpt import twinstar_from_spec
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = twinstar_from_spec(args.model, 'all', 'all', args.duet)
    for repeat in (1, 2):
        torch.manual_seed(20260929)
        rows = []
        start = time.monotonic()
        name = f'reference{repeat}'
        for i, window in enumerate(windows):
            logits = []
            generated = model.generate_batch([window['prompt']], 256, forced=[window['target']], logits_out=logits)
            if generated != [window['target']] or len(logits) != 256:
                raise AssertionError('reference teacher-forcing contract changed')
            stacked = torch.cat(logits, 0).float()
            targets = torch.tensor(window['target'], device=stacked.device, dtype=torch.long)
            losses = torch.nn.functional.cross_entropy(stacked, targets, reduction='none').cpu().tolist()
            if not all(math.isfinite(x) for x in losses):
                raise AssertionError('non-finite reference NLL')
            rows.append(dict(id=window['id'], losses=losses, nll=sum(losses) / 256,
                             prompt_sha256=window['prompt_sha256'], target=window['target']))
            del logits, stacked
            save(args.out / (name + '.json'), dict(name=name, complete=len(rows) == 32, windows=rows,
                                                  seconds=time.monotonic() - start, reference_sha=REFERENCE_SHA))
            print(f'{name} {i+1}/32 nll={rows[-1]["nll"]:.8f} seconds={time.monotonic()-start:.2f}', flush=True)



def coordinator(args):
    record = dict(cell='complete-numerical-guard', status='running', started_at=now(), job_id=os.environ.get('SLURM_JOB_ID'),
                  reference_sha=REFERENCE_SHA, windows_sha256=hashlib.sha256(args.windows.read_bytes()).hexdigest())
    start = time.monotonic()
    workers = []
    logs = []
    try:
        for mode, gpu in [('reference', 0), ('control', 1), ('engine', 2), ('options', 3)]:
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            env['TWINSTAR_DEVICES'] = 'cuda:0'
            env['PYTHONPATH'] = args.reference_repo + os.pathsep + env.get('PYTHONPATH', '')
            # Do not inherit diagnostic knobs that change the released reference.
            for key in ('LATENT_OFF', 'STATE_OFF', 'TWINSTAR_STATE_TRUNCATION', 'TWINSTAR_STATE_OVERSAMPLE', 'TWINSTAR_STATE_POWER'):
                env.pop(key, None)
            cmd = [sys.executable, __file__, '--worker', mode, '--gpu', str(gpu), '--model', args.model,
                   '--duet', args.duet, '--windows', str(args.windows), '--out', str(args.out)]
            handle = (args.out / (mode + '.log')).open('w'); logs.append(handle)
            workers.append((mode, subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT)))
        while any(proc.poll() is None for _,proc in workers):
            for mode, proc in workers:
                if proc.poll() not in (None, 0):
                    raise RuntimeError(f'{mode} worker exited {proc.returncode}; entire numerical cell failed')
            time.sleep(5)
        for mode, proc in workers:
            if proc.returncode:
                raise RuntimeError(f'{mode} worker exited {proc.returncode}')
        names = ('reference1', 'reference2', 'duet', 'duet2', 'stock1', 'stock2', 'off', 'off2')
        record['accuracy_first_options'] = json.loads((args.out / 'accuracy-first-options.json').read_text())['status']
        record.update(assess({name: json.loads((args.out / (name + '.json')).read_text()) for name in names}))
        if record['accuracy_first_options'] != 'pass':
            record['status'] = 'fail'
    except Exception as exc:
        record.update(status='fail', error=f'{type(exc).__name__}: {exc}')
        for _, proc in workers:
            if proc.poll() is None: proc.terminate()
        for _, proc in workers:
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
    finally:
        for log in logs: log.close()
        record.update(finished_at=now(), wall_seconds=time.monotonic()-start)
        save(args.out / 'result.json', record)
        print(json.dumps(record), flush=True)
    return int(record['status'] != 'pass')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['coordinator', 'reference', 'control', 'engine', 'options'], default='coordinator')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model', required=True)
    parser.add_argument('--duet', required=True)
    parser.add_argument('--windows', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference-repo', default='/reference')
    parser.add_argument('--duet-numerics', choices=['reference', 'production'], default='reference',
                        help='numerics profile for the DUET arms (docs/167 §4; guards use reference)')
    parser.add_argument('--cell-prefix', default='duet', help="engine cell names: <prefix>, <prefix>2 (e.g. duetp for production)")
    parser.add_argument('--port', type=int, default=31336, help='engine worker server port (distinct per concurrent engine worker)')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    windows = validate_windows(json.loads(args.windows.read_text()))
    if args.worker == 'coordinator': return coordinator(args)
    if args.worker == 'reference': reference_worker(args, windows)
    elif args.worker == 'options':
        args.port = 31339
        args.duet_cli = ['--no-prefill-layer-trim', '--prefill-saving-policy', 'kv-and-ssm',
                         '--decode-ssm-r', '0', '--decode-ssm-w', '0']
        out = args.out / 'accuracy-first-server'; out.mkdir(parents=True, exist_ok=True)
        result = dict(cell='accuracy-first-options-smoke', status='running', started_at=now())
        try:
            result['rounds'] = [load_cell(args, out / f'round{i}') for i in (1, 2)]
            result['rounds_complete'] = 2
            result['status'] = 'pass'
            result['prune_boundaries_exercised'] = []
            result['semantics'] = 'full-depth prefill and exact decode state; no latent or pruning'
        except Exception as exc:
            result.update(status='fail', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            result['finished_at'] = now(); save(args.out / 'accuracy-first-options.json', result)
    elif args.worker == 'control':
        with server(args, 'stock', args.out / 'stock-server', port=31335) as (endpoint, _):
            engine_pass(args, endpoint, windows, 'stock1'); engine_pass(args, endpoint, windows, 'stock2')
        with server(args, 'off', args.out / 'off-server', port=31335) as (endpoint, _):
            engine_pass(args, endpoint, windows, 'off'); engine_pass(args, endpoint, windows, 'off2')
    else:
        prefix = args.cell_prefix
        with server(args, 'duet', args.out / f'{prefix}-server', port=getattr(args, 'port', 31336)) as (endpoint, _):
            engine_pass(args, endpoint, windows, prefix); engine_pass(args, endpoint, windows, prefix + '2')
    return 0


if __name__ == '__main__':
    sys.exit(main())
