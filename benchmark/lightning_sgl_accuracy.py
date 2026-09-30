"""Paired GSM8K-200 through eval_thinking HTTP clients after the NLL guard.

The reference HTTP adapter calls the pinned PyTorch release loader/generator;
the engine endpoint is the native SGLang DUET implementation. This adapter is
an evaluation control, never the SGLang implementation under test.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from lightning_sgl_stage2 import now, request, save, server


def reference_server(args):
    import torch
    from transformers import AutoTokenizer
    from twinstar.models.ckpt import twinstar_from_spec
    audit = json.loads(Path('/task/reference-closure-audit.json').read_text())
    for name, expected in audit['files'].items():
        if hashlib.sha256((Path('/reference') / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'reference source changed: {name}')
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = twinstar_from_spec(args.model, 'all', 'all', args.duet)
    eos = {int(e) for e in (tok.eos_token_id, model.cfg.eos_token_id) if e is not None}
    for text in ('<|im_end|>', '<|endoftext|>'):
        token = tok.convert_tokens_to_ids(text)
        if isinstance(token, int) and token >= 0 and token != tok.unk_token_id: eos.add(token)
    pad = tok.pad_token_id or tok.eos_token_id or 0

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, data):
            payload = json.dumps(data, allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)

        def do_GET(self):
            self.respond(200 if self.path == '/health' else 404, {'reference_sha': audit['reference_sha']})

        def do_POST(self):
            if self.path != '/generate': return self.respond(404, {'error': 'unknown endpoint'})
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                ids, params = body['input_ids'], body['sampling_params']
                if not ids or not all(isinstance(t, int) for t in ids):
                    raise ValueError('reference adapter requires one nonempty token-ID prompt')
                maximum = int(params['max_new_tokens'])
                generator = torch.Generator(device=model.embed_tokens.weight.device)
                generator.manual_seed(int(params.get('sampling_seed', 20260929)))
                start = time.monotonic()
                output = model.generate_batch([ids], maximum, eos_ids=() if params.get('ignore_eos') else eos,
                    pad_id=pad, temperature=float(params.get('temperature', 0)), top_p=float(params.get('top_p', 1)),
                    top_k=int(params.get('top_k', 0)), generator=generator)[0]
                stop = bool(output and output[-1] in eos and not params.get('ignore_eos'))
                self.respond(200, {'text': tok.decode(output, skip_special_tokens=True), 'output_ids': output,
                    'meta_info': {'completion_tokens': len(output), 'prompt_tokens': len(ids),
                        'finish_reason': {'type': 'stop' if stop else 'length'}, 'e2e_latency': time.monotonic()-start}})
                print(f'reference generated {len(output)} tokens in {time.monotonic()-start:.3f}s', flush=True)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.respond(500, {'error': f'{type(exc).__name__}: {exc}'})

    print('reference endpoint ready', flush=True)
    HTTPServer(('127.0.0.1', 31337), Handler).serve_forever()


def evaluate(args, endpoint, label):
    output = args.out / label
    output.mkdir(parents=True, exist_ok=True)
    # Insert only the few changed evaluation modules into the existing package
    # search path. No source checkout, symlink or environment copy is needed.
    bootstrap = "import runpy,twinstar; twinstar.__path__.insert(0,'/task/eval-overlay'); runpy.run_module('twinstar.eval_thinking',run_name='__main__')"
    cmd = [sys.executable, '-c', bootstrap, '--model', args.model, '--endpoint', endpoint,
           '--tasks', 'gsm8k', '--gsm8k-file', '/task/gsm8k-test.jsonl', '--limit', '200', '--reps', '1',
           '--budgets', '0', '--max-new', '2048', '--temperature', '0', '--top-p', '1', '--top-k', '0',
           '--paired-sampling-seed', '20260929', '--generation-checkpoint', '--out', str(output)]
    env = os.environ.copy(); env['PYTHONPATH'] = '/reference' + os.pathsep + env.get('PYTHONPATH', '')
    with (output / 'eval.log').open('w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((output / 'gsm8k_b0.json').read_text())


def worker(args):
    if args.worker == 'reference':
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = '0'; env['TWINSTAR_DEVICES'] = 'cuda:0'
        env['PYTHONPATH'] = '/reference' + os.pathsep + env.get('PYTHONPATH', '')
        log = (args.out / 'reference-server.log').open('w')
        proc = subprocess.Popen([sys.executable, __file__, '--worker', 'reference-server', '--model', args.model,
                                 '--duet', args.duet, '--out', str(args.out)], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            start = time.monotonic()
            while True:
                if proc.poll() is not None: raise RuntimeError('reference server exited during startup')
                try: request('http://127.0.0.1:31337','/health',timeout=2); break
                except OSError:
                    if time.monotonic()-start>600: raise TimeoutError('reference startup exceeded 600s')
                    time.sleep(2)
            evaluate(args, 'http://127.0.0.1:31337', 'reference')
        finally:
            proc.terminate()
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            log.close()
    else:
        args.gpu = 1
        with server(args, 'duet', args.out / 'engine-server', port=31338) as (endpoint, _):
            evaluate(args, endpoint, 'engine')



def assess_accuracy(ref, engine):
    expected_ids = [f'gsm8k-{i}' for i in range(200)]
    for label, cell in [('reference', ref), ('engine', engine)]:
        if cell['n_questions'] != 200 or cell['reps'] != 1 or len(cell['results']) != 200:
            raise ValueError(f'{label}: accuracy requires exactly 200 questions x 1 repetition')
        if [row['id'] for row in cell['results']] != expected_ids:
            raise ValueError(f'{label}: GSM8K subset/order differs from the registered first 200 test rows')
    if ref['sampling'] != engine['sampling'] or ref['budget'] != engine['budget']:
        raise ValueError('paired accuracy sampling protocol differs')
    for key in ('gsm8k_file_sha256', 'chat_kwargs', 'paired_sampling_seed'):
        if ref['protocol'][key] != engine['protocol'][key]:
            raise ValueError(f'paired accuracy protocol differs: {key}')
    for a, b in zip(ref['results'], engine['results']):
        if (a['id'], a['prompt_hash'], a['gold'], a['sampling_seed']) != (b['id'], b['prompt_hash'], b['gold'], b['sampling_seed']):
            raise ValueError('paired accuracy sample or prompt identity mismatch')
    rc = sum(bool(row['correct']) for row in ref['results'])
    ec = sum(bool(row['correct']) for row in engine['results'])
    return dict(status='pass' if ec >= rc-4 else 'fail', reference_correct=rc, engine_correct=ec,
                reference_accuracy=rc/200, engine_accuracy=ec/200, delta_percentage_points=(ec-rc)/2,
                criterion='engine correct >= reference correct - 4 of 200')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['coordinator', 'reference', 'engine', 'reference-server'], default='coordinator')
    parser.add_argument('--model', required=True); parser.add_argument('--duet', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--guard-result', type=Path)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    if args.worker == 'reference-server': reference_server(args); return 0
    if args.worker != 'coordinator': worker(args); return 0
    if args.guard_result is None: parser.error('--guard-result is required before accuracy evaluation')
    guard = json.loads(args.guard_result.read_text())
    if guard.get('status') != 'pass' or guard.get('continuation_tokens') != 8192:
        raise ValueError('the complete numerical guard has not passed')
    record = dict(cell='gsm8k-200-paired', status='running', started_at=now(), job_id=os.environ.get('SLURM_JOB_ID'),
                  guard_sha256=hashlib.sha256(args.guard_result.read_bytes()).hexdigest())
    start=time.monotonic();workers=[];logs=[]
    try:
        for label in ('reference','engine'):
            log=(args.out/(label+'.log')).open('w');logs.append(log)
            workers.append(subprocess.Popen([sys.executable,__file__,'--worker',label,'--model',args.model,
                '--duet',args.duet,'--out',str(args.out)],stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in workers):
            if any(p.poll() not in (None,0) for p in workers): raise RuntimeError('accuracy worker failed')
            time.sleep(5)
        if any(p.returncode for p in workers): raise RuntimeError('accuracy worker failed')
        ref=json.loads((args.out/'reference/gsm8k_b0.json').read_text())
        engine=json.loads((args.out/'engine/gsm8k_b0.json').read_text())
        record.update(assess_accuracy(ref, engine))
    except Exception as exc:
        record.update(status='fail',error=f'{type(exc).__name__}: {exc}')
    finally:
        for p in workers:
            if p.poll() is None:p.terminate()
        for p in workers:
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()
        record.update(finished_at=now(),wall_seconds=time.monotonic()-start)
        save(args.out/'result.json',record);print(json.dumps(record),flush=True)
    return int(record['status']!='pass')


if __name__=='__main__': sys.exit(main())
