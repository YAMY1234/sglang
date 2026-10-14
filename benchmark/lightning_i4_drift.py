"""KDA frozen-reply protocol port: original/replay, compatible and QR-only.

Full production Lightning shallow prefill in both factor arms. Native original
has dense FP32 recurrent state. Same graph setting, dtype and frozen tokens.
24 replies x 256 decode predictions; prefill prediction is reported separately.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time

from lightning_dense_bridge_qualify import atomic, http, worker


def paired(a, b):
    delta = [y-x for x, y in zip(a, b)]
    assert len(delta) == 24
    mean = statistics.mean(delta)
    se = statistics.stdev(delta) / math.sqrt(len(delta))
    return dict(mean_delta=mean, paired_se=se, passed=mean <= 2 * se,
                criterion='mean(candidate_i-compatible_i) <= 2 * paired SE; one-sided; unit=reply',
                per_reply_delta=delta)


def gsm_pair(a, b):
    assert a['complete'] and b['complete']
    aa, bb = a['results'], b['results']; assert len(aa) == len(bb) == 200
    assert [(r['id'], r['gold'], r['prompt_hash']) for r in aa] == [
        (r['id'], r['gold'], r['prompt_hash']) for r in bb]
    loss = sum(x['correct'] and not y['correct'] for x, y in zip(aa, bb))
    gain = sum(y['correct'] and not x['correct'] for x, y in zip(aa, bb))
    return dict(reference_correct=sum(r['correct'] for r in aa),
        candidate_correct=sum(r['correct'] for r in bb), b=loss, c=gain,
        delta_points=(gain-loss)/2, passed=gain-loss >= -2*math.sqrt(gain+loss),
        criterion='docs/204 GSM paired gate: c-b >= -2 sqrt(b+c)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--variants', nargs='+', default=['qr'],
        choices=['qr', 'device', 'shallow', 'qr_shallow', 'device_shallow', 'fix', 'native_bf16', 'native_fp32'])
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    assert 1 <= len(args.variants) <= 2 and len(set(args.variants)) == len(args.variants)
    plan = json.loads(args.plan.read_text()); source = Path(plan['factored_source'])
    for name, expected in plan['source_file_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == expected, name
    from lightning_i4_kl_probe import FrozenReplyKL, cpu_self_test
    cpu_self_test(args.out/'cpu-self-test')
    atomic(args.out/'cpu-self-test.json', dict(passed=True, cuda_initialized=False,
        checks=['actual processor serialization', 'FP64 analytical KL', 'exact replay zero',
                'disk transfer', 'forced tokens', 'prefill excluded from decode mean']))
    os.environ.update(HF_HOME='/work/hf-cache', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        HF_DATASETS_CACHE=plan['datasets_cache'], HF_MODULES_CACHE=str(args.out/'hf-modules'))
    (args.out/'hf-modules').mkdir(exist_ok=True)
    sys.path[:0] = plan['helper_paths']
    from lightning_factored_service_pair import prepare_client
    prepare_client(plan, args.out/'gsm-preflight', 'gsm')
    import gsm_endpoint
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(plan['model'], trust_remote_code=True)
    # Exactly the first 24 of KDA's 48 newly generated self-reply prompts.
    topics = ['the water cycle', 'photosynthesis', 'the history of navigation',
        'prime numbers', 'how computers represent text', 'the causes of ocean tides',
        'urban public transport', 'how libraries organize knowledge']
    styles = ['a detailed tutorial', 'a dialogue between a teacher and a student',
              'an explanation using concrete examples']
    prompts = [f'Write {style} about {topic}. Develop the explanation over at least 1500 words, '
               'with concrete examples and connected paragraphs. Begin the response now.'
               for style in styles for topic in topics]
    ids = []
    for prompt in prompts:
        row = tokenizer.apply_chat_template([dict(role='user', content=prompt)],
            tokenize=True, add_generation_prompt=True, return_dict=False)
        if hasattr(row, 'keys'): row = row['input_ids']
        if hasattr(row, 'tolist'): row = row.tolist()
        assert row and all(isinstance(x, int) for x in row)
        ids.append(row)
    atomic(args.out/'prompts.json', dict(prompts=prompts, input_ids=ids))
    shared = args.out/'kl'; shared.mkdir(exist_ok=True)
    launch = dict(plan.get('launch_variants', {}))
    if 'fix_arm' in plan:
        launch.setdefault('fix', plan['fix_arm'])
    ready = threading.Event(); abort = threading.Event()
    processor = FrozenReplyKL.to_str()

    def evaluate(label):
        def run(url, local, root, env):
            # docs/209 §9/§11 arms carry their switches as launch env, not a probe override.
            if label != 'original' and label not in launch:
                mode =('reference' if label in ('compatible', 'shallow')
                        else label.split('_', 1)[0])
                atomic(root/'probe/i4-config.json', dict(
                    commit=mode, shallow='shallow' in label))
            def kl_pass(arm):
                start = time.monotonic()
                body = dict(input_ids=ids, sampling_params=[dict(temperature=0,
                    max_new_tokens=257, ignore_eos=True, custom_params=dict(root=str(shared),
                    arm=arm, sample_id=i, samples=24, steps=256)) for i in range(24)],
                    custom_logit_processor=processor)
                replies = http(url, '/generate', body)
                assert len(replies) == 24
                rows = [json.loads((shared/f'{arm}-{i}.json').read_text()) for i in range(24)]
                for response, row in zip(replies, rows):
                    assert response['meta_info']['completion_tokens'] == 257
                    assert response['output_ids'] == row['tokens']
                    assert row['decode_steps'] == 256
                receipt = dict(complete=True, seconds=time.monotonic()-start,
                    token_hashes=[r['token_sha256'] for r in rows])
                if arm != 'original':
                    receipt.update(per_reply_mean_kl=[r['mean_kl'] for r in rows],
                        mean_kl=statistics.mean(r['mean_kl'] for r in rows),
                        prefill_kl=[r['prefill_kl'] for r in rows])
                atomic(root/(arm+'-replies.json'), replies)
                atomic(args.out/(arm+'-kl.json'), receipt)
                print('LTNF_KL_ARM', arm, json.dumps(receipt), flush=True)
                return receipt
            if label == 'original':
                try:
                    kl_pass('original')
                    replay = kl_pass('replay')
                    assert replay['mean_kl'] == 0 and max(replay['prefill_kl']) == 0, replay
                    ready.set()
                except BaseException:
                    abort.set(); ready.set(); raise
            else:
                if not ready.wait(900) or abort.is_set():
                    raise RuntimeError('original/replay did not complete exactly')
                kl_pass(label)
            gsm_endpoint.run(url, plan['model'], root/'gsm200.json')
        return run

    failures = {}
    local = dict(plan, batched_extend=True, instrument_forward=True)
    arms = [('original', 'original-off', 0), ('compatible', 'factored', 1)]
    arms += [(label, 'factored', 2+i) for i, label in enumerate(args.variants)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(arms)) as executor:
        fs = {executor.submit(worker, dict(local, **launch[label]) if label in launch else local,
              arm, gpu, args.out/label,
              micro=False, gsm=False, evaluate=evaluate(label)): label
              for label, arm, gpu in arms}
        for future in concurrent.futures.as_completed(fs):
            try: future.result()
            except Exception as e:
                failures[fs[future]] = repr(e)
                if fs[future] == 'original':
                    abort.set(); ready.set()
            atomic(args.out/'collection.json', dict(complete=False, failures=failures))
    if failures:
        raise RuntimeError(failures)
    receipts = {a: json.loads((args.out/(a+'-kl.json')).read_text())
                for a in ['original', 'replay', 'compatible', *args.variants]}
    assert all(r['token_hashes'] == receipts['original']['token_hashes'] for r in receipts.values())
    drift = {a: paired(receipts['compatible']['per_reply_mean_kl'], receipts[a]['per_reply_mean_kl'])
             for a in args.variants}
    gsm = {a: json.loads((args.out/a/'gsm200.json').read_text())
           for a in ['original', 'compatible', *args.variants]}
    gates = dict(compatible_vs_original=gsm_pair(gsm['original'], gsm['compatible']))
    for a in args.variants:
        gates[a+'_vs_original'] = gsm_pair(gsm['original'], gsm[a])
        gates[a+'_vs_compatible'] = gsm_pair(gsm['compatible'], gsm[a])
    admitted = {a: bool(drift[a]['passed'] and gates['compatible_vs_original']['passed']
                       and gates[a+'_vs_original']['passed'] and gates[a+'_vs_compatible']['passed'])
                for a in args.variants}
    result = dict(complete=True, protocol='KDA first 24 prompts x 256 decode predictions; FP64 full vocab',
        replay_exact_zero=True, kl=receipts, drift=drift, gsm=gates,
        variants=args.variants, variant_passed=admitted, passed=all(admitted.values()),
        scope='Implementation variations; timing from uninstrumented 4483630; no service window')
    atomic(args.out/'result.json', result)
    atomic(args.out/'collection.json', dict(complete=True, failures={}))
    print('LTNF_KL_RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
