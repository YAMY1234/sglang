"""Server-side D warm-up, run in the benchmark step BEFORE the AgentX client starts (stdlib only). Adapted from an earlier
warm-up for another model: the served model from MODEL, decode lines with or without a DP tag (an attn-DP-1
decode logs [... TP0] as rank 0), and the first decision taken as the first request this rank got (no model-specific DSA log line).
Kimi K3.1: the client's warm-up (max_tokens=1) never decodes, so the first decode stalled ~29 s in window 1.

Why: on Rubin D does no decode during the client's warmup, so its first decode stalls ~48 s inside the scored
window. This sends waves of synthetic PD requests through the frontend so every D DP rank decodes at every batch size the
window uses (k = 1..20 running per rank), before the client exists. Each wave mixes context lengths (1k / 8k / 32k, plus one 64k
and one 128k per 8 requests) so the long-context D paths past DSA top-k 2048 (GVR top-k, cap8 logits, overflow, width splits) run too. Prompts are random token ids from a fixed seed (no trace
content: nothing the client sends can hit them in the prefix cache). Disclosed as a server-side warm-up.
Gate (exit 3): every D DP rank printed >= 1 'Decode batch' line in /logs/*_decode_w*.out. Writes /logs/D-WARMUP.json: per request
start / ISL / first chunk / second chunk / end (streamed; the first token comes from D), and per
rank the time from its first decode decision ('DSA draft-decode replay metadata') to its first 'Decode batch' line = the
first-40-step cost (decode-gap: ~47 s cold, ~1 s warm; re-read at profiling start in the readout).
usage: d_warmup.py --dp R [--isls 1024,8192,32768,1024,8192,32768,65536,131072] [--osl 128] [--ks 1,2,4,8,12,16,20]
"""
import argparse, concurrent.futures as cf, glob, json, os, random, re, time, urllib.request

FRONTEND = 'http://' + os.environ['SRT_FRONTEND_HOST'] + ':' + os.environ['SRT_FRONTEND_PORT']
LOGS = os.environ.get('GSM_LOGS', '/logs')
MODEL = os.environ.get('MODEL') or os.environ.get('SERVED_MODEL_NAME') or 'moonshotai/Kimi-K3.1-Pro'
DP = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?: DP(\d+))?\b[^\]]*\] Decode batch')
DECIDE = re.compile(r'(?!)')  # no Kimi decode log line marks the first decision; decision_to_first_line_s stays empty


def one(ids, osl):
    """Streamed PD request; in Dynamo the first token comes from D, so first -> second chunk times show a step-1 stall."""
    body = json.dumps(dict(model=MODEL, prompt=ids, max_tokens=osl, stream=True)).encode()
    req = urllib.request.Request(FRONTEND + '/v1/completions', data=body, headers={'Content-Type': 'application/json'})
    t = time.time(); first = second = None; n = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode(errors='replace').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                break
            try:
                ch = json.loads(data).get('choices') or []
            except ValueError:
                continue
            if ch and ch[0].get('text'):
                n += 1; now = time.time() - t
                if first is None:
                    first = now
                elif second is None:
                    second = now
    return dict(start=round(t, 3), isl=len(ids), ttft=first, t2=second, e2e=round(time.time() - t, 3), chunks=n)


def decode_ranks():
    first, decide = {}, {}
    for f in glob.glob(os.path.join(LOGS, '*_decode_w*.out')):
        for line in open(f, errors='replace'):
            m = DP.match(line)
            if m and (m[2] or '0') not in first:
                first[m[2] or '0'] = m[1]
            m = DECIDE.match(line)
            if m and (m[2] or '0') not in decide:
                decide[m[2] or '0'] = m[1]
    return first, decide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dp', type=int, required=True); ap.add_argument('--isls', default='1024,8192,32768,1024,8192,32768,65536,131072')
    ap.add_argument('--osl', type=int, default=128); ap.add_argument('--ks', default='1,2,4,8,12,16,20')
    a = ap.parse_args()
    isls = [int(x) for x in a.isls.split(',')]
    rng = random.Random(20261118); out = dict(dp=a.dp, isls=isls, osl=a.osl, waves=[], start=time.strftime('%FT%TZ', time.gmtime()))
    t0 = time.time()
    for k in map(int, a.ks.split(',')):
        n = a.dp * k
        prompts = [[rng.randrange(1000, 100000) for _ in range(isls[i % len(isls)])] for i in range(n)]
        t = time.time()
        with cf.ThreadPoolExecutor(max_workers=n) as ex:
            res = list(ex.map(lambda p: one(p, a.osl), prompts))
        lat = sorted(r['e2e'] for r in res); gap = [r['t2'] - r['ttft'] for r in res if r['t2'] is not None and r['ttft'] is not None]
        w = dict(k=k, requests=n, prompt_tokens=sum(map(len, prompts)), wall_s=round(time.time() - t, 2), e2e_p50=round(lat[len(lat) // 2], 2),
                 e2e_max=round(lat[-1], 2), ttft_max=round(max((r['ttft'] or 0) for r in res), 2),
                 first_to_second_chunk_max=round(max(gap), 2) if gap else None, chunks_min=min(r['chunks'] for r in res))
        out['waves'].append(w); out.setdefault('requests', []).extend(dict(r, k=k) for r in res)
        print('D_WARMUP_WAVE ' + json.dumps(w), flush=True)
    time.sleep(5)   # let the last decode-log interval flush
    first, decide = decode_ranks()
    ts = lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S'))
    stall = {d: ts(first[d]) - ts(decide[d]) for d in first if d in decide}
    out.update(wall_s=round(time.time() - t0, 1), end=time.strftime('%FT%TZ', time.gmtime()), dp_ranks_with_decode=len(first),
               first_decode_line_per_dp=first, first_decision_per_dp=decide, decision_to_first_line_s=stall,
               decision_to_first_line_s_max=max(stall.values()) if stall else None)
    open(os.path.join(LOGS, 'D-WARMUP.json'), 'w').write(json.dumps(out, indent=1))
    ok = len(first) >= a.dp
    print(('D_WARMUP_OK ' if ok else 'D_WARMUP_FAIL ') + json.dumps({k: out[k] for k in ('wall_s', 'dp_ranks_with_decode', 'decision_to_first_line_s_max')}), flush=True)
    raise SystemExit(0 if ok else 3)


if __name__ == '__main__':
    main()
