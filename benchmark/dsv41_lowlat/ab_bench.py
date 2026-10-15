"""Shared A/B client: any OpenAI-compatible server through the same /v1/completions requests.

Every arm gets identical token-id prompts (`prompt: [int, ...]`), streaming, greedy,
`ignore_eos`, so tokenizers and chat templates cannot differ. Logic follows L2's bench.py.

Points (contract "Shared client"):
  prefill  cold ISL 8K and 32K, OSL 1, c1 and c8           -> TTFT, input tok/s
  hit      64K prefix + 1K new and 7K + 1K, c1 and c16       -> TTFT, cached tokens
  decode   ISL 8K / OSL 1K, c1, c32, c64                     -> step, TPOT, tok/s per user and total
  prof     nsys windows: (a) one cold 8K prefill, (b) one c16 64K-hit wave, (c) 10 decode steps at c32;
           each window repeats its workload 3x in one capture and analysis keeps the last
Protocol: warm-up 2xC, timed 5xC (hit: 2 warm waves, 5 timed waves of C sent at once).
Prompts are seeded per point (crc32 of the point key), so every engine and arm order gets the
same token ids. Per-request rows: $OUT/bench/<phase>.jsonl; summaries: stdout SUMMARY + summary.jsonl.

Hit proof: per-request `usage.prompt_tokens_details.cached_tokens` (SGLang needs
`--enable-cache-report`) plus a /metrics counter diff per
point (SGLang needs `--enable-metrics`).
"""
import argparse
import asyncio
import json
import os
import random
import re
import time
import zlib

import aiohttp

VOCAB = (1000, 120000)
A = None  # parsed args
MODEL = None


def ids(n, key):
    rng = random.Random(zlib.crc32(key.encode()))
    return [rng.randrange(*VOCAB) for _ in range(n)]


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))] if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def r3(x, k=3):
    return None if x is None else round(x, k)


async def one(sess, prompt, max_new, tag, on_chunk=None, retry=1):
    try:
        return await _one(sess, prompt, max_new, tag, on_chunk)
    except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError) as e:
        # force_close makes this unexpected; one retry so a stale socket cannot kill an arm.
        if not retry:
            raise
        print(f"RETRY {tag} {type(e).__name__}", flush=True)
        return await one(sess, prompt, max_new, tag, on_chunk, retry - 1)


async def _one(sess, prompt, max_new, tag, on_chunk=None):
    body = {
        "model": MODEL, "prompt": prompt, "max_tokens": max_new, "temperature": 0.0,
        "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    chunks, usage, first_any = [], {}, None
    async with sess.post(A.base_url + "/v1/completions", json=body) as r:
        if r.status != 200:
            raise RuntimeError(f"{tag} HTTP {r.status}: {(await r.text())[:500]}")
        async for line in r.content:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices") or []:
                now = time.time()
                first_any = first_any or now
                # A chunk with text (or the final one) marks one engine step that emitted tokens.
                if c.get("text") or c.get("finish_reason"):
                    chunks.append(now)
                    if on_chunk:
                        on_chunk(len(chunks))
    t1 = time.time()
    n = usage.get("completion_tokens", 0)
    ptd = usage.get("prompt_tokens_details") or {}
    ttft = chunks[0] - t0 if chunks else None
    gaps = [b - a for a, b in zip(chunks, chunks[1:])]
    dec = t1 - chunks[0] if chunks else None
    return {
        "tag": tag, "t0": t0, "t1": t1, "ttft": ttft,
        "ttft_any": first_any - t0 if first_any else None, "e2e": t1 - t0,
        "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": ptd.get("cached_tokens"),
        "completion_tokens": n, "chunks": len(chunks),
        "tpot": dec / (n - 1) if n > 1 and dec else None,
        "tps_user": (n - 1) / dec if n > 1 and dec else None,
        "step_gaps": gaps,
    }


async def closed_loop(sess, prompts, max_new, conc, tag, on_chunk=None):
    sem = asyncio.Semaphore(conc)

    async def run(i, p):
        async with sem:
            return await one(sess, p, max_new, tag, on_chunk if i == 0 else None)

    t0 = time.time()
    res = await asyncio.gather(*(run(i, p) for i, p in enumerate(prompts)))
    return res, time.time() - t0


async def post(sess, path, body=None, ok=(200,)):
    try:
        async with sess.post(A.base_url + path, json=body) as r:
            txt = (await r.text()).strip()
            return r.status, txt[:300]
    except Exception as e:  # reported, never fatal
        return -1, repr(e)[:300]


async def flush(sess):
    if A.flush_path:
        st, txt = await post(sess, A.flush_path)
        print(f"FLUSH {A.flush_path} {st} {txt[:80]}", flush=True)
    else:
        # Servers without a cache-flush endpoint: unique
        # per-point prompts already make every cold request a miss.
        pass


METRIC_RE = re.compile(r"(cache|prefix|cached|spec_decode|accept)", re.I)


async def metrics(sess):
    """Counters from /metrics whose names mention cache/prefix/spec (summed over labels)."""
    try:
        async with sess.get(A.base_url + "/metrics") as r:
            if r.status != 200:
                return {}
            out = {}
            async for raw in r.content:
                line = raw.decode(errors="replace").strip()
                if not line or line[0] == "#":
                    continue
                name = line.split("{", 1)[0].split(" ", 1)[0]
                if not METRIC_RE.search(name) or not (name.endswith("_total") or "_sum" in name or "_count" in name):
                    continue
                try:
                    out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
            return out
    except Exception:
        return {}


def mdiff(m0, m1):
    return {k: round(m1[k] - m0.get(k, 0.0), 3) for k in m1 if m1[k] != m0.get(k, 0.0)}


def dump(phase, rows):
    with open(os.path.join(A.out, phase + ".jsonl"), "a") as f:
        for x in rows:
            f.write(json.dumps({**x, "phase": phase}) + "\n")


def summarize(phase, key, rows, wall, mdelta):
    gaps = [g for x in rows for g in x["step_gaps"]]
    ttfts = [x["ttft"] for x in rows]
    n_out = sum(x["completion_tokens"] for x in rows)
    s = {
        "phase": phase, **key, "label": A.label, "n": len(rows), "wall_s": r3(wall),
        "in_tok_s": round(sum(x["prompt_tokens"] or 0 for x in rows) / wall, 1),
        "out_tok_s": round(n_out / wall, 1),
        "ttft_mean_ms": r3(1e3 * mean(ttfts), 2), "ttft_p50_ms": r3(1e3 * pct(ttfts, 0.5), 2),
        "ttft_p90_ms": r3(1e3 * pct(ttfts, 0.9), 2), "ttft_max_ms": r3(1e3 * max(ttfts), 2),
        "step_p50_ms": r3(1e3 * pct(gaps, 0.5)) if gaps else None,
        "step_p10_ms": r3(1e3 * pct(gaps, 0.1)) if gaps else None,
        "step_p90_ms": r3(1e3 * pct(gaps, 0.9)) if gaps else None,
        "tpot_mean_ms": r3(1e3 * mean([x["tpot"] for x in rows])) if n_out > len(rows) else None,
        "tps_user_p50": r3(pct([x["tps_user"] for x in rows], 0.5), 1),
        "tps_user_p10": r3(pct([x["tps_user"] for x in rows], 0.1), 1),
        "tpot_p50_ms": r3(1e3 * pct([x["tpot"] for x in rows], 0.5)) if n_out > len(rows) else None,
        "tpot_p90_ms": r3(1e3 * pct([x["tpot"] for x in rows], 0.9)) if n_out > len(rows) else None,
        "tok_per_chunk": r3(n_out / max(1, sum(x["chunks"] for x in rows))),
        "prompt_tokens_p50": pct([x["prompt_tokens"] for x in rows], 0.5),
        "cached_p50": pct([x["cached_tokens"] for x in rows], 0.5),
        "cached_min": min((x["cached_tokens"] for x in rows if x["cached_tokens"] is not None), default=None),
        "metrics_delta": mdelta,
        "t0": min(x["t0"] for x in rows), "t1": max(x["t1"] for x in rows),
    }
    print("SUMMARY " + json.dumps(s), flush=True)
    with open(os.path.join(A.out, "summary.jsonl"), "a") as f:
        f.write(json.dumps(s) + "\n")
    return s


async def phase_cold(sess, phase, isl, osl, conc):
    """Unique prompts (no shared prefix): prefill or decode throughput."""
    await flush(sess)
    k = f"{phase}-{isl}-{osl}-c{conc}"
    await closed_loop(sess, [ids(isl, f"{k}-w{i}") for i in range(2 * conc)], osl, conc, k + "-warm")
    m0 = await metrics(sess)
    rows, wall = await closed_loop(sess, [ids(isl, f"{k}-t{i}") for i in range(5 * conc)], osl, conc, k)
    dump(phase, rows)
    summarize(phase, {"isl": isl, "osl": osl, "conc": conc}, rows, wall, mdiff(m0, await metrics(sess)))


async def phase_hit(sess, phase, prefix_len, new_len, conc, osl):
    """Shared prefix warmed once; each request = prefix + unique new tokens, waves of conc at once."""
    prefix = ids(prefix_len, f"prefix-{prefix_len}")
    k = f"{phase}-{prefix_len}+{new_len}-c{conc}"
    prime = await one(sess, prefix + ids(new_len, k + "-prime"), osl, k + "-prime")
    print(f"PRIME {k} prompt={prime['prompt_tokens']} cached={prime['cached_tokens']}", flush=True)
    for kind, waves in (("warm", 2), ("timed", 5)):
        m0 = await metrics(sess)
        rows, wall = [], 0.0
        for w in range(waves):
            r, t = await closed_loop(
                sess, [prefix + ids(new_len, f"{k}-{kind}{w}-{i}") for i in range(conc)], osl, conc,
                f"{k}-{kind}")
            rows += r
            wall += t
        if kind == "timed":
            dump(phase, rows)
            summarize(phase, {"prefix": prefix_len, "new": new_len, "osl": osl, "conc": conc},
                      rows, wall, mdiff(m0, await metrics(sess)))


async def prof_hook(sess, which, steps=None):
    path = A.prof_start_path if which == "start" else A.prof_stop_path
    body = A.prof_start_body if which == "start" else A.prof_stop_body
    if not path:
        return
    body = json.loads(body) if body else None
    for k, v in list((body or {}).items()):  # "{steps}" -> the window's step count, or dropped
        if v == "{steps}":
            if steps:
                body[k] = steps
            else:
                del body[k]
    st, txt = await post(sess, path, body)
    print(f"PROF_{which.upper()} {time.time():.3f} {path} {st} {txt[:120]}", flush=True)


async def phase_prof(sess, windows):
    """nsys capture windows; the server runs under `nsys --capture-range=cudaProfilerApi`."""
    for w in windows:
        print(f"PROF_WINDOW_BEGIN {w} {time.time():.3f}", flush=True)
        # Each window repeats its workload --prof-repeat times inside one capture, 1 s apart;
        # analysis keeps the last repeat, clear of nsys's capture-start cost on the first.
        if w == "prefill":  # (a) one cold 8K prefill forward (8192 = one chunk), repeated
            await flush(sess)
            await one(sess, ids(8192, "prof-prefill-warm"), 1, "prof-prefill-warm")
            await prof_hook(sess, "start", A.prof_repeat)
            rows = []
            for r in range(A.prof_repeat):
                rows.append(await one(sess, ids(8192, f"prof-prefill-{r}"), 1, f"prof-prefill-r{r}"))
                await asyncio.sleep(1.0)
            await prof_hook(sess, "stop")
            dump("prof", rows)
        elif w == "hit":  # (b) one c16 wave of 64K + 1K hits, repeated
            await flush(sess)
            prefix = ids(65536, "prefix-65536")
            await one(sess, prefix + ids(1024, "prof-hit-prime"), 1, "prof-hit-prime")
            await closed_loop(sess, [prefix + ids(1024, f"prof-hit-w{i}") for i in range(16)], 1, 16,
                              "prof-hit-warm")
            await prof_hook(sess, "start", A.prof_hit_steps)
            rows = []
            for r in range(A.prof_repeat):
                rr, _ = await closed_loop(
                    sess, [prefix + ids(1024, f"prof-hit-{r}-{i}") for i in range(16)], 1, 16, f"prof-hit-r{r}")
                rows += rr
                await asyncio.sleep(1.0)
            await prof_hook(sess, "stop")
            dump("prof", rows)
        elif w == "decode":  # (c) steady decode steps at c32, all 32 requests past prefill
            await flush(sess)
            conc, steps = A.prof_decode_conc, A.prof_decode_steps * A.prof_repeat
            started, n0, stop_at = [0], [0], [None]
            ready, enough = asyncio.Event(), asyncio.Event()

            def cb(i):
                def f(n):
                    if n == 1:
                        started[0] += 1
                        if started[0] == conc:
                            ready.set()
                    if i == 0:  # request 0 streams one chunk per decode step
                        n0[0] = n
                        if stop_at[0] is not None and n >= stop_at[0]:
                            enough.set()
                return f

            load = asyncio.ensure_future(asyncio.gather(
                *(one(sess, ids(A.decode_isl, f"prof-decode-{i}"), A.decode_osl, "prof-decode", cb(i)) for i in range(conc))))
            await ready.wait()
            await asyncio.sleep(1.0)
            stop_at[0] = n0[0] + steps
            await prof_hook(sess, "start", steps)
            await asyncio.wait_for(enough.wait(), 120)
            await prof_hook(sess, "stop")
            print(f"PROF_DECODE_CHUNKS start={stop_at[0] - steps} stop={n0[0]}", flush=True)
            rows = await load
            dump("prof", rows)
        print(f"PROF_WINDOW_END {w} {time.time():.3f}", flush=True)
        await asyncio.sleep(3)



async def main():
    global A, MODEL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", help="comma list of phases: prefill,hit,decode,prof (run in this order)")
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL"))
    ap.add_argument("--out", default=os.path.join(os.environ.get("OUT", "."), "bench"))
    ap.add_argument("--label", default=os.environ.get("ARM", ""), help="arm label stored in every summary")
    ap.add_argument("--reverse", action="store_true", help="reverse the order of points inside each phase")
    ap.add_argument("--prefill-isl", default="8192,32768")
    ap.add_argument("--prefill-conc", default="1,8")
    ap.add_argument("--hit", default="65536+1024,7168+1024", help="prefix+new list")
    ap.add_argument("--hit-conc", default="1,16")
    ap.add_argument("--hit-osl", type=int, default=8)
    ap.add_argument("--decode-isl", type=int, default=8192)
    ap.add_argument("--decode-osl", type=int, default=1024)
    ap.add_argument("--decode-conc", default="1,32,64")
    ap.add_argument("--flush-path", default="/flush_cache",
                    help="POST to drop the prefix cache before cold phases (or '' to skip)")
    ap.add_argument("--prof-windows", default="prefill,hit,decode")
    ap.add_argument("--prof-start-path", default="/start_profile")
    ap.add_argument("--prof-start-body", default='{"activities": ["CUDA_PROFILER"]}',
                    help='JSON; a value "{steps}" becomes the window step count (dropped when 0)')
    ap.add_argument("--prof-stop-path", default="/stop_profile")
    ap.add_argument("--prof-stop-body", default="")
    ap.add_argument("--prof-decode-steps", type=int, default=10,
                    help="decode steps analyzed; the capture holds prof-repeat times as many")
    ap.add_argument("--prof-repeat", type=int, default=3)
    ap.add_argument("--prof-decode-conc", type=int, default=32)  # set 1 for a bs=1 window
    ap.add_argument("--prof-hit-steps", type=int, default=0)
    A = ap.parse_args()
    assert A.base_url, "--base-url or BASE_URL"
    os.makedirs(A.out, exist_ok=True)
    tmo = aiohttp.ClientTimeout(total=3600)
    # force_close: pooled keep-alive sockets race uvicorn's 5 s idle close (L2, 10-08).
    async with aiohttp.ClientSession(timeout=tmo, connector=aiohttp.TCPConnector(limit=0, force_close=True)) as sess:
        async with sess.get(A.base_url + "/v1/models") as r:
            MODEL = (await r.json())["data"][0]["id"]
        print(f"CLIENT model={MODEL} base={A.base_url} out={A.out} label={A.label}", flush=True)
        rev = (lambda xs: xs[::-1]) if A.reverse else (lambda xs: xs)
        csv = lambda s: [x for x in s.split(",") if x]
        for ph in csv(A.plan):
            print(f"PHASE_START {ph} {time.time():.3f}", flush=True)
            if ph == "prefill":
                for isl in rev([int(x) for x in csv(A.prefill_isl)]):
                    for c in rev([int(x) for x in csv(A.prefill_conc)]):
                        await phase_cold(sess, "prefill", isl, 1, c)
            elif ph == "hit":
                for pn in rev(csv(A.hit)):
                    p, n = (int(x) for x in pn.split("+"))
                    await flush(sess)
                    for c in rev([int(x) for x in csv(A.hit_conc)]):
                        await phase_hit(sess, "hit", p, n, c, A.hit_osl)
            elif ph == "decode":
                for c in rev([int(x) for x in csv(A.decode_conc)]):
                    await phase_cold(sess, "decode", A.decode_isl, A.decode_osl, c)
            elif ph == "prof":
                await phase_prof(sess, csv(A.prof_windows))
            else:
                raise SystemExit(f"unknown phase {ph}")
            print(f"PHASE_END {ph} {time.time():.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
