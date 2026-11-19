"""Load client for one PD-prefill server without a decode side (see README.md).

Each request: /generate, input_ids = shared prefix (one of --groups, fixed by --seed) + fresh random tokens,
max_new_tokens 1, the fake bootstrap host 2.2.2.2 and its own room, so the server uses its fake KV sender.
Modes:
  warm     each prefix once (new tokens 0), --concurrency at a time
  probe    --requests requests one at a time; prints each meta_info
  closed   --concurrency lanes back to back for --lead + --window s, then waits for the requests in flight
  poisson  arrivals at --rate req/s (exponential gaps, seeded) for --lead + --window s
Writes one JSON line per request (send/done epoch s, latency, input, cached, ok) to --out and a summary to --out.sum.json:
new tok/s over requests completed inside [lead, lead+window] (closed) and the latency p50/p90 of requests sent inside it.
"""
import argparse
import asyncio
import json
import random
import time

import aiohttp
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--mode", choices=["warm", "probe", "closed", "poisson"], required=True)
ap.add_argument("--groups", type=int, default=0, help="0: no shared prefix")
ap.add_argument("--prefix", type=int, default=0)
ap.add_argument("--new", type=int, default=0)
ap.add_argument("--concurrency", type=int, default=1)
ap.add_argument("--rate", type=float, default=0.0)
ap.add_argument("--requests", type=int, default=2)
ap.add_argument("--lead", type=float, default=45.0)
ap.add_argument("--window", type=float, default=300.0)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--max-inflight", type=int, default=1024)
ap.add_argument("--out", required=True)
a = ap.parse_args()

LO, HI = 1000, 150000
prefixes = [np.random.default_rng([a.seed, g]).integers(LO, HI, a.prefix).tolist() for g in range(a.groups)]
pick = random.Random(a.seed * 7919 + 17)
room0 = random.Random().randrange(1 << 40)
counter = 0


def body(new_tokens, g, k):
    fresh = np.random.default_rng([a.seed, 99, k]).integers(LO, HI, new_tokens).tolist()
    ids = (prefixes[g] if g is not None else []) + fresh
    return len(ids), json.dumps({"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                                 "bootstrap_host": "2.2.2.2", "bootstrap_room": room0 + k}).encode()


async def one(s, rec, new_tokens, g, show=False):
    global counter
    counter += 1  # on the event loop: one room and one fresh-token seed per request
    n, data = await asyncio.to_thread(body, new_tokens, g, counter)
    t = time.time()
    r = {"send": t, "input": n, "group": g}
    try:
        async with s.post(a.url + "/generate", data=data, headers={"Content-Type": "application/json"}) as resp:
            j = await resp.json(content_type=None)
        ok = resp.status == 200 and "meta_info" in j
        r.update(ok=ok, cached=int(j["meta_info"].get("cached_tokens", 0)) if ok else 0)
        if not ok:
            r["err"] = str(j)[:300]
        if show:
            print(json.dumps({"status": resp.status, "meta_info": j.get("meta_info"), "text": j.get("text")})[:800], flush=True)
    except Exception as e:  # noqa: BLE001
        r.update(ok=False, cached=0, err=repr(e)[:300])
    r["done"] = time.time()
    r["lat"] = r["done"] - t
    rec.append(r)
    if not r["ok"] and sum(not x["ok"] for x in rec) <= 3:
        print("request failed:", r.get("err"), flush=True)


def group():
    return pick.randrange(a.groups) if a.groups else None


async def main():
    rec, t0 = [], time.time()
    timeout = aiohttp.ClientTimeout(total=3600)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as s:
        if a.mode == "warm":
            sem = asyncio.Semaphore(a.concurrency)

            async def w(g):
                async with sem:
                    await one(s, rec, 0, g)
            await asyncio.gather(*(w(g) for g in range(a.groups)))
        elif a.mode == "probe":
            for _ in range(a.requests):
                await one(s, rec, a.new, group(), show=True)
        elif a.mode == "closed":
            stop = t0 + a.lead + a.window

            async def lane():
                while time.time() < stop:
                    await one(s, rec, a.new, group())
            await asyncio.gather(*(lane() for _ in range(a.concurrency)))
        else:
            rng, tasks, t = random.Random(a.seed * 104729 + 3), [], t0
            stop = t0 + a.lead + a.window
            while True:
                t += rng.expovariate(a.rate)
                if t >= stop:
                    break
                await asyncio.sleep(max(0.0, t - time.time()))
                tasks = [x for x in tasks if not x.done()]
                if len(tasks) >= a.max_inflight:
                    rec.append({"send": time.time(), "ok": False, "err": "max_inflight", "input": 0, "cached": 0,
                                "done": time.time(), "lat": 0.0, "group": None})
                    continue
                tasks.append(asyncio.create_task(one(s, rec, a.new, group())))
            await asyncio.gather(*tasks)
    with open(a.out, "w") as f:
        for r in rec:
            f.write(json.dumps(r) + "\n")
    w0, w1 = t0 + a.lead, t0 + a.lead + a.window
    done_in = [r for r in rec if r["ok"] and w0 <= r["done"] < w1]
    sent_in = [r for r in rec if r["ok"] and w0 <= r["send"] < w1]
    lat = sorted(r["lat"] for r in sent_in) or [0.0]
    inp = sum(r["input"] for r in done_in)
    cached = sum(r["cached"] for r in done_in)
    summ = {"mode": a.mode, "requests": len(rec), "failed": sum(not r["ok"] for r in rec),
            "window_s": a.window, "done_in_window": len(done_in), "sent_in_window": len(sent_in),
            "new_tok_s": (inp - cached) / a.window if a.mode in ("closed", "poisson") else None,
            "hit": cached / inp if inp else None, "lat_p50": lat[len(lat) // 2], "lat_p90": lat[int(len(lat) * 0.9)],
            "args": vars(a), "t0": t0}
    json.dump(summ, open(a.out + ".sum.json", "w"))
    print(json.dumps({k: v for k, v in summ.items() if k != "args"}), flush=True)


asyncio.run(main())
