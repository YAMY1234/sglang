"""Accuracy gate client: GSM8K-200 cold/hit and needle first-token logprobs.

usage: python3 l6_gate.py --base-url URL --out DIR [--prompts PATH]
Prompts are pre-tokenized (gsm8k200.jsonl next to this file = the GSM8K records used). GSM8K: the first
200, a cold pass then a hit pass at concurrency 64. Needle: 20 x 32K and 10 x 64K, each cold, then two
hits, recording the first output token's top-20 logprobs (output only). Every request records its
cached_tokens so cold/hit coverage is checked by l6_gate_compare.py.
"""
import argparse
import os
import asyncio
import json
import re
from pathlib import Path

import aiohttp


def answer_number(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(nums[-1]) if nums else None


async def gen(session, url, rec, pass_name, max_new, top):
    ids = rec["input_ids"]
    body = {"input_ids": ids, "sampling_params": {"temperature": float(os.environ.get("GATE_TEMPERATURE", "0")), "max_new_tokens": max_new},
            "return_logprob": top}
    if top:
        body.update(logprob_start_len=len(ids), top_logprobs_num=20)
    async with session.post(url + "/generate", json=body) as r:
        text = await r.text()
        if r.status != 200:
            raise RuntimeError(f"{rec['id']} {pass_name}: HTTP {r.status} {text[:500]}")
    out = json.loads(text)
    meta = out["meta_info"]
    row = {"id": rec["id"], "pass": pass_name, "n_in": len(ids), "cached": meta.get("cached_tokens"),
           "text": out["text"]}
    if rec["id"].startswith("gsm8k"):
        row["correct"] = answer_number(out["text"]) == answer_number(rec["target"])
    else:
        row["correct"] = out["text"].strip() == rec["target"]
    if top:
        row["top"] = [[float(lp), int(t)] for lp, t, *_ in meta["output_top_logprobs"][0]]
    return row


async def run_pass(session, url, recs, pass_name, conc, max_new, top, sink):
    sem = asyncio.Semaphore(conc)

    async def one(rec):
        async with sem:
            row = await gen(session, url, rec, pass_name, max_new, top)
            sink.write(json.dumps(row) + "\n")
            return row

    rows = await asyncio.gather(*[one(r) for r in recs])
    ok = sum(r["correct"] for r in rows)
    print(f"PASS {pass_name}: {ok}/{len(rows)} correct, cached min/max "
          f"{min(r['cached'] or 0 for r in rows)}/{max(r['cached'] or 0 for r in rows)}", flush=True)


async def main(a):
    recs = [json.loads(l) for l in open(a.prompts)]
    gsm = [r for r in recs if r["id"].startswith("gsm8k")][:200]
    needle = [r for r in recs if r["id"].startswith("needle-32768")][:20] + \
             [r for r in recs if r["id"].startswith("needle-65536")][:10]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    conn = aiohttp.TCPConnector(force_close=True, limit=0)
    timeout = aiohttp.ClientTimeout(total=3600)
    with open(out / "gate.jsonl", "w", buffering=1) as sink:
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s:
            async with s.post(a.base_url + "/flush_cache") as r:
                print("FLUSH", r.status, flush=True)
            if a.warm:
                # One full 8K chunk first, so the decoder replay graphs are live for every pass.
                warm = {"id": "warm", "input_ids": [1000 + (i * 7919) % 100000 for i in range(8192)],
                        "target": ""}
                await gen(s, a.base_url, warm, "warm", 1, False)
                print("WARM done", flush=True)
            await run_pass(s, a.base_url, gsm, "gsm-cold", 64, 512, False, sink)
            if a.gsm_cold_only:
                print("GATE_DONE", flush=True)
                return
            await run_pass(s, a.base_url, gsm, "gsm-hit", 64, 512, False, sink)
            for rec in needle:
                for p in ("needle-cold", "needle-hit1", "needle-hit2"):
                    sink.write(json.dumps(await gen(s, a.base_url, rec, p, 24, True)) + "\n")
            print("GATE_DONE", flush=True)


p = argparse.ArgumentParser()
p.add_argument("--base-url", required=True)
p.add_argument("--out", required=True)
p.add_argument("--warm", action="store_true")
p.add_argument("--gsm-cold-only", action="store_true")
p.add_argument("--prompts", default=str(Path(__file__).with_name("gsm8k200.jsonl")))
asyncio.run(main(p.parse_args()))
