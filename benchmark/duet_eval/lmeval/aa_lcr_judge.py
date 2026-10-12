"""AA-LCR grading: the official v1.1 equality check (the dataset card's system and user prompts, verbatim) of every reply against
the official answer, on a local judge model served by SGLang.  The official checker is an external API model; it is not used
(no exam data leaves the cluster), which is the one deviation from the official protocol.  Both arms are judged by the same
judge with the same settings, and the judge never sees the arm.  A reply the judge cannot be read on is recorded as "none" and
scored incorrect; an empty reply is incorrect without a call.

    python -m benchmark.duet_eval.lmeval.aa_lcr_judge --endpoints http://127.0.0.1:30700,... --tag fn --runs OUT/stock OUT/duet \\
        [--chat-kwargs '{"enable_thinking": true, "reasoning_effort": "medium"}'] [--sampling '{"temperature": 1.0, ...}']
        -> <run>/rep*/aa_lcr/aa_lcr_judge.<tag>.jsonl, one record per reply (cached by the reply's sha1)
report_lmeval.py scores aa_lcr/judge-<tag> from these files (cell aa_lcr = the judge named by its --aa-lcr-judge)."""
import argparse
import asyncio
import glob
import json
import os
import re

from .common import SAMPLE_RE, response_of, sha1

SYSTEM = """Decide whether the CANDIDATE ANSWER is correct or incorrect against the OFFICIAL ANSWER.
Note the following points when assessing correctness:

- Numbers should still match when they are the same value written differently, e.g., a
  percentage, a count of percentage points, and the equivalent decimal fraction are the same
  value: 0.675, "67.5%" and "67.5 percentage points" all match. So do different scales
  (thousand, million, bn) and different notations (thousands separators, currency symbols,
  LaTeX markup, and numbers written as words).
- Where the question asks for a particular format (e.g., a percentage, a number of decimal
  places, a unit, a rounding, or an ordering) the CANDIDATE ANSWER must meet it. If the
  question asks for no particular format, accept any equivalent form.
- In cases where the question asks for an ordered list, a title, honorific or article added
  to an entry in the CANDIDATE ANSWER can change where that entry sorts. Accept the ordering
  if it is correct either with those additions or without them.
- Grade the value the CANDIDATE ANSWER finally commits to, and it must commit to one. Values
  reached while working, and alternatives it considers and sets aside, do not count. If it
  offers several values without selecting one, it is incorrect even if one of them is right.
  Hedging is fine as long as one clearly definitive answer is given."""
USER = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: START QUESTION {question}

END QUESTION

The OFFICIAL ANSWER: {official_answer}

END OFFICIAL ANSWER

BEGIN CANDIDATE ANSWER TO ASSESS

{candidate_answer}

END CANDIDATE ANSWER TO ASSESS

Reply as JSON, with a verdict of CORRECT or INCORRECT."""
VERDICT_JSON = re.compile(r'"verdict"\s*:\s*"\s*(CORRECT|INCORRECT)\s*"', re.I)
VERDICT_WORD = re.compile(r"\b(INCORRECT|CORRECT)\b")
JUDGE_VERSION = 1


def parse_verdict(text):
    """The verdict after any thinking: the JSON field, else the last bare CORRECT / INCORRECT, else none."""
    text = (text or "").rsplit("</think>", 1)[-1]
    m = VERDICT_JSON.findall(text)
    if m:
        return m[-1].upper()
    m = VERDICT_WORD.findall(text.upper())
    return m[-1] if m else "none"


def judge_file(rep_dir, tag):
    return os.path.join(rep_dir, "aa_lcr", f"aa_lcr_judge.{tag}.jsonl")


def load_judged(rep_dir, tag):
    """(doc_id, reply sha1) -> record."""
    out = {}
    path = judge_file(rep_dir, tag)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                r = json.loads(raw)
                if r.get("v") == JUDGE_VERSION:
                    out[(str(r["doc_id"]), r["resp_sha1"])] = r
    return out


def judge_tags(rep_dir):
    return sorted(os.path.basename(p)[len("aa_lcr_judge."):-len(".jsonl")] for p in glob.glob(judge_file(rep_dir, "*")))


def samples(rep_dir):
    for path in sorted(glob.glob(os.path.join(rep_dir, "aa_lcr", "**", "samples_*.jsonl"), recursive=True)):
        m = SAMPLE_RE.search(os.path.basename(path))
        if m and m.group("task") == "duet_aa_lcr":
            with open(path, encoding="utf-8") as fh:
                for raw in fh:
                    yield json.loads(raw)


async def judge(args):
    import aiohttp

    endpoints = args.endpoints.split(",")
    chat_kwargs = json.loads(args.chat_kwargs) if args.chat_kwargs else None
    sampling = json.loads(args.sampling) if args.sampling else {"temperature": 0.0}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout)) as session:
        models = {}
        for ep in endpoints:
            async with session.get(ep + "/v1/models") as r:
                models[ep] = (await r.json())["data"][0]["id"]
        todo = []
        for run in args.runs:
            for rep_dir in sorted(glob.glob(os.path.join(run, "rep[0-9]*"))):
                done, seen = load_judged(rep_dir, args.tag), set()
                for line in samples(rep_dir):
                    key = (str(line["doc_id"]), sha1(response_of(line)))
                    if key not in done and key not in seen:
                        seen.add(key)
                        todo.append((rep_dir, key, line))
        print(f"{len(todo)} replies to judge ({', '.join(sorted(set(models.values())))}; tag {args.tag})", flush=True)
        sem = asyncio.Semaphore(args.concurrency)
        files = {}

        async def one(i, rep_dir, key, line):
            reply = response_of(line).strip()
            if not reply:
                rec = {"verdict": "INCORRECT", "empty": True, "raw": ""}
            else:
                ep = endpoints[i % len(endpoints)]
                doc = line["doc"]
                user = USER.format(question=doc["question"], official_answer=doc["answer"], candidate_answer=reply)
                body = {"model": models[ep], "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                        "max_tokens": args.max_tokens, **sampling}
                if chat_kwargs:
                    body["chat_template_kwargs"] = chat_kwargs
                raw, finish = None, None
                async with sem:
                    for attempt in range(5):
                        try:
                            async with session.post(ep + "/v1/chat/completions", json=body) as r:
                                o = await r.json()
                            raw, finish = o["choices"][0]["message"]["content"] or "", o["choices"][0].get("finish_reason")
                            break
                        except Exception as e:                                  # noqa: BLE001 -- retried, then left unjudged
                            if attempt == 4:
                                print(f"judge failed on {rep_dir} {key}: {e!r}", flush=True)
                                return
                            await asyncio.sleep(2 * (attempt + 1))
                rec = {"verdict": parse_verdict(raw), "finish_reason": finish, "raw": raw.rsplit("</think>", 1)[-1][-400:]}
            rec.update(v=JUDGE_VERSION, tag=args.tag, doc_id=key[0], question_id=line["doc"].get("question_id"), resp_sha1=key[1],
                       judge=models.get(endpoints[i % len(endpoints)], ""))
            fh = files.setdefault(rep_dir, open(judge_file(rep_dir, args.tag), "a", encoding="utf-8"))
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

        await asyncio.gather(*(one(i, *t) for i, t in enumerate(todo)))
        for fh in files.values():
            fh.close()
    for run in args.runs:
        recs = [r for rep_dir in sorted(glob.glob(os.path.join(run, "rep[0-9]*"))) for r in load_judged(rep_dir, args.tag).values()]
        n = len(recs)
        if n:
            print(f"{run}: {n} replies, correct {100 * sum(r['verdict'] == 'CORRECT' for r in recs) / n:.1f}%, "
                  f"unreadable {sum(r['verdict'] == 'none' for r in recs)}, empty {sum(bool(r.get('empty')) for r in recs)}", flush=True)
    print("AA-LCR JUDGE DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoints", required=True, help="comma-separated SGLang servers of the judge model")
    ap.add_argument("--tag", required=True, help="judge name in the output file (e.g. fn, kimi)")
    ap.add_argument("--runs", nargs="+", required=True, help="arm directories (<run>/rep<k>/aa_lcr/samples_*.jsonl)")
    ap.add_argument("--chat-kwargs", default="", help="chat_template_kwargs of the judge, json")
    ap.add_argument("--sampling", default="", help="sampling fields of the request, json (default greedy)")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=7200)
    asyncio.run(judge(ap.parse_args()))


if __name__ == "__main__":
    main()
