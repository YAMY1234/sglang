"""First-token distribution of several served arms on LiveCodeBench prompts, under prompt variants.

The LCB loss of the Kimi DUET set is a mode switch at the first generated token ("Alright, let's see" -> "To solve this
problem").  This probe asks every arm for the top-k log-probabilities of the FIRST generated token only (max_new_tokens 1,
no sampling involved) for the official prompt and for variants that remove one ingredient at a time, and reports per arm
and variant the probability mass of the think-aloud openers, of the direct-answer openers, and the first-token KL to the
base arm (over the union of the arms' top-k, renormalised).

    python -m benchmark.duet_eval.lmeval.probe_opener --data /src/results/hfcache/lcb/lcb_v6_2408_2505.jsonl \\
        --tokenizer /work/models/Kimi-Linear-48B-A3B-Instruct --endpoints base=http://... noprune=http://... --n 150 --out DIR
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import statistics
import urllib.request

from benchmark.duet_eval.lmeval.lcb_utils import SYSTEM_MESSAGE, doc_to_text

THINK = ("Al", "Okay", "Let", "Hmm", "So", "First")                # Kimi tokenizes "Alright" as "Al" + "right"
DIRECT = ("To", "I", "We", "Looking", "Here", "The", "```")


def variants(doc):
    text = doc_to_text(doc)
    no_trailer = text.replace("### Answer: (use the provided format with backticks)\n\n", "")
    question_only = f"{doc['question_content']}\n\nWrite a Python solution."
    return {
        "official": [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": text}],
        "nosys": [{"role": "user", "content": SYSTEM_MESSAGE + "\n\n" + text}],
        "notrailer": [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": no_trailer}],
        "plain": [{"role": "user", "content": question_only}],
    }


def post(ep, payload, timeout=600):
    req = urllib.request.Request(ep + "/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def first_token_dist(ep, ids, k, rid):
    params = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 1}
    o = post(ep, {"input_ids": ids, "sampling_params": params, "return_logprob": True, "top_logprobs_num": k,
                  "return_text_in_logprobs": True, "rid": rid})
    top = o["meta_info"]["output_top_logprobs"][0]                     # [[logprob, token_id, text], ...]
    return {int(t[1]): (float(t[0]), t[2]) for t in top}


def mass(dist, prefixes):
    return sum(math.exp(lp) for lp, text in dist.values() if text.strip().startswith(prefixes))


def kl_topk(p, q):
    """KL(p || q) over the union of both top-k supports, each renormalised; missing tokens get the smallest kept log-prob - 2."""
    keys = set(p) | set(q)
    flo_p = min(lp for lp, _ in p.values()) - 2.0
    flo_q = min(lp for lp, _ in q.values()) - 2.0
    lp = {t: (p[t][0] if t in p else flo_p) for t in keys}
    lq = {t: (q[t][0] if t in q else flo_q) for t in keys}
    zp = math.log(sum(math.exp(v) for v in lp.values()))
    zq = math.log(sum(math.exp(v) for v in lq.values()))
    return sum(math.exp(lp[t] - zp) * ((lp[t] - zp) - (lq[t] - zq)) for t in keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--endpoints", nargs="+", required=True, help="name=url; the first is the base")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--reference-samples", default="", help="an lm-eval samples_duet_lcb_*.jsonl: assert the official variant renders the same prompt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    docs = [json.loads(l) for l in open(a.data, encoding="utf-8")][: a.n]
    arms = [e.split("=", 1) for e in a.endpoints]
    ref = {}
    if a.reference_samples:
        for l in open(a.reference_samples, encoding="utf-8"):
            d = json.loads(l)
            ref[d["doc"]["question_id"]] = d["arguments"]["gen_args_0"]["arg_0"]
    prompts = {}                                                        # (doc index, variant) -> token ids
    for i, d in enumerate(docs):
        for v, msgs in variants(d).items():
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            if v == "official" and d["question_id"] in ref and ref[d["question_id"]] != s:
                raise ValueError(f"official variant differs from the exam prompt for {d['question_id']}")
            prompts[(i, v)] = tok(s, add_special_tokens=False)["input_ids"]
    dists = {}

    def one(key):
        name, url, i, v = key
        d = first_token_dist(url, prompts[(i, v)], a.topk, rid=f"{os.getpid()}-{name}-{i}-{v}")
        # token text from our tokenizer, not the server's logprob text (that text did not match the opener prefixes: the
        # 2026-10-01 table printed P(think) 0.002 for a base whose stored "Al" mass was 0.93)
        dists[(name, i, v)] = {tid: (lp, tok.decode([tid])) for tid, (lp, _) in d.items()}

    keys = [(name, url, i, v) for name, url in arms for (i, v) in prompts]
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(one, keys))
    base = arms[0][0]
    os.makedirs(a.out, exist_ok=True)
    report = {"n": len(docs), "topk": a.topk, "arms": [n for n, _ in arms], "rows": []}
    print(f"{'variant':10s} {'arm':9s} {'P(think)':>9s} {'P(direct)':>10s} {'P(To)':>7s} {'P(Alright)':>11s} {'KL to base':>11s} {'SE':>7s}")
    for v in variants(docs[0]):
        for name, _ in arms:
            th = [mass(dists[(name, i, v)], THINK) for i in range(len(docs))]
            di = [mass(dists[(name, i, v)], DIRECT) for i in range(len(docs))]
            to = [mass(dists[(name, i, v)], ("To",)) for i in range(len(docs))]
            al = [mass(dists[(name, i, v)], ("Al",)) for i in range(len(docs))]
            kl = [kl_topk(dists[(base, i, v)], dists[(name, i, v)]) for i in range(len(docs))]
            row = {"variant": v, "arm": name, "p_think": statistics.mean(th), "p_direct": statistics.mean(di), "p_to": statistics.mean(to),
                   "p_alright": statistics.mean(al), "kl": statistics.mean(kl), "kl_se": statistics.stdev(kl) / math.sqrt(len(kl))}
            report["rows"].append(row)
            print(f"{v:10s} {name:9s} {row['p_think']:9.3f} {row['p_direct']:10.3f} {row['p_to']:7.3f} {row['p_alright']:11.3f} {row['kl']:11.4f} {row['kl_se']:7.4f}")
    report["dists"] = {f"{name}|{i}|{v}": d for (name, i, v), d in dists.items()}
    json.dump(report, open(os.path.join(a.out, "probe_opener.json"), "w"))
    print("PROBE OPENER DONE")


if __name__ == "__main__":
    main()
