"""Position-wise teacher-forced NLL gap between two served arms on the BASE arm's own generations.

    python -m benchmark.duet_eval.lmeval.forced_pairs --samples OUT/stock --task duet_aime25 --n 48 --max-cont 4096 \\
        --tokenizer /work/models/X --endpoints base=http://127.0.0.1:30303 duet=http://127.0.0.1:30301 --out DIR [--concurrency 32]

For each sampled (prompt, base response) pair the prompt is tokenized exactly as lm-eval sent it (the templated string in
`arguments`), the response is re-tokenized and truncated to --max-cont tokens, and both arms are made to emit exactly those
tokens (ForcedRawLogits custom logit processor, raw logits recorded before forcing).  Output: per-pair per-token NLL for
each arm, and the gap (other - base) by position bucket.  A gap that is flat from the first bucket points at the prompt
encoding (code + emitters); a gap that grows with position points at the decode-time state cuts."""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import random
import re
import statistics
import urllib.request
from pathlib import Path

from .common import first_string, response_of, sha1

BUCKETS = [(0, 64), (64, 256), (256, 1024), (1024, 4096), (4096, 1 << 30)]


def load_pairs(samples_root, task, n, seed, raw=False):
    """(prompt, response) pairs of the base arm.  raw: the response as the server returned it, thinking included (the sidecar's
    text_raw, joined on the sha1 of the templated prompt) -- lm-eval's resps of a thinking model hold the reply after the thinking."""
    rows = []
    for path in glob.glob(os.path.join(samples_root, "rep*", "**", f"samples_{task}_*.jsonl"), recursive=True):
        rep_dir = next(q for q in Path(path).parents if re.fullmatch(r"rep\d+", q.name))
        rep = int(rep_dir.name[3:])
        side = {}
        if raw:       # rep<k>/<kind>/sidecar.jsonl (one per process), or rep<k>/sidecar.jsonl (runs before 2026-10-02)
            kind_dir = Path(path).parents[1] if Path(path).parents[1] != rep_dir else rep_dir
            for sc in (kind_dir / "sidecar.jsonl", rep_dir / "sidecar.jsonl"):
                if sc.exists():
                    with open(sc, encoding="utf-8") as fh:
                        for line in fh:
                            r = json.loads(line)
                            side.setdefault(r["prompt_sha1"], r.get("text_raw", ""))
                    break
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                if d.get("filter", "none") not in ("none", "flexible-extract"):
                    continue
                prompt = first_string(d.get("arguments"))
                if raw:
                    text = side.get(sha1(prompt), "") if prompt else ""
                else:
                    text = response_of(d)
                if prompt and text:
                    rows.append({"doc_id": d["doc_id"], "rep": rep, "prompt": prompt, "response": text})
    random.Random(seed).shuffle(rows)
    return rows[:n]


def post(ep, payload, timeout=7200):
    req = urllib.request.Request(ep + "/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_arm(name, ep, prompts, targets, rec_root, concurrency):
    from benchmark.duet_eval.forced_logits import ForcedRawLogits, collect
    rec_dir = Path(rec_root) / f"{name}.raw"
    rec_dir.mkdir(parents=True, exist_ok=False)
    proc = ForcedRawLogits.to_str()

    def one(i):
        params = {"temperature": 0, "max_new_tokens": len(targets[i]), "ignore_eos": True,
                  "custom_params": {"targets": targets[i], "window": i, "record_dir": str(rec_dir), "hash_logits": False}}
        o = post(ep, {"input_ids": prompts[i], "sampling_params": params, "custom_logit_processor": proc, "rid": f"forced-{name}-{i}"})
        if o["meta_info"]["completion_tokens"] != len(targets[i]):
            raise ValueError(f"{name} pair {i}: forced continuation incomplete: {o['meta_info']}")

    with cf.ThreadPoolExecutor(concurrency) as ex:
        list(ex.map(one, range(len(targets))))
    ranks = sorted(rec_dir.glob("rank*.jsonl"))
    records = collect(rec_dir, len(ranks), targets, hash_logits=False, interleaved=concurrency > 1)
    return [r["nll"] for r in records]            # per pair: per-token NLL list


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True, help="base arm directory of an lm-eval exam (rep*/ inside)")
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--max-cont", type=int, default=4096)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--endpoints", nargs="+", required=True, help="name=url; the first is the base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--raw", action="store_true", help="force the full raw responses (thinking included) from the sidecars")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    pairs = load_pairs(a.samples, a.task, a.n, a.seed, raw=a.raw)
    prompts = [tok(p["prompt"], add_special_tokens=False)["input_ids"] for p in pairs]
    targets = [tok(p["response"], add_special_tokens=False)["input_ids"][: a.max_cont] for p in pairs]
    keep = [i for i, t in enumerate(targets) if len(t) >= 8]
    prompts, targets, pairs = [prompts[i] for i in keep], [targets[i] for i in keep], [pairs[i] for i in keep]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [e.split("=", 1) for e in a.endpoints]
    nll = {name: run_arm(name, url, prompts, targets, out, a.concurrency) for name, url in arms}
    base_name = arms[0][0]
    result = {"task": a.task, "pairs": len(pairs), "max_cont": a.max_cont, "arms": [n for n, _ in arms], "buckets": {}}
    for name, _ in arms[1:]:
        for lo, hi in BUCKETS:
            diffs, bvals, ovals = [], [], []
            for b, o in zip(nll[base_name], nll[name]):
                seg = slice(lo, min(hi, len(b)))
                if seg.stop > lo:
                    bvals += b[seg]
                    ovals += o[seg]
                    diffs += [x - y for x, y in zip(o[seg], b[seg])]
            if diffs:
                result["buckets"][f"{name}:{lo}-{hi}"] = {"tokens": len(diffs), "base_nll": statistics.mean(bvals),
                                                           f"{name}_nll": statistics.mean(ovals), "gap": statistics.mean(diffs)}
        per_pair = [statistics.mean(o) - statistics.mean(b) for b, o in zip(nll[base_name], nll[name])]
        result[f"{name}_per_pair_gap"] = per_pair
        print(f"FORCED-PAIRS {a.task} {name}-{base_name}: pairs {len(pairs)}, mean per-pair gap {statistics.mean(per_pair):+.5f} nats, "
              f"pairs with gap>0: {sum(g > 0 for g in per_pair)}/{len(per_pair)}", flush=True)
        for k, v in result["buckets"].items():
            if k.startswith(name):
                print(f"   pos {k.split(':')[1]:>12s}: tokens {v['tokens']:7d}  base {v['base_nll']:.4f}  {name} {v[name + '_nll']:.4f}  gap {v['gap']:+.5f}", flush=True)
    result["per_token"] = {name: nll[name] for name, _ in arms}
    result["pairs_meta"] = [{"doc_id": p["doc_id"], "rep": p["rep"], "cont_tokens": len(t)} for p, t in zip(pairs, targets)]
    (out / f"forced-pairs-{a.task}.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
