"""LongBench v2 for the paired DUET exams: the official zero-shot prompt, middle truncation and answer extraction.

Source: github.com/THUDM/LongBench at commit 2e00731f8d0bff23dc4325161044d0ed8af94c1e (MIT licence): pred.py (query_llm's
truncation, get_pred's prompt and extract_answer), prompts/0shot.txt (TEMPLATE_0SHOT below, byte for byte: sha256 68a16225...),
config/model2maxlen.json (120000 tokens for every 128K-context model: MAX_LEN), result.py (overall, easy / hard, short / medium
/ long).  Data: the 503 questions of the Hub dataset THUDM/LongBench-v2 (data.json at revision 2b48e494f2c7, prepare_data.sh),
in the dataset's order.

Prompt, as get_pred builds it without CoT, RAG or --no_context: the template with $DOC$ = context.strip(), $Q$ = question.strip(),
$C_A$ .. $C_D$ = the choices .strip(), replaced in that order (str.replace, as upstream); then query_llm's truncation with the
evaluated model's own tokenizer: ids = tokenizer.encode(prompt); if len(ids) > MAX_LEN, keep the first MAX_LEN // 2 and the last
MAX_LEN // 2 ids and decode them with skip_special_tokens=True.  The prompt is the single user message of the chat (no system
message), as upstream sends it.  Extraction (extract_answer): '*' removed, then the first "The correct answer is (X)", else the
first "The correct answer is X", X in A-D; applied to the reply after the thinking block (model.py strips it).

The prompts depend on the tokenizer, so they are built once per tokenizer (build_prompts, ~minutes of CPU for the 503 documents)
into <DUET_DATA_ROOT>/longbench-v2/prompts/<tokenizer name>-<sha1(path)[:8]>-<MAX_LEN>.jsonl and read by every later run (both
arms of a pairing read the same file); run.py passes the tokenizer path through lm-eval's --metadata.  Ahead of an exam:
    python -m benchmark.duet_eval.lmeval.longbench_v2_utils --tokenizer <base checkpoint> [--data-root DIR]
Copied next to the generated duet_longbench_v2.yaml by tasks.py."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import time

DATA_ROOT = os.environ.get("DUET_DATA_ROOT", "/src/results/hfcache")
N_DOCS = 503
MAX_LEN = 120000            # config/model2maxlen.json: every 128K-context model of the official leaderboard
TEMPLATE_0SHOT = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""
FIELDS = ("_id", "domain", "sub_domain", "difficulty", "length", "answer")


def official_prompt(item):
    """get_pred: the zero-shot template filled in upstream's replacement order."""
    return (TEMPLATE_0SHOT.replace("$DOC$", item["context"].strip()).replace("$Q$", item["question"].strip())
            .replace("$C_A$", item["choice_A"].strip()).replace("$C_B$", item["choice_B"].strip())
            .replace("$C_C$", item["choice_C"].strip()).replace("$C_D$", item["choice_D"].strip()))


def truncate(prompt, tokenizer, max_len=MAX_LEN):
    """query_llm's middle truncation (the branch of a model with a local HF tokenizer).  Returns (prompt, tokens before)."""
    input_ids = tokenizer.encode(prompt)
    n = len(input_ids)
    if n > max_len:
        input_ids = input_ids[:max_len // 2] + input_ids[-max_len // 2:]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    return prompt, n


def extract_answer(response):
    """pred.py extract_answer, verbatim."""
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None


def prompts_path(tokenizer, data_root=DATA_ROOT, max_len=MAX_LEN):
    tag = pathlib.Path(str(tokenizer)).name + "-" + hashlib.sha1(str(tokenizer).encode()).hexdigest()[:8]
    return pathlib.Path(data_root) / "longbench-v2" / "prompts" / f"{tag}-{max_len}.jsonl"


def build_prompts(tokenizer_path, data_root=DATA_ROOT, max_len=MAX_LEN):
    """Write the prompts file of a tokenizer once (a lock serialises concurrent builders); returns its path."""
    import fcntl
    path = prompts_path(tokenizer_path, data_root, max_len)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file():
            return path
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)     # pred.py: get_pred's tokenizer
        with open(pathlib.Path(data_root) / "longbench-v2" / "data.json", encoding="utf-8") as fh:
            data = json.load(fh)
        if len(data) != N_DOCS:
            raise SystemExit(f"LongBench v2 data.json holds {len(data)} items, {N_DOCS} expected (revision changed?)")
        t0, rows = time.time(), []
        for i, item in enumerate(data):
            prompt, n = truncate(official_prompt(item), tokenizer, max_len)
            rows.append({"lb_index": i, **{k: item[k] for k in FIELDS}, "tokens_before_truncation": n, "truncated": n > max_len,
                         "prompt": prompt})
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.rename(path)
        print(f"LongBench v2 prompts for {tokenizer_path}: {len(rows)} items, {sum(r['truncated'] for r in rows)} truncated to "
              f"{max_len} tokens, {time.time() - t0:.0f} s -> {path}", flush=True)
    return path


# ---------------------------------------------------------------- lm-eval task functions
def docs(tokenizer=None, **kwargs):
    """custom_dataset: the 503 questions with the prompts of this tokenizer (run.py's --metadata gives the base checkpoint)."""
    import datasets
    if not tokenizer:
        raise ValueError("LongBench v2 prompts depend on the tokenizer: run.py passes it in lm-eval --metadata")
    path = prompts_path(tokenizer)
    if not path.is_file():
        build_prompts(tokenizer)
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    if len(rows) != N_DOCS:
        raise ValueError(f"{path}: {len(rows)} rows, {N_DOCS} expected")
    return {"test": datasets.Dataset.from_list(rows)}


def doc_to_text(doc):
    return doc["prompt"]


def process_results(doc, results):
    pred = extract_answer(results[0].strip())
    return {"accuracy": float(pred == doc["answer"]), "no_answer": float(pred is None)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, nargs="+", help="base checkpoint directories (tokenizer + chat template)")
    ap.add_argument("--data-root", default=DATA_ROOT)
    a = ap.parse_args()
    for tok in a.tokenizer:
        print(build_prompts(tok, a.data_root), flush=True)


if __name__ == "__main__":
    main()
