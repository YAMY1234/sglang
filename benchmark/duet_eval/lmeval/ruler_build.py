"""Build the RULER documents of a long-context run ahead of the exam, one process per (subtask, length), into the cache that
run.py --ruler-build-samples N --ruler-cache DIR reads (the same patched lm-eval builders, so the exam gets these documents).
Per build it reports the build time, the RULER length field and the prompt the server receives (chat template + document + the
task's answer prefix, as model.py sends it) against the server context, and fails if a prompt leaves no room for the answer.

    python -m benchmark.duet_eval.lmeval.ruler_build --family kimi --tokenizer /models/Kimi-Linear-48B-A3B-Instruct \\
        --lengths 65536,131072 --n 100 --cache DIR --ruler-data $DUET_DATA_ROOT/ruler --server-context 262144"""
import argparse
import concurrent.futures as cf
import json
import multiprocessing
import time

from . import tasks
from .model import parse_kv
from .run import RULER_BUILDERS, patch_ruler_builds, patch_ruler_sources

ANSWER_ROOM = 64        # tokens a prompt must leave for the answer (RULER answers: <= ~40 tokens; model.py keeps 8 more)


def build_one(job):
    task, length, a = job
    patch_ruler_sources(a["ruler_data"])
    patch_ruler_builds(a["n"], seed=1234, cache_dir=a["cache"])
    import importlib
    from transformers import AutoTokenizer
    module, fn = RULER_BUILDERS[task]
    t0 = time.time()
    docs = getattr(importlib.import_module(f"lm_eval.tasks.ruler.{module}"), fn)(max_seq_lengths=[length], tokenizer=a["tokenizer"])["test"]
    seconds = time.time() - t0
    tok = AutoTokenizer.from_pretrained(a["tokenizer"], trust_remote_code=True)
    kwargs = parse_kv(tasks.FAMILIES[a["family"]]["ruler_chat_template_kwargs"])
    prompts = []
    for d in docs:
        # lm-eval with --apply_chat_template and a gen_prefix: [user: document, assistant: answer prefix], continued
        text = tok.apply_chat_template([{"role": "user", "content": d["input"]}, {"role": "assistant", "content": d["gen_prefix"]}],
                                       tokenize=False, add_generation_prompt=False, continue_final_message=True, **kwargs)
        prompts.append(len(tok(text, add_special_tokens=False).input_ids))
    fields = [d["length"] for d in docs]
    return dict(task=task, length=length, docs=len(docs), build_s=round(seconds, 1), ruler_length_min=min(fields),
                ruler_length_max=max(fields), prompt_min=min(prompts), prompt_max=max(prompts),
                room_min=a["server_context"] - max(prompts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=sorted(tasks.FAMILIES))
    ap.add_argument("--tokenizer", required=True, help="the base checkpoint directory, exactly as the exam passes it (cache key)")
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--n", type=int, required=True, help="documents per subtask and length")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ruler-data", required=True)
    ap.add_argument("--server-context", type=int, required=True)
    ap.add_argument("--tasks", default="all", help="comma list of RULER subtasks, 'all' or 'retrieval'")
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--out", default=None, help="json report")
    a = ap.parse_args()
    shared = dict(family=a.family, tokenizer=a.tokenizer, n=a.n, cache=a.cache, ruler_data=a.ruler_data, server_context=a.server_context)
    jobs = [(t, int(L), shared) for L in a.lengths.split(",") for t in tasks.parse_ruler_tasks(a.tasks) or tasks.RULER_TASKS]
    rows, failed = [], []
    with cf.ProcessPoolExecutor(a.processes or len(jobs), mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1) as pool:
        futures = {pool.submit(build_one, j): j[:2] for j in jobs}
        for f in cf.as_completed(futures):
            try:
                r = f.result()
            except BaseException as e:       # noqa: BLE001 - report every build, then fail
                failed.append((futures[f], repr(e)))
                print(f"BUILD FAILED {futures[f]}: {e!r}", flush=True)
                continue
            rows.append(r)
            print(json.dumps(r), flush=True)
    rows.sort(key=lambda r: (r["length"], r["task"]))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)
    short = [r for r in rows if r["room_min"] < ANSWER_ROOM]
    for r in short:
        print(f"PROMPT TOO LONG {r['task']}@{r['length']}: longest prompt {r['prompt_max']} leaves {r['room_min']} < {ANSWER_ROOM} tokens", flush=True)
    print(f"RULER BUILD {'FAILED' if failed or short else 'DONE'}: {len(rows)} builds, {len(failed)} failed, {len(short)} too long", flush=True)
    raise SystemExit(1 if failed or short else 0)


if __name__ == "__main__":
    main()
