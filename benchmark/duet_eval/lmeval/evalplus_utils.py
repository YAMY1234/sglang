"""HumanEval+ (EvalPlus) for the paired DUET exams: the official data and instruction prompt; graded afterwards by
humaneval_grade.py (EvalPlus's own sanitizer and evaluator, base and plus tests).

Source: github.com/evalplus/evalplus at commit 26d6d00bb1fd0fa37f39c99d5290da67891d1c5e (the README's install target, Apache-2.0).
Data: HumanEvalPlus v0.1.10 (evalplus/data/humaneval.py HUMANEVAL_PLUS_VERSION; the release file of evalplus/humanevalplus_release,
decompressed as EvalPlus caches it, prepare_data.sh), 164 tasks in the file's order.  Prompt: the request EvalPlus sends to a chat
model it does not prefill (the OpenAI-compatible backend: codegen.run_codegen's instruction prefix, codegen.codegen's
task["prompt"].strip() + "\\n", provider/openai.py's message `instruction_prefix + "\\n```python\\n{prompt.strip()}\\n```"`, one user
message, gen/util/openai_request.py).  EvalPlus's vLLM / HF backends instead prefill the assistant turn with a response prefix and an
open code fence, which a thinking template cannot continue (the thinking block comes first).  lm-eval records only whether a reply
was given; humaneval_grade.py sanitizes the reply after the thinking block (evalplus.sanitize.sanitize with the task's entry point,
as codegen does with a chat model's reply) and runs evalplus.evaluate.  Copied next to the generated duet_humaneval_plus.yaml by
tasks.py."""
import json
import os

EVALPLUS_CACHE = os.path.join(os.environ.get("DUET_DATA_ROOT", "/src/results/hfcache"), "evalplus-cache")   # XDG_CACHE_HOME of the grader
DATA = os.path.join(EVALPLUS_CACHE, "evalplus", "HumanEvalPlus-v0.1.10.jsonl")
N_DOCS = 164
# evalplus/codegen.py run_codegen (the default, non-EvalPerf instruction)
INSTRUCTION_PREFIX = ("Please provide a self-contained Python script that solves the following problem in a markdown code "
                      "block:")


def docs(**kwargs):
    """custom_dataset: task_id, prompt and entry point of each task (the test inputs stay in the file the grader reads)."""
    import datasets
    with open(DATA, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if len(rows) != N_DOCS:
        raise ValueError(f"{DATA}: {len(rows)} tasks, {N_DOCS} expected")
    return {"test": datasets.Dataset.from_list([{"task_id": r["task_id"], "prompt": r["prompt"], "entry_point": r["entry_point"]}
                                                for r in rows])}


def doc_to_text(doc):
    prompt = doc["prompt"].strip() + "\n"                                        # codegen.codegen
    return INSTRUCTION_PREFIX + f"\n```python\n{prompt.strip()}\n```"            # provider/openai.py OpenAIChatDecoder.codegen


def process_results(doc, results):
    return {"answered": int(bool(results[0].strip()))}
