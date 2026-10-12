"""Exam task configs and the exam protocol.

Protocol (the defaults of exam_lmeval_job.sh and report_lmeval.py; repetitions = EXAMS[exam]["reps"]): AIME 2024 + 2025 x8,
GPQA-Diamond x8 (gpqa_se), MATH-500 x8, LiveCodeBench x8, IFEval x8, MMLU test set x1 (mmlu_se), RULER at 32K, 64K and 128K x1
(13 tasks, 100 documents per task and length, greedy); since 2026-10-10 HumanEval+ x8 (humaneval_plus) and LongBench v2 x4
(longbench_v2).  An exam job runs one RULER length (default RULER_LENGTHS_P0, 32K); the
64K and 128K runs are jobs of their own that build their documents per task and length (README.md).  RULER at 256K and 1M
(261632 and 1048064 tokens, README.md "Long-context RULER") is no longer part of the protocol.
A run with more repetitions than the protocol is reported on its first N (report_lmeval.py).  AA-LCR (aa_lcr) is kept but is not
part of the protocol.

GPQA-Diamond and MMLU run as OpenAI simple-evals runs them (gpqa_se / mmlu_se, se_utils.py: its zero-shot prompt that asks for a
last line 'Answer: $LETTER', its answer extraction, GPQA's per-repeat option permutations, MMLU's test set); LiveCodeBench is not an
lm-eval task (lcb_utils.py: the official prompt and extraction; lcb_grade.py: the official grader), nor are HumanEval+
(evalplus_utils.py: EvalPlus's data and chat prompt; humaneval_grade.py: EvalPlus's sanitizer and evaluator), LongBench v2
(longbench_v2_utils.py: the official prompt, middle truncation and extraction) and AA-LCR (aa_lcr_utils.py, graded by
aa_lcr_judge.py); every other exam is lm-eval's standard task (same data, prompts, answer filters and graders) with only
these changes:

* generation settings: the vendor's recommended sampling for the family, max_gen_toks raised to a ceiling that model.py
  clamps to context - prompt per request, no completion-style stop strings ("Question:", "Q:") that would cut a chat
  model's reasoning; RULER keeps lm-eval's greedy decoding with the output limit raised to 1024 tokens, thinking off;
* `repeats: 1` -- repetitions are separate lm-eval processes (run.py --rep k) because lm-eval grades only the first of
  `repeats` responses;
* the Hub datasets load at pinned revisions (HF_SETS; prepare_data.sh caches the same ones);
* AIME 2024 from the offline-cached mirror HuggingFaceH4/aime_2024, its gold answers restored to lm-eval's integers by
  aime_utils.process_docs;
* MATH-500 and AIME: a prompt that asks for the boxed answer (MATH_PROMPT); MATH-500 graded by math500_utils (lm-eval's
  hendrycks_math500 grader reads the text between the first and the last "$" and scores chat replies ~0 on every model);
* IFEval: langdetect (its language checks) seeded in run.py, so a reply is graded the same on every run.

Each generated duet_*.yaml `include:`s the installed yaml by absolute path, so the task body is whatever lm-eval version is
installed (lm-eval records task versions in results.json).  `python -m benchmark.duet_eval.lmeval.tasks --family kimi --out DIR`
writes the files and a manifest."""
import argparse
import json
import os
import pathlib
import re
import shutil

import yaml

DATA_ROOT = os.environ.get("DUET_DATA_ROOT", "/src/results/hfcache")    # offline exam data (RULER sources, LiveCodeBench set)
MAX_GEN_TOKS = 229376       # ceiling only (model.py clamps every request to context - prompt); lm-eval left-truncates prompts
                            # longer than max_length - max_gen_toks, so this leaves 32K tokens of prompt room in a 256K context

FAMILIES = {
    # Kimi Linear paper (arXiv 2510.26692) evaluation setup: temperature 1.0; no top_p/top_k given; the model card and
    # generation_config give nothing; the Instruct model has no thinking mode (chat template has no switch).
    "kimi": dict(sampling=dict(temperature=1.0, top_p=1.0, top_k=-1), chat_template_kwargs="", think_end="", think_required=0,
                 ruler_chat_template_kwargs=""),
    # Qwen3.8-Flash-Next model card "Best Practices", thinking mode: temperature 1.0, top_p 0.95, top_k 20, min_p 0,
    # presence_penalty 0; reasoning_effort xhigh (default, highest).  The NVFP4 card: "Benchmarked with temperature=1.0,
    # top_p=0.95, max_new_tokens=131072, and reasoning_effort=xhigh".
    "flash_next": dict(sampling=dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0),
                       chat_template_kwargs="enable_thinking:true;reasoning_effort:xhigh", think_end="</think>", think_required=1,
                       ruler_chat_template_kwargs="enable_thinking:false"),
    # NVIDIA-Nemotron-3.5-Lightning model card "Recommended Sampling: Temperature 1.0, Top_P 0.95" (= generation_config.json; no
    # top_k); its examples switch thinking with chat_template_kwargs enable_thinking (the template opens "<think>\n").
    "lightning": dict(sampling=dict(temperature=1.0, top_p=0.95, top_k=-1), chat_template_kwargs="enable_thinking:true",
                      think_end="</think>", think_required=1, ruler_chat_template_kwargs="enable_thinking:false"),
    # GLM-5.3-Flash (zai-org/GLM-5.3-Flash, revision eb9eb208eb0d): generation_config.json and the model card's evaluation
    # footnotes: temperature 1.0, top_p 0.95 (no top_k); "reasoning_effort ... low, high, and max.  It defaults to max ...  For
    # benchmark and leaderboard reproduction, keep the default max" -- the template writes it as a first system turn
    # ("<|system|>Reasoning Effort: Max") and always opens the thinking block ("<|assistant|><think>"); it has no thinking switch.
    # RULER: every RULER task continues an assistant prefix (lm-eval's gen_prefix), which this template renders as an assistant
    # turn without reasoning ("<|assistant|><think></think>Answer:"), so RULER runs with thinking off as for the other thinking
    # models; the template arguments are the exams' own.
    "glm": dict(sampling=dict(temperature=1.0, top_p=0.95, top_k=-1), chat_template_kwargs="reasoning_effort:max",
                think_end="</think>", think_required=1, ruler_chat_template_kwargs="reasoning_effort:max"),
}

# lm-eval's completion-style math prompts ("Problem: ...\nAnswer:") give a chat model no reason to box its answer, and an
# unboxed correct answer scores 0 (smoke: 20% of MATH-500 answers on BOTH arms).  The NeMo-Skills math prompt (the colleague's
# harness, Qwen/DeepSeek evaluation practice) asks for the box; same prompt for both arms and for MATH-500 and AIME.
MATH_PROMPT = "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n\n{{problem}}"

# Hub datasets of the lm-eval exams at the commits the reported exams read (the version directories of their datasets cache);
# build() writes the revision into each yaml and prepare_data.sh caches these, so an online run cannot read a newer head
HF_SETS = {
    "aime24": dict(path="HuggingFaceH4/aime_2024", split="train", revision="2fe88a2f1091d5048c0f36abc874fb997b3dd99a"),
    "aime25": dict(path="math-ai/aime25", split="test", revision="563bb8404243c5f09de6ec262f2db674fe5bce9b"),
    "math500": dict(path="HuggingFaceH4/MATH-500", split="test", revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"),
    "ifeval": dict(path="google/IFEval", split="train", revision="966cd89545d6b6acfd7638bc708b98261ca58e84"),
}

RULER_TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
               "niah_multiquery", "niah_multivalue", "vt", "cwe", "fwe", "qa_squad", "qa_hotpot"]
RULER_RETRIEVAL = RULER_TASKS[:8]          # the needle-in-a-haystack family (reported as RULER-NIAH); the other five are canaries
SE_EXAMS = ("gpqa_se", "mmlu_se")          # simple-evals exams (se_utils.py): repetition and shard reach their docs as metadata
EXAMS = {
    # the cached mirror has lower-case columns (problem/answer); the grader already matches "answer" case-insensitively
    # the mirror's answers are zero-padded strings; aime_utils.process_docs restores lm-eval's integer answers (runs before
    # 2026-10-03 graded 7 problems against "025"-style strings: ../archive/regrade_aime24.py corrects their sample files)
    "aime24": dict(yamls=["aime/aime24.yaml"], reps=8, utils="aime_utils.py",
                   override=dict(dataset_path="HuggingFaceH4/aime_2024", doc_to_text=MATH_PROMPT, doc_to_target="{{answer}}",
                                 process_docs="!function aime_utils.process_docs")),
    "aime25": dict(yamls=["aime/aime25.yaml"], reps=8, override=dict(doc_to_text=MATH_PROMPT)),
    # lm-eval's hendrycks_math500 grader ($-delimited answer) mis-scores chat responses; same data and prompt, boxed answer,
    # math-verify + lm-eval string equivalence (math500_utils.py)
    "math500": dict(static=dict(task="duet_math500", dataset_path="HuggingFaceH4/MATH-500", test_split="test", output_type="generate_until",
                                doc_to_text=MATH_PROMPT, doc_to_target="{{answer}}",
                                process_results="!function math500_utils.process_results",
                                metric_list=[{"metric": "math_verify", "aggregation": "mean", "higher_is_better": True},
                                             {"metric": "exact_match", "aggregation": "mean", "higher_is_better": True},
                                             {"metric": "no_boxed", "aggregation": "mean", "higher_is_better": False}],
                                metadata={"version": 1}),
                    utils="math500_utils.py", reps=8),
    "ifeval": dict(yamls=["ifeval/ifeval.yaml"], reps=8),
    # LiveCodeBench v6, 454 problems (contest dates 2024-08 .. 2025-04; lcb/build_lcb_data.py); official LiveCodeBench prompt
    # and code extraction (lcb_utils.py), official grader run afterwards by lcb_grade.py (pass@1 over public + private tests)
    "lcb": dict(static=dict(task="duet_lcb", dataset_path="json", dataset_kwargs={"data_files": {"train": DATA_ROOT + "/lcb/lcb_v6_2408_2505.jsonl"}},
                            test_split="train", output_type="generate_until",
                            doc_to_text="!function lcb_utils.doc_to_text", doc_to_target="{{question_id}}",
                            process_results="!function lcb_utils.process_results",
                            metric_list=[{"metric": "code_extracted", "aggregation": "mean", "higher_is_better": True}],
                            metadata={"version": 1}),
                utils="lcb_utils.py", reps=8, n_docs=454),
    # AA-LCR v1.1 (aa_lcr_data.py): 100 questions over ~100K-token document sets, the official prompt; graded afterwards by the
    # official equality-check prompts on a local judge model (aa_lcr_judge.py); lm-eval records only that a reply was given
    "aa_lcr": dict(static=dict(task="duet_aa_lcr", dataset_path="json", dataset_kwargs={"data_files": {"train": DATA_ROOT + "/aa-lcr/aa_lcr_v1.1.jsonl"}},
                               test_split="train", output_type="generate_until", doc_to_text="{{prompt}}", doc_to_target="{{answer}}",
                               process_results="!function aa_lcr_utils.process_results",
                               metric_list=[{"metric": "answered", "aggregation": "mean", "higher_is_better": True}],
                               metadata={"version": 1}),
                   utils="aa_lcr_utils.py", reps=8),
    # HumanEval+ (EvalPlus v0.1.10, 164 tasks; evalplus_utils.py): EvalPlus's instruction prompt as its chat backend sends it (one
    # user message, no assistant prefill); graded afterwards by humaneval_grade.py (EvalPlus's sanitizer and evaluator, base + plus
    # tests, both arms in one call); lm-eval records only that a reply was given
    "humaneval_plus": dict(static=dict(task="duet_humaneval_plus", custom_dataset="!function evalplus_utils.docs", test_split="test",
                                       output_type="generate_until", doc_to_text="!function evalplus_utils.doc_to_text",
                                       doc_to_target="{{task_id}}", process_results="!function evalplus_utils.process_results",
                                       metric_list=[{"metric": "answered", "aggregation": "mean", "higher_is_better": True}],
                                       metadata={"version": 1}),
                           utils="evalplus_utils.py", reps=8, n_docs=164),
    # LongBench v2 (THUDM/LongBench-v2, 503 questions; longbench_v2_utils.py): the official zero-shot prompt with the official middle
    # truncation to 120000 tokens of the evaluated model's tokenizer, the official answer extraction; thinking on, no output cap
    "longbench_v2": dict(static=dict(task="duet_longbench_v2", custom_dataset="!function longbench_v2_utils.docs", test_split="test",
                                     output_type="generate_until", doc_to_text="!function longbench_v2_utils.doc_to_text",
                                     doc_to_target="{{answer}}", process_results="!function longbench_v2_utils.process_results",
                                     metric_list=[{"metric": "accuracy", "aggregation": "mean", "higher_is_better": True},
                                                  {"metric": "no_answer", "aggregation": "mean", "higher_is_better": False}],
                                     metadata={"version": 1}),
                         utils="longbench_v2_utils.py", reps=4, n_docs=503),
    # OpenAI simple-evals GPQA-Diamond (8 repeats, each with simple-evals' option permutation of that repeat) and MMLU (test set,
    # one pass): prompt, extraction and data in se_utils.py; run.py passes the repetition and the doc shard as lm-eval metadata
    "gpqa_se": dict(static=dict(task="duet_gpqa_se", custom_dataset="!function se_utils.gpqa_docs", test_split="test",
                                output_type="generate_until", doc_to_text="!function se_utils.doc_to_text", doc_to_target="{{answer}}",
                                process_results="!function se_utils.process_results_gpqa",
                                metric_list=[{"metric": "accuracy", "aggregation": "mean", "higher_is_better": True},
                                             {"metric": "no_answer", "aggregation": "mean", "higher_is_better": False}],
                                metadata={"version": 1}),
                    utils="se_utils.py", reps=8),
    "mmlu_se": dict(static=dict(task="duet_mmlu_se", custom_dataset="!function se_utils.mmlu_docs", test_split="test",
                                output_type="generate_until", doc_to_text="!function se_utils.doc_to_text", doc_to_target="{{answer}}",
                                process_results="!function se_utils.process_results_mmlu",
                                metric_list=[{"metric": "accuracy", "aggregation": "mean", "higher_is_better": True},
                                             {"metric": "no_answer", "aggregation": "mean", "higher_is_better": False}],
                                metadata={"version": 1}),
                    utils="se_utils.py", reps=1),
    # RULER keeps lm-eval's greedy decoding (a 30-token retrieval answer sampled at T=1 once would be noise) but raises the
    # output limit to 1024: lm-eval's vt limit of 30 tokens scores Kimi 0 on both arms because it restates the prompt before
    # naming the variables (colleague's RULER run, Kimi.md 4.6).  Thinking is switched off for RULER (FAMILIES ruler_chat_template_kwargs, run.py).
    "ruler": dict(yamls=[f"ruler/{t}.yaml" for t in RULER_TASKS], reps=1, keep_generation=True, generation_override={"max_gen_toks": 1024}),
}
P0 = ("aime24", "aime25", "gpqa_se", "math500", "lcb", "ifeval", "mmlu_se", "ruler", "humaneval_plus", "longbench_v2")
# exams that run as lm-eval processes of their own (run.py --exams <exam>): LCB (its system message), AA-LCR and LongBench v2
# (~100K-token prompts: lm-eval's max_length is raised so it never cuts one), HumanEval+ (graded by humaneval_grade.py); RULER too
# (greedy, its own documents).  DOC_SHARD_EXAMS: run.py --doc-shard i/n runs their documents i, i+n, ... (lm-eval --samples, so a
# sample keeps its doc_id in the full set and every document keeps the seed of the unsharded run)
OWN_PROCESS = ("lcb", "aa_lcr", "humaneval_plus", "longbench_v2")
LONG_PROMPT = ("aa_lcr", "longbench_v2")
DOC_SHARD_EXAMS = ("lcb", "humaneval_plus", "longbench_v2")
RULER_LENGTHS_P0 = (32768,)            # a job's default RULER length; 65536 and 131072 run as jobs of their own


def task_root():
    import lm_eval.tasks          # deferred: the report and prepare_data.sh read this module's tables where lm-eval is absent
    return pathlib.Path(lm_eval.tasks.__file__).parent


def exam_of_task(task):
    """lm-eval task name of a sample file -> its exam in EXAMS, or None (tasks of retired exams)."""
    base = task[len("duet_"):] if task.startswith("duet_") else task
    if base in EXAMS:
        return base
    ruler = base[len("ruler_"):] if base.startswith("ruler_") else base      # installed names: niah_*, ruler_vt, ruler_cwe, ...
    return "ruler" if ruler in RULER_TASKS else None


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_constructor("!function", lambda loader, node: f"!function {node.value}")


def read_config(path):
    return yaml.load(pathlib.Path(path).read_text(), Loader=_Loader)


def resolved(path, key):
    """Value of `key` along the include chain of an installed yaml (nearest wins)."""
    cfg = read_config(path)
    if key in cfg:
        return cfg[key]
    include = cfg.get("include")
    for inc in ([include] if isinstance(include, str) else include or []):
        value = resolved(pathlib.Path(path).parent / inc, key)
        if value is not None:
            return value
    return None


def task_name(installed):
    return f"duet_{resolved(installed, 'task')}"


def pinned(exam, dataset_path):
    """yaml keys that load `exam`'s Hub dataset at its HF_SETS revision ({} for an exam without a Hub dataset)."""
    pin = HF_SETS.get(exam)
    if pin is None:
        return {}
    if dataset_path != pin["path"]:
        raise ValueError(f"{exam}: the task reads {dataset_path!r}, HF_SETS pins {pin['path']!r} (installed lm-eval changed?)")
    return {"dataset_kwargs": {"revision": pin["revision"]}}


def build(family, out_dir, exams, ruler_tasks=None, temperature=None):
    """Write the yaml files for `exams`; returns {exam: {"tasks": [lm-eval task names], "reps": n}}.
    ruler_tasks: subset of RULER_TASKS to build (default all 13)."""
    fam = dict(FAMILIES[family])
    if temperature is not None:   # diagnostic arms only (e.g. the temperature dependence of a failure); exams use the vendor's value
        fam["sampling"] = dict(fam["sampling"], temperature=temperature)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = task_root()
    manifest = {}
    for exam in exams:
        spec = dict(EXAMS[exam])
        if exam == "ruler" and ruler_tasks:
            unknown = set(ruler_tasks) - set(RULER_TASKS)
            if unknown:
                raise ValueError(f"unknown RULER tasks {sorted(unknown)}")
            spec["yamls"] = [f"ruler/{t}.yaml" for t in RULER_TASKS if t in ruler_tasks]
        names = []
        if "static" in spec:
            cfg = dict(spec["static"], tag="duet_exam", repeats=1,
                       generation_kwargs={"until": [], "do_sample": True, **fam["sampling"], "max_gen_toks": MAX_GEN_TOKS})
            cfg.update(pinned(exam, cfg.get("dataset_path")))
            text = re.sub(r"'(!function [\w.]+)'", r"\1", yaml.safe_dump(cfg, sort_keys=False))
            (out / f"{cfg['task']}.yaml").write_text(text)
            shutil.copy(pathlib.Path(__file__).parent / spec["utils"], out / spec["utils"])
            manifest[exam] = {"tasks": [cfg["task"]], "reps": spec["reps"], "run_as": cfg["task"]}
            continue
        for rel in spec["yamls"]:
            installed = root / rel
            if not installed.is_file():
                raise FileNotFoundError(installed)
            name = task_name(installed)
            cfg = {"include": str(installed), "task": name, "tag": "duet_exam", "repeats": 1, **spec.get("override", {})}
            cfg.update(pinned(exam, cfg.get("dataset_path") or resolved(installed, "dataset_path")))
            if not spec.get("keep_generation"):
                cfg["generation_kwargs"] = {"until": [], "do_sample": True, **fam["sampling"], "max_gen_toks": MAX_GEN_TOKS}
            elif spec.get("generation_override"):
                cfg["generation_kwargs"] = {**(resolved(installed, "generation_kwargs") or {}), **spec["generation_override"]}
            (out / f"{name}.yaml").write_text(re.sub(r"'(!function [\w.]+)'", r"\1", yaml.safe_dump(cfg, sort_keys=False)))
            if spec.get("utils"):
                shutil.copy(pathlib.Path(__file__).parent / spec["utils"], out / spec["utils"])
            names.append(name)
        manifest[exam] = {"tasks": names, "reps": spec["reps"], "run_as": ",".join(names)}
    (out / "manifest.json").write_text(json.dumps({"family": family, "task_root": str(root), "exams": manifest}, indent=1))
    return manifest


def parse_ruler_tasks(spec):
    if spec in (None, "", "all"):
        return None
    if spec == "retrieval":
        return list(RULER_RETRIEVAL)
    return [x for x in spec.split(",") if x]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--exams", default=",".join(P0))
    ap.add_argument("--ruler-tasks", default="all", help="comma list of RULER subtasks, 'all' (13) or 'retrieval' (the 8 NIAH tasks)")
    ap.add_argument("--temperature", type=float, default=None, help="override the family's sampling temperature (diagnostic arms only)")
    args = ap.parse_args()
    manifest = build(args.family, args.out, args.exams.split(","), ruler_tasks=parse_ruler_tasks(args.ruler_tasks), temperature=args.temperature)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
