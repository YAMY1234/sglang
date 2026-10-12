# Paired DUET exams with lm-eval and simple-evals on SGLang

One job, one node: `bash exam_lmeval_job.sh <kimi|flash_next|lightning|glm> <out> [exams]` starts NDUET DUET servers and NSTOCK base
servers (identical `ServerArgs` except the DUET enablement, `../launch_duet.py`), a least-outstanding balancer per arm, one
lm-eval process per repetition and arm, and prints the paired table.  Without `[exams]` it runs every exam of the protocol below
(RULER at 32K).  More nodes = more jobs sharing `<out>`, with disjoint `REP_LIST`, MMLU shards (`SE_SHARD`) and one RULER length
or task group per job (`RULER_LENGTHS`, `RULER_TASKS`, `RULER_TAG`); `report_lmeval.py` merges everything.  Every knob and path
is listed at the top of `exam_lmeval_job.sh` (its defaults are the paths of the cluster the reported runs used: set them).
Installation, the releases, serving and the container: `../README.md`; to evaluate a released set straight from the Hugging Face
Hub (`Latent-Voyager/<original model>-SEED`: download and checks of the release and its base, every family, the whole sequence
from download to report): `../README.md`, section "Evaluate a released DUET set from the Hub".

## Protocol

| exam | repetitions | questions | source |
|---|---|---|---|
| AIME 2024 + 2025 (`aime24`, `aime25`) | 8 | 30 + 30 | lm-eval task, boxed-answer prompt |
| GPQA-Diamond (`gpqa_se`) | 8 | 198 | OpenAI simple-evals prompt and extraction |
| MATH-500 (`math500`) | 8 | 500 | boxed-answer prompt, math-verify |
| LiveCodeBench v6 (`lcb`) | 8 | 454 | official prompt and grader |
| IFEval (`ifeval`) | 8 | 541 | lm-eval task, prompt-level strict |
| MMLU (`mmlu_se`) | 1 | 14042 (test set) | OpenAI simple-evals prompt and extraction |
| RULER (`ruler`) at 32K, 64K and 128K | 1 | 13 tasks x 100 documents per length | lm-eval tasks, greedy |
| HumanEval+ (`humaneval_plus`) | 8 | 164 | EvalPlus: HumanEval+ v0.1.10, its chat prompt, sanitizer and evaluator (base + plus tests) |
| LongBench v2 (`longbench_v2`) | 4 | 503 | official zero-shot prompt, middle truncation to 120000 tokens, official answer extraction |

The repetition counts are `tasks.EXAMS[exam]["reps"]` and the defaults of the job and the report.  Pre-registered rule: a run
with more repetitions than the protocol is reported on its first N (repetitions 0..N-1); the report prints, per arm and exam,
the repetitions read, dropped and missing (`--reps exam=N,...` or `--reps all` override it, for analysis only).  RULER at 256K
and 1M is no longer part of the protocol (section "Long-context RULER" keeps the path); AA-LCR is not part of it either (its
code stays, section "AA-LCR").  HumanEval+ and LongBench v2 joined the protocol on 2026-10-10 (section "HumanEval+ and LongBench v2";
twinstar docs/90 has the official sources, every deviation and the first runs).

## What is compared

| | A: DUET | B: base |
|---|---|---|
| weights | base + a DUET release (`DUET_REL`: directory or Hub id) | base, unchanged |
| serving | this fork (the base arm is this fork with every DUET switch off, not upstream SGLang), the same `ServerArgs` except the DUET switches: context 262144 (Kimi-Linear RULER 64K / 128K: 270336), 128 running requests, decode CUDA graph, radix cache off, pytorch sampler; Kimi-Linear: chunked prefill off, fp32 conv and recurrent states; Lightning: chunked prefill off, fp32 Mamba states, the DUET arm with `--duet-numerics reference`; Flash-Next: chunked prefill 8192, fp32 recurrent state from the model config, triton GDN kernels, the DUET arm's emitters in fp32 with the exact prefix state (the reference profile's values, set explicitly) | same |
| prompts, graders | GPQA and MMLU as OpenAI simple-evals runs them, LiveCodeBench with its official prompt and grader, the other exams as lm-eval 0.4.13 tasks (below) | identical |
| sampling | vendor settings (below); `sampling_seed` = hash(pair seed + repetition, prompt tokens, occurrence of the prompt in the lm-eval process): the same random stream per question and repetition on both arms; a retried request keeps its seed.  The replies stay identical only until the first logit difference (kernels are not batch-invariant), so a rerun is not token-identical | identical |

A - B per question (mean over the repetitions both arms completed), SE over questions.  The two arms of a reported cell may come
from separate jobs (same settings, same seeds, same prompts, not served side by side).  `report_lmeval.py` checks every paired
sample: the same prompt (lm-eval `prompt_hash`) and the same sampling seed (the sidecar record of the reply on each arm), and
warns with counts and examples otherwise.  The occurrence count in the seed is per lm-eval process, so both arms of a pairing must
run the same exams per process and the same `SE_SHARD` split (MMLU has prompts that occur twice).

## Run the protocol for one release

Prerequisites (`../README.md`, sections "Install" and "Container"): this repository on a node with 4 GPUs (the reported runs:
GB300, 277 GiB each), the exam packages (`requirements-exams.txt`; `EXTRA_PYTHONPATH=<dir>` when they are installed with
`pip install --no-deps --target <dir>`, empty when they are importable already), the exam data (`prepare_data.sh`), the base
checkpoint at the revision of the release table, and the release directory: `python -m benchmark.duet_eval.fetch_duet --release
Latent-Voyager/<repository> --base` downloads and checks both and prints them as `DUET_REL` and the family's base variable
(`../README.md`, section "Evaluate a released DUET set from the Hub").  Every line below is one job on one
node; run them on separate nodes or one after another.  Jobs that run at the same time on one `<out>` need distinct
`SLURM_JOB_ID` values (Slurm sets them; set one per job elsewhere): the job names `tasks-<id>/`, `servers-<id>/` and `report-<id>`
by it.  Each job serves both arms (`NDUET` + `NSTOCK` servers); `NSTOCK=0` (DUET only) or `NDUET=0` (base only) runs one arm per
job with the same settings, as the reported v3.5 arms did (their base arms were run earlier, with the same seeds).

```bash
cd <this repository>
export DUET_DATA_ROOT=<exam-data> HF_HOME=<hf-cache> HF_OFFLINE=1 EXTRA_PYTHONPATH=   # or =<package dir>
J=benchmark/duet_eval/lmeval/exam_lmeval_job.sh
```

Per release, the family, the base and the layout of one 4-GPU node that serves both arms (the directories named as the twinstar
repository's `release/` names them; with `fetch_duet.py`, `DUET_REL` and the base variable are the ones it printed):

```bash
# Flash-Next BF16 v3.5 (the v3.5-b set, round 2): TP 2, one DUET and one base server per node
F=flash_next; export FN_BASE=/models/Qwen3.8-Flash-Next DUET_REL=/releases/duet-flash-next-v3.5-b-r2 TP=2 NDUET=1 NSTOCK=1
# Flash-Next NVFP4 v3.5: TP 1, two servers per arm
F=flash_next; export FN_BASE=/models/Qwen3.8-Flash-Next-NVFP4 DUET_REL=/releases/duet-flash-next-nvfp4-v3.5 TP=1 NDUET=2 NSTOCK=2
# Kimi-Linear v3.5: TP 1, two servers per arm
F=kimi; export KIMI_BASE=/models/Kimi-Linear-48B-A3B-Instruct DUET_REL=/releases/duet-kimi-v3.5 NDUET=2 NSTOCK=2
# Nemotron-3.5-Lightning v3.5: TP 1, two servers per arm
F=lightning; export LIGHTNING_BASE=/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 DUET_REL=/releases/duet-lightning-v3.5 NDUET=2 NSTOCK=2
# GLM-5.3-Flash: TP 4, one server per node -- the base arm (the family's defaults NDUET=0 NSTOCK=1 TP=4) ...
F=glm; export GLM_BASE=/models/GLM-5.3-Flash NDUET=0 NSTOCK=1 MRR=256
# ... and, in jobs of their own, the DUET arm (as the v3.5 finals of 2026-10-09 ran it)
F=glm; export GLM_BASE=/models/GLM-5.3-Flash DUET_REL=/releases/duet-glm53-flash-v3.5 NDUET=1 NSTOCK=0 MRR=256
```

The reported v3.5 DUET arms ran DUET only: Flash-Next BF16 `TP=2 NDUET=2 NSTOCK=0`, the other three `TP=1 NDUET=4 NSTOCK=0` (the
Lightning RULER 32K job served both arms, 2 + 2).

Then, with `O=out/<model>-v3.5`:

```bash
for k in 0 1 2 3 4 5 6 7; do
  REP_LIST=$k bash $J $F $O aime24,aime25,math500,ifeval
  REP_LIST=$k bash $J $F $O gpqa_se
  REP_LIST=$k bash $J $F $O lcb                      # graded at the end of the job by lcb_grade.py
done
SE_SHARD=0/2 bash $J $F $O mmlu_se                   # MMLU, half of the 14042 questions per job
SE_SHARD=1/2 bash $J $F $O mmlu_se
RULER_LENGTHS=32768 RULER_TAG=32k bash $J $F $O ruler
REP_LIST="0 1 2 3" bash $J $F $O humaneval_plus      # and "4 5 6 7"; graded afterwards over both arms (humaneval_grade.py, below)
for k in 0 1 2 3; do REP_LIST=$k bash $J $F $O longbench_v2; done
R="RULER_N=100 RULER_BUILD=100 RULER_CACHE=$O/ruler-cache"   # Kimi-Linear: R="$R CTX=270336", as the reported runs
env $R RULER_LENGTHS=65536 RULER_TAG=64k bash $J $F $O ruler
env $R RULER_LENGTHS=131072 RULER_TAG=128k-a RULER_TASKS=niah_single_1,niah_single_2,niah_single_3 bash $J $F $O ruler
env $R RULER_LENGTHS=131072 RULER_TAG=128k-b RULER_TASKS=niah_multikey_1,niah_multikey_2,niah_multikey_3 bash $J $F $O ruler
env $R RULER_LENGTHS=131072 RULER_TAG=128k-c RULER_TASKS=niah_multivalue,niah_multiquery,vt bash $J $F $O ruler
env $R RULER_LENGTHS=131072 RULER_TAG=128k-d RULER_TASKS=cwe,fwe bash $J $F $O ruler
env $R RULER_LENGTHS=131072 RULER_TAG=128k-e RULER_TASKS=qa_squad,qa_hotpot bash $J $F $O ruler
```

These reproduce the settings and seeds of the reported runs (twinstar repository: the `job.txt` of each series listed in
`results/acc-sgl/v35-finals/provenance.md`; `units/` holds the jobs of 2026-10-07).  The reported runs grouped exams and
repetitions into jobs differently and used one `<out>` per exam group; a seed does not depend on that grouping, only on the
`SE_SHARD` split of MMLU (the reported v3.5 runs: `0/2` and `1/2`; the FN-BF16 v3.5-b runs: 12 strided shards, `SE_SHARD=i/12`).
The Kimi-Linear RULER 64K and 128K runs (base, v3 and v3.5) served at `CTX=270336`, every other reported run at 262144.  RULER at
32K uses lm-eval's own document build (500 documents per task from one random sequence over the 13 tasks, the first 100 scored),
so it must run all 13 tasks in one process; at 64K and 128K every task and length is built from a fresh random state
(`RULER_BUILD=100`, section "Long-context RULER"), so the tasks can be split over jobs and the cache serves both arms.  A job
given RULER must include repetition 0 in `REP_LIST` (RULER runs once, as repetition 0).  A job killed at a time limit is finished
by rerunning the same line with `RESUME=1` (section "Resuming").  The job fails when a server log shows retracted requests (KV
cache full): rerun those repetitions with a lower `MRR`.

**HumanEval+ grading**, once every repetition of both arms is written (EvalPlus and tree-sitter from `requirements-evalplus.txt`,
installed apart: `pip install --no-deps --target <dir> -r requirements-evalplus.txt`, `<dir>` on `PYTHONPATH`; it runs model-written
code, so inside the container of a compute node; `--passes 2` grades twice and counts the verdicts that differ):

```bash
python -m benchmark.duet_eval.lmeval.humaneval_grade --out $O/humaneval-grade --passes 2 $O/stock $O/duet
```

**Report** (both arms in `$O`; add `--check-prompts benchmark/duet_eval/lmeval/reference/kimi-linear-prompts.json` or
`.../reference/flash-next-nvfp4-prompts.json` to compare every prompt with the reported Kimi-Linear / Flash-Next NVFP4 exams):

```bash
python -m benchmark.duet_eval.lmeval.report_lmeval --arms stock=$O/stock duet=$O/duet \
    --ruler $(for d in $O/stock-ruler*; do echo stock=$d; done) $(for d in $O/duet-ruler*; do echo duet=$d; done) \
    --md --out $O/report.json
```

The prompt check must report 0 different prompts and the seed check 0 different seeds (the reference files hold the exams of
the protocol except RULER 64K / 128K).  To compare two DUET sets on the same base (v3.5 against v3), give both as arms, the
baseline first: `--arms v3=<v3 out>/duet v35=<v3.5 out>/duet`.

**Reported v3.5 results** (twinstar repository, docs/67; DUET score, paired difference to the base +- SE, [repetitions read /
required]).  Sampling at temperature 1.0 makes a rerun differ question by question; the paired differences of a rerun should agree
within their SE.

| model | AIME 24+25 | GPQA | MATH-500 | LCB | IFEval | MMLU | RULER 32K | RULER 64K | RULER 128K |
|---|---|---|---|---|---|---|---|---|---|
| FN-BF16 v3.5 (v3.5-b, round 2) | 98.8 (-0.6 ± 0.5) [8/8] | 91.5 (-0.3 ± 0.4) [8/8] | 98.8 (-0.5 ± 0.1) [8/8] | 88.9 (+0.4 ± 0.4) [8/8] | 91.1 (-0.6 ± 0.3) [8/8] | 92.6 (+0.1 ± 0.1) [1/1] | 96.4 (+3.2 ± 0.6) [1/1] | 95.5 (+1.2 ± 0.5) [1/1] | 95.0 (-0.5 ± 0.4) [1/1] |
| FN-NVFP4 v3.5 | 99.6 (+0.0 ± 0.4) [8/8] | 91.2 (-1.1 ± 0.5) [8/8] | 98.9 (-0.3 ± 0.2) [8/8] | 88.8 (-0.1 ± 0.4) [8/8] | 91.5 (-0.1 ± 0.3) [8/8] | 92.5 (+0.2 ± 0.2) [1/1] | 96.1 (+1.4 ± 0.5) [1/1] | 95.4 (+0.5 ± 0.5) [1/1] | 94.6 (-0.7 ± 0.4) [1/1] |
| Kimi v3.5 | 66.9 (-0.6 ± 2.5) [8/8] | 68.0 (+0.4 ± 1.4) [8/8] | 92.4 (-0.8 ± 0.5) [8/8] | 43.4 (-0.9 ± 0.8) [8/8] | 71.7 (-1.5 ± 0.6) [8/8] | 78.3 (+0.6 ± 0.4) [1/1] | 95.0 (+0.0 ± 0.1) [1/1] | 93.9 (-0.1 ± 0.2) [1/1] | 94.1 (-0.4 ± 0.2) [1/1] |
| Lightning v3.5 | 88.8 (-0.8 ± 1.1) [8/8] | 75.0 (-1.3 ± 1.0) [8/8] | 98.3 (+0.1 ± 0.2) [8/8] | 77.6 (-0.6 ± 0.6) [8/8] | 92.0 (+0.4 ± 0.4) [8/8] | 88.0 (-0.2 ± 0.2) [1/1] | 96.4 (-0.0 ± 0.2) [1/1] | 96.0 (-0.3 ± 0.2) [1/1] | 95.6 (-0.3 ± 0.3) [1/1] |
| GLM-5.3-Flash v3.5 | 98.1 (+2.1 ± 1.3) [8/8] | 91.1 (-0.2 ± 0.5) [8/8] | 98.6 (-0.1 ± 0.2) [8/8] | 88.2 (+0.4 ± 0.4) [8/8] | 90.7 (+0.8 ± 0.4) [8/8] | 92.8 (-0.1 ± 0.1) [1/1] | 96.5 (-0.2 ± 0.1) [1/1] | 96.5 (+0.2 ± 0.2) [1/1] | 96.1 (-0.3 ± 0.2) [1/1] |

Notes on the reported runs (docs/67, docs/80 and docs/82 of the twinstar repository have the rest; the reports are its
`results/acc-sgl/v35-finals/reports/{fn-bf16-v35b2,fn-nvfp4,kimi,lightning,glm53-v35-final}.json`): the FN-BF16 row is the released v3.5-b set
(its second stage-2 round, examined from 2026-10-07 at fork commit 94f6b2a23a; the first BF16 v3.5 set, code r4096 + 512, is
superseded and its row is in docs/67); the base and v3 arms of a model reuse earlier runs at earlier commits of this branch (section
"Commits of the reported runs" of `../README.md`); MMLU arms split into different `SE_SHARD` parts give a few dozen of the repeated
MMLU questions different seeds (counted by the report's seed check); the Flash-Next RULER 32K / 64K cells above the base come from the
Flash-Next base's empty replies (it ends its turn right after RULER's answer prefix; on the documents both arms answered both
Flash-Next sets are below the base at every length, twinstar docs/70 §70.2.3 and the model cards); the Lightning base was re-run at
RULER 128K alone on the DUET runs' documents on 2026-10-08 (its first 128K run shared a process with its 64K run, so its documents
carried other ids and one task's documents differed), and the 128K cell pairs all 1300 documents (twinstar docs/82 §82.4); the
GLM-5.3-Flash arms ran one TP-4 server per node in jobs of their own (base at fork 0a5dc6ade0, DUET at 091d24da6c), and its DUET
LiveCodeBench cell is the restart-free rerun of 2026-10-09 in 6 document shards per repetition (`DOC_SHARD`; checkout c948d22edf =
091d24da6c + the shard commit, on this branch as 7901c6b2ec), which replaced a run whose time-limit restarts favoured short replies.

## Tasks (GPQA, MMLU: simple-evals; LiveCodeBench, HumanEval+, LongBench v2, AA-LCR: official prompt and grader, run through lm-eval; MATH-500: lm-eval's data with a boxed-answer prompt and grader; AIME, IFEval, RULER: lm-eval 0.4.13 task bodies, generation settings replaced)

| exam | task(s) | data | prompt / grader | reps |
|---|---|---|---|---|
| gpqa_se | duet_gpqa_se (`se_utils.py`) | GPQA-Diamond, 198 q (`simple-evals/gpqa_diamond.csv`, pinned in `prepare_data.sh`); repetition k uses simple-evals' option permutation of repeat k (`random.Random(0)` over examples x repeats) | simple-evals multiple-choice prompt (last line `Answer: $LETTER`) and extraction (first `Answer: X`), copied verbatim from simple-evals 652c89d0 | 8 |
| aime24, aime25 | aime24, aime25 | 30 + 30 q (aime24 from HuggingFaceH4/aime_2024, gold answers restored to integers as in lm-eval's own source, `aime_utils.py`) | NeMo-Skills math prompt (box requested); boxed + lm-eval `is_equiv` | 8 |
| math500 | duet_math500 (`math500_utils.py`) | HuggingFaceH4/MATH-500 | NeMo-Skills math prompt; last boxed, math-verify (primary) + `is_equiv` | 8 |
| mmlu_se | duet_mmlu_se (`se_utils.py`) | MMLU test set, 14042 q (`simple-evals/mmlu_test.parquet`, cais/mmlu pinned); `SE_SHARD=i/n` splits it over jobs | simple-evals prompt; `normalize_response`, the multilingual answer patterns, `normalize_extracted_answer` (verbatim) | 1 |
| ifeval | ifeval | 541 prompts | prompt-level strict (primary) and loose; langdetect seeded (`run.py`) so the language checks grade a reply the same on every run | 8 |
| lcb | duet_lcb (`lcb_utils.py`) | LiveCodeBench v6, 454 problems (2024-08 .. 2025-04) | official LiveCodeBench prompt and system message, code extraction; official grader (`lcb_grade.py`, public + private tests) | 8 |
| ruler | 13 RULER tasks (`RULER_TASKS=all`, the default; `retrieval` = the 8 NIAH tasks), 100 docs per task and length | synthetic; offline sources from `prepare_data.sh` | greedy, 1024-token answers, thinking off; `ruler@<length>` = the mean over the run's tasks, RULER-NIAH (8 retrieval tasks) and RULER-other (cwe, fwe, vt, qa) beside it | 1 |
| humaneval_plus | duet_humaneval_plus (`evalplus_utils.py`) | HumanEval+ v0.1.10, 164 tasks (EvalPlus's release file, pinned in `prepare_data.sh`) | EvalPlus's chat prompt, one user message, no assistant prefill; graded by `humaneval_grade.py`: EvalPlus's sanitizer on the reply after the thinking block, `evalplus.evaluate` over both arms at once (base tests = `humaneval`, base + plus = `humaneval_plus`, the primary cell) | 8 |
| longbench_v2 | duet_longbench_v2 (`longbench_v2_utils.py`) | THUDM/LongBench-v2, 503 questions (data.json at a pinned revision) | the official zero-shot template; the official middle truncation to 120000 tokens (the official length for 128K-context models) with the base's own tokenizer; the official `extract_answer` on the reply after the thinking block; cells `longbench_v2` and `longbench_v2/<easy, hard, short, medium, long>` | 4 |
| aa_lcr (not in the protocol) | duet_aa_lcr (`aa_lcr_utils.py`) | AA-LCR v1.1, 100 questions over document sets of ~100K tokens (`aa_lcr_data.py`, pinned Hub revision) | official prompt; the official v1.1 equality-check prompts on a local judge (`aa_lcr_judge.py`, below) | 8 |

The HF exam sets (AIME 2024 mirror, AIME 2025, MATH-500, IFEval) load at the revisions of `tasks.HF_SETS`, written into every
generated task yaml, so a run with network access reads the same data as an offline one.  Generation ceiling = min(229376,
context - prompt - 8), so the context must stay at 262144; a prompt lm-eval would left-truncate is refused.  Vendor sampling:
Kimi T 1.0 (Kimi Linear paper; it gives no top_p / top_k: top_p 1.0, no top_k), no thinking mode; Flash-Next thinking xhigh, T 1.0 / 0.95 / 20 (model
card); Lightning thinking on (`enable_thinking`), T 1.0 / 0.95 / no top_k (model card); GLM-5.3-Flash `reasoning_effort` max (the
template always thinks; RULER's answer prefix closes the thinking block: `<think></think>Answer:`), T 1.0 / 0.95 / no top_k
(generation_config.json, model card).  Thinking is stripped at the first
`</think>`; a response that never closes it is graded empty and counted (`思考未闭合` in the table, "thinking not closed").
`python -m benchmark.duet_eval.lmeval.tasks --family <f> --out <dir>` writes the task directory of a family without running
anything (the job writes one per job, `<out>/tasks-<job>/`).

## Why these deviations from stock lm-eval

* GPQA and MMLU run as simple-evals runs them: lm-eval's CoT prompts (`gpqa_diamond_cot_zeroshot`, `mmlu_flan_cot_zeroshot`) state
  no answer format and its flexible-extract reads the LAST "(X)" of a reply, so a reply that states its answer and then discusses
  the other options was scored on another letter, on both arms (Flash-Next's GPQA came out 7-9 points lower on both arms than
  under simple-evals).  simple-evals' prompt asks for a final `Answer: X` line, as the published GPQA / MMLU numbers of chat models
  do.  Where this differs from simple-evals itself:
  * the GPQA-Diamond csv comes from the Hub dataset zai-org/glm-simple-evals-dataset (simple-evals' own copy on openaipublic was not
    reachable from the cluster); its row order against simple-evals' copy is unverified, and the option permutations depend on
    the row order;
  * MMLU comes from cais/mmlu.  10 test questions have the literal option text "None" (se_id 1006, 1090, 1104, 1117, 1156, 1429,
    1961, 1988, 2027, 2037); simple-evals reads its csv with `pandas.read_csv`, which would turn that text into NaN (shown as
    "nan").  We render "None"; which one upstream's own file and reader produce is unverified;
  * sampling is the vendor's setting with the 229376-token generation ceiling (above), not simple-evals' sampler.
* `repeats` grades only the first response -> one lm-eval process per repetition.
* HumanEval+ and LongBench v2: section "HumanEval+ and LongBench v2".
* lm-eval's `hendrycks_math500` grader takes the text between the first and last `$` -> visibly correct boxed answers scored
  0; `duet_math500` grades the last `\boxed{}` with math-verify (or lm-eval's `is_equiv`), the grader of current MATH-500 reports (and lm-eval's own
  `minerva_math` `math_verify` metric).  Completion-style math prompts ("Problem: ... Answer:") gave 20% unboxed answers on
  both arms -> the NeMo-Skills prompt.
* AIME 2024 is read from the HuggingFaceH4/aime_2024 mirror (offline cache), whose `answer` column is zero-padded ("025");
  `aime_utils.process_docs` restores the integer answers of lm-eval's own source (Maxwell-Jia/AIME_2024), so the task scores
  exactly as lm-eval's.  Runs before 2026-10-03 graded 7 of 30 correct answers as wrong on both arms;
  `../archive/regrade_aime24.py` rescored their sample files (originals kept as `.orig`).
* IFEval's language checks call langdetect, which samples unless seeded -> `run.py` seeds it (since 2026-10-03), so a reply is
  graded the same on every run; `../archive/regrade_ifeval.py` regrades older sample logs with the seeded checker.
* RULER's lm-eval output limits -> 1024 tokens for every RULER task (`vt`'s limit of 30 tokens scored Kimi 0 on both arms: it
  restates the prompt first).
* SGLang honoured per-request seeds only in deterministic-inference mode -> `sampling_batch_info` honours them whenever a
  request asks (and merges seeded with unseeded batches).

## HumanEval+ and LongBench v2

Both run as lm-eval processes of their own (`tasks.OWN_PROCESS`), like LiveCodeBench; `DOC_SHARD=i/n` runs documents i, i+n, ... of
every listed repetition through lm-eval `--samples`, so a document keeps its doc_id and its seed (the seed never depends on the other
documents of a process).  Where they differ from the official procedures, and why (the same on both arms; twinstar docs/90 has the
sources and commits):

* **Sampling and budget.**  The vendor's sampling and thinking setting of the family and the protocol's generation ceiling (min(229376,
  context - prompt)), instead of EvalPlus's greedy / 768-token default and LongBench v2's temperature 0.1 / 128-token answer: the
  official caps would cut a reasoning model's thinking.  LongBench v2's official leaderboard runs reasoning models on the zero-shot
  prompt and lets them reason natively (its "w/ CoT" column for reasoning models), which is what this does; the answer is extracted
  from the text after the thinking block.
* **HumanEval+ prompt.**  The request of EvalPlus's OpenAI-compatible backend (one user message: the instruction prefix and the task
  prompt in a python fence).  EvalPlus's vLLM / HF backends also prefill the assistant turn with a response prefix and an open code
  fence; a thinking template opens the thinking block there, so no prefill.
* **HumanEval+ grader.**  EvalPlus at 26d6d00bb1fd (the commit its README installs; the 4-second minimum time limit) with its default
  time limits; EvalPlus's per-checker address-space cap raised from 4 to 8 GiB (`EVALPLUS_MAX_MEMORY_BYTES`): the cap counts the
  4.44 GiB a checker inherits from `evaluate`'s own process in the exam container, so under the default the canonical solutions of
  HumanEval/15 and /130 fail on their million-element inputs.  HumanEval/32's canonical solution fails its own plus oracle at any
  cap (evalplus issue #253); it stays in the set, as EvalPlus scores it.  `no_answer` (shown as 无答案) = the sanitized solution
  does not define the entry point; EvalPlus still grades it (its sanitizer returns the reply's first line when it finds no code).
* **LongBench v2 empty replies.**  `pred.py` skips an empty reply (the item is retried on its next run); here it is scored as no
  answer, never dropped.

Reported results (twinstar docs/90; runs at 722f62f1ed; DUET score, paired difference to the base +- SE, all 8 / 4 repetitions;
reports `results/acc-sgl/v35-finals/reports/<model>-extra.{md,json}` of the twinstar repository):

| model | HumanEval+ | HumanEval | LongBench v2 |
|---|---|---|---|
| FN-BF16 v3.5 (v3.5-b, round 2) | 94.8 (+0.9 ± 0.4) | 98.8 (+0.9 ± 0.4) | 65.3 (+0.9 ± 0.9) |
| FN-NVFP4 v3.5 | 94.6 (+0.2 ± 0.4) | 98.5 (+0.5 ± 0.3) | 64.6 (-0.7 ± 0.9) |
| Kimi v3.5 | 78.3 (-0.4 ± 0.8) | 83.2 (+0.0 ± 0.7) | 34.6 (-0.1 ± 1.1) |
| Lightning v3.5 | 92.0 (+0.1 ± 0.6) | 98.2 (+1.1 ± 0.4) | 45.7 (-1.4 ± 1.1) |
| GLM-5.3-Flash v3.5 | 94.7 (-0.2 ± 0.3) | 99.5 (+0.1 ± 0.2) | 64.1 (+0.8 ± 0.8) |

## Long-context RULER (64K and up)

lm-eval builds 500 documents per RULER task and length and tokenises every one, and the documents of a task depend on how many
the tasks before it consumed from the global random state.  With `RULER_BUILD=N` (`run.py --ruler-build-samples`) lm-eval's own
generators build N documents per task and length, each task and length from lm-eval's seed state (1234), so a document depends
only on (task, tokenizer, length, N): runs that split the tasks over jobs, and a later larger N, get the same documents.
`RULER_CACHE` keeps every build on disk for the other arm and later runs; `ruler_build.py` builds them ahead of the exam on CPU and
reports the length of every prompt as the server receives it.  The protocol's 64K and 128K runs use `RULER_N=100 RULER_BUILD=100`.
* Beyond the protocol (256K / 512K / 1M): lengths are the power of two minus 512 tokens (261632 / 523776 / 1048064, given
  with `RULER_LENGTHS`): lm-eval budgets the document and the answer but not the chat template and the answer prefix, and a
  prompt must leave room for the answer inside the model's native context (Flash-Next 262144 positions, Kimi-Linear 1048576).
  The server context (`CTX`) must hold the length plus the answer, at most the native context; the RULER process sets lm-eval's
  `max_length` to the context + 1024 so lm-eval never cuts a prompt, and the generation budget comes from the server context.
  Kimi-Linear prefills a prompt unchunked (the DUET prefill needs the whole prompt in one extend): `MEM_FRACTION` leaves headroom
  for the activations and `WATCHDOG` (seconds) covers a single forward longer than SGLang's 300 s default; both arms use the same
  settings.  `../longctx_probe_job.sh` sizes such runs (`../long_prompt_probe.py`).
* Corpus caps (lm-eval's generators, unchanged): `qa_squad` uses every SQuAD 2.0 dev paragraph once the length exceeds them
  (~180K tokens), and the essay-haystack tasks stop at the Paul Graham essay corpus; their prompts are then shorter than the
  nominal length (the build report gives the real lengths).

## AA-LCR (`aa_lcr_data.py`, `aa_lcr_judge.py`; not part of the protocol)

The questions, document order and prompt are the dataset card's (v1.1).  The official score is an LLM equality check against the
official answer with the card's system and user prompts; the official checker is an external API model and is not used (no exam
data leaves the cluster) -- the one deviation.  `aa_lcr_judge_job.sh fn|kimi <out> <arm dirs>` runs the same prompts on a local
judge: `fn` = the Flash-Next NVFP4 base, thinking at medium effort (the official checker reasons at medium effort), the vendor's
sampling; `kimi` = the Kimi-Linear base, greedy.  Both arms get the same judge and settings and the judge never sees the arm; a
reply it cannot be read on counts as incorrect.  `report_lmeval.py --aa-lcr-judge fn` makes that judge the `aa_lcr` cell and
lists every judge as `aa_lcr/judge-<tag>`.  Its data: `AA_LCR=1 prepare_data.sh`.

## Reading the table

`score (delta ± SE, n=questions) · 全部 N 轮 s · 截断 x% · 思考未闭合 z% · 无答案 y% · tokens`: every arm's score is its mean over
the repetitions both arms completed (the baseline's over those it shares with the first paired arm), so the score and the delta
cover the same samples; an arm that has more repetitions (within the protocol's first N) adds `全部 N 轮 s` ("all N
repetitions"), its mean over all of them.  截断 ("truncated") = finished by length; 思考未闭合 ("thinking not closed");
无答案 ("no answer") = no answer extracted (no simple-evals `Answer: X`, no boxed answer, or no LiveCodeBench code); these and
the token mean (`<n> tok`) cover all of the arm's samples read; 思考未闭合 and 无答案 appear only when nonzero, and the baseline
column has no delta.  The md header names the columns 任务 (cell) and 题数 × 共同轮次 (questions x common repetitions).  `--out`
writes every number as JSON.

## Layout of an output directory

`<out>/<arm>/rep<k>/<gen|lcb|aa_lcr|humaneval_plus|longbench_v2|gpqa_se|mmlu_se>[-s<i>of<n>]/` and `<out>/<arm>-ruler[-<tag>]/rep0/ruler/` hold one lm-eval
process's `sidecar.jsonl` (request id, seed, occurrence, budget, finish reason, token counts, retries, the sha1 of the templated
prompt and of the graded reply (thinking removed), the raw text of every response, and `resumed_from` for a replayed reply), `run.json` (argv, fork commit) and `claim.json` (the process that owns the directory; a second process on it is
refused), and lm-eval's `<model-path>/` subdirectory (the tokenizer path with `/` written as `__`) with its `samples_*.jsonl` and
`results_*.json`; runs before 2026-10-02 wrote these directly into `rep<k>/`.  A job's servers: `<out>/servers-<job>/<arm>-<i>/`
(`launch.json`, `server.log`); its task directory: `<out>/tasks-<job>/`.  `<out>/partial/` holds repetitions a time limit
interrupted (below).

## Resuming after a time limit (`RESUME=1`)

lm-eval writes a repetition's samples only when it ends, so a job killed at its time limit leaves the repetition's
`sidecar.jsonl` (every reply received so far) and no sample file.  Rerunning the same job line with `RESUME=1` moves every such
directory of the job -- `<out>/<arm>/rep<k>/<kind>` for the job's repetitions and exam kinds, and `<out>/<arm>-ruler[-<tag>]/rep0/ruler`
when the job runs RULER -- to `<out>/partial/<arm dir>/rep<k>/<kind>-<n>/`; the arm's balancer (`lb.py --resume`) answers every
request whose prompt, occurrence, sampling seed and budget match a recorded reply with that reply (`meta_info.duet_resumed` = the
original request id, written as `resumed_from` into the new sidecar), and the servers generate only the missing replies.  A
replayed reply is the sample the interrupted run drew; a regenerated one is a fresh draw with the same seed (not token
identical, kernels are not batch-invariant).  The regenerated replies are the ones the time limit cut, usually the longest:
report their number and outcome with the cell (sidecar records without `resumed_from`).  A directory with a sample file is never
moved; a directory of another job's unit (another RULER tag, a repetition outside `REP_LIST`) is not touched.  On Slurm, a resume
job can be queued ahead with `--dependency=afternotok:<job>`.

## Diagnostic tools

`forced_pairs.py` (position-wise teacher-forced NLL gap between served arms on the base arm's own generations; the scorer is
`../forced_logits.py`), `probe_opener.py` (first-token distribution under LiveCodeBench prompt variants).  Their job scripts
(`diag_forced_sweep_job.sh`, `diag_forced_fn_job.sh`, `probe_opener_job.sh`), `diagnose.py` (where the paired texts part) and
`probe_fn_tp_job.sh` (Flash-Next TP1 vs TP2) were one-off diagnostics of 2026-09-30 to 2026-10-03 and are in `../archive/`
(`../archive/README.md`): their paths are environment variables listed at the top of each, with the owner's cluster as default;
their release defaults are the v3 sets the recorded diagnostics ran on.

## Implementation notes

This harness and the served DUET path are the accuracy reference, not optimized for speed; no timing of them is a result (the
twinstar paper states the sets' performance as roofline counts).  The DUET decode-time work is the state cut (fp64 CholeskyQR2 and
a small eigh per (layer, slot) row; `benchmark/duet_eval/bench_state_cut.py --family kimi` profiles its operations).  Cuts are
batched across all due layers of a step (`sglang.srt.duet.dense_state_policy`, `DenseStatePolicy._cut_many`), two host syncs per
step.  Memory: the cut keeps one warm basis ((H, N, r) fp32) per (layer, slot) until SGLang reuses the slot; before fork
028001518c the basis was a view of the whole cut chunk's tensor, so every finished request pinned up to a (256, H, N, r) block,
and a node serving one repetition (no queue, so no slot reuse) lost 2 of 4 Lightning DUET servers to an out-of-memory error in the
cut's batched eigh.  The eager DUET prefill compiles each new shape with Triton on first use (`TRITON_CACHE` reuses an earlier
job's compiled kernels).
