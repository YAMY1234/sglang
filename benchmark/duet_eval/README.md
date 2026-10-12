# SEED (code name DUET) on this SGLang fork: serving, exams, parity

This branch serves the released SEED component sets and runs their paired exams against the unchanged base models.  SEED is the
method's name in the paper; the code, the release files (`spec.json`, `manifest.json`) and this directory call it DUET.
A DUET set adds parameters beside an unchanged base checkpoint and changes two things about a request's memory: the prompt runs
only the first k layers, the residual entering layer k is stored as a compact latent code, and trained *emitters* write the deeper
layers' prompt memories (recurrent states and attention caches) from the decoded latent; every recurrent state is kept as an
explicit sink term plus rank-r factors, re-pruned every W decode steps.  Attention K/V entries are kept exact for every token
(the emitters write the prompt tokens' entries above the cut, the layers themselves the generated tokens'); every generated token
runs all layers; no base weight is changed.  (The GLM-5.3-Flash set has no code: its emitters read the exact residual.)

This fork is the accuracy reference of the released sets: it serves both arms of the paired exams and of the parity checks with
the arithmetic the components were trained with, and it is not optimized for speed.  No timing of it is a result; the sets'
performance is stated as roofline counts (memory bytes, FLOPs, bytes per decode step) in the paper.

| what | where |
|---|---|
| run a released set from the Hugging Face Hub: download, serve, exams, report (all five sets) | section "Evaluate a released DUET set from the Hub" |
| install, container | sections "Install" and "Container" |
| serve a release (either arm), every launcher flag | `launch_duet.py` (this directory), section "Serve a release" |
| the exams (protocol, commands per model, report, reported results) | `lmeval/README.md` |
| served-vs-reference parity | `validate_fn_parity_job.sh`, `validate_hooked_parity_job.sh`, section "Parity check" |
| families, release resolution, arm check, environment | `python/sglang/srt/duet/README.md` |
| the Flash-Next model package | `flash_next/README.md` |

## Evaluate a released DUET set from the Hub

The five released sets are Hugging Face model repositories of the organization `Latent-Voyager`, each named after its original
model with the suffix `-SEED`.  A repository holds `duet_components.safetensors`, `spec.json`, `manifest.json`,
`training_history.json` and the model card; the serving code reads the first three.  The manifest's `name` is the set's name
inside the code (it differs from the repository name; nothing compares the two); the set's identity is the component file's
sha256 (`manifest.json` `sha256`).  Each set serves only on the unchanged base checkpoint it was trained on, at the revision the
manifest names (`manifest.json` `base`):

| set | Hub repository | `manifest.json` `name` | component sha256 | base (Hub id @ training revision) | family | serving |
|---|---|---|---|---|---|---|
| Flash-Next BF16 | `Latent-Voyager/Qwen3.8-Flash-Next-SEED` | `Qwen3.8-Flash-Next-BF16-DUET-v3.5` | `9a6654873a94...` | `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540` | `flash_next` | TP 2 |
| Flash-Next NVFP4 | `Latent-Voyager/Qwen3.8-Flash-Next-NVFP4-SEED` | `Qwen3.8-Flash-Next-NVFP4-DUET-v3.5` | `18a4ab8600b4...` | `nvidia/Qwen3.8-Flash-Next-NVFP4@fc694b54fb0174e0913e6adf86691ef85a4ead47` (the manifest records no revision: give it) | `flash_next` | TP 1 |
| Kimi-Linear | `Latent-Voyager/Kimi-Linear-48B-A3B-Instruct-SEED` | `Kimi-Linear-48B-A3B-Instruct-DUET-v3.5` | `8de5bd44c1af...` | `moonshotai/Kimi-Linear-48B-A3B-Instruct@e1df551a447157d4658b573f9a695d57658590e9` | `kimi` | TP 1 |
| Nemotron-3.5-Lightning | `Latent-Voyager/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-SEED` | `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-DUET-v3.5` | `0a8bca47f169...` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16@a9904d24bcc1d289a1950fa9d2b978c47cf903b9` | `lightning` | TP 1 |
| GLM-5.3-Flash | `Latent-Voyager/GLM-5.3-Flash-SEED` | `GLM-5.3-Flash-DUET-v3.5` | `1950555fc334...` | `zai-org/GLM-5.3-Flash@eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` (the FP8 release) | `glm` | TP 4 |

Hardware of the reported runs: nodes of 4 GB300 GPUs (277 GiB each), arm64, CUDA 13.  `launch_duet.py` serves Kimi-Linear and
Lightning at TP 1, Flash-Next at the given `--tp` and GLM-5.3-Flash at TP 4; each server needs GPUs that hold its share of the base
weights plus a KV cache for the 262144-token context of the exams (Flash-Next BF16 has 335 GiB of weights, hence TP 2).

What is checked on the way, so that a set is never served on another base or from altered files:

* `launch_duet.py --duet-release <owner>/<repo>[@<revision>]` and `fetch_duet.py --release ...` resolve the revision to a commit and
  download exactly `spec.json`, `manifest.json` and `duet_components.safetensors` of that commit, as regular files, into
  `$HF_HOME/duet-releases/<owner>--<repo>/<commit>`.  The component file must have the sha256 the Hub records for it (its LFS
  metadata) and the manifest's sha256 and size, and `spec.json` must equal the manifest's spec (`python/sglang/srt/duet/release.py`
  `fetch_release`, `verify_release`).  `--duet-release-sha256` / `--release-sha256` adds a pin of your own: the component sha256
  of the table, or a prefix of at least 12 hex digits.
* A base given as a Hub id (`launch_duet.py --model <owner>/<repo>[@<revision>]`, `fetch_duet.py --base`) is fetched into the
  standard cache `$HF_HOME/hub` only after its repository equals the manifest's `base.hf_id`, its revision resolves to
  `base.revision` (the default when no revision is given) and its `config.json` hashes to `base.config_sha256`
  (`release.fetch_base`).  Every launch checks the base directory against the manifest again (`release.check_base`), so a local
  base directory gets the same check.
* Before it reports READY, `launch_duet.py` checks the arm in `server.log`: the DUET arm loaded the component file with the
  release's sha256 and serves the release's cut (prefill depth k, state rank r, cadence W); the base arm loaded no DUET component.

The five repositories are private.  The account must have read access to them; give its token with `hf auth login` (stored as
`$HF_HOME/token`) or `HF_TOKEN` in the environment.  `huggingface_hub` reads the token itself: no script here reads, prints or
records it.  The base repositories of Kimi-Linear, Lightning, Flash-Next and GLM may ask to accept their license on the Hub first.
A missing or wrong token stops the download with "needs access on the Hub and a login".

```bash
# 0. once per machine: this branch, the exam packages, the exam data (with network); inside the image of section "Container"
#    /opt/duet-exams already holds the exam packages
git clone --branch <this branch> <this repository> seed && cd seed
pip install --no-deps --target /opt/duet-exams -r benchmark/duet_eval/lmeval/requirements-exams.txt
export PYTHONPATH=$PWD/python:$PWD:/opt/duet-exams HF_HOME=<hf-cache>
bash benchmark/duet_eval/lmeval/prepare_data.sh <exam-data>        # ~7 GB; checked against the reported copies' sha256
hf auth login                                                      # or: export HF_TOKEN=<token>

# 1. download and check one release and its base (the last line printed sets DUET_REL and the family's base variable)
eval "$(python -m benchmark.duet_eval.fetch_duet --release Latent-Voyager/Kimi-Linear-48B-A3B-Instruct-SEED \
          --release-sha256 8de5bd44c1af --base | tail -n 1)"                                        # -> DUET_REL, KIMI_BASE
#   Flash-Next BF16:  --release Latent-Voyager/Qwen3.8-Flash-Next-SEED --release-sha256 9a6654873a94 --base       -> FN_BASE
#   Flash-Next NVFP4: --release Latent-Voyager/Qwen3.8-Flash-Next-NVFP4-SEED --release-sha256 18a4ab8600b4 \
#                     --base nvidia/Qwen3.8-Flash-Next-NVFP4@fc694b54fb0174e0913e6adf86691ef85a4ead47           -> FN_BASE
#   Lightning:        --release Latent-Voyager/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-SEED --release-sha256 0a8bca47f169 --base
#                                                                                                                 -> LIGHTNING_BASE
#   GLM-5.3-Flash:    --release Latent-Voyager/GLM-5.3-Flash-SEED --release-sha256 1950555fc334 --base            -> GLM_BASE
#   (<repo>@<commit> pins a revision of the release; --base --config-only checks the base's identity without the download)

# 2. serve one arm by hand (optional: the exam jobs below start their own servers); --detach returns after READY
L="python -m benchmark.duet_eval.launch_duet --context-length 262144 --max-running-requests 128 --sampling-backend pytorch"
$L --model "$KIMI_BASE" --duet-release "$DUET_REL" --arm duet  --gpus 0 --port 30001 --out runs/kimi-duet  --detach
$L --model "$KIMI_BASE" --duet-release "$DUET_REL" --arm stock --gpus 1 --port 30002 --out runs/kimi-stock --detach
#   Flash-Next BF16: --model "$FN_BASE" ... --tp 2 --gpus 0,1      Flash-Next NVFP4: --model "$FN_BASE" ... --tp 1 --gpus 0
#   Lightning:       --model "$LIGHTNING_BASE" ... --gpus 0        GLM: --model "$GLM_BASE" ... --gpus 0,1,2,3 --max-running-requests 256
#   --model and --duet-release also take the Hub ids directly, e.g.
#   --model moonshotai/Kimi-Linear-48B-A3B-Instruct --duet-release Latent-Voyager/Kimi-Linear-48B-A3B-Instruct-SEED
#   (the base is then downloaded into $HF_HOME/hub at the first launch; the NVFP4 base is given with its revision:
#   --model nvidia/Qwen3.8-Flash-Next-NVFP4@fc694b54fb0174e0913e6adf86691ef85a4ead47)
curl -s http://127.0.0.1:30001/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model": "default", "messages": [{"role": "user", "content": "What is 17 * 23?"}], "max_tokens": 4096}'
kill -- -$(cat runs/kimi-duet/server.pid) -$(cat runs/kimi-stock/server.pid)       # stop both (each runs in its own session)

# 3. the paired exams, offline: one job per line on one 4-GPU node (lmeval/README.md "Run the protocol for one release" has
#    every family's node layout and the RULER 64K / 128K lines)
export DUET_DATA_ROOT=<exam-data> HF_OFFLINE=1 EXTRA_PYTHONPATH=/opt/duet-exams NDUET=2 NSTOCK=2   # DUET_REL, KIMI_BASE: step 1
J=benchmark/duet_eval/lmeval/exam_lmeval_job.sh; F=kimi; O=<out>/kimi-v3.5
for k in 0 1 2 3 4 5 6 7; do
  REP_LIST=$k bash $J $F $O aime24,aime25,math500,ifeval
  REP_LIST=$k bash $J $F $O gpqa_se
  REP_LIST=$k bash $J $F $O lcb
  REP_LIST=$k bash $J $F $O humaneval_plus
done
for k in 0 1 2 3; do REP_LIST=$k bash $J $F $O longbench_v2; done
SE_SHARD=0/2 bash $J $F $O mmlu_se; SE_SHARD=1/2 bash $J $F $O mmlu_se
RULER_LENGTHS=32768 RULER_TAG=32k bash $J $F $O ruler

# 4. HumanEval+ grading over both arms at once (EvalPlus runs model-written code: in the container of a compute node)
pip install --no-deps --target /opt/evalplus -r benchmark/duet_eval/lmeval/requirements-evalplus.txt
PYTHONPATH=$PYTHONPATH:/opt/evalplus python -m benchmark.duet_eval.lmeval.humaneval_grade --out $O/humaneval-grade --passes 2 \
    $O/stock $O/duet

# 5. the paired report: every cell, the prompt check and the seed check (both must report 0 differences)
python -m benchmark.duet_eval.lmeval.report_lmeval --arms stock=$O/stock duet=$O/duet \
    --ruler $(for d in $O/stock-ruler*; do echo stock=$d; done) $(for d in $O/duet-ruler*; do echo duet=$d; done) --md --out $O/report.json
```

The exam job serves both arms itself (NDUET DUET servers and NSTOCK base servers on one node, identical server arguments except the
DUET enablement), so step 2 is only for a look at the replies.  A job stopped by a time limit is finished by rerunning its line with
`RESUME=1`; LiveCodeBench, HumanEval+ and LongBench v2 can be split over nodes with `DOC_SHARD=i/n` (`lmeval/README.md`).  How to
read the table (`score (delta ± SE, n=questions)`, truncation and no-answer rates, the Chinese column labels):
`lmeval/README.md`, "Reading the table"; the reported numbers to compare with: `lmeval/README.md`, "Reported v3.5 results" and
"HumanEval+ and LongBench v2".  A rerun at temperature 1.0 differs question by question; its paired differences should agree with
the reported ones within their SE.

The protocol these lines run, as the harness code fixes it (no flag above changes it):

| item | value | where |
|---|---|---|
| exams and repetitions | AIME 2024 + 2025, GPQA Diamond (simple-evals), MATH-500, LiveCodeBench v6 (454 problems), IFEval, HumanEval+ x8; LongBench v2 x4; MMLU test split (simple-evals, 14,042) x1; RULER 32K / 64K / 128K (13 tasks x 100 documents) x1 | `lmeval/tasks.py` `EXAMS`, `P0` |
| sampling | Flash-Next: T 1.0, top-p 0.95, top-k 20, min-p 0, thinking on with `reasoning_effort` xhigh; Lightning: T 1.0, top-p 0.95, thinking on; Kimi-Linear: T 1.0, top-p 1.0 (no thinking mode); GLM: T 1.0, top-p 0.95, `reasoning_effort` max; RULER: greedy, at most 1024 output tokens, thinking off | `lmeval/tasks.py` `FAMILIES`, `EXAMS["ruler"]` |
| seeds | pair seed 20260930; a request's seed = hash(pair seed + repetition, prompt tokens, occurrence of the prompt in the process), the same on both arms; PyTorch sampler (`--sampling-backend pytorch`) | `exam_lmeval_job.sh` `PAIR_SEED`, `SAMPLING_BACKEND`; `lmeval/README.md` "What is compared" |
| budgets | context 262144 (Kimi-Linear RULER 64K / 128K: 270336); generation ceiling 229,376 tokens, clamped per request to context - prompt | `exam_lmeval_job.sh` `CTX`; `lmeval/tasks.py` `MAX_GEN_TOKS` |
| servers | identical server arguments on both arms except the DUET enablement; 128 running requests per server (GLM 256); radix cache off; decode CUDA graph | `launch_duet.py`, `exam_lmeval_job.sh` `MRR` |
| graders | lm-eval 0.4.13 task bodies (AIME integer gold, MATH-500 boxed answer + math-verify, IFEval prompt-level strict with langdetect seeded), simple-evals extraction for GPQA / MMLU, the official LiveCodeBench grader, EvalPlus for HumanEval+, the official LongBench v2 answer extraction | `lmeval/tasks.py` docstring, `lmeval/README.md` "Tasks" |

## Install

* **Image.** The reported runs used the SGLang image `lmsysorg/sglang:nightly-dev-cu13-20260913-fa663e72` (its arm64 build; built
  from fa663e72, the upstream commit this branch starts from), with this repository's `python/` first on `PYTHONPATH`; the job
  scripts set `PYTHONPATH=<repository>/python:<repository>` themselves and import nothing pip-installed from this checkout.  The
  same image with the exam packages: section "Container".
* **Exam packages** (lm-eval 0.4.13, flash-linear-attention 0.5.2 for the Kimi-Linear emitters, the graders' packages and their
  dependencies, at the versions of the reported runs): `lmeval/requirements-exams.txt`, installed beside the image's packages with
  `pip install --no-deps --target <dir> -r benchmark/duet_eval/lmeval/requirements-exams.txt`; the exam jobs take the directory as
  `EXTRA_PYTHONPATH=<dir>` (empty when the packages are importable already), the parity jobs as `PY_SITE=<dir>`.  The HumanEval+
  grader's packages (EvalPlus at 26d6d00bb1fd, tree-sitter): `lmeval/requirements-evalplus.txt`, in a directory of their own.
* **Exam data** (RULER sources, LiveCodeBench v6, the simple-evals GPQA / MMLU files, HumanEval+ v0.1.10, LongBench v2, the HF exam
  sets at the revisions the task yamls pin, NLTK punkt; downloads are checked against the reported copies' sha256), once, on a
  machine with internet: `bash benchmark/duet_eval/lmeval/prepare_data.sh <exam-data>` (from the repository root, with the exam
  packages importable), then `DUET_DATA_ROOT=<exam-data>` for the exams.
* **Paths.**  Every path the job scripts read is an environment variable listed at the top of the script; the defaults after `=`
  there are the paths of the cluster the reported runs used and do not exist elsewhere.  Set at least `HF_HOME`, `DUET_DATA_ROOT`,
  `EXTRA_PYTHONPATH`, `DUET_REL` and the family's base variable (`FN_BASE`, `KIMI_BASE`, `LIGHTNING_BASE`, `GLM_BASE`).
* **Base checkpoints**, unchanged, at the revisions of the table above (`launch_duet.py` refuses a base whose `config.json` sha256
  differs from the manifest's).  `fetch_duet.py --base` downloads and checks them; by hand:

```bash
hf download Qwen/Qwen3.8-Flash-Next --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 --local-dir <models>/Qwen3.8-Flash-Next
hf download nvidia/Qwen3.8-Flash-Next-NVFP4 --revision fc694b54fb0174e0913e6adf86691ef85a4ead47 --local-dir <models>/Qwen3.8-Flash-Next-NVFP4
hf download moonshotai/Kimi-Linear-48B-A3B-Instruct --revision e1df551a447157d4658b573f9a695d57658590e9 --local-dir <models>/Kimi-Linear-48B-A3B-Instruct
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9 --local-dir <models>/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
hf download zai-org/GLM-5.3-Flash --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a --local-dir <models>/GLM-5.3-Flash
```

  The NVFP4 release's manifest records no base revision (`base.revision` is null: there was no download record beside the
  checkpoint); the checkpoint the set was trained and examined on is that Hub revision, and the manifest's `config_sha256` is
  checked.
* **Releases.**  A release is a directory with `spec.json`, `manifest.json` and `duet_components.safetensors`; `fetch_duet.py`
  (above) puts it into `$HF_HOME/duet-releases/<owner>--<repo>/<commit>`.  `hf download Latent-Voyager/<repo> --local-dir <dir>`
  gives a directory the launcher takes as well (its files are verified at every launch).  An exam job starts several servers
  at once and runs offline (`HF_OFFLINE=1`), so give it the directory, not the Hub id.

## Container

The image of the reported runs is the SGLang image above plus the exam packages in `/opt/duet-exams` (not on `PYTHONPATH` by
default: the exam jobs take `EXTRA_PYTHONPATH=/opt/duet-exams`, the parity jobs `PY_SITE=/opt/duet-exams`).  It is published with
the twinstar release as an enroot / pyxis squashfs (`duet-v3.5.sqsh`) with its `Dockerfile` (the twinstar repository's
`release/container/README.md`).  The equivalent Docker build from this checkout (arm64; the nightly base tag may have been pruned
from Docker Hub since, in which case convert the squashfs):

```dockerfile
FROM lmsysorg/sglang:nightly-dev-cu13-20260913-fa663e72
COPY benchmark/duet_eval/lmeval/requirements-exams.txt /tmp/requirements-exams.txt
RUN python3 -m pip install --no-cache-dir --no-deps --no-build-isolation --target /opt/duet-exams -r /tmp/requirements-exams.txt
```

Code, data, base checkpoints, releases and outputs are mounted; the examples mount this checkout at `/fork`, the exam data and HF
cache under `/data`, the bases under `/models`, the releases under `/releases` and the outputs under `/out`
(`<checkout>`, `<data>`, ... are directories of the host):

```bash
# Slurm + pyxis (one 4-GPU node per job):
C="srun -N1 --gpus-per-node=4 --container-image=<image dir>/duet-v3.5.sqsh --container-workdir=/fork
   --container-mounts=<checkout>:/fork,<data>:/data,<models>:/models,<releases>:/releases,<out>:/out"
# docker:  C="docker run --rm --gpus all --ipc=host -w /fork -v <checkout>:/fork -v <data>:/data -v <models>:/models
#             -v <releases>:/releases -v <out>:/out <image>"
E="env DUET_DATA_ROOT=/data/duet-exams HF_HOME=/data/hf HF_OFFLINE=1 EXTRA_PYTHONPATH=/opt/duet-exams"

$C env PYTHONPATH=/opt/duet-exams bash benchmark/duet_eval/lmeval/prepare_data.sh /data/duet-exams      # once, with network
# one exam job (every job line of the protocol: lmeval/README.md "Run the protocol for one release")
$C $E FN_BASE=/models/Qwen3.8-Flash-Next DUET_REL=/releases/duet-flash-next-v3.5-b-r2 TP=2 NDUET=1 NSTOCK=1 REP_LIST=0 \
   bash benchmark/duet_eval/lmeval/exam_lmeval_job.sh flash_next /out/fn-bf16-v3.5 aime24,aime25,math500,ifeval
# the paired report, once every job of the release is done (CPU)
$C python -m benchmark.duet_eval.lmeval.report_lmeval --arms stock=/out/fn-bf16-v3.5/stock duet=/out/fn-bf16-v3.5/duet --md
```

The parity check (section "Parity check") also needs the reference implementation, the twinstar repository, and the model's
held-out token stream `eval/web/tokens.bin` (section "Parity check": rebuilt from public corpora by the twinstar recipe):

```bash
git clone --branch <twinstar branch> <twinstar repository> <twinstar>                                       # the reference
P="--container-mounts=<checkout>:/fork,<twinstar>:/twinstar,<data>:/data,<models>:/models,<releases>:/releases,<out>:/out"
R="env HF_HOME=/data/hf PY_SITE=/opt/duet-exams FORK=/fork REFERENCE_ROOT=/twinstar REPO_ROOT=/twinstar"
srun -N1 --gpus-per-node=4 --container-image=<image dir>/duet-v3.5.sqsh --container-workdir=/fork $P \
  $R MANIFEST=windows TOK=/data/tok-Flash-Next/eval/web/tokens.bin \
  bash benchmark/duet_eval/validate_fn_parity_job.sh /out/fn-bf16-v3.5-parity /models/Qwen3.8-Flash-Next /releases/duet-flash-next-v3.5-b-r2 2
srun -N1 --gpus-per-node=4 --container-image=<image dir>/duet-v3.5.sqsh --container-workdir=/fork $P \
  $R TOK=/data/tok-Kimi-Linear/eval/web/tokens.bin \
  bash benchmark/duet_eval/validate_hooked_parity_job.sh kimi /out/kimi-v3.5-parity /models/Kimi-Linear-48B-A3B-Instruct /releases/duet-kimi-v3.5
```

What the exam and parity jobs need from the image beyond the SGLang base image (the requirements the container is built to):

* the packages of `lmeval/requirements-exams.txt` at those versions, installed with `pip install --no-deps --target /opt/duet-exams`
  so that the base image's own packages stay as they were; the jobs put that directory on `PYTHONPATH` after this repository
  (`EXTRA_PYTHONPATH=/opt/duet-exams` for the exam job, `PY_SITE=/opt/duet-exams` for the parity jobs); the job's dependency check
  asserts lm-eval 0.4.13;
* from the base image unchanged: python 3.12, torch, the SGLang dependencies, transformers, datasets, pyarrow, numpy, aiohttp,
  requests, tqdm, pyyaml, huggingface_hub (with the `hf` command), safetensors, tiktoken; the reference implementation needs
  nothing beyond these and flash-linear-attention;
* command-line tools the jobs call: bash, curl, git (to record the fork commit), sha256sum, setsid, pkill, nvidia-smi;
* no DUET code, base model, release or exam data in the image: this checkout is mounted, and nothing is pip-installed from it
  (the jobs import `python/` from the checkout);
* shared memory for SGLang's processes (docker: `--ipc=host` or a large `--shm-size`); writable `<out>` (the jobs put `TMPDIR`
  and the Triton / torch-extension / flashinfer caches under it); no network during the exams (`HF_OFFLINE=1`).

## The v3.5 releases

| model | family | base (Hub id, revision) | DUET release (Hub) | twinstar release directory | component sha256 | serving | node layout of the v3.5 exam runs (DUET arm) |
|---|---|---|---|---|---|---|---|
| Flash-Next BF16 | `flash_next` | `Qwen/Qwen3.8-Flash-Next` de4b8e4d43b9 | `Latent-Voyager/Qwen3.8-Flash-Next-SEED` | `duet-flash-next-v3.5-b-r2` | `9a6654873a94...` | TP 2 (335 GiB of weights) | 2 DUET servers per node |
| Flash-Next NVFP4 | `flash_next` | `nvidia/Qwen3.8-Flash-Next-NVFP4` fc694b54fb01 | `Latent-Voyager/Qwen3.8-Flash-Next-NVFP4-SEED` | `duet-flash-next-nvfp4-v3.5` | `18a4ab8600b4...` | TP 1 | 4 DUET servers per node |
| Kimi-Linear-48B-A3B-Instruct | `kimi` | `moonshotai/Kimi-Linear-48B-A3B-Instruct` e1df551a4471 | `Latent-Voyager/Kimi-Linear-48B-A3B-Instruct-SEED` | `duet-kimi-v3.5` | `8de5bd44c1af...` | TP 1 | 4 DUET servers per node |
| Nemotron-3.5-Lightning-30B-A3B | `lightning` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` a9904d24bcc1 | `Latent-Voyager/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-SEED` | `duet-lightning-v3.5` | `0a8bca47f169...` | TP 1 | 4 DUET servers per node (RULER 32K: 2 DUET + 2 base) |
| GLM-5.3-Flash | `glm` | `zai-org/GLM-5.3-Flash` eb9eb208eb0d | `Latent-Voyager/GLM-5.3-Flash-SEED` | `duet-glm53-flash-v3.5` (a byte-identical copy of the export the exams served, `archive/duet-glm53-flash-v3.5-r2`) | `1950555fc334...` | TP 4 (the FP8 release) | 1 DUET server per node (the base arm in jobs of its own) |

The component sha256 is the release manifest's (`manifest.json` `sha256`); the DUET arm's `server.log` reports it in its
`DUET served cut: k=.. trim=.. r=.. W=.. sha256=..` line and `launch_duet.py` stops the server when it differs.  The examples
below that use local directories name them as the twinstar repository's `release/` does (`/releases/<twinstar release directory>`).

* **Flash-Next BF16** is released as the **v3.5-b** set (code r8192 + 256, the NVFP4 set's recipe on the BF16 base), its second
  stage-2 round (sha256 `9a6654873a94...`; the manifest was re-exported under its release name on 2026-10-08 with the component
  file byte-identical, twinstar docs/82): examined at fork commit 94f6b2a23a (DUET arm only, with the base arm of the earlier
  FN-BF16 runs; parity passed, section "Parity check").  Round 1 (sha256 `e7371d34d099...`) passed its gates and was not
  published; the first BF16 v3.5 set (code r4096 + 512, sha256 `8b3485838b81...`) is superseded and not published.
* The reported Lightning v3.5 exams served the same file from its earlier directory name `duet-lightning-v3c` (same sha256); its
  `spec.json` keeps the run name `duet-l35-k33-r2048-m128-v3c`.
* The base arms ran in earlier jobs with the same settings and seeds.  A job that serves both arms puts 1 + 1 servers
  (Flash-Next BF16) or 2 + 2 servers (the others) on a node (`lmeval/README.md`).

## Serve a release

`launch_duet.py` starts either arm of any family with identical server settings except the DUET enablement and checks the arm in
`server.log` before it reports READY: `--arm duet` serves the release, `--arm stock` the unchanged base.  `--model` and
`--duet-release` take a local directory or a Hub id (section "Evaluate a released DUET set from the Hub").  The exam settings,
with local directories:

```bash
L="python -m benchmark.duet_eval.launch_duet --context-length 262144 --max-running-requests 128 --sampling-backend pytorch"
$L --model /models/Qwen3.8-Flash-Next        --duet-release /releases/duet-flash-next-v3.5-b-r2  --arm duet --tp 2 --gpus 0,1 --port 30001
$L --model /models/Qwen3.8-Flash-Next-NVFP4  --duet-release /releases/duet-flash-next-nvfp4-v3.5 --arm duet --tp 1 --gpus 0   --port 30001
$L --model /models/Kimi-Linear-48B-A3B-Instruct --duet-release /releases/duet-kimi-v3.5          --arm duet --gpus 0 --port 30001
$L --model /models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --duet-release /releases/duet-lightning-v3.5 --arm duet --gpus 0 --port 30001
python -m benchmark.duet_eval.launch_duet --model /models/GLM-5.3-Flash --duet-release /releases/duet-glm53-flash-v3.5 --arm duet \
    --gpus 0,1,2,3 --port 30001 --context-length 262144 --max-running-requests 256 --sampling-backend pytorch
```

The same line with `--arm stock` serves the base arm (the release is still named: the launcher checks that the base is the
release's base and that the stock arm loads no DUET component; GLM's base arm may omit it).  `--dry-run` resolves and verifies
the release, checks the base, writes `launch.json` (command, the DUET-relevant environment, release identity) and prints the
command without starting anything; `--out DIR` holds `launch.json`, `server.log` and `server.pid` (default
`./duet_eval_out/<family>-<arm>-<port>`); `--detach` returns after READY.  The server is a normal SGLang HTTP server (`/generate`,
`/v1/chat/completions`); `--extra` takes only the ablation switches `--no-prefill-layer-trim`, `--decode-ssm-r N`,
`--decode-ssm-w N`.  What each family's launch does (the Kimi launcher, `sglang.launch_server --duet-release ... --duet-numerics
reference` for Lightning and GLM, the config view of the Flash-Next package) and the environment variables the server reads:
`python/sglang/srt/duet/README.md`.  `launch_duet.py` drops inherited `SGLANG_DUET_*`, `TWINSTAR_*`, `SGLANG_FLASHNEXT_*`,
`SGLANG_QWEN4_*` variables and `SGLANG_EXTERNAL_MODEL_PACKAGE`; other knobs of the Flash-Next serving work on this branch
(`SGLANG_GDN_*`, `SGLANG_QSA_*`, `SGLANG_PLE_*`, `SGLANG_P287_*`, `SGLANG_P886_*`, `SGLANG_PREFILL_GRAPH_*`) reach both arms
unchanged: the reported runs set none of them (`launch.json` records the environment), so leave them unset.

## GLM-5.3-Flash

`zai-org/GLM-5.3-Flash` at revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` (the FP8 release, 310 GB; 45 layers: 34 KDA + 11
DSA layers of MLA without RoPE with a lightning indexer, mHC residual, 288-expert MoE; vision tower loaded, unused).  Family `glm`
in `launch_duet.py` and `lmeval/tasks.py`; served by SGLang's own `models/glm5_next.py`.  The base arm runs without a release
(`--duet-release` may be omitted for `--arm stock`; given, it is verified and the base checked against it as for the other
families).  The DUET arm adds only `--duet-release <release dir> --duet-numerics reference` (`glm_duet_cli`): the release's spec
names the `glm5-next` family, and `python/sglang/srt/models/glm_duet` (twinstar docs/79) replaces `Glm5NextForConditionalGeneration`
-- the prompt runs layers 0..k-1, the 4 x 4096 mHC residual entering layer k goes through the release's residual code (the
released set has none: rank 0, the emitters read the exact residual), the emitters write every deeper
layer's KDA states and DSA latents / pooled indexer keys, the last prompt token runs as decode step 1, the KDA states are cut to
explicit sink + rank r at the prompt end and every W decode steps.  Chunked prefill, overlap scheduling, the decode CUDA graph and
every flag below stay the base arm's.  Served-vs-reference parity: `validate_glm_parity_job.sh` (one arm per
4-GPU node).

```bash
hf download zai-org/GLM-5.3-Flash --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a --local-dir /models/GLM-5.3-Flash
python -m benchmark.duet_eval.launch_duet --model /models/GLM-5.3-Flash --arm stock --gpus 0,1,2,3 --port 30001 \
    --context-length 262144 --max-running-requests 256 --sampling-backend pytorch
python -m benchmark.duet_eval.launch_duet --model /models/GLM-5.3-Flash --duet-release Latent-Voyager/GLM-5.3-Flash-SEED --arm duet \
    --gpus 0,1,2,3 --port 30002 --context-length 262144 --max-running-requests 256 --sampling-backend pytorch
```

The server it starts (one per 4-GPU GB300 node; arm64, CUDA 13, the image of section "Install"; `launch.json` records it):

| setting | value | why |
|---|---|---|
| parallelism | `--tp-size 4` (no expert parallelism, no DP attention) | 76 GB of weights per GPU (the bring-up also tried TP 4 + EP 4; twinstar docs/74) |
| MLA cache | `--kv-cache-dtype bfloat16` | SGLang's Blackwell default is an FP8 cache with unit scales ("no scaling factors provided"), which quantizes every cache entry; bf16 holds 12.5M tokens at mem-fraction 0.85 |
| DSA kernels | `--dsa-prefill-backend trtllm --dsa-decode-backend trtllm` | the vendor's Blackwell recipe (`test/registered/e2e/models/test_glm53_flash_b200.py`) and what SGLang runs on SM10x anyway (it reroutes `flashmla_sparse` to trtllm for the pooled indexer); the indexer's FP8 logits are DeepGEMM's in every configuration |
| FP8 GEMMs | `--moe-runner-backend flashinfer_trtllm`, `--fp8-gemm-backend` auto (DeepGEMM) | SGLang's default FP8 arithmetic on SM10x, the one the twinstar reference of GLM mirrors (twinstar docs/73 section 73.2): the routed experts on FlashInfer TRT-LLM with the checkpoint's fp32 block scales; the dense FP8 layers on DeepGEMM, whose weights SGLang requantizes to power-of-two (UE8M0) block scales at load (this checkpoint's scales are not powers of two).  `--fp8-gemm-backend flashinfer_trtllm` keeps the dense layers' fp32 scales; it changes the served model and would be changed together with the reference.  Note also that the FlashInfer FP8 MoE path does not apply the config's `swiglu_limit` 10 to the routed experts (DeepGEMM's MoE runner does, on UE8M0-requantized experts) |
| DeepGEMM warm-up | `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` | the warm-up walks every M for each weight shape (measured ~20 min per shape on GB300); kernels compile on first use instead; no numeric effect |
| KDA states | `SGLANG_MAMBA_CONV_DTYPE=float32 SGLANG_MAMBA_SSM_DTYPE=float32` (Triton KDA kernels) | as the Kimi-Linear exams pin them for both arms |
| context, batch | `--context-length 262144`, `--max-running-requests` = `--cuda-graph-max-bs-decode` = MRR, `--max-mamba-cache-size` MRR + 8, `--mem-fraction-static 0.85` | RULER 128K + generation; decode through the CUDA graph |
| other | radix cache off, chunked prefill at SGLang's default (16384), `--sampling-backend pytorch` (per-request seeds with top-p), `--watchdog-timeout 1800`, `--enable-custom-logit-processor` (the forced-NLL checks) | as the other families |

Start-up ~12 min (9 of them loading: the FlashInfer MoE weight layout is built shard by shard); the reply lengths the model produces
at each reasoning effort: twinstar docs/74.  Exam family `glm` (`lmeval/tasks.py`): the model's chat template (it writes
`<|system|>Reasoning Effort: Max` first and opens `<|assistant|><think>`), `reasoning_effort` max (the model card: "for benchmark
and leaderboard reproduction, keep the default max"), vendor sampling T 1.0 / top_p 0.95 / no top_k, stop on the
`generation_config.json` ids [154820 `<|endoftext|>`, 154827 `<|user|>`, 154829 `<|observation|>`] (the server's own stop set; the
model ends a turn with `<|user|>`), the reply graded after the first `</think>` (a reply that never closes it is graded empty and
counted).  RULER: every RULER task continues an assistant prefix (lm-eval's `gen_prefix`), which this template renders as an
assistant turn without reasoning (`<|assistant|><think></think>Answer:`), so RULER runs with thinking off as for the other thinking
models.  GPQA / MMLU through simple-evals and LiveCodeBench with its official prompt and grader, as for every family.  Job lines:
`lmeval/README.md`.

## Exams

`lmeval/README.md`: the protocol (AIME 2024 + 2025, GPQA-Diamond, MATH-500, LiveCodeBench, IFEval and HumanEval+ x8; LongBench v2
x4; MMLU x1; RULER at 32K, 64K and 128K x1), the job lines per model, the paired report (`lmeval/report_lmeval.py`), the reported v3.5 table, resuming after
a time limit, and every deviation from stock lm-eval / simple-evals.

## Parity check (served vs reference)

Before the first exam of a base / release pair, the served path is compared with the reference PyTorch implementation of the
twinstar repository on teacher-forced windows: 32 windows of the model's held-out token stream (prefix 3584 tokens, then 256
forced targets run as decode steps, so the decode-time state cut is inside the check).  The reference scores the base once and
the DUET set twice (the pass-to-pass difference is the reference noise); the exam serving path (`launch_duet.py`, exam settings)
scores both arms one request at a time (`seq`) and with 16 requests in flight (`conc16`).  One 4-GPU node per check.

```bash
# Flash-Next (BF16: TP 2, NVFP4: TP 1); the reference shards the base over the 4 GPUs and scores with flash_next/guard_reference.py
# (the guard's helpers; TWINSTAR_ROOT=<twinstar checkout at e91d2a7a6> scores with the original)
REFERENCE_ROOT=<twinstar checkout> MANIFEST=windows TOK=<tokens.bin> \
  bash benchmark/duet_eval/validate_fn_parity_job.sh <out> <base dir> <release dir> <tp>
# Kimi-Linear, Lightning (reference on GPU 0, the served arms on GPUs 0 and 1)
REFERENCE_ROOT=<twinstar checkout> TOK=<tokens.bin> \
  bash benchmark/duet_eval/validate_hooked_parity_job.sh <kimi|lightning> <out> <base dir> <release dir>
```

The token streams are the models' held-out web streams `eval/web/tokens.bin` (FineWeb-Edu + FineMath documents tokenized by the
base's tokenizer).  The training-data repositories of the release (Hugging Face datasets `Latent-Voyager/<original model>-SEED`, the
dataset repository beside each model repository) do not ship them: the twinstar recipe rebuilds them byte for byte from the public
sources (the twinstar repository's `release/datasets/README.md`).  `MANIFEST=windows` builds the 32 Flash-Next cases from the token
stream; the reported runs read them from an earlier admission manifest, whose first 32 cases are the same windows
(`parity_verdict.py` checks it).  The other paths are environment variables listed at the top of each script (defaults: the
cluster of the reported runs); section "Container" has a complete example.  The verdict (`parity_verdict.py`, also runnable
on a finished run's directory: `python -m benchmark.duet_eval.parity_verdict <out> --write <file>`) compares, per mode, the served
DUET - served base difference with the reference DUET - base difference, so that what moves both served arms of one job together
cancels: PASS (absolute) when they agree within max(0.002 nats, reference noise) for each reference DUET pass, otherwise PASS
(statistical) when the per-window difference of the two differences is within 2 SE; the last line is `PARITY PASS` or
`PARITY FAIL` and `verdict.json` holds every number.  Each arm's own served-vs-reference numbers are printed and written beside
it, as information.  Until 2026-10-07 each arm was judged alone against its reference; the served path's run-to-run offset on
these windows is ~0.002 nats, the size of the threshold, and the unchanged base arm of the FN-BF16 v3.5 parity missed that
per-arm gate in a rerun by as much as the DUET arm did in the first run.

The gate applied to the recorded parity runs (twinstar repository, `results/acc-sgl/<run>`; per-arm numbers identical to the
recorded `verdict.json` in all runs that record them):

| run | per-arm gate (recorded) | difference gate, seq | difference gate, conc16 |
|---|---|---|---|
| fn-bf16-v35b2-parity-20261008 (the released FN-BF16 v3.5-b set, round 2) | FAIL (DUET conc16, +0.0022 +- 0.0008: 0.0002 over the line) | PASS (absolute) | PASS (absolute) |
| fn-bf16-v35b-parity-20261007 (round 1 of v3.5-b, not released) | PASS | PASS (absolute) | PASS (absolute) |
| fn-bf16-v3b-parity-20261007 (its stage-1 set v3b, not released) | FAIL (base conc16) | **FAIL**: served -0.0017 vs reference +0.0010, paired -0.0028 +- 0.0013 | PASS (absolute) |
| fn-bf16-v35-parity-20261007 (the first BF16 v3.5 set, superseded) | FAIL (DUET seq) | PASS (statistical) | PASS (absolute) |
| fn-bf16-v35-parity-rerun-20261007 (same set) | FAIL (base conc16) | PASS (absolute) | PASS (absolute) |
| fn-bf16-v3-parity-v35pin-20261007 | PASS | PASS (absolute) | PASS (absolute) |
| fn-nvfp4-v35-parity-20261006 | PASS | PASS (absolute) | PASS (absolute) |
| kimi-v35-parity-20261006 | PASS | PASS (absolute) | PASS (absolute) |
| lightning-v3c-parity-20261006 (the Lightning v3.5 set) | PASS | PASS (statistical) | PASS (statistical) |
| fn-bf16-v3-parity-20261003 | PASS | PASS (absolute) | PASS (absolute) |
| fn-nvfp4-v3-parity-20261003 | PASS | **FAIL**: served +0.0026 vs reference -0.0087 / -0.0041, paired +0.0089 +- 0.0040 | PASS (statistical) |
| kimi-v3c-parity-20261003 (the Kimi v3 set) | PASS | PASS (absolute) | PASS (absolute) |
| fn-bf16-parity-20261003, l35-fast-parity-20261003 (v1 sets) | PASS | PASS | PASS |

In the FN-NVFP4 v3 run the two served arms moved in opposite directions (DUET +0.0043, base -0.0047 against their references, each
within 2 SE alone); the served NVFP4 path is not run-to-run deterministic (two passes of one server differ by up to 0.006 nats).
In the FN-BF16 v3b run the DUET arm agrees with its reference in both modes (seq -0.0012 +- 0.0009) and the base arm moved up
(seq +0.0016, conc16 +0.0026); the FN-BF16 v3.5-b and v3b parity jobs ran the pinned copy 94f6b2a23a (per-arm gate) and were judged
with this branch's `parity_verdict.py` afterwards.

`validate_fn_port_job.sh` checks that the in-fork Flash-Next package serves what the original twinstar tree served
(`flash_next/README.md`; the original tree, a twinstar checkout at `e91d2a7a6`, is its fifth
argument).  CPU unit tests of the serving layer and of the exam tools: `test/registered/unit/duet/` (run from that directory,
`python -m unittest <module>`; `test_duet_hub`, `test_exam_resume`, `test_parity_verdict`, `test_extra_exams` (its EvalPlus and
lm-eval cases skip without those packages) and most of `test_duet_adapters_numerics` need no GPU and no torch; the others need
torch and run on CPU in the exam image).

## Commits of the reported runs

The reported numbers (twinstar repository, docs/67) were produced at the fork commits that
`results/acc-sgl/v35-finals/provenance.md` lists per model, arm, series and repetition: the runs launched for the final matrix
(the v3.5 arms of Flash-Next BF16, Flash-Next NVFP4 and Kimi-Linear, their RULER 64K / 128K units, the late v3 and Lightning
v3-b repetitions) at 94f6b2a23a, from a pinned checkout; the Lightning v3.5 arms at 6b82207d32 and 96d264d30d; base and v3 arms
reused from earlier series at earlier commits of this branch (ce8630c3, 5f23c08e59, a81c871a34, 7505b416ba, 9433abc3bf,
45f5c4188f, 05f1e4a1 (whose job log names 703861bb), 7d06bb5c, 28417aaf, 84c960f320, ed29a633af; one Lightning base series did not
record its commit -- its job script matches 2334110d9c, kept as an archive tag of this repository; the commit of every sample file: twinstar
results/acc-sgl/v35-fairness-20261008/pair_config_check.json and docs/86).  The Flash-Next BF16 v3.5-b exams (2026-10-07) run at 94f6b2a23a as well.  All of these commits are ancestors of this
branch.  `git log <commit>..94f6b2a23a -- benchmark/duet_eval python/sglang/srt/duet` shows what changed between an older series
and the final pin.  The GLM-5.3-Flash finals (2026-10-08 / 09) ran from two pinned checkouts: the base arm at 0a5dc6ade0, the
DUET arm at 091d24da6c (the GLM DUET adapter with the boundary fix of twinstar docs/79 section 79.12); between the two commits only
the row "GLM DUET adapter" of the table below changed (the adapter package, the DUET layer's default-off `omega_heads`, the launcher's DUET-arm
flags, the parity tools), and the recorded `launch.json` command lines of the two arms differ only by the DUET flags.  The GLM DUET
arm's LiveCodeBench cell is the restart-free rerun of 2026-10-09 at c948d22edf (091d24da6c plus the LiveCodeBench doc shards); that
commit is on this branch as the cherry-pick 7901c6b2ec (identical patch), and kept as a tag of this repository's archive.

Commits after 94f6b2a23a change no prompt, sampling setting, seed or grader, and no computed value of the Flash-Next, Kimi-Linear
or Lightning served paths: their model files changed by two unused local assignments (row "Archive and vendoring"), the Kimi
release verification now done by `sglang.srt.duet.release` with identical reports and unused import aliases (cleanup 2); the
common layer gained the GLM and Hub-fetch code, which leaves the other families' arithmetic as it was (`omega_heads` defaults to
the old draw).  The GLM-5.3-Flash family and its adapter are new after it (the GLM rows; cleanup 3 removed parameters nothing read):

| commit | touches |
|---|---|
| 6362194b8b | `exam_lmeval_job.sh` `RESUME=1` block: also resumes the job's RULER unit (only with `RESUME=1`); test `test_exam_resume.py` |
| 8e142d8283 | parity verdict: `parity_verdict.py` (new), the verdict step of both parity jobs; test `test_parity_verdict.py` |
| ef01ed63fe | the header line of the three gate jobs (fork commit inside the container) |
| 995287cccb | `lmeval/regrade_aime24.py`, `lmeval/regrade_ifeval.py` moved to `archive/` (one-off regrades of runs before 2026-10-03) |
| 6161ca0887 | READMEs, `CODEOWNERS`, docstrings, comments and help text of `tasks.py`, `run.py`, `report_lmeval.py` and the exam job, the exam job's error message for a missing `DUET_REL` |
| 8e8b464ba8 | the pin note of this README; the `--check-prompts` path in `lmeval/README.md` (READMEs only) |
| c07e550e75 | `flash_next/twinstar_sgl/twinstar_qwen4_exp.py`: the mixer bridge, the state-only emitters and the prompt hidden-state capture removed (their code is not in this fork; no release reaches them) and refused where they were entered; `flash_next/README.md` |
| 5b00fb230c | `exam_lmeval_job.sh`: `RESUME=1` finds a run of two simple-evals exams with `SE_SHARD` (`gen-s<i>of<n>`); header text; the start line names the arms; test `test_exam_resume.py` |
| fafaf67a8c | `validate_fn_parity_job.sh`: `MANIFEST=windows` builds the 32 guard cases from the token stream |
| 79e8f98c09 | text and pins: `requirements-exams.txt` (the image tag, the complete package set of the reported runs), the `prepare_data.sh` header, docstrings and help text of `run.py`, `lb.py`, `lcb_grade.py`, `lcb/build_lcb_data.py`, `ruler_build.py`, `lmeval/__init__.py` |
| d0254a516a | `exam_lmeval_job.sh`: a `gpqa_se` or `mmlu_se` run alone logs to `<out>/<arm>.<exam>.rep<k>.log` (log names only) |
| the commit after it | READMEs (this table, section "Container", the v3.5-b release, the corrections of 2026-10-07), `CODEOWNERS` |
| GLM DUET adapter (2026-10-08, twinstar docs/79) | `python/sglang/srt/models/glm_duet/` (new: geometry / release contract, the emitters' training arithmetic, the model class with the chunk-aware prompt pass, the boundary decode step and the state cut on the KDA pool); `sglang.srt.duet`: the `glm5-next` spec model and adapter row, `project_state_batched(omega_heads=)` (a TP rank's head slice draws the unsharded sketch; default unchanged); `launch_duet.glm_duet_cli` (the DUET arm's flags), `--disable-overlap-schedule` (glm, both arms, for the teacher-forced checks); `forced_nll.py --save-logits-every` and the window hashes / per-token NLLs in its output; `validate_glm_parity_job.sh` (new); tests `test_glm_duet.py` (new) and the GLM launcher test.  The other families' code paths, launch commands and environments are unchanged (`omega_heads` defaults to the old draw) |
| Hub fetch (2026-10-09) | `python/sglang/srt/duet/release.py`: `fetch_release` also checks the downloaded component file against the Hub's LFS sha256; new `hub_lfs_sha256`, `check_pin`, `is_hub_id`, `fetch_base` (a base checkpoint from the Hub, checked against the manifest's `base` before the download); `launch_duet.py`: `--model` takes a Hub id, `--duet-release-sha256` (pin); `fetch_duet.py` (new); section "Evaluate a released DUET set from the Hub"; test `test_duet_hub.py` (new).  A launch from local directories builds the same command, environment and `launch.json` as before (`test_duet_hub`, `test_glm_duet`); no served code path changed |
| GLM family (2026-10-07, f999c8faad and the commit after it, which serves the dense FP8 layers on SGLang's default DeepGEMM path as the GLM reference does) | a new family `glm` (section "GLM-5.3-Flash"): `tasks.FAMILIES["glm"]`, `launch_duet.py` (family detection, `glm_command`, `--duet-release` optional for the stock arm of `RELEASE_OPTIONAL_STOCK` families), `exam_lmeval_job.sh` (the `glm` case, `GLM_BASE`, the release-free base arm); `reply_survey.py` (new: reply lengths, decode throughput, sanity replies); a launcher unit test.  The other families' launch commands and environments (`--dry-run`, both arms) and their task directories are unchanged |
| Archive and vendoring (2026-10-09, twinstar docs/86 section 86.4) | one-off diagnostics moved to `archive/` (`archive/README.md`): `lmeval/diagnose.py`, `lmeval/diag_forced_{fn,sweep}_job.sh`, `lmeval/probe_opener_job.sh`, `lmeval/probe_fn_tp_job.sh`, `profile_decode_job.sh`, `trace_summary.py`, `bench_state_budget.py`, `python/sglang/srt/models/kimi_duet/tests/cut_quality.py`; `kimi_duet/tests/test_batched_cut.py` -> test `test_kimi_batched_cut.py` (the package `kimi_duet.tests` removed); `flash_next/guard_reference.py` (new: the guard's reference helpers, copied from twinstar e91d2a7a6) read by `validate_fn_parity_job.sh` instead of a twinstar checkout (`TWINSTAR_ROOT` optional), test `test_fn_guard_reference.py`; `flash_next/twinstar_sgl/twinstar_qwen4_exp.py`: the unused locals `rec` and `n` of `_twinstar_prefill` removed; READMEs.  No served computation changed: the two removed assignments were never read |
| 975636da62 (cleanup 1, launcher) | `launch_duet.py`: the options no job, test or procedure passes are removed -- `--stock-native-state-dtype` (a Kimi stock arm without the fp32 state pin), `--no-fork-pythonpath`, `--radix` and `--fn-chunk` (Flash-Next; the chunk size is the constant `FN_CHUNK` = 8192, the old default) -- and the unused `dry_run` argument of `run_steps`; the server command, environment, release identity and Flash-Next settings file of every family and arm are unchanged (`--dry-run` of the five v3.5 sets on their bases, both arms, the GLM base arm without a release and two `--extra` ablations, the Flash-Next pre-steps run on CPU and their settings files and config views compared); `validate_fn_port_job.sh`: the original tree is required (its default, a checkout path on the cluster of the reported runs, no longer exists); comments of `exam_lmeval_job.sh` (the GLM DUET arm) and `benchmark/duet_eval/__init__.py`; the environment table of `python/sglang/srt/duet/README.md` (GLM rows) |
| a3c58f609f (cleanup 2, Kimi package) | `kimi_duet_checkpoint.py`: its copies of `read_header`, the per-tensor audit, the sha256 loop and the Hub download (`resolve_release`) replaced by `sglang.srt.duet.release`'s (`read_header`, `audit_metadata` on the module's own contract, `sha256_of`; `launch.py` fetches with `fetch_release`, which also checks the Hub's LFS sha256); the Kimi-specific spec / base checks and the tensor contract unchanged.  `verify_release` returns the identical report for the Kimi v3 and v3.5 releases on their base config and refuses the same 13 tampered synthetic releases and a truncated component file as before; new test `test_kimi_checkpoint.py`.  `kimi_duet_math.py` / `duet_options.py`: the aliases nothing imports (`quantize`, `_orthonormalize`, `truncate_rank`, `project_state`, `ROW_CHUNK`; `DuetOptions`, `ENV_PREFIX`, `POLICIES`, `boolean`) removed (test `test_kimi_batched_cut` updated).  No computed value changes |
| 8fa2a3d205 (cleanup 3, GLM adapter) | `glm_duet/model.py`: parameters no body reads removed -- `last` of `_KDAEmitter.emit` (and the list `_prompt_pass` built only to pass it on), `prefix` of `_boundary` -- and an unread unpacked value of the diagnostic `_selfcheck`; `glm_duet/geometry.py`: the unused `BASE_ARCH` and `Geometry.gate_lower_bound` (the emitter reads the bound from the layer config).  Every call matches its new signature (AST); no computed value changes |
| c97134ea6d (cleanup 4, exam tables) | `lmeval/tasks.py`: the constant `RULER_LENGTHS_P1` (the 256K / 1M RULER lengths, read by no code since those runs left the protocol) removed, its lengths named in the docstring and in `lmeval/README.md` ("Long-context RULER").  The task directories of all four families (every exam, the protocol's group, the RULER task groups) are byte-identical to those of 94f6b2a23a (Kimi-Linear, Flash-Next, Lightning) and of d4e342dbde (all four) |
| 9258aa7598 (cleanup 5, archive) | `throughput.py` -> `archive/throughput.py` (its two callers, `probe_fn_tp_job.sh` and `profile_decode_job.sh`, were archived on 2026-10-09; `reply_survey.py throughput` measures the same for current work); the archived callers name the new module path; `archive/README.md` row; README contents (the probes, the GLM parity job) |
| the commit after 9258aa7598 (cleanup 6, text) | this table's hashes and check paragraph; the docstrings of `sglang.srt.duet.release._sibling` and `latent_codec` (the loaders and tests they name exist on this branch) |
| 7901c6b2ec, 0d62329f71 (LiveCodeBench doc shards, cherry-picked from c948d22edf / ec0ffe6ef3, the checkout the GLM v3.5 LCB runs of 2026-10-09 used) | `run.py --doc-shard i/n` for `lcb` (lm-eval `--samples`, a problem keeps its doc_id and seed), `exam_lmeval_job.sh` `LCB_SHARD`; nothing else |
| 470b70901a, 259b87f9db, c3d5e8f0f6, 477b841f75, 722f62f1ed (HumanEval+ and LongBench v2, 2026-10-10, twinstar docs/90) | two exams in `tasks.EXAMS` and `tasks.P0`: `humaneval_plus` (`evalplus_utils.py`, graded by `humaneval_grade.py` with EvalPlus, `requirements-evalplus.txt`) and `longbench_v2` (`longbench_v2_utils.py`); `tasks.OWN_PROCESS` / `LONG_PROMPT` / `DOC_SHARD_EXAMS` and the `n_docs` of LCB and the two exams; `run.py`: `--doc-shard` for the three; `exam_lmeval_job.sh`: the own-process exams in one loop (the LCB and AA-LCR command lines as before), `DOC_SHARD` (`LCB_SHARD` its earlier name), the resume kinds with a shard suffix; `report_lmeval.py`: the new cells and a warning for ungraded HumanEval+ samples; `prepare_data.sh`: the two data files; `lmeval/README.md`; tests `test_extra_exams.py` (new), `test_exam_resume.py`.  The task files of the earlier exams are byte-identical (check below); every HumanEval+ / LongBench v2 run of docs/90 ran at 722f62f1ed |
| the commit after cleanup 6 (release layout) | the twinstar repository keeps only the five v3.5 release directories in `release/` (2026-10-09; the others moved to `release/archive/`): this README's owner's-directory names; the default `DUET_REL` of `lmeval/aa_lcr_judge_job.sh` (the judge's stock-arm release check) and of `longctx_probe_job.sh` (the out-of-protocol long-context probe) is now the family's v3.5 set; the archived diagnostics' defaults point at `release/archive/`.  No exam job, task or server argument changes |
| the commits after 37b19fd7dc (audit of 2026-10-10) | the Hub path for the `Latent-Voyager/<model>-SEED` repositories: `release.fetch_release` / `fetch_base` name the cause of a private-repository access error and warn when the Hub reports no LFS sha256; `launch_duet.py` turns release errors into one-line exits, notes a base fetched without the revision the manifest lacks (NVFP4), refuses `--tp` other than 1 for Kimi-Linear and Lightning (their commands are TP 1; the exam job passes TP 1); the Kimi launcher parses `--duet <repo>@<rev>`; `sglang.launch_server --duet-release <Hub id>` downloads the release once while the server arguments resolve (`adapters.localize_release`; a directory or a path is left as given); tests `test_duet_hub.py`, `test_kimi_checkpoint.py`.  The exam harness: `run.json` never records a variable with TOKEN in its name; `exam_lmeval_job.sh` requires `DUET_REL` (the defaults to the archived v3 sets are gone); `validate_hooked_parity_job.sh` finds the git dirs under `REPO_ROOT`.  The Kimi model file lost an unused module constant and its import.  READMEs (an outside user's path from the Hub, the GLM row of the reported table), docstrings and comments.  The task files of every exam and family are byte-identical to 37b19fd7dc's, and those of the earlier exams to 94f6b2a23a's; no served computation changes |

Check (up to the release-layout commit; the LCB-shard and HumanEval+ / LongBench v2 rows above add the shard paths and two exams,
and the task files of every earlier exam stay byte-identical, twinstar docs/90): the task directories that `python -m
benchmark.duet_eval.lmeval.tasks` builds for every family and exam (all of `tasks.EXAMS`, the protocol's exam groups and the RULER
task groups) are byte-identical at 94f6b2a23a and at the branch head;
`lmeval/model.py`, `common.py`, `se_utils.py`, the `*_utils.py` graders and `lcb/` (except the docstring of `build_lcb_data.py`) are
unchanged since 94f6b2a23a; `launch_duet.py` and the code under `python/` changed only by the GLM rows, the Hub-fetch row and
cleanups 1-3 of the table above (`python/sglang/srt/duet/README.md` too) and by the Kimi tests moved out of `python/` (row
"Archive and vendoring"); `run.py`, `tasks.py`, `report_lmeval.py`, `lb.py`, `lcb_grade.py`, `ruler_build.py`, `lmeval/__init__.py`
and `prepare_data.sh` differ only in docstrings, comments and help text (and `tasks.py` by the unused constant of cleanup 4); `requirements-exams.txt` gained comments and the pins of 15 dependencies;
`exam_lmeval_job.sh` changed in its `RESUME=1` block, comments, messages, log names and the start line; the Flash-Next package
lost three paths no release reaches (c07e550e75) and two unused locals (row "Archive and vendoring").  To repeat the check where
lm-eval is installed:

```bash
for c in 94f6b2a23a HEAD; do
  git worktree add --detach /tmp/duet-$c $c
  for f in kimi flash_next lightning; do
    (cd /tmp/duet-$c && python -m benchmark.duet_eval.lmeval.tasks --family $f --out /tmp/tasks-$c/$f \
        --exams aime24,aime25,gpqa_se,math500,lcb,ifeval,mmlu_se,ruler,aa_lcr > /dev/null)
  done
done
diff -r /tmp/tasks-94f6b2a23a /tmp/tasks-HEAD && echo identical
git diff --stat 94f6b2a23a HEAD -- python benchmark/duet_eval/launch_duet.py benchmark/duet_eval/lmeval
```

## Contents (DUET code on this branch, relative to upstream fa663e72)

**Current canonical serving.** Performance numbers come from the canonical native entry points.
The benchmark launcher has been redirected and qualified on CPU; it has not been measured
separately on GPU. Historical exam numbers and recipes earlier in this document retain their
original source identities and are not evidence for this redirect.

* `python/sglang/srt/duet/`: the model-independent layer -- release spec, fetch and verification (`release.py`, `spec.py`),
  options and numerics profiles, adapter registration (`adapters.py`), latent code (`latent_codec.py`), the state cut
  (`state_factor.py`, `dense_state_policy.py`), slot state pool, invariants, the retraction guard.
* `python/sglang/srt/models/kimi_linear_duet/` (canonical Kimi-Linear adapter and checkpoint verification), `python/sglang/srt/models/lightning_duet/`
  (Nemotron-3.5-Lightning), `python/sglang/srt/models/flash_next_duet/` (canonical in-tree Flash-Next and preserved factored-pool performance path), `python/sglang/srt/models/glm_duet/` (GLM-5.3-Flash), `benchmark/duet_eval/launch_duet.py` (both arms; Kimi/Flash-Next route through native `sglang.launch_server`, no external package/config-view alternative), `benchmark/duet_eval/fetch_duet.py` (a release and its base from the Hub, checked).
* Generic SGLang changes on the exam path: DUET server arguments and their resolution (`server_args.py`, `arg_groups/`), adapter
  registration (`models/registry.py`), per-request sampling seeds honoured outside deterministic mode (`sampling/sampling_batch_info.py`,
  both arms), `/server_info` `duet` (`entrypoints/http_server.py`), the retraction guard (`managers/schedule_batch.py`), the decode
  CUDA-graph hooks of the state cut (`model_executor/runner/decode_cuda_graph_runner.py`), the Kimi KDA state-cut hooks
  (`layers/attention/linear/kda_backend.py`, `kda_state_prune.py`), the Flash-Next GDN path with the factored recurrent-state pool and
  its kernels (`layers/attention/linear/gdn_backend.py`, `kernels/gdn_factored*.py`, `gdn_prefill_reference.py`, `gdn_truncate.py`,
  `gdn_k31_eigh.py`, `mem_cache/gdn_factored_pool.py`, `gdn_continuation_capacity.py`, `flashnext_scheme_c*.py`, `memory_pool.py`,
  `kv_cache_configurator.py`, `model_executor/duet_policy.py`, `fullstack_policy.py`), the fp32 emitters (`layers/twinstar_emitter_fp32.py`),
  and small shared fixes (`causal_conv1d_triton.py` state-dtype store, `utils/common.py`).
* Not used by the exams (behind flags or modes the exam launch does not set): the Flash-Next prefill/decode disaggregation and state
  staging (`disaggregation/`, `sgl-model-gateway/src/routers/http/pd_*`), the latent page pool and arrival pipeline
  (`mem_cache/flashnext_*` except `flashnext_scheme_c*`), the GDN prefill graphs (`mem_cache/gdn_prefill_*`, `gdn_pside_*`), the QSA
  code pool (`layers/attention/qsa/code*.py`, `mem_cache/qsa_code*`), HiCache host pools (`mem_cache/pool_host/flashnext_*`), and their
  tests (`test/manual/`, `test/registered/unit/{mem_cache,disaggregation,layers}`).

**Exams** (`lmeval/`): `tasks.py` (protocol and task yamls), `run.py` (one repetition = one lm-eval process), `model.py` (lm-eval
model `sglang-duet`: paired seeds, budgets, sidecar), `lb.py` (balancer, replay on resume), `se_utils.py` (simple-evals GPQA /
MMLU), `aime_utils.py`, `math500_utils.py`, `lcb_utils.py` + `lcb/` + `lcb_grade.py` (LiveCodeBench), `evalplus_utils.py` +
`humaneval_grade.py` + `requirements-evalplus.txt` (HumanEval+), `longbench_v2_utils.py` (LongBench v2), `ruler_build.py`,
`common.py`, `report_lmeval.py`, `exam_lmeval_job.sh`, `prepare_data.sh`, `requirements-exams.txt`, `reference/` (prompt hashes of
the reported Kimi-Linear and Flash-Next NVFP4 exams); AA-LCR (`aa_lcr_*`, not part of the protocol).

**Gates and parity**: `validate_fn_parity_job.sh` (its reference scoring: `flash_next/guard_reference.py`),
`validate_hooked_parity_job.sh`, `parity_verdict.py`, `validate_glm_parity_job.sh` (GLM: the served side), `forced_nll.py`, `forced_logits.py`, `validate_fn_port_job.sh`; unit tests in
`test/registered/unit/duet/`.

**Probes and diagnostics** (development tools; their timings are not results): `bench_state_cut.py`, `reply_survey.py`, `long_prompt_probe.py`, `longctx_probe_job.sh`;
`lmeval/forced_pairs.py`, `lmeval/probe_opener.py`.  The Lightning drivers under
`benchmark/lightning_*` and `benchmark/lightning_duet_probe/` are the colleague's integration guards, written for an out-of-tree
plugin layout and its cluster paths; the exams do not use them.

**Archive** (`archive/`): one-off regrades of sample logs written before 2026-10-03 and the one-off diagnostics of 2026-09-30 to
2026-10-04 (`diagnose.py`, `diag_forced_*_job.sh`, `probe_opener_job.sh`, `probe_fn_tp_job.sh`, `profile_decode_job.sh`,
`trace_summary.py`, `bench_state_budget.py`, `cut_quality.py`, and since 2026-10-09 `throughput.py`; `archive/README.md`).
