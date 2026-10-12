#!/bin/bash
# Paired DUET exam on one node with lm-eval: N data-parallel servers per arm (DUET on the first N*TP GPUs, the unchanged base on
# the next N*TP), one least-outstanding balancer per arm, one lm-eval process per repetition and arm (run.py), then the paired
# report.  Both arms: identical ServerArgs, identical task yamls, identical per-request sampling seeds.
#   bash exam_lmeval_job.sh <kimi|flash_next|lightning|glm> <out dir> [exams]
# exams: comma list of tasks.EXAMS; default the protocol's exams (tasks.P0: AIME 24+25, GPQA-Diamond, MATH-500, LiveCodeBench, IFEval,
# MMLU, RULER, HumanEval+, LongBench v2), each exam at its protocol repetitions (tasks.EXAMS reps) and RULER at RULER_LENGTHS (default 32K,
# tasks.RULER_LENGTHS_P0; the protocol's 64K and 128K runs are jobs of their own, README.md).
# One node, 4 GPUs: NDUET DUET servers + NSTOCK base servers at TP=1 (default 3 + 1: the base decodes faster).  More nodes = more jobs
# sharing <out dir> with disjoint REP_LIST (e.g. job A REP_LIST="0 1 2 3", job B REP_LIST="4 5 6 7"), one RULER length per job
# (RULER_LENGTHS + RULER_TAG); run report_lmeval.py over <out dir> once every job is done.  Both arms of a pairing must run the same
# exams per job and the same SE_SHARD split: the seed of a request counts the repeats of its prompt inside one lm-eval process.
# Cluster submit (a copy outside the repository cannot find the repository from its own path, so FORK is required there):
#   cp exam_lmeval_job.sh $OUT/job-$SLURM_JOB_ID.sh; FORK=<repository> bash $OUT/job-$SLURM_JOB_ID.sh ... > $OUT/job-$SLURM_JOB_ID.log
# Paths, all environment variables.  The defaults after '=' are the owner's cluster (/src = the twinstar checkout, /work = models,
# HF cache and python packages, /me = the owner's models); on any other machine set them (README.md "Run the protocol for one
# release"; ../README.md "Container"):
#   FORK            this repository; default: the repository this script is in
#   DUET_DATA_ROOT  =/src/results/hfcache  exam data: ruler/ (RULER sources), lcb/ (LiveCodeBench v6 set + tests), simple-evals/,
#                   evalplus-cache/ (HumanEval+), longbench-v2/ (data.json and the prompts built per tokenizer), datasets/ (HF
#                   datasets cache), nltk_data/ -- prepare_data.sh builds it
#   HF_HOME         =/work/hf-cache        HF cache; launch_duet fetches a Hub release into $HF_HOME/duet-releases
#   HF_OFFLINE      =1                     1 = no network; 0 = fetch (a Hub release, the pinned HF datasets)
#   EXTRA_PYTHONPATH =/work/pyuser/lib/python3.12/site-packages:/src/results/pyextra   python packages beside the image's
#                   (requirements-exams.txt); set it empty when they are installed
#   KIMI_BASE       =/work/models/Kimi-Linear-48B-A3B-Instruct            base checkpoints, unchanged
#   FN_BASE         =/work/hf-cache/nvidia-nvfp4-fc694b54                 (the NVFP4 base)
#   LIGHTNING_BASE  =/me/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
#   GLM_BASE        =/src/results/models/GLM-5.3-Flash          (zai-org/GLM-5.3-Flash, FP8 release, revision eb9eb208eb0d)
#   DUET_REL        DUET release (required; the base arm checks its base against it): a directory or a Hub id
#                   <owner>/<repo>[@rev], e.g. Latent-Voyager/Kimi-Linear-48B-A3B-Instruct-SEED (../README.md, "Evaluate a
#                   released DUET set from the Hub").  Preferably the directory fetch_duet.py downloaded once, as the servers of
#                   one job would otherwise fetch the same release in parallel.  With HF_OFFLINE=1 a Hub id needs its full
#                   40-hex @<commit> and is served from the copy fetched earlier ($HF_HOME/duet-releases/<owner>--<repo>/<commit>).
#                   On the cluster of the reported runs the v3.5 directories are /src/release/duet-<flash-next-v3.5-b-r2 |
#                   flash-next-nvfp4-v3.5 | kimi-v3.5 | lightning-v3.5 | glm53-flash-v3.5>.
#                   glm: optional while the job serves the base only (NDUET=0, the glm default: one TP-4 base server per node);
#                   the GLM DUET arm (NDUET=1 NSTOCK=0) serves the release through models/glm_duet (launch_duet.glm_duet_cli)
#   TWINSTAR_ROOT   flash_next only, optional: serve the original twinstar_sgl tree of this twinstar checkout (tag
#                   commit e91d2a7a6) instead of this fork's copy (../flash_next; A/B checks only)
# Exam knobs: NDUET / NSTOCK (N sets both; ARMS "duet stock" by default, one arm when the other has 0 servers; glm: 0 + 1), TP (1; the BF16
#   Flash-Next base needs 2; glm 4), DUET_EXTRA (duet-arm server flags, ablations, see below), REP_LIST (default all), CTX (262144, the minimum: the 229376-token generation ceiling + 32768 of prompt; the Kimi-Linear RULER
#   64K / 128K runs used 270336),
#   MRR (max running requests per server, 128), CONC (lm-eval requests in flight per process, MRR x the larger server count; 0 or 1 would make lm-eval send one request at a time), LIMIT (questions per task,
#   smoke tests), REPS_CAP, RULER_LENGTHS (32768), RULER_N (docs per task and length, 100), RULER_BUILD (docs built per
#   task and length, each from a fresh RNG state; default lm-eval's 500 in one sequence; the 64K and 128K runs set it to RULER_N), RULER_CACHE
#   (directory of those builds, shared by the arms and jobs), MEM_FRACTION / WATCHDOG (kimi launcher: static memory share, default
#   0.7, and scheduler watchdog seconds, default 300 -- 512K-1M prompts prefill unchunked), RULER_TASKS (all = the 13 tasks,
#   default, as the protocol reports RULER | retrieval = the 8 NIAH tasks | list), RULER_TAG, PAIR_SEED (20260930), SAMPLING_BACKEND (pytorch: per-request seeds with
#   top-k/top-p), TEMPERATURE (diagnostic arms only; exams use the vendor's), ALL_REPS=1 (every listed exam at every rep in
#   REP_LIST), DOC_SHARD=i/n (LiveCodeBench / HumanEval+ / LongBench v2 documents i, i+n, ... of every listed rep, each document with
#   the seed of the unsharded run; logs and rep directories carry the shard, LiveCodeBench grading covers only this shard; LCB_SHARD
#   = its earlier, LiveCodeBench-only name), SE_SHARD=i/n (gpqa_se / mmlu_se runs only: this job runs docs i, i+n, ... of every listed rep; logs and rep
#   directories carry the shard), PROFILE=1 (DUET hook timings in the server logs), LCB_PROCESSES (LiveCodeBench grader processes, 32), TRITON_CACHE (a
#   Triton cache the kernels were already compiled into; a cold cache compiles inside the first requests, and a forward that compiles
#   for more than 300 s trips SGLang's scheduler watchdog), RESUME=1 (finish the listed repetitions, and this job's RULER unit,
#   that an earlier job on this <out> left unfinished, e.g. killed at its time limit: their recorded replies are replayed, only the
#   missing ones generated), SLURM_JOB_ID (names <out>/tasks-<id>/, servers-<id>/ and report-<id>: jobs that run at the same time
#   on one <out> need distinct values; outside Slurm it is "local" unless set).
# The job fails (exit 1) when a server does not come up or fails its arm check, and after the report when a server log shows
# retracted requests (KV cache full: a retracted request leaves the uninterrupted decode both arms are compared on).
FAM=${1:?kimi|flash_next|lightning|glm}; OUT=${2:?out}; EXAMS=${3:-}
mkdir -p "$OUT"
FORK=${FORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
[ -f "$FORK/benchmark/duet_eval/launch_duet.py" ] || { echo "FORK=$FORK is not this repository (a copy of the job script needs FORK=<repository>)"; exit 1; }
DUET_DATA_ROOT=${DUET_DATA_ROOT:-/src/results/hfcache}; HF_OFFLINE=${HF_OFFLINE:-1}
EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH-/work/pyuser/lib/python3.12/site-packages:/src/results/pyextra}   # pyextra: tenacity, math_verify, sympy, latex2sympy2_extended, antlr4, mpmath
TWINSTAR_ROOT=${TWINSTAR_ROOT:-}
PP=$FORK/python:$FORK
export FORK DUET_DATA_ROOT PYTHONPATH=$PP${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HOME=${HF_HOME:-/work/hf-cache} HF_HUB_OFFLINE=$HF_OFFLINE HF_DATASETS_OFFLINE=$HF_OFFLINE
export HF_DATASETS_CACHE=$DUET_DATA_ROOT/datasets NLTK_DATA=$DUET_DATA_ROOT/nltk_data TOKENIZERS_PARALLELISM=false NVIDIA_TF32_OVERRIDE=0
export TMPDIR=$OUT/tmp XDG_CACHE_HOME=$OUT/cache TRITON_CACHE_DIR=${TRITON_CACHE:-$OUT/cache/triton} TORCH_EXTENSIONS_DIR=$OUT/cache/torch FLASHINFER_WORKSPACE_BASE=$OUT/cache/flashinfer SGLANG_CACHE_DIR=$OUT/cache/sglang
export SGLANG_DUET_HOOK_PROFILE=${PROFILE:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # no allocator fragmentation across extend-batch sizes
mkdir -p $TMPDIR $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR $FLASHINFER_WORKSPACE_BASE $SGLANG_CACHE_DIR
cd $FORK
# the protocol (tasks.py): PROTO[<exam>] = its repetitions, P0 = the P0 exams, RULER_P0 = the P0 RULER lengths
declare -A PROTO
while read -r key value; do case $key in P0|RULER_P0|OWN_EXAMS|LONG_PROMPT_EXAMS|DOC_SHARD_EXAMS) printf -v $key %s "$value" ;; *) PROTO[$key]=$value ;; esac; done < <(python - <<'PY'
from benchmark.duet_eval.lmeval import tasks
for exam, spec in tasks.EXAMS.items():
    print(exam, spec["reps"])
print("P0", ",".join(tasks.P0))
print("RULER_P0", ",".join(map(str, tasks.RULER_LENGTHS_P0)))
print("OWN_EXAMS", " ".join(tasks.OWN_PROCESS))
print("LONG_PROMPT_EXAMS", " ".join(tasks.LONG_PROMPT))
print("DOC_SHARD_EXAMS", " ".join(tasks.DOC_SHARD_EXAMS))
PY
)
[ -n "$P0" ] || { echo "cannot read the exam protocol from benchmark.duet_eval.lmeval.tasks (PYTHONPATH=$PYTHONPATH)"; exit 1; }
EXAMS=${EXAMS:-$P0}
for e in ${EXAMS//,/ }; do [ -n "${PROTO[$e]}" ] || { echo "unknown exam $e (tasks.EXAMS: ${!PROTO[*]})"; exit 1; }; done
[ "$FAM" = glm ] && { NDUET=${NDUET:-${N:-0}}; NSTOCK=${NSTOCK:-${N:-1}}; }     # one TP-4 server per node, the base arm
NDUET=${NDUET:-${N:-3}}; NSTOCK=${NSTOCK:-${N:-1}}; CTX=${CTX:-262144}; MRR=${MRR:-128}; CONC=${CONC:-$((MRR * (NDUET > NSTOCK ? NDUET : NSTOCK)))}; PAIR_SEED=${PAIR_SEED:-20260930}; RULER_LENGTHS=${RULER_LENGTHS:-$RULER_P0}
case $FAM in
  kimi)        BASE=${KIMI_BASE:-/work/models/Kimi-Linear-48B-A3B-Instruct} ;;
  flash_next)  BASE=${FN_BASE:-/work/hf-cache/nvidia-nvfp4-fc694b54} ;;
  lightning)   BASE=${LIGHTNING_BASE:-/me/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16} ;;
  glm)         BASE=${GLM_BASE:-/src/results/models/GLM-5.3-Flash} ;;
  *) echo "unknown family $FAM"; exit 1 ;;
esac
TP=${TP:-$([ "$FAM" = glm ] && echo 4 || echo 1)}; SB=${SAMPLING_BACKEND:-pytorch}     # pytorch: seeds in mixed batches (greedy RULER requests carry top_k=1; flashinfer asserts)
[ -f "$BASE/config.json" ] || { echo "$FAM base checkpoint $BASE has no config.json (KIMI_BASE / FN_BASE / LIGHTNING_BASE / GLM_BASE)"; exit 1; }
REL=${DUET_REL:-}
# glm serves its base arm without a release (launch_duet RELEASE_OPTIONAL_STOCK); every other arm names it (the base arm: identity check)
if [ -z "$REL" ] && ! { [ "$FAM" = glm ] && [ "$NDUET" = 0 ]; }; then
  echo "$FAM on $BASE: set DUET_REL to the release directory (fetch_duet.py prints it) or a Hub id <owner>/<repo>@<commit>"; exit 1
fi
# a Hub id offline: the copy launch_duet fetched earlier (sglang.srt.duet.release.fetch_release: <hf>/duet-releases/<owner>--<repo>/<commit>)
if [ -n "$REL" ] && [ ! -d "$REL" ] && [ "$HF_OFFLINE" = 1 ]; then
  REL_COPY=$HF_HOME/duet-releases/$(echo "${REL%@*}" | sed 's#/#--#g')/${REL#*@}
  [ "${REL#*@}" != "$REL" ] && [ -d "$REL_COPY" ] || { echo "DUET_REL=$REL is a Hub id and HF_OFFLINE=1: set HF_OFFLINE=0 to fetch it, or DUET_REL to a local copy (none at $REL_COPY)"; exit 1; }
  REL=$REL_COPY
fi
SBARG=""; [ -n "$SB" ] && SBARG="--sampling-backend $SB"
ARMS=${ARMS:-"duet stock"}; [ "$NSTOCK" = 0 ] && ARMS=duet; [ "$NDUET" = 0 ] && ARMS=stock
# DUET_EXTRA: extra server arguments for the duet arm only (ablations: "--decode-ssm-r 0 --decode-ssm-w 0" = DUET prefill without
# state cuts; "--no-prefill-layer-trim" = full prefill with cuts; "--decode-ssm-r 32" / "--decode-ssm-w 8" = other cut specs).
# An ablation run uses NSTOCK=0 and is paired by report_lmeval.py against the base run that used the same pair seed.
DUET_EXTRA=${DUET_EXTRA:-}
# a worktree's .git names its git dir by the host path; inside the container the checkout is /src: the git dir is the
# longest tail of that path that exists under /src
tree_git() { local g; g=$(git -C "$1" rev-parse --absolute-git-dir 2>/dev/null) && { echo "$g"; return; }; g=$(sed -n 's#^gitdir: ##p' "$1/.git" 2>/dev/null)
  while [ -n "$g" ] && [ "$g" != "${g#*/}" ]; do g=${g#*/}; [ -d "/src/$g" ] && { echo "/src/$g"; return; }; done; echo /nonexistent; }
tree_head() { git --git-dir="$(tree_git $1)" rev-parse --short HEAD 2>/dev/null || echo unknown; }
tree_dirty() { git --git-dir="$(tree_git $1)" --work-tree="$1" status --short 2>/dev/null | wc -l; }
export FORK_COMMIT=$(tree_head $FORK)       # run.py records it in run.json (git cannot resolve a host-path worktree in the container)
echo "== $(date +%F\ %T) family=$FAM base=$BASE release=$REL exams=$EXAMS arms=[$ARMS] duet_extra=[$DUET_EXTRA] servers duet=$NDUET stock=$NSTOCK TP=$TP ctx=$CTX mrr=$MRR conc=$CONC pair_seed=$PAIR_SEED sampling_backend=${SB:-default} limit=${LIMIT:-none} fork=$FORK_COMMIT ($(tree_dirty $FORK) dirty)$([ "$FAM" = flash_next ] && echo " fn_package=$([ -n "$TWINSTAR_ROOT" ] && echo "$TWINSTAR_ROOT@$(tree_head $TWINSTAR_ROOT)" || echo in-fork)")" | tee -a $OUT/job.txt
FAM=$FAM python - <<'PY' || { echo "dependency check failed"; exit 1; }
import importlib.util
import os
need = ["lm_eval", "datasets", "yaml", "aiohttp", "tenacity", "tqdm", "requests", "transformers", "pyarrow", "numpy", "sympy",
        "math_verify", "langdetect", "immutabledict", "nltk", "wonderwords", "huggingface_hub"]
if os.environ["FAM"] == "kimi":
    need += ["tiktoken", "fla"]          # Kimi tokenizer; fla = the chunked KDA kernel of the DUET emitters (without it: a per-token loop)
missing = [m for m in need if importlib.util.find_spec(m) is None]
print("python deps missing:", missing or "none")
assert not missing
from importlib.metadata import version
assert version("lm_eval") == "0.4.13", f"lm-eval {version('lm_eval')}: the task bodies and model.py's overrides are for 0.4.13"
PY
TASKS=$OUT/tasks-${SLURM_JOB_ID:-local}; python -m benchmark.duet_eval.lmeval.tasks --family $FAM --out $TASKS --exams $EXAMS --ruler-tasks ${RULER_TASKS:-all} ${TEMPERATURE:+--temperature $TEMPERATURE} > $TASKS.json || exit 1
# ---------------------------------------------------------------- servers (launch_duet --detach returns after /health)
TSR=""; [ "$FAM" = flash_next ] && [ -n "$TWINSTAR_ROOT" ] && TSR="--twinstar-root $TWINSTAR_ROOT"
SRV=$OUT/servers-${SLURM_JOB_ID:-local}; mkdir -p $SRV; gpu=0; declare -A URLS LAUNCHES LOG0
stop_all() { for f in $SRV/*/server.pid $SRV/lb-*.pid; do [ -f $f ] && kill $(cat $f) 2>/dev/null; done; pkill -f -- "--settings $SRV"; sleep 5; }
for arm in $ARMS; do
  n=$NDUET; [ $arm = stock ] && n=$NSTOCK
  extra=""; [ $arm = duet ] && [ -n "$DUET_EXTRA" ] && extra="--extra=\"$DUET_EXTRA\""     # '=' form: the value may start with '--'
  for i in $(seq 1 $n); do
    gpus=$(seq -s, $gpu $((gpu + TP - 1))); gpu=$((gpu + TP)); port=$((30600 + (gpu - TP)))
    LOG0[$arm-$i]=$(stat -c %s $SRV/$arm-$i/server.log 2>/dev/null || echo 0)     # launch_duet appends to a reused --out
    eval python -m benchmark.duet_eval.launch_duet --model $BASE ${REL:+--duet-release $REL} --arm $arm --port $port --gpus $gpus --tp $TP \
      --context-length $CTX --max-running-requests $MRR $SBARG $extra --hf-cache $HF_HOME $TSR --out $SRV/$arm-$i --detach --health-timeout 3000 \
      ${MEM_FRACTION:+--mem-fraction $MEM_FRACTION} ${WATCHDOG:+--watchdog-timeout $WATCHDOG} \
      > $SRV/$arm-$i.launch.log 2>&1 & LAUNCHES[$!]=$arm-$i
    URLS[$arm]="${URLS[$arm]:+${URLS[$arm]},}http://127.0.0.1:$port"
  done
done
FAILED=""; for pid in "${!LAUNCHES[@]}"; do wait $pid || FAILED="$FAILED ${LAUNCHES[$pid]}"; done     # launch_duet: 1 = no /health or arm check failed
if [ -n "$FAILED" ]; then
  for s in $FAILED; do echo "!! server $s failed to launch ($SRV/$s.launch.log):"; tail -n 5 $SRV/$s.launch.log; done
  stop_all; echo "EXAM LMEVAL FAILED: server launch ($(date +%H:%M))"; exit 1
fi
for arm in $ARMS; do for u in ${URLS[$arm]//,/ }; do curl -sf $u/health >/dev/null || { echo "server $u not healthy"; grep -h "Error\|error" $SRV/*.launch.log | tail -n 5; stop_all; exit 1; }; done; done
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tee $SRV/gpu-memory.txt
echo "== servers up ($(date +%H:%M))"
MAXREP=0; for e in ${EXAMS//,/ }; do [ "${PROTO[$e]}" -gt $MAXREP ] && MAXREP=${PROTO[$e]}; done; [ -n "$REPS_CAP" ] && [ "$REPS_CAP" -lt "$MAXREP" ] && MAXREP=$REPS_CAP
REP_LIST=${REP_LIST:-$(seq -s' ' 0 $((MAXREP - 1)))}
# RULER and tasks.OWN_PROCESS (LCB, AA-LCR, HumanEval+, LongBench v2) run as lm-eval processes of their own (RULER: greedy, --samples;
# LCB: system message; AA-LCR, LongBench v2: ~100K-token prompts; HumanEval+: graded by humaneval_grade.py)
GEN_EXAMS=$(echo $EXAMS | tr , '\n' | grep -v '^ruler$' | grep -vxF -f <(echo $OWN_EXAMS | tr ' ' '\n') | paste -sd,)
SHARD_TAG=${SE_SHARD:+.s${SE_SHARD/\//of}}
# DOC_SHARD=i/n: documents i, i+n, ... of each listed rep of the tasks.DOC_SHARD_EXAMS (LCB, HumanEval+, LongBench v2; run.py
# --doc-shard); LCB_SHARD is its earlier name (LiveCodeBench only, the GLM v3.5 LCB runs of 2026-10-09)
DOC_SHARD=${DOC_SHARD:-${LCB_SHARD:-}}
DOC_TAG=${DOC_SHARD:+.s${DOC_SHARD/\//of}}
# a simple-evals exam run alone logs under its name, so a job of the other exams of the same repetition on this <out> keeps its log
SE_TAG=""; case $GEN_EXAMS in gpqa_se|mmlu_se) SE_TAG=.$GEN_EXAMS ;; esac
# ---------------------------------------------------------------- RESUME=1: finish repetitions an earlier job left unfinished
# (killed at its time limit: lm-eval writes samples only when a repetition ends).  Such a repetition directory of this job (a
# claim.json or a sidecar.jsonl -- runs before 2026-10-06 wrote no claim -- and no sample file) moves to
# <out>/partial/<arm dir>/rep<k>/<kind>-<n>; the arm's balancer replays every reply recorded in those sidecars (lb.py --resume: same
# prompt, occurrence, seed and budget) and the servers generate only the missing ones.  This job's repetition directories:
# <out>/<arm>/rep<k>/<gen | gpqa_se | mmlu_se[-s<i>of<n>] | lcb | aa_lcr | humaneval_plus | longbench_v2 (DOC_SHARD: -s<i>of<n>)> for k
# in REP_LIST, and <out>/<arm>-ruler[-<tag>]/rep0/ruler
# when the job runs RULER (listed, and repetition 0 in REP_LIST).
declare -A RESUME_FILES
resume_unit() {   # resume_unit <arm> <arm directory under <out>> <rep> <kind>
  local arm=$1 adir=$2 k=$3 kind=$4 d=$OUT/$2/rep$3/$4 p=$OUT/partial/$2/rep$3 n f
  if { [ -f $d/claim.json ] || [ -f $d/sidecar.jsonl ]; } && ! compgen -G "$d/*/samples_*.jsonl" >/dev/null; then
    n=1; while [ -e $p/$kind-$n ]; do n=$((n + 1)); done
    mkdir -p $p; mv $d $p/$kind-$n
    echo "== resume: $d (claim $(cat $p/$kind-$n/claim.json 2>/dev/null | tr -d '\n')) -> partial/$adir/rep$k/$kind-$n" | tee -a $OUT/job.txt
  fi
  for f in $p/$kind-[0-9]*/sidecar.jsonl; do
    [ -f $f ] && { RESUME_FILES[$arm]="${RESUME_FILES[$arm]:+${RESUME_FILES[$arm]},}$f"; echo "   $adir rep$k $kind: $(wc -l < $f) replies in $f"; }
  done
}
if [ "${RESUME:-0}" = 1 ]; then
  # run.py's kind: one simple-evals exam alone -> its name, any other set -> gen; SE_SHARD adds -s<i>of<n> to either
  KINDS=""; [ -n "$GEN_EXAMS" ] && { case $GEN_EXAMS in *,*) KINDS=gen ;; *_se) KINDS=$GEN_EXAMS ;; *) KINDS=gen ;; esac; KINDS=$KINDS${SE_SHARD:+-s${SE_SHARD/\//of}}; }
  for e in $OWN_EXAMS; do echo ",$EXAMS," | grep -q ",$e," || continue
    case " $DOC_SHARD_EXAMS " in *" $e "*) KINDS="$KINDS $e${DOC_SHARD:+-s${DOC_SHARD/\//of}}" ;; *) KINDS="$KINDS $e" ;; esac; done
  for arm in $ARMS; do
    for k in $REP_LIST; do for kind in $KINDS; do resume_unit $arm $arm $k $kind; done; done
    if echo ",$EXAMS," | grep -q ",ruler," && echo " $REP_LIST " | grep -q " 0 "; then resume_unit $arm $arm-ruler${RULER_TAG:+-$RULER_TAG} 0 ruler; fi
  done
fi
# ---------------------------------------------------------------- balancers
LBP=31001; declare -A LB
for arm in $ARMS; do
  setsid python -m benchmark.duet_eval.lmeval.lb --port $LBP --backends ${URLS[$arm]} ${RESUME_FILES[$arm]:+--resume ${RESUME_FILES[$arm]}} \
    > $SRV/lb-$arm.log 2>&1 < /dev/null & echo $! > $SRV/lb-$arm.pid
  LB[$arm]=http://127.0.0.1:$LBP; LBP=$((LBP + 1))
done
for arm in $ARMS; do for i in $(seq 1 30); do curl -sf ${LB[$arm]}/health >/dev/null && break; sleep 2; [ $i = 30 ] && { echo "balancer $arm not healthy"; cat $SRV/lb-$arm.log; stop_all; exit 1; }; done; done
# ---------------------------------------------------------------- repetitions: one lm-eval process per (rep, arm), all in parallel
echo "== lm-eval: repetitions [$REP_LIST] x arms [$ARMS], $CONC requests in flight each ($(date +%H:%M))"
PIDS=""
for k in $REP_LIST; do [ -z "$ALL_REPS" ] && [ "$k" -ge "$MAXREP" ] && continue; [ -z "$GEN_EXAMS" ] && continue; for arm in $ARMS; do
  python -m benchmark.duet_eval.lmeval.run --family $FAM --arm $arm --base-url ${LB[$arm]} --tokenizer $BASE --out $OUT/$arm --rep $k \
    --exams $GEN_EXAMS --num-concurrent $CONC --max-length $CTX --pair-seed $PAIR_SEED --tasks-dir $TASKS ${ALL_REPS:+--all-reps} \
    ${LIMIT:+--limit $LIMIT} ${SE_SHARD:+--doc-shard $SE_SHARD} > $OUT/$arm$SE_TAG.rep$k$SHARD_TAG.log 2>&1 & PIDS="$PIDS $!"
done; done
# the exams of tasks.OWN_PROCESS, one process per (exam, rep, arm).  LongBench v2 and AA-LCR: lm-eval's max_length is raised by the
# generation ceiling so it never cuts a ~100K-token prompt; the budget per request is the server context minus the prompt (model.py)
for e in $OWN_EXAMS; do echo ",$EXAMS," | grep -q ",$e," || continue
  len="--max-length $CTX"; case " $LONG_PROMPT_EXAMS " in *" $e "*) len="--max-length $((CTX + 229376)) --server-context $CTX" ;; esac
  shard=""; tag=""; case " $DOC_SHARD_EXAMS " in *" $e "*) shard=${DOC_SHARD:+--doc-shard $DOC_SHARD}; tag=$DOC_TAG ;; esac
  for k in $REP_LIST; do [ -z "$ALL_REPS" ] && [ "$k" -ge "${PROTO[$e]}" ] && continue; for arm in $ARMS; do
    python -m benchmark.duet_eval.lmeval.run --family $FAM --arm $arm --base-url ${LB[$arm]} --tokenizer $BASE --out $OUT/$arm --rep $k \
      --exams $e --num-concurrent $CONC $len --pair-seed $PAIR_SEED --tasks-dir $TASKS ${ALL_REPS:+--all-reps} \
      ${LIMIT:+--limit $LIMIT} $shard > $OUT/$arm.$e.rep$k$tag.log 2>&1 & PIDS="$PIDS $!"
  done; done
done
if echo ",$EXAMS," | grep -q ",ruler," && ! echo " $REP_LIST " | grep -q " 0 "; then
  echo "!! WARNING: RULER is listed but REP_LIST [$REP_LIST] lacks repetition 0 (RULER runs once, as repetition 0): this job does NOT run RULER" | tee -a $OUT/job.txt
fi
if echo ",$EXAMS," | grep -q ",ruler," && echo " $REP_LIST " | grep -q " 0 "; then for arm in $ARMS; do
  python -m benchmark.duet_eval.lmeval.run --family $FAM --arm $arm --base-url ${LB[$arm]} --tokenizer $BASE --out $OUT/$arm-ruler${RULER_TAG:+-$RULER_TAG} --rep 0 \
    --exams ruler --num-concurrent $CONC --max-length $((CTX + 1024)) --server-context $CTX --pair-seed $PAIR_SEED --ruler-lengths $RULER_LENGTHS \
    --tasks-dir $TASKS --ruler-tasks ${RULER_TASKS:-all} --ruler-data $DUET_DATA_ROOT/ruler --ruler-per-length ${RULER_N:-100} \
    ${RULER_BUILD:+--ruler-build-samples $RULER_BUILD} ${RULER_CACHE:+--ruler-cache $RULER_CACHE} > $OUT/$arm.ruler${RULER_TAG:+-$RULER_TAG}.log 2>&1 & PIDS="$PIDS $!"
done; fi
NFAIL=0; for p in $PIDS; do wait $p || NFAIL=$((NFAIL + 1)); done     # every lm-eval process's own status (a bare wait hid crashes)
[ $NFAIL -gt 0 ] && echo "!! $NFAIL lm-eval process(es) exited non-zero; their logs: $OUT/*.log" | tee -a $OUT/job.txt
# only this job's logs: <out> is shared with earlier jobs (*.rep$k.log also matches <arm>.lcb.rep$k.log and <arm>.<gpqa_se|mmlu_se>.rep$k.log;
# RULER runs with repetition 0)
grep -h "REP DONE\|Error\|error:" $(for k in $REP_LIST; do echo $OUT/*.rep$k$SHARD_TAG.log ${DOC_TAG:+$OUT/*.rep$k$DOC_TAG.log}; done) \
  $(echo " $REP_LIST " | grep -q " 0 " && echo $OUT/*.ruler*.log) 2>/dev/null | grep -v "Warning" | tail -n 20
for arm in $ARMS; do curl -s ${LB[$arm]}/stats; echo; done | tee $SRV/lb-stats.txt
if echo ",$EXAMS," | grep -q ",lcb,"; then
  echo "== LiveCodeBench grading (official grader, $(nproc) cpus) ($(date +%H:%M))"
  # grade only this job's repetitions (other jobs sharing <out> may still be writing theirs)
  python -m benchmark.duet_eval.lmeval.lcb_grade --tests $DUET_DATA_ROOT/lcb/lcb_v6_2408_2505_tests.pkl --processes ${LCB_PROCESSES:-32} \
    $(for arm in $ARMS; do for k in $REP_LIST; do d=$OUT/$arm/rep$k${DOC_SHARD:+/lcb-s${DOC_SHARD/\//of}}; [ -d $d ] && echo $d; done; done) 2>&1 | grep -v Warning | tail -n 12
fi
# HumanEval+ is graded once every repetition of both arms is written, both arms in one EvalPlus call (humaneval_grade.py; its
# packages: requirements-evalplus.txt), not here: the report shows the HumanEval+ cells after that grading
echo ",$EXAMS," | grep -q ",humaneval_plus," && echo "== HumanEval+ written; grade both arms together: python -m benchmark.duet_eval.lmeval.humaneval_grade --out $OUT/humaneval-grade $OUT/stock $OUT/duet"
# ---------------------------------------------------------------- report
compgen -G "$OUT/stock*" >/dev/null && python -m benchmark.duet_eval.lmeval.report_lmeval --arms stock=$OUT/stock duet=$OUT/duet --ruler $(for d in $OUT/stock-ruler*; do [ -d $d ] && echo stock=$d; done) $(for d in $OUT/duet-ruler*; do [ -d $d ] && echo duet=$d; done) --md --out $OUT/report-${SLURM_JOB_ID:-local}.json | tee $OUT/report-${SLURM_JOB_ID:-local}.md
grep -h "DUET-HOOK" $SRV/duet-*/server.log 2>/dev/null | tail -n 4
stop_all
# SGLang logs "KV cache pool is full. Retract requests." (scheduler.py) when it retracts running requests
RETRACTED=$(for s in "${!LOG0[@]}"; do n=$(tail -c +$((LOG0[$s] + 1)) $SRV/$s/server.log 2>/dev/null | grep -c "Retract"); [ "$n" -gt 0 ] && echo "$s: $n"; done)
if [ -n "$RETRACTED" ]; then
  echo "!! EXAM LMEVAL FAILED: requests were retracted (KV cache full); server: lines with 'Retract' in its server.log" | tee -a $OUT/job.txt
  echo "$RETRACTED" | tee -a $OUT/job.txt
  echo "!! a retracted request leaves the uninterrupted decode both arms are compared on: rerun these repetitions with a lower MRR ($(date +%H:%M))"
  exit 1
fi
if [ $NFAIL -gt 0 ]; then
  echo "!! EXAM LMEVAL FAILED: $NFAIL lm-eval process(es) exited non-zero ($(date +%H:%M))" | tee -a $OUT/job.txt
  exit 1
fi
echo "EXAM LMEVAL DONE ($(date +%H:%M))"
