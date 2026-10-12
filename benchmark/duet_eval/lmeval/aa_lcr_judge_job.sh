#!/bin/bash
# AA-LCR grading (aa_lcr_judge.py) on one node: four TP1 servers of a judge model (a base checkpoint, no DUET), the official v1.1
# equality check over every reply of the given arm directories, both arms with the same judge and settings.
#   bash aa_lcr_judge_job.sh <judge: fn|kimi> <out dir> <arm dir> [<arm dir> ...]
# fn   = Qwen3.8-Flash-Next NVFP4 base, thinking at reasoning_effort medium (the official checker reasons at medium effort),
#        the vendor's sampling;  kimi = Kimi-Linear-48B-A3B-Instruct base (no thinking mode), greedy.
# Submit as: cp aa_lcr_judge_job.sh $OUT/job-$SLURM_JOB_ID.sh; FORK=<repository> bash $OUT/job-$SLURM_JOB_ID.sh ... > $OUT/job-$SLURM_JOB_ID.log 2>&1
# Paths (env; the defaults after '=' are the owner's cluster, as in exam_lmeval_job.sh): FORK (default: the repository this script
#   is in), HF_HOME (=/work/hf-cache), EXTRA_PYTHONPATH (=/work/pyuser/lib/python3.12/site-packages:/src/results/pyextra),
#   FN_BASE (=/work/hf-cache/nvidia-nvfp4-fc694b54), KIMI_BASE (=/work/models/Kimi-Linear-48B-A3B-Instruct), DUET_REL (the
#   release launch_duet verifies for the judge's stock-arm launch, =/src/release/duet-flash-next-nvfp4-v3.5 | duet-kimi-v3.5)
JUDGE=${1:?judge fn|kimi}; OUT=${2:?out dir}; shift 2
mkdir -p "$OUT"
FORK=${FORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
[ -f "$FORK/benchmark/duet_eval/launch_duet.py" ] || { echo "FORK=$FORK is not this repository (a copy of the job script needs FORK=<repository>)"; exit 1; }
HF_HOME=${HF_HOME:-/work/hf-cache}; EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH-/work/pyuser/lib/python3.12/site-packages:/src/results/pyextra}
export FORK PYTHONPATH=$FORK/python:$FORK${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 HF_HOME HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false NVIDIA_TF32_OVERRIDE=0
export TMPDIR=$OUT/tmp XDG_CACHE_HOME=$OUT/cache TRITON_CACHE_DIR=${TRITON_CACHE:-$OUT/cache/triton} TORCH_EXTENSIONS_DIR=$OUT/cache/torch FLASHINFER_WORKSPACE_BASE=$OUT/cache/flashinfer SGLANG_CACHE_DIR=$OUT/cache/sglang
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $TMPDIR $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR $FLASHINFER_WORKSPACE_BASE $SGLANG_CACHE_DIR
cd $FORK
case $JUDGE in
  fn)    BASE=${FN_BASE:-/work/hf-cache/nvidia-nvfp4-fc694b54}; REL=${DUET_REL:-/src/release/duet-flash-next-nvfp4-v3.5}
         KW='{"enable_thinking": true, "reasoning_effort": "medium"}'; SAMPLING='{"temperature": 1.0, "top_p": 0.95, "top_k": 20}' ;;
  kimi)  BASE=${KIMI_BASE:-/work/models/Kimi-Linear-48B-A3B-Instruct}; REL=${DUET_REL:-/src/release/duet-kimi-v3.5}; KW=""; SAMPLING='{"temperature": 0.0}' ;;
  *) echo "unknown judge $JUDGE"; exit 1 ;;
esac
echo "== $(date +%F\ %T) AA-LCR judge $JUDGE ($BASE) over $# arm dirs; fork $(git rev-parse --short HEAD) ($(git status --short | wc -l) dirty)" | tee -a $OUT/job.txt
SRV=$OUT/servers-${SLURM_JOB_ID:-local}; mkdir -p $SRV; PIDS=""; EPS=""
for g in 0 1 2 3; do
  python -m benchmark.duet_eval.launch_duet --model $BASE --duet-release $REL --arm stock --port $((30700 + g)) --gpus $g --tp 1 \
    --context-length 131072 --max-running-requests 128 --sampling-backend pytorch --hf-cache $HF_HOME --out $SRV/judge-$g --detach --health-timeout 3000 \
    > $SRV/judge-$g.launch.log 2>&1 & PIDS="$PIDS $!"
  EPS="${EPS:+$EPS,}http://127.0.0.1:$((30700 + g))"
done
wait $PIDS
for u in ${EPS//,/ }; do curl -sf $u/health >/dev/null || { echo "server $u not healthy"; tail -n 5 $SRV/*.launch.log; exit 1; }; done
echo "== judge servers up ($(date +%H:%M))"
python -m benchmark.duet_eval.lmeval.aa_lcr_judge --endpoints $EPS --tag $JUDGE --runs "$@" ${KW:+--chat-kwargs "$KW"} --sampling "$SAMPLING" 2>&1 | grep -v Warning
for f in $SRV/*/server.pid; do [ -f $f ] && kill $(cat $f) 2>/dev/null; done; pkill -f -- "--settings $SRV"; sleep 5
echo "AA-LCR JUDGE JOB DONE ($(date +%H:%M))"
