#!/bin/bash
# docs/222 §20 P12 pd1n job (holder inbox / mini, $HOLDER_JOB): bash p12_hjob.sh <kind> <package fp>
# One node: P TP2 on GPU0-1 (task 0), D TP2 on GPU2-3 (task 1).  Each run is the unchanged j4569012 per-arm harness
# (pkg support/job.sh, as trio.sh starts it for one arm; P12_ONE_NODE=1 gives D its own node-local root), one after another:
#   positive  PC+ arm 1, P12_PROOF=positive                 -> p12_wire_client positive; p12_wire_check --snapshots
#   negative  PC+ arm 1 event; PC+ arm 1 abort (two restarts) -> client event / abort;  p12_wire_check --negative x2
#   micro     C+ 300 s; PC+ arm 1 300 s; PC+ arm 0 300 s       -> AgentX C8 (window client), no proof keys
#   ab        PC+ arm 0 600 s; PC+ arm 1 600 s                 -> AgentX C8 (window client), no proof keys
# Runs land in results/p3pd-20261017/runs/j<job>-<kind>-<label>[/remote-node]; job log + summary in $OUT.
# Lines: P12HJOB RUN <label> rc=..; P12GATE_DRIVER {..} (instance log); P12_LIVE_WIRE_TORCH_EQUAL_PASS|FAIL {..};
#        P12RUN <label> {..} (p12_runcheck.py); P12HJOB END rc=..
set -uo pipefail
KIND=$1; FP=$2
W=/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
R=$W/results/p3pd-20261017; B=$R/inputs/pkg-$FP
T=${P12_JOBDIR:-$W/results/ssm-kernel-219/ws2/p12/pkg-$FP}   # not dirname $0: plain sbatch runs a spool copy (LHR 252802)
JOB=${HOLDER_JOB:-${SLURM_JOB_ID:?}}
NODE=$(hostname -s)
OUT=$R/p12/$KIND-$JOB-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT
IMG=$W/containers/lmsysorg+sglang+nightly-dev-cu13-20260913-fa663e72-arm64-enroot.sqsh
SRC=/work/results/p3pd-20261017/inputs/src-p12-51d15dce06c
log() { echo "P12HJOB $(date -u +%FT%TZ) $*" | tee -a $OUT/job.log; }
log "START kind=$KIND fp=$FP job=$JOB node=$NODE cvd=${HOLDER_CVD:-} run_dir=${RUN_DIR:-} tools=$T"
[[ -z ${HOLDER_JOB:-} || ${HOLDER_CVD:-} == 0,1,2,3 ]] || { log "needs the whole node (HOLDER_CVD=${HOLDER_CVD:-})"; exit 2; }
[[ -f $B/manifest.json && -f $R/CPU-ready-$FP.json ]] || { log "package or its CPU gate receipt missing"; exit 2; }
for v in P12_ARM P12_PROOF P12_WINDOW_SECONDS SGLANG_FLASHNEXT_PD_STAGING_PROOF_DIR SGLANG_FLASHNEXT_PD_STAGING_PROOF_TENSORS \
         SGLANG_GDN_PD_P12_WIRE_REPLAY SGLANG_GDN_PD_P12_WIRE_NEGATIVE SGLANG_GDN_PD_SHALLOW_SEND_PIPELINE; do unset $v; done

gpus_idle() {
  local n i
  for i in $(seq 1 36); do
    n=$(srun --overlap --jobid=$JOB -N1 -n1 -w $NODE --gres=gpu:4 nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
    [[ $n == 0 ]] && return 0; sleep 5
  done
  return 1
}

run_one() {  # run_one <label> <variant> <arm|-> <proof|-> <seconds|->
  local lab=$1 variant=$2 arm=$3 proof=$4 secs=$5 parm pocs=1 fix=1 prelude rc
  case $variant in C+) parm=factor; prelude=Cplus ;; PC+) parm=A; prelude=PCplus ;; *) log "bad variant $variant"; return 2 ;; esac
  gpus_idle || { log "RUN $lab GPUS-BUSY"; return 9; }
  log "RUN $lab variant=$variant arm=$arm proof=$proof seconds=$secs run=j$JOB-$KIND-$lab"
  (
    export PARITY_BUNDLE=$B PARITY_FORK_SHA=7a841e4189445d108085423c3d6054bd9a7b486c \
      FINAL_FORK=/work/results/remeasure-v6-fivearm-tp1-20261004/inputs/source-42bbc39e-capsule-f28999a8 PDREF_MODE=grid \
      RECON_MODE=PD V3_DEPLOYMENT=PD V3_MODEL=bf16 V3_SERVING_GPUS=4 PSIDE_C=8 REMEASURE_REPEAT=1 PARITY_STOCK_DTYPE=bfloat16 \
      B35_PAIR_STAGE=0 PFACTOR3_PAIR=0 REMEASURE_COHORT=ladder SGLANG_FLASHNEXT_ALLOFF_PREFILL_GRAPH=1 \
      SGLANG_GDN_DEFER_SHALLOW_FACTOR_PLAN=1 REMEASURE_GPU_AUTHORITY=V6_OVERLAP_ON V6_HARNESS_DRY_RUN=0 V6_GPU_ADMISSION=PASS
    export V6_VARIANT=$variant PSIDE_ARM=$parm PARITY_ARM_LABEL=$parm
    export REMEASURE_PROMPT_ONLY_STATE_CACHE=$pocs SGLANG_GDN_PROMPT_ONLY_STATE_CACHE=$pocs
    export SGLANG_GDN_K31_MIXED_EIGH=$fix SGLANG_GDN_PREFILL_RESTORE_GRAPH=$fix
    export REMEASURE_SEGMENT_ID=$JOB-$KIND-$lab TRIO_LABEL=$lab
    unset TRIO_P_NODE TRIO_D_NODE   # one node: entry.py / archive_node.py pick D by SLURM_PROCID == 1
    export TRIO_P_DEVICES=0,1 TRIO_D_DEVICES=2,3 TRIO_PD_NODE_LIST=$NODE,$NODE P12_ONE_NODE=1
    export TRIO_SRUN_ARGS="--overlap --jobid=$JOB -N1 -n2 --ntasks-per-node=2 --cpus-per-task=64 --mem=0 --gres=gpu:4 -w $NODE"
    [[ $arm != - ]] && export P12_ARM=$arm
    [[ $proof != - ]] && export P12_PROOF=$proof
    [[ $secs != - ]] && export P12_WINDOW_SECONDS=$secs
    srun --overlap --jobid=$JOB -N1 -n1 -w $NODE --cpus-per-task=1 \
      python3 "$B/support/prelude/cache_seed_node.py" "$B/support/prelude/$prelude.json" || exit 7
    exec bash "$B/support/job.sh"
  ) > $OUT/instance-$lab.log 2>&1
  rc=$?
  log "RUN $lab rc=$rc $(grep -h -E 'P12GATE_DRIVER|REMEASURE_CELL_COMPLETE' $OUT/instance-$lab.log | cut -c1-400 | tr '\n' ' ')"
  return $rc
}

check_tensors() {  # check_tensors <out json> <args...>: P3's checker in the image (torch), over the archived proof dirs
  local out=$1; shift
  srun --overlap --jobid=$JOB -N1 -n1 -w $NODE --cpus-per-task=16 --mem=64G --container-image="$IMG" \
    --container-mounts="$W:/work:ro,$R:/work/results/p3pd-20261017" --no-container-mount-home --container-workdir=/tmp \
    /opt/sglang/bin/python $SRC/p3/p12_wire_check.py "$@" --out $out 2>&1 | tee -a $OUT/job.log | grep -E '^P12_LIVE_WIRE'
  return ${PIPESTATUS[0]}
}

run() { echo /work/results/p3pd-20261017/runs/j$JOB-$KIND-$1; }
rc=0
keep() { local r=$?; [[ $rc == 0 && $r != 0 ]] && rc=$r; return 0; }   # first failure wins; later runs still run
case $KIND in
  positive)
    run_one pos PC+ 1 positive -; keep
    check_tensors $(run pos)/p12-gate-equal.json --snapshots "$(run pos)/p12-proof/**/*.pt"; keep ;;
  negative)
    run_one event PC+ 1 event -; keep
    run_one abort PC+ 1 abort -; keep
    check_tensors $(run abort)/p12-gate-negative.json --negative "$(run event)/p12-proof/**/*.pt" \
      --negative "$(run abort)/p12-proof/**/*.pt"; keep ;;
  micro)
    for x in "Cp C+ - - 300" "PCp1 PC+ 1 - 300" "PCp0 PC+ 0 - 300"; do run_one $x; keep; done ;;
  ab)
    for x in "PCp0 PC+ 0 - 600" "PCp1 PC+ 1 - 600"; do run_one $x; keep; done ;;
  *) log "bad kind $KIND"; exit 2 ;;
esac
for d in $R/runs/j$JOB-$KIND-*; do
  [[ -d $d ]] && python3 $T/p12_runcheck.py $d --kind $KIND 2>&1 | tee -a $OUT/job.log | grep -E '^P12RUN' >/dev/null
done
log "END rc=$rc runs=$(ls -d $R/runs/j$JOB-$KIND-* 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
exit $rc
