#!/bin/bash
# docs/222 §15 PD serving numerics gate: holder inbox job body (sourced by the per-gate wrapper from pdgate_submit.sh,
# which sets OUT; queued with --gpus 4 --excl --stage).  One phase per arm in plan.json order (before, cand1 [, cand2]):
#   host  : seed the arm's compile caches from its package prelude (as the window's trio.sh does)
#   GPU0-1: P TP2, GPU2-3: D TP2, both from g-<arm>-<role>.{env,argv} in the window's container mounts (+ /out)
#   client: pdgate_phase.sh (ready -> PD smoke -> KL 48x256 -> GSM200), then the pair is stopped and the GPUs must drain
# then pdgate_summary.py -> gate-summary.json / summary.json (agg gate schema + PD role split + activation lines).
set -uo pipefail
: "${OUT:?OUT}" "${HOLDER_JOB:?HOLDER_JOB}"
source /lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911/results/ssm-kernel-219/holder/scripts/hlib.sh
IMG=$(h_image); MODEL=$(h_set model $R/staging/model); BH=$(hostname)
log() { echo "PDGATE $(date -u +%FT%TZ) $*" | tee -a $OUT/job.log; }
log "START holder=$HOLDER_JOB node=$BH run=${RUN_DIR:-} cvd=${HOLDER_CVD:-} image=$IMG model=$MODEL"
[[ ${HOLDER_CVD:-} == 0,1,2,3 ]] || { log "needs the whole node (HOLDER_CVD=${HOLDER_CVD:-})"; exit 2; }
PHASES=$(python3 -c "import json;print(' '.join(json.load(open('$OUT/plan.json'))['run_phases']))")  # before may be reused
gpu_idle() {
  local i n
  for i in $(seq 1 36); do
    n=$(srun --overlap --jobid=$HOLDER_JOB -N1 -n1 --gres=gpu:4 nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
    [[ $n == 0 ]] && return 0; sleep 5
  done; return 1
}
serve() {  # serve <arm> <role> -> pid of the srun step
  local m g
  m=$(sed "s|@MODEL@|$MODEL|" $OUT/g-$1.mounts)
  g=$(python3 -c "import json;print(json.load(open('$OUT/g-$1-$2.json'))['gpus'])")
  srun --overlap --jobid=$HOLDER_JOB -N1 -n1 --gres=gpu:4 --cpus-per-task=48 --mem=300G \
    --container-image="$IMG" --container-mounts="$m" --no-container-mount-home --container-workdir=/out bash -c "
      python3 -c 'import json;[print(d) for d in json.load(open(\"/out/g-$1-$2.json\"))[\"mkdirs\"]]' | xargs -r mkdir -p
      set -a; . /out/g-$1-$2.env; set +a
      export CUDA_VISIBLE_DEVICES=$g
      env | grep -E '^(SGLANG|TWINSTAR|PD|PDGATE|V4)_' | sort | sed 's/^/PDGATE_ENV /'
      eval exec \"\$(cat /out/g-$1-$2.argv)\"" > $OUT/srv-$1-$2.log 2>&1 &
  echo $!
}
client() {  # client <command...>: the image, /out and /work read-only, no GPU work
  srun --overlap --jobid=$HOLDER_JOB -N1 -n1 --gres=gpu:4 --cpus-per-task=8 --mem=64G --container-image="$IMG" \
    --container-mounts="$OUT:/out,$W:/work:ro" --no-container-mount-home --container-workdir=/out "$@"
}
RC=0
for ARM in $PHASES; do
  python3 $OUT/tools/pdgate_seed.py $OUT $ARM > $OUT/seed-$ARM.log 2>&1 || { log "SEED-FAIL $ARM $(tail -1 $OUT/seed-$ARM.log)"; RC=8; break; }
  gpu_idle || { log "GPUS-BUSY before $ARM"; RC=9; break; }
  log "PHASE $ARM start"
  SP=$(serve $ARM P); SD=$(serve $ARM D)
  client bash /out/tools/pdgate_phase.sh $ARM $BH > $OUT/phase-$ARM.log 2>&1
  rc=$?; log "PHASE $ARM rc=$rc $(grep -E 'READY|SMOKE|SCORE|GSM|PHASE|DEAD|TIMEOUT|FAIL' $OUT/phase-$ARM.log | tr '\n' ' ')"
  kill $SP $SD 2>/dev/null
  for i in $(seq 1 60); do kill -0 $SP 2>/dev/null || kill -0 $SD 2>/dev/null || break; sleep 2; done
  kill -9 $SP $SD 2>/dev/null
  [[ $rc == 0 ]] || { RC=$rc; break; }
done
client python3 /out/tools/pdgate_summary.py /out > $OUT/summary.log 2>&1
log "END rc=$RC $(tail -1 $OUT/summary.log)"
exit $RC
