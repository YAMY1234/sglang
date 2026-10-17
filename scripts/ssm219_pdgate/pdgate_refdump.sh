#!/bin/bash
# HOLDER_GPUS=2
# HOLDER_EXCL=0
# HOLDER_STAGE=0
# HOLDER_MINUTES=35
# docs/222 §15 PD gate: S_fp32 teacher-forced reference dump over an extra window set (WS-4 P6 (b) long set, lead 22:43),
# with the recipe of the standard set's reference (docs/218 kl-fdt-2, job 4538682): the agg S32 server of that job
# (v7b-s32.{env,argv,json}, v7b.mounts, release copy rel-v7stock) on 2 GPUs, acc_audit_kl dump (top 8192 + tail) of
# every forced step.  Only the ports change (30101/30111 are inside the nodes' ephemeral range -> 65311/65315).
# Gates before the dump: the pool must be fp32 (ssm_state 25.37GB, as kl-fdt-2); windows sha must match the plan.
# Holder inbox body; the wrapper sets L (dir with windows-<name>.json; ref-s32/ is created there) and NAME.
#   qsubmit.sh <wrapper> --gpus 2 --minutes 35        wrapper:  L=<dir>; NAME=long; source <this file>
set -uo pipefail
: "${L:?L}" "${NAME:?NAME}" "${HOLDER_JOB:?HOLDER_JOB}"
W=/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
F=$W/results/fp32-state-20261015/kl-fdt-2
source ${PDGATE_HLIB:-$W/results/ssm-kernel-219/holder/scripts/hlib.sh}
IMG=$(h_image)
log() { echo "REFDUMP $(date -u +%FT%TZ) $*" | tee -a $L/refdump.log; }
[[ -d $L/ref-s32 ]] && { log "ref-s32 exists in $L: refusing to overwrite"; exit 2; }
log "START holder=$HOLDER_JOB node=$(hostname) cvd=${HOLDER_CVD:-} windows=$(sha256sum $L/windows-$NAME.json | cut -c1-16)"
# the job's own copy of the S32 launch, ports moved out of the ephemeral range
sed -e 's/--port 30101/--port 65311/' -e 's/--nccl-port 30111/--nccl-port 65315/' $F/v7b-s32.argv > $L/s32.argv
grep -q -- '--port 65311' $L/s32.argv && grep -q -- '--nccl-port 65315' $L/s32.argv || { log "port rewrite failed"; exit 3; }
MODEL=$(h_set model $W/results/bf16-regress-20261013/staging/model)  # the holder's verified local copy when staged
MNT="$(sed -e "s|$F:/out|$F:/out:ro|" -e "s|$W/results/bf16-regress-20261013/staging/model:/model|$MODEL:/model|" $F/v7b.mounts),$L:/o"
srun --overlap --jobid=$HOLDER_JOB -N1 -n1 --gres=gpu:4 --cpus-per-task=48 --mem=300G --container-image="$IMG" \
  --container-mounts="$MNT" --no-container-mount-home --container-workdir=/o bash -c "
    python3 -c 'import json;[print(d) for d in json.load(open(\"/out/v7b-s32.json\"))[\"mkdirs\"]]' | xargs -r mkdir -p
    set -a; . /out/v7b-s32.env; set +a
    export CUDA_VISIBLE_DEVICES=$HOLDER_CVD
    eval exec \"\$(cat /o/s32.argv)\"" > $L/srv-s32.log 2>&1 &
SP=$!
srun --overlap --jobid=$HOLDER_JOB -N1 -n1 --gres=gpu:4 --cpus-per-task=8 --mem=64G --container-image="$IMG" \
  --container-mounts="$W:/work:ro,$F:/out:ro,$L:/o" --no-container-mount-home --container-workdir=/o bash -c "
    set -uo pipefail
    export PYTHONPATH=/out/tools:/work/results/bf16-regress-20261013/source-7a/python:/work/pyuser/lib/python3.12/site-packages TMPDIR=/tmp
    t0=\$(date +%s)
    until curl -sf -m 10 http://127.0.0.1:65311/get_model_info >/dev/null; do
      [ \$(( \$(date +%s) - t0 )) -gt 1500 ] && { echo TIMEOUT; exit 3; }
      grep -q -E 'Scheduler hit an exception|Received sigquit|Capture cuda graph failed' /o/srv-s32.log && { echo DEAD; exit 4; }
      sleep 10; done
    echo \"READY after \$(( \$(date +%s) - t0 )) s\"
    grep -q 'ssm_state size: 25.37GB' /o/srv-s32.log || { echo 'ABORT pool is not fp32'; exit 7; }
    curl -s -m 900 http://127.0.0.1:65311/generate -H 'Content-Type: application/json' \
      -d '{\"text\": \"Hello, warm-up.\", \"sampling_params\": {\"max_new_tokens\": 32, \"temperature\": 0}}' >/dev/null
    /opt/sglang/bin/python -m acc_audit_kl score --endpoint http://127.0.0.1:65311 --windows /o/windows-$NAME.json \
      --arm s32 --mode dump --out-root /o --conc 16 || exit 5
    echo DUMP-DONE" > $L/client.log 2>&1
RC=$?; kill $SP 2>/dev/null; sleep 20; kill -9 $SP 2>/dev/null
log "END rc=$RC $(grep -E 'READY|ABORT|SCORE|DUMP-DONE|TIMEOUT|DEAD' $L/client.log | tr '\n' ' ')"
[[ $RC == 0 ]] && ls -la $L/ref-s32 | tee -a $L/refdump.log
exit $RC
