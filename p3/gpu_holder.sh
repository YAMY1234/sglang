# HOLDER_GPUS=2
# HOLDER_EXCL=0
# HOLDER_STAGE=1
# HOLDER_MINUTES=30
# Existing allocation only: exactly the assigned two visible GPUs do work.
set -Eeuo pipefail
source /lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911/results/ssm-kernel-219/holder/scripts/hlib.sh
: "${P3_TREE:?P3_TREE}"
P3_STAGE=${P3_STAGE:-p5}
P3_MODE=${P3_MODE:-micro}
A=$W/results/ssm-kernel-219
O=${P3_OUTPUT:-$A/p3-own/$P3_STAGE-h$HOLDER_JOB-$(basename "$RUN_DIR")}
G=${P3_GATE:-$A/pdgate/p3b-9827e05cb25}
mkdir -p "$O"
MODEL=$(h_set model "$W/results/bf16-regress-20261013/staging/model")
MOUNTS=$(sed "s|@MODEL@|$MODEL|" "$G/g-cand1.mounts")
MOUNTS="$MOUNTS,$P3_TREE:/p3/src:ro,$O:/p3out"
h_mounts() { echo "$MOUNTS"; }
echo "STEP START holder=$HOLDER_JOB cvd=$HOLDER_CVD stage=$P3_STAGE mode=$P3_MODE tree=$P3_TREE out=$O at=$(date -u +%FT%TZ)"
cat "$P3_TREE/SHA" > "$O/SHA"
H_MEM=400G H_CPUS=32 h_srun "$HOLDER_CVD" \
    "export CUDA_VISIBLE_DEVICES=$HOLDER_CVD P3_RUN_ID=$(basename "$RUN_DIR") P3_STAGE=$P3_STAGE P3_MODE=$P3_MODE; bash /p3/src/p3/gpu_inner.sh" 2>&1 | tee "$O/driver.log"
echo "STEP END holder=$HOLDER_JOB stage=$P3_STAGE at=$(date -u +%FT%TZ) out=$O"
