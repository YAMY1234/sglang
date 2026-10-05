#!/bin/bash
#SBATCH --job-name=r12-qwen38-e2e
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=144
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/yangminl/pp-verify-20260917/R12-kimi-tep8-vs-pp4/qwen-e2e/logs/%x-%j.out

set -euo pipefail

R12_ROOT=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/yangminl/pp-verify-20260917/R12-kimi-tep8-vs-pp4
if [[ "${R12_QWEN_DRY_RUN:-0}" == 1 ]]; then
  SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
else
  SCRIPT_DIR=${R12_QWEN_HARNESS_DIR:-$R12_ROOT/qwen-e2e/sbatch}
fi
source "$SCRIPT_DIR/r12_qwen_e2e_lib.sh"
source "$SCRIPT_DIR/r12_qwen_runtime_lib.sh"
source "$SCRIPT_DIR/r12_point_lib.sh"
source "$SCRIPT_DIR/r12_workflow.sh"

if [[ "${R12_QWEN_DRY_RUN:-0}" == 1 ]]; then
  out=${R12_QWEN_DRY_RUN_OUT:-$SCRIPT_DIR/qwen-dry-run}
  export P_NODE=${P_NODE:-dryrun-prefill} D_NODE=${D_NODE:-dryrun-decode}
  export P_PORT=${P_PORT:-11001} D_PORT=${D_PORT:-21001} ROUTER_PORT=${ROUTER_PORT:-31001}
  export P_NCCL_PORT=${P_NCCL_PORT:-51001} D_NCCL_PORT=${D_NCCL_PORT:-61001}
  export BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-41001}
  bash -n "$0" "$SCRIPT_DIR"/*.sh
  exec bash "$SCRIPT_DIR/render_r12_qwen_e2e.sh" "$out"
fi

ROOT=$R12_ROOT/qwen-e2e
IMG=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/yangminl/qwen35-perf-20260927/containers/lmsysorg+sglang+nightly-dev-cu13-20260926-1f6ce4b0-arm64-enroot.sqsh
MODEL=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/minma/models/Qwen3.8-Flash-Next
R10_ROOT=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/yangminl/pp-verify-20260917/R10-pp-bubble-vs-dep-tuning
SOURCE_ARCHIVE=$R10_ROOT/upload/src-info-r10-job5.tar.gz
SOURCE_MANIFEST=$R10_ROOT/upload/source-input-r10-job5.sha256
RUNTIME=$ROOT/runtime/$SLURM_JOB_ID
SRC=$RUNTIME/source
PIPDEPS=$RUNTIME/pipdeps
CACHE=$RUNTIME/cache
JOB_LOGS=$ROOT/logs/$SLURM_JOB_ID
JOB_RESULTS=$ROOT/results/$SLURM_JOB_ID
JOB_PROOF=$ROOT/proof/$SLURM_JOB_ID
RENDERED=$JOB_PROOF/rendered
ACTUAL=$JOB_PROOF/actual
R12_JOB=qwen-e2e
R12_PP_CHUNK=8192
R12_EXPECTED_SOURCE_SHA=$R12_QWEN_SOURCE_SHA
R12_CONCURRENCIES=16
R12_DISCARD_PROMPTS=16
R12_WARMUP_PROMPTS=16
R12_FORMAL_PROMPTS=32
R12_FORMAL_ROUNDS=1
R12_INCLUDE_A_REPEAT=0
R12_ROLE_RANKS=0
JOB_START_EPOCH=$(date +%s)
R12_JOB_END_EPOCH=$((JOB_START_EPOCH + 5400))

for path in "$IMG" "$MODEL/config.json" "$SOURCE_ARCHIVE" "$SOURCE_MANIFEST" \
  "$SCRIPT_DIR/render_r12_qwen_e2e.sh" "$SCRIPT_DIR/run_r12_qwen_role.sh" \
  "$SCRIPT_DIR/check_qwen_feature_support.py" "$SCRIPT_DIR/r12_qwen_runtime_lib.sh" \
  "$SCRIPT_DIR/r12_point_lib.sh" \
  "$SCRIPT_DIR/r12_workflow.sh"; do
  test -f "$path" || { echo "SETUP_INVALID reason=MISSING_QWEN_PREREQUISITE path=$path"; exit 1; }
done
mkdir -p "$SRC" "$PIPDEPS" "$CACHE" "$JOB_LOGS" "$JOB_RESULTS" "$RENDERED" "$ACTUAL"
P_NODE=
D_NODE=
PREFILL_PID=
DECODE_PID=
ROUTER_PID=
ACTIVE_DECODE_SERVICE=
P_PORT=
D_PORT=
ROUTER_PORT=
BOOTSTRAP_PORT=
P_NCCL_PORT=
D_NCCL_PORT=
NORMAL_COMPLETE=0
trap cleanup EXIT
tar -xzf "$SOURCE_ARCHIVE" -C "$SRC"
test "$(<"$SRC/SOURCE_COMMIT")" = "$R12_QWEN_SOURCE_SHA" || {
  echo "SETUP_INVALID reason=QWEN_SOURCE_SHA"; exit 1;
}
if ! (cd "$SRC" && sha256sum -c --quiet "$SOURCE_MANIFEST"); then
  echo "SETUP_INVALID reason=QWEN_SOURCE_MANIFEST"; exit 1
fi

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#NODES[@]}" == 2 ]] || { echo "SETUP_INVALID reason=QWEN_NODE_COUNT count=${#NODES[@]}"; exit 1; }
P_NODE=${NODES[0]}; D_NODE=${NODES[1]}
MOUNTS="$SRC:/src:ro,$MODEL:/model:ro,$SCRIPT_DIR:/harness:ro,$JOB_LOGS:/logs,$JOB_RESULTS:/results,$JOB_PROOF:/proof,$RUNTIME:/runtime"

echo "JOB_START $(date -u +%FT%TZ) job=$SLURM_JOB_ID experiment=qwen-e2e nodes=${NODES[*]} source=$R12_QWEN_SOURCE_SHA time_limit=01:30:00 scripted_time_kill=disabled"
for node in "${NODES[@]}"; do
  collect_topology "$node"
done

PORT_SELECTED=0
for attempt in $(seq 0 49); do
  offset=$(((SLURM_JOB_ID % 500 + attempt * 17) % 500))
  P_PORT=$((11000+offset)); D_PORT=$((21000+offset)); ROUTER_PORT=$((31000+offset))
  BOOTSTRAP_PORT=$((41000+offset))
  P_NCCL_PORT=$((51000+offset)); D_NCCL_PORT=$((61000+offset))
  if port_probe "$P_NODE" "$P_PORT" "$ROUTER_PORT" "$BOOTSTRAP_PORT" "$P_NCCL_PORT" \
    && port_probe "$D_NODE" "$D_PORT" "$BOOTSTRAP_PORT" "$D_NCCL_PORT"; then
    PORT_SELECTED=1; break
  fi
done
[[ "$PORT_SELECTED" == 1 ]] || { echo "SETUP_INVALID reason=QWEN_NO_FREE_PORT_SET"; exit 1; }
export P_NODE D_NODE P_PORT D_PORT ROUTER_PORT BOOTSTRAP_PORT P_NCCL_PORT D_NCCL_PORT
echo "PORT_SELECTION_PASS p=$P_PORT d=$D_PORT router=$ROUTER_PORT bootstrap=$BOOTSTRAP_PORT shared_pd=1 generations=1"
bash "$SCRIPT_DIR/render_r12_qwen_e2e.sh" "$RENDERED" | tee "$JOB_LOGS/render.out"
python3 "$SCRIPT_DIR/check_qwen_feature_support.py" --source-root "$SRC" \
  --rendered-root "$RENDERED" --output "$JOB_PROOF/feature-support-proof.json" \
  | tee "$JOB_LOGS/feature-support.out"

PACKAGE_OK=0
srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$P_NODE" \
  --container-image="$IMG" --no-container-entrypoint --no-container-mount-home \
  --container-mounts="$MOUNTS" --container-workdir=/src --container-remap-root \
  python3 -c "import importlib.metadata as m; assert m.version('sglang-kernel') == '0.4.8'; assert m.version('flashinfer-python') == '0.6.18'" \
  && PACKAGE_OK=1
if [[ "$PACKAGE_OK" != 1 ]]; then
  srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$P_NODE" \
    --container-image="$IMG" --no-container-entrypoint --no-container-mount-home \
    --container-mounts="$MOUNTS" --container-workdir=/src --container-remap-root \
    python3 -m pip install --target /runtime/pipdeps --no-deps \
      sglang-kernel==0.4.8 'flashinfer-python[cu13]==0.6.18' \
      >"$JOB_LOGS/package-install.out" 2>&1
fi
srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$P_NODE" \
  --container-image="$IMG" --no-container-entrypoint --no-container-mount-home \
  --container-mounts="$MOUNTS" --container-workdir=/src --container-remap-root \
  env PYTHONPATH=/runtime/pipdeps python3 -c \
    "import importlib.metadata as m; assert m.version('sglang-kernel') == '0.4.8'; assert m.version('flashinfer-python') == '0.6.18'; print('PACKAGE_EFFECTIVE_GATE_PASS sglang-kernel=0.4.8 flashinfer-python=0.6.18')"
echo "PACKAGE_CHECK_END $(date -u +%FT%TZ) isolated=$((1-PACKAGE_OK))"

r12_run_workflow
