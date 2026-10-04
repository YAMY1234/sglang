#!/bin/bash
#SBATCH --job-name=r12-job7
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=144
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/yangminl/pp-verify-20260917/R12-kimi-tep8-vs-pp4/logs/%x-%j.out

set -euo pipefail

R12_ROOT=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/yangminl/pp-verify-20260917/R12-kimi-tep8-vs-pp4
if [[ "${R12_DRY_RUN:-0}" == 1 ]]; then
  SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
else
  # Slurm executes a spool copy, so BASH_SOURCE[0] is not the submit directory.
  SCRIPT_DIR=${R12_HARNESS_DIR:-$R12_ROOT/sbatch}
fi
source "$SCRIPT_DIR/r12_launch_lib.sh"
source "$SCRIPT_DIR/r12_runtime_lib.sh"
source "$SCRIPT_DIR/r12_point_lib.sh"
source "$SCRIPT_DIR/r12_workflow.sh"
: "${R12_JOB:?R12_JOB=job7b|job7c}"
: "${R12_PP_CHUNK:?R12_PP_CHUNK=8192|16384}"
case "$R12_JOB:$R12_PP_CHUNK" in
  job7b:8192|job7c:16384) ;;
  *) echo "SETUP_INVALID reason=INVALID_JOB_CHUNK job=$R12_JOB pp_chunk=$R12_PP_CHUNK"; exit 2 ;;
esac
R12_MEM_FRACTION=${R12_MEM_FRACTION:-0.90}
case "$R12_MEM_FRACTION" in 0.90|0.85) ;; *) echo "SETUP_INVALID reason=INVALID_MEM_FRACTION value=$R12_MEM_FRACTION"; exit 2;; esac

if [[ "${R12_DRY_RUN:-0}" == 1 ]]; then
  out=${R12_DRY_RUN_OUT:-$SCRIPT_DIR/dry-run-$R12_JOB}
  export P0=${P0:-dryrun-p0} P1=${P1:-dryrun-p1} D0=${D0:-dryrun-d0} D1=${D1:-dryrun-d1}
  export P_PORT=${P_PORT:-10001} D_PORT=${D_PORT:-15001} ROUTER_PORT=${ROUTER_PORT:-20001}
  export P_DIST_PORT=${P_DIST_PORT:-25001} D_DIST_PORT=${D_DIST_PORT:-30001}
  export P_NCCL_PORT=${P_NCCL_PORT:-35001} D_NCCL_PORT=${D_NCCL_PORT:-40001}
  export BOOT_NORMAL=${BOOT_NORMAL:-45001} BOOT_FALLBACK=${BOOT_FALLBACK:-46001}
  bash -n "$0" "$SCRIPT_DIR"/*.sh
  echo "DRY_RUN_BASH_N_PASS harness=$0"
  exec env R12_JOB="$R12_JOB" R12_PP_CHUNK="$R12_PP_CHUNK" \
    R12_MEM_FRACTION="$R12_MEM_FRACTION" bash "$SCRIPT_DIR/render_r12_launches.sh" "$out"
fi

ROOT=$R12_ROOT
IMG=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/yangminl/qwen35-perf-20260927/containers/lmsysorg+sglang+nightly-dev-cu13-20260926-1f6ce4b0-arm64-enroot.sqsh
MODEL=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/models/kimi-k3
DRAFT=/lustre/fsw/portfolios/coreai/projects/coreai_tensorrt_ci/llm-models/RadixArk-Kimi-K3-DSpark
SOURCE_ARCHIVE=$ROOT/prereq/sglang-$R12_EXPECTED_SOURCE_SHA.tar.gz
SOURCE_ARCHIVE_CHECKSUM=$SOURCE_ARCHIVE.sha256
RUNTIME=$ROOT/runtime/$SLURM_JOB_ID
SRC=$RUNTIME/source
PIPDEPS=$RUNTIME/pipdeps
CACHE=$RUNTIME/cache
JOB_LOGS=$ROOT/logs/$R12_JOB-$SLURM_JOB_ID
JOB_RESULTS=$ROOT/results/$R12_JOB-$SLURM_JOB_ID
JOB_PROOF=$ROOT/proof/$R12_JOB-$SLURM_JOB_ID
RENDERED=$JOB_PROOF/rendered
ACTUAL=$JOB_PROOF/actual
JOB_START_EPOCH=$(date +%s)

for path in "$IMG" "$MODEL/config.json" "$DRAFT/config.json" "$SOURCE_ARCHIVE" \
  "$SOURCE_ARCHIVE_CHECKSUM" "$SCRIPT_DIR/render_r12_launches.sh" \
  "$SCRIPT_DIR/check_feature_support.py" \
  "$SCRIPT_DIR/run_r12_role.sh" "$SCRIPT_DIR/analyze_r12.py" \
  "$SCRIPT_DIR/summarize_r12.py" "$SCRIPT_DIR/r12_runtime_lib.sh" \
  "$SCRIPT_DIR/r12_point_lib.sh" \
  "$SCRIPT_DIR/r12_workflow.sh"; do
  test -f "$path" || { echo "SETUP_INVALID reason=MISSING_PREREQUISITE path=$path"; exit 1; }
done
mkdir -p "$ROOT/runtime" "$JOB_LOGS" "$JOB_RESULTS" "$RENDERED" "$ACTUAL"
RACK_FILE=$ROOT/runtime/rack-$R12_JOB.txt
release_rack_file() {
  if [[ -f "$RACK_FILE" ]] && grep -Fq "job_id=$SLURM_JOB_ID " "$RACK_FILE"; then
    rm -f "$RACK_FILE"
  fi
}
early_cleanup() {
  local rc=$?
  release_rack_file
  case "$RUNTIME" in "$ROOT/runtime/$SLURM_JOB_ID") ;; *) echo "REFUSE_EARLY_DELETE_RUNTIME path=$RUNTIME"; exit "$rc";; esac
  if [[ -e "$RUNTIME" ]]; then find "$RUNTIME" -depth -delete; fi
  rmdir "$ROOT/runtime" 2>/dev/null || true
  echo "JOB_EARLY_CLEANUP $(date -u +%FT%TZ) job=$SLURM_JOB_ID rc=$rc source_pipdeps_cache_deleted=1"
  exit "$rc"
}
trap early_cleanup EXIT

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#NODES[@]}" -eq 4 ]] || { echo "SETUP_INVALID reason=NODE_COUNT count=${#NODES[@]}"; exit 1; }
P0=${NODES[0]}; P1=${NODES[1]}; D0=${NODES[2]}; D1=${NODES[3]}
P_NODELIST=$P0,$P1
D_NODELIST=$D0,$D1
RACKS=$(printf '%s\n' "${NODES[@]}" | sed 's/-T.*//' | LC_ALL=C sort -u | paste -sd,)
[[ "$RACKS" != *,* ]] || { echo "SETUP_INVALID reason=CROSS_RACK nodes=${NODES[*]} racks=$RACKS"; exit 1; }
if [[ -n "${R12_FORBIDDEN_RACK:-}" && "$RACKS" == "$R12_FORBIDDEN_RACK" ]]; then
  echo "SETUP_INVALID reason=FORBIDDEN_RACK rack=$RACKS forbidden=$R12_FORBIDDEN_RACK nodes=${NODES[*]}"
  exit 1
fi

RACK_EXCLUDE=$RACKS-T[01-18]
RACK_TMP=$RACK_FILE.tmp.$SLURM_JOB_ID
printf 'job_id=%s job_name=%s rack=%s nodes=%s state=RUNNING timestamp=%s\n' \
  "$SLURM_JOB_ID" "$R12_JOB" "$RACKS" "${NODES[*]}" "$(date -u +%FT%TZ)" >"$RACK_TMP"
mv "$RACK_TMP" "$RACK_FILE"
case "$R12_JOB" in
  job7b) OTHER_JOB_NAME=r12-job7c ;;
  job7c) OTHER_JOB_NAME=r12-job7b ;;
esac
echo "RACK_COORDINATION_BEGIN job=$SLURM_JOB_ID experiment=$R12_JOB rack=$RACKS other=$OTHER_JOB_NAME before_source_extract=1"
for attempt in 1 2 3 4 5; do
  other_row=$(squeue -h -u "$USER" -n "$OTHER_JOB_NAME" -t PENDING,RUNNING \
    -o '%A|%T|%N' | awk -F'|' -v self="$SLURM_JOB_ID" '$1 != self {print; exit}')
  if [[ -z "$other_row" ]]; then
    echo "RACK_COORDINATION_NO_OTHER_ACTIVE other=$OTHER_JOB_NAME"
    break
  fi
  IFS='|' read -r other_id other_state other_nodes <<<"$other_row"
  if [[ "$other_state" == PENDING ]]; then
    if scontrol update JobId="$other_id" ExcNodeList="$RACK_EXCLUDE"; then
      echo "RACK_COORDINATION_EXCLUDED_PENDING other_job=$other_id exclude=$RACK_EXCLUDE"
      break
    fi
    echo "RACK_COORDINATION_RETRY attempt=$attempt other_job=$other_id observed=PENDING"
    sleep 1
    continue
  fi
  if [[ "$other_state" == RUNNING && -n "$other_nodes" ]]; then
    mapfile -t other_hostnames < <(scontrol show hostnames "$other_nodes")
    other_racks=$(printf '%s\n' "${other_hostnames[@]}" | sed 's/-T.*//' | LC_ALL=C sort -u | paste -sd,)
    if [[ "$other_racks" == *,* ]]; then
      echo "SETUP_INVALID reason=OTHER_JOB_CROSS_RACK other_job=$other_id nodes=$other_nodes racks=$other_racks"
      exit 1
    fi
    if [[ "$other_racks" == "$RACKS" ]]; then
      if (( SLURM_JOB_ID > other_id )); then
        echo "RACK_COORDINATION_REQUEUE job=$SLURM_JOB_ID winner=$other_id rack=$RACKS exclude=$RACK_EXCLUDE before_source_extract=1"
        scontrol update JobId="$SLURM_JOB_ID" ExcNodeList="$RACK_EXCLUDE"
        release_rack_file
        scontrol requeue "$SLURM_JOB_ID"
        exit 0
      fi
      echo "RACK_COORDINATION_SAME_RACK_WINNER job=$SLURM_JOB_ID loser=$other_id rack=$RACKS reason=lower_job_id"
    else
      echo "RACK_COORDINATION_DIFFERENT_RACK_PASS other_job=$other_id other_rack=$other_racks"
    fi
    break
  fi
done
echo "RACK_COORDINATION_PASS job=$SLURM_JOB_ID rack=$RACKS"

mkdir -p "$SRC" "$PIPDEPS" "$CACHE"
(cd "$ROOT/prereq" && sha256sum -c "$(basename "$SOURCE_ARCHIVE_CHECKSUM")") \
  | tee "$JOB_RESULTS/source-archive-verify.out"
tar -xzf "$SOURCE_ARCHIVE" -C "$SRC"
test -f "$SRC/python/pyproject.toml" || { echo "SETUP_INVALID reason=SOURCE_ARCHIVE_INCOMPLETE"; exit 1; }

MOUNTS="$SRC:/src:ro,$MODEL:/model:ro,$DRAFT:/draft:ro,$SCRIPT_DIR:/harness:ro,$JOB_LOGS:/logs,$JOB_RESULTS:/results,$JOB_PROOF:/proof,$RUNTIME:/runtime"
PREFILL_PID=
DECODE_PID=
ROUTER_PID=
ACTIVE_DECODE_SERVICE=
NORMAL_COMPLETE=0

safe_delete_runtime() {
  [[ "$RUNTIME" == "$ROOT/runtime/$SLURM_JOB_ID" ]] || {
    echo "REFUSE_DELETE_RUNTIME path=$RUNTIME"
    return 1
  }
  r12_safe_delete_runtime "$ROOT" "$RUNTIME"
}
cleanup() {
  local rc=$?
  stop_router || true
  stop_prefill || true
  stop_decode || true
  safe_delete_runtime || true
  release_rack_file
  echo "JOB_CLEANUP $(date -u +%FT%TZ) job=$SLURM_JOB_ID experiment=$R12_JOB rc=$rc normal_complete=$NORMAL_COMPLETE source_pipdeps_cache_deleted=1"
  exit "$rc"
}
trap cleanup EXIT

echo "JOB_START $(date -u +%FT%TZ) job=$SLURM_JOB_ID experiment=$R12_JOB pp_chunk=$R12_PP_CHUNK mem_fraction=$R12_MEM_FRACTION nodes=${NODES[*]} rack=$RACKS source=$R12_EXPECTED_SOURCE_SHA"
echo "JOB_ENV_BEGIN"; env | LC_ALL=C sort; echo "JOB_ENV_END"

collect_topology() {
  local node=$1
  local output=$JOB_LOGS/topology-$node.out
  srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$node" \
    --container-image="$IMG" --no-container-entrypoint --no-container-mount-home \
    --container-mounts="$MOUNTS" --container-workdir=/src --container-remap-root \
    python3 -c 'import os,subprocess; a=set(os.sched_getaffinity(0)); by={}; [by.setdefault(int(x.split(",")[1]),set()).add(int(x.split(",")[0])) for x in subprocess.check_output(["lscpu","-p=CPU,NODE"],text=True).splitlines() if x and not x.startswith("#")]; hit={k:len(v&a) for k,v in by.items()}; print("AFFINITY",min(a),max(a),len(a),sorted(a)); print("AFFINITY_BY_NUMA",hit); assert hit=={0:72,1:72},hit' \
    >"$output" 2>&1 || { cat "$output"; echo "SETUP_INVALID reason=AFFINITY_BY_NUMA node=$node"; exit 1; }
  cat "$output"
}
for node in "${NODES[@]}"; do collect_topology "$node"; done

port_probe() {
  local node=$1; shift
  srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$node" \
    python3 -c 'import socket,sys; held=[]; ports=list(map(int,sys.argv[1:])); [(lambda s,p:(s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1),s.bind(("0.0.0.0",p)),held.append(s)))(socket.socket(),p) for p in ports]; print("PORTS_FREE",socket.gethostname(),ports); [s.close() for s in held]' "$@"
}
PORT_SELECTED=0
for attempt in $(seq 0 49); do
  offset=$(((SLURM_JOB_ID % 500 + attempt * 17) % 500))
  P_PORT=$((10000+offset)); D_PORT=$((15000+offset)); ROUTER_PORT=$((20000+offset))
  P_DIST_PORT=$((25000+offset)); D_DIST_PORT=$((30000+offset))
  P_NCCL_PORT=$((35000+offset)); D_NCCL_PORT=$((40000+offset))
  BOOT_NORMAL=$((45000+offset)); BOOT_FALLBACK=$((46000+offset))
  if port_probe "$P0" "$P_PORT" "$ROUTER_PORT" "$P_DIST_PORT" "$P_NCCL_PORT" \
      "$BOOT_NORMAL" "$BOOT_FALLBACK" \
    && port_probe "$P1" "$P_NCCL_PORT" \
    && port_probe "$D0" "$D_PORT" "$D_DIST_PORT" "$D_NCCL_PORT" "$BOOT_NORMAL" "$BOOT_FALLBACK" \
    && port_probe "$D1" "$D_NCCL_PORT"; then PORT_SELECTED=1; break; fi
done
[[ "$PORT_SELECTED" == 1 ]] || { echo "SETUP_INVALID reason=NO_FREE_PORT_SET"; exit 1; }
export P0 P1 D0 D1 P_PORT D_PORT ROUTER_PORT P_DIST_PORT D_DIST_PORT \
  P_NCCL_PORT D_NCCL_PORT BOOT_NORMAL BOOT_FALLBACK
echo "PORT_SELECTION_PASS p=$P_PORT d=$D_PORT router=$ROUTER_PORT p_dist=$P_DIST_PORT d_dist=$D_DIST_PORT p_nccl=$P_NCCL_PORT d_nccl=$D_NCCL_PORT bootstrap_normal=$BOOT_NORMAL bootstrap_fallback=$BOOT_FALLBACK shared_pd=1"

R12_JOB=$R12_JOB R12_PP_CHUNK=$R12_PP_CHUNK R12_MEM_FRACTION=$R12_MEM_FRACTION \
  bash "$SCRIPT_DIR/render_r12_launches.sh" "$RENDERED" \
  | tee "$JOB_LOGS/render-launches.out"
python3 "$SCRIPT_DIR/check_feature_support.py" --source-root "$SRC" \
  --rendered-root "$RENDERED" --job "$R12_JOB" \
  --output "$JOB_PROOF/feature-support-proof.json" \
  | tee "$JOB_LOGS/feature-support-gate.out"

PACKAGE_LOG=$JOB_LOGS/package-versions.out
PACKAGE_OK=0
srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$P0" \
  --container-image="$IMG" --no-container-entrypoint --no-container-mount-home \
  --container-mounts="$MOUNTS" --container-workdir=/src --container-remap-root \
  python3 -m pip show sglang-kernel flashinfer-python | tee "$PACKAGE_LOG"
srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$P0" \
  --container-image="$IMG" --no-container-entrypoint --no-container-mount-home \
  --container-mounts="$MOUNTS" --container-workdir=/src --container-remap-root \
  python3 -c "import importlib.metadata as m; assert m.version('sglang-kernel') == '0.4.8'; assert m.version('flashinfer-python') == '0.6.18'" \
  && PACKAGE_OK=1
if [[ "$PACKAGE_OK" != 1 ]]; then
  srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$P0" \
    --container-image="$IMG" --no-container-entrypoint --no-container-mount-home \
    --container-mounts="$MOUNTS" --container-workdir=/src --container-remap-root \
    python3 -m pip install --target /runtime/pipdeps --no-deps \
      sglang-kernel==0.4.8 'flashinfer-python[cu13]==0.6.18' \
      2>&1 | tee "$JOB_LOGS/package-install.out"
fi
echo "PACKAGE_CHECK_END $(date -u +%FT%TZ) isolated=$((1-PACKAGE_OK))"

r12_run_workflow
