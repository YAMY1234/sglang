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
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/yangminl/pp-verify-20260917/R12-kimi-tep8-vs-pp4/qwen-e2e/logs/%x-%j.out

set -euo pipefail

R12_ROOT=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/yangminl/pp-verify-20260917/R12-kimi-tep8-vs-pp4
if [[ "${R12_QWEN_DRY_RUN:-0}" == 1 ]]; then
  SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
else
  SCRIPT_DIR=${R12_QWEN_HARNESS_DIR:-$R12_ROOT/qwen-e2e/sbatch}
fi
source "$SCRIPT_DIR/r12_qwen_e2e_lib.sh"
source "$SCRIPT_DIR/r12_point_lib.sh"
source "$SCRIPT_DIR/r12_workflow.sh"

if [[ "${R12_QWEN_DRY_RUN:-0}" == 1 ]]; then
  out=${R12_QWEN_DRY_RUN_OUT:-$SCRIPT_DIR/qwen-dry-run}
  export P_NODE=${P_NODE:-dryrun-prefill} D_NODE=${D_NODE:-dryrun-decode}
  export P_PORT=${P_PORT:-11001} D_PORT=${D_PORT:-21001} ROUTER_PORT=${ROUTER_PORT:-31001}
  export P_NCCL_PORT=${P_NCCL_PORT:-51001} D_NCCL_PORT=${D_NCCL_PORT:-61001}
  export BOOT_NORMAL=${BOOT_NORMAL:-41001} BOOT_FALLBACK=${BOOT_FALLBACK:-42001}
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
R12_REPEAT_HANDOFF_PROBE=1
R12_INJECT_REPEAT_PROBE_FAILURE=1
R12_ROLE_RANKS=0
JOB_START_EPOCH=$(date +%s)

for path in "$IMG" "$MODEL/config.json" "$SOURCE_ARCHIVE" "$SOURCE_MANIFEST" \
  "$SCRIPT_DIR/render_r12_qwen_e2e.sh" "$SCRIPT_DIR/run_r12_qwen_role.sh" \
  "$SCRIPT_DIR/check_qwen_feature_support.py" "$SCRIPT_DIR/r12_point_lib.sh" \
  "$SCRIPT_DIR/r12_workflow.sh"; do
  test -f "$path" || { echo "SETUP_INVALID reason=MISSING_QWEN_PREREQUISITE path=$path"; exit 1; }
done
mkdir -p "$SRC" "$PIPDEPS" "$CACHE" "$JOB_LOGS" "$JOB_RESULTS" "$RENDERED" "$ACTUAL"
early_cleanup() {
  local rc=$?
  trap - EXIT
  case "$RUNTIME" in "$ROOT/runtime/$SLURM_JOB_ID") ;; *) echo "REFUSE_EARLY_DELETE_RUNTIME path=$RUNTIME"; exit "$rc";; esac
  if [[ -e "$RUNTIME" ]]; then find "$RUNTIME" -depth -delete; fi
  rmdir "$ROOT/runtime" 2>/dev/null || true
  echo "JOB_EARLY_CLEANUP $(date -u +%FT%TZ) job=$SLURM_JOB_ID rc=$rc source_pipdeps_cache_deleted=1"
  exit "$rc"
}
trap early_cleanup EXIT
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
PREFILL_PID=; DECODE_PID=; ROUTER_PID=; ACTIVE_DECODE_SERVICE=; NORMAL_DECODE_PID=
P_PORT=; D_PORT=; ROUTER_PORT=; BOOT_NORMAL=; BOOT_FALLBACK=; P_NCCL_PORT=; D_NCCL_PORT=
NORMAL_COMPLETE=0
BUDGET_WATCHDOG_PID=

stop_pid() {
  local pid=$1
  [[ -n "$pid" ]] || return 0
  set +e
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  set -e
}
wait_endpoint_down() {
  local url=$1 label=$2
  for _ in $(seq 1 60); do
    if ! curl -fsS --max-time 1 "$url" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  echo "SETUP_INVALID reason=QWEN_ENDPOINT_STILL_LIVE label=$label url=$url"
  return 1
}
stop_router() { stop_pid "${ROUTER_PID:-}"; ROUTER_PID=; [[ -z "$ROUTER_PORT" ]] || wait_endpoint_down "http://$P_NODE:$ROUTER_PORT/health" router; }
stop_prefill() { stop_pid "${PREFILL_PID:-}"; PREFILL_PID=; [[ -z "$P_PORT" ]] || wait_endpoint_down "http://$P_NODE:$P_PORT/health" prefill; }
stop_decode() { stop_pid "${DECODE_PID:-}"; DECODE_PID=; [[ -z "$D_PORT" ]] || wait_endpoint_down "http://$D_NODE:$D_PORT/health" decode; }
cleanup() {
  local rc=$?
  trap - EXIT
  if [[ -n "${BUDGET_WATCHDOG_PID:-}" ]]; then
    kill "$BUDGET_WATCHDOG_PID" 2>/dev/null || true
    wait "$BUDGET_WATCHDOG_PID" 2>/dev/null || true
  fi
  stop_router || true; stop_prefill || true; stop_decode || true
  case "$RUNTIME" in "$ROOT/runtime/$SLURM_JOB_ID") ;; *) echo "REFUSE_DELETE_RUNTIME path=$RUNTIME"; exit "$rc";; esac
  if [[ -e "$RUNTIME" ]]; then find "$RUNTIME" -depth -delete; fi
  rmdir "$ROOT/runtime" 2>/dev/null || true
  echo "JOB_CLEANUP $(date -u +%FT%TZ) job=$SLURM_JOB_ID experiment=qwen-e2e rc=$rc normal_complete=$NORMAL_COMPLETE source_pipdeps_cache_deleted=1"
  exit "$rc"
}
trap cleanup EXIT

# Slurm keeps the requested 00:40 safety window for orderly cleanup, but the
# R12 accounting cap is 4 GPU.h.  Terminate the 8-GPU validation after 1700 s
# of wrapper time (leaving allocation/cleanup margin below 30 billed minutes).
(
  sleep 1700
  echo "QWEN_E2E_BUDGET_TIMEOUT elapsed_wrapper_s=1700 hard_gpu_hours=4 action=terminate"
  kill -TERM $$
) &
BUDGET_WATCHDOG_PID=$!

echo "JOB_START $(date -u +%FT%TZ) job=$SLURM_JOB_ID experiment=qwen-e2e nodes=${NODES[*]} source=$R12_QWEN_SOURCE_SHA time_limit=00:40:00"
for node in "${NODES[@]}"; do
  output=$JOB_LOGS/topology-$node.out
  srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$node" \
    python3 -c 'import os,subprocess; a=set(os.sched_getaffinity(0)); by={}; [by.setdefault(int(x.split(",")[1]),set()).add(int(x.split(",")[0])) for x in subprocess.check_output(["lscpu","-p=CPU,NODE"],text=True).splitlines() if x and not x.startswith("#")]; hit={k:len(v&a) for k,v in by.items()}; print("AFFINITY_BY_NUMA",hit); assert hit=={0:72,1:72},hit' \
    >"$output" 2>&1 || { cat "$output"; echo "SETUP_INVALID reason=QWEN_AFFINITY node=$node"; exit 1; }
  cat "$output"
done

port_probe() {
  local node=$1; shift
  srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$node" \
    python3 -c 'import socket,sys; held=[]; ports=list(map(int,sys.argv[1:])); [(lambda s,p:(s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1),s.bind(("0.0.0.0",p)),held.append(s)))(socket.socket(),p) for p in ports]; print("PORTS_FREE",socket.gethostname(),ports); [s.close() for s in held]' "$@"
}
PORT_SELECTED=0
for attempt in $(seq 0 49); do
  offset=$(((SLURM_JOB_ID % 500 + attempt * 17) % 500))
  P_PORT=$((11000+offset)); D_PORT=$((21000+offset)); ROUTER_PORT=$((31000+offset))
  BOOT_NORMAL=$((41000+offset)); BOOT_FALLBACK=$((42000+offset))
  P_NCCL_PORT=$((51000+offset)); D_NCCL_PORT=$((61000+offset))
  if port_probe "$P_NODE" "$P_PORT" "$ROUTER_PORT" "$BOOT_NORMAL" "$BOOT_FALLBACK" "$P_NCCL_PORT" \
    && port_probe "$D_NODE" "$D_PORT" "$BOOT_NORMAL" "$BOOT_FALLBACK" "$D_NCCL_PORT"; then
    PORT_SELECTED=1; break
  fi
done
[[ "$PORT_SELECTED" == 1 ]] || { echo "SETUP_INVALID reason=QWEN_NO_FREE_PORT_SET"; exit 1; }
export P_NODE D_NODE P_PORT D_PORT ROUTER_PORT BOOT_NORMAL BOOT_FALLBACK P_NCCL_PORT D_NCCL_PORT
echo "PORT_SELECTION_PASS p=$P_PORT d=$D_PORT router=$ROUTER_PORT bootstrap_normal=$BOOT_NORMAL bootstrap_fallback=$BOOT_FALLBACK shared_pd=1"
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

launch_prefill() {
  local service=$1 variant=$2 chunk=$3 boot=$4
  srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$P_NODE" \
    --container-image="$IMG" --no-container-entrypoint --no-container-mount-home \
    --container-mounts="$MOUNTS" --container-workdir=/src --container-remap-root \
    env R12_ROLE=prefill R12_VARIANT="$variant" R12_CHUNK="$chunk" R12_SERVICE="$service" \
      R12_HTTP_PORT="$P_PORT" R12_NCCL_PORT="$P_NCCL_PORT" R12_BOOTSTRAP_PORT="$boot" \
      R12_FROZEN_PROOF_DIR=/proof/rendered R12_ACTUAL_PROOF_DIR=/proof/actual \
      /bin/bash /harness/run_r12_qwen_role.sh >"$JOB_LOGS/$service-prefill-rank-0.out" 2>&1 &
  PREFILL_PID=$!
}
launch_decode() {
  local service=$1 boot=$2
  srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$D_NODE" \
    --container-image="$IMG" --no-container-entrypoint --no-container-mount-home \
    --container-mounts="$MOUNTS" --container-workdir=/src --container-remap-root \
    env R12_ROLE=decode R12_VARIANT=TEP R12_CHUNK=32768 R12_SERVICE="$service" \
      R12_HTTP_PORT="$D_PORT" R12_NCCL_PORT="$D_NCCL_PORT" R12_BOOTSTRAP_PORT="$boot" \
      R12_FROZEN_PROOF_DIR=/proof/rendered R12_ACTUAL_PROOF_DIR=/proof/actual \
      /bin/bash /harness/run_r12_qwen_role.sh >"$JOB_LOGS/$service-decode-rank-0.out" 2>&1 &
  DECODE_PID=$!; ACTIVE_DECODE_SERVICE=$service
  if [[ "$service" == decode-normal ]]; then NORMAL_DECODE_PID=$DECODE_PID; fi
}
wait_servers() {
  local service=$1 ready=0
  for _ in $(seq 1 240); do
    if curl -fsS --max-time 2 "http://$P_NODE:$P_PORT/health" >/dev/null 2>&1 \
      && curl -fsS --max-time 2 "http://$D_NODE:$D_PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$PREFILL_PID" 2>/dev/null || ! kill -0 "$DECODE_PID" 2>/dev/null; then break; fi
    sleep 5
  done
  if [[ "$ready" != 1 ]]; then
    tail -240 "$JOB_LOGS/$service-prefill-rank-0.out" "$JOB_LOGS/$ACTIVE_DECODE_SERVICE-decode-rank-0.out" 2>/dev/null || true
    echo "SETUP_INVALID reason=QWEN_SERVER_START service=$service"; return 1
  fi
  local skip
  skip=$(awk '/skipping NUMA binding for GPU/{s++} END{print s+0}' \
    "$JOB_LOGS/$service-prefill-rank-0.out" "$JOB_LOGS/$ACTIVE_DECODE_SERVICE-decode-rank-0.out")
  echo "NUMA_BIND_LOG_GATE service=$service skip_count=$skip expected=0"
  [[ "$skip" == 0 ]] || { echo "SETUP_INVALID reason=QWEN_NUMA_LOG service=$service"; return 1; }
  echo "SERVER_START_PASS $(date -u +%FT%TZ) service=$service"
  if [[ "$ACTIVE_DECODE_SERVICE" == decode-normal && "$service" != B-main ]]; then
    [[ "$DECODE_PID" == "$NORMAL_DECODE_PID" ]] && kill -0 "$NORMAL_DECODE_PID"
    echo "DECODE_RESIDENT_GATE_PASS service=$service decode_service=decode-normal pid=$NORMAL_DECODE_PID"
  fi
}
launch_router() {
  local service=$1 actual=$ACTUAL/$service-router.out frozen=$RENDERED/$service-router.out
  r12_qwen_build_router_command "$P_NODE" "$P_PORT" "$D_NODE" "$D_PORT" "$ROUTER_PORT"
  r12_qwen_emit_command ROUTER_LAUNCH "$actual"
  cmp "$frozen" "$actual" || { echo "SETUP_INVALID reason=QWEN_ROUTER_PROOF service=$service"; return 1; }
  srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$P_NODE" \
    --container-image="$IMG" --no-container-entrypoint --no-container-mount-home \
    --container-mounts="$MOUNTS" --container-workdir=/src --container-remap-root \
    "${R12_COMMAND[@]}" >"$JOB_LOGS/$service-router.out" 2>&1 &
  ROUTER_PID=$!
  local ready=0
  for _ in $(seq 1 120); do
    if curl -fsS --max-time 2 "http://$P_NODE:$ROUTER_PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    kill -0 "$ROUTER_PID" 2>/dev/null || break; sleep 2
  done
  [[ "$ready" == 1 ]] || { echo "SETUP_INVALID reason=QWEN_ROUTER_START service=$service"; return 1; }
  echo "ROUTER_START_PASS $(date -u +%FT%TZ) service=$service"
}
run_bench() {
  local prefix=$1 window=$2 concurrency=$3 prompts=$4 label=$5 seed=$6
  echo "BENCH_BEGIN $(date -u +%FT%TZ) C=$concurrency label=$label prompts=$prompts output_len=2 seed=$seed" | tee -a "$window"
  set +e
  srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$P_NODE" \
    --container-image="$IMG" --no-container-entrypoint --no-container-mount-home \
    --container-mounts="$MOUNTS" --container-workdir=/src --container-remap-root \
    env PYTHONPATH=/runtime/pipdeps:/src/python PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    python3 -m sglang.bench_serving --backend sglang --base-url "http://$P_NODE:$ROUTER_PORT" \
      --model Qwen/Qwen3.8-Flash-Next --tokenizer /model --dataset-name random \
      --random-input-len 8192 --random-output-len 2 --random-range-ratio 1 \
      --num-prompts "$prompts" --max-concurrency "$concurrency" --request-rate inf \
      --warmup-requests 0 --flush-cache --disable-tqdm --disable-stream --seed "$seed" \
      --output-file "/logs/bench-$prefix-$label.json" \
      >"$JOB_LOGS/bench-$prefix-$label.out" 2>&1
  local rc=$?
  set -e
  [[ "$rc" == 0 ]] || return "$rc"
  python3 - "$JOB_LOGS/bench-$prefix-$label.json" "$prompts" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert int(d["completed"]) == int(sys.argv[2]), d
PY
  echo "BENCH_END $(date -u +%FT%TZ) C=$concurrency label=$label prompts=$prompts output_len=2 seed=$seed" | tee -a "$window"
}
scan_fatal() {
  local service=$1 label=$2 output=$JOB_LOGS/$service-$label-error-scan.out
  : >"$output"
  grep -E -n 'Fatal Python error|CUDA error|out of memory|OutOfMemory|OOM|Killed|Address already|NCCL.*(Error|error|failed|timeout|abort)' \
    "$JOB_LOGS/$service-prefill-rank-0.out" "$JOB_LOGS/$ACTIVE_DECODE_SERVICE-decode-rank-0.out" \
    "$JOB_LOGS/$service-router.out" >>"$output" 2>/dev/null || true
  [[ ! -s "$output" ]] || { cat "$output"; echo "SETUP_INVALID reason=QWEN_FATAL service=$service label=$label"; return 1; }
  echo "ERROR_SCAN_PASS service=$service label=$label"
}
r12_analyze_point() {
  local service=$1 arm=$2 concurrency=$3 topology=$4 prefix=$5 window=$6 expected=$7 rounds=$8
  [[ "$rounds" == 1 ]]
  local output=$JOB_RESULTS/$arm/point-summary.json
  mkdir -p "$(dirname "$output")"
  python3 - "$JOB_LOGS/bench-$prefix-formal1.json" "$output" "$arm" "$topology" "$expected" "$window" <<'PY'
import json,pathlib,sys
bench_path,out,arm,topology,expected,window=sys.argv[1:]
d=json.load(open(bench_path)); expected=int(expected); completed=int(d["completed"])
assert completed == expected, (completed,expected)
value=float(d["input_throughput"])
row={"arm":arm,"topology":topology,"benchmark":{"rounds":[{"formal":1,"completed":completed,"expected":expected,"incomplete":expected-completed,"input_throughput":value}],"median_input_throughput":value,"median_per_prefill_gpu":value/4,"max_over_min":1.0,"request_trace":{"completed":completed,"expected":expected,"incomplete":expected-completed,"rating":"PASS"}},"mechanism":{"trace_mode":"qwen_e2e_validation","trace_completeness":"N/A_VALIDATION","kv_transfer":"NOT_REPORTED"},"windows":window}
p=pathlib.Path(out); p.write_text(json.dumps(row,indent=2,sort_keys=True)+"\n")
print("R12_POINT",json.dumps(row,sort_keys=True,separators=(",",":")))
PY
  scan_fatal "$service" "C$concurrency"
  find "$JOB_LOGS" "$JOB_RESULTS/$arm" -type f \
    \( -name "*$service*C$concurrency*" -o -path "$JOB_RESULTS/$arm/*" \) -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum >"$JOB_RESULTS/$arm/input-files.sha256"
  echo "TRACE_MODE=qwen_e2e_validation arm=$arm completed=$expected/$expected"
}
r12_summarize_results() {
  local fallback=$1 skipped=$2 output=$JOB_RESULTS/qwen-e2e-summary.json
  python3 - "$JOB_RESULTS" "$output" "$fallback" "$skipped" "$JOB_PROOF/raw-diff-B-main-vs-A-main-rank0.out" <<'PY'
import hashlib,json,pathlib,sys
root,out,fallback,skipped,diff=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),pathlib.Path(sys.argv[5])
names=("B-C16","A-C16","B-C16-repeat","A-C16-repeat")
arms={name:json.load(open(root/name/"point-summary.json")) for name in names}
assert all(row["benchmark"]["request_trace"]["rating"] == "PASS" for row in arms.values())
assert fallback == 1 and skipped == 0 and diff.exists()
record={"job":"qwen-e2e","performance_reportable":False,"arms":list(names),"sampling":{"warmup_prompts":16,"formal_rounds":1,"formal_prompts":32},"gates":{"normal_handoff_probe60":"PASS","resident_decode":"PASS","repeat_probe_failure_injected":1,"DECODE_RESTART_FALLBACK":fallback,"A_REPEAT_SKIPPED_TIMEOUT":skipped,"proof_diff":"PASS","request_completion":"128/128"},"trace_mode":"qwen_e2e_validation","proof_diff_sha256":hashlib.sha256(diff.read_bytes()).hexdigest(),"verdict":"PASS"}
out.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
print("R12_SUMMARY",json.dumps(record,sort_keys=True,separators=(",",":")))
PY
}

r12_run_workflow
