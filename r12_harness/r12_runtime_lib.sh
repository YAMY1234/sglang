#!/bin/bash

# Workflow-facing runtime functions shared verbatim by the allocated Kimi
# wrapper and the CPU walkthrough.  The walkthrough replaces only external
# commands (`srun` and `curl`), never these harness functions.

stop_pid() {
  local pid=$1
  [[ -n "$pid" ]] || return 0
  set +e
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 60); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null; fi
  wait "$pid" 2>/dev/null
  set -e
}

wait_endpoint_down() {
  local url=$1 label=$2
  for _ in $(seq 1 60); do
    if ! curl -fsS --max-time 1 "$url" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  echo "SETUP_INVALID reason=ENDPOINT_STILL_LIVE label=$label url=$url"
  return 1
}

stop_router() {
  stop_pid "${ROUTER_PID:-}"; ROUTER_PID=
  [[ -z "${P0:-}" || -z "${ROUTER_PORT:-}" ]] || wait_endpoint_down "http://$P0:$ROUTER_PORT/health" router
}

stop_prefill() {
  stop_pid "${PREFILL_PID:-}"; PREFILL_PID=
  [[ -z "${P0:-}" || -z "${P_PORT:-}" ]] || wait_endpoint_down "http://$P0:$P_PORT/health" prefill
}

stop_decode() {
  stop_pid "${DECODE_PID:-}"; DECODE_PID=
  [[ -z "${D0:-}" || -z "${D_PORT:-}" ]] || wait_endpoint_down "http://$D0:$D_PORT/health" decode
}

launch_prefill() {
  local service=$1 variant=$2 chunk=$3 boot=$4
  srun --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=144 --cpu-bind=none \
    --nodelist="$P_NODELIST" --container-image="$IMG" --no-container-entrypoint \
    --no-container-mount-home --container-mounts="$MOUNTS" --container-workdir=/src \
    --container-remap-root --output="$JOB_LOGS/$service-prefill-rank-%t.out" \
    --error="$JOB_LOGS/$service-prefill-rank-%t.out" \
    env R12_ROLE=prefill R12_VARIANT="$variant" R12_CHUNK="$chunk" R12_SERVICE="$service" \
      R12_MASTER="$P0" R12_DIST_PORT="$P_DIST_PORT" R12_HTTP_PORT="$P_PORT" \
      R12_NCCL_PORT="$P_NCCL_PORT" R12_BOOTSTRAP_PORT="$boot" \
      R12_MEM_FRACTION="$R12_MEM_FRACTION" R12_FROZEN_PROOF_DIR=/proof/rendered \
      R12_ACTUAL_PROOF_DIR=/proof/actual /bin/bash /harness/run_r12_role.sh &
  PREFILL_PID=$!
}

launch_decode() {
  local service=$1 boot=$2
  srun --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=144 --cpu-bind=none \
    --nodelist="$D_NODELIST" --container-image="$IMG" --no-container-entrypoint \
    --no-container-mount-home --container-mounts="$MOUNTS" --container-workdir=/src \
    --container-remap-root --output="$JOB_LOGS/$service-decode-rank-%t.out" \
    --error="$JOB_LOGS/$service-decode-rank-%t.out" \
    env R12_ROLE=decode R12_VARIANT=TEP R12_CHUNK=8192 R12_SERVICE="$service" \
      R12_MASTER="$D0" R12_DIST_PORT="$D_DIST_PORT" R12_HTTP_PORT="$D_PORT" \
      R12_NCCL_PORT="$D_NCCL_PORT" R12_BOOTSTRAP_PORT="$boot" \
      R12_MEM_FRACTION="$R12_MEM_FRACTION" R12_FROZEN_PROOF_DIR=/proof/rendered \
      R12_ACTUAL_PROOF_DIR=/proof/actual /bin/bash /harness/run_r12_role.sh &
  DECODE_PID=$!
  ACTIVE_DECODE_SERVICE=$service
}

prewarm_dynamic_module_role() {
  local service=$1
  local role=$2
  local nodelist=$3
  local proof_dir=$JOB_PROOF/prewarm
  local rank
  mkdir -p "$proof_dir"
  echo "DYNAMIC_MODULE_PREWARM_BEGIN $(date -u +%FT%TZ) service=$service role=$role nodes=$nodelist isolated_role_cache=1"
  srun --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=144 --cpu-bind=none \
    --nodelist="$nodelist" --container-image="$IMG" --no-container-entrypoint \
    --no-container-mount-home --container-mounts="$MOUNTS" --container-workdir=/src \
    --container-remap-root --output="$proof_dir/$service-$role-rank-%t.out" \
    --error="$proof_dir/$service-$role-rank-%t.out" \
    env R12_ROLE="$role" R12_SERVICE="$service" \
      /bin/bash /harness/prewarm_r12_cache.sh
  for rank in 0 1; do
    grep -Fq "DYNAMIC_MODULE_PREWARM_PASS service=$service role=$role rank=$rank " \
      "$proof_dir/$service-$role-rank-$rank.out"
  done
  sha256sum "$proof_dir/$service-$role-rank-0.out" \
    "$proof_dir/$service-$role-rank-1.out" \
    >"$proof_dir/$service-$role.sha256"
  echo "DYNAMIC_MODULE_PREWARM_ROLE_PASS $(date -u +%FT%TZ) service=$service role=$role proof=$proof_dir/$service-$role.sha256"
}

prewarm_dynamic_module_caches() {
  local service=$1
  prewarm_dynamic_module_role "$service" prefill "$P_NODELIST"
  prewarm_dynamic_module_role "$service" decode "$D_NODELIST"
  echo "DYNAMIC_MODULE_PREWARM_GROUP_PASS $(date -u +%FT%TZ) service=$service caches=4"
}

r12_collect_start_failure() {
  local service=$1
  local attempt=$2
  local forced_reason=${3:-}
  local summary=$JOB_LOGS/$service-start-attempt$attempt-traceback-summary.out
  local dynamic_log=
  local host=unknown
  local node=unknown
  local ranks=unknown
  local reason=UNKNOWN_PROCESS_EXIT
  local oom_evidence=0
  local log
  local -a logs=()
  while IFS= read -r log; do logs+=("$log"); done < <(
    find "$JOB_LOGS" "$JOB_PROOF/prewarm" -maxdepth 1 -type f \
      \( -name "$service-prefill-rank-*.out" -o -name "$service-decode-rank-*.out" \
      -o -name "$service-router.out" \) | LC_ALL=C sort
  )
  {
    echo "START_FAILURE_TRACEBACK_SUMMARY_BEGIN service=$service attempt=$attempt logs=${#logs[@]}"
    for log in "${logs[@]}"; do
      echo "LOG_BEGIN path=$log"
      grep -n -m 24 -E 'Traceback|ModuleNotFoundError|ImportError|OutOfMemory|CUDA out of memory|Memory pool.*(allocat|fail)|Failed to allocate.*memory pool|Connection closed by peer|Scheduler hit an exception' \
        "$log" || true
      echo "LOG_END path=$log"
    done
    echo "START_FAILURE_TRACEBACK_SUMMARY_END service=$service attempt=$attempt"
  } | tee "$summary"

  for log in "${logs[@]}"; do
    if grep -Eq "No module named 'transformers_modules[.]model[.]|No module named 'encoding_k3'" "$log"; then
      dynamic_log=$log
      break
    fi
  done
  if [[ -n "$forced_reason" ]]; then
    reason=$forced_reason
  elif [[ -n "$dynamic_log" ]]; then
    host=$(grep -m1 'ROLE_START ' "$dynamic_log" | sed -n 's/.* host=\([^ ]*\).*/\1/p' || true)
    [[ -n "$host" ]] || host=unknown
    node=${host##*-}
    ranks=$(grep -oE 'TP[0-9]+' "$dynamic_log" | sed 's/^TP//' | LC_ALL=C sort -nu | paste -sd, || true)
    [[ -n "$ranks" ]] || ranks=unknown
    reason=DYNAMIC_MODULE_IMPORT_RACE
  else
    for log in "${logs[@]}"; do
      if grep -Eq 'OutOfMemory|CUDA out of memory|Memory pool.*(allocat|fail)|Failed to allocate.*memory pool' "$log"; then
        oom_evidence=1
        break
      fi
    done
    if [[ "$oom_evidence" == 1 ]]; then reason=STARTUP_OOM_EVIDENCE; fi
  fi
  R12_SERVER_FAILURE_REASON=$reason
  echo "SERVER_START_FAIL $(date -u +%FT%TZ) service=$service attempt=$attempt reason=$reason node=$node host=$host ranks=$ranks mem_fraction=$R12_MEM_FRACTION summary=$summary"
  if [[ "$oom_evidence" == 1 ]]; then
    echo "MEMORY_RETRY_BOTH_ARMS=0.85 reason=STARTUP_OOM action=NEED_LEAD automatic_change=0 current_mem_fraction=$R12_MEM_FRACTION"
    echo "NEED_LEAD reason=STARTUP_OOM_EVIDENCE service=$service attempt=$attempt automatic_mem_change=forbidden"
  fi
}

wait_servers() {
  local service=$1
  local start_epoch
  local next_progress
  local now
  start_epoch=$(date +%s)
  next_progress=$((start_epoch + 60))
  while true; do
    if curl -fsS --max-time 2 "http://$P0:$P_PORT/health" >/dev/null 2>&1 \
      && curl -fsS --max-time 2 "http://$D0:$D_PORT/health" >/dev/null 2>&1; then
      local skip
      skip=$(r12_count_numa_skip_logs "$JOB_LOGS" "$service" "$ACTIVE_DECODE_SERVICE")
      echo "NUMA_BIND_LOG_GATE service=$service skip_count=$skip expected=0"
      [[ "$skip" == 0 ]] || { echo "SETUP_INVALID reason=NUMA_BIND_LOG_MISMATCH service=$service"; return 1; }
      echo "SERVER_START_PASS $(date -u +%FT%TZ) service=$service"
      return 0
    fi
    if ! kill -0 "$PREFILL_PID" 2>/dev/null || ! kill -0 "$DECODE_PID" 2>/dev/null; then
      echo "SERVER_PROCESS_EXIT $(date -u +%FT%TZ) service=$service prefill_alive=$([[ -n ${PREFILL_PID:-} ]] && kill -0 "$PREFILL_PID" 2>/dev/null && echo 1 || echo 0) decode_alive=$([[ -n ${DECODE_PID:-} ]] && kill -0 "$DECODE_PID" 2>/dev/null && echo 1 || echo 0)"
      return 1
    fi
    now=$(date +%s)
    if (( now >= next_progress )); then
      echo "SERVER_START_PROGRESS $(date -u +%FT%TZ) service=$service elapsed_s=$((now-start_epoch)) prefill_alive=1 decode_alive=1 wait_limit=slurm_only"
      next_progress=$((now + 60))
    fi
    sleep 5
  done
}

launch_router() {
  local service=$1
  r12_build_router_command "$P0" "$P_PORT" "$D0" "$D_PORT" "$ROUTER_PORT"
  local actual=$ACTUAL/$service-router.out
  local frozen=$RENDERED/$service-router.out
  r12_emit_command ROUTER_LAUNCH "$actual"
  cmp "$frozen" "$actual" || { echo "SETUP_INVALID reason=ROUTER_PROOF_MISMATCH service=$service"; return 1; }
  echo "LAUNCH_RENDER_MATCH service=$service role=router sha256=$(sha256sum "$actual" | awk '{print $1}')"
  srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$P0" \
    --container-image="$IMG" --no-container-entrypoint --no-container-mount-home \
    --container-mounts="$MOUNTS" --container-workdir=/src --container-remap-root \
    "${R12_COMMAND[@]}" >"$JOB_LOGS/$service-router.out" 2>&1 &
  ROUTER_PID=$!
  local start_epoch
  local next_progress
  local now
  start_epoch=$(date +%s)
  next_progress=$((start_epoch + 60))
  while true; do
    if curl -fsS --max-time 2 "http://$P0:$ROUTER_PORT/health" >/dev/null 2>&1; then
      echo "ROUTER_START_PASS $(date -u +%FT%TZ) service=$service"
      return 0
    fi
    if ! kill -0 "$ROUTER_PID" 2>/dev/null; then
      echo "ROUTER_PROCESS_EXIT $(date -u +%FT%TZ) service=$service"
      return 1
    fi
    now=$(date +%s)
    if (( now >= next_progress )); then
      echo "ROUTER_START_PROGRESS $(date -u +%FT%TZ) service=$service elapsed_s=$((now-start_epoch)) alive=1 wait_limit=slurm_only"
      next_progress=$((now + 60))
    fi
    sleep 2
  done
}

run_bench() {
  local prefix=$1 window=$2 concurrency=$3 prompts=$4 label=$5 seed=$6
  echo "BENCH_BEGIN $(date -u +%FT%TZ) C=$concurrency label=$label prompts=$prompts output_len=2 seed=$seed" | tee -a "$window"
  set +e
  srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$P0" \
    --container-image="$IMG" --no-container-entrypoint --no-container-mount-home \
    --container-mounts="$MOUNTS" --container-workdir=/src --container-remap-root \
    env PYTHONPATH=/runtime/pipdeps:/src/python PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
      HF_HOME=/runtime/cache/client/huggingface XDG_CACHE_HOME=/runtime/cache/client/xdg \
      SGLANG_CACHE_DIR=/runtime/cache/client/sglang \
    python3 -m sglang.bench_serving --backend sglang --base-url "http://$P0:$ROUTER_PORT" \
      --model moonshotai/Kimi-K3 --tokenizer /model --dataset-name random \
      --random-input-len 8192 --random-output-len 2 --random-range-ratio 1 \
      --num-prompts "$prompts" --max-concurrency "$concurrency" --request-rate inf \
      --warmup-requests 0 --flush-cache --disable-tqdm --disable-stream --seed "$seed" \
      --output-file "/logs/bench-$prefix-$label.json" 2>&1 | tee "$JOB_LOGS/bench-$prefix-$label.out"
  local rc=${PIPESTATUS[0]}
  set -e
  [[ "$rc" == 0 ]] || return "$rc"
  if ! python3 - "$JOB_LOGS/bench-$prefix-$label.json" "$prompts" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); expected=int(sys.argv[2]); assert d["completed"] == expected,(d["completed"],expected)
PY
  then
    return 1
  fi
  echo "BENCH_END $(date -u +%FT%TZ) C=$concurrency label=$label prompts=$prompts output_len=2 seed=$seed" | tee -a "$window"
}

scan_fatal() {
  local service=$1 label=$2
  local output=$JOB_LOGS/$service-$label-error-scan.out
  : >"$output"
  grep -E -n 'Fatal Python error|CUDA error|out of memory|OutOfMemory|OOM|Killed|Address already|NCCL.*(Error|error|failed|timeout|abort)' \
    "$JOB_LOGS/$service-prefill-rank-"*.out "$JOB_LOGS/$ACTIVE_DECODE_SERVICE-decode-rank-"*.out "$JOB_LOGS/$service-router.out" \
    >>"$output" 2>/dev/null || true
  grep -E -n 'KVTransferError' "$JOB_LOGS/$service-prefill-rank-"*.out "$JOB_LOGS/$ACTIVE_DECODE_SERVICE-decode-rank-"*.out "$JOB_LOGS/$service-router.out" \
    | grep -v 'AbortReq' >>"$output" 2>/dev/null || true
  if [[ -s "$output" ]]; then cat "$output"; echo "SETUP_INVALID reason=FATAL_OR_KV_TRANSFER service=$service label=$label"; return 1; fi
  echo "ERROR_SCAN_PASS service=$service label=$label"
}

release_rack_file() {
  if [[ -f "${RACK_FILE:-}" ]] && grep -Fq "job_id=$SLURM_JOB_ID " "$RACK_FILE"; then
    rm -f "$RACK_FILE"
  fi
}

safe_delete_runtime() {
  [[ "$RUNTIME" == "$ROOT/runtime/$SLURM_JOB_ID" ]] || {
    echo "REFUSE_DELETE_RUNTIME path=$RUNTIME"
    return 1
  }
  r12_safe_delete_runtime "$ROOT" "$RUNTIME"
}

cleanup() {
  local rc=$?
  trap - EXIT
  stop_router || true
  stop_prefill || true
  stop_decode || true
  safe_delete_runtime || true
  release_rack_file
  echo "JOB_CLEANUP $(date -u +%FT%TZ) job=$SLURM_JOB_ID experiment=$R12_JOB rc=$rc normal_complete=$NORMAL_COMPLETE source_pipdeps_cache_deleted=1"
  exit "$rc"
}

collect_topology() {
  local node=$1
  local output=$JOB_LOGS/topology-$node.out
  srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$node" \
    --container-image="$IMG" --no-container-entrypoint --no-container-mount-home \
    --container-mounts="$MOUNTS" --container-workdir=/src --container-remap-root \
    python3 -c 'import os,subprocess; a=set(os.sched_getaffinity(0)); by={}; [by.setdefault(int(x.split(",")[1]),set()).add(int(x.split(",")[0])) for x in subprocess.check_output(["lscpu","-p=CPU,NODE"],text=True).splitlines() if x and not x.startswith("#")]; hit={k:len(v&a) for k,v in by.items()}; print("AFFINITY",min(a),max(a),len(a),sorted(a)); print("AFFINITY_BY_NUMA",hit); assert hit=={0:72,1:72},hit' \
    >"$output" 2>&1 || { cat "$output"; echo "SETUP_INVALID reason=AFFINITY_BY_NUMA node=$node"; return 1; }
  cat "$output"
}

port_probe() {
  local node=$1
  shift
  srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$node" \
    python3 -c 'import socket,sys; held=[]; ports=list(map(int,sys.argv[1:])); [(lambda s,p:(s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1),s.bind(("0.0.0.0",p)),held.append(s)))(socket.socket(),p) for p in ports]; print("PORTS_FREE",socket.gethostname(),ports); [s.close() for s in held]' "$@"
}
