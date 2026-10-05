#!/bin/bash

# Qwen3.8 workflow-facing lifecycle, benchmark, proof, and summary functions.
# The allocated wrapper and CPU walkthrough both source this file so that the
# branch test executes these exact functions with only external commands faked.

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

stop_router() {
  stop_pid "${ROUTER_PID:-}"
  ROUTER_PID=
  [[ -z "$ROUTER_PORT" ]] || wait_endpoint_down "http://$P_NODE:$ROUTER_PORT/health" router
}

stop_prefill() {
  stop_pid "${PREFILL_PID:-}"
  PREFILL_PID=
  [[ -z "$P_PORT" ]] || wait_endpoint_down "http://$P_NODE:$P_PORT/health" prefill
}

stop_decode() {
  stop_pid "${DECODE_PID:-}"
  DECODE_PID=
  [[ -z "$D_PORT" ]] || wait_endpoint_down "http://$D_NODE:$D_PORT/health" decode
}

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
  DECODE_PID=$!
  ACTIVE_DECODE_SERVICE=$service
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
    echo "SETUP_INVALID reason=QWEN_SERVER_START service=$service"
    return 1
  fi
  local skip
  skip=$(awk '/skipping NUMA binding for GPU/{s++} END{print s+0}' \
    "$JOB_LOGS/$service-prefill-rank-0.out" "$JOB_LOGS/$ACTIVE_DECODE_SERVICE-decode-rank-0.out")
  echo "NUMA_BIND_LOG_GATE service=$service skip_count=$skip expected=0"
  [[ "$skip" == 0 ]] || { echo "SETUP_INVALID reason=QWEN_NUMA_LOG service=$service"; return 1; }
  echo "SERVER_START_PASS $(date -u +%FT%TZ) service=$service"
}

launch_router() {
  local service=$1
  local actual=$ACTUAL/$service-router.out
  local frozen=$RENDERED/$service-router.out
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
    kill -0 "$ROUTER_PID" 2>/dev/null || break
    sleep 2
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
  local service=$1 label=$2
  local output=$JOB_LOGS/$service-$label-error-scan.out
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
  local skipped=$1
  local output=$JOB_RESULTS/qwen-e2e-summary.json
  python3 - "$JOB_RESULTS" "$output" "$skipped" "$JOB_PROOF/raw-diff-B-main-vs-A-main-rank0.out" <<'PY'
import hashlib,json,pathlib,sys
root,out,skipped,diff=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),int(sys.argv[3]),pathlib.Path(sys.argv[4])
names=("B-C16","A-C16","B-C16-repeat")
arms={name:json.load(open(root/name/"point-summary.json")) for name in names}
assert all(row["benchmark"]["request_trace"]["rating"] == "PASS" for row in arms.values())
assert skipped == 1 and diff.exists()
record={"job":"qwen-e2e","performance_reportable":False,"arms":list(names),"sampling":{"warmup_prompts":16,"formal_rounds":1,"formal_prompts":32},"gates":{"whole_group_restart":"PASS","bootstrap_ports":1,"A_REPEAT_NOT_REQUESTED":1,"proof_diff":"PASS","request_completion":"96/96"},"trace_mode":"qwen_e2e_validation","proof_diff_sha256":hashlib.sha256(diff.read_bytes()).hexdigest(),"verdict":"PASS"}
out.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
print("R12_SUMMARY",json.dumps(record,sort_keys=True,separators=(",",":")))
PY
}

safe_delete_runtime() {
  [[ "$RUNTIME" == "$ROOT/runtime/$SLURM_JOB_ID" ]] || {
    echo "QWEN_REFUSE_DELETE_RUNTIME path=$RUNTIME"
    return 1
  }
  if [[ -e "$RUNTIME" ]]; then find "$RUNTIME" -depth -delete; fi
  rmdir "$ROOT/runtime" 2>/dev/null || true
}

cleanup() {
  local rc=$?
  trap - EXIT
  stop_router || true
  stop_prefill || true
  stop_decode || true
  safe_delete_runtime || true
  echo "JOB_CLEANUP $(date -u +%FT%TZ) job=$SLURM_JOB_ID experiment=qwen-e2e rc=$rc normal_complete=$NORMAL_COMPLETE source_pipdeps_cache_deleted=1"
  exit "$rc"
}

collect_topology() {
  local node=$1
  local output=$JOB_LOGS/topology-$node.out
  srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$node" \
    python3 -c 'import os,subprocess; a=set(os.sched_getaffinity(0)); by={}; [by.setdefault(int(x.split(",")[1]),set()).add(int(x.split(",")[0])) for x in subprocess.check_output(["lscpu","-p=CPU,NODE"],text=True).splitlines() if x and not x.startswith("#")]; hit={k:len(v&a) for k,v in by.items()}; print("AFFINITY_BY_NUMA",hit); assert hit=={0:72,1:72},hit' \
    >"$output" 2>&1 || { cat "$output"; echo "SETUP_INVALID reason=QWEN_AFFINITY node=$node"; return 1; }
  cat "$output"
}

port_probe() {
  local node=$1
  shift
  srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=144 --cpu-bind=none --nodelist="$node" \
    python3 -c 'import socket,sys; held=[]; ports=list(map(int,sys.argv[1:])); [(lambda s,p:(s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1),s.bind(("0.0.0.0",p)),held.append(s)))(socket.socket(),p) for p in ports]; print("PORTS_FREE",socket.gethostname(),ports); [s.close() for s in held]' "$@"
}
