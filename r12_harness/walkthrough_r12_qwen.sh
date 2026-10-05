#!/bin/bash

set -euo pipefail

OUT_ROOT=${1:?usage: walkthrough_r12_qwen.sh NEW_OUTPUT_DIRECTORY}
[[ ! -e "$OUT_ROOT" ]] || { echo "QWEN_WALKTHROUGH_REFUSE_EXISTING_OUTPUT path=$OUT_ROOT" >&2; exit 1; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT_ROOT"

run_walkthrough() (
  declare -F | awk '{print $3}' | LC_ALL=C sort -u >"$OUT_ROOT/functions-before.txt"
  source "$SCRIPT_DIR/r12_qwen_e2e_lib.sh"
  source "$SCRIPT_DIR/r12_qwen_runtime_lib.sh"
  source "$SCRIPT_DIR/r12_point_lib.sh"
  source "$SCRIPT_DIR/r12_workflow.sh"
  declare -F | awk '{print $3}' | LC_ALL=C sort -u >"$OUT_ROOT/functions-after.txt"
  comm -13 "$OUT_ROOT/functions-before.txt" "$OUT_ROOT/functions-after.txt" \
    >"$OUT_ROOT/functions-expected.txt"

  mkdir -p "$OUT_ROOT"/{logs,results,proof/rendered,proof/actual,root/runtime}
  ROOT=$OUT_ROOT/root
  RUNTIME=$ROOT/runtime/qwen-walkthrough-v2
  JOB_LOGS=$OUT_ROOT/logs
  JOB_RESULTS=$OUT_ROOT/results
  JOB_PROOF=$OUT_ROOT/proof
  RENDERED=$JOB_PROOF/rendered
  ACTUAL=$JOB_PROOF/actual
  mkdir -p "$RUNTIME"/{source,pipdeps,cache}
  touch "$RUNTIME/source/sentinel" "$RUNTIME/pipdeps/sentinel" "$RUNTIME/cache/sentinel"

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
  SLURM_JOB_ID=qwen-walkthrough-v2
  R12_JOB_END_EPOCH=$(($(date +%s) + 1800))
  NORMAL_COMPLETE=0

  P_NODE=fake-prefill
  D_NODE=fake-decode
  P_PORT=11001
  D_PORT=21001
  ROUTER_PORT=31001
  P_NCCL_PORT=51001
  D_NCCL_PORT=61001
  BOOTSTRAP_PORT=41001
  export P_NODE D_NODE P_PORT D_PORT ROUTER_PORT P_NCCL_PORT D_NCCL_PORT BOOTSTRAP_PORT
  IMG=fake-image
  MOUNTS=fake-mounts
  PREFILL_PID=
  DECODE_PID=
  ROUTER_PID=
  ACTIVE_DECODE_SERVICE=

  bash "$SCRIPT_DIR/render_r12_qwen_e2e.sh" "$RENDERED"

  srun() {
    local arg role= service= variant= chunk= boot= output_file= prompts=0
    local is_router=0 is_bench=0 is_topology=0 is_port_probe=0 previous=
    for arg in "$@"; do
      case "$arg" in
        R12_ROLE=*) role=${arg#*=} ;;
        R12_SERVICE=*) service=${arg#*=} ;;
        R12_VARIANT=*) variant=${arg#*=} ;;
        R12_CHUNK=*) chunk=${arg#*=} ;;
        R12_BOOTSTRAP_PORT=*) boot=${arg#*=} ;;
        /logs/bench-*.json) output_file=${arg#/logs/} ;;
        sglang_router.launch_router) is_router=1 ;;
        sglang.bench_serving) is_bench=1 ;;
        *AFFINITY_BY_NUMA*) is_topology=1 ;;
        *PORTS_FREE*) is_port_probe=1 ;;
      esac
      [[ "$previous" != --num-prompts ]] || prompts=$arg
      previous=$arg
    done
    if [[ "$is_topology" == 1 ]]; then
      echo 'AFFINITY_BY_NUMA {0: 72, 1: 72}'
      return 0
    fi
    if [[ "$is_port_probe" == 1 ]]; then
      echo 'PORTS_FREE fake-node [41001]'
      return 0
    fi
    if [[ -n "$role" ]]; then
      [[ "$boot" == "$BOOTSTRAP_PORT" ]]
      local label=PREFILL_LAUNCH http=$P_PORT nccl=$P_NCCL_PORT
      if [[ "$role" == decode ]]; then
        label=DECODE_LAUNCH
        http=$D_PORT
        nccl=$D_NCCL_PORT
      fi
      r12_qwen_build_role_command "$role" "$variant" "$chunk" "$http" "$nccl" "$boot" "$service"
      r12_qwen_emit_command "$label" "$ACTUAL/$service-$role-rank0.out"
      cmp "$RENDERED/$service-$role-rank0.out" "$ACTUAL/$service-$role-rank0.out"
      printf 'FAKE_EXTERNAL_SERVER role=%s service=%s bootstrap=%s\n' \
        "$role" "$service" "$boot" >"$JOB_LOGS/$service-$role-rank-0.out"
      touch "$JOB_LOGS/fake-$role-$BASHPID.ready"
      exec sleep 600
    fi
    if [[ "$is_router" == 1 ]]; then
      touch "$JOB_LOGS/fake-router-$BASHPID.ready"
      exec sleep 600
    fi
    if [[ "$is_bench" == 1 ]]; then
      local value=8000
      [[ "$output_file" != *A-* ]] || value=10000
      printf '{"completed":%s,"incomplete":0,"input_throughput":%s,"median_ttft_ms":10,"p90_ttft_ms":12}\n' \
        "$prompts" "$value" >"$JOB_LOGS/$output_file"
      echo "FAKE_EXTERNAL_BENCH_PASS output=$output_file completed=$prompts"
      return 0
    fi
    echo "FAKE_EXTERNAL_SRUN_UNHANDLED $*" >&2
    return 2
  }

  curl() {
    local url=${*: -1}
    case "$url" in
      *:$ROUTER_PORT/*) [[ -n "${ROUTER_PID:-}" ]] && kill -0 "$ROUTER_PID" 2>/dev/null \
        && [[ -f "$JOB_LOGS/fake-router-$ROUTER_PID.ready" ]] ;;
      *:$P_PORT/*) [[ -n "${PREFILL_PID:-}" ]] && kill -0 "$PREFILL_PID" 2>/dev/null \
        && [[ -f "$JOB_LOGS/fake-prefill-$PREFILL_PID.ready" ]] ;;
      *:$D_PORT/*) [[ -n "${DECODE_PID:-}" ]] && kill -0 "$DECODE_PID" 2>/dev/null \
        && [[ -f "$JOB_LOGS/fake-decode-$DECODE_PID.ready" ]] ;;
      *) return 1 ;;
    esac
  }

  : >"$OUT_ROOT/functions-executed.raw"
  set -T
  trap 'printf "%s\n" "${FUNCNAME[0]:-MAIN}" >>"$OUT_ROOT/functions-executed.raw"' DEBUG
  trap cleanup EXIT

  collect_topology "$P_NODE"
  port_probe "$P_NODE" "$BOOTSTRAP_PORT"
  echo "JOB_START $(date -u +%FT%TZ) job=$SLURM_JOB_ID experiment=qwen-e2e lifecycle=whole_group_restart walkthrough=true-functions"
  r12_run_workflow
)

run_walkthrough >"$OUT_ROOT/transcript.out" 2>&1
LC_ALL=C sort -u "$OUT_ROOT/functions-executed.raw" >"$OUT_ROOT/functions-executed.txt"
comm -23 "$OUT_ROOT/functions-expected.txt" "$OUT_ROOT/functions-executed.txt" \
  >"$OUT_ROOT/functions-uncovered.txt"
expected_count=$(wc -l <"$OUT_ROOT/functions-expected.txt")
executed_count=$(comm -12 "$OUT_ROOT/functions-expected.txt" "$OUT_ROOT/functions-executed.txt" | wc -l)
uncovered_count=$(wc -l <"$OUT_ROOT/functions-uncovered.txt")
{
  echo "COVERAGE_METHOD=trap_DEBUG_functrace"
  echo "EXPECTED_FUNCTIONS=$expected_count"
  echo "EXECUTED_FUNCTIONS=$executed_count"
  echo "UNCOVERED_FUNCTIONS=$uncovered_count"
  echo "UNCOVERED_BEGIN"
  cat "$OUT_ROOT/functions-uncovered.txt"
  echo "UNCOVERED_END"
} | tee "$OUT_ROOT/coverage.txt"
[[ "$uncovered_count" == 0 ]]
[[ ! -e "$OUT_ROOT/root/runtime/qwen-walkthrough-v2" ]]
grep -Fq 'SERVICE_LIFECYCLE mode=whole_group_restart bootstrap_generations=1' "$OUT_ROOT/transcript.out"
grep -Fq 'GROUP_START_PASS' "$OUT_ROOT/transcript.out"
grep -Fq 'GROUP_STOP_PASS' "$OUT_ROOT/transcript.out"
grep -Fq 'A_REPEAT_NOT_REQUESTED validation_mode=1' "$OUT_ROOT/transcript.out"
grep -Fq 'R12_SUMMARY ' "$OUT_ROOT/transcript.out"
grep -Fq 'request_completion":"96/96"' "$OUT_ROOT/transcript.out"
grep -Fq 'JOB_END ' "$OUT_ROOT/transcript.out"
grep -Fq 'JOB_CLEANUP ' "$OUT_ROOT/transcript.out"
grep -Fq 'normal_complete=1 source_pipdeps_cache_deleted=1' "$OUT_ROOT/transcript.out"
echo "QWEN_WALKTHROUGH_PASS output=$OUT_ROOT uncovered_functions=0"
