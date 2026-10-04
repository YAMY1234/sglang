#!/bin/bash

set -euo pipefail

OUT_ROOT=${1:?usage: walkthrough_r12_qwen.sh NEW_OUTPUT_DIRECTORY}
[[ ! -e "$OUT_ROOT" ]] || { echo "QWEN_WALKTHROUGH_REFUSE_EXISTING_OUTPUT path=$OUT_ROOT" >&2; exit 1; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/r12_qwen_e2e_lib.sh"
source "$SCRIPT_DIR/r12_qwen_runtime_lib.sh"
source "$SCRIPT_DIR/r12_point_lib.sh"
source "$SCRIPT_DIR/r12_workflow.sh"

mkdir -p "$OUT_ROOT"/{logs,results,proof/rendered,proof/actual,root/runtime}
ROOT=$OUT_ROOT/root
RUNTIME=$ROOT/runtime/qwen-walkthrough
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
R12_REPEAT_HANDOFF_PROBE=1
R12_INJECT_REPEAT_PROBE_FAILURE=1
R12_ROLE_RANKS=0
SLURM_JOB_ID=qwen-walkthrough
JOB_START_EPOCH=$(date +%s)
NORMAL_COMPLETE=0

P_NODE=fake-prefill
D_NODE=fake-decode
P_PORT=11001
D_PORT=21001
ROUTER_PORT=31001
P_NCCL_PORT=51001
D_NCCL_PORT=61001
BOOT_NORMAL=41001
BOOT_FALLBACK=42001
export P_NODE D_NODE P_PORT D_PORT ROUTER_PORT P_NCCL_PORT D_NCCL_PORT \
  BOOT_NORMAL BOOT_FALLBACK

IMG=fake-image
MOUNTS=fake-mounts
PREFILL_PID=
DECODE_PID=
ROUTER_PID=
ACTIVE_DECODE_SERVICE=
NORMAL_DECODE_PID=

bash "$SCRIPT_DIR/render_r12_qwen_e2e.sh" "$RENDERED"

walkthrough_cleanup() {
  local rc=$?
  trap - EXIT
  stop_router || true
  stop_prefill || true
  stop_decode || true
  case "$RUNTIME" in "$ROOT/runtime/$SLURM_JOB_ID") ;; *)
    echo "QWEN_WALKTHROUGH_REFUSE_DELETE path=$RUNTIME" >&2
    exit "$rc"
  esac
  if [[ -e "$RUNTIME" ]]; then find "$RUNTIME" -depth -delete; fi
  echo "QWEN_WALKTHROUGH_EXIT_CLEANUP rc=$rc normal_complete=$NORMAL_COMPLETE runtime_deleted=1"
  exit "$rc"
}
trap walkthrough_cleanup EXIT

# Fake only the allocated external processes.  Lifecycle, proof, benchmark
# validation, point analysis, fallback, and summary functions are the exact
# functions sourced by run_r12_qwen_e2e.sbatch.
srun() {
  local arg role= service= variant= chunk= boot= output_file= prompts=0
  local is_router=0 is_bench=0 previous=
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
    esac
    [[ "$previous" != --num-prompts ]] || prompts=$arg
    previous=$arg
  done

  if [[ -n "$role" ]]; then
    local expected=$BOOT_NORMAL label=PREFILL_LAUNCH
    [[ "$service" != *-fallback ]] || expected=$BOOT_FALLBACK
    [[ "$role" != decode ]] || label=DECODE_LAUNCH
    [[ "$boot" == "$expected" ]]
    r12_qwen_build_role_command "$role" "$variant" "$chunk" \
      "$([[ "$role" == prefill ]] && echo "$P_PORT" || echo "$D_PORT")" \
      "$([[ "$role" == prefill ]] && echo "$P_NCCL_PORT" || echo "$D_NCCL_PORT")" \
      "$boot" "$service"
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
    [[ "$output_file" != *A-main* && "$output_file" != *A-repeat* ]] || value=10000
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

echo "JOB_START $(date -u +%FT%TZ) job=$SLURM_JOB_ID experiment=qwen-e2e walkthrough=true-functions"
r12_run_workflow

[[ ! -e "$RUNTIME" ]] || true
