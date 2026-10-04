#!/bin/bash

set -euo pipefail

OUT_ROOT=${1:?usage: walkthrough_r12.sh NEW_OUTPUT_DIRECTORY}
[[ ! -e "$OUT_ROOT" ]] || { echo "WALKTHROUGH_REFUSE_EXISTING_OUTPUT path=$OUT_ROOT" >&2; exit 1; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/r12_launch_lib.sh"
source "$SCRIPT_DIR/r12_runtime_lib.sh"
source "$SCRIPT_DIR/r12_point_lib.sh"
source "$SCRIPT_DIR/r12_workflow.sh"
mkdir -p "$OUT_ROOT"

run_scenario() (
  local scenario=$1
  local scenario_root=$OUT_ROOT/$scenario
  mkdir -p "$scenario_root"/{logs,results,proof/rendered,proof/actual,root/runtime}

  ROOT=$scenario_root/root
  RUNTIME=$ROOT/runtime/walkthrough-$scenario
  JOB_LOGS=$scenario_root/logs
  JOB_RESULTS=$scenario_root/results
  JOB_PROOF=$scenario_root/proof
  RENDERED=$JOB_PROOF/rendered
  ACTUAL=$JOB_PROOF/actual
  mkdir -p "$RUNTIME"/{source,pipdeps,cache}
  touch "$RUNTIME/source/sentinel" "$RUNTIME/pipdeps/sentinel" "$RUNTIME/cache/sentinel"

  R12_JOB=job7b
  R12_PP_CHUNK=8192
  R12_MEM_FRACTION=0.90
  SLURM_JOB_ID=walkthrough-$scenario
  NORMAL_COMPLETE=0
  P0=fake-p0; P1=fake-p1; D0=fake-d0; D1=fake-d1
  P_PORT=10123; D_PORT=15123; ROUTER_PORT=20123
  P_DIST_PORT=25123; D_DIST_PORT=30123
  P_NCCL_PORT=35123; D_NCCL_PORT=40123
  BOOT_NORMAL=45123; BOOT_FALLBACK=46123
  export R12_JOB R12_PP_CHUNK R12_MEM_FRACTION P0 P1 D0 D1 P_PORT D_PORT \
    ROUTER_PORT P_DIST_PORT D_DIST_PORT P_NCCL_PORT D_NCCL_PORT \
    BOOT_NORMAL BOOT_FALLBACK
  if [[ "$scenario" == timeout ]]; then
    JOB_START_EPOCH=$(($(date +%s)-6001))
  else
    JOB_START_EPOCH=$(date +%s)
  fi

  bash "$SCRIPT_DIR/render_r12_launches.sh" "$RENDERED"

  walkthrough_cleanup() {
    local rc=$?
    trap - EXIT
    r12_safe_delete_runtime "$ROOT" "$RUNTIME"
    echo "WALKTHROUGH_EXIT_CLEANUP scenario=$scenario rc=$rc normal_complete=$NORMAL_COMPLETE runtime_deleted=1"
    exit "$rc"
  }
  trap walkthrough_cleanup EXIT

  assert_boot() {
    local proof=$1 boot=$2
    grep -Fq -- "--disaggregation-bootstrap-port $boot" "$proof"
  }

  # Only external processes are faked.  The workflow-facing launch/stop/wait,
  # benchmark, scan, point-analysis, proof, and summary functions above are the
  # exact functions sourced by the allocated wrapper.
  srun() {
    local arg role= service= variant= chunk= boot= output_file= prompts=0 rank
    local is_router=0 is_bench=0
    for arg in "$@"; do
      case "$arg" in
        R12_ROLE=*) role=${arg#*=} ;;
        R12_SERVICE=*) service=${arg#*=} ;;
        R12_VARIANT=*) variant=${arg#*=} ;;
        R12_CHUNK=*) chunk=${arg#*=} ;;
        R12_BOOTSTRAP_PORT=*) boot=${arg#*=} ;;
        /logs/bench-*.json) output_file=${arg#/logs/} ;;
        --num-prompts) ;;
        sglang_router.launch_router) is_router=1 ;;
        sglang.bench_serving) is_bench=1 ;;
      esac
    done
    local previous=
    for arg in "$@"; do
      [[ "$previous" != --num-prompts ]] || prompts=$arg
      previous=$arg
    done
    if [[ -n "$role" ]]; then
      local expected=$BOOT_NORMAL
      [[ "$service" != *-fallback ]] || expected=$BOOT_FALLBACK
      [[ "$boot" == "$expected" ]]
      for rank in 0 1; do
        r12_build_role_command "$role" "$variant" "$chunk" "$rank" \
          "$([[ "$role" == prefill ]] && echo "$P0" || echo "$D0")" \
          "$([[ "$role" == prefill ]] && echo "$P_DIST_PORT" || echo "$D_DIST_PORT")" \
          "$([[ "$role" == prefill ]] && echo "$P_PORT" || echo "$D_PORT")" \
          "$([[ "$role" == prefill ]] && echo "$P_NCCL_PORT" || echo "$D_NCCL_PORT")" \
          "$boot" "$service" "$R12_MEM_FRACTION"
        local label=PREFILL_LAUNCH
        [[ "$role" != decode ]] || label=DECODE_LAUNCH
        r12_emit_command "$label" "$ACTUAL/$service-$role-rank$rank.out"
        cmp "$RENDERED/$service-$role-rank$rank.out" "$ACTUAL/$service-$role-rank$rank.out"
        printf 'FAKE_EXTERNAL_SERVER role=%s service=%s rank=%s bootstrap=%s\n' \
          "$role" "$service" "$rank" "$boot" >"$JOB_LOGS/$service-$role-rank-$rank.out"
      done
      touch "$JOB_LOGS/fake-$role-$BASHPID.ready"
      exec sleep 600
    fi
    if [[ "$is_router" == 1 ]]; then
      touch "$JOB_LOGS/fake-router-$BASHPID.ready"
      exec sleep 600
    fi
    if [[ "$is_bench" == 1 ]]; then
      local label=${output_file%.json}
      if [[ "$scenario" == fallback && "$label" == *-handoff_probe ]]; then
        echo "FAKE_EXTERNAL_BENCH_FAIL label=handoff_probe completed=0 expected=$prompts"
        return 1
      fi
      local value=8000
      [[ "$label" != bench-A-* ]] || value=$((10000 + prompts))
      [[ "$label" != bench-B-repeat-* ]] || value=7960
      [[ "$label" != bench-A-repeat-* ]] || value=10020
      printf '{"completed":%s,"incomplete":0,"input_throughput":%s,"median_ttft_ms":10,"p90_ttft_ms":12}\n' \
        "$prompts" "$value" >"$JOB_LOGS/$output_file"
      echo "FAKE_EXTERNAL_BENCH_PASS output=$output_file completed=$prompts"
      return
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

  IMG=fake-image
  MOUNTS=fake-mounts
  P_NODELIST=$P0,$P1
  D_NODELIST=$D0,$D1
  PREFILL_PID=; DECODE_PID=; ROUTER_PID=; ACTIVE_DECODE_SERVICE=

  echo "JOB_START $(date -u +%FT%TZ) job=$SLURM_JOB_ID experiment=$R12_JOB pp_chunk=$R12_PP_CHUNK mem_fraction=$R12_MEM_FRACTION walkthrough=$scenario"
  r12_run_workflow
)

for scenario in normal fallback timeout; do
  scenario_root=$OUT_ROOT/$scenario
  run_scenario "$scenario" >"$scenario_root.transcript.tmp" 2>&1
  mkdir -p "$scenario_root"
  mv "$scenario_root.transcript.tmp" "$scenario_root/transcript.out"
  [[ ! -e "$scenario_root/root/runtime/walkthrough-$scenario" ]]
  grep -Fq 'NUMA_BIND_LOG_GATE service=B-main skip_count=0 expected=0' "$scenario_root/transcript.out"
  grep -Fq 'R12_SUMMARY ' "$scenario_root/transcript.out"
  grep -Fq 'JOB_END ' "$scenario_root/transcript.out"
  grep -Fq "WALKTHROUGH_EXIT_CLEANUP scenario=$scenario rc=0 normal_complete=1 runtime_deleted=1" "$scenario_root/transcript.out"
done

grep -Fq 'SERVER_START_PASS' "$OUT_ROOT/normal/transcript.out"
grep -Fq 'DECODE_RESTART_FALLBACK=0 probe_completed=60' "$OUT_ROOT/normal/transcript.out"
grep -Fq 'TRACE_MODE=native_logs arm=A-C16-repeat' "$OUT_ROOT/normal/transcript.out"

grep -Fq 'FAKE_EXTERNAL_BENCH_FAIL label=handoff_probe' "$OUT_ROOT/fallback/transcript.out"
grep -Fq 'DECODE_RESTART_FALLBACK=1 probe_failed=1 signature=0' "$OUT_ROOT/fallback/transcript.out"
grep -Fq 'SERVER_START_PASS' "$OUT_ROOT/fallback/transcript.out"
grep -Fq 'SERVER_START_PASS' "$OUT_ROOT/fallback/transcript.out"

grep -Fq 'TIMEOUT_CUT arm=A-C16-repeat' "$OUT_ROOT/timeout/transcript.out"
if grep -Fq 'TRACE_MODE=native_logs arm=A-C16-repeat' "$OUT_ROOT/timeout/transcript.out"; then
  echo "WALKTHROUGH_TIMEOUT_FAILED_TO_CUT_A_REPEAT" >&2
  exit 1
fi

python3 - "$OUT_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected = {
    "normal": (0, 0),
    "fallback": (1, 0),
    "timeout": (0, 1),
}
for scenario, (fallback, timeout) in expected.items():
    data = json.loads((root / scenario / "results" / "job7b-summary.json").read_text())
    assert data["gates"]["DECODE_RESTART_FALLBACK"] == fallback
    assert data["gates"]["A_REPEAT_SKIPPED_TIMEOUT"] == timeout
record = {
    "branches": {
        "normal_handoff_probe_pass": "PASS",
        "probe_failure_whole_group_fallback": "PASS",
        "timeout_cut_a_repeat": "PASS",
        "exit_cleanup_exact_runtime": "PASS",
        "real_lifecycle_functions_fake_external_only": "PASS",
        "zero_match_numa_log_gate": "PASS",
        "shared_bootstrap_normal_and_fallback": "PASS",
        "real_run_point_set_u_and_analysis": "PASS",
        "summary_all_scenarios": "PASS",
    },
    "scenarios": expected,
    "verdict": "PASS",
}
(root / "walkthrough-verdict.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print("WALKTHROUGH_BRANCH_ASSERTIONS_PASS", json.dumps(record, sort_keys=True))
PY

find "$OUT_ROOT" -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
  >"$OUT_ROOT/walkthrough-files.sha256"
echo "WALKTHROUGH_PASS output=$OUT_ROOT"
