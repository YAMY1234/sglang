#!/bin/bash

set -euo pipefail

OUT_ROOT=${1:?usage: walkthrough_r12.sh NEW_OUTPUT_DIRECTORY}
[[ ! -e "$OUT_ROOT" ]] || { echo "WALKTHROUGH_REFUSE_EXISTING_OUTPUT path=$OUT_ROOT" >&2; exit 1; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/r12_launch_lib.sh"
source "$SCRIPT_DIR/r12_workflow.sh"
mkdir -p "$OUT_ROOT"

write_fake_point() {
  local output=$1 arm=$2 concurrency=$3
  local value=8000
  [[ "$arm" != A-* ]] || value=$((10000 + concurrency * 10))
  [[ "$arm" != B-C16-repeat ]] || value=7960
  [[ "$arm" != A-C16-repeat ]] || value=10020
  mkdir -p "$output"
  python3 - "$output/point-summary.json" "$arm" "$value" <<'PY'
import json, pathlib, sys
path, arm, value = pathlib.Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
row = {
    "arm": arm,
    "benchmark": {
        "median_input_throughput": value,
        "median_per_prefill_gpu": value / 8,
        "max_over_min": (value + 5) / (value - 5),
        "rounds": [
            {"input_throughput": value - 5},
            {"input_throughput": value},
            {"input_throughput": value + 5},
        ],
    },
}
path.write_text(json.dumps(row, sort_keys=True) + "\n")
PY
}

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

  launch_decode() {
    local service=$1 boot=$2 rank
    local expected=$BOOT_NORMAL
    [[ "$service" != decode-fallback ]] || expected=$BOOT_FALLBACK
    [[ "$boot" == "$expected" ]]
    for rank in 0 1; do
      assert_boot "$RENDERED/$service-decode-rank$rank.out" "$boot"
      cp "$RENDERED/$service-decode-rank$rank.out" "$ACTUAL/$service-decode-rank$rank.out"
      printf 'STUB_SERVER_READY role=decode service=%s rank=%s bootstrap=%s\n' \
        "$service" "$rank" "$boot" >"$JOB_LOGS/$service-decode-rank-$rank.out"
    done
    ACTIVE_DECODE_SERVICE=$service
    echo "STUB_LAUNCH role=decode service=$service bootstrap=$boot immediate_return=1"
  }

  launch_prefill() {
    local service=$1 variant=$2 chunk=$3 boot=$4 rank
    local expected=$BOOT_NORMAL
    [[ "$service" != A-main-fallback ]] || expected=$BOOT_FALLBACK
    [[ "$boot" == "$expected" ]]
    for rank in 0 1; do
      assert_boot "$RENDERED/$service-prefill-rank$rank.out" "$boot"
      cp "$RENDERED/$service-prefill-rank$rank.out" "$ACTUAL/$service-prefill-rank$rank.out"
      printf 'STUB_SERVER_READY role=prefill service=%s rank=%s variant=%s chunk=%s bootstrap=%s\n' \
        "$service" "$rank" "$variant" "$chunk" "$boot" >"$JOB_LOGS/$service-prefill-rank-$rank.out"
    done
    echo "STUB_LAUNCH role=prefill service=$service variant=$variant chunk=$chunk bootstrap=$boot immediate_return=1"
  }

  wait_servers() {
    local service=$1 skip
    skip=$(r12_count_numa_skip_logs "$JOB_LOGS" "$service" "$ACTIVE_DECODE_SERVICE")
    echo "NUMA_BIND_LOG_GATE service=$service skip_count=$skip expected=0"
    [[ "$skip" == 0 ]]
    echo "SERVER_START_PASS $(date -u +%FT%TZ) service=$service stub=1"
  }

  launch_router() {
    local service=$1
    cp "$RENDERED/$service-router.out" "$ACTUAL/$service-router.out"
    printf 'STUB_ROUTER_READY service=%s p=%s:%s d=%s:%s router_port=%s\n' \
      "$service" "$P0" "$P_PORT" "$D0" "$D_PORT" "$ROUTER_PORT" \
      >"$JOB_LOGS/$service-router.out"
    echo "ROUTER_START_PASS $(date -u +%FT%TZ) service=$service stub=1"
  }

  stop_router() { echo "STUB_STOP role=router"; }
  stop_prefill() { echo "STUB_STOP role=prefill"; }
  stop_decode() { echo "STUB_STOP role=decode service=${ACTIVE_DECODE_SERVICE:-none}"; }

  run_bench() {
    local prefix=$1 window=$2 concurrency=$3 prompts=$4 label=$5 seed=$6
    echo "BENCH_BEGIN $(date -u +%FT%TZ) C=$concurrency label=$label prompts=$prompts output_len=2 seed=$seed stub=1" | tee -a "$window"
    if [[ "$scenario" == fallback && "$label" == handoff_probe ]]; then
      echo "STUB_BENCH_FAIL label=$label completed=0 expected=$prompts"
      return 1
    fi
    printf '{"completed":%s,"incomplete":0}\n' "$prompts" >"$JOB_LOGS/bench-$prefix-$label.json"
    echo "BENCH_END $(date -u +%FT%TZ) C=$concurrency label=$label prompts=$prompts output_len=2 seed=$seed stub=1" | tee -a "$window"
  }

  run_point() {
    local service=$1 result_arm=$2 concurrency=$3 topology=$4
    write_fake_point "$JOB_RESULTS/$result_arm" "$result_arm" "$concurrency"
    echo "R12_POINT_STUB service=$service arm=$result_arm C=$concurrency topology=$topology completed=480 expected=480"
  }

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
grep -Fq 'R12_POINT_STUB service=A-repeat arm=A-C16-repeat' "$OUT_ROOT/normal/transcript.out"

grep -Fq 'STUB_BENCH_FAIL label=handoff_probe' "$OUT_ROOT/fallback/transcript.out"
grep -Fq 'DECODE_RESTART_FALLBACK=1 probe_failed=1 signature=0' "$OUT_ROOT/fallback/transcript.out"
grep -Fq "STUB_LAUNCH role=decode service=decode-fallback bootstrap=46123" "$OUT_ROOT/fallback/transcript.out"
grep -Fq 'SERVER_START_PASS' "$OUT_ROOT/fallback/transcript.out"

grep -Fq 'TIMEOUT_CUT arm=A-C16-repeat' "$OUT_ROOT/timeout/transcript.out"
if grep -Fq 'R12_POINT_STUB service=A-repeat arm=A-C16-repeat' "$OUT_ROOT/timeout/transcript.out"; then
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
        "zero_match_numa_log_gate": "PASS",
        "shared_bootstrap_normal_and_fallback": "PASS",
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
