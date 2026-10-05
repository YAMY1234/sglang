#!/bin/bash

# Shared benchmark-point implementation.  Both the allocated wrapper and the
# CPU walkthrough source this exact function so `set -u`/path construction is
# exercised before GPU submission.

run_point() {
  local service=$1 result_arm=$2 concurrency=$3 topology=$4
  local prefix=$service-C$concurrency
  local window=$JOB_LOGS/$prefix-windows.out
  local warmup_prompts=${R12_WARMUP_PROMPTS:-64}
  local formal_prompts=${R12_FORMAL_PROMPTS:-160}
  local formal_rounds=${R12_FORMAL_ROUNDS:-3}
  local round
  : >"$window"
  run_bench "$prefix" "$window" "$concurrency" "$warmup_prompts" warmup 41
  for round in $(seq 1 "$formal_rounds"); do
    run_bench "$prefix" "$window" "$concurrency" "$formal_prompts" \
      "formal$round" "$((41 + round))"
  done
  if declare -F r12_analyze_point >/dev/null; then
    r12_analyze_point "$service" "$result_arm" "$concurrency" "$topology" \
      "$prefix" "$window" "$formal_prompts" "$formal_rounds"
    return
  fi
  python3 "$SCRIPT_DIR/analyze_r12.py" --arm "$result_arm" --topology "$topology" \
    --windows "$window" --prefill "$JOB_LOGS/$service-prefill-rank-0.out" \
    --prefill "$JOB_LOGS/$service-prefill-rank-1.out" \
    --server-log-timezone "${R12_SERVER_LOG_TIMEZONE:-America/Los_Angeles}" \
    --bench-json "$JOB_LOGS/bench-$prefix-formal1.json" \
    --bench-json "$JOB_LOGS/bench-$prefix-formal2.json" \
    --bench-json "$JOB_LOGS/bench-$prefix-formal3.json" --out-dir "$JOB_RESULTS/$result_arm"
  python3 - "$JOB_RESULTS/$result_arm/point-summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
m=d["mechanism"]
print("PYTHON_TRACE_COUNTS", "arm="+d["arm"], "engine_returns="+str(sum(m["r7_formal_engine_returns"].values())))
print("TRACE_MODE="+m["trace_mode"], "arm="+d["arm"], "kv_transfer="+m["kv_transfer"])
print("TRACE_COMPLETENESS="+m["trace_completeness"], "arm="+d["arm"], m["r7_formal_engine_returns"])
PY
  scan_fatal "$service" "C$concurrency"
  local r7_combined=$JOB_RESULTS/$result_arm/r7-trace-combined.out
  grep -h 'R7_TRACE' "$JOB_LOGS/$service-prefill-rank-"*.out | LC_ALL=C sort >"$r7_combined" || true
  if [[ "$topology" == pp && -s "$r7_combined" ]]; then
    set +e
    python3 "$SCRIPT_DIR/analyze_r9_pp.py" --wrapper "$window" --prefill "$r7_combined" \
      --out-dir "$JOB_RESULTS/$result_arm/r9-pp" >"$JOB_RESULTS/$result_arm/r9-pp.stdout" 2>&1
    local r7_rc=$?
    set -e
    echo "R7_PP_ANALYZER arm=$result_arm rc=$r7_rc output=$JOB_RESULTS/$result_arm/r9-pp"
  else
    echo "R7_TRACE_UNAVAILABLE_OR_NOT_PP arm=$result_arm topology=$topology source=$R12_EXPECTED_SOURCE_SHA fallback=batch-log-mechanism"
  fi
  find "$JOB_LOGS" "$JOB_RESULTS/$result_arm" -type f \
    \( -name "*$service*C$concurrency*" -o -path "$JOB_RESULTS/$result_arm/*" \) -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum >"$JOB_RESULTS/$result_arm/input-files.sha256"
}
