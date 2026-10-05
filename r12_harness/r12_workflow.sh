#!/bin/bash

# Shared whole-group workflow. Every service generation starts prefill,
# decode, and router together on one bootstrap port and stops all three before
# the next generation. Allocated wrappers and CPU walkthroughs source this file.

r12_run_scan() {
  local service=$1
  local arm=$2
  local topology=$3
  local concurrency
  for concurrency in ${R12_CONCURRENCIES:-16 32 64}; do
    echo "GRID_BEGIN $(date -u +%FT%TZ) service=$service arm=$arm C=$concurrency isl=8192 osl=2"
    run_point "$service" "$arm-C$concurrency" "$concurrency" "$topology"
    echo "GRID_END $(date -u +%FT%TZ) service=$service arm=$arm C=$concurrency isl=8192 osl=2"
  done
}

r12_start_group() {
  local service=$1
  local variant=$2
  local chunk=$3
  local attempt
  local max_attempts=${R12_GROUP_START_MAX_ATTEMPTS:-3}
  R12_SERVER_FAILURE_REASON=UNKNOWN
  for attempt in $(seq 1 "$max_attempts"); do
    echo "GROUP_START_BEGIN $(date -u +%FT%TZ) service=$service variant=$variant chunk=$chunk bootstrap=$BOOTSTRAP_PORT attempt=$attempt max_attempts=$max_attempts"
    if ! prewarm_dynamic_module_caches "$service"; then
      r12_collect_start_failure "$service" "$attempt" DYNAMIC_MODULE_PREWARM_FAIL
    else
      launch_decode "$service" "$BOOTSTRAP_PORT"
      launch_prefill "$service" "$variant" "$chunk" "$BOOTSTRAP_PORT"
      if wait_servers "$service"; then
        if launch_router "$service"; then
          echo "GROUP_START_PASS $(date -u +%FT%TZ) service=$service bootstrap=$BOOTSTRAP_PORT attempt=$attempt"
          return 0
        fi
        r12_collect_start_failure "$service" "$attempt" ROUTER_PROCESS_EXIT
      else
        r12_collect_start_failure "$service" "$attempt"
      fi
    fi
    stop_router || true
    stop_prefill || true
    stop_decode || true
    echo "GROUP_START_ATTEMPT_CLEANUP $(date -u +%FT%TZ) service=$service attempt=$attempt reason=$R12_SERVER_FAILURE_REASON"
    if (( attempt < max_attempts )); then
      echo "GROUP_START_RETRY $(date -u +%FT%TZ) service=$service completed_attempt=$attempt next_attempt=$((attempt+1)) reason=$R12_SERVER_FAILURE_REASON config_unchanged=1"
    fi
  done

  python3 - "$R12_JOB" "$service" "$R12_SERVER_FAILURE_REASON" "$max_attempts" <<'PY'
import json
import sys
job, service, reason, attempts = sys.argv[1:]
record = {
    "job": job,
    "service": service,
    "reason": reason,
    "attempts": int(attempts),
    "trace_mode": "native_logs",
    "trace_completeness": "N/A_NO_TRACE",
    "verdict": "SERVER_START_FAIL",
}
print("R12_SUMMARY", json.dumps(record, sort_keys=True, separators=(",", ":")))
PY
  NORMAL_COMPLETE=1
  R12_WORKFLOW_TERMINAL=1
  echo "JOB_END $(date -u +%FT%TZ) job=$SLURM_JOB_ID experiment=$R12_JOB reason=server_start_fail service=$service attempts=$max_attempts"
}

r12_stop_group() {
  local service=$1
  stop_router
  stop_prefill
  stop_decode
  echo "GROUP_STOP_PASS $(date -u +%FT%TZ) service=$service"
}

r12_remaining_seconds() {
  local now
  now=$(date +%s)
  echo "$((R12_JOB_END_EPOCH - now))"
}

r12_run_workflow() {
  echo "SERVICE_LIFECYCLE mode=whole_group_restart bootstrap_generations=1 source=$R12_EXPECTED_SOURCE_SHA"
  R12_WORKFLOW_TERMINAL=0

  r12_start_group B-main TEP 8192
  [[ "$R12_WORKFLOW_TERMINAL" != 1 ]] || return 0
  local discard_window=$JOB_LOGS/B-main-C16-discard-windows.out
  : >"$discard_window"
  local discard_prompts=${R12_DISCARD_PROMPTS:-160}
  run_bench B-main-C16-discard "$discard_window" 16 "$discard_prompts" discard 40
  echo "DISCARD_ROUND_EXCLUDED arm=B-C16 prompts=$discard_prompts seed=40 before=warmup"
  r12_run_scan B-main B tep
  r12_stop_group B-main

  r12_start_group A-main PP "$R12_PP_CHUNK"
  [[ "$R12_WORKFLOW_TERMINAL" != 1 ]] || return 0
  r12_run_scan A-main A pp
  r12_stop_group A-main

  r12_start_group B-repeat TEP 8192
  [[ "$R12_WORKFLOW_TERMINAL" != 1 ]] || return 0
  run_point B-repeat B-C16-repeat 16 tep
  r12_stop_group B-repeat

  local a_repeat_skipped=1
  local remaining
  remaining=$(r12_remaining_seconds)
  if [[ "${R12_INCLUDE_A_REPEAT:-1}" == 1 && "$remaining" -ge "${R12_A_REPEAT_MIN_REMAINING:-1800}" ]]; then
    a_repeat_skipped=0
    r12_start_group A-repeat PP "$R12_PP_CHUNK"
    [[ "$R12_WORKFLOW_TERMINAL" != 1 ]] || return 0
    run_point A-repeat A-C16-repeat 16 pp
    r12_stop_group A-repeat
  elif [[ "${R12_INCLUDE_A_REPEAT:-1}" == 1 ]]; then
    echo "TIMEOUT_CUT arm=A-C16-repeat remaining_s=$remaining threshold_s=${R12_A_REPEAT_MIN_REMAINING:-1800} deltaA=max_over_min_upper_bound"
  else
    echo "A_REPEAT_NOT_REQUESTED validation_mode=1 remaining_s=$remaining"
  fi

  set +e
  diff -u "$ACTUAL/B-main-prefill-rank0.out" "$ACTUAL/A-main-prefill-rank0.out" \
    >"$JOB_PROOF/raw-diff-B-main-vs-A-main-rank0.out"
  local raw_diff_rc=$?
  set -e
  [[ "$raw_diff_rc" == 0 || "$raw_diff_rc" == 1 ]]
  echo "RAW_PROOF_DIFF_DONE rc=$raw_diff_rc output=$JOB_PROOF/raw-diff-B-main-vs-A-main-rank0.out"
  python3 "$SCRIPT_DIR/prove_r12_semantic_diff.py" --root "$ACTUAL" \
    --pp-chunk "$R12_PP_CHUNK" --output "$JOB_PROOF/actual-semantic-proof.json"

  if declare -F r12_summarize_results >/dev/null; then
    r12_summarize_results "$a_repeat_skipped"
  else
    python3 "$SCRIPT_DIR/summarize_r12.py" --root "$JOB_RESULTS" --job "$R12_JOB" \
      --pp-chunk "$R12_PP_CHUNK" --decode-restart-fallback 0 \
      --a-repeat-skipped "$a_repeat_skipped" --output "$JOB_RESULTS/$R12_JOB-summary.json"
  fi
  find "$JOB_LOGS" "$JOB_RESULTS" "$JOB_PROOF" -type f -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum >"$JOB_RESULTS/job-input-files.sha256"
  NORMAL_COMPLETE=1
  echo "JOB_END $(date -u +%FT%TZ) job=$SLURM_JOB_ID experiment=$R12_JOB pp_chunk=$R12_PP_CHUNK lifecycle=whole_group_restart a_repeat_skipped=$a_repeat_skipped"
}
