#!/bin/bash

# Shared R12 orchestration.  The allocated-job wrapper supplies the real
# lifecycle/benchmark functions; walkthrough_r12.sh supplies deterministic
# CPU-only stubs.  Keeping the branch graph here makes the full path test run
# the same normal, fallback, timeout, proof, and summary control flow as GPU
# jobs.

r12_run_scan() {
  local service=$1 arm=$2 topology=$3
  for concurrency in 16 32 64; do
    echo "GRID_BEGIN $(date -u +%FT%TZ) service=$service arm=$arm C=$concurrency isl=8192 osl=2"
    run_point "$service" "$arm-C$concurrency" "$concurrency" "$topology"
    echo "GRID_END $(date -u +%FT%TZ) service=$service arm=$arm C=$concurrency isl=8192 osl=2"
  done
}

r12_start_service() {
  local service=$1 variant=$2 chunk=$3 boot=$4
  launch_prefill "$service" "$variant" "$chunk" "$boot"
  wait_servers "$service"
  launch_router "$service"
}

r12_run_workflow() {
  echo "DECODE_LIFECYCLE normal=resident fallback=whole-group-restart source=$R12_EXPECTED_SOURCE_SHA"
  launch_decode decode-normal "$BOOT_NORMAL"
  launch_prefill B-main TEP 8192 "$BOOT_NORMAL"
  wait_servers B-main
  launch_router B-main
  discard_window=$JOB_LOGS/B-main-C16-discard-windows.out
  : >"$discard_window"
  run_bench B-main-C16-discard "$discard_window" 16 160 discard 40
  echo "DISCARD_ROUND_EXCLUDED arm=B-C16 prompts=160 seed=40 before=warmup"
  r12_run_scan B-main B tep

  stop_router; stop_prefill
  r12_start_service A-main PP "$R12_PP_CHUNK" "$BOOT_NORMAL"
  probe_window=$JOB_LOGS/A-main-handoff-probe-windows.out
  : >"$probe_window"
  probe_failed=0
  run_bench A-main-handoff-probe "$probe_window" 16 60 handoff_probe 45 || probe_failed=1
  handoff_signature=0
  if grep -Ehi 'session not alive|Failed to get kvcache|bootstrap.*(fail|timeout)|handshake.*(fail|timeout)|KVTransferError' \
      "$JOB_LOGS/A-main-prefill-rank-"*.out "$JOB_LOGS/decode-normal-decode-rank-"*.out "$JOB_LOGS/A-main-router.out" \
      | grep -vq 'AbortReq'; then handoff_signature=1; fi
  DECODE_RESTART_FALLBACK=0
  A_MAIN_SERVICE=A-main
  if [[ "$probe_failed" == 1 || "$handoff_signature" == 1 ]]; then
    DECODE_RESTART_FALLBACK=1
    echo "DECODE_RESTART_FALLBACK=1 probe_failed=$probe_failed signature=$handoff_signature action=restart_decode_prefill_router"
    stop_router; stop_prefill; stop_decode
    launch_decode decode-fallback "$BOOT_FALLBACK"
    launch_prefill A-main-fallback PP "$R12_PP_CHUNK" "$BOOT_FALLBACK"
    wait_servers A-main-fallback
    launch_router A-main-fallback
    A_MAIN_SERVICE=A-main-fallback
  else
    echo "DECODE_RESTART_FALLBACK=0 probe_completed=60"
  fi
  r12_run_scan "$A_MAIN_SERVICE" A pp

  stop_router; stop_prefill
  r12_start_service B-repeat TEP 8192 "$BOOT_NORMAL"
  run_point B-repeat B-C16-repeat 16 tep

  stop_router; stop_prefill
  A_REPEAT_SKIPPED=0
  elapsed=$(($(date +%s)-JOB_START_EPOCH))
  remaining=$((7200-elapsed))
  if (( remaining < 1500 )); then
    A_REPEAT_SKIPPED=1
    echo "TIMEOUT_CUT arm=A-C16-repeat remaining_s=$remaining threshold_s=1500 deltaA=max_over_min_upper_bound"
  else
    r12_start_service A-repeat PP "$R12_PP_CHUNK" "$BOOT_NORMAL"
    run_point A-repeat A-C16-repeat 16 pp
    stop_router; stop_prefill
  fi
  stop_decode

  for rank in 0 1; do
    cmp "$RENDERED/decode-normal-decode-rank$rank.out" "$ACTUAL/decode-normal-decode-rank$rank.out"
  done
  echo "LAUNCH_RENDER_COMBINED_MATCH normal_decode=1 fallback=$DECODE_RESTART_FALLBACK"
  set +e
  diff -u "$ACTUAL/B-main-prefill-rank0.out" "$ACTUAL/$A_MAIN_SERVICE-prefill-rank0.out" \
    >"$JOB_PROOF/raw-diff-B-main-vs-A-main-rank0.out"
  raw_diff_rc=$?
  set -e
  [[ "$raw_diff_rc" == 0 || "$raw_diff_rc" == 1 ]]
  echo "RAW_PROOF_DIFF_DONE rc=$raw_diff_rc output=$JOB_PROOF/raw-diff-B-main-vs-A-main-rank0.out"

  python3 "$SCRIPT_DIR/summarize_r12.py" --root "$JOB_RESULTS" --job "$R12_JOB" \
    --pp-chunk "$R12_PP_CHUNK" --decode-restart-fallback "$DECODE_RESTART_FALLBACK" \
    --a-repeat-skipped "$A_REPEAT_SKIPPED" --output "$JOB_RESULTS/$R12_JOB-summary.json"
  find "$JOB_LOGS" "$JOB_RESULTS" "$JOB_PROOF" -type f -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum >"$JOB_RESULTS/job-input-files.sha256"
  NORMAL_COMPLETE=1
  echo "JOB_END $(date -u +%FT%TZ) job=$SLURM_JOB_ID experiment=$R12_JOB pp_chunk=$R12_PP_CHUNK decode_restart_fallback=$DECODE_RESTART_FALLBACK a_repeat_skipped=$A_REPEAT_SKIPPED"
}
