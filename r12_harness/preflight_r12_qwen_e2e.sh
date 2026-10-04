#!/bin/bash

set -euo pipefail

OUT=${1:?usage: preflight_r12_qwen_e2e.sh NEW_OUTPUT_DIRECTORY [SOURCE_ARCHIVE]}
SOURCE_ARCHIVE=${2:-}
[[ ! -e "$OUT" ]] || { echo "QWEN_PREFLIGHT_REFUSE_EXISTING_OUTPUT path=$OUT" >&2; exit 1; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT"
shellcheck_status=NOT_INSTALLED
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck -S warning "$SCRIPT_DIR/run_r12_qwen_e2e.sbatch" \
    "$SCRIPT_DIR/r12_workflow.sh" "$SCRIPT_DIR/r12_point_lib.sh" \
    "$SCRIPT_DIR/r12_qwen_runtime_lib.sh" "$SCRIPT_DIR/walkthrough_r12_qwen.sh" \
    "$SCRIPT_DIR/run_r12_qwen_role.sh" "$SCRIPT_DIR/render_r12_qwen_e2e.sh" \
    >"$OUT/shellcheck.out"
  shellcheck_status=PASS
fi
R12_QWEN_DRY_RUN=1 R12_QWEN_DRY_RUN_OUT="$OUT/rendered" \
  bash "$SCRIPT_DIR/run_r12_qwen_e2e.sbatch" >"$OUT/dry-run.out" 2>&1
if [[ -n "$SOURCE_ARCHIVE" ]]; then
  python3 "$SCRIPT_DIR/check_qwen_feature_support.py" --source-archive "$SOURCE_ARCHIVE" \
    --rendered-root "$OUT/rendered" --output "$OUT/feature-support-proof.json" \
    >"$OUT/feature-support.out"
  grep -Fq 'FEATURE_SUPPORT_GATE_PASS' "$OUT/feature-support.out"
fi
grep -Fq 'QWEN_RENDER_ASSERTIONS_PASS' "$OUT/dry-run.out"
grep -Fq 'QWEN_RENDER_PASS' "$OUT/dry-run.out"
bash "$SCRIPT_DIR/walkthrough_r12_qwen.sh" "$OUT/walkthrough" \
  >"$OUT/walkthrough.out" 2>&1
grep -Fq 'SERVER_START_PASS' "$OUT/walkthrough.out"
grep -Fq 'ROUTER_START_PASS' "$OUT/walkthrough.out"
grep -Fq 'DECODE_RESIDENT_GATE_PASS service=A-main' "$OUT/walkthrough.out"
grep -Fq 'REPEAT_PROBE_FAILURE_INJECTED=1 completed_real_probe=60' "$OUT/walkthrough.out"
grep -Fq 'DECODE_RESTART_FALLBACK=1 repeat_probe_failed=1' "$OUT/walkthrough.out"
grep -Fq 'SERVER_START_PASS ' "$OUT/walkthrough.out"
grep -Fq 'service=A-repeat-fallback' "$OUT/walkthrough.out"
grep -Fq 'ERROR_SCAN_PASS service=A-repeat-fallback label=C16' "$OUT/walkthrough.out"
grep -Fq 'R12_SUMMARY ' "$OUT/walkthrough.out"
grep -Fq 'JOB_END ' "$OUT/walkthrough.out"
grep -Fq 'QWEN_WALKTHROUGH_EXIT_CLEANUP rc=0 normal_complete=1 runtime_deleted=1' \
  "$OUT/walkthrough.out"
[[ ! -e "$OUT/walkthrough/root/runtime/qwen-walkthrough" ]]
python3 - "$OUT/walkthrough/results/qwen-e2e-summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d["verdict"] == "PASS", d
assert d["gates"]["DECODE_RESTART_FALLBACK"] == 1, d
assert d["gates"]["A_REPEAT_SKIPPED_TIMEOUT"] == 0, d
assert d["gates"]["request_completion"] == "128/128", d
PY
find "$SCRIPT_DIR" -maxdepth 1 -type f \
  \( -name 'r12_*.sh' -o -name 'walkthrough_r12_qwen.sh' \
     -o -name 'run_r12_qwen_e2e.sbatch' -o -name 'check_qwen_feature_support.py' \) \
  -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >"$OUT/preflight-inputs.sha256"
printf 'QWEN_PREFLIGHT_PASS shellcheck=%s feature_gate=%s\n' "$shellcheck_status" \
  "$([[ -n "$SOURCE_ARCHIVE" ]] && echo PASS || echo DEFERRED_TO_AGA)" | tee "$OUT/verdict.out"
