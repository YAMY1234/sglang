#!/bin/bash

set -euo pipefail

OUT=${1:?usage: preflight_r12_qwen_e2e.sh NEW_OUTPUT_DIRECTORY [SOURCE_ARCHIVE]}
SOURCE_ARCHIVE=${2:-}
[[ ! -e "$OUT" ]] || { echo "QWEN_PREFLIGHT_REFUSE_EXISTING_OUTPUT path=$OUT" >&2; exit 1; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT/static-sweep-v1"

bash -n "$SCRIPT_DIR"/*.sh "$SCRIPT_DIR"/*.sbatch
python3 - "$SCRIPT_DIR" <<'PY'
import pathlib,sys
for path in pathlib.Path(sys.argv[1]).glob("*.py"):
    compile(path.read_text(),str(path),"exec")
print("QWEN_PREFLIGHT_SYNTAX_PASS")
PY

shellcheck_status=NOT_INSTALLED
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck -S warning "$SCRIPT_DIR"/*.sh "$SCRIPT_DIR"/*.sbatch \
    >"$OUT/static-sweep-v1/shellcheck.out"
  shellcheck_status=PASS
fi
scan_files=("$SCRIPT_DIR"/*.sh "$SCRIPT_DIR"/*.sbatch)
printf 'python3 %q --output %q' "$SCRIPT_DIR/static_sweep_r12.py" \
  "$OUT/static-sweep-v1/declaration-sweep.json" >"$OUT/static-sweep-v1/command.txt"
printf ' %q' "${scan_files[@]}" >>"$OUT/static-sweep-v1/command.txt"
printf '\n' >>"$OUT/static-sweep-v1/command.txt"
python3 "$SCRIPT_DIR/static_sweep_r12.py" \
  --output "$OUT/static-sweep-v1/declaration-sweep.json" "${scan_files[@]}" \
  | tee "$OUT/static-sweep-v1/result.out"
grep -Fq 'violations=0 verdict=PASS' "$OUT/static-sweep-v1/result.out"

R12_QWEN_DRY_RUN=1 R12_QWEN_DRY_RUN_OUT="$OUT/rendered" \
  bash "$SCRIPT_DIR/run_r12_qwen_e2e.sbatch" >"$OUT/dry-run.out" 2>&1
grep -Fq 'QWEN_RENDER_ASSERTIONS_PASS roles=6 staging=1 bootstrap_ports=1' "$OUT/dry-run.out"
grep -Fq 'QWEN_RENDER_PASS' "$OUT/dry-run.out"

if [[ -n "$SOURCE_ARCHIVE" ]]; then
  python3 "$SCRIPT_DIR/check_qwen_feature_support.py" --source-archive "$SOURCE_ARCHIVE" \
    --rendered-root "$OUT/rendered" --output "$OUT/feature-support-proof.json" \
    | tee "$OUT/feature-support.out"
  grep -Fq 'FEATURE_SUPPORT_GATE_PASS' "$OUT/feature-support.out"
  feature_status=PASS
else
  feature_status=DEFERRED_TO_AGA
fi

HARNESS=$SCRIPT_DIR/run_r12_qwen_e2e.sbatch
WORKFLOW=$SCRIPT_DIR/r12_workflow.sh
grep -Fqx '#SBATCH --time=01:30:00' "$HARNESS"
grep -Fqx '#SBATCH --nodes=2' "$HARNESS"
grep -Fqx '#SBATCH --gpus-per-node=4' "$HARNESS"
grep -Fq 'scripted_time_kill=disabled' "$HARNESS"
grep -Fq 'bootstrap=$BOOTSTRAP_PORT shared_pd=1 generations=1' "$HARNESS"
grep -Fq 'SERVICE_LIFECYCLE mode=whole_group_restart bootstrap_generations=1' "$WORKFLOW"
if grep -Eq 'BUDGET_WATCHDOG|BUDGET_TIMEOUT|scancel|kill -TERM \$\$' \
    "$HARNESS" "$WORKFLOW" "$SCRIPT_DIR/r12_qwen_runtime_lib.sh" \
    "$SCRIPT_DIR/run_r12_job7.sbatch" "$SCRIPT_DIR/r12_runtime_lib.sh"; then
  echo "PREFLIGHT_SCRIPTED_TIME_KILL_FORBIDDEN" >&2
  exit 1
fi
if grep -Eqi 'probe60|handoff_probe|decode-normal|decode-fallback|BOOT_FALLBACK|BOOT_NORMAL' \
    "$HARNESS" "$WORKFLOW" "$SCRIPT_DIR/render_r12_qwen_e2e.sh"; then
  echo "QWEN_PREFLIGHT_COMPLEX_LIFECYCLE_REGRESSION" >&2
  exit 1
fi
if grep -Eq '^[A-Za-z_][A-Za-z0-9_]*\(\)' "$HARNESS"; then
  echo "QWEN_PREFLIGHT_WRAPPER_FUNCTION_NOT_EXTRACTED" >&2
  exit 1
fi

bash "$SCRIPT_DIR/walkthrough_r12_qwen.sh" "$OUT/walkthrough-v2" \
  >"$OUT/walkthrough-v2.stdout" 2>&1
grep -Fq 'UNCOVERED_FUNCTIONS=0' "$OUT/walkthrough-v2/coverage.txt"
grep -Fq 'QWEN_WALKTHROUGH_PASS' "$OUT/walkthrough-v2.stdout"
grep -Fq 'R12_SUMMARY ' "$OUT/walkthrough-v2/transcript.out"
grep -Fq 'request_completion":"96/96"' "$OUT/walkthrough-v2/transcript.out"
grep -Fq 'JOB_END ' "$OUT/walkthrough-v2/transcript.out"
grep -Fq 'JOB_CLEANUP ' "$OUT/walkthrough-v2/transcript.out"
[[ ! -e "$OUT/walkthrough-v2/root/runtime/qwen-walkthrough-v2" ]]

find "$SCRIPT_DIR" -maxdepth 1 -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
  >"$OUT/preflight-inputs.sha256"
printf 'QWEN_PREFLIGHT_PASS lifecycle=whole_group_restart static_sweep=PASS shellcheck=%s feature_gate=%s uncovered_functions=0\n' \
  "$shellcheck_status" "$feature_status" | tee "$OUT/verdict.out"
