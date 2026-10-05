#!/bin/bash

set -euo pipefail

OUT_ROOT=${1:?usage: preflight_r12.sh NEW_OUTPUT_DIRECTORY}
[[ ! -e "$OUT_ROOT" ]] || { echo "PREFLIGHT_REFUSE_EXISTING_OUTPUT path=$OUT_ROOT" >&2; exit 1; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT_ROOT"/{job7b,job7c,static-sweep-v1}

for script in "$SCRIPT_DIR"/*.sh "$SCRIPT_DIR"/*.sbatch; do bash -n "$script"; done
python3 - "$SCRIPT_DIR" <<'PY'
import pathlib,sys
for path in pathlib.Path(sys.argv[1]).glob("*.py"):
    compile(path.read_text(), str(path), "exec")
print("PREFLIGHT_SYNTAX_PASS")
PY

shellcheck_status=NOT_INSTALLED
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck -S warning "$SCRIPT_DIR"/*.sh "$SCRIPT_DIR"/*.sbatch \
    >"$OUT_ROOT/static-sweep-v1/shellcheck.out"
  shellcheck_status=PASS
fi

scan_files=("$SCRIPT_DIR"/*.sh "$SCRIPT_DIR"/*.sbatch)
printf 'python3 %q --output %q' "$SCRIPT_DIR/static_sweep_r12.py" \
  "$OUT_ROOT/static-sweep-v1/declaration-sweep.json" \
  >"$OUT_ROOT/static-sweep-v1/command.txt"
printf ' %q' "${scan_files[@]}" >>"$OUT_ROOT/static-sweep-v1/command.txt"
printf '\n' >>"$OUT_ROOT/static-sweep-v1/command.txt"
python3 "$SCRIPT_DIR/static_sweep_r12.py" \
  --output "$OUT_ROOT/static-sweep-v1/declaration-sweep.json" \
  "${scan_files[@]}" | tee "$OUT_ROOT/static-sweep-v1/result.out"
grep -Fq 'violations=0 verdict=PASS' "$OUT_ROOT/static-sweep-v1/result.out"

env R12_DRY_RUN=1 R12_DRY_RUN_OUT="$OUT_ROOT/job7b" \
  R12_JOB=job7b R12_PP_CHUNK=8192 bash "$SCRIPT_DIR/run_r12_job7.sbatch" \
  >"$OUT_ROOT/job7b-dry-run.out"
env R12_DRY_RUN=1 R12_DRY_RUN_OUT="$OUT_ROOT/job7c" \
  R12_JOB=job7c R12_PP_CHUNK=16384 bash "$SCRIPT_DIR/run_r12_job7.sbatch" \
  >"$OUT_ROOT/job7c-dry-run.out"

FEATURE_SOURCE_ROOT=${R12_FEATURE_SOURCE_ROOT:-$SCRIPT_DIR/..}
FEATURE_SOURCE_ARCHIVE=${R12_SOURCE_ARCHIVE:-$SCRIPT_DIR/../prereq/sglang-cb0b3498fcc2f398229b0b8cb9df0a5825e1438a.tar.gz}
if [[ -f "$FEATURE_SOURCE_ROOT/python/sglang/srt/disaggregation/utils.py" ]]; then
  feature_source=(--source-root "$FEATURE_SOURCE_ROOT")
elif [[ -f "$FEATURE_SOURCE_ARCHIVE" ]]; then
  feature_source=(--source-archive "$FEATURE_SOURCE_ARCHIVE")
else
  echo "PREFLIGHT_FEATURE_SOURCE_MISSING root=$FEATURE_SOURCE_ROOT archive=$FEATURE_SOURCE_ARCHIVE" >&2
  exit 1
fi
python3 "$SCRIPT_DIR/check_feature_support.py" "${feature_source[@]}" \
  --rendered-root "$OUT_ROOT" --output "$OUT_ROOT/feature-support-proof.json" \
  | tee "$OUT_ROOT/feature-support-gate.out"

HARNESS=$SCRIPT_DIR/run_r12_job7.sbatch
WORKFLOW=$SCRIPT_DIR/r12_workflow.sh
grep -Fqx '#SBATCH --qos=short' "$HARNESS"
grep -Fqx '#SBATCH --nodes=4' "$HARNESS"
grep -Fqx '#SBATCH --cpus-per-task=144' "$HARNESS"
grep -Fqx '#SBATCH --exclusive' "$HARNESS"
grep -Fqx '#SBATCH --time=02:00:00' "$HARNESS"
if grep -Eq '^#SBATCH --(nice|hold|nodelist|dependency|constraint)\b' "$HARNESS"; then
  echo "PREFLIGHT_FORBIDDEN_SBATCH_OPTION" >&2
  exit 1
fi
grep -Fq 'bootstrap=$BOOTSTRAP_PORT shared_pd=1 generations=1' "$HARNESS"
grep -Fq 'SERVICE_LIFECYCLE mode=whole_group_restart bootstrap_generations=1' "$WORKFLOW"
grep -Fq 'TIMEOUT_CUT arm=A-C16-repeat' "$WORKFLOW"
if grep -Eq 'BUDGET_WATCHDOG|BUDGET_TIMEOUT|scancel|kill -TERM \$\$' \
    "$HARNESS" "$WORKFLOW" "$SCRIPT_DIR/r12_runtime_lib.sh" \
    "$SCRIPT_DIR/run_r12_qwen_e2e.sbatch" "$SCRIPT_DIR/r12_qwen_runtime_lib.sh"; then
  echo "PREFLIGHT_SCRIPTED_TIME_KILL_FORBIDDEN" >&2
  exit 1
fi
if grep -Eqi 'probe60|handoff_probe|decode-normal|decode-fallback|BOOT_FALLBACK|BOOT_NORMAL' \
    "$HARNESS" "$WORKFLOW" "$SCRIPT_DIR/render_r12_launches.sh"; then
  echo "PREFLIGHT_COMPLEX_LIFECYCLE_REGRESSION" >&2
  exit 1
fi
if grep -Eq '^[A-Za-z_][A-Za-z0-9_]*\(\)' "$HARNESS"; then
  echo "PREFLIGHT_WRAPPER_FUNCTION_NOT_EXTRACTED" >&2
  exit 1
fi

python3 - "$OUT_ROOT" <<'PY'
import json,pathlib,shlex,sys
root=pathlib.Path(sys.argv[1])
def command(path):
    label,raw=path.read_text().strip().split(" ",1)
    return label,shlex.split(raw)
def value(tokens,flag):
    assert tokens.count(flag)==1,(flag,tokens)
    return tokens[tokens.index(flag)+1]
services=("B-main","A-main","B-repeat","A-repeat")
for directory,chunk in ((root/"job7b","8192"),(root/"job7c","16384")):
    plan=directory.joinpath("service-plan.tsv").read_text().splitlines()
    assert len(plan)==5 and all(row.endswith("whole_group_restart") for row in plan[1:])
    boots=set()
    for service in services:
        prefill=command(directory/f"{service}-prefill-rank0.out")[1]
        decode=command(directory/f"{service}-decode-rank0.out")[1]
        boots.add(value(prefill,"--disaggregation-bootstrap-port"))
        boots.add(value(decode,"--disaggregation-bootstrap-port"))
        assert value(prefill,"--mem-fraction-static")=="0.90"
        assert value(decode,"--mem-fraction-static")=="0.90"
    assert len(boots)==1,boots
    assert value(command(directory/"A-main-prefill-rank0.out")[1],"--chunked-prefill-size")==chunk
for rank in (0,1):
    assert (root/"job7b"/f"B-main-prefill-rank{rank}.out").read_bytes()==(root/"job7c"/f"B-main-prefill-rank{rank}.out").read_bytes()
    for service in services:
        assert (root/"job7b"/f"{service}-decode-rank{rank}.out").read_bytes()==(root/"job7c"/f"{service}-decode-rank{rank}.out").read_bytes()
record={"jobs":{"job7b":8192,"job7c":16384},"services":list(services),"lifecycle":"whole_group_restart","bootstrap_ports":1,"decode_restarted_each_service":True,"probe_paths":0,"fallback_paths":0,"verdict":"PASS"}
(root/"proof-verdict.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
print("PREFLIGHT_PROOF_DIFF_PASS",json.dumps(record,sort_keys=True))
PY

bash "$SCRIPT_DIR/walkthrough_r12.sh" "$OUT_ROOT/walkthrough-v2" \
  >"$OUT_ROOT/walkthrough-v2.stdout" 2>&1
grep -Fq 'UNCOVERED_FUNCTIONS=0' "$OUT_ROOT/walkthrough-v2/coverage.txt"
grep -Fq 'WALKTHROUGH_PASS' "$OUT_ROOT/walkthrough-v2.stdout"

find "$SCRIPT_DIR" -maxdepth 1 -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
  >"$OUT_ROOT/preflight-inputs.sha256"
printf 'PREFLIGHT_PASS lifecycle=whole_group_restart static_sweep=PASS shellcheck=%s uncovered_functions=0\n' \
  "$shellcheck_status" | tee "$OUT_ROOT/verdict.out"
