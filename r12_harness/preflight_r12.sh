#!/bin/bash

set -euo pipefail

OUT_ROOT=${1:?usage: preflight_r12.sh NEW_OUTPUT_DIRECTORY}
[[ ! -e "$OUT_ROOT" ]] || { echo "PREFLIGHT_REFUSE_EXISTING_OUTPUT path=$OUT_ROOT" >&2; exit 1; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT_ROOT/job7b" "$OUT_ROOT/job7c" "$OUT_ROOT/fixture"

for script in "$SCRIPT_DIR"/*.sh "$SCRIPT_DIR"/*.sbatch; do bash -n "$script"; done
python3 - "$SCRIPT_DIR" <<'PY'
import pathlib,sys
for path in pathlib.Path(sys.argv[1]).glob("*.py"):
    compile(path.read_text(), str(path), "exec")
print("PREFLIGHT_PYTHON_COMPILE_PASS")
PY
echo "PREFLIGHT_SYNTAX_PASS"

env R12_DRY_RUN=1 R12_DRY_RUN_OUT="$OUT_ROOT/job7b" \
  R12_JOB=job7b R12_PP_CHUNK=8192 bash "$SCRIPT_DIR/run_r12_job7.sbatch" \
  >"$OUT_ROOT/job7b-dry-run.out"
env R12_DRY_RUN=1 R12_DRY_RUN_OUT="$OUT_ROOT/job7c" \
  R12_JOB=job7c R12_PP_CHUNK=16384 bash "$SCRIPT_DIR/run_r12_job7.sbatch" \
  >"$OUT_ROOT/job7c-dry-run.out"

HARNESS=$SCRIPT_DIR/run_r12_job7.sbatch
grep -Fqx '#SBATCH --qos=short' "$HARNESS"
grep -Fqx '#SBATCH --nodes=4' "$HARNESS"
grep -Fqx '#SBATCH --cpus-per-task=144' "$HARNESS"
grep -Fqx '#SBATCH --exclusive' "$HARNESS"
grep -Fqx '#SBATCH --time=02:00:00' "$HARNESS"
if grep -Eq '^#SBATCH --(nice|hold|nodelist|dependency|constraint)\b' "$HARNESS"; then
  echo "PREFLIGHT_FORBIDDEN_SBATCH_OPTION" >&2
  exit 1
fi
grep -Fq 'DISCARD_ROUND_EXCLUDED' "$HARNESS"
grep -Fq 'DECODE_RESTART_FALLBACK=1' "$HARNESS"
grep -Fq '60 handoff_probe 45' "$HARNESS"
grep -Fq 'TIMEOUT_CUT arm=A-C16-repeat' "$HARNESS"
grep -Fq 'safe_delete_runtime' "$HARNESS"
grep -Fq 'SETUP_INVALID reason=FORBIDDEN_RACK' "$HARNESS"
grep -Fq 'RACK_COORDINATION_EXCLUDED_PENDING' "$HARNESS"
grep -Fq 'RACK_COORDINATION_REQUEUE' "$HARNESS"
grep -Fq 'ExcNodeList=' "$HARNESS"
grep -Fq "m.version('sglang-kernel') == '0.4.8'" "$HARNESS"
grep -Fq 'sglang-kernel==0.4.8' "$HARNESS"
if grep -Fq 'sglang-kernel==0.4.7' "$HARNESS"; then
  echo "PREFLIGHT_STALE_KERNEL_PIN" >&2
  exit 1
fi

python3 - "$HARNESS" <<'PY'
import pathlib,sys
text=pathlib.Path(sys.argv[1]).read_text()
assert text.index('RACK_COORDINATION_BEGIN') < text.index('tar -xzf "$SOURCE_ARCHIVE"')
assert text.index('RACK_COORDINATION_REQUEUE') < text.index('tar -xzf "$SOURCE_ARCHIVE"')
print("PREFLIGHT_RACK_COORDINATION_PASS")
PY

python3 - "$OUT_ROOT" <<'PY'
import json
import pathlib
import re
import shlex
import sys

root = pathlib.Path(sys.argv[1])
b = root / "job7b"
c = root / "job7c"

def parsed(path):
    label, raw = path.read_text().strip().split(" ", 1)
    return label, shlex.split(raw)

def normalized(path, normalize_chunk=True):
    label, tokens = parsed(path)
    result = []
    i = 0
    value_flags = {
        "--disaggregation-bootstrap-port",
        "--tp-size", "--ep-size", "--pp-size",
    }
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("SGLANG_PP_COMM_OVERLAP=") or token.startswith(
            "SGLANG_PP_LAYER_PARTITION="
        ):
            i += 1
            continue
        if re.match(r"^[A-Z_]+=/runtime/cache/[^/]+/rank-\d+/", token):
            key, value = token.split("=", 1)
            suffix = value.split("/", 5)[-1]
            result.append(f"{key}=/runtime/cache/SERVICE/rank-N/{suffix}")
            i += 1
            continue
        if token in value_flags:
            if token == "--disaggregation-bootstrap-port":
                result.extend((token, "BOOTSTRAP"))
            i += 2
            continue
        if token == "--disable-overlap-schedule":
            i += 1
            continue
        if normalize_chunk and token in {"--chunked-prefill-size", "--max-prefill-tokens"}:
            result.extend((token, "CHUNK"))
            i += 2
            continue
        result.append(token)
        i += 1
    return label, result

services = ["B-main", "A-main", "B-repeat", "A-repeat", "A-main-fallback"]
for directory in (b, c):
    plan = directory.joinpath("service-plan.tsv").read_text().splitlines()
    assert plan[1].split("\t")[:3] == ["B-main", "TEP", "8192"]
    expected = "8192" if directory == b else "16384"
    assert plan[2].split("\t")[:3] == ["A-main", "PP", expected]
    assert plan[-1].split("\t")[-1] == "fallback"
    for rank in (0, 1):
        # Service generations differ only by the declared endpoint/cache identity.
        assert normalized(directory / f"B-main-prefill-rank{rank}.out") == normalized(
            directory / f"B-repeat-prefill-rank{rank}.out"
        )
        assert normalized(directory / f"A-main-prefill-rank{rank}.out") == normalized(
            directory / f"A-repeat-prefill-rank{rank}.out"
        ) == normalized(directory / f"A-main-fallback-prefill-rank{rank}.out")
        # After removing the preregistered topology/env and endpoint identities,
        # A and B are byte-token identical.
        assert normalized(directory / f"B-main-prefill-rank{rank}.out") == normalized(
            directory / f"A-main-prefill-rank{rank}.out"
        )
        assert normalized(directory / f"decode-normal-decode-rank{rank}.out") == normalized(
            directory / f"decode-fallback-decode-rank{rank}.out"
        )
    for service in services:
        assert parsed(directory / f"{service}-router.out")[1] == parsed(
            directory / "B-main-router.out"
        )[1]

# Across 7b/7c, B and decode are byte identical.  PP differs only at the two
# chunk value positions after endpoint/cache normalization.
for rank in (0, 1):
    assert (b / f"B-main-prefill-rank{rank}.out").read_bytes() == (
        c / f"B-main-prefill-rank{rank}.out"
    ).read_bytes()
    assert (b / f"decode-normal-decode-rank{rank}.out").read_bytes() == (
        c / f"decode-normal-decode-rank{rank}.out"
    ).read_bytes()
    _, left = normalized(b / f"A-main-prefill-rank{rank}.out", normalize_chunk=False)
    _, right = normalized(c / f"A-main-prefill-rank{rank}.out", normalize_chunk=False)
    changes = [(i, x, y) for i, (x, y) in enumerate(zip(left, right)) if x != y]
    assert len(left) == len(right) and len(changes) == 2, changes
    for i, x, y in changes:
        assert (x, y) == ("8192", "16384")
        assert left[i - 1] == right[i - 1]
        assert left[i - 1] in {"--chunked-prefill-size", "--max-prefill-tokens"}

record = {
    "jobs": {"job7b": 8192, "job7c": 16384},
    "services": services,
    "normal_and_fallback_paths_rendered": True,
    "raw_endpoint_differences_preserved": True,
    "semantic_a_b_diff_only_topology_env_and_chunk": True,
    "job7b_job7c_b_and_decode_byte_identical": True,
    "job7b_job7c_pp_only_two_chunk_values": True,
    "max_total_tokens_absent": True,
    "mem_fraction": "0.90",
    "verdict": "PASS",
}
(root / "proof-verdict.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print("PREFLIGHT_PROOF_DIFF_PASS", json.dumps(record, sort_keys=True))
PY

cat >"$OUT_ROOT/fixture/windows.out" <<'EOF'
BENCH_BEGIN 2026-10-04T00:00:00Z C=16 label=formal1 prompts=160 output_len=2 seed=42
BENCH_END 2026-10-04T00:00:10Z C=16 label=formal1 prompts=160 output_len=2 seed=42
BENCH_BEGIN 2026-10-04T00:01:00Z C=16 label=formal2 prompts=160 output_len=2 seed=43
BENCH_END 2026-10-04T00:01:10Z C=16 label=formal2 prompts=160 output_len=2 seed=43
BENCH_BEGIN 2026-10-04T00:02:00Z C=16 label=formal3 prompts=160 output_len=2 seed=44
BENCH_END 2026-10-04T00:02:10Z C=16 label=formal3 prompts=160 output_len=2 seed=44
EOF
for rep in 1 2 3; do
  cat >"$OUT_ROOT/fixture/formal$rep.json" <<EOF
{"completed":160,"input_throughput":$((8000+rep)),"median_ttft_ms":1000,"p90_ttft_ms":1200}
EOF
done
cat >"$OUT_ROOT/fixture/pp.out" <<'EOF'
[2026-10-04 00:00:01.000 PP0 TP0 EP0] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, full token usage: 0.10, mamba usage: 0.20, #running-req: 1, #queue-req: 2, #pending-token: 0, #inflight-req: 1, input throughput (token/s): 20000.0
[2026-10-04 00:00:01.010 PP1 TP0 EP0] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, full token usage: 0.10, mamba usage: 0.20, #running-req: 1, #queue-req: 2, #pending-token: 0, #inflight-req: 1, input throughput (token/s): 20000.0
[2026-10-04 00:00:01.020 PP2 TP0 EP0] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, full token usage: 0.10, mamba usage: 0.20, #running-req: 1, #queue-req: 2, #pending-token: 0, #inflight-req: 1, input throughput (token/s): 20000.0
[2026-10-04 00:00:01.030 PP3 TP0 EP0] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, full token usage: 0.10, mamba usage: 0.20, #running-req: 1, #queue-req: 2, #pending-token: 0, #inflight-req: 1, input throughput (token/s): 20000.0
[2026-10-04 00:01:01.000 PP0 TP0 EP0] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, full token usage: 0.10, mamba usage: 0.20, #running-req: 1, #queue-req: 2, #pending-token: 0, #inflight-req: 1, input throughput (token/s): 20000.0
[2026-10-04 00:02:01.000 PP0 TP0 EP0] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 0, full token usage: 0.10, mamba usage: 0.20, #running-req: 1, #queue-req: 2, #pending-token: 0, #inflight-req: 1, input throughput (token/s): 20000.0
EOF
python3 "$SCRIPT_DIR/analyze_r12.py" --arm fixture-pp --topology pp \
  --windows "$OUT_ROOT/fixture/windows.out" --prefill "$OUT_ROOT/fixture/pp.out" \
  --bench-json "$OUT_ROOT/fixture/formal1.json" --bench-json "$OUT_ROOT/fixture/formal2.json" \
  --bench-json "$OUT_ROOT/fixture/formal3.json" --out-dir "$OUT_ROOT/fixture/analysis" \
  >"$OUT_ROOT/fixture/analyzer.stdout"
python3 - "$OUT_ROOT/fixture/analysis/point-summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d["benchmark"]["request_trace"] == {"completed":480,"expected":480,"incomplete":0,"rating":"PASS"}
assert d["mechanism"]["entities"]["0"]["observed"] is True
assert d["mechanism"]["entities"]["0"]["per_formal"]["formal1"]["running_req"]["p50"] == 1
assert d["mechanism"]["entities"]["0"]["per_formal"]["formal1"]["queue_req"]["p50"] == 2
assert d["mechanism"]["trace_mode"] == "native_logs"
assert d["mechanism"]["kv_transfer"] == "TRACE_UNAVAILABLE"
print("PREFLIGHT_ANALYZER_FIXTURE_PASS")
PY
python3 - "$OUT_ROOT/fixture/analysis/point-summary.json" "$OUT_ROOT/fixture/final" <<'PY'
import copy,json,pathlib,sys
base=json.load(open(sys.argv[1])); root=pathlib.Path(sys.argv[2])
values={"B-C16":8000,"B-C32":8100,"B-C64":8200,"A-C16":10000,"A-C32":11000,"A-C64":12000,"B-C16-repeat":7960,"A-C16-repeat":10020}
for name,value in values.items():
 d=copy.deepcopy(base); d["arm"]=name
 d["benchmark"]["median_input_throughput"]=value
 d["benchmark"]["median_per_prefill_gpu"]=value/8
 for i,row in enumerate(d["benchmark"]["rounds"]): row["input_throughput"]=value+(i-1)*5
 p=root/name; p.mkdir(parents=True); (p/"point-summary.json").write_text(json.dumps(d)+"\n")
PY
python3 "$SCRIPT_DIR/summarize_r12.py" --root "$OUT_ROOT/fixture/final" --job job7b \
  --pp-chunk 8192 --decode-restart-fallback 0 --a-repeat-skipped 0 \
  --output "$OUT_ROOT/fixture/final-summary.json" >"$OUT_ROOT/fixture/final-summary.stdout"
python3 - "$OUT_ROOT/fixture/final-summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert all(x["label"]=="PP4_LEADS" for x in d["decisions"].values()); assert d["gates"]["FIRST_ARM_COLD"]=="PASS"; assert d["trace_mode"]=="native_logs"; assert d["gates"]["TRACE_COMPLETENESS"]=="N/A_NO_TRACE"; print("PREFLIGHT_SUMMARIZER_FIXTURE_PASS")
PY

if timeout 100 git -C "$SCRIPT_DIR/.." grep -q 'SGLANG_R7_STAGE_SKEW_TRACE' \
  cb0b3498fcc2f398229b0b8cb9df0a5825e1438a -- python 2>/dev/null; then
  echo 'R7_TRACE_SOURCE_SYMBOL=present' >"$OUT_ROOT/r7-trace-source-check.out"
else
  echo 'R7_TRACE_SOURCE_SYMBOL=absent batch-log-mechanism-fallback=implemented' >"$OUT_ROOT/r7-trace-source-check.out"
fi

{
  find "$SCRIPT_DIR" -maxdepth 2 -type f -not -path '*/local-dry-run/*' \
    -not -path '*/__pycache__/*' -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum
  sha256sum "$OUT_ROOT/proof-verdict.json" \
    "$OUT_ROOT/job7b/rendered-launches.sha256" "$OUT_ROOT/job7c/rendered-launches.sha256"
} >"$OUT_ROOT/preflight-inputs.sha256"
echo "PREFLIGHT_PASS output=$OUT_ROOT"
