#!/bin/bash

set -euo pipefail

OUT_DIR=${1:?usage: render_r12_qwen_e2e.sh OUT_DIR}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/r12_qwen_e2e_lib.sh"
for name in P_NODE D_NODE P_PORT D_PORT ROUTER_PORT P_NCCL_PORT D_NCCL_PORT \
  BOOTSTRAP_PORT; do
  : "${!name:?$name is required}"
done
mkdir -p "$OUT_DIR"

render_role() {
  local label=$1 role=$2 variant=$3 chunk=$4 http=$5 nccl=$6 boot=$7 service=$8
  r12_qwen_build_role_command "$role" "$variant" "$chunk" "$http" "$nccl" "$boot" "$service"
  r12_qwen_emit_command "$label" "$OUT_DIR/$service-$role-rank0.out"
}

render_router() {
  local service=$1
  r12_qwen_build_router_command "$P_NODE" "$P_PORT" "$D_NODE" "$D_PORT" "$ROUTER_PORT"
  r12_qwen_emit_command ROUTER_LAUNCH "$OUT_DIR/$service-router.out"
}

services=(B-main A-main B-repeat)
variants=(TEP PP TEP)
for i in "${!services[@]}"; do
  service=${services[$i]}
  render_role PREFILL_LAUNCH prefill "${variants[$i]}" 8192 "$P_PORT" "$P_NCCL_PORT" \
    "$BOOTSTRAP_PORT" "$service"
  render_role DECODE_LAUNCH decode TEP 32768 "$D_PORT" "$D_NCCL_PORT" \
    "$BOOTSTRAP_PORT" "$service"
  render_router "$service"
done

python3 - "$OUT_DIR" "$BOOTSTRAP_PORT" <<'PY'
import pathlib, shlex, sys
root = pathlib.Path(sys.argv[1]); bootstrap = sys.argv[2]
def tokens(name):
    _, raw = (root / name).read_text().strip().split(" ", 1)
    return shlex.split(raw)
def value(row, flag):
    assert row.count(flag) == 1, (flag, row)
    return row[row.index(flag) + 1]
services = ("B-main", "A-main", "B-repeat")
role_files = sorted(root.glob("*-prefill-rank0.out")) + sorted(root.glob("*-decode-rank0.out"))
assert len(role_files) == 6, role_files
for path in role_files:
    row = tokens(path.name)
    assert row.count("SGLANG_DISAGG_STAGING_BUFFER=1") == 1, path
    assert row.count("SGLANG_DISAGG_STAGING_BUFFER=0") == 0, path
    assert value(row, "--mem-fraction-static") == "0.85", path
    assert value(row, "--disaggregation-bootstrap-port") == bootstrap, path
for service in services:
    decode = tokens(f"{service}-decode-rank0.out")
    assert value(decode, "--tp-size") == "4" and value(decode, "--ep-size") == "4"
for service in ("A-main",):
    row = tokens(f"{service}-prefill-rank0.out")
    assert value(row, "--tp-size") == "1" and value(row, "--pp-size") == "4"
for service in ("B-main", "B-repeat"):
    row = tokens(f"{service}-prefill-rank0.out")
    assert value(row, "--tp-size") == "4" and "--pp-size" not in row
print("QWEN_RENDER_ASSERTIONS_PASS roles=6 staging=1 bootstrap_ports=1 TEP4=1 PP4=1 lifecycle=whole_group_restart")
PY
find "$OUT_DIR" -maxdepth 1 -type f -name '*.out' -print0 | LC_ALL=C sort -z \
  | xargs -0 sha256sum >"$OUT_DIR/rendered-files.sha256"
echo "QWEN_RENDER_PASS sha256=$(sha256sum "$OUT_DIR/rendered-files.sha256" | awk '{print $1}')"
