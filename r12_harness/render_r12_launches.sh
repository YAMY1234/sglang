#!/bin/bash

set -euo pipefail

OUT_DIR=${1:?usage: render_r12_launches.sh OUT_DIR}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/r12_launch_lib.sh"

: "${R12_JOB:?R12_JOB=job7b|job7c}"
: "${R12_PP_CHUNK:?R12_PP_CHUNK=8192|16384}"
for name in P0 P1 D0 D1 P_PORT D_PORT ROUTER_PORT P_DIST_PORT D_DIST_PORT \
  P_NCCL_PORT D_NCCL_PORT P_BOOT_1 P_BOOT_2 P_BOOT_3 P_BOOT_4 P_BOOT_5 \
  D_BOOT_1 D_BOOT_2; do
  : "${!name:?$name is required}"
done

case "$R12_JOB:$R12_PP_CHUNK" in
  job7b:8192|job7c:16384) ;;
  *) echo "DRY_RUN_INVALID_JOB_CHUNK job=$R12_JOB pp_chunk=$R12_PP_CHUNK" >&2; exit 2 ;;
esac
R12_MEM_FRACTION=${R12_MEM_FRACTION:-0.90}
case "$R12_MEM_FRACTION" in 0.90|0.85) ;; *) echo "DRY_RUN_INVALID_MEM_FRACTION value=$R12_MEM_FRACTION" >&2; exit 2;; esac

ports=("$P_PORT" "$D_PORT" "$ROUTER_PORT" "$P_DIST_PORT" "$D_DIST_PORT"
  "$P_NCCL_PORT" "$D_NCCL_PORT" "$P_BOOT_1" "$P_BOOT_2" "$P_BOOT_3"
  "$P_BOOT_4" "$P_BOOT_5" "$D_BOOT_1" "$D_BOOT_2")
for port in "${ports[@]}"; do
  [[ "$port" =~ ^[0-9]+$ ]] && ((port >= 1 && port <= 65535)) || {
    echo "DRY_RUN_INVALID_PORT value=$port" >&2
    exit 2
  }
done
mapfile -t unique_ports < <(printf '%s\n' "${ports[@]}" | LC_ALL=C sort -u)
[[ "${#unique_ports[@]}" -eq "${#ports[@]}" ]] || {
  echo "DRY_RUN_DUPLICATE_PORTS values=${ports[*]}" >&2
  exit 2
}

mkdir -p "$OUT_DIR"
printf 'service\tvariant\tchunk\tp_boot\tdecode_path\n' >"$OUT_DIR/service-plan.tsv"

render_role() {
  local label=$1 role=$2 variant=$3 chunk=$4 master=$5 dist=$6 http=$7 nccl=$8 boot=$9 service=${10}
  local rank
  for rank in 0 1; do
    r12_build_role_command "$role" "$variant" "$chunk" "$rank" "$master" "$dist" \
      "$http" "$nccl" "$boot" "$service" "$R12_MEM_FRACTION"
    r12_emit_command "$label" "$OUT_DIR/$service-$role-rank$rank.out"
  done
}

render_router() {
  local service=$1
  r12_build_router_command "$P0" "$P_PORT" "$D0" "$D_PORT" "$ROUTER_PORT"
  r12_emit_command ROUTER_LAUNCH "$OUT_DIR/$service-router.out"
}

render_role DECODE_LAUNCH decode TEP 8192 "$D0" "$D_DIST_PORT" "$D_PORT" \
  "$D_NCCL_PORT" "$D_BOOT_1" decode-normal
render_role DECODE_LAUNCH decode TEP 8192 "$D0" "$D_DIST_PORT" "$D_PORT" \
  "$D_NCCL_PORT" "$D_BOOT_2" decode-fallback

services=(B-main A-main B-repeat A-repeat A-main-fallback)
variants=(TEP PP TEP PP PP)
chunks=(8192 "$R12_PP_CHUNK" 8192 "$R12_PP_CHUNK" "$R12_PP_CHUNK")
boots=("$P_BOOT_1" "$P_BOOT_2" "$P_BOOT_3" "$P_BOOT_4" "$P_BOOT_5")
decode_paths=(normal normal normal normal fallback)
for i in "${!services[@]}"; do
  service=${services[$i]}
  render_role PREFILL_LAUNCH prefill "${variants[$i]}" "${chunks[$i]}" "$P0" \
    "$P_DIST_PORT" "$P_PORT" "$P_NCCL_PORT" "${boots[$i]}" "$service"
  render_router "$service"
  printf '%s\t%s\t%s\t%s\t%s\n' "$service" "${variants[$i]}" "${chunks[$i]}" \
    "${boots[$i]}" "${decode_paths[$i]}" >>"$OUT_DIR/service-plan.tsv"
done

python3 - "$OUT_DIR" "$R12_JOB" "$R12_PP_CHUNK" "$R12_MEM_FRACTION" <<'PY'
import pathlib
import shlex
import sys

root = pathlib.Path(sys.argv[1])
job, pp_chunk, mem_fraction = sys.argv[2], sys.argv[3], sys.argv[4]

def command(path):
    label, raw = path.read_text().strip().split(" ", 1)
    return label, shlex.split(raw)

for path in sorted(root.glob("*.out")):
    label, tokens = command(path)
    assert tokens and "" not in tokens, path
    assert "--max-total-tokens" not in tokens, path
    if label in {"PREFILL_LAUNCH", "DECODE_LAUNCH"}:
        assert tokens.count("SGLANG_DISAGG_STAGING_BUFFER=1") == 1, path
        assert tokens.count("SGLANG_DISAGG_STAGING_BUFFER=0") == 0, path
        assert tokens[tokens.index("--mem-fraction-static") + 1] == mem_fraction, path
        assert tokens[tokens.index("--nnodes") + 1] == "2", path
        assert tokens.count("SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S=1200") == 1, path
        if label == "DECODE_LAUNCH":
            assert tokens[tokens.index("--tp-size") + 1] == "8", path
            assert tokens[tokens.index("--ep-size") + 1] == "8", path
            assert tokens[tokens.index("--chunked-prefill-size") + 1] == "8192", path
            assert "--pp-size" not in tokens and "--enable-dp-attention" not in tokens, path
        elif "A-" in path.name:
            assert tokens[tokens.index("--tp-size") + 1] == "2", path
            assert tokens[tokens.index("--ep-size") + 1] == "2", path
            assert tokens[tokens.index("--pp-size") + 1] == "4", path
            assert "--disable-overlap-schedule" in tokens, path
            assert tokens.count("SGLANG_PP_COMM_OVERLAP=1") == 1, path
            assert tokens.count("SGLANG_PP_LAYER_PARTITION=24,23,23,23") == 1, path
            assert tokens[tokens.index("--chunked-prefill-size") + 1] == pp_chunk, path
        else:
            assert tokens[tokens.index("--tp-size") + 1] == "8", path
            assert tokens[tokens.index("--ep-size") + 1] == "8", path
            assert "--pp-size" not in tokens and "--enable-dp-attention" not in tokens, path
            assert "--disable-overlap-schedule" not in tokens, path
            assert not any(x.startswith("SGLANG_PP_") for x in tokens), path
            assert tokens[tokens.index("--chunked-prefill-size") + 1] == "8192", path
print(f"DRY_RUN_STATIC_ASSERTIONS_PASS job={job} pp_chunk={pp_chunk} services=5 decode_paths=2")
PY

find "$OUT_DIR" -maxdepth 1 -type f -name '*.out' -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum >"$OUT_DIR/rendered-files.sha256"
sha256sum "$OUT_DIR/rendered-files.sha256" | awk '{print $1}' >"$OUT_DIR/rendered-launches.sha256"
echo "DRY_RUN_RENDER_SHA256 $(<"$OUT_DIR/rendered-launches.sha256")"
echo "DRY_RUN_PORT_GATE_PASS count=${#ports[@]}"
