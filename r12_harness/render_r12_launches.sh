#!/bin/bash

set -euo pipefail

OUT_DIR=${1:?usage: render_r12_launches.sh OUT_DIR}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/r12_launch_lib.sh"

: "${R12_JOB:?R12_JOB=job7b|job7c}"
: "${R12_PP_CHUNK:?R12_PP_CHUNK=8192|16384}"
for name in P0 P1 D0 D1 P_PORT D_PORT ROUTER_PORT P_DIST_PORT D_DIST_PORT \
  P_NCCL_PORT D_NCCL_PORT BOOTSTRAP_PORT; do
  : "${!name:?$name is required}"
done

case "$R12_JOB:$R12_PP_CHUNK" in
  job7b:8192|job7c:16384) ;;
  *) echo "DRY_RUN_INVALID_JOB_CHUNK job=$R12_JOB pp_chunk=$R12_PP_CHUNK" >&2; exit 2 ;;
esac
R12_MEM_FRACTION=${R12_MEM_FRACTION:-0.90}
case "$R12_MEM_FRACTION" in 0.90|0.85) ;; *) echo "DRY_RUN_INVALID_MEM_FRACTION value=$R12_MEM_FRACTION" >&2; exit 2 ;; esac

ports=("$P_PORT" "$D_PORT" "$ROUTER_PORT" "$P_DIST_PORT" "$D_DIST_PORT"
  "$P_NCCL_PORT" "$D_NCCL_PORT" "$BOOTSTRAP_PORT")
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
printf 'service\tvariant\tchunk\tbootstrap\tlifecycle\n' >"$OUT_DIR/service-plan.tsv"

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

services=(B-main A-main B-repeat A-repeat)
variants=(TEP PP TEP PP)
chunks=(8192 "$R12_PP_CHUNK" 8192 "$R12_PP_CHUNK")
for i in "${!services[@]}"; do
  service=${services[$i]}
  render_role PREFILL_LAUNCH prefill "${variants[$i]}" "${chunks[$i]}" "$P0" \
    "$P_DIST_PORT" "$P_PORT" "$P_NCCL_PORT" "$BOOTSTRAP_PORT" "$service"
  render_role DECODE_LAUNCH decode TEP 8192 "$D0" "$D_DIST_PORT" "$D_PORT" \
    "$D_NCCL_PORT" "$BOOTSTRAP_PORT" "$service"
  render_router "$service"
  printf '%s\t%s\t%s\t%s\twhole_group_restart\n' "$service" "${variants[$i]}" \
    "${chunks[$i]}" "$BOOTSTRAP_PORT" >>"$OUT_DIR/service-plan.tsv"
done

python3 - "$OUT_DIR" "$R12_JOB" "$R12_PP_CHUNK" "$R12_MEM_FRACTION" "$BOOTSTRAP_PORT" <<'PY'
import pathlib, shlex, sys
root = pathlib.Path(sys.argv[1])
job, pp_chunk, mem_fraction, bootstrap = sys.argv[2:]
def command(path):
    label, raw = path.read_text().strip().split(" ", 1)
    return label, shlex.split(raw)
def value(tokens, flag):
    assert tokens.count(flag) == 1, (flag, tokens)
    return tokens[tokens.index(flag) + 1]
services = ("B-main", "A-main", "B-repeat", "A-repeat")
for service in services:
    for role in ("prefill", "decode"):
        label, tokens = command(root / f"{service}-{role}-rank0.out")
        assert label == ("PREFILL_LAUNCH" if role == "prefill" else "DECODE_LAUNCH")
        assert "--max-total-tokens" not in tokens
        assert tokens.count("SGLANG_DISAGG_STAGING_BUFFER=0") == 1
        assert tokens.count("SGLANG_DISAGG_STAGING_BUFFER=1") == 0
        cache = f"/runtime/cache/{service}/{role}-rank-0"
        assert tokens.count(f"HF_HOME={cache}/xdg/huggingface") == 1
        assert tokens.count(f"XDG_CACHE_HOME={cache}/xdg") == 1
        assert value(tokens, "--mem-fraction-static") == mem_fraction
        assert value(tokens, "--disaggregation-bootstrap-port") == bootstrap
        assert value(tokens, "--nnodes") == "2"
        if role == "decode":
            assert value(tokens, "--tp-size") == "8"
            assert value(tokens, "--ep-size") == "8"
            assert value(tokens, "--chunked-prefill-size") == "8192"
        elif service.startswith("A-"):
            assert value(tokens, "--tp-size") == "2"
            assert value(tokens, "--ep-size") == "2"
            assert value(tokens, "--pp-size") == "4"
            assert value(tokens, "--chunked-prefill-size") == pp_chunk
            assert tokens.count("SGLANG_PP_COMM_OVERLAP=1") == 1
        else:
            assert value(tokens, "--tp-size") == "8"
            assert value(tokens, "--ep-size") == "8"
            assert "--pp-size" not in tokens
            assert value(tokens, "--chunked-prefill-size") == "8192"
print(f"DRY_RUN_STATIC_ASSERTIONS_PASS job={job} pp_chunk={pp_chunk} groups=4 bootstrap_ports=1 lifecycle=whole_group_restart")
PY

python3 "$SCRIPT_DIR/prove_r12_semantic_diff.py" --root "$OUT_DIR" \
  --pp-chunk "$R12_PP_CHUNK" --output "$OUT_DIR/semantic-proof.json"

find "$OUT_DIR" -maxdepth 1 -type f -name '*.out' -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum >"$OUT_DIR/rendered-files.sha256"
sha256sum "$OUT_DIR/rendered-files.sha256" | awk '{print $1}' >"$OUT_DIR/rendered-launches.sha256"
echo "DRY_RUN_RENDER_SHA256 $(<"$OUT_DIR/rendered-launches.sha256")"
echo "DRY_RUN_PORT_GATE_PASS count=${#ports[@]}"
