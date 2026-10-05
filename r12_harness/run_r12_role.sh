#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/r12_launch_lib.sh"

: "${R12_ROLE:?prefill|decode}"
: "${R12_VARIANT:?PP|TEP}"
: "${R12_CHUNK:?}"
: "${R12_SERVICE:?}"
: "${R12_MASTER:?}"
: "${R12_DIST_PORT:?}"
: "${R12_HTTP_PORT:?}"
: "${R12_NCCL_PORT:?}"
: "${R12_BOOTSTRAP_PORT:?}"
: "${R12_FROZEN_PROOF_DIR:?}"
: "${R12_ACTUAL_PROOF_DIR:?}"

rank=${SLURM_PROCID:?}
r12_build_role_cache_env "$R12_SERVICE" "$R12_ROLE" "$rank"
cache=$R12_ROLE_CACHE
mkdir -p "$cache"/{sglang,sglang-jit,xdg,triton,torchinductor,torch-extensions,cuda}
mkdir -p "$R12_ACTUAL_PROOF_DIR"

r12_build_role_command "$R12_ROLE" "$R12_VARIANT" "$R12_CHUNK" "$rank" \
  "$R12_MASTER" "$R12_DIST_PORT" "$R12_HTTP_PORT" "$R12_NCCL_PORT" \
  "$R12_BOOTSTRAP_PORT" "$R12_SERVICE" "${R12_MEM_FRACTION:-0.90}"

label=PREFILL_LAUNCH
[[ "$R12_ROLE" != decode ]] || label=DECODE_LAUNCH
actual=$R12_ACTUAL_PROOF_DIR/$R12_SERVICE-$R12_ROLE-rank$rank.out
frozen=$R12_FROZEN_PROOF_DIR/$R12_SERVICE-$R12_ROLE-rank$rank.out
r12_emit_command "$label" "$actual"
if ! cmp "$frozen" "$actual"; then
  echo "SETUP_INVALID reason=LAUNCH_PROOF_MISMATCH service=$R12_SERVICE role=$R12_ROLE rank=$rank frozen=$frozen actual=$actual"
  exit 1
fi
echo "LAUNCH_RENDER_MATCH service=$R12_SERVICE role=$R12_ROLE rank=$rank sha256=$(sha256sum "$actual" | awk '{print $1}') source=$R12_EXPECTED_SOURCE_SHA"
echo "ROLE_START $(date -u +%FT%TZ) service=$R12_SERVICE role=$R12_ROLE variant=$R12_VARIANT chunk=$R12_CHUNK rank=$rank host=${SLURMD_NODENAME:-unknown} source=$R12_EXPECTED_SOURCE_SHA"
exec "${R12_COMMAND[@]}"
