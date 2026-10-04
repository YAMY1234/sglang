#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/r12_qwen_e2e_lib.sh"

: "${R12_ROLE:?prefill|decode}"
: "${R12_VARIANT:?PP|TEP}"
: "${R12_CHUNK:?}"
: "${R12_SERVICE:?}"
: "${R12_HTTP_PORT:?}"
: "${R12_NCCL_PORT:?}"
: "${R12_BOOTSTRAP_PORT:?}"
: "${R12_FROZEN_PROOF_DIR:?}"
: "${R12_ACTUAL_PROOF_DIR:?}"

mkdir -p "/runtime/cache/$R12_SERVICE" "$R12_ACTUAL_PROOF_DIR"
r12_qwen_build_role_command "$R12_ROLE" "$R12_VARIANT" "$R12_CHUNK" \
  "$R12_HTTP_PORT" "$R12_NCCL_PORT" "$R12_BOOTSTRAP_PORT" "$R12_SERVICE"
label=PREFILL_LAUNCH
[[ "$R12_ROLE" != decode ]] || label=DECODE_LAUNCH
actual=$R12_ACTUAL_PROOF_DIR/$R12_SERVICE-$R12_ROLE-rank0.out
frozen=$R12_FROZEN_PROOF_DIR/$R12_SERVICE-$R12_ROLE-rank0.out
r12_qwen_emit_command "$label" "$actual"
cmp "$frozen" "$actual" || {
  echo "SETUP_INVALID reason=QWEN_LAUNCH_PROOF_MISMATCH service=$R12_SERVICE role=$R12_ROLE"
  exit 1
}
echo "LAUNCH_RENDER_MATCH service=$R12_SERVICE role=$R12_ROLE rank=0 sha256=$(sha256sum "$actual" | awk '{print $1}') source=$R12_QWEN_SOURCE_SHA"
exec "${R12_COMMAND[@]}"
