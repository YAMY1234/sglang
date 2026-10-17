#!/bin/bash
# p4/run_gate.sh <logical GPU>; run inside the qualified service container.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export CUDA_VISIBLE_DEVICES=${1:?usage: p4/run_gate.sh GPU}
export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/p4-torch-extensions}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/p4-triton}
export MAX_JOBS=${MAX_JOBS:-8}
export SGLANG_GDN_EXPIRY_SPLIT_KERNEL=1 SGLANG_GDN_EXPIRY_SPLIT_MIN_BATCH=16
O=${P4_OUT:-$ROOT/p4/results}; mkdir -p "$O"
python -c 'from sglang.srt.layers.attention.linear.kernels.gdn_factored_cuda import load_extension; assert load_extension() is not None' > "$O/build.log" 2>&1
cuobjdump --dump-resource-usage "$TORCH_EXTENSIONS_DIR"/gdn_factored_step_sm103a_*/*.so > "$O/resources.txt"
python "$ROOT/p4/resources.py" "$O/build.log" "$O/resources.txt"
EXTRA=()
[[ -z ${P4_SERVICE_DIR:-} ]] || EXTRA+=(--service-dir "$P4_SERVICE_DIR")
[[ -z ${P4_BASELINE_TREE:-} ]] || EXTRA+=(--baseline-tree "$P4_BASELINE_TREE")
python "$ROOT/p4/gate.py" --out "$O/gate.json" --model "${P4_MODEL:-/model}" "${EXTRA[@]}" 2>&1 | tee "$O/gate.log"
