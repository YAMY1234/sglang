#!/bin/bash
# p4/micro.sh <logical GPU>; strict gate with the same binary must already pass.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export CUDA_VISIBLE_DEVICES=${1:?usage: p4/micro.sh GPU}
export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/p4-torch-extensions}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/p4-triton}
O=${P4_OUT:-$ROOT/p4/results}; mkdir -p "$O"
python "$ROOT/p4/micro.py" --gate "$O/gate.json" --out "$O/micro.json" --model "${P4_MODEL:-/model}" 2>&1 | tee "$O/micro.log"
