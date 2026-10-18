#!/usr/bin/env bash
# Single P worker group, TP2. The holder supplies a frozen workload factory.
# Factory contract and route semantics are documented in route_micro.py.
set -Eeuo pipefail
P3_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$P3_DIR/env.sh"
export SGLANG_GDN_K31_AB_DUAL=1
export SGLANG_GDN_K31_AB_ENV=SGLANG_GDN_K31_FUSED,SGLANG_GDN_K31_CHOLQR_MIXED
export PYTHONUNBUFFERED=1
exec python -m torch.distributed.run --standalone --nproc-per-node=2 "$P3_DIR/route_micro.py" "$@"
