#!/bin/bash
# Holder inbox after capture: --gpus 1 --minutes 5 (no timing, no exclusivity).
set -euo pipefail
W=/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
source "$W/results/ssm-kernel-219/holder/scripts/hlib.sh"
SOURCE=$(sed -n 's/^# HOLDER_SOURCE=//p' "${BASH_SOURCE[0]}" | head -1)
TREE=${P4_TREE:-${SOURCE%/p4/hjob_service_gate.sh}}
C_TREE=/work/${TREE#"$W/"}
DUMP=$(find "$W/results/ssm-kernel-231" -maxdepth 1 -type d -name 'service-dump-p4cap-*' | sort | tail -1)
[[ -n $DUMP ]]
C_DUMP=/work/${DUMP#"$W/"}
h_srun "$HOLDER_CVD" "set -euo pipefail
cd '$C_TREE'
export PYTHONPATH='$C_TREE/python:/work/pyuser/lib/python3.12/site-packages'
export TORCH_EXTENSIONS_DIR=/tmp/p4-torch-extensions TRITON_CACHE_DIR=/tmp/p4-triton MAX_JOBS=8
python p4/gate.py --service-only --service-dir '$C_DUMP' --baseline-tree /work/src/ssm219-int-d053bd5e346 --out '$C_DUMP/final-service-gate.json' --model /model > '$C_DUMP/final-service-gate.log' 2>&1
cat '$C_DUMP/final-service-gate.log'
"
