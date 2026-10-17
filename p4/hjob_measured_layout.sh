#!/bin/bash
# Holder inbox after service capture: --gpus 1 --excl --minutes 15.
set -euo pipefail
W=/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
source "$W/results/ssm-kernel-219/holder/scripts/hlib.sh"
SOURCE=$(sed -n 's/^# HOLDER_SOURCE=//p' "${BASH_SOURCE[0]}" | head -1)
TREE=${P4_TREE:-${SOURCE%/p4/hjob_measured_layout.sh}}
C_TREE=/work/${TREE#"$W/"}
DUMP=${P4_SERVICE_DIR:-}
if [[ -z $DUMP ]]; then
    LAYOUT=$(find "$W/results/ssm-kernel-231" -maxdepth 2 -path '*/service-dump-p4cap-*/z-layouts.json' | sort | tail -1)
    DUMP=${LAYOUT%/z-layouts.json}
fi
if [[ ! -f $DUMP/z-layouts.json ]]; then
    echo 'BLOCKED measured z-layouts.json missing; no timing'
    exit 1
fi
C_DUMP=/work/${DUMP#"$W/"}
O=/work/results/ssm-kernel-231/measured-$(basename "$RUN_DIR")
mkdir -p "${O/#\/work/$W}"
h_srun "$HOLDER_CVD" "set -euo pipefail
export PYTHONPATH=/work/pyuser/lib/python3.12/site-packages
export P4_OUT='$O' P4_Z_LAYOUTS='$C_DUMP/z-layouts.json' P4_SERVICE_DIR='$C_DUMP'
export P4_BASELINE_TREE=/work/src/ssm219-int-d053bd5e346
cat '$C_DUMP/z-layouts.json'
nvidia-smi -q > '$O/nvidia-before.txt'
bash '$C_TREE/p4/run_gate.sh' '$HOLDER_CVD'
bash '$C_TREE/p4/micro.sh' '$HOLDER_CVD'
nvidia-smi -q > '$O/nvidia-after.txt'
"
