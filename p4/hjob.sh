#!/bin/bash
# Holder inbox: --gpus 1 --excl --minutes 15 (model mount, no full model load).
set -euo pipefail
W=/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
source "$W/results/ssm-kernel-219/holder/scripts/hlib.sh"
SOURCE=$(sed -n 's/^# HOLDER_SOURCE=//p' "${BASH_SOURCE[0]}" | head -1)
TREE=${P4_TREE:-${SOURCE%/p4/hjob.sh}}
[[ -n $TREE && -f $TREE/SHA ]]
# qsubmit canonicalizes HOLDER_SOURCE to /scratch, while W is a /lustre alias.
TREE_REL=$(realpath --relative-to="$W" "$TREE")
[[ $TREE_REL != ..* ]]
C_TREE=/work/$TREE_REL
O=/work/results/ssm-kernel-231/$(basename "$RUN_DIR")
mkdir -p "${O/#\/work/$W}"
h_srun "$HOLDER_CVD" "set -euo pipefail
export PYTHONPATH=/work/pyuser/lib/python3.12/site-packages
export P4_OUT='$O'
nvidia-smi -q > '$O/nvidia-before.txt'
bash '$C_TREE/p4/run_gate.sh' '$HOLDER_CVD'
bash '$C_TREE/p4/micro.sh' '$HOLDER_CVD'
nvidia-smi -q > '$O/nvidia-after.txt'
"
