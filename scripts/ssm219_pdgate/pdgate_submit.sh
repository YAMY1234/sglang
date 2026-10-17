#!/bin/bash
# docs/222 §15 PD gate: queue a prepared gate ($OUT from pdgate_prep.py) on an ssm219 holder.
# Integration line only (WS-2 does not submit GPU work).
#   pdgate_submit.sh <OUT> [minutes, default 45] [--node NODE | --next-q _next_X]
# Without a target qsubmit.sh uses queue/CURRENT; with several holders alive name one (--node <its node>) or a holder that
# has not started yet (--next-q _next_X, adopted at its start).  Minutes: two phases 45, one phase (--before-from) 30,
# three phases (cand2) 60; + 2 per phase with --hit-pass.
set -euo pipefail
OUT=$(readlink -f ${1:?OUT}); shift
MIN=45; [[ ${1:-} =~ ^[0-9]+$ ]] && { MIN=$1; shift; }
H=/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911/results/ssm-kernel-219/holder
[[ -f $OUT/plan.json && -f $OUT/tools/pdgate_hjob.sh ]] || { echo "not prepared: $OUT"; exit 2; }
J=$OUT/pdgate-$(basename $OUT).sh
{ echo "# docs/222 §15 PD gate $(basename $OUT): $(python3 -c "import json;p=json.load(open('$OUT/plan.json'));print(' '.join(a+'='+p['arms'][a]['cls'] for a in p['phases']))")"
  echo "OUT=$OUT"; echo "source $OUT/tools/pdgate_hjob.sh"; } > $J
bash $H/scripts/qsubmit.sh $J --gpus 4 --excl --stage --minutes $MIN "$@"
