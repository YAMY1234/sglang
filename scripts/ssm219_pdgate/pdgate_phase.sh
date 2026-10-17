#!/bin/bash
# docs/222 §15 PD gate client, one phase, inside the image: pdgate_phase.sh <arm> <bootstrap host>
#   waits for the arm's P and D, one PD smoke request (canary: hand-off + 32 decoded tokens), then KL(48x256) of the pair
#   against /out/ref-s32 (conc 16, cold prefix) and GSM200 (conc 64), both dispatched to P and D with one bootstrap room.
set -uo pipefail
ARM=$1; BH=$2
export PYTHONPATH=/out/tools:/work/results/bf16-regress-20261013/source-7a/python:/work/pyuser/lib/python3.12/site-packages TMPDIR=/tmp
K="/opt/sglang/bin/python -m pdgate_kl"
q() { python3 -c "import json;p=json.load(open('/out/plan.json'));print($1)"; }
PP=$(q "p['ports']['P']['http']"); DP=$(q "p['ports']['D']['http']"); BP=$(q "p['ports']['P']['bootstrap']")
S0=$(q "p['arms']['$ARM']['step0_owner']")
PAIR="--p http://127.0.0.1:$PP --d http://127.0.0.1:$DP --bootstrap-host $BH --bootstrap-port $BP"
ready() { curl -sf -m 10 http://127.0.0.1:$1/get_model_info >/dev/null; }
t0=$(date +%s)
for r in P D; do p=$([ $r = P ] && echo $PP || echo $DP)
  until ready $p; do
    [ $(( $(date +%s) - t0 )) -gt ${PDGATE_READY_S:-1500} ] && { echo "TIMEOUT $ARM-$r"; exit 3; }
    # fail fast when a server died (agg gate K1 4528168 waited 20 min on a capture crash)
    grep -q -E "Scheduler hit an exception|Received sigquit|Capture cuda graph failed" \
      /out/srv-$ARM-$r.log 2>/dev/null && { echo "DEAD $ARM-$r"; exit 4; }
    sleep 10; done
  echo "READY $ARM-$r after $(( $(date +%s) - t0 )) s"
done
$K smoke $PAIR --out /out/smoke-$ARM.json || { echo "SMOKE-FAIL $ARM"; exit 5; }
$K score $PAIR --windows /out/windows.json --arm $ARM --refs s32=/out/ref-s32 --step0 $S0 || { echo "SCORE-FAIL $ARM"; exit 6; }
$K gsm $PAIR --data /work/data/unified-gsm8k-test.jsonl --arm $ARM --out /out/gsm-$ARM.json --conc 64 || { echo "GSM-FAIL $ARM"; exit 7; }
echo "PHASE-DONE $ARM $(date -u +%T)"
