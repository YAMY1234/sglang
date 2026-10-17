#!/usr/bin/env bash
# docs/222 §21 paper-ladder 1P:1D row on the v12 stack, LHR (copy of p3pd/submit_window_p3pd.sh: QOS argument for the
# short / normal twin, no node exclusion by default, optional canary window length).  Integration submits.
# Usage (LHR login): submit_window_v12pd.sh <FP> <C> "<variants in placement order>" <short|normal> [time] [canary seconds]
#   e.g. N3 "PC+ S C+" short 01:20:00   |   canary: <FP> 8 "PC+ S C+" short 00:50:00 300
set -euo pipefail
FP=$1; C=$2; VARIANTS=$3; QOS=$4; TIME=${5:-01:20:00}; SECS=${6:-}
[[ $QOS == short || $QOS == normal ]] || { echo "qos short|normal"; exit 2; }
[[ -z $SECS || $SECS == 300 ]] || { echo "canary seconds: 300 only"; exit 2; }
EXCLUDE=${EXCLUDE-}
W=/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
R=$W/results/p3pd-20261017
B=$R/inputs/pkg-$FP
[[ -f $R/CPU-ready-$FP.json ]] || { echo "no CPU gate receipt for $FP on this cluster"; exit 2; }
T=$(dirname "$(readlink -f "$0")")/..
python3 $T/p12/lhr/depcheck.py $T/p12/lhr/items.txt $B || exit 3  # docs/222 §20.3: every dependency exists on this cluster
read -ra V <<< "$VARIANTS"; N=${#V[@]}
export TRIO_VARIANTS="$VARIANTS"
NAME="v12pd-C$C-$(echo "$VARIANTS" | tr -d ' +' )${SECS:+-canary}"
job=$(sbatch --parsable -A coreai_comparch_inferencex --job-name="$NAME" --partition=batch --qos=$QOS \
  -N"$N" --segment="$N" --ntasks="$N" --ntasks-per-node=1 --cpus-per-task=140 --mem=900G --gres=gpu:4 \
  --time="$TIME" ${EXCLUDE:+--exclude=$EXCLUDE} --no-requeue --output="$R/logs/v12pd-win-C$C-%j.out" \
  --export=ALL,PARITY_BUNDLE=$B,PARITY_FORK_SHA=7a841e4189445d108085423c3d6054bd9a7b486c,FINAL_FORK=/work/results/remeasure-v6-fivearm-tp1-20261004/inputs/source-42bbc39e-capsule-f28999a8,PDREF_MODE=grid,RECON_MODE=PD,V3_DEPLOYMENT=PD,V3_MODEL=bf16,V3_SERVING_GPUS=4,PSIDE_C=$C,REMEASURE_REPEAT=1,PARITY_STOCK_DTYPE=bfloat16,B35_PAIR_STAGE=0,PFACTOR3_PAIR=0,REMEASURE_COHORT=ladder,SGLANG_FLASHNEXT_ALLOFF_PREFILL_GRAPH=1,SGLANG_GDN_DEFER_SHALLOW_FACTOR_PLAN=1,REMEASURE_GPU_AUTHORITY=V6_OVERLAP_ON,V6_HARNESS_DRY_RUN=0,V6_GPU_ADMISSION=PASS${SECS:+,LADDER_WINDOW_SECONDS=$SECS} \
  "$B/support/trio.sh")
echo "{\"job\": \"$job\", \"cluster\": \"LHR\", \"C\": $C, \"variants\": \"$VARIANTS\", \"fp\": \"$FP\", \"qos\": \"$QOS\", \"time\": \"$TIME\", \"canary_seconds\": \"${SECS:-}\", \"exclude\": \"$EXCLUDE\", \"epoch\": $(date +%s)}" >> $R/submissions-v12pd.jsonl
echo $job
