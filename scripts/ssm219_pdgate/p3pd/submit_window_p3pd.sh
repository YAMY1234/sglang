#!/usr/bin/env bash
# docs/222 §16 P3 route-0 C8 formal window (copy of integration submit_window_ssm219pd.sh; own root / job prefix). Integration submits.
# Usage (CMH login): submit_window_ssm219pd.sh <FP> <C> "<variants in placement order>" [time]   e.g. N2 "S C+" / "C+ S"
# EXCLUDE (default nvl72d137-T07, lead 10-16 00:5x): that node reports T 274.13 GB (all other 338/342 logs 274.69) and its
# X term is 2-3 pages larger, so the S D-side X identity +-1 page gate would fail there; submission parameter only.
set -euo pipefail
FP=$1; C=$2; VARIANTS=$3; TIME=${4:-01:20:00}
EXCLUDE=${EXCLUDE-nvl72d137-T07}
W=/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
R=$W/results/p3pd-20261017
B=$R/inputs/pkg-$FP
read -ra V <<< "$VARIANTS"; N=${#V[@]}
export TRIO_VARIANTS="$VARIANTS"
NAME="p3pd-C$C-$(echo "$VARIANTS" | tr -d ' +' )"
job=$(sbatch --parsable -A coreai_comparch_inferencex --job-name="$NAME" --partition=batch --qos=short \
  -N"$N" --segment="$N" --ntasks="$N" --ntasks-per-node=1 --cpus-per-task=140 --mem=900G --gres=gpu:4 \
  --time="$TIME" ${EXCLUDE:+--exclude=$EXCLUDE} --no-requeue --output="$R/logs/cmh-win-C$C-%j.out" \
  --export=ALL,PARITY_BUNDLE=$B,PARITY_FORK_SHA=7a841e4189445d108085423c3d6054bd9a7b486c,FINAL_FORK=/work/results/remeasure-v6-fivearm-tp1-20261004/inputs/source-42bbc39e-capsule-f28999a8,PDREF_MODE=grid,RECON_MODE=PD,V3_DEPLOYMENT=PD,V3_MODEL=bf16,V3_SERVING_GPUS=4,PSIDE_C=$C,REMEASURE_REPEAT=1,PARITY_STOCK_DTYPE=bfloat16,B35_PAIR_STAGE=0,PFACTOR3_PAIR=0,REMEASURE_COHORT=ladder,SGLANG_FLASHNEXT_ALLOFF_PREFILL_GRAPH=1,SGLANG_GDN_DEFER_SHALLOW_FACTOR_PLAN=1,REMEASURE_GPU_AUTHORITY=V6_OVERLAP_ON,V6_HARNESS_DRY_RUN=0,V6_GPU_ADMISSION=PASS \
  "$B/support/trio.sh")
echo "{\"job\": \"$job\", \"C\": $C, \"variants\": \"$VARIANTS\", \"fp\": \"$FP\", \"time\": \"$TIME\", \"exclude\": \"$EXCLUDE\", \"epoch\": $(date +%s)}" >> $R/submissions.jsonl
echo $job
