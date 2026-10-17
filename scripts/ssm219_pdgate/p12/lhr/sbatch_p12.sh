#!/bin/bash
# docs/222 §20 LHR: plain sbatch for a P12 pd1n job (no holder): bash sbatch_p12.sh <kind> <package fp> <short|normal> [--dry]
#   runs <job dir>/p12_hjob.sh <kind> <fp> on one exclusive 4-GPU node (HOLDER_JOB unset -> srun --jobid=$SLURM_JOB_ID).
#   Twin rule (lead 08:47): submit short and normal, keep the first that starts, scancel the other.
set -euo pipefail
KIND=$1; FP=$2; QOS=$3
W=/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
D=$W/results/ssm-kernel-219/ws2/p12/pkg-$FP${P12_TAG:+-$P12_TAG}; R=$W/results/p3pd-20261017   # P12_TAG: job dir made by prep_p12.sh <fp> <tag>
case $KIND in positive) M=45;; negative) M=75;; micro) M=100;; ab) M=85;; *) echo "bad kind $KIND"; exit 2;; esac
[[ $QOS == short || $QOS == normal ]] || { echo "qos short|normal"; exit 2; }
CMD=(sbatch --parsable -A coreai_comparch_inferencex -p batch --qos=$QOS -N1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=140
     --mem=0 --exclusive --time=$M --no-requeue --export=ALL,P12_JOBDIR=$D --job-name=ws2-p12-$KIND --output=$R/logs/p12-$KIND-%j.out
     $D/p12_hjob.sh $KIND $FP)
python3 "$(dirname "$(readlink -f "$0")")/depcheck.py" "$(dirname "$(readlink -f "$0")")/items.txt" $R/inputs/pkg-$FP || exit 3  # docs/222 §20.3
if [[ ${4:-} == --dry ]]; then echo "${CMD[@]}"; else "${CMD[@]}"; fi
