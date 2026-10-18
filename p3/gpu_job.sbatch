#!/bin/bash
#SBATCH --job-name=p3-p5-route
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=48
#SBATCH --mem=600G
#SBATCH --time=00:30:00
# Set P3_TREE to the git-archived host tree. Same body also runs as a 2-slot
# holder job if the cluster rejects a two-GPU normal allocation.
set -Eeuo pipefail
: "${P3_TREE:?P3_TREE}"
W=/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
A=$W/results/ssm-kernel-219
P3_STAGE=${P3_STAGE:-p5}
P3_MODE=${P3_MODE:-micro}
O=${P3_OUTPUT:-$A/p3-own/$P3_STAGE-$SLURM_JOB_ID}
G=${P3_GATE:-$A/pdgate/p3b-9827e05cb25}
mkdir -p "$O"
IMG=$W/containers/lmsysorg+sglang+nightly-dev-cu13-20260913-fa663e72-arm64-enroot.sqsh
MODEL=$W/results/bf16-regress-20261013/staging/model
MOUNTS=$(sed "s|@MODEL@|$MODEL|" "$G/g-cand1.mounts")
MOUNTS="$MOUNTS,$P3_TREE:/p3/src:ro,$O:/p3out"
date -u +START=%FT%TZ | tee "$O/job.txt"
echo "JOB=$SLURM_JOB_ID STAGE=$P3_STAGE MODE=$P3_MODE TREE=$P3_TREE NODE=$(hostname)" | tee -a "$O/job.txt"
cat "$P3_TREE/SHA" | tee "$O/SHA"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$O/gpus.csv"
srun -N1 -n1 --gres=gpu:2 --cpus-per-task=48 --container-image="$IMG" --container-mounts="$MOUNTS" \
    --no-container-mount-home --container-workdir=/p3out \
    --export=ALL,P3_STAGE="$P3_STAGE",P3_MODE="$P3_MODE",CUDA_VISIBLE_DEVICES=0,1 \
    bash /p3/src/p3/gpu_inner.sh 2>&1 | tee "$O/driver.log"
echo "END $(date -u +%FT%TZ)" | tee -a "$O/job.txt"
