#!/bin/bash
#SBATCH -A coreai_comparch_inferencex
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -t 00:17:00
#SBATCH -J KDAF-delayed-gate
#SBATCH -o /lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911/results/kdaf-206/%j.out
set -euo pipefail
W=/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
mkdir -p "$W/results/kdaf-206/$SLURM_JOB_ID"
export OUT=/work/results/kdaf-206/$SLURM_JOB_ID
export KDA_SOURCE=/work/scratch/kdaf-a4d88c1c0f3
srun --cpu-bind=none --container-image="$W/containers/lmsysorg+sglang+nightly-dev-cu13-20260913-fa663e72-arm64-enroot.sqsh" --container-mounts="$W:/work" --no-container-mount-home bash -c '
set -euo pipefail
export PYTHONPATH=$KDA_SOURCE/python:$KDA_SOURCE/test/manual/kda_factored:/work/pyuser/lib/python3.12/site-packages
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TRITON_CACHE_DIR=/work/scratch/kdaf-206/triton
export CUDA_VISIBLE_DEVICES=0
python "$KDA_SOURCE/test/manual/kda_factored/immutable_gate.py" | tee "$OUT/immutable-gate.out"
python "$KDA_SOURCE/test/manual/kda_factored/pool_gate.py" | tee "$OUT/fp32-regression.out"
'
