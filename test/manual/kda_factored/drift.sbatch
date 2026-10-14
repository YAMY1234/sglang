#!/bin/bash
#SBATCH -A coreai_comparch_inferencex
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -t 00:32:00
#SBATCH -J KDAF-drift
#SBATCH -o /lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911/results/kdaf-206/%j.out
set -euo pipefail
W=/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
: "${KDA_SOURCE_COMMIT:?}" "${FAMILY:?}"
export KDA_SOURCE=/work/scratch/kdaf-$KDA_SOURCE_COMMIT
export OUT=/work/results/kdaf-206/$SLURM_JOB_ID
mkdir -p "$W/results/kdaf-206/$SLURM_JOB_ID"
srun --cpu-bind=none --container-image="$W/containers/lmsysorg+sglang+nightly-dev-cu13-20260913-fa663e72-arm64-enroot.sqsh" --container-mounts="$W:/work,/tmp:/tmp" --no-container-mount-home bash -c '
set -euo pipefail
if [[ $FAMILY == glm ]]; then
 export KDA_BUNDLE=/work/results/seed-serve-20261011/inputs-glm-seed35-r6
 export KDA_RELEASE=/work/hf-cache/duet-releases/Latent-Voyager--GLM-5.3-Flash-SEED/f0876024a34df39c76130eb0c5601c342e7d5425
 export SGLANG_MAMBA_SSM_DTYPE=float32 SGLANG_MAMBA_CONV_DTYPE=float32 NVIDIA_TF32_OVERRIDE=0 SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
else
 export KDA_BUNDLE=/work/results/seed-serve-20261011/inputs-v35-mem065-r1
 export KDA_RELEASE=/work/hf-cache/duet-releases/Latent-Voyager--Kimi-Linear-48B-A3B-Instruct-SEED/373e9d62c2e56941b13a981a3b56a0bdfbee8e60
fi
export PYTHONPATH=$KDA_SOURCE/python:$KDA_SOURCE/test/manual/kda_factored:/work/results/remeasure-v6-fivearm-tp1-20261004/inputs/helper-runtime/helpers:$KDA_BUNDLE:/work/results/seed-serve-20261011/inputs-v35-mem065-r1:/work/pyuser/lib/python3.12/site-packages
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/work/hf-cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_MODULES_CACHE=/tmp/kdaf-$SLURM_JOB_ID/hf-modules OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MAX_JOBS=4 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/work/scratch/kdaf-206/triton
unset SGLANG_EXTERNAL_MODEL_PACKAGE SGLANG_DUET_DIR
mkdir -p "$HF_MODULES_CACHE"
python "$KDA_BUNDLE/runtime_env.py" "$KDA_SOURCE/test/manual/kda_factored/drift_service.py"
'
