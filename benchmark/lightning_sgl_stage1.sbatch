#!/bin/bash
#SBATCH -A coreai_comparch_inferencex
#SBATCH -p batch_long
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH -t 00:20:00
#SBATCH -J lightning-sgl-stock
#SBATCH -o /scratch/fsw/portfolios/coreai/users/yangminl/twinstar-20260911/results/lightning-sgl-20260929/stock-%j.out
set -euo pipefail
W=/scratch/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
R=$W/results/lightning-sgl-20260929
I=$W/containers/lmsysorg+sglang+nightly-dev-cu13-20260913-fa663e72-arm64-enroot.sqsh
M=/lustre/fsw/portfolios/coreai/projects/coreai_tensorrt_ci/llm-models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
printf 'job=%s started=%s\n' "$SLURM_JOB_ID" "$(date --iso-8601=seconds)"
mkdir -p "$R/cache" "$R/tmp" "$R/stock-$SLURM_JOB_ID"
srun --ntasks=1 --container-image="$I" --container-mounts="$R:/task,$M:/model:ro,$W/hf-cache:/hf-cache,$R/cache:/root/.cache,$R/tmp:/tmp" --no-container-mount-home --container-workdir=/task bash -c '
set -euo pipefail
unset PYTHONPATH SGLANG_EXTERNAL_MODEL_PACKAGE
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HOME=/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/task/cache TMPDIR=/task/tmp TRITON_CACHE_DIR=/task/cache/triton TORCHINDUCTOR_CACHE_DIR=/task/cache/inductor TORCH_EXTENSIONS_DIR=/task/cache/torch_extensions CUDA_CACHE_PATH=/task/cache/cuda
export SGLANG_CACHE_DIR=/task/cache/sglang FLASHINFER_WORKSPACE_BASE=/task/cache/flashinfer
export CUDA_VISIBLE_DEVICES=0
python -u /task/lightning_sgl_stage1.py --model /model --out "/task/stock-$SLURM_JOB_ID"
'
