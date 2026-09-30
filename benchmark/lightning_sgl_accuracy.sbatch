#!/bin/bash
#SBATCH -A coreai_comparch_inferencex
#SBATCH -p batch_long
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH -t 04:00:00
#SBATCH -J lightning-sgl-gsm200
#SBATCH -o /scratch/fsw/portfolios/coreai/users/yangminl/twinstar-20260911/results/lightning-sgl-20260929/phase2-%j.out
set -euo pipefail
W=/scratch/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
R=$W/results/lightning-sgl-20260929
I=$W/containers/lmsysorg+sglang+nightly-dev-cu13-20260913-fa663e72-arm64-enroot.sqsh
M=/lustre/fsw/portfolios/coreai/projects/coreai_tensorrt_ci/llm-models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
P=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/minma/twinstar-pd-models
G=${1:?pass the complete numerical guard result.json path}
LIGHTNING_GUARD_REL=$(python3 - "$R" "$G" <<'PY_PATH'
from pathlib import Path
import sys
root, guard = (Path(arg).resolve(strict=True) for arg in sys.argv[1:])
relative = guard.relative_to(root)
if len(relative.parts) != 2 or not relative.parts[0].startswith('guard-') or relative.name != 'result.json':
    raise SystemExit('guard result must be a guard cell result under this line results directory')
print(relative.as_posix())
PY_PATH
)
export LIGHTNING_GUARD_REL
printf 'job=%s started=%s\n' "$SLURM_JOB_ID" "$(date --iso-8601=seconds)"
mkdir -p "$R/cache" "$R/tmp" "$R/gsm200-$SLURM_JOB_ID"
srun --ntasks=1 --container-image="$I" --container-mounts="$R:/task,$M:/model:ro,$P:/reference:ro,$W/hf-cache:/hf-cache,$R/cache:/root/.cache,$R/tmp:/tmp" --no-container-mount-home --container-workdir=/task bash -c '
set -euo pipefail
export PYTHONPATH=/task/plugin PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HOME=/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/task/cache TMPDIR=/task/tmp TRITON_CACHE_DIR=/task/cache/triton TORCHINDUCTOR_CACHE_DIR=/task/cache/inductor TORCH_EXTENSIONS_DIR=/task/cache/torch_extensions CUDA_CACHE_PATH=/task/cache/cuda
export SGLANG_CACHE_DIR=/task/cache/sglang FLASHINFER_WORKSPACE_BASE=/task/cache/flashinfer
python -u /task/lightning_sgl_accuracy.py --model /model --duet /hf-cache/lightning-duet-93530e4621bb3a2d480407a357d6416b7b1c6701 --guard-result "/task/$LIGHTNING_GUARD_REL" --out "/task/gsm200-$SLURM_JOB_ID"
'
