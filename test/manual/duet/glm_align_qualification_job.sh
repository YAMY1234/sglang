#!/usr/bin/env bash
# Submit exactly one GPU job after the CPU gate. Allocation flags are supplied by sbatch.
set -Eeuo pipefail
: "${GLM_ALIGN_HOST:?owned immutable qualification bundle}"
: "${GLM_ALIGN_SOURCE_SHA:?source commit}"
: "${GLM_ALIGN_MODE:?cpu or gpu}"
W=/scratch/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
IMAGE=$W/containers/lmsysorg+sglang+nightly-dev-cu13-20260913-fa663e72-arm64-enroot.sqsh
OUT=$GLM_ALIGN_HOST/jobs/$SLURM_JOB_ID
mkdir -p "$OUT"
python3 - "$GLM_ALIGN_HOST" "$GLM_ALIGN_MODE" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
for name,sha in manifest['files'].items():
 assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha,name
if sys.argv[2]=='gpu':
 proof=json.loads((root/'CPU_PASS.json').read_text())
 assert proof['passed'] and not proof['cuda_initialized'] and proof['source_sha']==manifest['source_sha']
PY
MOUNTS="$W:/work:ro,$GLM_ALIGN_HOST:/glm-align,$GLM_ALIGN_HOST/source:/glm-align/source:ro,/tmp:/tmp,/scratch/fsw/portfolios/coreai/projects/coreai_tensorrt_ci/llm-models/GLM-5.3-Flash:/lustre/fsw/portfolios/coreai/projects/coreai_tensorrt_ci/llm-models/GLM-5.3-Flash:ro"
srun --ntasks=1 --cpu-bind=none --container-image="$IMAGE" --container-mounts="$MOUNTS" --no-container-mount-home --container-workdir=/glm-align/source bash -c '
set -Eeuo pipefail
export PYTHONPATH=/glm-align/source/python:/glm-align/source:/work/results/remeasure-v6-fivearm-tp1-20261004/inputs/helper-runtime/helpers:/work/pyuser/lib/python3.12/site-packages
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/work/hf-cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/glm-align/cache/accuracy-datasets
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MAX_JOBS=4 TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=0 SGLANG_GLM_ALIGN_PREPARE=0 SGLANG_GLM_SAFE_PACKED_DECODE=0
unset TRITON_INTERPRET SGLANG_EXTERNAL_MODEL_PACKAGE
RUN_OUT=/glm-align/jobs/$SLURM_JOB_ID
export HF_MODULES_CACHE=$RUN_OUT/hf-modules
export TMPDIR=/tmp/glmalign-$SLURM_JOB_ID XDG_CACHE_HOME=$RUN_OUT/cache TRITON_CACHE_DIR=$RUN_OUT/triton TORCH_EXTENSIONS_DIR=$RUN_OUT/torch FLASHINFER_WORKSPACE_BASE=$RUN_OUT/flashinfer SGLANG_CACHE_DIR=$RUN_OUT/sglang
mkdir -p "$HF_MODULES_CACHE" "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$FLASHINFER_WORKSPACE_BASE" "$SGLANG_CACHE_DIR"
if [[ $GLM_ALIGN_MODE == cpu ]]; then
 export CUDA_VISIBLE_DEVICES=""
 python test/manual/duet/glm_align_cpu_gate.py --out "$RUN_OUT/cpu"
else
 python -m torch.distributed.run --standalone --nproc_per_node=4 test/manual/duet/glm_align_gpu_gate.py --stage micro --out "$RUN_OUT/micro"
 python test/manual/duet/glm_align_gpu_gate.py --stage sanity --out "$RUN_OUT/sanity" --helpers /work/results/seed-serve-20261011/inputs-v35-mem065-r1 --reference-root /work/results/remeasure-v6-fivearm-tp1-20261004/inputs/helper-runtime/helpers
fi
'
