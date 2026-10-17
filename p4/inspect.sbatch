#!/bin/bash
#SBATCH -A coreai_comparch_inferencex
#SBATCH -p batch
#SBATCH --qos=normal
#SBATCH -N1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=400G
#SBATCH -t 00:05:00
#SBATCH -J p4-lowering
set -euo pipefail
W=/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
P4_TREE=${P4_TREE:?}
O=$W/results/ssm-kernel-231/inspect-j$SLURM_JOB_ID
mkdir -p "$O"
srun -N1 -n1 --gres=gpu:4 --cpus-per-task=32 --container-image="$W/containers/lmsysorg+sglang+nightly-dev-cu13-20260913-fa663e72-arm64-enroot.sqsh" --container-mounts="$W:/work,/tmp:/tmp" --no-container-mount-home --container-workdir=/work --export=ALL,CUDA_VISIBLE_DEVICES=0 bash -c '
set -euo pipefail
export PYTHONPATH="${P4_TREE/#\/lustre\/fsw\/portfolios\/coreai\/users\/yangminl\/twinstar-20260911/\/work}/python:/work/pyuser/lib/python3.12/site-packages"
export TRITON_CACHE_DIR=/tmp/p4-norm-inspect
python "${PYTHONPATH%%/python:*}/p4/inspect_norm.py" /work/results/ssm-kernel-231/inspect-j$SLURM_JOB_ID
'
