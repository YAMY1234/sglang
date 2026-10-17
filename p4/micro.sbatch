#!/bin/bash
#SBATCH -A coreai_comparch_inferencex
#SBATCH -p batch
#SBATCH --qos=normal
#SBATCH -N1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=400G
#SBATCH -t 00:15:00
#SBATCH -J p4-micro
set -euo pipefail
W=/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
source "$W/results/ssm-kernel-219/holder/scripts/hlib.sh"
P4_TREE=${P4_TREE:?}; P4_GATE_JSON=${P4_GATE_JSON:?}
P4_CONTAINER_TREE=/work/${P4_TREE#"$W/"}; export P4_CONTAINER_TREE
O=$W/results/ssm-kernel-231/micro-j$SLURM_JOB_ID
mkdir -p "$O"
cp "$P4_GATE_JSON" "$O/gate.json"
srun -N1 -n1 --gres=gpu:4 --cpus-per-task=32 --container-image="$(h_image)" --container-mounts="$(h_mounts)" --no-container-mount-home --container-workdir=/work --export=ALL,CUDA_VISIBLE_DEVICES=0 bash -c '
set -euo pipefail
export PYTHONPATH=/work/pyuser/lib/python3.12/site-packages
export P4_OUT=/work/results/ssm-kernel-231/micro-j$SLURM_JOB_ID
nvidia-smi -q > "$P4_OUT/nvidia-before.txt"
bash "$P4_CONTAINER_TREE/p4/micro.sh" 0
nvidia-smi -q > "$P4_OUT/nvidia-after.txt"
'
