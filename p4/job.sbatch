#!/bin/bash
#SBATCH -A coreai_comparch_inferencex
#SBATCH -p batch
#SBATCH --qos=normal
#SBATCH -N1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=400G
#SBATCH -t 00:15:00
#SBATCH -J p4-gate
set -euo pipefail
W=/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
P4_TREE=${P4_TREE:?}
R=$W/results/bf16-regress-20261013
source "$W/results/ssm-kernel-219/holder/scripts/hlib.sh"
O=$W/results/ssm-kernel-231/gate-j$SLURM_JOB_ID
mkdir -p "$O"
P4_TREE_REL=$(realpath --relative-to="$W" "$P4_TREE")
[[ $P4_TREE_REL != ..* ]]
P4_CONTAINER_TREE=/work/$P4_TREE_REL
export P4_CONTAINER_TREE
srun -N1 -n1 --gres=gpu:4 --cpus-per-task=32 --container-image="$(h_image)" --container-mounts="$(h_mounts)" --no-container-mount-home --container-workdir=/work --export=ALL,CUDA_VISIBLE_DEVICES=0 bash -c '
set -euo pipefail
export PYTHONPATH=/work/pyuser/lib/python3.12/site-packages
export P4_OUT=/work/results/ssm-kernel-231/gate-j$SLURM_JOB_ID
nvidia-smi -q > "$P4_OUT/nvidia-before.txt"
bash "$P4_CONTAINER_TREE/p4/run_gate.sh" 0
if [[ ${P4_MICRO:-0} == 1 ]]; then bash "$P4_CONTAINER_TREE/p4/micro.sh" 0; fi
nvidia-smi -q > "$P4_OUT/nvidia-after.txt"
'
