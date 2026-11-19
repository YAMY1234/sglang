#!/bin/bash
# Example: one arm on 2 exclusive VR200 nodes (4 GPUs each) with Slurm + a container runtime plugin (pyxis/enroot).
# Submit both arms at the same time so they run in the same window, e.g.
#   sbatch -A <account> -p <partition> slurm_example.sh bench-pp8
#   sbatch -A <account> -p <partition> slurm_example.sh bench-tep8
# Set the paths below (or export them before sbatch). WORK must be on a filesystem both nodes see.
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --job-name=k31-pp-prefill
set -eu
PLAN=${1:?plan name, e.g. bench-pp8}
IMAGE=${IMAGE:-lmsysorg/sglang:dev-kimi-k31-pro-cu134@sha256:a42079a8d1299b94824eab821de5f261c016797434cf69dc1fb3b15bdfddfc0e}
SGLANG_REPO=${SGLANG_REPO:?path to this repo checkout (branch k31-pp8-prefill-repro)}
MODEL_PATH=${MODEL_PATH:?local snapshot of moonshotai/Kimi-K3.1-Pro}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:?local snapshot of RadixArk/Kimi-K3.1-Dflash2-Preview}
DEEP_GEMM_DIR=${DEEP_GEMM_DIR:?directory with sgl-deep-gemm 0.2.8 installed via pip --target}
WORK=${WORK:?shared output directory}
RUST_EXT_CACHE=${RUST_EXT_CACHE:-$WORK/rust-ext-cache}  # prebuilt Rust TreeCore (README "Software"), under WORK
RUN=$WORK/$PLAN-$(date -u +%m%dT%H%M%S)
mkdir -p "$RUN"
B=$SGLANG_REPO/benchmark/k31_pp_prefill_repro
MOUNTS=$SGLANG_REPO,$MODEL_PATH,$DRAFT_MODEL_PATH,$DEEP_GEMM_DIR,$WORK
srun --ntasks-per-node=1 --container-image="$IMAGE" --container-mounts="$MOUNTS" --no-container-mount-home \
  --export=ALL,SGLANG_REPO="$SGLANG_REPO",MODEL_PATH="$MODEL_PATH",DRAFT_MODEL_PATH="$DRAFT_MODEL_PATH",PYTHONPATH="$DEEP_GEMM_DIR",RUST_EXT_CACHE="$RUST_EXT_CACHE" \
  bash "$B/run_arm.sh" "$RUN" "$B/plans/$PLAN.txt" > "$RUN/driver.log" 2>&1
python3 "$B/parse_results.py" "$WORK" "$(basename "$RUN")" > "$RUN/summary.jsonl" || true
