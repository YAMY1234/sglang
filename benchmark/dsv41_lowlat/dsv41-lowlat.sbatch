#!/bin/bash
#SBATCH --job-name=dsv41-lowlat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Add --account / --partition / --qos for your cluster on the sbatch command line.
#
# DSV4.1-Flash low-latency decode on one GB300 node (TP4). UNIT_TESTS = GPU pytest files before any server.
# SCREEN: OSL=512 ROUNDS=1 PROF=0 ACC=0 CONC=1,4 MAXBS=8. FORMAL: OSL=1024 ROUNDS=2 CONC=1,2,4,8.
# Per arm (ARMS="name:srcdir:envfile ..."), two server launches from the same source:
#   perf: synthetic acceptance 3.51 (as the InferenceX AgentX recipe), under nsys capture-range;
#         decode ISL 8K / OSL 1K at c=1,2,4,8, two rounds (round 2 reversed), then one bs=1 nsys window
#   acc : real acceptance; GSM8K-200 greedy cold + decode c=1 (real acceptance length)
set -euo pipefail
# ---- cluster-specific paths (CMH values from the published runs as defaults) ----
ROOT=${ROOT:-/lustre/fsw/portfolios/coreai/users/yangminl/dsv41-lowlat-cmh}   # shared cache dir lives here
D=${D:?set D to an output directory (runs/ is written below it)}
BENCH=${BENCH:?set BENCH to this directory (benchmark/dsv41_lowlat of the checkout)}
CL=$BENCH/ab_bench.py
MODEL=${MODEL:-/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/weiliangl/models/DeepSeek-V4.1-Flash}
IMG=${IMG:-$ROOT/containers/lmsysorg+sglang+nightly-dev-cu13-20261012-e6072935-arm64.sqsh}
# -------------------------------------------------------------------------------
for _ in $(seq 1 135); do [[ -s "$IMG" ]] && break; sleep 20; done; echo "IMG_READY $(date -u +%FT%TZ) $(ls -la "$IMG")"
NSYS=${NSYS:-nsys}  # nsys 2026.5.1 binary visible inside the container (only used with PROF=1)
W=/local/dsv41-lowlat-$SLURM_JOB_ID
mkdir -p "$W/cache/home" 2>/dev/null || { W=/tmp/${W##*/}; mkdir -p "$W/cache/home"; }
echo "ALLOCATION_START $(date -u +%FT%TZ) node=$(hostname) job=$SLURM_JOB_ID arms=${ARMS:?}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader; uname -m; echo "W=$W"
PORT=$((5000 + (SLURM_JOB_ID % 30) * 100))
export BASE_URL=http://127.0.0.1:$PORT
SERVER_PID=
cleanup() {
  [[ -n "${SERVER_PID:-}" ]] || return 0
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 3; done
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=
  for _ in $(seq 1 30); do
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] && break
    sleep 3
  done
}
trap 'cleanup; rm -rf "$W"' EXIT

serve() {  # serve <srcdir> <outdir> <nsys 0|1> <extra env...>
  local SRC=$1 OUT=$2 PROF=$3; shift 3
  ARGS=(--model-path /model --trust-remote-code --tp 4 --ep-size 4 --mem-fraction-static 0.8
    --cuda-graph-max-bs-decode ${MAXBS:-16} --max-running-requests 16 --chunked-prefill-size 8192
    --speculative-algorithm DSPARK --speculative-dspark-block-size 5
    ${REPLAY_ARGS-}
    --served-model-name deepseek-ai/DeepSeek-V4.1-Flash --watchdog-timeout 7200
    --enable-cache-report --enable-metrics --host 0.0.0.0 --port "$PORT" "${XARGS[@]}")
  SENV=(PYTHONPATH=/src/python PYTHONDONTWRITEBYTECODE=1 HOME=$W/cache/home PYTHONUNBUFFERED=1
    SGLANG_RUST_BUILD_MODE=auto SGLANG_CACHE_DIR=$ROOT/cache/sglang XDG_CACHE_HOME=$W/cache
    CARGO_HOME=/root/.cargo RUSTUP_HOME=/root/.rustup
    TRITON_CACHE_DIR=$W/cache/triton TORCHINDUCTOR_CACHE_DIR=$W/cache/inductor "$@")
  CT=(--container-image="$IMG" --container-mounts="$SRC:/src:ro,$MODEL:/model:ro,$ROOT:$ROOT,$D:$D,$BENCH:$BENCH,$W:$W"
    --container-workdir="$OUT" --container-remap-root --no-container-mount-home)
  mkdir -p "$OUT"
  printf '%s\n' "${SENV[@]}" "ARGS ${ARGS[*]}" "SRC_HEAD $(git -C "$SRC" rev-parse HEAD)" \
    "SRC_DIRTY $(git -C "$SRC" status --porcelain --untracked-files=no | wc -l)" "IMG $IMG" "NSYS $PROF" "HOST $(hostname)" > "$OUT/launch.txt"
  local PRE=()
  [[ "$PROF" == 1 ]] && PRE=("$NSYS" profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi
    --capture-range-end=stop --trace-fork-before-exec=true -f true -o "$OUT/prof/decode-bs1")
  mkdir -p "$OUT/prof"
  echo "SERVER_START $(date -u +%FT%TZ) out=$OUT"
  srun --overlap --ntasks=1 "${CT[@]}" env "${SENV[@]}" "${PRE[@]}" python3 -m sglang.launch_server "${ARGS[@]}" > "$OUT/server.log" 2>&1 &
  SERVER_PID=$!
  local T0=$(date +%s) READY=0
  for _ in $(seq 1 360); do
    if curl -fsS "$BASE_URL/health_generate" >/dev/null 2>&1; then READY=1; break; fi
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 5
  done
  if [[ "$READY" != 1 ]]; then echo "START_FAIL after $(( $(date +%s) - T0 )) s"; tail -60 "$OUT/server.log"; cleanup; return 1; fi
  echo "START_PASS $(date -u +%FT%TZ) ready_s=$(( $(date +%s) - T0 ))"
  curl -fsS "$BASE_URL/get_server_info" > "$OUT/server-info.json" || true
  RUNC=(srun --overlap --ntasks=1 "${CT[@]}" env BASE_URL="$BASE_URL" OUT="$OUT" PYTHONDONTWRITEBYTECODE=1)
}

CONC=${CONC:-1,2,4,8}
if [[ "${UNIT:-0}" == 1 ]]; then  # GPU tests of the branch before any server; UNIT_TESTS="file ..."; UNIT_GPUS=0 or 0,1,2,3
  IFS=: read -r _ USRC _ <<< "${ARMS%% *}"
  for t in ${UNIT_TESTS:?}; do
    srun --overlap --ntasks=1 --container-image="$IMG" --container-mounts="$USRC:/src:ro,$ROOT:$ROOT,$D:$D,$W:$W" \
      --container-workdir=/src --container-remap-root --no-container-mount-home \
      env PYTHONPATH=/src/python PYTHONDONTWRITEBYTECODE=1 HOME=$W/cache/home XDG_CACHE_HOME=$W/cache CUDA_VISIBLE_DEVICES=${UNIT_GPUS:-0} \
      python3 -m pytest -q -p no:cacheprovider -W ignore --tb=short $t 2>&1 | tail -40 | sed "s|^|UNIT ${t##*/} |" || true
  done
  for t in ${UNIT_SCRIPTS:-}; do  # standalone multi-GPU test scripts: "file" runs as python3 file --num-gpu 4
    srun --overlap --ntasks=1 --container-image="$IMG" --container-mounts="$USRC:/src:ro,$ROOT:$ROOT,$D:$D,$W:$W" \
      --container-workdir=/src --container-remap-root --no-container-mount-home \
      env PYTHONPATH=/src/python PYTHONDONTWRITEBYTECODE=1 HOME=$W/cache/home XDG_CACHE_HOME=$W/cache \
      python3 $t --num-gpu 4 2>&1 | grep -v Warning | tail -40 | sed "s|^|UNIT ${t##*/} |" || true
  done
  echo "UNIT_END $(date -u +%FT%TZ)"
fi

for spec in $ARMS; do
  IFS=: read -r ARM SRC ENVF <<< "$spec"
  XENV=() XARGS=()
  # env file: KEY=VALUE lines are server env; lines starting with -- are extra server args
  [[ -n "$ENVF" ]] && mapfile -t XENV < <(grep -v '^#' "$ENVF" | sed '/^$/d' | grep -v '^--')
  [[ -n "$ENVF" ]] && read -ra XARGS <<< "$(grep '^--' "$ENVF" | tr '\n' ' ')"
  echo "ARM $ARM src=$SRC env=${XENV[*]:-none} args=${XARGS[*]:-none}"
  # perf launch: synthetic acceptance, nsys capture range
  OUT=$D/runs/$ARM-$SLURM_JOB_ID/perf
  if [[ "${PERF:-1}" == 1 ]] && serve "$SRC" "$OUT" "${PROF:-1}" SGLANG_SIMULATE_ACC_LEN=3.51 SGLANG_SIMULATE_ACC_METHOD=match-expected \
      SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token SGLANG_TORCH_PROFILER_DIR=$OUT/prof "${XENV[@]}"; then
    set +e
    "${RUNC[@]}" python3 $CL decode --label "$ARM" --decode-conc "$CONC" --decode-osl "${OSL:-1024}" --out "$OUT/bench" 2>&1 | tee "$OUT/client.log"
    [[ "${ROUNDS:-2}" == 2 ]] && "${RUNC[@]}" python3 $CL decode --label "$ARM" --decode-conc "$CONC" --decode-osl "${OSL:-1024}" --reverse --out "$OUT/bench" 2>&1 | tee -a "$OUT/client.log"
    [[ "${PROF:-1}" == 1 ]] && "${RUNC[@]}" python3 $CL prof --label "$ARM" --prof-windows decode \
      --prof-decode-conc 1 --prof-decode-steps 10 --out "$OUT/bench" 2>&1 | tee -a "$OUT/client.log"
    set -e
    echo "PERF_END $ARM $(date -u +%FT%TZ)"
    grep -c -E "Traceback|RuntimeError|ERROR" "$OUT/server.log" | sed 's/^/SERVER_ERROR_LINES /' || true
    cleanup
    ls -la "$OUT/prof" || true
  fi
  [[ "${ACC:-1}" == 1 ]] || continue
  [[ " ${NOACC_ARMS:-} " == *" $ARM "* ]] && continue
  # accuracy launch: real acceptance
  OUT=$D/runs/$ARM-$SLURM_JOB_ID/acc
  if serve "$SRC" "$OUT" 0 "${XENV[@]}"; then
    set +e
    "${RUNC[@]}" python3 $BENCH/l6_gate.py --prompts $BENCH/gsm8k200.jsonl --gsm-cold-only --base-url "$BASE_URL" --out "$OUT/gate" 2>&1 | tee "$OUT/client.log"
    "${RUNC[@]}" python3 $CL decode --label "$ARM-realacc" --decode-conc 1 --out "$OUT/bench" 2>&1 | tee -a "$OUT/client.log"
    set -e
    echo "ACC_END $ARM $(date -u +%FT%TZ)"
    grep -c -E "Traceback|RuntimeError|ERROR" "$OUT/server.log" | sed 's/^/SERVER_ERROR_LINES /' || true
    cleanup
  fi
done
echo "JOB_END $(date -u +%FT%TZ)"
