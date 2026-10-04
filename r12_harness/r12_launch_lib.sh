#!/bin/bash

# Shared launch construction for the renderer and the real rank launcher.
# Callers must use bash and set -euo pipefail themselves.

R12_EXPECTED_SOURCE_SHA=cb0b3498fcc2f398229b0b8cb9df0a5825e1438a

r12_common_env() {
  R12_ENV=(
    PYTHONPATH=/runtime/pipdeps:/src/python
    PYTHONDONTWRITEBYTECODE=1
    PYTHONUNBUFFERED=1
    PYTHONNOUSERSITE=1
    TZ=UTC
    SGLANG_IS_IN_CI=1
    SGLANG_LOG_MS=1
    SGLANG_R7_STAGE_SKEW_TRACE=1
    SGLANG_RUST_BUILD_MODE=auto
    SGLANG_DISAGG_STAGING_BUFFER=1
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
    SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S=1200
    MC_FORCE_MNNVL=1
    NCCL_MNNVL_ENABLE=1
    NCCL_CUMEM_ENABLE=1
    NCCL_NVLS_ENABLE=0
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  )
}

r12_build_role_command() {
  local role=$1 variant=$2 chunk=$3 rank=$4 master=$5 dist_port=$6
  local http_port=$7 nccl_port=$8 bootstrap_port=$9 service=${10}
  local mem_fraction=${11:-0.90}
  local cache=/runtime/cache/$service/rank-$rank
  local -a role_env topology

  r12_common_env
  role_env=(
    SGLANG_CACHE_DIR=$cache/sglang
    SGLANG_JIT_CACHE_DIR=$cache/sglang-jit
    XDG_CACHE_HOME=$cache/xdg
    TRITON_CACHE_DIR=$cache/triton
    TORCHINDUCTOR_CACHE_DIR=$cache/torchinductor
    TORCH_EXTENSIONS_DIR=$cache/torch-extensions
    CUDA_CACHE_PATH=$cache/cuda
  )
  topology=()
  if [[ "$role" == prefill ]]; then
    case "$variant" in
      PP)
        role_env+=(SGLANG_PP_COMM_OVERLAP=1 SGLANG_PP_LAYER_PARTITION=24,23,23,23)
        topology=(--tp-size 2 --ep-size 2 --pp-size 4 --disable-overlap-schedule)
        ;;
      TEP)
        topology=(--tp-size 8 --ep-size 8)
        ;;
      *) echo "invalid prefill variant=$variant" >&2; return 2 ;;
    esac
  elif [[ "$role" == decode ]]; then
    [[ "$variant" == TEP ]] || { echo "invalid decode variant=$variant" >&2; return 2; }
    [[ "$chunk" == 8192 ]] || { echo "invalid decode chunk=$chunk" >&2; return 2; }
    topology=(--tp-size 8 --ep-size 8)
  else
    echo "invalid role=$role" >&2
    return 2
  fi

  R12_COMMAND=(
    env "${R12_ENV[@]}" "${role_env[@]}"
    python3 -m sglang.launch_server
    --model-path /model --served-model-name moonshotai/Kimi-K3
    --trust-remote-code
    --nnodes 2 --node-rank "$rank" --dist-init-addr "$master:$dist_port"
    --host 0.0.0.0 --port "$http_port" --nccl-port "$nccl_port" --random-seed 0
    --moe-runner-backend flashinfer_mxfp4
    --attention-backend trtllm_mla --decode-attention-backend trtllm_mla
    --mem-fraction-static "$mem_fraction" --context-length 65536
    --chunked-prefill-size "$chunk" --max-prefill-tokens "$chunk"
    --max-running-requests 128
    --cuda-graph-max-bs-decode 128 --mamba-full-memory-ratio 0.86
    --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer
    --reasoning-parser kimi_k3 --tool-call-parser kimi_k3
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":3}'
    --disable-flashinfer-autotune --cuda-graph-backend-prefill breakable
    --cuda-graph-backend-decode disabled --skip-server-warmup
    --watchdog-timeout 7200 --decode-log-interval 100
    --speculative-algorithm DSPARK --speculative-draft-model-path /draft
    --speculative-dspark-block-size 7
    --disaggregation-mode "$role" --disaggregation-transfer-backend mooncake
    --disaggregation-bootstrap-port "$bootstrap_port"
    --enable-metrics "${topology[@]}"
  )
}

r12_build_router_command() {
  local p_node=$1 p_port=$2 d_node=$3 d_port=$4 router_port=$5
  R12_COMMAND=(
    env PYTHONPATH=/runtime/pipdeps:/src/python PYTHONDONTWRITEBYTECODE=1
    PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TZ=UTC
    python3 -m sglang_router.launch_router --pd-disaggregation --mini-lb
    --prefill "http://$p_node:$p_port" --decode "http://$d_node:$d_port"
    --host 0.0.0.0 --port "$router_port"
  )
}

r12_emit_command() {
  local label=$1 output=$2
  {
    printf '%s ' "$label"
    printf '%q ' "${R12_COMMAND[@]}"
    printf '\n'
  } >"$output"
}
