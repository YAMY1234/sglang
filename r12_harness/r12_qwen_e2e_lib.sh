#!/bin/bash

# Qwen3.8 launch construction for the budget fallback validation.  It uses the
# same R12_COMMAND/r12_emit_command contract as the Kimi renderer and role
# launcher, while replacing only model/topology-specific tokens with R14's
# already validated configuration.

R12_QWEN_SOURCE_SHA=10edb05e72afb79ecfdf7dbe7603f90ba05ad3ee

r12_qwen_common_env() {
  R12_ENV=(
    PYTHONPATH=/runtime/pipdeps:/src/python
    PYTHONDONTWRITEBYTECODE=1
    PYTHONUNBUFFERED=1
    PYTHONNOUSERSITE=1
    TZ=UTC
    SGLANG_IS_IN_CI=1
    SGLANG_LOG_MS=1
    SGLANG_RUST_BUILD_MODE=auto
    SGLANG_DISAGG_STAGING_BUFFER=1
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
    MC_FORCE_MNNVL=1
    NCCL_MNNVL_ENABLE=1
    NCCL_CUMEM_ENABLE=1
    NCCL_NVLS_ENABLE=0
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  )
}

r12_qwen_build_role_command() {
  local role=$1 variant=$2 chunk=$3 http_port=$4 nccl_port=$5
  local bootstrap_port=$6 service=$7
  local cache=/runtime/cache/$service
  local -a role_env topology

  r12_qwen_common_env
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
        role_env+=(SGLANG_PP_COMM_OVERLAP=1 SGLANG_PP_LAYER_PARTITION=12,12,12,12)
        topology=(--tp-size 1 --ep-size 1 --pp-size 4 --disable-overlap-schedule)
        ;;
      TEP) topology=(--tp-size 4 --ep-size 4) ;;
      *) echo "invalid Qwen prefill variant=$variant" >&2; return 2 ;;
    esac
  elif [[ "$role" == decode ]]; then
    [[ "$variant" == TEP ]] || { echo "invalid Qwen decode variant=$variant" >&2; return 2; }
    topology=(--tp-size 4 --ep-size 4)
  else
    echo "invalid Qwen role=$role" >&2
    return 2
  fi

  R12_COMMAND=(
    env "${R12_ENV[@]}" "${role_env[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3
    python3 -m sglang.launch_server
    --model-path /model --served-model-name Qwen/Qwen3.8-Flash-Next
    --trust-remote-code --random-seed 42
    --host 0.0.0.0 --port "$http_port" --nccl-port "$nccl_port"
    --disaggregation-transfer-backend mooncake
    --disaggregation-bootstrap-port "$bootstrap_port"
    --mem-fraction-static 0.85 --max-running-requests 128
    --max-prefill-tokens 32768 --chunked-prefill-size "$chunk" --page-size 64
    --attention-backend trtllm_mha --moe-runner-backend flashinfer_trtllm
    --linear-attn-prefill-backend flashinfer --linear-attn-decode-backend flashinfer
    --mamba-ssm-dtype bfloat16 --disable-radix-cache
    --disable-shared-experts-fusion --reasoning-parser qwen3-thinking
    --watchdog-timeout 2400 --weight-loader-prefetch-checkpoints
    --weight-loader-prefetch-num-threads 4 --enable-metrics --skip-server-warmup
    --cuda-graph-backend-prefill disabled
    --speculative-algorithm NEXTN --speculative-num-steps 3
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
    --disaggregation-mode "$role" "${topology[@]}"
  )
}

r12_qwen_build_router_command() {
  local p_node=$1 p_port=$2 d_node=$3 d_port=$4 router_port=$5
  R12_COMMAND=(
    env PYTHONPATH=/runtime/pipdeps:/src/python PYTHONDONTWRITEBYTECODE=1
    PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TZ=UTC
    python3 -m sglang_router.launch_router --pd-disaggregation --mini-lb
    --prefill "http://$p_node:$p_port" --decode "http://$d_node:$d_port"
    --host 0.0.0.0 --port "$router_port"
  )
}

r12_qwen_emit_command() {
  local label=$1 output=$2
  {
    printf '%s ' "$label"
    printf '%q ' "${R12_COMMAND[@]}"
    printf '\n'
  } >"$output"
}
