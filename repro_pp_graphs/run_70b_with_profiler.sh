#!/usr/bin/env bash
set -euxo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG="$ROOT/logs_70b_profiler"; mkdir -p "$LOG"
TRACE_DIR="$ROOT/trace"; mkdir -p "$TRACE_DIR"

echo "=== 70B模型 Pipeline Parallelism CUDA Graphs 延迟测试 (带PyTorch Profiler) ==="

run_70b_test_with_profiler () {
  local CASE="$1" ; local ARGS_FILE="$2" ; local PORT="$3"
  
  echo "启动 $CASE 测试 (70B模型，4张GPU，带profiler)..."
  
  # 设置PyTorch Profiler环境变量
  export SGLANG_TORCH_PROFILER_DIR="${TRACE_DIR}"
  export SGLANG_PROFILE_WITH_STACK=true
  
  # 启动服务器 - 使用4张GPU
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
    $(cat "$ROOT/server_args_base_70b_fixed.txt") \
    $(cat "$ARGS_FILE") \
    --host 127.0.0.1 --port "${PORT}" \
    > "${LOG}/${CASE}_server.log" 2>&1 &
  SERVER_PID=$!

  # 等待就绪（70B模型需要更长时间）
  echo "等待服务器启动（70B模型加载需要较长时间）..."
  for i in {1..240}; do  # 4分钟等待
    if nc -z 127.0.0.1 $PORT 2>/dev/null; then
      echo "服务器就绪 (${i}s)"
      break
    fi
    sleep 1
    if [ $((i % 30)) -eq 0 ]; then
      echo "  仍在启动中... (${i}s)"
    fi
  done

  # 检查服务器是否真的就绪
  if ! nc -z 127.0.0.1 $PORT 2>/dev/null; then
    echo "服务器启动失败，检查日志: ${LOG}/${CASE}_server.log"
    kill ${SERVER_PID} || true
    return 1
  fi

  # 等待服务器完全就绪
  sleep 5
  
  # 启动profiling
  echo "启动profiling..."
  python3 -m sglang.profiler \
    --url "http://127.0.0.1:${PORT}" \
    --num-steps 10 \
    --cpu --gpu \
    --output-dir "${TRACE_DIR}" \
    --profile-name "${CASE}" &
  PROFILER_PID=$!
  
  # 等待profiler准备好
  sleep 3

  # 压测 - 触发decode阶段
  echo "开始压测，触发decode阶段..."
  for round in {1..3}; do
    echo "测试轮次 $round/3"
    python3 "$ROOT/client_benchmark.py" --host 127.0.0.1 --port "${PORT}" \
      --batch 8 --max_new_tokens 64 \
      > "${LOG}/${CASE}_client_round${round}.log" 2>&1 || true
    sleep 2
  done

  echo "$CASE 测试完成"
  
  # 等待profiler完成
  wait $PROFILER_PID || true
  
  # 关闭服务器
  kill ${SERVER_PID} || true
  wait ${SERVER_PID} || true
  sleep 5
}

# Case A: CUDA Graphs开启（问题态）
echo "=== Case A: 70B + CUDA Graphs 开启 + Profiler ==="
run_70b_test_with_profiler "CASE_A_70B_graphs_on_profiler" "$ROOT/server_args_graphs_on.txt" 9000

# Case B: CUDA Graphs关闭（对照）
echo "=== Case B: 70B + CUDA Graphs 关闭 + Profiler ==="  
run_70b_test_with_profiler "CASE_B_70B_graphs_off_profiler" "$ROOT/server_args_graphs_off.txt" 9001

echo ""
echo "=== Profiler测试完成 ==="
echo ""
echo "生成的文件："
echo "- 日志: $LOG"
echo "- Profiler traces: $TRACE_DIR"
echo ""
echo "分析方法："
echo "1. 查找 $TRACE_DIR 中的 .json 文件"
echo "2. 用 ui.perfetto.dev 打开 JSON 文件"
echo "3. 搜索 'cudagraph' 和 'send_tensor_dict' 查看具体的PP通信延迟"
echo "4. 对比两个case中最后一个PP rank发送token的延迟差异"

# 列出生成的trace文件
echo ""
echo "生成的trace文件："
find "$TRACE_DIR" -name "*.json" -o -name "*.pt.trace.json" 2>/dev/null || echo "未找到trace文件" 