#!/usr/bin/env bash
set -euxo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG="$ROOT/logs"; mkdir -p "$LOG"

echo "=== 快速验证 DeepSeek-V3 PP CUDA Graphs 延迟问题 ==="

run_quick_test () {
  local CASE="$1" ; local ARGS_FILE="$2" ; local PORT="$3"
  
  echo "启动 $CASE 测试..."
  
  # 启服（使用较小模型快速测试）
  CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
    $(cat "$ROOT/server_args_base_quick.txt") \
    $(cat "$ARGS_FILE") \
    --host 127.0.0.1 --port "${PORT}" \
    > "${LOG}/${CASE}_server.log" 2>&1 &
  SERVER_PID=$!

  # 等待就绪
  echo "等待服务器启动..."
  for i in {1..60}; do
    if nc -z 127.0.0.1 $PORT 2>/dev/null; then
      echo "服务器就绪 (${i}s)"
      break
    fi
    sleep 1
  done

  # 快速压测
  echo "开始压测..."
  for round in {1..3}; do
    echo "测试轮次 $round/3"
    python3 "$ROOT/client_benchmark.py" --host 127.0.0.1 --port "${PORT}" \
      --batch 2 --max_new_tokens 32 \
      > "${LOG}/${CASE}_client_round${round}.log" 2>&1 || true
    sleep 1
  done

  echo "$CASE 测试完成"
  
  # 收尾
  kill ${SERVER_PID} || true
  wait ${SERVER_PID} || true
  sleep 2
}

# A: 开启 CUDA Graphs（问题态）
echo "=== Case A: CUDA Graphs 开启 ==="
run_quick_test "CASE_A_graphs_on" "$ROOT/server_args_graphs_on.txt" 8000

# B: 关闭 CUDA Graphs（对照）
echo "=== Case B: CUDA Graphs 关闭 ==="  
run_quick_test "CASE_B_graphs_off" "$ROOT/server_args_graphs_off.txt" 8001

echo ""
echo "=== 测试完成，分析结果 ==="
echo ""

# 分析客户端日志中的TPOT
for case in "CASE_A_graphs_on" "CASE_B_graphs_off"; do
  echo "=== $case 结果 ==="
  for round in {1..3}; do
    if [ -f "${LOG}/${case}_client_round${round}.log" ]; then
      echo "Round $round:"
      cat "${LOG}/${case}_client_round${round}.log" | grep -E "(total_time_s|approx_tpot_s)" || echo "无结果数据"
    fi
  done
  echo ""
done

echo "服务器日志位置: $LOG"
echo ""
echo "预期结果:"
echo "- Case A (CUDA Graphs开启): TPOT 较高，~100ms延迟"
echo "- Case B (CUDA Graphs关闭): TPOT 较低，~3ms延迟" 