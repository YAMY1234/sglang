#!/usr/bin/env bash
set -euxo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG="$ROOT/logs_70b"; mkdir -p "$LOG"
NSYS_DIR="$ROOT/nsys"; mkdir -p "$NSYS_DIR"

echo "=== 70B 模型 4GPU Pipeline Parallelism CUDA Graphs 延迟测试 ==="

run_70b_test () {
  local CASE="$1" ; local ARGS_FILE="$2" ; local PORT="$3"
  
  echo "启动 $CASE 测试 (70B模型，4张GPU)..."
  
  # 启动服务器 - 使用4张GPU + nsys profiling
  CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile \
    --trace=cuda,nvtx,osrt \
    --force-overwrite=true \
    --sample=none \
    --cpuctxsw=process-tree \
    -o "${NSYS_DIR}/${CASE}" \
    python3 -m sglang.launch_server \
    $(cat "$ROOT/server_args_base_70b_fixed.txt") \
    $(cat "$ARGS_FILE") \
    --host 127.0.0.1 --port "${PORT}" \
    > "${LOG}/${CASE}_server.log" 2>&1 &
  SERVER_PID=$!

  # 等待就绪（70B模型需要更长时间）
  echo "等待服务器启动（70B模型加载需要较长时间）..."
  for i in {1..180}; do  # 增加到3分钟等待时间
    if nc -z 127.0.0.1 $PORT 2>/dev/null; then
      echo "服务器就绪 (${i}s)"
      break
    fi
    sleep 1
    if [ $i -eq 60 ]; then
      echo "1分钟后仍在加载..."
    elif [ $i -eq 120 ]; then
      echo "2分钟后仍在加载..."
    fi
  done

  # 检查服务器是否真的就绪
  if ! nc -z 127.0.0.1 $PORT 2>/dev/null; then
    echo "服务器启动失败，检查日志: ${LOG}/${CASE}_server.log"
    kill ${SERVER_PID} || true
    return 1
  fi

  # 压测 - 增加batch size来触发更多的PP通信
  echo "开始压测..."
  for round in {1..5}; do
    echo "测试轮次 $round/5"
    python3 "$ROOT/client_benchmark.py" --host 127.0.0.1 --port "${PORT}" \
      --batch 4 --max_new_tokens 64 \
      > "${LOG}/${CASE}_client_round${round}.log" 2>&1 || true
    sleep 2
  done

  echo "$CASE 测试完成"
  
  # 收尾
  kill ${SERVER_PID} || true
  wait ${SERVER_PID} || true
  sleep 5  # 给更多时间清理
}

# A: 开启 CUDA Graphs（问题态）
echo "=== Case A: 70B + CUDA Graphs 开启 ==="
run_70b_test "CASE_A_70B_graphs_on" "$ROOT/server_args_graphs_on.txt" 9000

# B: 关闭 CUDA Graphs（对照）
echo "=== Case B: 70B + CUDA Graphs 关闭 ==="  
run_70b_test "CASE_B_70B_graphs_off" "$ROOT/server_args_graphs_off.txt" 9001

echo ""
echo "=== 70B 测试完成，分析结果 ==="
echo ""

# 分析结果
for case in "CASE_A_70B_graphs_on" "CASE_B_70B_graphs_off"; do
  echo "=== $case 结果 ==="
  total_time_sum=0
  tpot_sum=0
  valid_rounds=0
  
  for round in {1..5}; do
    if [ -f "${LOG}/${case}_client_round${round}.log" ]; then
      echo "Round $round:"
      result=$(cat "${LOG}/${case}_client_round${round}.log" | grep -E "(total_time_s|approx_tpot_s)" || echo "")
      if [ -n "$result" ]; then
        echo "$result"
        # 提取数值进行计算
        total_time=$(echo "$result" | grep "total_time_s" | grep -o '[0-9.]*')
        tpot=$(echo "$result" | grep "approx_tpot_s" | grep -o '[0-9.]*')
        if [ -n "$total_time" ] && [ -n "$tpot" ]; then
          total_time_sum=$(echo "$total_time_sum + $total_time" | bc -l)
          tpot_sum=$(echo "$tpot_sum + $tpot" | bc -l)
          valid_rounds=$((valid_rounds + 1))
        fi
      else
        echo "无结果数据"
      fi
    fi
  done
  
  if [ $valid_rounds -gt 0 ]; then
    avg_total_time=$(echo "scale=3; $total_time_sum / $valid_rounds" | bc -l)
    avg_tpot=$(echo "scale=6; $tpot_sum / $valid_rounds" | bc -l)
    avg_tpot_ms=$(echo "scale=3; $avg_tpot * 1000" | bc -l)
    echo "平均总时间: ${avg_total_time}s"
    echo "平均TPOT: ${avg_tpot}s (${avg_tpot_ms}ms)"
  fi
  echo ""
done

echo "服务器日志位置: $LOG"
echo ""
echo "预期结果（基于Issue #10865）:"
echo "- Case A (CUDA Graphs开启): 应该出现~100ms的PP通信延迟"
echo "- Case B (CUDA Graphs关闭): 应该只有~3ms的正常延迟" 