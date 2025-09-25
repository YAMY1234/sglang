#!/usr/bin/env bash
set -euxo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG="$ROOT/logs"; TRACE="$ROOT/trace"
mkdir -p "$LOG" "$TRACE"

# 检查nsys是否可用
if ! command -v nsys &> /dev/null; then
    echo "警告: nsys 未安装，将跳过nsys分析"
    echo "如需安装nsys，请运行: apt-get install nvidia-nsight-systems-cli"
    exit 1
fi

# 记录环境
bash "$ROOT/collect_env.sh" > "$LOG/env.txt" 2>&1 || true

run_case_nsys () {
  local CASE="$1" ; local ARGS_FILE="$2" ; local PORT="$3"
  
  # nsys配置
  local NSYS_OUTPUT="${TRACE}/${CASE}_nsys_profile.qdrep"
  local NSYS_ARGS=(
    profile
    -o "$NSYS_OUTPUT"
    --trace=cuda,nvtx,nccl
    --cuda-memory-usage=true
    --force-overwrite=true
    --sample=none
    --backtrace=dwarf
    --delay=10  # 延迟10秒开始profiling，等待模型加载
    --duration=30  # 只profile 30秒，避免文件过大
  )

  echo "启动服务器 (Case: $CASE, Port: $PORT)"
  
  # 使用nsys启动服务器
  nsys "${NSYS_ARGS[@]}" python3 -m sglang.launch_server \
    $(cat "$ROOT/server_args_base.txt") \
    $(cat "$ARGS_FILE") \
    --host 127.0.0.1 --port "${PORT}" \
    > "${LOG}/${CASE}_server.log" 2>&1 &
  SERVER_PID=$!

  # 等待就绪（最多 120s）
  echo "等待服务器启动..."
  python3 - <<'PY' "${PORT}"
import time, socket, sys
port=int(sys.argv[1])
for i in range(120):
    try:
        s=socket.socket(); s.settimeout(0.2); s.connect(("127.0.0.1",port)); s.close()
        print(f"server ready after {i+1}s"); break
    except Exception: time.sleep(1)
else: raise SystemExit("server not ready")
PY

  echo "服务器就绪，开始压测..."
  
  # 运行多轮测试以确保进入decode阶段
  for round in {1..3}; do
    echo "运行测试轮次 $round/3..."
    python3 "$ROOT/client_benchmark.py" --host 127.0.0.1 --port "${PORT}" \
      --batch 4 --max_new_tokens 64 \
      > "${LOG}/${CASE}_client_round${round}.log" 2>&1 || true
    sleep 2
  done

  echo "测试完成，等待nsys结束..."
  # 等待nsys完成
  wait ${SERVER_PID} || true
  
  echo "Case $CASE 完成，nsys输出: $NSYS_OUTPUT"
}

run_case_regular () {
  local CASE="$1" ; local ARGS_FILE="$2" ; local PORT="$3"
  
  echo "启动常规服务器 (Case: $CASE, Port: $PORT)"
  
  # 常规启动（用于对比）
  python3 -m sglang.launch_server \
    $(cat "$ROOT/server_args_base.txt") \
    $(cat "$ARGS_FILE") \
    --host 127.0.0.1 --port "${PORT}" \
    > "${LOG}/${CASE}_server.log" 2>&1 &
  SERVER_PID=$!

  # 等待就绪
  python3 - <<'PY' "${PORT}"
import time, socket, sys
port=int(sys.argv[1])
for i in range(120):
    try:
        s=socket.socket(); s.settimeout(0.2); s.connect(("127.0.0.1",port)); s.close()
        print(f"server ready after {i+1}s"); break
    except Exception: time.sleep(1)
else: raise SystemExit("server not ready")
PY

  # 压测
  python3 "$ROOT/client_benchmark.py" --host 127.0.0.1 --port "${PORT}" \
    --batch 4 --max_new_tokens 64 \
    > "${LOG}/${CASE}_client.log" 2>&1 || true

  # 收尾
  kill ${SERVER_PID} || true
  wait ${SERVER_PID} || true
}

echo "开始 nsys 深度分析..."

# A: Decode + CUDA Graphs（问题态）- 使用nsys
echo "=== Case A: CUDA Graphs + nsys 分析 ==="
run_case_nsys "CASE_A_decode_graphs_nsys" "$ROOT/server_args_graphs_on.txt" 8000

# B: 关闭 CUDA Graphs（对照）- 常规测试
echo "=== Case B: 无 CUDA Graphs 对照 ==="
run_case_regular "CASE_B_no_graphs" "$ROOT/server_args_graphs_off.txt" 8001

echo ""
echo "=== 分析完成 ==="
echo "日志文件: $LOG"
echo "nsys文件: $TRACE/CASE_A_decode_graphs_nsys_nsys_profile.qdrep"
echo ""
echo "分析方法:"
echo "1. 使用 nsys-ui 打开 .qdrep 文件进行可视化分析"
echo "2. 或使用命令行: nsys stats $TRACE/CASE_A_decode_graphs_nsys_nsys_profile.qdrep"
echo "3. 重点关注 CUDA Graph replay 后的 NCCL 通信延迟" 