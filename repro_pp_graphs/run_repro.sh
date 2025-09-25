#!/usr/bin/env bash
set -euxo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG="$ROOT/logs"; TRACE="$ROOT/trace"
mkdir -p "$LOG" "$TRACE"

# 记录环境
bash "$ROOT/collect_env.sh" > "$LOG/env.txt" 2>&1 || true

run_case () {
  local CASE="$1" ; local ARGS_FILE="$2" ; local PORT="$3"
  # 用 PyTorch Profiler（SGLang 会把 chrome trace 写到这个路径；若你们没做内置，也可先跳过）
  export SGLANG_ENABLE_TORCH_PROFILER=1
  export TORCH_PROFILER_OUTPUT="${TRACE}/${CASE}_chrome_trace.json"

  # 启服
  python3 -m sglang.launch_server \
    $(cat "$ROOT/server_args_base.txt") \
    $(cat "$ARGS_FILE") \
    --host 127.0.0.1 --port "${PORT}" \
    > "${LOG}/${CASE}_server.log" 2>&1 &
  SERVER_PID=$!

  # 等待就绪（最多 120s）
  python3 - <<'PY' "${PORT}"
import time, socket, sys
port=int(sys.argv[1])
for _ in range(120):
    try:
        s=socket.socket(); s.settimeout(0.2); s.connect(("127.0.0.1",port)); s.close()
        print("server ready"); break
    except Exception: time.sleep(1)
else: raise SystemExit("server not ready")
PY

  # 压测（触发 prefill+decode）
  python3 "$ROOT/client_benchmark.py" --host 127.0.0.1 --port "${PORT}" \
    > "${LOG}/${CASE}_client.log" 2>&1 || true

  # 收尾
  kill ${SERVER_PID} || true
  wait ${SERVER_PID} || true
}

# A: Decode + CUDA Graphs（问题态）
run_case "CASE_A_decode_graphs" "$ROOT/server_args_graphs_on.txt" 8000

# B: 关闭 CUDA Graphs（对照）
run_case "CASE_B_no_graphs" "$ROOT/server_args_graphs_off.txt" 8001

echo "Done. Logs in $LOG ; traces in $TRACE" 