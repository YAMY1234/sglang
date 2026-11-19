#!/bin/bash
# Prefill-only benchmark driver: one Kimi-K3.1-Pro PD-prefill server over 2 nodes (8 GPUs), no decode side.
# pfab_client.py sends the fake bootstrap host, so the server uses its fake KV sender (transfer completes at once).
# Run it once per node, inside the container, at the same time on both nodes (e.g. `srun -N2 --ntasks-per-node=1`).
# Node 0 leads (rank 0 server + client); node 1 follows through marker files in RUN_DIR, which must be on a
# filesystem both nodes see. Nodes must be exclusive: leftover GPU processes are killed between server boots.
#
# usage: run_arm.sh RUN_DIR PLAN_FILE
# env:   MODEL_PATH, DRAFT_MODEL_PATH        local snapshots of the two checkpoints (required)
#        SGLANG_REPO                         this repo; its python/ is put first on PYTHONPATH (optional)
#        NODE_RANK                           default: $SLURM_NODEID, else 0
#        LEADER_IP                           default: node 0's resolved hostname
#        ARMS_JSON                           default: arms.json next to this script
#        LOCAL_TMP                           node-local scratch for JIT caches (default /tmp); cleaned at the end
#        RUST_EXT_CACHE                      prebuilt Rust TreeCore cache (README "Software"); copied in before boot
#        BUDGET_S                            seconds this run may use (default 6600); later plan steps are skipped
#        NSYS, NSYS_SCHED                    profiling (README "Profiling")
# Plan lines: boot ARM | restart ARM | probe WL N | flush | warm WL
#             | closed WL CONC LEAD WINDOW TAG | poisson WL RATE LEAD WINDOW TAG
set -u
R=${1:?run dir} PLAN=${2:?plan file}
BIN=$(cd "$(dirname "$0")" && pwd)
ARMS=${ARMS_JSON:-$BIN/arms.json}
: "${MODEL_PATH:?set MODEL_PATH}" "${DRAFT_MODEL_PATH:?set DRAFT_MODEL_PATH}"
export MODEL_PATH DRAFT_MODEL_PATH NNODES=2
NODE=${NODE_RANK:-${SLURM_NODEID:-0}}
H=${NODE_NAME:-$(hostname -s)}
T0=$(date +%s)
DEADLINE=$((T0 + ${BUDGET_S:-6600}))
PORT=${PORT:-30000}
TAG=$(basename "$R")
log() { echo "[$(date -u +%FT%TZ) $H] $*"; }

# fresh node-local caches per run (no JIT / autotune cache carried over from earlier runs)
C=${LOCAL_TMP:-/tmp}/pfab-$TAG
mkdir -p "$C"
export HOME=$C/home PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=$C/pycache
export SGLANG_CACHE_DIR=$C/sglang TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor CUDA_CACHE_PATH=$C/nv \
  TILELANG_CACHE_DIR=$C/tilelang FLASHINFER_WORKSPACE_BASE=$C/fi TVM_FFI_CACHE_DIR=$C/tvm-ffi XDG_CACHE_HOME=$C/xdg \
  SGLANG_JIT_CACHE_DIR=$C/jit SGLANG_DG_CACHE_DIR=$C/dg SGLANG_CUTE_AOT_CACHE_DIR=$C/cute_aot
[ -n "${SGLANG_REPO:-}" ] && export PYTHONPATH=$SGLANG_REPO/python${PYTHONPATH:+:$PYTHONPATH}
# the Rust TreeCore built once from this tree's rust/ (else the first boot builds it with cargo on every node)
[ -n "${RUST_EXT_CACHE:-}" ] && mkdir -p "$SGLANG_CACHE_DIR" && cp -r "$RUST_EXT_CACHE/rust_extensions" "$SGLANG_CACHE_DIR/"

mkdir -p "$R"
{
  date -u +%FT%TZ; echo "NODE_RANK=$NODE CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  nvidia-smi --query-gpu=index,name,uuid,memory.total,clocks.max.sm --format=csv
  timeout 300 python3 -c 'import sglang, deep_gemm, torch; print("sglang", sglang.__file__); print("deep_gemm", deep_gemm.__file__, getattr(deep_gemm, "__version__", "?")); print("torch", torch.__version__)' 2>&1 | tail -3
  [ -n "${SGLANG_REPO:-}" ] && git -C "$SGLANG_REPO" rev-parse HEAD 2>&1
  [ -n "${RUST_EXT_CACHE:-}" ] && find "$SGLANG_CACHE_DIR/rust_extensions" -name '*.so' -exec sha256sum {} + 2>&1
} > "$R/facts-$H.txt" 2>&1

# profiling: server under `nsys launch` (NSYS), per-node sidecar for nsys start/stop and py-spy (NSYS_SCHED)
ulimit -n "$(ulimit -Hn)" 2>/dev/null
SIDE=
if [ -n "${NSYS:-}" ]; then
  export NSYS NSYS_SESSION=pf-$H
  [ -n "${NSYS_SCHED:-}" ] && { bash "$BIN/profile/nsys_window.sh" "$R" "$NSYS_SCHED" > "$R/side-$H.log" 2>&1 & SIDE=$!; }
fi
finish_prof() {  # stop the sidecar if still waiting, then export this node's reports (server already stopped)
  [ -n "$SIDE" ] && { kill "$SIDE" 2>/dev/null; wait "$SIDE" 2>/dev/null; }
  [ -n "${NSYS:-}" ] || return 0
  for f in "$R"/prof/nsys/*-"$H".nsys-rep; do
    [ -f "$f" ] || continue
    log "export $f"
    timeout 2400 "$NSYS" export --type=sqlite --include-blobs=true --force-overwrite=true -o "${f%.nsys-rep}.sqlite" "$f" \
      > "${f%.nsys-rep}.export.log" 2>&1 || timeout 2400 "$NSYS" export --type=sqlite --force-overwrite=true \
      -o "${f%.nsys-rep}.sqlite" "$f" >> "${f%.nsys-rep}.export.log" 2>&1
    log "export rc=$?"
  done
}

SPID= GPID=
start_server() {  # $1 step dir, $2 arm, $3 dist addr
  setsid python3 -I "${PFAB_LAUNCH:-$BIN/launch_server.py}" "$ARMS" "$2" "$NODE" "$3" "$PORT" > "$1/server-$H.log" 2>&1 < /dev/null &
  SPID=$!
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 2 > "$1/gpu-$H.csv" 2>&1 &
  GPID=$!
  log "server $2 started pid=$SPID addr=$3"
}

stop_server() {
  local p=$SPID left=
  [ -n "${GPID:-}" ] && kill "$GPID" 2>/dev/null; GPID=
  [ -n "$p" ] || return 0
  kill -TERM -- "-$p" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$p" 2>/dev/null || break; sleep 2; done
  kill -KILL -- "-$p" 2>/dev/null
  SPID=
  for _ in $(seq 1 30); do
    left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | grep -v '^$' | tr '\n' ' ')
    [ -z "$left" ] && break
    sleep 3
  done
  if [ -n "$left" ]; then log "leftover GPU pids: $left"; for x in $left; do [ "$x" != $$ ] && kill -KILL "$x" 2>/dev/null; done; sleep 5; fi
  log "server stopped"
}
trap 'stop_server; exit 1' TERM INT

extract() {  # rank-0 batch lines and the engagement / error lines, small enough to copy around
  grep -h 'Prefill batch' "$1/server-$H.log" > "$1/prefill-$H.log" 2>/dev/null
  grep -h -i -E 'LAUNCH|server_args=|Load weight end|KV Cache is allocated|Mamba Cache is allocated|max_total_num_tokens|available_gpu_mem|HiCache|Tree ?Core|path cap|SP collective|attention-residual carry|rows split across|fused gather|fused KV|prims_ts|KDA prefill kernel|out of memory|Traceback|Error|error:|fired up' \
    "$1/server-$H.log" 2>/dev/null | cut -c1-4000 | head -800 > "$1/server-key-$H.log"
}

# ---------------- follower (node 1) ----------------
if [ "$NODE" != 0 ]; then
  touch "$R/READY-$H"
  i=1
  while :; do
    S="$R/s$i"
    if [ -f "$S/START" ]; then
      read -r arm addr < "$S/START"
      start_server "$S" "$arm" "$addr"
      while [ ! -f "$S/STOP" ]; do
        if ! kill -0 "$SPID" 2>/dev/null; then log "server died"; touch "$S/FAILED-$H"; break; fi
        sleep 5
      done
      stop_server; extract "$S"; touch "$S/STOPPED-$H"
      i=$((i + 1)); continue
    fi
    [ -f "$R/ALLDONE" ] && break
    sleep 5
  done
  finish_prof; rm -rf "$C"; log "follower done"; exit 0
fi

# ---------------- leader (node 0) ----------------
ADDR_IP=${LEADER_IP:-$(python3 -I -c 'import socket; print(socket.gethostbyname(socket.gethostname()))')}
for _ in $(seq 1 120); do ls "$R"/READY-* > /dev/null 2>&1 && break; sleep 5; done
log "leader ip=$ADDR_IP follower: $(ls "$R" | grep READY- | tr '\n' ' ')"
cp "$PLAN" "$R/plan.txt"; cp "$ARMS" "$R/arms.json"

wl() {  # workload -> client args
  case $1 in
    W128_90) echo "--groups 16 --prefix 117760 --new 13312" ;;  # 131,072-token requests, 89.8% shared prefix
    W32_0)   echo "--groups 0 --prefix 0 --new 32768" ;;        # 32,768-token requests, no shared prefix
    *) echo "unknown workload $1" >&2; return 1 ;;
  esac
}
phase() {  # $1 name, rest: client args; records epoch and local time so the server log can be windowed
  local ph=$1; shift
  local s; s=$(date +%s)
  curl -s -m 20 "http://127.0.0.1:$PORT/metrics" > "$S/metrics-$ph-start.txt" 2>/dev/null
  echo -e "$ph\tstart\t$s\t$(date '+%F %T')" >> "$S/phases.tsv"
  timeout $(( DEADLINE - s > 60 ? DEADLINE - s : 60 )) python3 -I "$BIN/pfab_client.py" --url "http://127.0.0.1:$PORT" \
    --out "$S/$ph.jsonl" "$@" > "$S/$ph.log" 2>&1 < /dev/null
  local rc=$?
  echo -e "$ph\tend\t$(date +%s)\t$(date '+%F %T')\trc=$rc" >> "$S/phases.tsv"
  curl -s -m 20 "http://127.0.0.1:$PORT/metrics" > "$S/metrics-$ph-end.txt" 2>/dev/null
  log "phase $ph rc=$rc $(tail -1 "$S/$ph.log" | cut -c1-300)"
  return $rc
}
end_server() {
  [ -n "${S:-}" ] || return 0
  touch "$S/STOP"; stop_server; extract "$S"
  for _ in $(seq 1 60); do ls "$S"/STOPPED-* > /dev/null 2>&1 && break; sleep 5; done
}

i=0 S= alive=0
while read -r op x1 x2 x3 x4 x5 <&3; do
  case "$op" in ''|\#*) continue ;; esac
  left=$(( DEADLINE - $(date +%s) ))
  case "$op" in
    boot|restart)
      end_server; alive=0
      [ "$left" -gt 900 ] || { log "SKIP $op $x1: $left s left"; echo "SKIP $op $x1 left=$left" >> "$R/skipped.txt"; continue; }
      i=$((i + 1)); S="$R/s$i"; mkdir -p "$S"
      echo "$x1 $ADDR_IP:$((5757 + i))" > "$S/START"
      t=$(date +%s); start_server "$S" "$x1" "$ADDR_IP:$((5757 + i))"
      for _ in $(seq 1 180); do
        if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then alive=1; break; fi
        kill -0 "$SPID" 2>/dev/null || break
        ls "$S"/FAILED-* > /dev/null 2>&1 && break
        sleep 10
      done
      echo -e "variant\t$x1\nboot_s\t$(( $(date +%s) - t ))\nalive\t$alive" > "$S/boot.tsv"
      log "step $i $x1 alive=$alive after $(( $(date +%s) - t )) s"
      [ $alive = 1 ] || { echo boot_failed > "$S/STATUS"; end_server; } ;;
    *)
      [ $alive = 1 ] || { log "skip $op (no live server)"; continue; }
      kill -0 "$SPID" 2>/dev/null || { log "server died before $op"; echo server_died > "$S/STATUS"; alive=0; end_server; continue; }
      case "$op" in
        flush) r=$(curl -s -m 120 -X POST "http://127.0.0.1:$PORT/flush_cache"); log "flush: $r"; echo -e "flush\t$(date +%s)\t$r" >> "$S/phases.tsv" ;;
        probe) phase "probe-$x1" --mode probe $(wl "$x1") --requests "$x2" ;;
        warm) phase "warm-$x1-$(date +%s)" --mode warm $(wl "$x1") --concurrency 16 ;;
        closed|poisson)
          need=$(( ${x3%.*} + ${x4%.*} + 120 ))
          [ "$left" -gt "$need" ] || { log "SKIP $x5: $left s left < $need"; echo "SKIP $x5 left=$left" >> "$R/skipped.txt"; continue; }
          if [ "$op" = closed ]; then phase "$x5" --mode closed $(wl "$x1") --concurrency "$x2" --lead "$x3" --window "$x4"
          else phase "$x5" --mode poisson $(wl "$x1") --rate "$x2" --lead "$x3" --window "$x4"; fi ;;
        *) log "unknown op $op" ;;
      esac ;;
  esac
done 3< "$PLAN"
end_server
touch "$R/ALLDONE"
finish_prof
rm -rf "$C"
log "leader done"
