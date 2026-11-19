#!/bin/bash
# Per-node profiling sidecar for run_arm.sh. Inside the container, next to this node's server.
# Schedule file lines: PHASE NSYS_DELAY_S NSYS_DUR_S PYSPY_DELAY_S PYSPY_DUR_S  (delays from the phase's start epoch in
# RUN/s*/phases.tsv; 0 duration = skip). Both nodes compute the same absolute times from the same epoch.
# nsys: `nsys start/stop --session=$NSYS_SESSION` on the session that launch_server.py opened with `nsys launch`.
# py-spy (optional, $PYSPY, default `py-spy` on PATH): repeated `dump --nonblocking` of this node's first scheduler (PP0/PP4 or TP0/TP4), MainThread only
# (py-spy record hangs on these schedulers).
# usage: nsys_window.sh RUN_DIR SCHEDULE_FILE
set -u
R=${1:?} SCHED=${2:?}
H=$(hostname -s)
NSYS=${NSYS:?} SES=${NSYS_SESSION:?}
PS=${PYSPY:-py-spy}
O=$R/prof; mkdir -p "$O/nsys"
T=$O/side-$H.tsv
note() { echo -e "$(date +%s.%N)\t$*" >> "$T"; }
note "begin sched=$SCHED"
{ $NSYS --version; echo ---; $NSYS start --help; echo ---; $NSYS export --help | head -80; } > "$O/nsys-help-$H.txt" 2>&1

sleep_until() { local now; now=$(date +%s.%N); local d; d=$(python3 -I -c "print(max(0.0, $1 - $now))"); sleep "$d"; }

nsys_window() {  # $1 phase, $2 abs start, $3 dur
  local f=$O/nsys/$1-$H rc
  sleep_until "$2"
  note "nsys_start_req $1"
  timeout 120 $NSYS start --session="$SES" -o "$f" --force-overwrite=true --sample=process-tree \
    --sampling-period=16000000 --cpuctxsw=process-tree > "$O/nsys-start-$1-$H.log" 2>&1
  rc=$?
  if [ $rc != 0 ]; then
    note "nsys_start_full_rc $rc; retry minimal"
    timeout 120 $NSYS start --session="$SES" -o "$f" --force-overwrite=true --sample=none --cpuctxsw=none \
      >> "$O/nsys-start-$1-$H.log" 2>&1
    rc=$?
  fi
  note "nsys_start_ret $1 rc=$rc"
  sleep_until "$(python3 -I -c "print($2 + $3)")"
  note "nsys_stop_req $1"
  timeout 600 $NSYS stop --session="$SES" > "$O/nsys-stop-$1-$H.log" 2>&1
  note "nsys_stop_ret $1 rc=$?"
}

pyspy_window() {  # $1 phase, $2 abs start, $3 dur
  local d=$O/pyspy-$1-$H P= p i=0 stop t
  mkdir -p "$d"
  command -v "$PS" > /dev/null 2>&1 || { note "pyspy_not_found $PS"; return; }
  sleep_until "$2"
  for p in $(pgrep -f "sglang::sched"); do
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q -E "(PP|TP)[04]([^0-9]|$)" && { P=$p; break; }
  done
  pgrep -af "sglang::sched" > "$d/pgrep.txt" 2>&1
  [ -n "$P" ] || { note "pyspy_no_target $1"; return; }
  note "pyspy_start $1 pid=$P $(tr '\0' ' ' < /proc/$P/cmdline)"
  stop=$(( $(date +%s) + $3 ))
  while [ "$(date +%s)" -lt "$stop" ]; do
    t=$(date +%s.%N)
    timeout 10 $PS dump --pid "$P" --nonblocking 2>&1 | awk -v t="$t" 'NR==1{print "# t=" t} /^Thread /{m=($0 ~ /MainThread/)} m' > "$d/d$(printf %05d $i).txt"
    i=$((i + 1)); sleep 0.2
  done
  note "pyspy_end $1 dumps=$i"
}

while read -r ph nd ndur pd pdur <&3; do
  case "$ph" in ''|\#*) continue ;; esac
  note "wait_phase $ph"
  t0=
  for _ in $(seq 1 1440); do  # up to 2 h
    t0=$(grep -h -P "^$ph\tstart\t" "$R"/s*/phases.tsv 2>/dev/null | head -1 | cut -f3)
    [ -n "$t0" ] && break
    [ -f "$R/ALLDONE" ] && break
    sleep 5
  done
  [ -n "$t0" ] || { note "phase_not_seen $ph"; continue; }
  note "phase_seen $ph t0=$t0"
  [ "${pdur:-0}" -gt 0 ] && { pyspy_window "$ph" $((t0 + pd)) "$pdur" & }
  [ "${ndur:-0}" -gt 0 ] && nsys_window "$ph" $((t0 + nd)) "$ndur"
  wait
done 3< "$SCHED"
note "end"
touch "$O/SIDE_DONE-$H"
