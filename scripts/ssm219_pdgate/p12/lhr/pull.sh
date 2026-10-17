#!/bin/bash
# docs/222 §20 LHR port, on the LHR login (detached): pull every `copy` item of items.txt from CMH by tar over ssh
# (LHR's own key, authorized at CMH), N streams in parallel; `mkdir` items as empty dirs; never overwrite an existing path.
#   nohup setsid bash pull.sh <items.txt> <log dir> [streams] > /dev/null 2>&1 &
set -uo pipefail
ITEMS=$1; LOG=$2; N=${3:-4}
W=/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
CMH="ssh -o BatchMode=yes -o ServerAliveInterval=30 -i $HOME/.ssh/id_ed25519 yangminl@aws-cmh-slurm-1-login-01.nvidia.com"
mkdir -p $LOG
pull() {
  local p=$1 k=${1//\//__}
  if [[ -e $W/$p || -L $W/$p ]]; then echo "SKIP-EXISTS $p" >> $LOG/pull.log; return; fi
  mkdir -p "$W/$(dirname $p)"
  local t0=$(date +%s)
  $CMH "tar cf - -C $W $p" 2> $LOG/$k.ssh.err | tar xf - -C $W 2> $LOG/$k.tar.err
  local rc=("${PIPESTATUS[@]}")
  echo "PULLED $p ssh_rc=${rc[0]} tar_rc=${rc[1]} seconds=$(( $(date +%s) - t0 ))" >> $LOG/pull.log
}
export -f pull; export W CMH LOG
echo "START $(date -u +%FT%TZ)" >> $LOG/pull.log
grep -v '^#' $ITEMS | awk '$2=="mkdir"{print $1}' | while read p; do mkdir -p $W/$p && echo "MKDIR $p" >> $LOG/pull.log; done
grep -v '^#' $ITEMS | awk '$2=="copy"{print $1}' | xargs -P $N -I{} bash -c 'pull {}'
echo "END $(date -u +%FT%TZ)" >> $LOG/pull.log
