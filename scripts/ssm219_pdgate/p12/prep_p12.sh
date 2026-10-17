#!/bin/bash
# docs/222 §20: the P12 job directory (CMH / LHR login): bash prep_p12.sh <package fp> [tag]   (tag: a new dir pkg-<fp>-<tag>)
#   $A/ws2/p12/pkg-<fp>/ <- p12_hjob.sh, p12_runcheck.py (this upload) + four holder inbox wrappers pinned to that dir
#   (the holder does not pass the submitter's env).  Prints each wrapper with its git blob and header.
set -euo pipefail
FP=$1; TAG=${2:-}
W=/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
A=$W/results/ssm-kernel-219; R=$W/results/p3pd-20261017
T=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
D=$A/ws2/p12/pkg-$FP${TAG:+-$TAG}
[[ -f $R/inputs/pkg-$FP/manifest.json ]] || { echo "no package $FP"; exit 2; }
[[ ! -e $D ]] || { echo "$D exists (never recreated under the same name)"; exit 2; }
mkdir -p $D
cp $T/p12_hjob.sh $T/p12_runcheck.py $D/
wrap() {  # wrap <kind> <minutes> <what>
  cat > $D/hjob_p12_$1.sh <<EOF
# HOLDER_GPUS=4
# HOLDER_EXCL=1
# HOLDER_STAGE=0
# HOLDER_MINUTES=$2
# holder inbox job: docs/222 §20 P12 $1 (WS-2; package pkg-$FP; pd1n P GPU0-1 / D GPU2-3): $3
# queue: bash \$H/scripts/qsubmit.sh $D/hjob_p12_$1.sh --gpus 4 --excl --minutes $2
P12_JOBDIR=$D exec bash $D/p12_hjob.sh $1 $FP
EOF
}
wrap positive 45 "live received-byte gate (replay on P, proof dirs both roles; client positive + torch.equal checker)"
wrap negative 75 "live negative gates (event, abort restarts; client 1 failure + 5 healthy each; checker --negative x2)"
wrap micro 100 "C+ / PC+ arm 1 / PC+ arm 0, AgentX C8 300 s each, no proof keys"
wrap ab 85 "PC+ env_p12 0 / 1, AgentX C8 600 s each, request-paired, no proof keys"
cd $D
for f in *; do printf '%s %s\n' "$(git hash-object $f 2>/dev/null || python3 -c "import hashlib,sys;d=open(sys.argv[1],'rb').read();print(hashlib.sha1(b'blob %d\0'%len(d)+d).hexdigest())" $f)" "$D/$f"; done
