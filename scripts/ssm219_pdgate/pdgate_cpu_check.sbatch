#!/bin/bash
#SBATCH --job-name=ws2-pdgate-cpu
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=cpu
#SBATCH --qos=cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
# docs/222 §15 PD gate CPU self-test, zero GPU: the units (probe / dispatch / summary) inside the serving image with the real
# sglang of (1) the client tree source-7a and (2) the C+ window's server capsule (dill round trip by reference on both).
#   T=<uploaded tools dir> O=<result dir> sbatch -o $O/cpu-%j.log pdgate_cpu_check.sbatch
set -uo pipefail
: "${T:?T}" "${O:?O}"
W=/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
A=$W/results/ssm-kernel-219
IMG=$W/containers/lmsysorg+sglang+nightly-dev-cu13-20260913-fa663e72-arm64-enroot.sqsh
ACC=/work/results/ssm-kernel-219/gate/STACK-bf16-4/tools/acc_audit_kl.py   # the agg gate's copy (acc-audit 481041ee0)
GS=/work/results/ssm-kernel-219/tools/gate_summary.py
echo "START $(date -u +%FT%TZ) job=$SLURM_JOB_ID node=$(hostname)"
rc=0
for tree in /work/results/bf16-regress-20261013/source-7a/python /work/results/ssm219-pd-20261016/inputs/src-ssm219-int-1fde0db52ee/python; do
  srun -N1 -n1 --container-image=$IMG --container-mounts="$W:/work:ro,$T:/tools:ro,$O:/o" --no-container-mount-home \
    --container-workdir=/o bash -c "export PYTHONPATH=$tree:/work/pyuser/lib/python3.12/site-packages PYTHONDONTWRITEBYTECODE=1
      /opt/sglang/bin/python /tools/pdgate_selftest.py units --acc-audit $ACC --gate-summary $GS" > $O/units-$(basename $(dirname $tree)).log 2>&1
  r=$?; echo "units tree=$tree rc=$r $(tail -1 $O/units-$(basename $(dirname $tree)).log)"; [ $r = 0 ] || rc=1
done
echo "END rc=$rc $(date -u +%FT%TZ)"; exit $rc
