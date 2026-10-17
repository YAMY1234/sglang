#!/bin/bash
#SBATCH --job-name=ws2-pdgate-arg
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=cpu
#SBATCH --qos=cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:40:00
# docs/222 §15 PD gate launch check, zero GPU: for every arm / role of a prepared gate, the serving image with the gate's
# own mounts (+ Lustre model) and env: (1) `sglang` must resolve to the declared tree (sglang_expect), (2) the fork's
# ServerArgs parser must accept the gate's full argv.  Catches what killed 4560175 / 4560392 (stock sglang shadowing).
#   OUT=<prepared gate> sbatch -o $OUT/argcheck-%j.log pdgate_argcheck.sbatch
set -uo pipefail
: "${OUT:?OUT}"
W=/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911
IMG=$W/containers/lmsysorg+sglang+nightly-dev-cu13-20260913-fa663e72-arm64-enroot.sqsh
MODEL=$W/results/bf16-regress-20261013/staging/model
echo "START $(date -u +%FT%TZ) job=$SLURM_JOB_ID node=$(hostname) out=$OUT"
rc=0
for arm in $(python3 -c "import json;print(' '.join(json.load(open('$OUT/plan.json'))['phases']))"); do
  for role in P D; do
    srun -N1 -n1 --container-image=$IMG --container-mounts="$(sed "s|@MODEL@|$MODEL|" $OUT/g-$arm.mounts)" \
      --no-container-mount-home --container-workdir=/out bash -c "
        set -a; . /out/g-$arm-$role.env; set +a
        want=\$(python3 -c 'import json;print(json.load(open(\"/out/g-$arm-$role.json\"))[\"sglang_expect\"])')
        /opt/sglang/bin/python - <<'PY'
import json, os, shlex, sys
spec = json.load(open('/out/g-$arm-$role.json'))
import sglang
got = os.path.dirname(sglang.__file__) + '/'
print('PDGATE_SGLANG', got, 'expect', spec['sglang_expect'])
assert got == spec['sglang_expect'], 'sglang resolves to the wrong tree'
import argparse
from sglang.srt.server_args import ServerArgs
argv = shlex.split(open('/out/g-$arm-$role.argv').read())
i = argv.index('sglang.launch_server')
p = argparse.ArgumentParser()
ServerArgs.add_cli_args(p)
ns = p.parse_args(argv[i + 1:])
print('ARGV-OK', len(argv[i + 1:]), 'tokens; port', ns.port, 'mode', ns.disaggregation_mode)
PY" > $OUT/argcheck-$arm-$role.log 2>&1
    r=$?; echo "$arm/$role rc=$r $(grep -E 'PDGATE_SGLANG|ARGV-OK|Error|error:' $OUT/argcheck-$arm-$role.log | tail -2 | tr '\n' ' ')"
    [ $r = 0 ] || rc=1
  done
done
echo "END rc=$rc $(date -u +%FT%TZ)"; exit $rc
