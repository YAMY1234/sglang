#!/bin/bash
#SBATCH --job-name=ws2-pdgate-arg
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=cpu
#SBATCH --qos=cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
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
# one container start per distinct mount table (all arms of a gate usually share one): each start of the 31.7 GB image on
# a CPU node costs 8-13 min, four sequential starts ran into the old 40 min limit (4566155 / 4566156, 10-17)
cat > $OUT/argcheck-inner.sh <<'INNER'
#!/bin/bash
# runs inside the image: argcheck-inner.sh <arm> [<arm> ...]; every arm / role in its own subshell and log
rc=0
for arm in "$@"; do for role in P D; do
  (
    set -a; . /out/g-$arm-$role.env; set +a
    ARM=$arm ROLE=$role /opt/sglang/bin/python - <<'PY'
import argparse, hashlib, json, os, shlex
arm, role = os.environ["ARM"], os.environ["ROLE"]
spec = json.load(open(f"/out/g-{arm}-{role}.json"))
import sglang
got = os.path.dirname(sglang.__file__) + "/"
print("PDGATE_SGLANG", got, "expect", spec["sglang_expect"])
assert got == spec["sglang_expect"], "sglang resolves to the wrong tree"
from sglang.srt.server_args import ServerArgs
argv = shlex.split(open(f"/out/g-{arm}-{role}.argv").read())
i = argv.index("sglang.launch_server")
p = argparse.ArgumentParser()
ServerArgs.add_cli_args(p)
ns = p.parse_args(argv[i + 1:])
print("ARGV-OK", len(argv[i + 1:]), "tokens; port", ns.port, "mode", ns.disaggregation_mode)
for hf in json.load(open("/out/plan.json"))["arms"][arm].get("helper_files") or []:
    h = hashlib.sha256(open(hf["cont"], "rb").read()).hexdigest()
    print("HELPER-FILE", hf["cont"].rsplit("/", 1)[-1], "ok" if h == hf["sha256"] else "MISMATCH", hf["git_blob"][:12])
    assert h == hf["sha256"], hf
PY
  ) > /out/argcheck-$arm-$role.log 2>&1
  r=$?; echo "$arm/$role rc=$r $(grep -E 'PDGATE_SGLANG|ARGV-OK|HELPER-FILE|Error|error:' /out/argcheck-$arm-$role.log | tail -3 | tr '\n' ' ')"
  [ $r = 0 ] || rc=1
done; done
exit $rc
INNER
rc=0
groups=$(python3 - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
g = {}
for arm in json.load(open(f"{out}/plan.json"))["phases"]:
    g.setdefault(open(f"{out}/g-{arm}.mounts").read(), []).append(arm)
print("\n".join(",".join(v) for v in g.values()))
PY
)
for grp in $groups; do
  first=${grp%%,*}
  srun -N1 -n1 --container-image=$IMG --container-mounts="$(sed "s|@MODEL@|$MODEL|" $OUT/g-$first.mounts)" \
    --no-container-mount-home --container-workdir=/out bash /out/argcheck-inner.sh ${grp//,/ } || rc=1
done
echo "END rc=$rc $(date -u +%FT%TZ)"; exit $rc
