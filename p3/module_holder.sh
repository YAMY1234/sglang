# HOLDER_GPUS=2
# HOLDER_EXCL=0
# HOLDER_STAGE=0
# HOLDER_MINUTES=10
# Runs the same-state numerical gate and publication protocol on real P dumps.
set -Eeuo pipefail
source /lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911/results/ssm-kernel-219/holder/scripts/hlib.sh
: "${P3_TREE:?}" "${P3_DUMP_DIR:?}"
P3_STAGE=${P3_STAGE:-p3}
O=$W/results/ssm-kernel-219/p3-own/module-$P3_STAGE-h$HOLDER_JOB-$(basename "$RUN_DIR")
mkdir -p "$O"
SRC=${P3_TREE/#$W/\/work}
DUMPS=${P3_DUMP_DIR/#$W/\/work}
DST=${O/#$W/\/work}
echo "STEP START module stage=$P3_STAGE holder=$HOLDER_JOB cvd=$HOLDER_CVD $(date -u +%FT%TZ) out=$O"
H_MEM=96G H_CPUS=16 h_srun "$HOLDER_CVD" "
set -Eeuo pipefail
export PYTHONPATH=$SRC/python:/work/pyuser/lib/python3.12/site-packages
export PYTHONUNBUFFERED=1 SGLANG_GDN_K31_EIGH=jacobi
export TORCH_EXTENSIONS_DIR=/tmp/p3-module-extensions TRITON_CACHE_DIR=/tmp/p3-module-triton MAX_JOBS=8
python -c 'import sglang; print(\"P3_MODULE_SOURCE\", sglang.__file__); assert sglang.__file__.startswith(\"$SRC/\")'
module_rc=0
python $SRC/p3/module_gate.py --stage $P3_STAGE --dumps '$DUMPS/k31dump-r*-*.pt' --out $DST/module.json > $DST/module.log 2>&1 || module_rc=\$?
publication_rc=0
python $SRC/p3/publication_gate.py --dumps '$DUMPS/k31dump-r*-*.pt' --out $DST/publication.json > $DST/publication.log 2>&1 || publication_rc=\$?
tail -1 $DST/module.log
tail -1 $DST/publication.log
exit \$((module_rc || publication_rc))" 2>&1 | tee "$O/driver.log"
echo "STEP END module stage=$P3_STAGE holder=$HOLDER_JOB $(date -u +%FT%TZ) out=$O"
