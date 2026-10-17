# HOLDER_GPUS=2
# HOLDER_EXCL=0
# HOLDER_STAGE=0
# HOLDER_MINUTES=5
set -Eeuo pipefail
source /lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911/results/ssm-kernel-219/holder/scripts/hlib.sh
: "${P3_TREE:?}" "${P3_GATE_SCRIPT:?}"
O=$W/results/ssm-kernel-219/p3-own/publication-h$HOLDER_JOB-$(basename "$RUN_DIR")
mkdir -p "$O"
SRC=${P3_TREE/#$W/\/work}
SCRIPT=${P3_GATE_SCRIPT/#$W/\/work}
DST=${O/#$W/\/work}
echo "STEP START publication holder=$HOLDER_JOB cvd=$HOLDER_CVD $(date -u +%FT%TZ) out=$O"
H_MEM=96G H_CPUS=16 h_srun "$HOLDER_CVD" "
set -Eeuo pipefail
export PYTHONPATH=$SRC/python:/work/pyuser/lib/python3.12/site-packages
export PYTHONUNBUFFERED=1 SGLANG_GDN_K31_EIGH=jacobi
export TORCH_EXTENSIONS_DIR=/tmp/p3-pub-extensions TRITON_CACHE_DIR=/tmp/p3-pub-triton MAX_JOBS=8
python -c 'import sglang; print(\"P3_MODULE_SOURCE\", sglang.__file__); assert sglang.__file__.startswith(\"$SRC/\")'
python $SCRIPT --synthetic --out $DST/result.json > $DST/worker.log 2>&1
tail -3 $DST/worker.log" 2>&1 | tee "$O/driver.log"
echo "STEP END publication holder=$HOLDER_JOB $(date -u +%FT%TZ) out=$O"
