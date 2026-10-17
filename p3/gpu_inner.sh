#!/bin/bash
# Uses the PD receipt mounts plus explicit /p3/src and /p3out mounts.
set -Eeuo pipefail
P3_SRC=/p3/src
P3_OUT=/p3out
P3_STAGE=${P3_STAGE:-p5}
export P3_SRC P3_OUT P3_STAGE
ARM=Cp
[[ $P3_STAGE == p3 ]] && ARM=PCp
export P3_LAUNCH_JSON=/work/results/fp32-pd-20261015/runs/j4529207-$ARM/workers/P0/actual-launch.json
export P3_HELPER_DIR=/work/results/remeasure-v6-fivearm-tp1-20261004/inputs/helper-runtime/helpers
python3 "$P3_SRC/p3/frozen_env.py" --launch "$P3_LAUNCH_JSON" --src "$P3_SRC" --out "$P3_OUT" \
    --helper-dir "$P3_HELPER_DIR" > "$P3_OUT/env.sh"
source "$P3_OUT/env.sh"
export PYTHONPATH="$P3_SRC/p3:$PYTHONPATH"
export SGLANG_GDN_K31_STATE_DUMP_DIR=$P3_OUT/dumps
export SGLANG_GDN_K31_STATE_DUMP_MAX=2
mkdir -p "$SGLANG_GDN_K31_STATE_DUMP_DIR"
python3 - <<'PY'
import argparse, json, os
import sglang
import torch
from p3_factory import frozen_argv
from sglang.srt.server_args import ServerArgs
got = os.path.realpath(sglang.__file__)
assert got == '/p3/src/python/sglang/__init__.py', got
print('P3_SGLANG_PATH_PASS', got, flush=True)
assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 2
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
print('P3_GPU_SLOTS_PASS', os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.device_count(), flush=True)
p = argparse.ArgumentParser()
ServerArgs.add_cli_args(p)
argv = frozen_argv(json.load(open(os.environ['P3_LAUNCH_JSON']))['command'], 30795)
p.parse_args(argv)
print('P3_FULL_PD_ARGCHECK_PASS', len(argv), flush=True)
PY
if [[ ${P3_MODE:-micro} == checkpoint ]]; then
    source "$P3_SRC/p3/env.sh"
    export SGLANG_GDN_K31_AB_DUAL=1 SGLANG_GDN_K31_AB_ENV=SGLANG_GDN_K31_FUSED,SGLANG_GDN_K31_CHOLQR_MIXED
    export SGLANG_GDN_PD_SHALLOW_FULLN=1 TWINSTAR_PD_EMITTER_GRAPH=1
    checkpoint_rc=0
    python -m torch.distributed.run --standalone --nproc-per-node=2 "$P3_SRC/p3/checkpoint_gate.py" \
        > "$P3_OUT/worker.log" 2>&1 || checkpoint_rc=$?
    module_rc=0
    python "$P3_SRC/p3/module_gate.py" --stage p3 --dumps "$P3_OUT/dumps/k31dump-r*-*.pt" \
        --out "$P3_OUT/module.json" > "$P3_OUT/module.log" 2>&1 || module_rc=$?
    publication_rc=0
    python "$P3_SRC/p3/publication_gate.py" --dumps "$P3_OUT/dumps/k31dump-r*-*.pt" \
        --out "$P3_OUT/publication.json" > "$P3_OUT/publication.log" 2>&1 || publication_rc=$?
    printf 'P3_CHECKPOINT_MODULE_RC checkpoint=%s module=%s publication=%s\n' \
        "$checkpoint_rc" "$module_rc" "$publication_rc"
    (( checkpoint_rc == 0 && module_rc == 0 && publication_rc == 0 )) || exit 1
elif [[ ${P3_MODE:-micro} == smoke ]]; then
    source "$P3_SRC/p3/env.sh"
    export SGLANG_GDN_K31_AB_DUAL=1 SGLANG_GDN_K31_AB_ENV=SGLANG_GDN_K31_FUSED,SGLANG_GDN_K31_CHOLQR_MIXED
    [[ $P3_STAGE == p3 ]] && export SGLANG_GDN_PD_SHALLOW_FULLN=1 TWINSTAR_PD_EMITTER_GRAPH=1
    python -m torch.distributed.run --standalone --nproc-per-node=2 "$P3_SRC/p3/p3_factory.py" \
        --smoke --stage "$P3_STAGE" --out "$P3_OUT" > "$P3_OUT/worker.log" 2>&1
else
    bash "$P3_SRC/p3/route_micro.sh" --factory p3_factory:build --stage "$P3_STAGE" --out "$P3_OUT" \
        > "$P3_OUT/worker.log" 2>&1
    python3 "$P3_SRC/p3/reduce_micro.py" "$P3_OUT" --stage "$P3_STAGE" | tee "$P3_OUT/reduce.txt"
fi
python3 "$P3_SRC/p3/canary.py" "$P3_OUT/worker.log" | tee "$P3_OUT/canary.txt"
if [[ $P3_STAGE == p3 ]]; then
    grep -m1 'GDN_PD_SHALLOW_FULLN_ACTIVE role=prefill shallow_layers=24 deep_layers=12' "$P3_OUT/worker.log"
fi
echo "P3_GPU_MICRO_DONE $P3_STAGE $(date -u +%FT%TZ)"
