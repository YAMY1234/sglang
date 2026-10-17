"""docs/229 §3 P3 / P5 holder micro: shell `export` lines that rebuild the frozen PD P-role environment of an fp32-pd P0 worker
(workers/P0/actual-launch.json) for an in-process P worker group (p3_factory.py), with the fork source swapped.

Kept: the launch's model / runtime switches (SGLANG_*, TWINSTAR_*, PSIDE_*, PFACTOR3_*, PD_FACTOR_ONLY_CONTRACT, NCCL_*, MC_*,
REMEASURE_PROMPT_ONLY_STATE_CACHE, HF_* offline) plus the execution_argv `env K=V` prefix; REMEASURE_NATIVE_LAUNCH=0 (no
native-observer harness hooks in the measured process, so the observer keys can stay dropped); per-worker /tmp paths of the frozen
run are remapped under <out>/runtime.  Dropped (recorded in <out>/frozen-env.json): the PD observer / router / client hooks that
need a live D side or HTTP (TWINSTAR_AGENTX_*, TWINSTAR_PD_FORMAL_TRACE, TWINSTAR_PD_TORCH_PROFILE, PD_REMOTE_HTTP_FD, PD_ROUTER_*,
PD_SCREEN_WORKER, PD_TOPOLOGY_INPUTS), SGLANG_MOONCAKE_CUSTOM_MEM_POOL (no transfer engine in-process), Slurm / PMIx / shell
variables, CUDA_VISIBLE_DEVICES (holder slots) and the frozen caches.
PYTHONPATH = [HELPER_DIR (P3 helper copy, optional)] + <src>/python + the launch's helper / support / extra entries that exist,
with the frozen fork capsule replaced by <src>/python; missing entries are dropped and recorded.

  python3 p3_frozen_env.py --launch <actual-launch.json> --src /work/src/<tree> --out <dir> [--helper-dir <dir>] > env.sh
"""
import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path

KEEP_PREFIX = ('SGLANG_', 'TWINSTAR_', 'PSIDE_', 'PFACTOR3_', 'NCCL_', 'MC_', 'HF_')
KEEP_NAMES = {'PD_FACTOR_ONLY_CONTRACT', 'REMEASURE_PROMPT_ONLY_STATE_CACHE', 'TOKENIZERS_PARALLELISM'}
DROP_NAMES = {'TWINSTAR_AGENTX_INVENTORY', 'TWINSTAR_AGENTX_PRIVATE_OCCUPANCY', 'TWINSTAR_PD_FORMAL_TRACE',
              'TWINSTAR_PD_TORCH_PROFILE', 'SGLANG_MOONCAKE_CUSTOM_MEM_POOL'}
DROP_PREFIX = ('PD_ROUTER_', 'PD_REMOTE_', 'PD_SCREEN_', 'PD_TOPOLOGY_', 'SLURM', 'PMIX', 'OMPI', 'PRTE')
CACHE_NAMES = {'TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH', 'SGLANG_CACHE_DIR',
               'XDG_CACHE_HOME', 'PYTHONPYCACHEPREFIX', 'TMPDIR'}
CAPSULE_MARK = '-capsule-'


def frozen_worker_root(env):
    # the per-worker /tmp root of the frozen run, e.g. /tmp/fp32-pd-20261015/j4529207-Cp
    for key in ('V4_LOCAL_ROOT', 'PDREF_ROOT'):
        if env.get(key, '').startswith('/tmp/'):
            return env[key]
    return None


def build(launch, src, out, helper_dir=None, exists=os.path.exists):
    env = dict(launch['environment'])
    for tok in launch.get('execution_argv', [])[1:]:  # ['env', 'K=V', ..., python, -m, ...]
        if '=' not in tok or tok.startswith('-'):
            break
        k, v = tok.split('=', 1)
        env[k] = v
    root = frozen_worker_root(env)
    runtime = f'{out}/runtime'
    kept, dropped, remapped = {}, {}, {}
    for k, v in sorted(env.items()):
        if k in DROP_NAMES or k.startswith(DROP_PREFIX) or k in CACHE_NAMES:
            dropped[k] = v
            continue
        if not (k.startswith(KEEP_PREFIX) or k in KEEP_NAMES):
            continue
        if k == 'CUDA_VISIBLE_DEVICES':
            dropped[k] = v
            continue
        if root and root in v:
            nv = v.replace(root, runtime)
            remapped[k] = (v, nv)
            v = nv
        elif v.startswith('/tmp/'):
            nv = f'{runtime}/{k.lower()}'
            remapped[k] = (v, nv)
            v = nv
        kept[k] = v
    cache = f'/tmp/p3micro-{hashlib.sha1(out.encode()).hexdigest()[:10]}'
    for k in ('TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'SGLANG_CACHE_DIR', 'XDG_CACHE_HOME'):
        kept[k] = f'{cache}/{k.lower()}'
    path, path_dropped = [], []
    if helper_dir:
        path.append(helper_dir)
    path.append(f'{src}/python')
    for p in env.get('PYTHONPATH', '').split(':'):
        if not p or CAPSULE_MARK in p:  # the frozen fork source is replaced by <src>/python
            continue
        if p in path:
            continue
        (path if exists(p) else path_dropped).append(p)
    kept['PYTHONPATH'] = ':'.join(path)
    kept['PYTHONDONTWRITEBYTECODE'] = '1'
    # WS-2 10-17: the receipt's native-observer sitecustomize (/pdrefresh/native-observer) installs harness observers
    # (native_pd_inventory reads TWINSTAR_AGENTX_INVENTORY inside model.forward) only when REMEASURE_NATIVE_LAUNCH=1; the
    # measured in-process P runs without them, so the observer keys stay dropped
    kept['REMEASURE_NATIVE_LAUNCH'] = '0'
    receipt = dict(launch_command=launch.get('command'), src=src, helper_dir=helper_dir, out=out, frozen_root=root,
                   kept=kept, dropped=sorted(dropped), remapped=remapped, pythonpath_dropped=path_dropped)
    return kept, receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--launch', required=True, type=Path)
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--helper-dir')
    a = ap.parse_args()
    kept, receipt = build(json.loads(a.launch.read_text()), a.src, a.out, a.helper_dir)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, 'frozen-env.json').write_text(json.dumps(receipt, indent=1) + '\n')
    for k, v in kept.items():
        print(f'export {k}={shlex.quote(v)}')
    for d in sorted({v for k, v in kept.items() if v.startswith(f'{a.out}/runtime')}):
        print(f'mkdir -p {shlex.quote(d)}')


if __name__ == '__main__':
    main()
