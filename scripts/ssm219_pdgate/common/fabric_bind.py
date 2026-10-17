"""docs/222 §20.2 (LHR port): the NCCL / UCX network keys are facts of the cluster a job runs on, not of the recipe.

The j4569012 baselines were captured on CMH GB300 nodes, whose job environment carries NCCL_SOCKET_IFNAME=eth0,
NCCL_IB_HCA==rdma_vf_rail0..3, NCCL_NET_PLUGIN=spcx, ... (not in the image).  On LHR there is no eth0 and the job
environment sets none of them, so the cloned CMH values made ncclGetUniqueId fail ("invalid usage", jobs 252358 / 252360).
bind(): every FABRIC key takes the current job's value (unset when the cluster sets none) under a binding reason, like the
Slurm keys; on CMH the job values equal the baseline, so CMH launches do not change.  check(): before a real (non-CPU)
launch, every named interface / HCA / net plugin must exist on this node, else fail fast with the missing names.
The harness's own keys (job.sh: NCCL_MNNVL_ENABLE, NCCL_CUMEM_ENABLE, MC_FORCE_MNNVL) are not touched.
"""
import glob
import os

FABRIC = ('NCCL_SOCKET_IFNAME', 'NCCL_IB_HCA', 'NCCL_NET_PLUGIN', 'NCCL_IB_ADDR_FAMILY', 'NCCL_IB_TC', 'NCCL_IB_TIMEOUT',
          'NCCL_IB_ADAPTIVE_ROUTING', 'UCX_NET_DEVICES', 'UCX_TLS')
REASON = 'Current cluster fabric binding (docs/222 §20.2): NCCL / UCX network keys follow the job environment'


def bind(env, inherited, setv):
    for k in FABRIC:
        setv(k, inherited.get(k), REASON)


def _names(value):
    return [x.split(':')[0] for x in value.lstrip('=^').split(',') if x and x not in ('all',)]


def check(env, net='/sys/class/net', ib='/sys/class/infiniband'):
    missing = []
    for k in ('NCCL_SOCKET_IFNAME', 'UCX_NET_DEVICES'):
        v = env.get(k)
        if v and not v.startswith('^'):  # '^x' excludes; prefixes match NCCL-style (eth -> eth0 ...)
            for n in _names(v):
                if not any(d.startswith(n) for d in os.listdir(net)):
                    missing.append(f'{k}={n}')
    v = env.get('NCCL_IB_HCA')
    if v and not v.startswith('^'):
        for n in _names(v):
            have = os.listdir(ib) if os.path.isdir(ib) else []
            if not (n in have if v.startswith('=') else any(d.startswith(n) for d in have)):
                missing.append(f'NCCL_IB_HCA={n}')
    v = env.get('NCCL_NET_PLUGIN')
    if v and v not in ('none',):
        libs = glob.glob(f'/usr/lib/*/libnccl-net-{v}.so*') + glob.glob(f'/usr/lib/libnccl-net-{v}.so*') + \
            glob.glob(f'/usr/local/lib/libnccl-net-{v}.so*') + glob.glob(f'/opt/*/lib*/libnccl-net-{v}.so*')
        if not libs:
            missing.append(f'NCCL_NET_PLUGIN={v}')
    if missing:
        raise RuntimeError('FABRIC_CHECK_FAIL ' + ' '.join(missing) + ' (docs/222 §20.2)')
    return dict(passed=True, keys={k: env.get(k) for k in FABRIC})
