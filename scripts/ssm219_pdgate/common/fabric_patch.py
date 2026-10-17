"""docs/222 §20.2: apply the fabric binding to a derived package (called by derive_p12.py / derive_v12pd.py)."""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPU_BLOCK = '''# docs/222 §20.2 (LHR port): the NCCL / UCX fabric keys follow the job environment; check() fails on a missing name.
import fabric_bind
fb = {}
for variant in ('S', 'C+', 'PC+'):
    for role in ('P', 'D'):
        inh = dict(os.environ, V4_RUN_ID='jcpu-fab', PSIDE_C='8', V4_COORD_ROOT=str(root / 'coord'), P12_ARM='1')
        for k in fabric_bind.FABRIC:
            inh.pop(k, None)
        e = build(variant, role, inh, root / 'run', root / f'fab-{variant}-{role}', root / 'coord', bundle, '0,1', 99,
                  {'--port': 65011, '--nccl-port': 65015, '--disaggregation-bootstrap-port': 65013}, cpu=True)[0]
        assert not set(fabric_bind.FABRIC) & set(e), (variant, role, sorted(set(fabric_bind.FABRIC) & set(e)))
        base = json.loads(Path(f'/pdrefresh/baselines/{variant}/role-{role}-actual-launch.json').read_text())['environment']
        inh.update({k: base[k] for k in fabric_bind.FABRIC if k in base})
        e2 = build(variant, role, inh, root / 'run', root / f'fab2-{variant}-{role}', root / 'coord', bundle, '0,1', 99,
                   {'--port': 65011, '--nccl-port': 65015, '--disaggregation-bootstrap-port': 65013}, cpu=True)[0]
        assert {k: e2.get(k) for k in fabric_bind.FABRIC} == {k: base.get(k) for k in fabric_bind.FABRIC}, (variant, role)
        fb[f'{variant}/{role}'] = dict(baseline={k: base.get(k) for k in fabric_bind.FABRIC if k in base}, without_job_env='unset')
for bad in ({'NCCL_SOCKET_IFNAME': 'eth0-not-here'}, {'NCCL_IB_HCA': '=rdma_vf_rail0'}, {'NCCL_NET_PLUGIN': 'spcx-not-here'}):
    try:
        fabric_bind.check(bad)
        raise AssertionError(f'fabric check accepted {bad}')
    except RuntimeError as exc:
        assert 'FABRIC_CHECK_FAIL' in str(exc)
fb['empty_env_check'] = fabric_bind.check({})
result['checks']['fabric'] = fb
'''


def apply(out, patch, patches):
    """out: the package work dir; patch(rel, old, new, count=1): the deriver's asserting text patch."""
    shutil.copyfile(HERE / 'fabric_bind.py', out / 'support/fabric_bind.py')
    patches.append(dict(file='support/fabric_bind.py', added=True))
    patch('support/clone_role.py',
          "    setv('PDREF_ROOT', str(root), 'Current local output root')\n",
          "    from fabric_bind import bind as fabric_bind, check as fabric_check  # docs/222 §20.2 (LHR port)\n"
          "    fabric_bind(env, inherited, setv)\n"
          "    setv('PDREF_ROOT', str(root), 'Current local output root')\n")
    patch('support/clone_role.py', "    assert argv[argv.index('--tp')+1] == '2'\n",
          "    if not cpu:\n        fabric_check(env)  # docs/222 §20.2: fail fast on a fabric name this node lacks\n"
          "    assert argv[argv.index('--tp')+1] == '2'\n")
    patch('support/r16_cpu_gate.py', "           'Current service TP2 GPU binding', 'Current authorized concurrency/run/census metadata'}\n",
          "           'Current service TP2 GPU binding', 'Current authorized concurrency/run/census metadata',\n"
          "           'Current cluster fabric binding (docs/222 §20.2): NCCL / UCX network keys follow the job environment'}\n")
    patch('support/r16_cpu_gate.py', "result['passed'] = True\n", CPU_BLOCK + "result['passed'] = True\n")
