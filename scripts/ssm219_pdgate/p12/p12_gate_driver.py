"""docs/222 §20 P12 live wire gate, P task: entry.py runs this instead of run_v2.py when P12_PROOF is set.

Same native P / D launches as the window harness (v3_services.services: clone_role.build + apply_p12 -> runtime_preflight
-> native launch with the window observers; D = v3_services.remote_main on task 1), but no router, warmup or AgentX:
P3's p12_wire_client.py (frozen capsule copy) posts its requests to P and D directly.  Then the P / D log lines.
The torch.equal checker (p12_wire_check.py) runs after the archive, from the hjob, over runs/<run>/p12-proof/{P,D}.
  P12_PROOF=positive : P has SGLANG_GDN_PD_P12_WIRE_REPLAY=1; client --mode positive (1025 + 4097 tokens)
  P12_PROOF=event|abort : P has SGLANG_GDN_PD_P12_WIRE_NEGATIVE=<mode>; client --mode <mode> (1 failure + 5 healthy)
Writes <root>/p12-gate/{client.json,driver.json,client.log}; prints P12GATE_DRIVER ...; exit 1 on any failure.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from static_client import atomic
from ladder_services import services

# full-N / deferred publication activation (both TP ranks log each) and the P12 sender line
REQUIRED_P = {'fulln': r'GDN_PD_SHALLOW_FULLN_ACTIVE role=prefill shallow_layers=24 deep_layers=12',
              'deferred_join_offload': r'PD full-N deferred publication enabled[^\n]*join_offload=True',
              'pipeline_staged': r'PD_SHALLOW_SEND_PIPELINE_ACTIVE staged=True'}
MODE_P = {'positive': {'replay': r'P12_WIRE_REPLAY_ACTIVE frozen_source=True aux_success=once timing_eligible=False'},
          'event': {'injected': r'P12_WIRE_NEGATIVE_RECEIPT phase=injected'},
          'abort': {'injected': r'P12_WIRE_NEGATIVE_RECEIPT phase=injected', 'abort': r'P12_WIRE_NEGATIVE_P mode=abort'}}
MODE_D = {'positive': {}, 'event': {'reserved': r'P12_WIRE_NEGATIVE_D phase=reserved', 'drained': r'P12_WIRE_NEGATIVE_D phase=drained'},
          'abort': {'reserved': r'P12_WIRE_NEGATIVE_D phase=reserved', 'drained': r'P12_WIRE_NEGATIVE_D phase=drained'}}
FORBID = {'positive': ('P12_WIRE_NEGATIVE',), 'event': ('P12_WIRE_REPLAY_ACTIVE',), 'abort': ('P12_WIRE_REPLAY_ACTIVE',)}


def lines(text, pattern):
    return [x for x in text.splitlines() if re.search(pattern, x)]


def main():
    root = Path(os.environ['PDREF_ROOT']); root.mkdir(parents=True, exist_ok=True)
    mode = os.environ['P12_PROOF']
    assert mode in MODE_P and os.environ['V6_VARIANT'] == 'PC+' and os.environ['P12_ARM'] == '1'
    assert os.environ['PDREF_MODE'] == 'grid' and os.environ['V3_DEPLOYMENT'] == 'PD'
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 4
    out = root / 'p12-gate'; out.mkdir(exist_ok=True)
    res = dict(mode=mode, passed=False, epoch=time.time(), run=os.environ['V4_RUN_ID'])
    src = None
    try:
        with services(root, 'PD', os.environ['PSIDE_ARM'], False) as (workers, actors, groups):
            P, D = workers
            assert (P['role'], D['role']) == ('P', 'D'), (P['role'], D['role'])
            env = json.loads((Path(P['directory']) / 'actual-launch.json').read_text())['environment']
            src = env['V4_SOURCE']
            res.update(P=dict(url=P['url'], bootstrap_port=P['bootstrap_port'], directory=P['directory']),
                       D=dict(url=D['url'], directory=D.get('remote_directory', D['directory'])), source=src,
                       source_sha=env['V4_SHA'])
            cmd = [sys.executable, src + '/p3/p12_wire_client.py', '--p', P['url'], '--d', D['url'],
                   '--bootstrap-host', P['host'], '--bootstrap-port', str(P['bootstrap_port']), '--mode', mode,
                   '--out', str(out / 'client.json')]
            res['client_command'] = cmd
            with (out / 'client.log').open('w') as sink:
                proc = subprocess.Popen(cmd, stdout=sink, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 1800
                while proc.poll() is None:
                    for g in groups:
                        g.check()
                    if time.monotonic() > deadline:
                        proc.terminate(); proc.wait(timeout=30)
                        raise TimeoutError('p12_wire_client')
                    time.sleep(1)
            res['client_rc'] = proc.returncode
            time.sleep(10)  # receipts / log lines of the last request
            for g in groups:
                g.check()
            plog = (Path(P['directory']) / 'server.log').read_text(errors='replace')
            dlog = (Path(D.get('remote_directory', D['directory'])) / 'server.log').read_text(errors='replace')
        req = {**REQUIRED_P, **MODE_P[mode]}
        found_p = {k: lines(plog, v) for k, v in req.items()}
        found_d = {k: lines(dlog, v) for k, v in MODE_D[mode].items()}
        forbid = {k: [x for x in plog.splitlines() + dlog.splitlines() if k in x] for k in FORBID[mode]}
        pipe = lines(plog, r'PD_SHALLOW_SEND_PIPELINE_ACTIVE')
        client = json.loads((out / 'client.json').read_text()) if (out / 'client.json').exists() else {}
        res.update(found_P={k: v[:4] for k, v in found_p.items()}, found_D={k: v[:4] for k, v in found_d.items()},
                   forbidden=forbid, pipeline_lines=pipe, decision_counts=dict(
                       P_tracebacks=plog.count('Traceback'), D_tracebacks=dlog.count('Traceback')),
                   client_passed=client.get('passed'), client_complete=client.get('complete'))
        res['passed'] = bool(res['client_rc'] == 0 and client.get('passed') and client.get('complete')
                             and all(found_p.values()) and all(found_d.values()) and not any(forbid.values())
                             and all('staged=True' in x for x in pipe))
    except BaseException as exc:
        res['error'] = repr(exc)
        raise
    finally:
        atomic(out / 'driver.json', res)
        print('P12GATE_DRIVER', json.dumps(dict(mode=mode, passed=res['passed'], client_rc=res.get('client_rc'),
                                                missing_P=[k for k, v in res.get('found_P', {}).items() if not v],
                                                missing_D=[k for k, v in res.get('found_D', {}).items() if not v],
                                                forbidden={k: len(v) for k, v in res.get('forbidden', {}).items()},
                                                error=res.get('error'))), flush=True)
    if not res['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
