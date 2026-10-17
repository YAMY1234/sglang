"""docs/222 §20: per-run P12 checks of a finished pd1n run (CMH host python3, zero GPU; reads the archived run dir).

  python3 p12_runcheck.py <runs/j<job>-<kind>-<label>> --kind positive|negative|micro|ab   -> P12RUN <label> {json}

timing runs (micro / ab): window complete; prewindow admission passed; none of the four proof / replay / negative keys in
  the P or D launch env and no P12_WIRE_ line; the P12 switch as the label says (P actual env); P12 path per room from
  P's TP0 "Flash-Next staging transfer: {json}" lines: arm 1 -> every line p12=True with kv_bytes > 0 and kv_segments >= 1
  (p12_path_fallbacks = lines that are not = the pre-registered "fallback 0"); arm 0 / C+ -> no p12 line.
  Reported: full-N prefill summary fallbacks (last per TP rank; the j4569012 PC+ window had 16 at 600 forwards).
gate runs: p12-gate/driver.json passed (client + required P / D lines, none forbidden) and the torch.equal checker json.
"""
import argparse
import json
import re
from pathlib import Path

DEBUG = ('SGLANG_FLASHNEXT_PD_STAGING_PROOF_DIR', 'SGLANG_FLASHNEXT_PD_STAGING_PROOF_TENSORS',
         'SGLANG_GDN_PD_P12_WIRE_REPLAY', 'SGLANG_GDN_PD_P12_WIRE_NEGATIVE')
KEY = 'SGLANG_GDN_PD_SHALLOW_SEND_PIPELINE'


def read(p):
    return Path(p).read_text(errors='replace') if Path(p).exists() else ''


def env_of(p):
    return json.loads(Path(p).read_text())['environment'] if Path(p).exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--kind', required=True, choices=('positive', 'negative', 'micro', 'ab'))
    a = ap.parse_args()
    d = Path(a.run)
    label = d.name.split(f'-{a.kind}-', 1)[1]
    plog, dlog = read(d / 'workers/P0/server.log'), read(d / 'remote-node/workers/D0/server.log')
    penv, denv = env_of(d / 'workers/P0/actual-launch.json'), env_of(d / 'remote-node/workers/D0/actual-launch.json')
    out = dict(run=d.name, kind=a.kind, label=label, P_env_found=penv is not None, D_env_found=denv is not None)
    if penv:
        out.update(variant_sha=penv.get('V4_SHA'), arm=penv.get(KEY),
                   debug_env=sorted(k for k in DEBUG if k in penv or k in (denv or {})))
    stag = []
    for line in plog.splitlines():
        if 'Flash-Next staging transfer:' in line and 'TP0' in line:
            try:
                stag.append(json.loads(line.split('Flash-Next staging transfer:', 1)[1]))
            except ValueError:
                pass
    p12 = [s for s in stag if s.get('p12')]
    good = [s for s in p12 if s.get('kv_bytes', 0) > 0 and s.get('kv_segments', 0) >= 1]
    fulln = {}
    for m in re.finditer(r'\[[^\]]* (TP\d)\] GDN full-N prefill: role=prefill forwards=(\d+)[^\n]*fallbacks=(\d+)', plog):
        fulln[m.group(1)] = dict(forwards=int(m.group(2)), fallbacks=int(m.group(3)))
    out.update(staging_lines_tp0=len(stag), p12_lines=len(p12), p12_kv_first_lines=len(good),
               pipeline_lines=[x[-120:] for x in plog.splitlines() if 'PD_SHALLOW_SEND_PIPELINE_ACTIVE' in x],
               wire_lines=sum('P12_WIRE_' in x for x in (plog + dlog).splitlines()), fulln_last=fulln)
    if a.kind in ('micro', 'ab'):
        pre = json.loads(read(d / 'lead-prewindow-admission.json') or '{}')
        cell = json.loads(read(d / 'complete.json') or '{}')
        want_arm = {'Cp': None, 'PCp0': '0', 'PCp1': '1'}[label]
        fb = len(stag) - len(good) if want_arm == '1' else None
        out.update(complete=bool(cell), prewindow_passed=pre.get('passed'), p12_path_fallbacks=fb,
                   tps_gpu=cell.get('tps_gpu'), completed=(cell.get('summary') or {}).get('completed_in_window'))
        ok = (bool(cell) and pre.get('passed') is True and penv is not None and not out['debug_env']
              and out['wire_lines'] == 0 and penv.get(KEY) == want_arm
              and ((fb == 0 and len(stag) > 0) if want_arm == '1' else not p12))
    else:
        drv = json.loads(read(d / 'p12-gate/driver.json') or '{}')
        eq = d / ('p12-gate-equal.json' if a.kind == 'positive' else 'p12-gate-negative.json')
        chk = json.loads(read(eq) or '{}')
        out.update(driver_passed=drv.get('passed'), client_rc=drv.get('client_rc'),
                   missing_P=[k for k, v in (drv.get('found_P') or {}).items() if not v],
                   missing_D=[k for k, v in (drv.get('found_D') or {}).items() if not v],
                   checker=str(eq.name) if eq.exists() else None, checker_passed=chk.get('passed') if chk else None,
                   checker_error=chk.get('error'))
        ok = drv.get('passed') is True and penv is not None and penv.get(KEY) == '1'
        if chk:
            ok = ok and chk.get('passed') is True
    out['ok'] = bool(ok)
    print('P12RUN', label, json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
