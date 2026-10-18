"""docs/222 §23 C8 canary verdict (pre-registered before the canary runs): python3 f32_canary_check.py <runs dir> <job> [n2]
n2 (docs/222 §24): the N2 {S, PC+f32c} window; PCpf32c is checked as Cpf32c, plus P0 GDN_PD_SHALLOW_FULLN_ACTIVE >= 1;
   by role per lead 22:5x (pre-registration update made before the N2 canary / windows ran): P0 FACTORED_CUDA_FP32_ACTIVE is
   NOT expected on the shallow P (prefill only) and only reported; D0 must show the executed set (FP32 step + expiry,
   STEP_CUDA, FUSE_NORM, TRUNC_CUDA); K31_FUSED / K31_MIXED_EIGH_ANY_BATCH / PREFIX_RESTORE_FUSED / BOOKKEEPING_FOLD are
   structurally not executed on D (captured boundary graph) and are not checked.

Per arm (run dir runs/j<job>-<label>):
  all     : complete.json (reducer cell: N > 0, tps_gpu > 0, user_p90 > 0); lead-prewindow-admission.json passed.
  Cpf32c  : prewindow f32 gate passed on P and D (pool `dtype=fp32 ... factors float32`, no BUILD_FAILED);
            D0 log: FACTORED_CUDA_FP32_ACTIVE step and expiry, FACTORED_STEP_CUDA_ACTIVE, FACTORED_STEP_FUSE_NORM_ACTIVE,
            FACTORED_TRUNC_CUDA_ACTIVE each >= 1; no BUILD_FAILED, no Traceback on either role during service (lead 23:3x
            clarification after the 261411 reading: before the 'SIGTERM received' line; the post-SIGTERM graceful-shutdown race
            after 'Remaining number of requests 0' is counted separately as post_sigterm_shutdown_race, not a failure).
            P0 log: FACTORED_CUDA_FP32_ACTIVE counted and reported.  Lead 19:3x asks for it on P too; WS-2 expects 0 on P
            (P serves prefill only; the line is logged only by the CUDA decode step / expiry).  P = 0 is reported as
            P_FP32_ACTIVE_ABSENT for the lead's ruling, never silently passed.
  Cpk15   : prewindow gate passed (pool `dtype=fp16 ... factors float16`); D0 log: FACTORED_STEP_CUDA_ACTIVE,
            FACTORED_STEP_FUSE_NORM_ACTIVE, FACTORED_TRUNC_CUDA_ACTIVE >= 1 and FACTORED_CUDA_FP32_ACTIVE == 0 on both roles.
  P->D    : the factor arms' complete.json exists (every request is prefilled on P and decoded on D, so a served window
            with N > 0 is the transfer working); D0 logs of both arms show the pool line with the arm's dtype.
"""
import json
import sys
from pathlib import Path

ACT = ('FACTORED_CUDA_FP32_ACTIVE step', 'FACTORED_CUDA_FP32_ACTIVE expiry', 'FACTORED_STEP_CUDA_ACTIVE',
       'FACTORED_STEP_FUSE_NORM_ACTIVE', 'FACTORED_TRUNC_CUDA_ACTIVE')


def log_of(run, role):
    for p in (run / 'remote-node/workers' / f'{role}0/server.log', run / 'workers' / f'{role}0/server.log'):
        if p.is_file() and p.stat().st_size:
            return p, p.read_text(errors='replace')
    return None, ''


def main(runs, job, kind='n3'):
    out = dict(job=job, kind=kind, arms={}, verdict='PASS', notes=[])
    for lab in (('S', 'Cpf32c', 'Cpk15') if kind == 'n3' else ('S', 'PCpf32c')):
        run = Path(runs) / f'j{job}-{lab}'
        a = dict(run=str(run), fails=[])
        c = run / 'complete.json'
        if c.is_file():
            cj = json.loads(c.read_text())
            res = cj.get('result', cj)
            a['complete'] = {k: res.get(k) for k in ('N', 'tps_gpu', 'user_p90', 'ttft_p50_s', 'hit')}
            if not (res.get('N') and res.get('tps_gpu') and res.get('user_p90')):
                a['fails'].append('complete.json without a served cell')
        else:
            a['fails'].append('no complete.json')
        pw = run / 'lead-prewindow-admission.json'
        pwj = json.loads(pw.read_text()) if pw.is_file() else {}
        if not pwj.get('passed'):
            a['fails'].append('prewindow not passed')
        if lab != 'S':
            a['f32_gate'] = {r: (g.get('passed'), g.get('factored_line', '')[:200]) for r, g in pwj.get('f32', {}).items()}
            if set(a['f32_gate']) != {'P', 'D'} or not all(v[0] for v in a['f32_gate'].values()):
                a['fails'].append('prewindow f32 gate missing / failed on a role')
            for role in ('P', 'D'):
                path, log = log_of(run, role)
                lines = log.splitlines()
                a[f'{role}_log'] = str(path)
                a[f'{role}_active'] = {k: sum(k in l for l in lines) for k in ACT}
                a[f'{role}_build_failed'] = sum('BUILD_FAILED' in l for l in lines)
                sig = next((i for i, l in enumerate(lines) if 'SIGTERM received' in l), len(lines))
                a[f'{role}_traceback'] = sum('Traceback' in l for l in lines[:sig])  # during service (lead 23:3x)
                a[f'{role}_post_sigterm_shutdown_race'] = sum('Traceback' in l for l in lines[sig:])
                if a[f'{role}_post_sigterm_shutdown_race'] and not any('Remaining number of requests 0' in l for l in lines[sig:]):
                    a['fails'].append(f'{role}0 traceback after SIGTERM without a drained shutdown')
                dt = 'dtype=fp16' if lab == 'Cpk15' else 'dtype=fp32'
                fl = 'factors float16' if lab == 'Cpk15' else 'factors float32'
                pool = [l for l in lines if 'Factored GDN pool allocated' in l]
                if not pool or not all(dt in l and fl in l for l in pool):
                    a['fails'].append(f'{role}0 pool line not {dt} / {fl}')
                if a[f'{role}_build_failed'] or a[f'{role}_traceback']:
                    a['fails'].append(f'{role}0 BUILD_FAILED / Traceback')
            d = a['D_active']
            if lab == 'PCpf32c':
                path, log = log_of(run, 'P')
                a['P_shallow_fulln'] = sum('GDN_PD_SHALLOW_FULLN_ACTIVE' in l for l in log.splitlines())
                if a['P_shallow_fulln'] < 1:
                    a['fails'].append('P0 lacks GDN_PD_SHALLOW_FULLN_ACTIVE')
            if lab in ('Cpf32c', 'PCpf32c'):
                for k in ACT:
                    if d[k] < 1:
                        a['fails'].append(f'D0 lacks {k}')
                if lab == 'PCpf32c':  # lead 22:5x (before the N2 canary ran): shallow P only prefills -> not expected, reported
                    a['P_fp32_active_not_expected'] = a['P_active']['FACTORED_CUDA_FP32_ACTIVE step'] + a['P_active']['FACTORED_CUDA_FP32_ACTIVE expiry']
                elif a['P_active']['FACTORED_CUDA_FP32_ACTIVE step'] + a['P_active']['FACTORED_CUDA_FP32_ACTIVE expiry'] == 0:
                    a['P_FP32_ACTIVE_ABSENT'] = True
                    out['notes'].append(f'{lab} P0 has no FACTORED_CUDA_FP32_ACTIVE line (expected: P is prefill-only); '
                                        'lead 19:3x criterion on P not met -> lead rules')
            else:
                for k in ACT[2:]:
                    if d[k] < 1:
                        a['fails'].append(f'D0 lacks {k}')
                if any(a[f'{r}_active'][k] for r in ('P', 'D') for k in ACT[:2]):
                    a['fails'].append('fp16 control logged FACTORED_CUDA_FP32_ACTIVE')
        out['arms'][lab] = a
        if a['fails']:
            out['verdict'] = 'FAIL'
    if out['verdict'] == 'PASS' and any(a.get('P_FP32_ACTIVE_ABSENT') for a in out['arms'].values()):
        out['verdict'] = 'PASS_D__P_FP32_ACTIVE_ABSENT_LEAD_RULES'
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], *(sys.argv[3:4] or ['n3']))
