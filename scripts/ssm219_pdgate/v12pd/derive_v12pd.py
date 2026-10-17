"""docs/222 §21 (lead 09:17 / 09:38 / 09:44, integration 10-17): paper-ladder 1P:1D row on the "v12" stack, LHR.
Package = the j4569012 window package pkg-p30305a3bb41198b (S = S_fp32 stock; C+ = the 4569012 C+ recipe on the stock
capsule 7a841e4189, fp16 factors R16; PC+ = the 4569012 PC+ recipe: P3 tree 6fb0b992d93 + P3 helper, route 0 + deferred
on P) with the U1 overlay on EVERY variant and role, as the agg row sets F219_U1A=1 + F219_U1C=1 on every slot:

  U1-a : SGLANG_GDN_U1A_RESULT_BEFORE_EXTEND=1 + WS-3's u1a site dir FIRST on PYTHONPATH (its sitecustomize installs
         ws3_u1a_register.py on sglang.srt.managers.scheduler import and chains to the next sitecustomize on sys.path, i.e.
         the window's /pdrefresh/native-observer one) + SGLANG_AB_OUT=<worker dir>/u1a (decision log, archived with the run).
         The two WS-3 files are frozen read-only under the campaign inputs (u1-overlay-<blobs>), same bytes as
         $W/results/ws3-ttft-223/tools/u1a.  Structurally inert in PD (WS-3): P runs --disable-overlap-schedule, D has
         only decode / prebuilt batches; set anyway for "same stack" (lead 09:44).
  U1-c : SGLANG_QSA_PREFILL_HOIST=1 (stock switch, present in both PD trees: environ.py:309, qsa/metadata.py:135).

Points (each text patch asserts its exact hit count):
  r16_delta apply_u1 + constants; clone_role apply_u1 after apply_p3; prewindow_gate U1 env + "WS3_U1A installed" on both
  roles (WS3_U1A_ACTIVE reported); harness_mode LADDER_WINDOW_SECONDS in (300, 900) (300 = canary only); entry_guard frozen
  overlay; CPU gate (science env + the U1 keys on every variant x role, runtime_preflight with the U1 path, scheduler import
  on both trees installs U1-a).
Usage (CMH or LHR login):  python3 derive_v12pd.py
"""
import difflib
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path

W = '/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911'
ROOT = Path(f'{W}/results/p3pd-20261017')
PARENT_FP = 'p30305a3bb41198b'
PARENT = ROOT / f'inputs/pkg-{PARENT_FP}'
OUT = ROOT / 'inputs/pkg-work-v12pd'
U1_SRC = Path(f'{W}/results/ws3-ttft-223/tools/u1a')
U1_FILES = {'site/sitecustomize.py': None, 'ws3_u1a_register.py': None}  # filled with git blobs at derive time
HERE = Path(__file__).resolve().parent
patches = []


def git_blob(p):
    data = Path(p).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def sha256(p):
    return hashlib.file_digest(open(p, 'rb'), 'sha256').hexdigest()


def patch(rel, old, new, count=1):
    p = OUT / rel
    t = p.read_text()
    assert t.count(old) == count, (rel, t.count(old), count, old[:80])
    p.write_text(t.replace(old, new))
    patches.append(dict(file=rel, old=old[:100], new=new[:100], count=count))


def freeze_u1():
    blobs = {rel: git_blob(U1_SRC / rel) for rel in U1_FILES}
    name = 'u1-overlay-' + blobs['site/sitecustomize.py'][:7] + '-' + blobs['ws3_u1a_register.py'][:7]
    host = ROOT / 'inputs' / name
    if not host.exists():
        tmp = ROOT / 'inputs' / (name + '.tmp')
        if tmp.exists():
            shutil.rmtree(tmp)
        for rel in U1_FILES:
            (tmp / 'u1a' / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(U1_SRC / rel, tmp / 'u1a' / rel)
        for p in sorted(tmp.rglob('*'), key=lambda x: -len(x.parts)):
            os.chmod(p, 0o444 if p.is_file() else 0o555)
        os.chmod(tmp, 0o555)
        tmp.rename(host)
    for rel, b in blobs.items():
        assert git_blob(host / 'u1a' / rel) == b, rel
    assert not any(os.stat(p).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) for p in [host, *host.rglob('*')])
    return name, host, blobs


def main():
    m0 = json.loads((PARENT / 'manifest.json').read_text())
    assert m0['fingerprint'] == PARENT_FP
    name, host, blobs = freeze_u1()
    site = f'/work/results/p3pd-20261017/inputs/{name}/u1a/site'
    sc = (host / 'u1a/site/sitecustomize.py').read_text()
    assert '_chain()' in sc and 'ws3_u1a_register.py' in sc and 'SGLANG_GDN_U1A_RESULT_BEFORE_EXTEND' in sc
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(PARENT, OUT, ignore=shutil.ignore_patterns('__pycache__'))

    # ---- r16_delta: apply_u1 ----
    p = OUT / 'support/r16_delta.py'
    p.write_text(p.read_text() + (
        "\n\n# docs/222 §21 (paper ladder 1P:1D row, v12 stack, lead 09:17 / 09:44): U1-a + U1-c on every variant and role.\n"
        f"U1_SITE = {site!r}\n"
        "U1_ENV = {'SGLANG_GDN_U1A_RESULT_BEFORE_EXTEND': '1', 'SGLANG_QSA_PREFILL_HOIST': '1'}\n\n\n"
        "def apply_u1(variant, role, env, setv, worker_root):\n"
        "    \"\"\"U1-a (site dir first on PYTHONPATH + switch + decision-log dir) and U1-c (QSA prefill hoist).\"\"\"\n"
        "    pp = env['PYTHONPATH'].split(':')\n"
        "    assert U1_SITE not in pp and not set(U1_ENV) & set(env) and 'SGLANG_AB_OUT' not in env, (variant, role)\n"
        "    why = 'docs/222 §21: v12 stack U1-a (WS-3 u1a site, frozen) + U1-c on every variant / role (lead 09:44)'\n"
        "    setv('PYTHONPATH', ':'.join([U1_SITE] + pp), why)\n"
        "    for k, v in U1_ENV.items():\n"
        "        setv(k, v, why)\n"
        "    setv('SGLANG_AB_OUT', str(worker_root) + '/u1a', why)\n"
        "    return sorted(U1_ENV) + ['SGLANG_AB_OUT']\n"))
    patches.append(dict(file='support/r16_delta.py', appended='apply_u1 + constants'))

    # ---- clone_role ----
    patch('support/clone_role.py', "    apply_p3(variant, role, env, setv)  # docs/222 §16\n",
          "    apply_p3(variant, role, env, setv)  # docs/222 §16\n"
          "    from r16_delta import apply_u1\n    apply_u1(variant, role, env, setv, worker_root)  # docs/222 §21\n")

    # ---- prewindow_gate: U1 on both roles of every variant ----
    patch('support/prewindow_gate.py', "   assert 'The server is fired up and ready to roll!' in log\n",
          "   assert 'The server is fired up and ready to roll!' in log\n"
          "   from r16_delta import U1_SITE,U1_ENV  # docs/222 §21: v12 U1-a + U1-c, every variant / role\n"
          "   ae=actual['environment'];assert ae['PYTHONPATH'].split(':')[0]==U1_SITE and all(ae.get(k)==v for k,v in U1_ENV.items()) and ae.get('SGLANG_AB_OUT'),(role,'U1 env')\n"
          "   u1i=[line for line in log.splitlines() if 'WS3_U1A installed on' in line];assert u1i,(role,'WS3_U1A installed line missing')\n"
          "   out.setdefault('u1',{})[role]=dict(installed=len(u1i),active=sum('WS3_U1A_ACTIVE' in line for line in log.splitlines()),hoist=ae['SGLANG_QSA_PREFILL_HOIST'])\n")

    # ---- harness_mode: canary-only short window ----
    patch('support/harness_mode.py', "        return 60 if self.dry_run else 900\n",
          "        if self.dry_run:\n            return 60\n"
          "        seconds = int(os.environ.get('LADDER_WINDOW_SECONDS', '900'))  # docs/222 §21: 300 s = LHR canary only\n"
          "        assert seconds in (300, 900), seconds\n        return seconds\n")

    # ---- entry_guard: the frozen U1 overlay ----
    patch('support/entry_guard.py', "if a.mode=='grid':\n",
          "# docs/222 §21: the frozen U1 overlay (exact files)\n"
          "uo=Path(m['u1_overlay_host'])/'u1a'\n"
          "for rel,b in m['u1_overlay_blobs'].items():\n"
          " d=(uo/rel).read_bytes();assert hashlib.sha1(b'blob %d\\0'%len(d)+d).hexdigest()==b,rel\n"
          "assert sorted(str(f.relative_to(uo)) for f in uo.rglob('*') if f.is_file())==sorted(m['u1_overlay_blobs']),'U1 overlay files'\n"
          "if a.mode=='grid':\n")

    # ---- CPU gate ----
    cg = 'support/r16_cpu_gate.py'
    patch(cg, "        assert sorted(science_env) == sorted(want), (variant, role, science_env)\n",
          "        want = sorted(set(want) | {'PYTHONPATH', 'SGLANG_AB_OUT'} | set(r16_delta.U1_ENV))  # docs/222 §21\n"
          "        assert sorted(science_env) == sorted(want), (variant, role, science_env)\n"
          "        assert env['PYTHONPATH'].split(':')[0] == r16_delta.U1_SITE and env['SGLANG_AB_OUT'].endswith('/u1a')\n"
          "        assert all(env[k] == v for k, v in r16_delta.U1_ENV.items()), (variant, role)\n")
    patch(cg, "        rec['environment']['CUDA_VISIBLE_DEVICES'] = '0,1'\n",
          "        rec['environment']['CUDA_VISIBLE_DEVICES'] = '0,1'\n"
          "        rec['environment']['PYTHONPATH'] = r16_delta.U1_SITE + ':' + rec['environment']['PYTHONPATH']  # docs/222 §21\n"
          "        rec['environment'].update(r16_delta.U1_ENV)\n")
    patch(cg, "result['passed'] = True\n", (HERE / 'cpu_gate_u1.py.in').read_text() + "result['passed'] = True\n")

    # ---- manifest + fingerprint ----
    files = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
    m = dict(m0)
    fp = 'v12pd' + hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:11]
    m.update(fingerprint=fp, p3pd_parent_fingerprint=PARENT_FP, files=files, u1_overlay_host=str(host),
             u1_overlay_site=site, u1_overlay_blobs=blobs, u1_overlay_source=str(U1_SRC), v12pd_patches=patches)
    (OUT / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
    final = ROOT / f'inputs/pkg-{fp}'
    assert not final.exists(), final
    OUT.rename(final)
    lines = []
    for rel in sorted(files):
        a, b = PARENT / rel, final / rel
        ta = a.read_text(errors='replace').splitlines(True) if a.exists() else []
        tb = b.read_text(errors='replace').splitlines(True)
        if ta != tb:
            lines += difflib.unified_diff(ta, tb, 'p3pd/' + rel, 'v12pd/' + rel)
    (ROOT / f'inputs/pkg-{fp}.diff').write_text(''.join(lines))
    print(fp, len(files), len(patches), name, json.dumps(blobs))


if __name__ == '__main__':
    assert len(sys.argv) == 1
    main()
