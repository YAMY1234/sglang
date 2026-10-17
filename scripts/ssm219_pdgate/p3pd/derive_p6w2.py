"""docs/222 §17 (lead 03:10 pre-registration, integration 10-17): second 1P:1D C8 window package = the first window's
package pkg-p30305a3bb41198b (window 4569012) with exactly one change, on the PC+ D role only:

  PC+ D        : SGLANG_GDN_PD_BOUNDARY_ASYNC=ordered + WS-4's P6 (b) overlay (twinstar 85477ffe4, the files of
                 $A/tools/p6/85477ffe4/overlay, frozen read-only under the campaign inputs) right after the P3 tree's python
                 on PYTHONPATH, as in pdgate p6b2-ordered-d053bd5e346 cand1.  No SGLANG_GDN_PD_BOUNDARY_AB, no MODGATE.
  PC+ P, S, C+, P : unchanged (byte-identical launches to the parent)

The overlay's twinstar_sgl/__init__.py appends the helper-runtime directory (TWINSTAR_P6_HELPER_BASE, default the
helper-runtime mount, which the PC+ instance fills with the P3 helper) to __path__, so only its own four modules shadow the
P3 helper: p6_boundary.py, p6_modgate.py (new), pd_boundary_graph.py (frozen + MODGATE hook, inert without
TWINSTAR_P6_MODGATE), pd_shallow.py (frozen + the P6 decode_boundary hunk).  The P3 helper's pd_shallow.py is frozen + one
hunk in SplitBoundaryPhase.before_send, which only the prefill sender calls (disaggregation/prefill.py dispatch_handoff
"before_send"): on D it is unreachable, so D's effective change against 4569012 is the P6 hunk.  derive asserts both diffs.

Points (each text patch asserts its exact hit count):
  r16_delta      apply_p6d(variant, role, env, setv): PC+ D only; overlay right after the P3 python; env key; absent keys
  clone_role     apply_p6d after apply_p3
  admitted       PC+:D0 += SGLANG_GDN_PD_BOUNDARY_ASYNC (prewindow_gate then requires it in the recipe and the actual env)
  preflight      PC+ D: twinstar_sgl resolves to the frozen overlay (exact files), its __path__ carries the helper, and the
                 non-shadowed modules (pd_shallow_install, pd_emitter_graph, pd_dense) resolve in the helper
  prewindow      PC+ D server log: `PD_BOUNDARY_ASYNC_ACTIVE mode=ordered ab=` once per TP rank (2); absent elsewhere
  entry_guard    frozen overlay files
  CPU gate       PC+ D science env (+ the key), overlay placement per variant x role, preflight on the apply_p6d-rendered
                 PC+ D0, P6 import check (mode ordered, canary once, decode_boundary hook, module origins)
Usage (CMH login):  python3 derive_p6w2.py
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
OUT = ROOT / 'inputs/pkg-work-p6'
HR = '/work/results/remeasure-v6-fivearm-tp1-20261004/inputs/helper-runtime'
OV_SRC = Path(f'{W}/results/ssm-kernel-219/tools/p6/85477ffe4/overlay')
OV_NAME = 'p6-overlay-85477ffe4'
OV_HOST = ROOT / 'inputs' / OV_NAME
OV_CONT = f'/work/results/p3pd-20261017/inputs/{OV_NAME}'
# twinstar 85477ffe4 (docs/233 P6 (b) M2): source of every overlay file (sha256 of `git show 85477ffe4:<path>`)
OV_FILES = {'__init__.py': 'fd4abfe55bcd8b99db8e614aab5c3230a95b8e655634dd04f06a2f9a7280b186',
            'p6_boundary.py': '8785fe72e998c1946aec5e4d017cdc47ab0d1827d25368da1ed39c44d548d2f4',
            'p6_modgate.py': 'ee40af5d65b7b1d8d469cc8e4fa3e9c29d49247f0df98a347794db7597114193',
            'pd_boundary_graph.py': '46fd5667422b3d6275e5a9d60fe39fa3684d757e6e32d535ce904fb249ce68be',
            'pd_shallow.py': '555c8e78ee3577867994c6b1cab5ba77b63b9bf865bd1a755b676d63a3e4721c'}
OV_GIT = {'__init__.py': 'docs/scripts/p6_233/overlay/twinstar_sgl__init__.py'} | {
    n: f'infra/sglang/twinstar_sgl/{n}' for n in OV_FILES if n != '__init__.py'}
FROZEN_TS = f'{W}/results/r16-pd-20261014/overlay/helper-runtime/helpers/twinstar_sgl'
P3_TS = f'{W}/src/ssm219-p3helper-e2f2a76f5f2/helpers/twinstar_sgl'
P6_D_ENV = {'SGLANG_GDN_PD_BOUNDARY_ASYNC': 'ordered'}
patches = []


def sha256(p):
    return hashlib.file_digest(open(p, 'rb'), 'sha256').hexdigest()


def patch(rel, old, new, count=1):
    p = OUT / rel
    t = p.read_text()
    assert t.count(old) == count, (rel, t.count(old), count, old[:80])
    p.write_text(t.replace(old, new))
    patches.append(dict(file=rel, old=old[:100], new=new[:100], count=count))


def hunks(a, b):
    """(removed, added) lines of a -> b."""
    d = list(difflib.unified_diff(Path(a).read_text().splitlines(), Path(b).read_text().splitlines(), lineterm='', n=0))
    return ([x[1:] for x in d if x.startswith('-') and not x.startswith('---')],
            [x[1:] for x in d if x.startswith('+') and not x.startswith('+++')])


def freeze_overlay():
    src = OV_SRC / 'twinstar_sgl'
    assert {p.name for p in src.iterdir() if p.is_file()} == set(OV_FILES), sorted(src.iterdir())
    assert {n: sha256(src / n) for n in OV_FILES} == OV_FILES
    dst = OV_HOST / 'twinstar_sgl'
    if not dst.exists():
        tmp = ROOT / 'inputs' / (OV_NAME + '.tmp')
        if tmp.exists():
            shutil.rmtree(tmp)
        (tmp / 'twinstar_sgl').mkdir(parents=True)
        for n in OV_FILES:
            shutil.copyfile(src / n, tmp / 'twinstar_sgl' / n)
            os.chmod(tmp / 'twinstar_sgl' / n, 0o444)
        for d in (tmp / 'twinstar_sgl', tmp):
            os.chmod(d, 0o555)
        tmp.rename(OV_HOST)
    assert {p.name for p in dst.iterdir()} == set(OV_FILES), sorted(dst.iterdir())
    assert {n: sha256(dst / n) for n in OV_FILES} == OV_FILES
    assert not any(os.stat(p).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) for p in [dst, *dst.iterdir()])


def main():
    m0 = json.loads((PARENT / 'manifest.json').read_text())
    assert m0['fingerprint'] == PARENT_FP
    freeze_overlay()
    ov = OV_HOST / 'twinstar_sgl'
    # the overlay's __path__ append defaults to the helper-runtime mount (P3 helper on the PC+ instance)
    init = (ov / '__init__.py').read_text()
    assert f"'{HR}/helpers/twinstar_sgl'" in init and '__path__.append' in init
    # pd_shallow: overlay = frozen + P6 hunk; P3 helper = frozen + before_send hunk (prefill sender only)
    rm6, add6 = hunks(f'{FROZEN_TS}/pd_shallow.py', ov / 'pd_shallow.py')
    rm3, add3 = hunks(f'{FROZEN_TS}/pd_shallow.py', f'{P3_TS}/pd_shallow.py')
    assert not rm6 and any('p6_boundary' in x for x in add6), (rm6, add6)
    assert not rm3 and len(add3) == 6 and any('_pfactor_agg_contract' in x for x in add3), (rm3, add3)
    assert any('def before_send' in x for x in Path(f'{P3_TS}/pd_shallow.py').read_text().splitlines()[160:170])
    rmg, addg = hunks(f'{FROZEN_TS}/pd_boundary_graph.py', ov / 'pd_boundary_graph.py')
    assert not rmg and len(addg) == 3 and 'TWINSTAR_P6_MODGATE' in addg[0], addg
    assert sha256(f'{FROZEN_TS}/pd_boundary_graph.py') == sha256(f'{P3_TS}/pd_boundary_graph.py')
    for n in ('p6_boundary.py', 'p6_modgate.py'):
        assert not Path(P3_TS, n).exists() and not Path(FROZEN_TS, n).exists(), n

    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(PARENT, OUT, ignore=shutil.ignore_patterns('__pycache__'))

    # ---- r16_delta: apply_p6d ----
    p = OUT / 'support/r16_delta.py'
    p.write_text(p.read_text() + (
        "\n\n# docs/222 §17 (second C8 window, lead 03:10): PC+ D adds P6 (b) ordered boundary admission (WS-4 overlay,\n"
        "# twinstar 85477ffe4, frozen read-only copy) right after the P3 tree; PC+ P, S, C+, P unchanged.\n"
        f"P6_OVERLAY = {OV_CONT!r}\n"
        f"P6_OVERLAY_FILES = {OV_FILES!r}\n"
        f"P6_D_ENV = {P6_D_ENV!r}\n"
        "P6_ABSENT = ('SGLANG_GDN_PD_BOUNDARY_AB', 'TWINSTAR_P6_MODGATE', 'TWINSTAR_P6_HELPER_BASE',\n"
        "             'TWINSTAR_PD_BOUNDARY_GRAPH_GUARD', 'TWINSTAR_PD_SHALLOW_AUDIT')\n\n\n"
        "def apply_p6d(variant, role, env, setv):\n"
        "    \"\"\"PC+ D -> P6 (b) ordered (overlay after the P3 python + env); every other variant / role untouched.\"\"\"\n"
        "    pp = env['PYTHONPATH'].split(':')\n"
        "    assert P6_OVERLAY not in pp and not set(P6_D_ENV) & set(env), (variant, role)\n"
        "    if (variant, role) != ('PC+', 'D'):\n"
        "        return []\n"
        "    assert not set(P6_ABSENT) & set(env), sorted(set(P6_ABSENT) & set(env))\n"
        "    assert pp.count(P3_SOURCE + '/python') == 1, pp\n"
        "    i = pp.index(P3_SOURCE + '/python')\n"
        "    assert pp[i + 1].endswith('/helper-runtime/helpers'), pp  # the helper the overlay's __path__ extends\n"
        "    pp.insert(i + 1, P6_OVERLAY)\n"
        "    why = 'docs/222 §17: P6 (b) ordered boundary admission on PC+ D (WS-4 overlay 85477ffe4), lead 03:10'\n"
        "    setv('PYTHONPATH', ':'.join(pp), why)\n"
        "    for k, v in P6_D_ENV.items():\n"
        "        setv(k, v, why)\n"
        "    return sorted(P6_D_ENV)\n"))
    patches.append(dict(file='support/r16_delta.py', appended='apply_p6d + constants'))

    # ---- clone_role ----
    patch('support/clone_role.py', "    apply_p3(variant, role, env, setv)  # docs/222 §16\n",
          "    apply_p3(variant, role, env, setv)  # docs/222 §16\n"
          "    from r16_delta import apply_p6d\n    apply_p6d(variant, role, env, setv)  # docs/222 §17\n")

    # ---- runtime_preflight: PC+ D imports twinstar_sgl through the frozen overlay ----
    old = ("spec=importlib.util.find_spec('twinstar_sgl')\n"
           "assert str((helper/'twinstar_sgl').resolve()) in [str(Path(path).resolve()) for path in spec.submodule_search_locations]\n")
    patch('support/runtime_preflight.py', old,
          "import os\n"
          "if env.get('SGLANG_GDN_PD_BOUNDARY_ASYNC'):  # docs/222 §17: PC+ D, P6 (b) overlay right after the P3 tree\n"
          "    assert P3 and env['SGLANG_GDN_PD_BOUNDARY_ASYNC']=='ordered'\n"
          f"    ov=Path({OV_CONT!r})/'twinstar_sgl'\n"
          "    got={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ov.iterdir() if f.is_file()}\n"
          f"    assert got=={OV_FILES!r},got\n"
          "    spec=importlib.util.find_spec('twinstar_sgl')\n"
          "    assert [str(Path(x).resolve()) for x in spec.submodule_search_locations]==[str(ov.resolve())],spec.submodule_search_locations\n"
          "    os.environ.pop('TWINSTAR_P6_HELPER_BASE',None)\n"
          "    if 'TWINSTAR_P6_HELPER_BASE' in env:os.environ['TWINSTAR_P6_HELPER_BASE']=env['TWINSTAR_P6_HELPER_BASE']  # CPU gate fixture only\n"
          "    import twinstar_sgl\n"
          "    assert str((helper/'twinstar_sgl').resolve()) in [str(Path(x).resolve()) for x in twinstar_sgl.__path__],twinstar_sgl.__path__\n"
          "    for mod,where in (('pd_shallow',ov),('p6_boundary',ov),('pd_boundary_graph',ov),('pd_shallow_install',helper/'twinstar_sgl'),\n"
          "                      ('pd_emitter_graph',helper/'twinstar_sgl'),('pd_dense',helper/'twinstar_sgl')):\n"
          "        assert Path(importlib.util.find_spec('twinstar_sgl.'+mod).origin).resolve().parent==where.resolve(),mod\n"
          "else:\n"
          "    assert 'SGLANG_GDN_PD_BOUNDARY_ASYNC' not in env\n"
          "    " + old.replace("\n", "\n    ", 1))

    # ---- prewindow_gate: the D-side canary on PC+, absent elsewhere ----
    patch('support/prewindow_gate.py', "   assert 'The server is fired up and ready to roll!' in log\n",
          "   assert 'The server is fired up and ready to roll!' in log\n"
          "   canary=[line for line in log.splitlines() if 'PD_BOUNDARY_ASYNC_ACTIVE' in line]  # docs/222 §17: P6 (b), once per TP rank\n"
          "   if (variant,role)==('PC+','D'):assert len(canary)==2 and all(line.endswith('PD_BOUNDARY_ASYNC_ACTIVE mode=ordered ab=') for line in canary),canary\n"
          "   else:assert not canary,(role,canary)\n")

    # ---- entry_guard: the frozen overlay ----
    patch('support/entry_guard.py', "if a.mode=='grid':\n",
          "# docs/222 §17: the frozen P6 overlay (exact file set)\n"
          "ov=Path(m['p6_overlay_host'])/'twinstar_sgl'\n"
          "assert {f.name:hashlib.file_digest(f.open('rb'),'sha256').hexdigest() for f in ov.iterdir()}==m['p6_overlay_files'],'P6 overlay changed'\n"
          "if a.mode=='grid':\n")

    # ---- CPU gate ----
    patch('support/r16_cpu_gate.py',
          "                [k for k, v in r16_delta.ROUTE0_P_ENV.items() if base.get(k) != v] if role == 'P' else [])\n",
          "                [k for k, v in r16_delta.ROUTE0_P_ENV.items() if base.get(k) != v] if role == 'P' else\n"
          "                sorted(r16_delta.P6_D_ENV))  # docs/222 §17\n")
    patch('support/r16_cpu_gate.py', "        assert sorted(science_env) == sorted(want), (variant, role, science_env)\n",
          "        assert sorted(science_env) == sorted(want), (variant, role, science_env)\n"
          "        pp = env['PYTHONPATH'].split(':')  # docs/222 §17: the P6 overlay right after the P3 tree, on PC+ D only\n"
          "        if (variant, role) == ('PC+', 'D'):\n"
          "            assert pp[pp.index(r16_delta.P3_SOURCE + '/python') + 1] == r16_delta.P6_OVERLAY, pp\n"
          "            assert env['SGLANG_GDN_PD_BOUNDARY_ASYNC'] == 'ordered'\n"
          "        else:\n"
          "            assert r16_delta.P6_OVERLAY not in pp and 'SGLANG_GDN_PD_BOUNDARY_ASYNC' not in env, (variant, role)\n")
    patch('support/r16_cpu_gate.py',
          "            if wr == 'P0':\n                e.update(r16_delta.ROUTE0_P_ENV)\n",
          "            if wr == 'P0':\n                e.update(r16_delta.ROUTE0_P_ENV)\n"
          "            else:  # docs/222 §17: the real apply_p6d on D0; the overlay extends the /pdp3 P3 helper here\n"
          "                r16_delta.apply_p6d('PC+', 'D', e, lambda k, v, why: e.__setitem__(k, v))\n"
          "                e['TWINSTAR_P6_HELPER_BASE'] = '/pdp3/helper-runtime/helpers/twinstar_sgl'\n")
    patch('support/r16_cpu_gate.py', "result['checks']['runtime_preflight'] = pre\n",
          "result['checks']['runtime_preflight'] = pre\n"
          "# docs/222 §17: P6 (b) on the PC+ D import path (P3 tree, overlay, P3 helper at /pdp3): mode, canary once,\n"
          "# the decode_boundary hook, module origins\n"
          "d0 = json.loads((root / 'preflight-PC+-D0.json').read_text())['environment']\n"
          "ppd = d0['PYTHONPATH'].split(':')\n"
          "i = ppd.index(r16_delta.P3_SOURCE + '/python')\n"
          "assert ppd[i + 1:i + 3] == [r16_delta.P6_OVERLAY, '/pdp3/helper-runtime/helpers'], ppd\n"
          "out = subprocess.run([sys.executable, '-c', 'import importlib.util,inspect,json,logging,sys\\n'\n"
          "    'logging.getLogger().addHandler(logging.StreamHandler(sys.stdout));logging.getLogger().setLevel(logging.INFO)\\n'\n"
          "    'import twinstar_sgl,twinstar_sgl.p6_boundary as b\\n'\n"
          "    'assert b.mode()==\"ordered\" and b.mode()==\"ordered\"\\n'\n"
          "    'import twinstar_sgl.pd_shallow as s\\n'\n"
          "    'assert \"p6_boundary\" in inspect.getsource(s.decode_boundary)\\n'\n"
          "    'o={m:importlib.util.find_spec(\"twinstar_sgl.\"+m).origin for m in (\"pd_shallow\",\"p6_boundary\",\"pd_boundary_graph\",\"pd_shallow_install\",\"pd_emitter_graph\",\"pd_dense\")}\\n'\n"
          "    'print(json.dumps(dict(path=list(twinstar_sgl.__path__),origins=o)))'],\n"
          "    capture_output=True, text=True, timeout=900,\n"
          "    env=dict(os.environ, PYTHONPATH=':'.join(ppd[i:i + 3] + ['/work/pyuser/lib/python3.12/site-packages']),\n"
          "             SGLANG_GDN_PD_BOUNDARY_ASYNC='ordered', TWINSTAR_P6_HELPER_BASE=d0['TWINSTAR_P6_HELPER_BASE']))\n"
          "assert out.returncode == 0, out.stderr[-2000:]\n"
          "lines = out.stdout.strip().splitlines()\n"
          "assert [x for x in lines if 'PD_BOUNDARY_ASYNC_ACTIVE' in x] == ['PD_BOUNDARY_ASYNC_ACTIVE mode=ordered ab='], lines\n"
          "p6 = json.loads(lines[-1])\n"
          "for mod, where in p6['origins'].items():\n"
          "    want = r16_delta.P6_OVERLAY if mod in ('pd_shallow', 'p6_boundary', 'pd_boundary_graph') else '/pdp3/helper-runtime/helpers'\n"
          "    assert where == f'{want}/twinstar_sgl/{mod}.py', (mod, where)\n"
          "result['checks']['p6_import'] = p6\n")

    # ---- admitted recipes: PC+:D0 += P6 env ----
    refs = json.loads((OUT / 'support/admitted-recipes.json').read_text())
    env = refs['PC+:D0']['environment']
    assert not set(P6_D_ENV) & set(env)
    for k in refs:
        assert 'SGLANG_GDN_PD_BOUNDARY_ASYNC' not in refs[k]['environment'], k
    env.update(P6_D_ENV)
    (OUT / 'support/admitted-recipes.json').write_text(json.dumps(refs, indent=2) + '\n')
    patches.append(dict(file='support/admitted-recipes.json', PCplus_D0_added=P6_D_ENV))

    # ---- manifest + fingerprint ----
    files = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
    m = dict(m0)
    fp = 'p6' + hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:14]
    m.update(fingerprint=fp, p3pd_parent_fingerprint=PARENT_FP, files=files,
             p6_overlay_host=str(OV_HOST), p6_overlay_container=OV_CONT, p6_overlay_files=OV_FILES,
             p6_overlay_source=dict(dir=str(OV_SRC), twinstar_commit='85477ffe412b64cd77b0a68dcb37012aa41540a0', paths=OV_GIT),
             p6_d_env=P6_D_ENV, p6_pd_shallow_vs_frozen=dict(overlay_added=add6, p3_added=add3),
             p6_patches=patches)
    (OUT / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
    final = ROOT / f'inputs/pkg-{fp}'
    if final.exists():
        shutil.rmtree(final)
    OUT.rename(final)
    lines = []
    for rel in sorted(files):
        a, b = PARENT / rel, final / rel
        ta = a.read_text(errors='replace').splitlines(True) if a.exists() else []
        tb = b.read_text(errors='replace').splitlines(True)
        if ta != tb:
            lines += difflib.unified_diff(ta, tb, 'p3pd/' + rel, 'p6w2/' + rel)
    (ROOT / f'inputs/pkg-{fp}.diff').write_text(''.join(lines))
    print(fp, len(files), len(patches))


if __name__ == '__main__':
    assert len(sys.argv) == 1
    main()
