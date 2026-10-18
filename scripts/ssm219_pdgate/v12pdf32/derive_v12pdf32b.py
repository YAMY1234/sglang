"""docs/222 §24 (lead 21:4x / 21:5x via integration): PD both_fp32 arm PC+f32c, N2 {S, PC+f32c} windows on LHR.
Parent: pkg-v12f326d361c149c (tonight's N3 {S, C+f32c, C+k15} package; C+f32c / C+k15 unchanged here).

  PC+f32c (trio token; V6_VARIANT PC+, F32_ARM f32, run label PCpf32c) = the v12pd PC+ PD recipe (P3 route 0 + P3 helper
           e2f2a76 + pd_emitter_graph 99125ac43, U1) served from the frozen capsule of WS-3's merged tree (P3 6fb0b992d93
           merged into d45106d8b54) instead of the P3 tree, with
    P role : the route-0 env exactly as before (K31_FUSED=0, CHOLQR_MIXED=0, PD_SHALLOW_FULLN=1, ...) + TWINSTAR_FACTOR_DTYPE=fp32
             + SGLANG_GDN_FACTORED_CUDA_FP32=1 (as tonight's C+f32c P);
    D role : the 15 decode-side keys of the (d)-gated env + TWINSTAR_FACTOR_DTYPE=fp32 + SGLANG_GDN_FACTORED_CUDA_FP32=1.
  The D role's baseline has SGLANG_GDN_AGG_FULLN_PREFILL / _COMPACT_BUFFERS=0; the 15 keys set them to 1 (as for C+).
  A bare PC+ token is refused (this package serves PC+ only as PC+f32c).
  gate     : prewindow factor-dtype gate (dtype=fp32 / factors float32, no BUILD_FAILED) on both PC+ roles.
  cpu gate : clone science env per role; ServerArgs parse + sglang resolution on the merged capsule; native config admission
             (A, fp32); runtime_preflight on rendered PC+ P0 / D0 with the P3 helper; U1 on the merged capsule; CUDA_FP32 and
             selectors read; PD_SHALLOW_FULLN present in the merged tree's code.
Usage (CMH or LHR login, after the capsule is frozen):  python3 derive_v12pdf32b.py <capsule dir name> <40-hex sha>
"""
import difflib
import os
import hashlib
import json
import shutil
import sys
from pathlib import Path

W = '/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911'
ROOT = Path(f'{W}/results/p3pd-20261017')
PARENT_FP = 'v12f326d361c149c'
PARENT = ROOT / f'inputs/pkg-{PARENT_FP}'
OUT = ROOT / 'inputs/pkg-work-v12pdf32b'
patches = []


def sha256(p):
    return hashlib.file_digest(open(p, 'rb'), 'sha256').hexdigest()


def patch(rel, old, new, count=1):
    p = OUT / rel
    t = p.read_text()
    assert t.count(old) == count, (rel, t.count(old), count, old[:80])
    p.write_text(t.replace(old, new))
    patches.append(dict(file=rel, old=old[:100], new=new[:100], count=count))


def main(name, sha):
    host = ROOT / 'inputs' / name
    cont = f'/work/results/p3pd-20261017/inputs/{name}'
    assert (host / 'SHA').read_text().strip() == sha and len(sha) == 40
    cat = sha256(host / 'CANDIDATE_FILE_SHA256.json')
    n = len(json.loads((host / 'CANDIDATE_FILE_SHA256.json').read_text()))
    dry = os.environ.get('DERIVE_DRYRUN') == '1'  # patch-application self-test on a stand-in capsule; nothing is kept
    if not dry:
        assert any('SGLANG_GDN_PD_SHALLOW_FULLN' in p.read_text(errors='replace') for p in (host / 'python/sglang').rglob('*.py'))
        assert any('SGLANG_GDN_FACTORED_CUDA_FP32' in p.read_text(errors='replace') for p in (host / 'python/sglang').rglob('*.py'))
    m0 = json.loads((PARENT / 'manifest.json').read_text())
    assert m0['fingerprint'] == PARENT_FP
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(PARENT, OUT, ignore=shutil.ignore_patterns('__pycache__'))

    # ---- r16_delta: PC+f32c ----
    p = OUT / 'support/r16_delta.py'
    p.write_text(p.read_text() + (
        "\n\n# docs/222 §24 (lead 21:4x / 21:5x): PC+f32c = PC+ (route 0, P3 helper) on the merged capsule; P route-0 + fp32,\n"
        "# D 15 keys + fp32.\n"
        f"PCF32_SOURCE, PCF32_SHA = {cont!r}, {sha!r}\n"
        f"PCF32_CATALOG_SHA256, PCF32_CATALOG_N = {cat!r}, {n}\n"
        "SOURCE_SHA = {'S': OLD_SHA, 'C+': F32_SHA, 'P': OLD_SHA, 'PC+': PCF32_SHA}\n"
        "_apply_f32_cplus = apply_f32\n\n\n"
        "def pcf32_role_env(role):\n"
        "    return dict(F32_EXTRA) if role == 'P' else f32_env('f32')\n\n\n"
        "def apply_f32(variant, role, env, setv, f32arm):  # noqa: F811  (C+ / S / P: the §23 function)\n"
        "    if variant != 'PC+':\n"
        "        return _apply_f32_cplus(variant, role, env, setv, f32arm)\n"
        "    assert f32arm == 'f32', ('docs/222 §24: PC+ is served only as PC+f32c', f32arm)\n"
        "    assert env['V4_SOURCE'] == P3_SOURCE and env['FINAL_FORK'] == P3_SOURCE and env['V4_SHA'] == P3_SHA  # after apply_p3\n"
        "    want = pcf32_role_env(role)\n"
        "    over = {k: env[k] for k in want if k in env}\n"
        "    assert over == ({} if role == 'P' else {'SGLANG_GDN_AGG_FULLN_PREFILL': '0', 'SGLANG_GDN_AGG_FULLN_COMPACT_BUFFERS': '0'}), (role, over)\n"
        "    pp = env['PYTHONPATH'].split(':')\n"
        "    assert pp.count(P3_SOURCE + '/python') == 1, pp\n"
        "    pp[pp.index(P3_SOURCE + '/python')] = PCF32_SOURCE + '/python'\n"
        "    why = 'docs/222 §24: PC+f32c serves the merged tree (P3 PD full-N into d45106d8b54)'\n"
        "    setv('V4_SOURCE', PCF32_SOURCE, why)\n"
        "    setv('FINAL_FORK', PCF32_SOURCE, why)\n"
        "    setv('V4_SHA', PCF32_SHA, why)\n"
        "    setv('PYTHONPATH', ':'.join(pp), why)\n"
        "    for k, v in want.items():\n"
        "        setv(k, v, 'docs/222 §24 PC+f32c: ' + ('route-0 P + fp32 factors / CUDA fp32' if role == 'P' else\n"
        "                                              '(d)-gated decode-side 15 keys + fp32 factors / CUDA fp32'))\n"
        "    return sorted(want)\n"))
    patches.append(dict(file='support/r16_delta.py', appended='docs/222 §24 PCF32 constants, pcf32_role_env, apply_f32 (PC+)'))

    # ---- trio.sh ----
    patch('support/trio.sh', "    PC+) arm=A;      pocs=1;   fix=1; prelude=PCplus ;;\n",
          "    PC+) echo 'docs/222 §24: PC+ must be PC+f32c in this package'; exit 2 ;;\n"
          "    PC+f32c) v6=PC+; f32arm=f32; arm=A; pocs=1; fix=1; prelude=PCplus ;;\n")

    # ---- runtime_preflight / entry_guard ----
    patch('support/runtime_preflight.py', "ALLOWED={'7a841e4189445d108085423c3d6054bd9a7b486c':",
          f"ALLOWED={{{sha!r}:({cat!r},{n}),  # docs/222 §24\n         '7a841e4189445d108085423c3d6054bd9a7b486c':")
    patch('support/runtime_preflight.py', "P3=recipe['source_sha']=='6fb0b992d9381e995fed7bd73a652c3c218bc875'\n",
          f"P3=recipe['source_sha'] in ('6fb0b992d9381e995fed7bd73a652c3c218bc875',{sha!r})  # docs/222 §24: merged tree serves the P3 helper\n")
    patch('support/entry_guard.py', "if a.mode=='grid':\n",
          "# docs/222 §24: the merged capsule (PC+f32c)\n"
          "ms=Path(m['pcf32_source_host']);assert (ms/'SHA').read_text().strip()==m['pcf32_source_sha'];mc=ms/'CANDIDATE_FILE_SHA256.json'\n"
          "assert hashlib.file_digest(mc.open('rb'),'sha256').hexdigest()==m['pcf32_catalog_sha256']\n"
          "mf=json.loads(mc.read_text());assert len(mf)==m['pcf32_catalog_files']\n"
          "for name,digest in mf.items():assert hashlib.file_digest((ms/name).open('rb'),'sha256').hexdigest()==digest,name\n"
          "if a.mode=='grid':\n")

    # ---- prewindow: the factor-dtype gate also on PC+ ----
    patch('support/prewindow_gate.py', "   if variant=='C+':\n    f32=f32_role_gate(",
          "   if variant in ('C+','PC+'):  # docs/222 §24\n    f32=f32_role_gate(")

    # ---- admitted recipes: PC+:D0 takes the ruled values of the two overridden keys; P0 unchanged ----
    refs = json.loads((OUT / 'support/admitted-recipes.json').read_text())
    env = refs['PC+:D0']['environment']
    hit = {k: env[k] for k in m0['f32_k15_env'] if k in env}
    assert hit == {'SGLANG_GDN_AGG_FULLN_PREFILL': '0', 'SGLANG_GDN_AGG_FULLN_COMPACT_BUFFERS': '0'}, hit
    for k in hit:
        env[k] = m0['f32_k15_env'][k]
    pe = refs['PC+:P0']['environment']
    assert not set(m0['f32_extra_env']) & set(pe) and not set(m0['f32_extra_env']) & set(env)
    (OUT / 'support/admitted-recipes.json').write_text(json.dumps(refs, indent=2) + '\n')
    patches.append(dict(file='support/admitted-recipes.json', PCplus_D0_overrides={k: [v, m0['f32_k15_env'][k]] for k, v in hit.items()}))

    # ---- CPU gate ----
    cg = 'support/r16_cpu_gate.py'
    patch(cg, "('P', ''), ('PC+', '')):  # docs/222 §23\n    for role, devices in",
          "('P', ''), ('PC+', 'f32')):  # docs/222 §23 / §24\n    for role, devices in")
    patch(cg, "        if variant == 'C+':  # docs/222 §23: tree swap + the 15 keys (+ fp32 keys)\n",
          "        if variant == 'PC+':  # docs/222 §24: P3 tree -> merged tree; P + fp32 keys, D + 15 keys + fp32 keys\n"
          "            want = want + sorted(r16_delta.pcf32_role_env(role))\n"
          "            assert all(env[k] == v for k, v in r16_delta.pcf32_role_env(role).items()) and env['V4_SHA'] == r16_delta.PCF32_SHA\n"
          "            assert env['PYTHONPATH'].split(':').count(r16_delta.PCF32_SOURCE + '/python') == 1, env['PYTHONPATH']\n"
          "            if role == 'P':\n"
          "                assert all(env[k] == v for k, v in r16_delta.ROUTE0_P_ENV.items()), 'route-0 P env'\n"
          "        if variant == 'C+':  # docs/222 §23: tree swap + the 15 keys (+ fp32 keys)\n")
    patch(cg, "            tree = r16_delta.P3_SOURCE if variant == 'PC+' else r16_delta.F32_SOURCE  # docs/222 §23\n",
          "            tree = r16_delta.PCF32_SOURCE if variant == 'PC+' else r16_delta.F32_SOURCE  # docs/222 §23 / §24\n")
    patch(cg, "            r16_delta.OLD_SOURCE + '/python', r16_delta.P3_SOURCE + '/python').replace(\n"
              "            '/work/results/remeasure-v6-fivearm-tp1-20261004/inputs/helper-runtime/helpers', '/pdp3/helper-runtime/helpers'))\n",
          "            r16_delta.OLD_SOURCE + '/python', r16_delta.PCF32_SOURCE + '/python').replace(  # docs/222 §24\n"
          "            '/work/results/remeasure-v6-fivearm-tp1-20261004/inputs/helper-runtime/helpers', '/pdp3/helper-runtime/helpers'),\n"
          "            **r16_delta.F32_EXTRA)\n")
    patch(cg, "    if arm == 'A':\n        assert receipt['state'] == FROZEN.format(rel=r16_delta.NEW_RELEASE), receipt['state']\n",
          "    if arm == 'A':  # docs/222 §24: merged tree, fp32 factors (no 15 keys in the native config: ring 16)\n"
          "        assert receipt['state'] == FROZEN.format(rel=r16_delta.NEW_RELEASE).replace('dtype=fp16', 'dtype=fp32'), receipt['state']\n")
    patch(cg, "(('S', ''), ('P', ''), ('PC+', ''), ('C+', 'f32'), ('C+', 'k15')):  # docs/222 §23\n",
          "(('S', ''), ('P', ''), ('PC+', 'f32'), ('C+', 'f32'), ('C+', 'k15')):  # docs/222 §23 / §24\n")
    patch(cg, "e['PYTHONPATH'].replace(r16_delta.OLD_SOURCE + '/python', r16_delta.P3_SOURCE + '/python')",
          "e['PYTHONPATH'].replace(r16_delta.OLD_SOURCE + '/python', r16_delta.PCF32_SOURCE + '/python')")
    patch(cg, "e.update(V4_SOURCE=r16_delta.P3_SOURCE, FINAL_FORK=r16_delta.P3_SOURCE, V4_SHA=r16_delta.P3_SHA,",
          "e.update(V4_SOURCE=r16_delta.PCF32_SOURCE, FINAL_FORK=r16_delta.PCF32_SOURCE, V4_SHA=r16_delta.PCF32_SHA,")
    patch(cg, "            rec['source_sha'] = r16_delta.P3_SHA\n",
          "            e.update(r16_delta.pcf32_role_env(wr[0]))  # docs/222 §24\n"
          "            rec['source_sha'] = r16_delta.PCF32_SHA\n")
    patch(cg, "for tree in (r16_delta.OLD_SOURCE, r16_delta.P3_SOURCE, r16_delta.F32_SOURCE):  # docs/222 §23\n",
          "for tree in (r16_delta.OLD_SOURCE, r16_delta.P3_SOURCE, r16_delta.F32_SOURCE, r16_delta.PCF32_SOURCE):  # docs/222 §23 / §24\n")
    patch(cg, "result['checks']['f32'] = f32chk\n",
          "result['checks']['f32'] = f32chk\n"
          "# docs/222 §24: the merged capsule reads the same fp32 admission / selectors, and carries P3's PD full-N switch\n"
          "out = subprocess.run([sys.executable, '-c', 'import json,os,sglang\\n'\n"
          "                      'import sglang.srt.layers.attention.linear.kernels.gdn_factored_norm as n\\n'\n"
          "                      'import sglang.srt.layers.attention.linear.kernels.gdn_factored as g\\n'\n"
          "                      'print(json.dumps(dict(sglang=os.path.dirname(sglang.__file__), cuda_fp32=n.CUDA_FP32, fuse_norm=n.enabled(),\\n'\n"
          "                      '    impl32=g.STEP_IMPL_BY_RMAX[32], batches=sorted(g.STEP_CUDA_BATCHES))))'],\n"
          "                     capture_output=True, text=True, timeout=1200,\n"
          "                     env=dict(os.environ, PYTHONPATH=r16_delta.PCF32_SOURCE + '/python:/work/pyuser/lib/python3.12/site-packages',\n"
          "                              REMEASURE_SOURCE_ONLY='1', **r16_delta.f32_env('f32')))\n"
          "assert out.returncode == 0, out.stderr[-3000:]\n"
          "got = json.loads(out.stdout.strip().splitlines()[-1])\n"
          "assert got['sglang'] == r16_delta.PCF32_SOURCE + '/python/sglang' and got['cuda_fp32'] is True and got['impl32'] == 'cuda', got\n"
          "fulln = sorted(str(p.relative_to(r16_delta.PCF32_SOURCE)) for p in Path(r16_delta.PCF32_SOURCE, 'python/sglang').rglob('*.py')\n"
          "               if 'SGLANG_GDN_PD_SHALLOW_FULLN' in p.read_text(errors='replace'))\n"
          "assert fulln, 'merged tree lacks SGLANG_GDN_PD_SHALLOW_FULLN'\n"
          "result['checks']['pcf32'] = dict(got, pd_shallow_fulln_files=fulln)\n")
    patch(cg, "F32_ARM='f32' if variant == 'C+' else '')  # docs/222 §23\n",
          "F32_ARM='f32' if variant in ('C+', 'PC+') else '')  # docs/222 §23 / §24\n")

    # ---- manifest + fingerprint ----
    files = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
    m = dict(m0)
    fp = 'v12f32b' + hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:9]
    m.update(fingerprint=fp, v12f32_parent_fingerprint=PARENT_FP, files=files, pcf32_source_host=str(host),
             pcf32_source_container=cont, pcf32_source_sha=sha, pcf32_catalog_sha256=cat, pcf32_catalog_files=n,
             f32_arms=dict(m0['f32_arms'], **{'PC+f32c': dict(V6_VARIANT='PC+', F32_ARM='f32', label='PCpf32c', table_arm='both_fp32')}),
             pcf32_patches=patches)
    (OUT / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
    final = ROOT / f'inputs/pkg-{fp}'
    if dry:
        shutil.rmtree(OUT)
        print('DRYRUN-OK', fp, len(files), len(patches))
        return
    assert not final.exists(), final
    OUT.rename(final)
    lines = []
    for rel in sorted(files):
        a, b = PARENT / rel, final / rel
        ta = a.read_text(errors='replace').splitlines(True) if a.exists() else []
        tb = b.read_text(errors='replace').splitlines(True)
        if ta != tb:
            lines += difflib.unified_diff(ta, tb, 'v12f32/' + rel, 'v12f32b/' + rel)
    (ROOT / f'inputs/pkg-{fp}.diff').write_text(''.join(lines))
    changed = sorted({l[4:].split()[0] for l in lines if l.startswith('+++ ')})
    print(fp, len(files), len(patches), changed)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
