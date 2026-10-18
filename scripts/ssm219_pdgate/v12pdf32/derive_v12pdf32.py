"""docs/222 §23 (lead 19:2x / 19:3x via integration): LHR 1P:1D fp32-factor row for the all-fp32 main figure.
Windows N3 {S, C+f32c, C+k15}, 9 C, 900 s.  Both factor arms are V6_VARIANT=C+ (same baselines, admitted recipes and
prelude) told apart by F32_ARM (trio.sh variant tokens C+f32c / C+k15 -> run labels Cpf32c / Cpk15):

  C+f32c : the v12pd C+ PD recipe + the 15 decode-side keys of the (d)-gated C+f32x env + TWINSTAR_FACTOR_DTYPE=fp32
           + SGLANG_GDN_FACTORED_CUDA_FP32=1, P and D.
  C+k15  : the same without the two fp32 keys (fp16 control).
  tree   : both serve the frozen capsule src-f32step-d45106d8b54 (git archive of d45106d8b54: fp32 gates a / b / c / d
           PASS on CMH; the stock capsule 7a841e4189 is its ancestor, 104 commits, srt/disaggregation unchanged).  The
           window container mounts /work/src from stable-src, so the tree lives under the campaign inputs.
  S      : unchanged (S_fp32 stock).  P / PC+ stay defined as in the parent but are not part of this row.
Two of the 15 keys override explicit PD-baseline values (SGLANG_GDN_AGG_FULLN_PREFILL / _COMPACT_BUFFERS 0 -> 1); the
admitted C+ recipes take the ruled values.  Every text patch asserts its exact hit count.
  gate     : before the formal window, every C+ role: pool line `dtype=fp32 ... factors float32` (f32) or `dtype=fp16 ...
             factors float16` (k15), no FACTORED_*_BUILD_FAILED; activation lines counted (canary checks D, reports P).
  cpu gate : clone science env = the authorised keys only (both modes); ServerArgs parse + sglang resolution on the
             capsule; native config admission on the capsule per mode; runtime_preflight on rendered C+ P0 / D0; U1 on
             the capsule; CUDA_FP32 / selectors read on CPU per mode.
Usage (CMH or LHR login):  python3 derive_v12pdf32.py
"""
import difflib
import hashlib
import json
import shutil
import sys
from pathlib import Path

W = '/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911'
ROOT = Path(f'{W}/results/p3pd-20261017')
PARENT_FP = 'v12pded3ebbe77fd'
PARENT = ROOT / f'inputs/pkg-{PARENT_FP}'
OUT = ROOT / 'inputs/pkg-work-v12pdf32'
F32_SHA = 'd45106d8b54cf7091be1a6b62f233b3c587f5806'
F32_NAME = 'src-f32step-d45106d8b54'
F32_HOST = ROOT / 'inputs' / F32_NAME
F32_CONT = f'/work/results/p3pd-20261017/inputs/{F32_NAME}'
F32_CAT, F32_N = 'b4ce993e851bd06f1b63e42df3fbdd79e2a764dd80a891e91cb369e542a8806d', 9421
# the 15 decode-side keys of the (d)-gated C+f32x env (formal j4579798 launch.json; docs/222 §22 prep.sh `s2`)
K15_ENV = {'SGLANG_GDN_K31_MIXED_EIGH_ANY_BATCH': '1', 'SGLANG_GDN_AGG_FULLN_PREFILL': '1', 'SGLANG_GDN_AGG_FULLN_OVERLAP_OK': '1',
           'SGLANG_GDN_AGG_FULLN_COMPACT_BUFFERS': '1', 'SGLANG_GDN_AGG_BATCH_PUBLISH_DEFERRED': '1',
           'SGLANG_GDN_FACTORED_BOOKKEEPING_FOLD': '1', 'SGLANG_GDN_FACTORED_STEP_IMPL32': 'cuda',
           'SGLANG_GDN_FACTORED_STEP_CUDA_BATCHES': '1,2,3,4,5,6,7,8', 'SGLANG_GDN_PREFIX_RESTORE_FUSED': '1',
           'SGLANG_GDN_FACTORED_TRUNC_CUDA': '1', 'SGLANG_GDN_K31_FUSED': '1', 'SGLANG_GDN_FACTORED_RING': '2',
           'SGLANG_GDN_FULLN_CKPT_N_MINUS_1': '1', 'SGLANG_GDN_FACTORED_STEP_FUSE_NORM': '1', 'SGLANG_GDN_FULLN_CKPT_FP32_SNAPSHOT': '1'}
F32_EXTRA = {'TWINSTAR_FACTOR_DTYPE': 'fp32', 'SGLANG_GDN_FACTORED_CUDA_FP32': '1'}
patches = []


def sha256(p):
    return hashlib.file_digest(open(p, 'rb'), 'sha256').hexdigest()


def patch(rel, old, new, count=1):
    p = OUT / rel
    t = p.read_text()
    assert t.count(old) == count, (rel, t.count(old), count, old[:80])
    p.write_text(t.replace(old, new))
    patches.append(dict(file=rel, old=old[:100], new=new[:100], count=count))


NATIVE_NEW = """native = {}
for arm, f32arm in (('stock', ''), ('factor', 'f32'), ('factor', 'k15'), ('arm2', ''), ('A', '')):  # docs/222 §23
    out = root / f'native-{arm}{f32arm}.json'
    if arm == 'A':
        nenv = dict(os.environ, PYTHONPATH=os.environ['PYTHONPATH'].replace(
            r16_delta.OLD_SOURCE + '/python', r16_delta.P3_SOURCE + '/python').replace(
            '/work/results/remeasure-v6-fivearm-tp1-20261004/inputs/helper-runtime/helpers', '/pdp3/helper-runtime/helpers'))
    elif arm == 'factor':
        nenv = dict(os.environ, PYTHONPATH=os.environ['PYTHONPATH'].replace(
            r16_delta.OLD_SOURCE + '/python', r16_delta.F32_SOURCE + '/python'), **r16_delta.f32_env(f32arm))
    else:
        nenv = None
    with (root / f'native-{arm}{f32arm}.log').open('w') as log:
        subprocess.run([sys.executable, '/pdrefresh/native_launch.py', arm, str(out)], env=nenv, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=1800)
    receipt = json.loads(out.read_text())
    assert receipt['passed'] and receipt['release_argument'] == r16_delta.spec(r16_delta.ARM_VARIANT[arm])[0]
    if arm == 'A':
        assert receipt['state'] == FROZEN.format(rel=r16_delta.NEW_RELEASE), receipt['state']
    if arm == 'factor':  # the f32 capsule with the 15 keys: ring 2; dtype by mode
        want = FROZEN.format(rel=r16_delta.NEW_RELEASE).replace('ring=16', 'ring=2')
        if f32arm == 'f32':
            want = want.replace('dtype=fp16', 'dtype=fp32')
        assert receipt['state'] == want, (f32arm, receipt['state'], want)
    if arm in ('factor', 'A'):
        assert Path(r16_delta.NEW_RELEASE, '.cache/state_sink_vbar.pt').is_file()
    native[arm + f32arm] = dict(state=receipt['state'], effective_rank=receipt['effective'].get('gdn_rank'),
                       effective_every=receipt['effective'].get('gdn_every'),
                       release=receipt['release_argument'], name=receipt['server_info_schema'].get('name'),
                       spec=receipt['server_info_schema'].get('spec'))
result['checks']['native'] = native
"""


def main():
    m0 = json.loads((PARENT / 'manifest.json').read_text())
    assert m0['fingerprint'] == PARENT_FP
    assert (F32_HOST / 'SHA').read_text().strip() == F32_SHA
    assert sha256(F32_HOST / 'CANDIDATE_FILE_SHA256.json') == F32_CAT
    assert len(json.loads((F32_HOST / 'CANDIDATE_FILE_SHA256.json').read_text())) == F32_N
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(PARENT, OUT, ignore=shutil.ignore_patterns('__pycache__'))

    # ---- r16_delta: apply_f32 + f32 gate ----
    p = OUT / 'support/r16_delta.py'
    p.write_text(p.read_text() + (
        "\n\n# docs/222 §23 (lead 19:2x / 19:3x): C+f32c / C+k15 = V6_VARIANT C+ with F32_ARM f32 / k15 on the f32step capsule.\n"
        f"F32_SOURCE, F32_SHA = {F32_CONT!r}, {F32_SHA!r}\n"
        f"F32_CATALOG_SHA256, F32_CATALOG_N = {F32_CAT!r}, {F32_N}\n"
        f"K15_ENV = {K15_ENV!r}\n"
        f"F32_EXTRA = {F32_EXTRA!r}\n"
        "F32_ARMS = ('f32', 'k15')\n"
        "SOURCE_SHA = {'S': OLD_SHA, 'C+': F32_SHA, 'P': OLD_SHA, 'PC+': P3_SHA}\n"
        "F32_ACTIVE = ('FACTORED_CUDA_FP32_ACTIVE step', 'FACTORED_CUDA_FP32_ACTIVE expiry', 'FACTORED_STEP_CUDA_ACTIVE',\n"
        "              'FACTORED_STEP_FUSE_NORM_ACTIVE', 'FACTORED_TRUNC_CUDA_ACTIVE')\n\n\n"
        "def f32_env(f32arm):\n"
        "    assert f32arm in F32_ARMS, f32arm\n"
        "    return dict(K15_ENV, **(F32_EXTRA if f32arm == 'f32' else {}))\n\n\n"
        "def apply_f32(variant, role, env, setv, f32arm):\n"
        "    \"\"\"C+ -> f32step capsule (both roles) + the 15 keys (+ fp32 keys for f32); S / P / PC+ untouched.\"\"\"\n"
        "    if variant != 'C+':\n"
        "        assert not f32arm, (variant, f32arm)\n"
        "        return []\n"
        "    want = f32_env(f32arm)\n"
        "    assert env['V4_SOURCE'] == OLD_SOURCE and env['FINAL_FORK'] == OLD_SOURCE and env['V4_SHA'] == OLD_SHA\n"
        "    # the only baseline keys the ruling overrides: the agg-contract pair (explicit 0 in every 4569012 C+ role)\n"
        "    assert {k: env[k] for k in want if k in env} == {'SGLANG_GDN_AGG_FULLN_PREFILL': '0', 'SGLANG_GDN_AGG_FULLN_COMPACT_BUFFERS': '0'}, \\\n"
        "        {k: env[k] for k in want if k in env}\n"
        "    pp = env['PYTHONPATH'].split(':')\n"
        "    assert pp.count(OLD_SOURCE + '/python') == 1, pp\n"
        "    pp[pp.index(OLD_SOURCE + '/python')] = F32_SOURCE + '/python'\n"
        "    why = 'docs/222 §23: C+f32c / C+k15 serve the f32step tree d45106d8b54 (fp32 gates a-d PASS on CMH)'\n"
        "    setv('V4_SOURCE', F32_SOURCE, why)\n"
        "    setv('FINAL_FORK', F32_SOURCE, why)\n"
        "    setv('V4_SHA', F32_SHA, why)\n"
        "    setv('PYTHONPATH', ':'.join(pp), why)\n"
        "    for k, v in want.items():\n"
        "        setv(k, v, f'docs/222 §23 ({f32arm}): (d)-gated decode-side 15 keys' + (' + fp32 factors / CUDA fp32' if f32arm == 'f32' else ''))\n"
        "    return sorted(want)\n\n\n"
        "def f32_role_gate(variant, role, log, f32arm):\n"
        "    lines = log.splitlines()\n"
        "    pool = next((l for l in lines if 'Factored GDN pool allocated' in l), '')\n"
        "    failed = [l[-300:] for l in lines if 'BUILD_FAILED' in l]\n"
        "    dt, fl = ('fp32', 'float32') if f32arm == 'f32' else ('fp16', 'float16')\n"
        "    g = dict(variant=variant, role=role, f32arm=f32arm, factored_line=pool[pool.find('Factored'):][:400],\n"
        "             build_failed=failed, active={k: sum(k in l for l in lines) for k in F32_ACTIVE},\n"
        "             traceback=sum('Traceback' in l for l in lines))\n"
        "    g['passed'] = f'dtype={dt}' in pool and f'factors {fl}' in pool and not failed\n"
        "    return g\n"))
    patches.append(dict(file='support/r16_delta.py', appended='docs/222 §23 constants, f32_env, apply_f32, f32_role_gate'))

    # ---- clone_role ----
    patch('support/clone_role.py', "    from r16_delta import apply_u1\n",
          "    from r16_delta import apply_f32\n    apply_f32(variant, role, env, setv, inherited.get('F32_ARM', ''))  # docs/222 §23\n"
          "    from r16_delta import apply_u1\n")
    patch('support/clone_role.py', "source_and_helper_unchanged=(variant != 'PC+'),",
          "source_and_helper_unchanged=(variant not in ('C+', 'PC+')),")

    # ---- trio.sh: the two C+ arm tokens ----
    patch('support/trio.sh', "  v=${VARS[$i]}; lab=${v//+/p}\n",
          "  v=${VARS[$i]}; lab=${v//+/p}; v6=$v; f32arm=''\n")
    patch('support/trio.sh', "    C+)  arm=factor; pocs=1;   fix=1; prelude=Cplus ;;\n",
          "    C+)  echo 'docs/222 §23: C+ must be C+f32c or C+k15 in this package'; exit 2 ;;\n"
          "    C+f32c) v6=C+; f32arm=f32; arm=factor; pocs=1; fix=1; prelude=Cplus ;;\n"
          "    C+k15)  v6=C+; f32arm=k15; arm=factor; pocs=1; fix=1; prelude=Cplus ;;\n")
    patch('support/trio.sh', "    export V6_VARIANT=$v PSIDE_ARM=$arm PARITY_ARM_LABEL=$arm\n",
          "    export V6_VARIANT=$v6 F32_ARM=$f32arm PSIDE_ARM=$arm PARITY_ARM_LABEL=$arm\n")

    # ---- runtime_preflight: the f32 capsule is the third authorised tree ----
    patch('support/runtime_preflight.py',
          "ALLOWED={'7a841e4189445d108085423c3d6054bd9a7b486c':('256951278686462ffb28c5d471f65d250aa23421b55ff789d918895ad01440a1',9379),\n",
          "ALLOWED={'7a841e4189445d108085423c3d6054bd9a7b486c':('256951278686462ffb28c5d471f65d250aa23421b55ff789d918895ad01440a1',9379),\n"
          f"         {F32_SHA!r}:({F32_CAT!r},{F32_N}),  # docs/222 §23\n")

    # ---- entry_guard: the f32 capsule catalog ----
    patch('support/entry_guard.py', "if a.mode=='grid':\n",
          "# docs/222 §23: the f32step capsule (C+f32c / C+k15)\n"
          "fs=Path(m['f32_source_host']);assert (fs/'SHA').read_text().strip()==m['f32_source_sha'];fc=fs/'CANDIDATE_FILE_SHA256.json'\n"
          "assert hashlib.file_digest(fc.open('rb'),'sha256').hexdigest()==m['f32_catalog_sha256']\n"
          "ff=json.loads(fc.read_text());assert len(ff)==m['f32_catalog_files']\n"
          "for name,digest in ff.items():assert hashlib.file_digest((fs/name).open('rb'),'sha256').hexdigest()==digest,name\n"
          "if a.mode=='grid':\n")

    # ---- prewindow_gate: factor dtype by mode + no CUDA build failure on C+ ----
    patch('support/prewindow_gate.py', "   ready=json.loads((w/'ready-contract.json').read_text());",
          "   from r16_delta import f32_role_gate  # docs/222 §23\n"
          "   if variant=='C+':\n"
          "    f32=f32_role_gate(variant,role,log,os.environ['F32_ARM']);out.setdefault('f32',{})[role]=f32\n"
          "    (root/('f32-gate-'+role+'.json')).write_text(json.dumps(f32,indent=2)+'\\n');assert f32['passed'],f32\n"
          "   ready=json.loads((w/'ready-contract.json').read_text());")

    # ---- admitted recipes: C+ takes the ruled values of the two overridden keys ----
    refs = json.loads((OUT / 'support/admitted-recipes.json').read_text())
    over = {}
    for key in ('C+:P0', 'C+:D0'):
        env = refs[key]['environment']
        hit = {k: env[k] for k in K15_ENV if k in env}
        assert hit == {'SGLANG_GDN_AGG_FULLN_PREFILL': '0', 'SGLANG_GDN_AGG_FULLN_COMPACT_BUFFERS': '0'}, (key, hit)
        assert not set(F32_EXTRA) & set(env)
        for k in hit:
            env[k] = K15_ENV[k]
        over[key] = {k: [v, K15_ENV[k]] for k, v in hit.items()}
    (OUT / 'support/admitted-recipes.json').write_text(json.dumps(refs, indent=2) + '\n')
    patches.append(dict(file='support/admitted-recipes.json', Cplus_overrides=over))

    # ---- CPU gate ----
    cg = 'support/r16_cpu_gate.py'
    patch(cg, "rows = []\nfor variant in ['S', 'C+', 'P', 'PC+']:\n    for role, devices in (('P', '0,1'), ('D', '2,3')):\n"
              "        w = root / f'clone-{variant}-{role}'\n"
              "        inherited = dict(os.environ, V4_RUN_ID='jcpu', PSIDE_C='32', V4_COORD_ROOT=str(root / 'coord'))\n",
          "rows = []\nfor variant, f32arm in (('S', ''), ('C+', 'f32'), ('C+', 'k15'), ('P', ''), ('PC+', '')):  # docs/222 §23\n"
          "    for role, devices in (('P', '0,1'), ('D', '2,3')):\n"
          "        w = root / f'clone-{variant}{f32arm}-{role}'\n"
          "        inherited = dict(os.environ, V4_RUN_ID='jcpu', PSIDE_C='32', V4_COORD_ROOT=str(root / 'coord'), F32_ARM=f32arm)\n")
    patch(cg, "        want = sorted(set(want) | {'PYTHONPATH', 'SGLANG_AB_OUT', 'SGLANG_GDN_U1A_RESULT_BEFORE_EXTEND'})",
          "        if variant == 'C+':  # docs/222 §23: tree swap + the 15 keys (+ fp32 keys)\n"
          "            want = want + ['V4_SOURCE', 'FINAL_FORK', 'V4_SHA'] + sorted(r16_delta.f32_env(f32arm))\n"
          "            assert all(env[k] == v for k, v in r16_delta.f32_env(f32arm).items()) and env['V4_SHA'] == r16_delta.F32_SHA\n"
          "            assert ('TWINSTAR_FACTOR_DTYPE' in env) == (f32arm == 'f32') and ('SGLANG_GDN_FACTORED_CUDA_FP32' in env) == (f32arm == 'f32')\n"
          "            assert env['PYTHONPATH'].split(':').count(r16_delta.F32_SOURCE + '/python') == 1, env['PYTHONPATH']\n"
          "        want = sorted(set(want) | {'PYTHONPATH', 'SGLANG_AB_OUT', 'SGLANG_GDN_U1A_RESULT_BEFORE_EXTEND'})")
    patch(cg, "        if variant == 'PC+':  # argcheck: the P3 tree's ServerArgs parses the full clone argv; sglang resolves to it\n",
          "        if variant in ('PC+', 'C+'):  # argcheck: the served tree's ServerArgs parses the full clone argv; sglang resolves to it\n"
          "            tree = r16_delta.P3_SOURCE if variant == 'PC+' else r16_delta.F32_SOURCE  # docs/222 §23\n")
    patch(cg, "env=dict(os.environ, PYTHONPATH=r16_delta.P3_SOURCE + '/python:/work/pyuser/lib/python3.12/site-packages'))\n",
          "env=dict(os.environ, PYTHONPATH=tree + '/python:/work/pyuser/lib/python3.12/site-packages'))\n")
    patch(cg, "            assert got['sglang'] == r16_delta.P3_SOURCE + '/python/sglang', got\n"
              "            result.setdefault('checks', {}).setdefault('argcheck_pcplus', {})[role] = got\n",
          "            assert got['sglang'] == tree + '/python/sglang', got\n"
          "            result.setdefault('checks', {}).setdefault('argcheck', {})[f'{variant}{f32arm}/{role}'] = got\n")
    patch(cg, "        rows.append(dict(variant=variant, role=role,", "        rows.append(dict(variant=variant, f32arm=f32arm, role=role,")
    t = (OUT / cg).read_text()
    a = t.index("native = {}\nfor arm in ['stock', 'factor', 'arm2', 'A']:\n")
    b = t.index("result['checks']['native'] = native\n") + len("result['checks']['native'] = native\n")
    assert t.count("native = {}\n") == 1
    (OUT / cg).write_text(t[:a] + NATIVE_NEW + t[b:])
    patches.append(dict(file=cg, replaced='native admission block (f32 capsule per mode)'))
    patch(cg, "for v in ('S', 'P', 'PC+'):\n    for wr in ('P0', 'D0'):\n",
          "for v, f32arm in (('S', ''), ('P', ''), ('PC+', ''), ('C+', 'f32'), ('C+', 'k15')):  # docs/222 §23\n    for wr in ('P0', 'D0'):\n")
    patch(cg, "        if v == 'PC+':  # docs/222 §16: the PC+ recipe as clone_role renders it (P3 tree; P3 helper at /pdp3)\n",
          "        if v == 'C+':  # docs/222 §23: the C+ recipe as clone_role renders it (f32 capsule; frozen helper)\n"
          "            rec = json.loads(Path(f'/pdrefresh/baselines/C+/role-{wr[0]}-runtime-recipe.json').read_text())\n"
          "            e = rec['environment']\n"
          "            assert e['PYTHONPATH'].count(r16_delta.OLD_SOURCE + '/python') == 1\n"
          "            e['PYTHONPATH'] = e['PYTHONPATH'].replace(r16_delta.OLD_SOURCE + '/python', r16_delta.F32_SOURCE + '/python')\n"
          "            e.update(V4_SOURCE=r16_delta.F32_SOURCE, FINAL_FORK=r16_delta.F32_SOURCE, V4_SHA=r16_delta.F32_SHA,\n"
          "                     **r16_delta.f32_env(f32arm))\n"
          "            rec['source_sha'] = r16_delta.F32_SHA\n"
          "        elif v == 'PC+':  # docs/222 §16: the PC+ recipe as clone_role renders it (P3 tree; P3 helper at /pdp3)\n")
    patch(cg, "        tmp = root / f'preflight-{v}-{wr}.json'\n", "        tmp = root / f'preflight-{v}{f32arm}-{wr}.json'\n")
    patch(cg, "        pre.append(dict(variant=v, role=wr,", "        pre.append(dict(variant=v + f32arm, role=wr,")
    patch(cg, "for tree in (r16_delta.OLD_SOURCE, r16_delta.P3_SOURCE):\n",
          "for tree in (r16_delta.OLD_SOURCE, r16_delta.P3_SOURCE, r16_delta.F32_SOURCE):  # docs/222 §23\n")
    patch(cg, "result['checks']['u1'] = u1\n",
          "result['checks']['u1'] = u1\n"
          "# docs/222 §23: on the f32 capsule the fp32 admission and the kernel selectors read as each mode sets them (CPU import)\n"
          "f32chk = {}\n"
          "for f32arm in r16_delta.F32_ARMS:\n"
          "    out = subprocess.run([sys.executable, '-c', 'import json,os,sglang\\n'\n"
          "                          'import sglang.srt.layers.attention.linear.kernels.gdn_factored_norm as n\\n'\n"
          "                          'import sglang.srt.layers.attention.linear.kernels.gdn_factored as g\\n'\n"
          "                          'import sglang.srt.layers.attention.linear.kernels.gdn_factored_expiry_cuda as x\\n'\n"
          "                          'print(json.dumps(dict(sglang=os.path.dirname(sglang.__file__), cuda_fp32=n.CUDA_FP32, fuse_norm=n.enabled(),\\n'\n"
          "                          '    impl32=g.STEP_IMPL_BY_RMAX[32], batches=sorted(g.STEP_CUDA_BATCHES), expiry_variant=x.VARIANT)))'],\n"
          "                         capture_output=True, text=True, timeout=1200,\n"
          "                         env=dict(os.environ, PYTHONPATH=r16_delta.F32_SOURCE + '/python:/work/pyuser/lib/python3.12/site-packages',\n"
          "                                  REMEASURE_SOURCE_ONLY='1', **r16_delta.f32_env(f32arm)))\n"
          "    assert out.returncode == 0, (f32arm, out.stderr[-3000:])\n"
          "    got = json.loads(out.stdout.strip().splitlines()[-1])\n"
          "    assert got['sglang'] == r16_delta.F32_SOURCE + '/python/sglang' and got['cuda_fp32'] is (f32arm == 'f32'), got\n"
          "    assert got['fuse_norm'] is True and got['impl32'] == 'cuda' and got['batches'] == list(range(1, 9)) and got['expiry_variant'] == 0, got\n"
          "    f32chk[f32arm] = got\n"
          "result['checks']['f32'] = f32chk\n")
    patch(cg, "for variant in ('S', 'C+', 'PC+'):\n    for role in ('P', 'D'):\n"
              "        inh = dict(os.environ, V4_RUN_ID='jcpu-fab', PSIDE_C='8', V4_COORD_ROOT=str(root / 'coord'), P12_ARM='1')\n",
          "for variant in ('S', 'C+', 'PC+'):\n    for role in ('P', 'D'):\n"
          "        inh = dict(os.environ, V4_RUN_ID='jcpu-fab', PSIDE_C='8', V4_COORD_ROOT=str(root / 'coord'), P12_ARM='1',\n"
          "                   F32_ARM='f32' if variant == 'C+' else '')  # docs/222 §23\n")

    # ---- manifest + fingerprint ----
    files = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
    m = dict(m0)
    fp = 'v12f32' + hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:10]
    m.update(fingerprint=fp, v12pd_parent_fingerprint=PARENT_FP, files=files, f32_source_host=str(F32_HOST),
             f32_source_container=F32_CONT, f32_source_sha=F32_SHA, f32_catalog_sha256=F32_CAT, f32_catalog_files=F32_N,
             f32_k15_env=K15_ENV, f32_extra_env=F32_EXTRA,
             f32_arms={'C+f32c': dict(V6_VARIANT='C+', F32_ARM='f32', label='Cpf32c', table_arm='factor_fp32'),
                       'C+k15': dict(V6_VARIANT='C+', F32_ARM='k15', label='Cpk15', table_arm='factor_k15')},
             f32_patches=patches)
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
            lines += difflib.unified_diff(ta, tb, 'v12pd/' + rel, 'v12f32/' + rel)
    (ROOT / f'inputs/pkg-{fp}.diff').write_text(''.join(lines))
    changed = sorted({l[4:].split()[0] for l in lines if l.startswith('+++ ')})
    print(fp, len(files), len(patches), changed)


if __name__ == '__main__':
    assert len(sys.argv) == 1
    main()
