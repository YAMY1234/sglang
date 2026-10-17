"""docs/222 §16 (lead 21:51 / 22:55 / 23:22 pre-authorisation; integration 10-17): 1P:1D C8 formal window package in
which the both arm PC+ serves P3 route 0 ("0 + deferred", env_service_route0.sh) exactly as the PD serving gate
check-p3svc0b-6fb0b992d93 passed it:

  PC+ P and D  : P3 tree 6fb0b992d93 (frozen capsule, git archive) + P3 helper (e2f2a76 base + pd_emitter_graph.py
                 99125ac43 single-file bind) -- the package's pd_dense.py overlay is kept (inert for factored state)
  PC+ P only   : the env of p3/env_service_route0.sh 0 (d35b0fc1097 == 6fb0b992d93) on top of the frozen PC+ P env
  S, C+, P     : untouched (fp32-pd v5 receipts)

Parent: fp32-pd v5 pkg-f329df728b6b135b (S = S_fp32 stock, C+ / PC+ fp16 factors, no FACTOR_DTYPE).  Patch style and
points follow integration's derive_ssm219pd.py (every text patch asserts its exact hit count):
  campaign  fp32-pd-20261015 -> p3pd-20261017
  r16_delta apply_p3(variant, role, env, setv): PC+ source capsule (both roles), route-0 env (P role)
  clone_role per-variant V4_SHA; apply_p3 after the R16 apply; source_and_helper_unchanged = variant != PC+
  runtime_preflight  two authorised trees; PC+ helper = frozen helper-source.json + three authorised overrides
  entry_guard  P3 capsule catalog, P3 helper base digest, pd_emitter_graph sha
  job.sh    PC+ instance (P and D) mounts the P3 helper; cpu mode also mounts it at /pdp3/helper-runtime
  CPU gate  PC+ science env, native_launch A on the P3 tree + helper, runtime_preflight on rendered PC+ P0 / D0,
            ServerArgs parse of the PC+ clones (argcheck)
  admitted-recipes  PC+:P0 += route-0 env
Usage (CMH login):  python3 derive_p3pd.py <capsule catalog sha256> <capsule file count>
"""
import difflib
import hashlib
import json
import shutil
import sys
from pathlib import Path

W = '/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911'
OLD, NEW = 'fp32-pd-20261015', 'p3pd-20261017'
PARENT = Path(f'{W}/results/{OLD}/inputs/pkg-f329df728b6b135b')
ROOT = Path(f'{W}/results/{NEW}')
OUT = ROOT / 'inputs/pkg-work'
P3_SHA = '6fb0b992d9381e995fed7bd73a652c3c218bc875'
CAP_NAME = 'src-p3-own-6fb0b992d93'
CAP_HOST = ROOT / 'inputs' / CAP_NAME
CAP_CONT = f'/work/results/{NEW}/inputs/{CAP_NAME}'
OLD_SHA = '7a841e4189445d108085423c3d6054bd9a7b486c'
OLD_SOURCE = '/work/results/remeasure-v6-fivearm-tp1-20261004/inputs/source-42bbc39e-capsule-f28999a8'
HR = '/work/results/remeasure-v6-fivearm-tp1-20261004/inputs/helper-runtime'
FROZEN_HELPER_HOST = f'{W}/results/r16-pd-20261014/overlay/helper-runtime'
P3_HELPER_HOST = f'{W}/src/ssm219-p3helper-e2f2a76f5f2'
P3_GRAPH_HOST = f'{W}/src/p3-helper-99125ac43/pd_emitter_graph.py'
GRAPH_BLOB = '1b2345d1a0b66c133f15bea75288954ac108b91f'
# p3/env_service_route0.sh 0 at d35b0fc1097 (unchanged at 6fb0b992d93), sourced in `env -i` (SGLANG_ / TWINSTAR_ exports)
ROUTE0_P_ENV = {'SGLANG_FLASHNEXT_PD_TRUNK_PREFILL_GRAPH': '0', 'SGLANG_GDN_FACTORED_HOST_SYNC_FREE': '1',
                'SGLANG_GDN_K31_CHOLQR_MIXED': '0', 'SGLANG_GDN_K31_FUSED': '0', 'SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED': '1',
                'SGLANG_GDN_PD_PUBLISH_JOIN_OFFLOAD': '1', 'SGLANG_GDN_PD_SHALLOW_FULLN': '1',
                'SGLANG_GDN_PREFILL_AGG_CONTRACT': '1', 'TWINSTAR_PD_EMITTER_GRAPH': '1'}
patches = []


def sha256(p):
    return hashlib.file_digest(open(p, 'rb'), 'sha256').hexdigest()


def git_blob(p):
    data = Path(p).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def helpers_digest(root):
    """sha256 over (path, sha256) of every file under <root>/helpers (what PYTHONPATH sees)."""
    h = hashlib.sha256()
    for p in sorted(Path(root, 'helpers').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            h.update(f'{p.relative_to(root)}\0{sha256(p)}\n'.encode())
    return h.hexdigest()


def patch(rel, old, new, count=1):
    p = OUT / rel
    t = p.read_text()
    assert t.count(old) == count, (rel, t.count(old), count, old[:80])
    p.write_text(t.replace(old, new))
    patches.append(dict(file=rel, old=old[:100], new=new[:100], count=count))


def main(cat_sha, cat_n):
    cat_n = int(cat_n)
    assert (CAP_HOST / 'SHA').read_text().strip() == P3_SHA
    assert sha256(CAP_HOST / 'CANDIDATE_FILE_SHA256.json') == cat_sha
    assert len(json.loads((CAP_HOST / 'CANDIDATE_FILE_SHA256.json').read_text())) == cat_n
    assert git_blob(P3_GRAPH_HOST) == GRAPH_BLOB, git_blob(P3_GRAPH_HOST)
    frozen_manifest = Path(FROZEN_HELPER_HOST, 'helper-source.json')
    assert sha256(frozen_manifest) == '980aad64464b75fdb5d9d6f292b0540bd8b405438df958d4d440ec934d700eee'
    # the P3 helper base differs from the frozen helper in exactly pd_shallow.py (and, by the bind, pd_emitter_graph.py)
    frozen = json.loads(frozen_manifest.read_text())['files']
    p3_over = {}
    for name, digest in frozen.items():
        if name.startswith('helpers/') and Path(P3_HELPER_HOST, name).is_file():
            d = sha256(Path(P3_HELPER_HOST, name))
            if d != digest:
                p3_over[name] = d
    graph = 'helpers/twinstar_sgl/pd_emitter_graph.py'
    p3_over[graph] = sha256(P3_GRAPH_HOST)
    assert set(p3_over) == {'helpers/twinstar_sgl/pd_shallow.py', graph}, sorted(p3_over)
    base_digest = helpers_digest(P3_HELPER_HOST)
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(PARENT, OUT, ignore=shutil.ignore_patterns('__pycache__'))

    # ---- campaign paths (same files / counts as derive_ssm219pd.py on this parent) ----
    for rel, n in [('support/job.sh', 7), ('support/archive_node.py', 5), ('support/trio.sh', 1),
                   ('support/entry_guard.py', 1), ('support/r16_cpu_gate.py', 1), ('support/prelude/S.json', 1),
                   ('support/prelude/Cplus.json', 1), ('support/prelude/P.json', 1), ('support/prelude/PCplus.json', 1),
                   ('support/knee-binding.json', 1)]:
        patch(rel, OLD, NEW, n)

    # ---- r16_delta: apply_p3 ----
    p = OUT / 'support/r16_delta.py'
    p.write_text(p.read_text() + (
        "\n\n# docs/222 §16 (P3 route 0 C8 window): PC+ serves the P3 tree + helper on P and D; route-0 env on P only.\n"
        f"OLD_SOURCE, OLD_SHA = {OLD_SOURCE!r}, {OLD_SHA!r}\n"
        f"P3_SOURCE, P3_SHA = {CAP_CONT!r}, {P3_SHA!r}\n"
        f"P3_CATALOG_SHA256, P3_CATALOG_N = {cat_sha!r}, {cat_n}\n"
        f"ROUTE0_P_ENV = {ROUTE0_P_ENV!r}\n"
        "SOURCE_SHA = {'S': OLD_SHA, 'C+': OLD_SHA, 'P': OLD_SHA, 'PC+': P3_SHA}\n\n\n"
        "def apply_p3(variant, role, env, setv):\n"
        "    \"\"\"PC+ -> P3 capsule (both roles) + p3/env_service_route0.sh 0 (P role); S / C+ / P untouched.\"\"\"\n"
        "    if variant != 'PC+':\n"
        "        return []\n"
        "    assert env['V4_SOURCE'] == OLD_SOURCE and env['FINAL_FORK'] == OLD_SOURCE and env['V4_SHA'] == OLD_SHA\n"
        "    pp = env['PYTHONPATH'].split(':')\n"
        "    assert pp.count(OLD_SOURCE + '/python') == 1, pp\n"
        "    pp[pp.index(OLD_SOURCE + '/python')] = P3_SOURCE + '/python'\n"
        "    why = 'docs/222 §16: PC+ serves the P3 tree 6fb0b992d93 (PD serving gate check-p3svc0b PASS)'\n"
        "    setv('V4_SOURCE', P3_SOURCE, why)\n"
        "    setv('FINAL_FORK', P3_SOURCE, why)\n"
        "    setv('V4_SHA', P3_SHA, why)\n"
        "    setv('PYTHONPATH', ':'.join(pp), why)\n"
        "    if role != 'P':\n"
        "        return []\n"
        "    for k, v in ROUTE0_P_ENV.items():\n"
        "        setv(k, v, 'docs/222 §16: p3/env_service_route0.sh 0 (route 0 + deferred), P role only')\n"
        "    return sorted(ROUTE0_P_ENV)\n"))
    patches.append(dict(file='support/r16_delta.py', appended='apply_p3 + constants'))

    # ---- clone_role ----
    patch('support/clone_role.py', "    r16_flags = r16_apply(variant, argv, env, setv)\n",
          "    r16_flags = r16_apply(variant, argv, env, setv)\n"
          "    from r16_delta import apply_p3\n    apply_p3(variant, role, env, setv)  # docs/222 §16\n")
    patch('support/clone_role.py', "    assert env['V4_SHA'] == '7a841e4189445d108085423c3d6054bd9a7b486c'\n",
          "    from r16_delta import SOURCE_SHA\n    assert env['V4_SHA'] == SOURCE_SHA[variant]  # docs/222 §16\n")
    patch('support/clone_role.py', "source_and_helper_unchanged=True,", "source_and_helper_unchanged=(variant != 'PC+'),")

    # ---- runtime_preflight: two trees; PC+ helper against the frozen record + the authorised P3 overrides ----
    patch('support/runtime_preflight.py',
          "assert recipe['source_sha']=='7a841e4189445d108085423c3d6054bd9a7b486c'\n"
          "expected_manifest='256951278686462ffb28c5d471f65d250aa23421b55ff789d918895ad01440a1'\n",
          "# docs/222 §16: exactly two authorised trees (stock capsule / P3 capsule for PC+)\n"
          "ALLOWED={'7a841e4189445d108085423c3d6054bd9a7b486c':('256951278686462ffb28c5d471f65d250aa23421b55ff789d918895ad01440a1',9379),\n"
          f"         {P3_SHA!r}:({cat_sha!r},{cat_n})}}\n"
          "assert recipe['source_sha'] in ALLOWED,recipe['source_sha']\n"
          "expected_manifest,expected_files=ALLOWED[recipe['source_sha']]\n")
    patch('support/runtime_preflight.py', "assert len(source_files)==9379\n", "assert len(source_files)==expected_files\n")
    patch('support/runtime_preflight.py', "helper_manifest=helper.parent/'helper-source.json'\n",
          "helper_manifest=helper.parent/'helper-source.json'\n"
          f"P3=recipe['source_sha']=={P3_SHA!r}\n"
          "if P3:  # the P3 helper base carries no record of its own: check it against the frozen helper's record\n"
          "    helper_manifest=Path('/work/results/r16-pd-20261014/overlay/helper-runtime/helper-source.json')\n")
    patch('support/runtime_preflight.py', "assert set(AUTHORIZED)<=set(helper_files),AUTHORIZED\n",
          f"if P3:AUTHORIZED.update({p3_over!r})  # docs/222 §16: P3 pd_shallow.py (e2f2a76) + pd_emitter_graph.py (99125ac43)\n"
          "assert set(AUTHORIZED)<=set(helper_files),AUTHORIZED\n")

    # ---- entry_guard: the P3 capsule, helper base and graph file ----
    patch('support/entry_guard.py', "if a.mode=='grid':\n",
          "# docs/222 §16: PC+ inputs (P3 capsule catalog, helper base, pd_emitter_graph bind)\n"
          "cs=Path(m['p3_source_host']);assert (cs/'SHA').read_text().strip()==m['p3_source_sha'];cc=cs/'CANDIDATE_FILE_SHA256.json'\n"
          "assert hashlib.file_digest(cc.open('rb'),'sha256').hexdigest()==m['p3_catalog_sha256']\n"
          "cf=json.loads(cc.read_text());assert len(cf)==m['p3_catalog_files']\n"
          "for name,digest in cf.items():assert hashlib.file_digest((cs/name).open('rb'),'sha256').hexdigest()==digest,name\n"
          "assert hashlib.file_digest(open(m['p3_graph_host'],'rb'),'sha256').hexdigest()==m['p3_graph_sha256']\n"
          "hb=Path(m['p3_helper_host']);hd=hashlib.sha256()\n"
          "for f in sorted((hb/'helpers').rglob('*')):\n"
          " if f.is_file() and '__pycache__' not in f.parts:hd.update(f'{f.relative_to(hb)}\\0{hashlib.file_digest(f.open(\"rb\"),\"sha256\").hexdigest()}\\n'.encode())\n"
          "assert hd.hexdigest()==m['p3_helper_digest'],'P3 helper base changed'\n"
          "if a.mode=='grid':\n")

    # ---- job.sh: PC+ instance (P and D nodes) mounts the P3 helper; cpu mode also exposes it at /pdp3 ----
    frozen_mount = f"{FROZEN_HELPER_HOST}:{HR}:ro"
    patch('support/job.sh', frozen_mount, "${PDP3_HELPER_MOUNT}")
    patch('support/job.sh', f"${{PARITY_BUNDLE}}/support/pd_dense.py:{HR}/helpers/twinstar_sgl/pd_dense.py:ro\" \\\n",
          f"${{PARITY_BUNDLE}}/support/pd_dense.py:{HR}/helpers/twinstar_sgl/pd_dense.py:ro${{PDP3_TAIL}}\" \\\n")
    patch('support/job.sh', " exec srun ${TRIO_SRUN_ARGS",
          " # docs/222 §16: the PC+ instance serves the P3 helper (e2f2a76 base + pd_emitter_graph 99125ac43 bind) on P and D\n"
          f" PDP3_HELPER_MOUNT='{frozen_mount}'; PDP3_TAIL=''\n"
          " if [[ ${V6_VARIANT:-} == PC+ && $PDREF_MODE == grid ]]; then\n"
          f"  PDP3_HELPER_MOUNT='{P3_HELPER_HOST}:{HR}:ro'; PDP3_TAIL=',{P3_GRAPH_HOST}:{HR}/helpers/twinstar_sgl/pd_emitter_graph.py:ro'\n"
          " fi\n"
          " if [[ $PDREF_MODE == cpu ]]; then  # the CPU gate checks the PC+ launches against the P3 helper at /pdp3\n"
          f"  PDP3_TAIL=\",{P3_HELPER_HOST}:/pdp3/helper-runtime:ro,{P3_GRAPH_HOST}:/pdp3/helper-runtime/helpers/twinstar_sgl/pd_emitter_graph.py:ro,${{PARITY_BUNDLE}}/support/pd_dense.py:/pdp3/helper-runtime/helpers/twinstar_sgl/pd_dense.py:ro\"\n"
          " fi\n"
          " exec srun ${TRIO_SRUN_ARGS")

    # ---- CPU gate ----
    patch('support/r16_cpu_gate.py',
          "        assert science_env == (['SGLANG_DUET_DIR'] if '--duet-release' in EXPECTED[variant] else []), (variant, role, science_env)\n",
          "        want = (['SGLANG_DUET_DIR'] if '--duet-release' in EXPECTED[variant] else [])\n"
          "        if variant == 'PC+':  # docs/222 §16\n"
          "            base = json.loads(Path(f'/pdrefresh/baselines/PC+/role-{role}-actual-launch.json').read_text())['environment']\n"
          "            want = want + ['V4_SOURCE', 'FINAL_FORK', 'V4_SHA', 'PYTHONPATH'] + (\n"
          "                [k for k, v in r16_delta.ROUTE0_P_ENV.items() if base.get(k) != v] if role == 'P' else [])\n"
          "        assert sorted(science_env) == sorted(want), (variant, role, science_env)\n"
          "        if variant == 'PC+':  # argcheck: the P3 tree's ServerArgs parses the full clone argv; sglang resolves to it\n"
          "            ac = subprocess.run([sys.executable, '-c', 'import json,sys,argparse,os,sglang;'\n"
          "                'from sglang.srt.server_args import ServerArgs;a=json.loads(sys.argv[1]);i=a.index(\"sglang.launch_server\");'\n"
          "                'p=argparse.ArgumentParser();ServerArgs.add_cli_args(p);ns=p.parse_args(a[i+1:]);'\n"
          "                'print(json.dumps(dict(sglang=os.path.dirname(sglang.__file__),tokens=len(a)-i-1,mode=ns.disaggregation_mode)))',\n"
          "                json.dumps(argv)], capture_output=True, text=True, timeout=900,\n"
          "                env=dict(os.environ, PYTHONPATH=r16_delta.P3_SOURCE + '/python:/work/pyuser/lib/python3.12/site-packages'))\n"
          "            assert ac.returncode == 0, (variant, role, ac.stderr[-2000:])\n"
          "            got = json.loads(ac.stdout.strip().splitlines()[-1])\n"
          "            assert got['sglang'] == r16_delta.P3_SOURCE + '/python/sglang', got\n"
          "            result.setdefault('checks', {}).setdefault('argcheck_pcplus', {})[role] = got\n")
    patch('support/r16_cpu_gate.py',
          "        subprocess.run([sys.executable, '/pdrefresh/native_launch.py', arm, str(out)], stdout=log,",
          "        subprocess.run([sys.executable, '/pdrefresh/native_launch.py', arm, str(out)],\n"
          "                       env=(dict(os.environ, PYTHONPATH=os.environ['PYTHONPATH'].replace(\n"
          "                           r16_delta.OLD_SOURCE + '/python', r16_delta.P3_SOURCE + '/python').replace(\n"
          f"                           '{HR}/helpers', '/pdp3/helper-runtime/helpers')) if arm == 'A' else None),\n"
          "                       stdout=log,")
    patch('support/r16_cpu_gate.py', "for v in ('S', 'P'):\n", "for v in ('S', 'P', 'PC+'):\n")
    patch('support/r16_cpu_gate.py',
          "        rec = json.loads(Path(f'/pdrefresh/prewindow-fixtures/{v}/workers/{wr}/runtime-recipe.json').read_text())\n",
          "        if v == 'PC+':  # docs/222 §16: the PC+ recipe as clone_role renders it (P3 tree; P3 helper at /pdp3)\n"
          "            rec = json.loads(Path(f'/pdrefresh/baselines/PC+/role-{wr[0]}-runtime-recipe.json').read_text())\n"
          "            e = rec['environment']\n"
          "            assert e['PYTHONPATH'].count(r16_delta.OLD_SOURCE + '/python') == 1\n"
          f"            e['PYTHONPATH'] = e['PYTHONPATH'].replace(r16_delta.OLD_SOURCE + '/python', r16_delta.P3_SOURCE + '/python').replace('{HR}/helpers', '/pdp3/helper-runtime/helpers')\n"
          "            e.update(V4_SOURCE=r16_delta.P3_SOURCE, FINAL_FORK=r16_delta.P3_SOURCE, V4_SHA=r16_delta.P3_SHA,\n"
          "                     V4_HELPER='/pdp3/helper-runtime/helpers')\n"
          "            if wr == 'P0':\n"
          "                e.update(r16_delta.ROUTE0_P_ENV)\n"
          "            rec['source_sha'] = r16_delta.P3_SHA\n"
          "        else:\n"
          "            rec = json.loads(Path(f'/pdrefresh/prewindow-fixtures/{v}/workers/{wr}/runtime-recipe.json').read_text())\n")

    # ---- admitted recipes: route-0 env on PC+:P0 ----
    refs = json.loads((OUT / 'support/admitted-recipes.json').read_text())
    env = refs['PC+:P0']['environment']
    before = {k: env.get(k) for k in ROUTE0_P_ENV}
    env.update(ROUTE0_P_ENV)
    (OUT / 'support/admitted-recipes.json').write_text(json.dumps(refs, indent=2) + '\n')
    patches.append(dict(file='support/admitted-recipes.json', PCplus_P0_before=before, PCplus_P0_after=ROUTE0_P_ENV))

    # ---- manifest + fingerprint ----
    files = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
    m = json.loads((PARENT / 'manifest.json').read_text())
    fp = 'p3' + hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:14]
    m.update(campaign=NEW, fingerprint=fp, fp32pd_parent_fingerprint=m['fingerprint'], files=files,
             p3_source_host=str(CAP_HOST), p3_source_container=CAP_CONT, p3_source_sha=P3_SHA,
             p3_catalog_sha256=cat_sha, p3_catalog_files=cat_n, p3_helper_host=P3_HELPER_HOST,
             p3_helper_digest=base_digest, p3_graph_host=P3_GRAPH_HOST, p3_graph_sha256=sha256(P3_GRAPH_HOST),
             p3_graph_blob=GRAPH_BLOB, p3_helper_overrides=p3_over, p3_route0_p_env=ROUTE0_P_ENV,
             p3_patches=patches)
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
            lines += difflib.unified_diff(ta, tb, 'fp32pd/' + rel, 'p3pd/' + rel)
    (ROOT / f'inputs/pkg-{fp}.diff').write_text(''.join(lines))
    print(fp, len(files), len(patches))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
