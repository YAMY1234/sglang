"""docs/222 §20 (lead 08:23 / 08:56, integration 10-17): P12 job package = the j4569012 launch package
pkg-p30305a3bb41198b with the PC+ fork checkout replaced, P12's switch on PC+ P, the gate-only proof switches, and the
one-node (pd1n) placement of the unchanged per-arm harness.

  PC+ P and D : P3 tree 6fb0b992d93 -> P12 tree 51d15dce06c (frozen capsule src-p12-51d15dce06c: git archive of the
                fork commit, every blob checked against `git ls-tree`, byte-equal to integration's
                /work/src/ssm219-p12-51d15dce06c); helper deb51f492f5 + p3helper e2f2a76f5f2 + pd_emitter_graph 99125ac43,
                helper-runtime, final-extra, release v3.5-b-r2, Flash-Next staging 4 x 1024 MiB, TP2, instrumentation:
                unchanged (the parent's job.sh / clone_role / mounts)
  PC+ P only  : SGLANG_GDN_PD_SHALLOW_SEND_PIPELINE = $P12_ARM (= p3/env_p12.sh 0|1 after the frozen PC+ env; apply_p12
                asserts env_p12.sh's recipe precondition; the CPU gate runs the real env_p12.sh on the rendered env)
  gate runs   : P12_PROOF=positive|event|abort -> both roles SGLANG_FLASHNEXT_PD_STAGING_PROOF_DIR=
                /work/results/p3pd-20261017/runs/<run>/p12-proof/<role> + ..._PROOF_TENSORS=1; P: WIRE_REPLAY=1 (positive)
                or WIRE_NEGATIVE=<mode>; P12_PROOF unset (timing) -> none of the four keys, asserted again before the window
  S, C+, P    : unchanged (C+ = the micro's same-run reference)
  pd1n        : P12_ONE_NODE=1 -> job.sh task 1 (D, chosen by SLURM_PROCID as before) keeps its own node-local root
                /tmp/p3pd-20261017/<run>-dnode, as on a second node (the P task copies D0 files into its own workers/D0)
  window      : P12_WINDOW_SECONDS in (300, 600, 900) (harness_mode; 900 = the window)
  gate driver : entry.py runs support/p12_gate_driver.py on the P task when P12_PROOF is set (no router / AgentX)

Points (each text patch asserts its exact hit count):
  r16_delta apply_p12 + constants; clone_role apply_p12 after apply_p3; runtime_preflight P12 catalog + P3 helper rule;
  entry_guard P12 capsule catalog; prewindow_gate P12 switch / canary / no debug key; harness_mode; job.sh; entry.py;
  CPU gate (PC+ tree = P12, science env + the key, renders of every arm x proof mode, real env_p12.sh, the tree reads it)
Usage (CMH login):  python3 derive_p12.py <capsule catalog sha256> <capsule file count>
"""
import difflib
import hashlib
import json
import shutil
import sys
from pathlib import Path

W = '/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911'
ROOT = Path(f'{W}/results/p3pd-20261017')
PARENT_FP = 'p30305a3bb41198b'
PARENT = ROOT / f'inputs/pkg-{PARENT_FP}'
OUT = ROOT / 'inputs/pkg-work-p12'
P12_SHA = '51d15dce06ce2d709484fd1a697ffbba1449e40a'
CAP_NAME = 'src-p12-51d15dce06c'
CAP_HOST = ROOT / 'inputs' / CAP_NAME
CAP_CONT = f'/work/results/p3pd-20261017/inputs/{CAP_NAME}'
STAGED = Path(f'{W}/src/ssm219-p12-51d15dce06c')  # integration's staging of the same commit
# integration 10-17 (P3 delivery): key blobs (git sha1 prefixes)
BLOBS = {'p3/P12.md': '63abb1c63951', 'p3/p12_wire_client.py': 'de70d9ea2e5a', 'p3/p12_wire_check.py': '86467d379f0a',
         'python/sglang/srt/disaggregation/flashnext_staging_transport.py': 'd1c11170610a',
         'python/sglang/srt/disaggregation/pd_shallow_send_proof.py': '14ebb30547b9',
         'python/sglang/srt/disaggregation/flashnext_staging_audit.py': '7494c7aebfaa', 'p3/env_p12.sh': 'cc46aac3f84a'}
HERE = Path(__file__).resolve().parent
patches = []


def sha256(p):
    return hashlib.file_digest(open(p, 'rb'), 'sha256').hexdigest()


def git_blob(p):
    data = Path(p).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def patch(rel, old, new, count=1):
    p = OUT / rel
    t = p.read_text()
    assert t.count(old) == count, (rel, t.count(old), count, old[:80])
    p.write_text(t.replace(old, new))
    patches.append(dict(file=rel, old=old[:100], new=new[:100], count=count))


def main(cat_sha, cat_n):
    cat_n = int(cat_n)
    m0 = json.loads((PARENT / 'manifest.json').read_text())
    assert m0['fingerprint'] == PARENT_FP
    assert (CAP_HOST / 'SHA').read_text().strip() == P12_SHA == (STAGED / 'SHA').read_text().strip()
    assert sha256(CAP_HOST / 'CANDIDATE_FILE_SHA256.json') == cat_sha
    cat = json.loads((CAP_HOST / 'CANDIDATE_FILE_SHA256.json').read_text())
    assert len(cat) == cat_n
    for rel, blob in BLOBS.items():
        assert git_blob(CAP_HOST / rel).startswith(blob) and git_blob(STAGED / rel).startswith(blob), rel
    env_p12 = (CAP_HOST / 'p3/env_p12.sh').read_text()
    assert 'export SGLANG_GDN_PD_SHALLOW_SEND_PIPELINE=$P12_ARM' in env_p12

    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(PARENT, OUT, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(HERE / 'p12_gate_driver.py', OUT / 'support/p12_gate_driver.py')
    patches.append(dict(file='support/p12_gate_driver.py', added=git_blob(HERE / 'p12_gate_driver.py')))

    # ---- r16_delta: apply_p12 ----
    p = OUT / 'support/r16_delta.py'
    p.write_text(p.read_text() + (
        "\n\n# docs/222 §20 (P12 shallow P KV-first send, lead 08:23 / 08:56): PC+ serves the P12 tree 51d15dce06c on P and D\n"
        "# (= the window tree 6fb0b992d93 + P12); P: SGLANG_GDN_PD_SHALLOW_SEND_PIPELINE = P12_ARM (p3/env_p12.sh 0|1);\n"
        "# live wire gates only (P12_PROOF positive|event|abort): proof dir / tensors on both roles, replay / negative on P.\n"
        f"P12_SOURCE, P12_SHA = {CAP_CONT!r}, {P12_SHA!r}\n"
        f"P12_CATALOG_SHA256, P12_CATALOG_N = {cat_sha!r}, {cat_n}\n"
        "P12_KEY = 'SGLANG_GDN_PD_SHALLOW_SEND_PIPELINE'\n"
        "P12_DEBUG = ('SGLANG_FLASHNEXT_PD_STAGING_PROOF_DIR', 'SGLANG_FLASHNEXT_PD_STAGING_PROOF_TENSORS',\n"
        "             'SGLANG_GDN_PD_P12_WIRE_REPLAY', 'SGLANG_GDN_PD_P12_WIRE_NEGATIVE')\n"
        "# p3/env_p12.sh refuses unless the frozen PC+ P recipe is the qualified R1-off shallow full-N / deferred / JOIN one\n"
        "P12_RECIPE = {'SGLANG_GDN_PD_SHALLOW_FULLN': '1', 'SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED': '1',\n"
        "              'SGLANG_GDN_PD_PUBLISH_JOIN_OFFLOAD': '1', 'SGLANG_GDN_K31_FUSED': '0', 'SGLANG_GDN_K31_CHOLQR_MIXED': '0'}\n"
        "P12_PROOF_ROOT = '/work/results/p3pd-20261017/runs/{run}/p12-proof'\n"
        "SOURCE_SHA['PC+'] = P12_SHA\n\n\n"
        "def p12_env(role, arm, proof, run):\n"
        "    \"\"\"The P12 keys of one PC+ role: {key: value}.\"\"\"\n"
        "    out = {P12_KEY: arm} if role == 'P' else {}\n"
        "    if proof:\n"
        "        out['SGLANG_FLASHNEXT_PD_STAGING_PROOF_DIR'] = P12_PROOF_ROOT.format(run=run) + '/' + role\n"
        "        out['SGLANG_FLASHNEXT_PD_STAGING_PROOF_TENSORS'] = '1'\n"
        "        if role == 'P' and proof == 'positive':\n"
        "            out['SGLANG_GDN_PD_P12_WIRE_REPLAY'] = '1'\n"
        "        elif role == 'P':\n"
        "            out['SGLANG_GDN_PD_P12_WIRE_NEGATIVE'] = proof\n"
        "    return out\n\n\n"
        "def apply_p12(variant, role, env, setv, inherited):\n"
        "    \"\"\"PC+ -> P12 capsule (both roles) + the P12 keys (P12_ARM, P12_PROOF of the harness env); others untouched.\"\"\"\n"
        "    assert P12_KEY not in env and not set(P12_DEBUG) & set(env), (variant, role)\n"
        "    arm, proof = inherited.get('P12_ARM'), inherited.get('P12_PROOF', '')\n"
        "    if variant != 'PC+':\n"
        "        assert not proof, (variant, proof)\n"
        "        return []\n"
        "    assert arm in ('0', '1'), ('PC+ needs P12_ARM 0|1', arm)\n"
        "    assert proof in ('', 'positive', 'event', 'abort') and (not proof or arm == '1'), (arm, proof)\n"
        "    assert env['V4_SOURCE'] == P3_SOURCE and env['FINAL_FORK'] == P3_SOURCE and env['V4_SHA'] == P3_SHA\n"
        "    pp = env['PYTHONPATH'].split(':')\n"
        "    assert pp.count(P3_SOURCE + '/python') == 1, pp\n"
        "    pp[pp.index(P3_SOURCE + '/python')] = P12_SOURCE + '/python'\n"
        "    why = 'docs/222 §20: PC+ serves the P12 tree 51d15dce06c (window tree 6fb0b992d93 + P12), lead 08:23'\n"
        "    setv('V4_SOURCE', P12_SOURCE, why)\n"
        "    setv('FINAL_FORK', P12_SOURCE, why)\n"
        "    setv('V4_SHA', P12_SHA, why)\n"
        "    setv('PYTHONPATH', ':'.join(pp), why)\n"
        "    if role == 'P':\n"
        "        assert {k: env.get(k) for k in P12_RECIPE} == P12_RECIPE, {k: env.get(k) for k in P12_RECIPE}\n"
        "    keys = p12_env(role, arm, proof, inherited['V4_RUN_ID'])\n"
        "    for k, v in keys.items():\n"
        "        setv(k, v, 'docs/222 §20: p3/env_p12.sh ' + arm + ' (P only)' if k == P12_KEY else\n"
        "             'docs/222 §20: P12 live wire gate ' + proof + ' only (p3/P12.md), never in a timing run')\n"
        "    return sorted(keys)\n"))
    patches.append(dict(file='support/r16_delta.py', appended='apply_p12 + constants'))

    # ---- clone_role ----
    patch('support/clone_role.py', "    apply_p3(variant, role, env, setv)  # docs/222 §16\n",
          "    apply_p3(variant, role, env, setv)  # docs/222 §16\n"
          "    from r16_delta import apply_p12\n    apply_p12(variant, role, env, setv, inherited)  # docs/222 §20\n")

    # ---- runtime_preflight: the P12 capsule; it serves the P3 helper like the P3 tree ----
    patch('support/runtime_preflight.py',
          "         '6fb0b992d9381e995fed7bd73a652c3c218bc875':('3506e5e243772967387315d7500a8c9f4332c0e60c0b8721a6b2a438bb379d1b',9423)}\n",
          "         '6fb0b992d9381e995fed7bd73a652c3c218bc875':('3506e5e243772967387315d7500a8c9f4332c0e60c0b8721a6b2a438bb379d1b',9423),\n"
          f"         {P12_SHA!r}:({cat_sha!r},{cat_n})}}  # docs/222 §20: P12 capsule\n")
    patch('support/runtime_preflight.py', "P3=recipe['source_sha']=='6fb0b992d9381e995fed7bd73a652c3c218bc875'\n",
          f"P3=recipe['source_sha'] in ('6fb0b992d9381e995fed7bd73a652c3c218bc875',{P12_SHA!r})  # docs/222 §20: P12 serves the same P3 helper\n")

    # ---- entry_guard: the P12 capsule catalog ----
    patch('support/entry_guard.py', "if a.mode=='grid':\n",
          "# docs/222 §20: PC+ serves the P12 capsule (catalog, every file)\n"
          "cs=Path(m['p12_source_host']);assert (cs/'SHA').read_text().strip()==m['p12_source_sha'];cc=cs/'CANDIDATE_FILE_SHA256.json'\n"
          "assert hashlib.file_digest(cc.open('rb'),'sha256').hexdigest()==m['p12_catalog_sha256']\n"
          "cf=json.loads(cc.read_text());assert len(cf)==m['p12_catalog_files']\n"
          "for name,digest in cf.items():assert hashlib.file_digest((cs/name).open('rb'),'sha256').hexdigest()==digest,name\n"
          "if a.mode=='grid':\n")

    # ---- prewindow_gate: P12 switch on PC+ P only, canary, no debug key / line in a timing window ----
    patch('support/prewindow_gate.py', "   assert 'The server is fired up and ready to roll!' in log\n",
          "   assert 'The server is fired up and ready to roll!' in log\n"
          "   from r16_delta import P12_KEY,P12_DEBUG  # docs/222 §20\n"
          "   assert not set(P12_DEBUG)&set(actual['environment']),(role,sorted(set(P12_DEBUG)&set(actual['environment'])))\n"
          "   assert 'P12_WIRE_' not in log,(role,'P12 gate line in a timing run')\n"
          "   pipe=[line for line in log.splitlines() if 'PD_SHALLOW_SEND_PIPELINE_ACTIVE' in line]\n"
          "   if (variant,role)==('PC+','P'):\n"
          "    arm=actual['environment'][P12_KEY];assert arm==os.environ['P12_ARM'],(arm,os.environ.get('P12_ARM'))\n"
          "    assert (pipe and all('PD_SHALLOW_SEND_PIPELINE_ACTIVE staged=True' in line for line in pipe)) if arm=='1' else not pipe,(arm,pipe)\n"
          "    assert re.search(r'GDN_PD_SHALLOW_FULLN_ACTIVE role=prefill shallow_layers=24 deep_layers=12',log) and re.search(r'PD full-N deferred publication enabled[^\\n]*join_offload=True',log),'P12 recipe activation lines'\n"
          "   else:assert P12_KEY not in actual['environment'] and not pipe,(role,pipe)\n"
          "   out.setdefault('p12',{})[role]=dict(arm=actual['environment'].get(P12_KEY),pipeline_lines=pipe)\n")

    # ---- harness_mode: the pd1n window length ----
    patch('support/harness_mode.py', "        return 60 if self.dry_run else 900\n",
          "        if self.dry_run:\n            return 60\n"
          "        seconds = int(os.environ.get('P12_WINDOW_SECONDS', '900'))  # docs/222 §20: pd1n micro 300 s, A/B 600 s\n"
          "        assert seconds in (300, 600, 900), seconds\n        return seconds\n")

    # ---- job.sh: pd1n keeps D's node-local root apart from P's ----
    patch('support/job.sh', 'export V4_LOCAL_ROOT="/tmp/p3pd-20261017/${V4_RUN_ID}"\n',
          'export V4_LOCAL_ROOT="/tmp/p3pd-20261017/${V4_RUN_ID}"\n'
          '# docs/222 §20 pd1n: P (task 0) and D (task 1) on one node keep separate node-local roots, as on two nodes\n'
          'if [[ ${1:-} == inner && ${P12_ONE_NODE:-0} == 1 && ${SLURM_PROCID:-0} == 1 ]]; then\n'
          ' export V4_LOCAL_ROOT="/tmp/p3pd-20261017/${V4_RUN_ID}-dnode"\n'
          'fi\n')

    # ---- entry.py: the gate driver on the P task ----
    patch('support/entry.py', "else:runpy.run_path('/pdrefresh/run_v2.py',run_name='__main__')",
          "else:runpy.run_path('/pdrefresh/p12_gate_driver.py' if os.environ.get('P12_PROOF') else '/pdrefresh/run_v2.py',run_name='__main__')  # docs/222 §20")

    # ---- CPU gate ----
    cg = 'support/r16_cpu_gate.py'
    t = (OUT / cg).read_text()
    n3, s3 = t.count('r16_delta.P3_SOURCE'), t.count('r16_delta.P3_SHA')
    assert (n3, s3) == (6, 2), (n3, s3)
    patch(cg, 'r16_delta.P3_SOURCE', 'r16_delta.P12_SOURCE', n3)  # docs/222 §20: the PC+ tree is the P12 capsule
    patch(cg, 'r16_delta.P3_SHA', 'r16_delta.P12_SHA', s3)
    patch(cg, "                [k for k, v in r16_delta.ROUTE0_P_ENV.items() if base.get(k) != v] if role == 'P' else [])\n",
          "                [k for k, v in r16_delta.ROUTE0_P_ENV.items() if base.get(k) != v] + [r16_delta.P12_KEY] if role == 'P' else [])\n")
    patch(cg, "result['passed'] = True\n", (HERE / 'cpu_gate_p12.py.in').read_text() + "result['passed'] = True\n")

    # ---- manifest + fingerprint ----
    files = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
    m = dict(m0)
    fp = 'p12' + hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:13]
    m.update(fingerprint=fp, p3pd_parent_fingerprint=PARENT_FP, files=files,
             p12_source_host=str(CAP_HOST), p12_source_container=CAP_CONT, p12_source_sha=P12_SHA,
             p12_catalog_sha256=cat_sha, p12_catalog_files=cat_n, p12_staged_equal=str(STAGED), p12_key_blobs=BLOBS,
             p12_patches=patches)
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
            lines += difflib.unified_diff(ta, tb, 'p3pd/' + rel, 'p12/' + rel)
    (ROOT / f'inputs/pkg-{fp}.diff').write_text(''.join(lines))
    print(fp, len(files), len(patches))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
