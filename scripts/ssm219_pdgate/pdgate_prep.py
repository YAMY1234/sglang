"""docs/222 §15 PD serving numerics gate, prep (CMH login, python3 only; zero GPU).

Builds the launch of every arm's P and D server from the actual-launch receipts of a 1P:1D window, the way gate_prep.py
builds agg arms from the v7 receipts: same argv and environment as served, minus instrumentation and harness plumbing,
plus the gate's ports and the probe.  Arms are phases of one 4-GPU holder job (P TP2 on GPU0-1, D TP2 on GPU2-3); the
first phase is `before`, then `cand1` [, `cand2`].

  arm classes (receipts = workers/{P0,D0}/actual-launch.json of a finished window):
    C+   ssm219-pd-20261016 j4537841-Cp  PD stack v1 (pkg v2 s193cab2d1d9047c; integration tree v3 capsule 1fde0db52ee)
    PC+  fp32-pd-20261015   j4529207-PCp both arm, shallow P (pkg f325b2ae29ef52f7; stock capsule 7a841e4189)
    S    ssm219-pd-20261016 j4537841-S   S_fp32 stock pair (record-only floor arm)

    python3 pdgate_prep.py --out $OUT --src /work/src/<tree> --sha <40> \
        --before C+ --cand1 C+ --cand1-p-env '{"SGLANG_GDN_K31_FUSED":"1","SGLANG_GDN_K31_CHOLQR_MIXED":"1"}'
    (per arm: --<arm>-src/--<arm>-sha override the tree, --<arm>-p-env / --<arm>-d-env add role env, --<arm>-receipt DIR
     replaces the class receipt; --expect cand1:P:<regex> adds an activation line; --no-line KEY='<reason>' accepts a
     switch that has no activation line.)

Writes $OUT/g-<arm>-<P|D>.{env,argv,json}, $OUT/g-<arm>.mounts, $OUT/plan.json, the reference (hard links) and tools.
Every change against a receipt is listed in g-<arm>-<role>.json `changed`.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
from pathlib import Path

W = "/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911"
RES = f"{W}/results"
RECEIPTS = {"C+": f"{RES}/ssm219-pd-20261016/runs/j4537841-Cp", "PC+": f"{RES}/fp32-pd-20261015/runs/j4529207-PCp",
            "S": f"{RES}/ssm219-pd-20261016/runs/j4537841-S"}
PRELUDE = {"C+": "Cplus", "PC+": "PCplus", "S": "S"}
# reference = the agg S_fp32 dump on the agg gate's 48 windows (docs/218 kl-fdt-2, job 4538682; windows sha af5cd62b)
REF = f"{RES}/fp32-state-20261015/kl-fdt-2/ref-s32"
WINDOWS = f"{RES}/fp32-state-20261015/kl-fdt-2/windows.json"
WINDOWS_SHA = "af5cd62b294ff245510c19f2cae19d93d091139fe71c7278b1974f5c8e471df9"
MODEL_HOST = f"{RES}/bf16-regress-20261013/staging/model"
# ports of the pair (phases run one after another; the job waits for the previous pair to exit)
PORTS = {"P": dict(http=30301, bootstrap=30303, nccl=30305), "D": dict(http=30302, bootstrap=30304, nccl=30306)}
GPUS = {"P": "0,1", "D": "2,3"}
# dropped from the receipt environment (pre-registered, docs/222 §15.2): instrumentation, harness plumbing, process state
DROP = {"TWINSTAR_AGENTX_INVENTORY", "TWINSTAR_CUDA_TIMELINE", "TWINSTAR_PD_CACHE_AUDIT", "TWINSTAR_AGENTX_PRIVATE_OCCUPANCY",
        "TWINSTAR_PD_FORMAL_TRACE", "TWINSTAR_PD_TORCH_PROFILE", "PD_REMOTE_HTTP_FD", "STAGE2_HTTP_FD", "PD_SCREEN_WORKER",
        "REMEASURE_CENSUS_CONTROL", "PD_NODE_LIST", "PD_SHALLOW_LOADING_AUDIT", "TWINSTAR_EXPECT_FACTOR_P_POOL",
        "TWINSTAR_PD_EXPECT_STATE_SLOTS", "HOSTNAME", "PWD", "OLDPWD", "SHLVL", "_", "CUDA_VISIBLE_DEVICES"}
DROP_PREFIX = ("SLURM_", "SLURMD_", "_SLURM_", "PMIX_", "PMI_", "OMPI_", "ENROOT_", "PYXIS_")
# output / audit paths moved under the gate's node-local directory (the receipt paths belong to the window's run)
LOCAL = ("PFACTOR3_OUTPUT", "V3_GRAPH_OUTPUT", "V4_LOCAL_ROOT", "PDREF_ROOT", "PD_FACTOR_ONLY_CONTRACT")
# compile-cache paths keep the receipt value and are seeded from the package prelude, like the window (trio.sh)
CACHE = ("SGLANG_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR",
         "PYTHONPYCACHEPREFIX", "XDG_CACHE_HOME", "TMPDIR")
# activation lines (docs/219 §8 rule 7b: a result without the switch's first-activation line is void); key -> (regex, note)
KNOWN = {
    "SGLANG_GDN_K31_FUSED": (r"K31_FUSED_ACTIVE 36 \d+ (24|48)", "gdn_factored_pool.py factorize_layers"),
    "SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED": (r"PD full-N deferred publication enabled", "gdn_pd_publication.py"),
    "SGLANG_GDN_PD_PUBLISH_JOIN_OFFLOAD": (r"PD full-N deferred publication enabled.*join_offload=True", "gdn_pd_publication.py"),
    "SGLANG_GDN_FACTORED_TRUNC_CUDA": (r"FACTORED_TRUNC_CUDA_ACTIVE", "gdn_factored_expiry_cuda.py"),
    "SGLANG_GDN_FACTORED_BOOKKEEPING_FOLD": (r"GDN_FACTORED_BOOKKEEPING_FOLD_ACTIVE", "gdn_factored_pool.py"),
    "SGLANG_GDN_K31_MIXED_EIGH_ANY_BATCH": (r"K31_MIXED_EIGH_ANY_BATCH_ACTIVE", "fires only when b > 1 rows end one prefill"),
    # P3 (fork 9827e05cb25 gdn_prefill_agg_contract.py; helper e2f2a76 twinstar_sgl/pd_emitter_graph.py receipt())
    "SGLANG_GDN_PD_SHALLOW_FULLN": (r"GDN_PD_SHALLOW_FULLN_ACTIVE role=prefill shallow_layers=24 deep_layers=12 phase=r/r",
                                    "P3 fork gdn_prefill_agg_contract.py"),
    "TWINSTAR_PD_EMITTER_GRAPH": (r"Flash-Next P emitter graph: ", "helper pd_emitter_graph.py receipt (per capture)"),
}
# switches whose code path this gate cannot show on its own
COVERED_BY = {"SGLANG_GDN_K31_CHOLQR_MIXED": ("SGLANG_GDN_K31_FUSED",
              "no own line; K31_FUSED_ACTIVE implies it (fused = CHOLQR_MIXED and fused_supported, gdn_factored_pool.py)"),
              "SGLANG_GDN_FACTORED_HOST_SYNC_FREE": ("SGLANG_GDN_PD_PUBLISH_JOIN_OFFLOAD",
              "no own line; join_offload=True implies it (offload raises without HOST_SYNC_FREE, gdn_pd_publication.py)")}
NOT_EXERCISED = {"SGLANG_GDN_PREFIX_RESTORE_FUSED": "prefix hits only; the gate's windows are cold (P cached_tokens == 0)"}
FALSE = ("", "0", "false", "False", "off")


def sha256(path):
    return hashlib.file_digest(open(path, "rb"), "sha256").hexdigest()


def host_path(p):
    return p.replace("/work", W, 1) if p == "/work" or p.startswith("/work/") else p


def load_receipt(d, role):
    rec = json.loads((Path(d) / "workers" / f"{role}0" / "actual-launch.json").read_text())
    assert rec["role"] == role and not rec["cpu_service_stub"], (d, role, rec["role"])
    cmd, ex = rec["command"], rec["execution_argv"]
    i = ex.index(cmd[0])
    assert ex[i:] == cmd and ex[0] == "env" and all("=" in x for x in ex[1:i]), ex[:i + 1]
    return rec, dict(x.split("=", 1) for x in ex[1:i])


def arg_set(argv, flag, value, changed):
    i = argv.index(flag)
    changed[flag] = [argv[i + 1], value]
    argv[i + 1] = value


def build_role(arm, role, cls, receipt, src, sha, extra_env, gate_root):
    rec, exec_env = load_receipt(receipt, role)
    env = dict(rec["environment"])
    changed = {}
    mounts = env["_SLURM_SPANK_OPTION_pyxis_container_mounts"].split(",")
    for k in sorted(env):
        if k in DROP or k.startswith(DROP_PREFIX):
            changed[k] = [env.pop(k), None]
    for k, v in exec_env.items():  # the launcher's `env K=V` prefix (SGLANG_GDN_PROMPT_ONLY_STATE_CACHE)
        changed[k] = [env.get(k), v]
        env[k] = v
    local = f"{gate_root}/{arm}-{role}"
    for k in LOCAL:
        if k in env:
            leaf = Path(env[k]).name if k in ("PFACTOR3_OUTPUT", "PD_FACTOR_ONLY_CONTRACT", "V3_GRAPH_OUTPUT") else ""
            changed[k] = [env[k], f"{local}/{k.lower()}" + (f"-{leaf}" if leaf else "")]
            env[k] = changed[k][1]
    changed["REMEASURE_NATIVE_LAUNCH"] = [env.get("REMEASURE_NATIVE_LAUNCH"), "0"]  # as gate_prep: no harness observers
    env["REMEASURE_NATIVE_LAUNCH"] = "0"
    # the tree: replace the receipt's source capsule (PYTHONPATH element + V4_SOURCE / FINAL_FORK / V4_SHA)
    old_src = env["V4_SOURCE"]
    pp = env["PYTHONPATH"].split(":")
    assert pp.count(old_src + "/python") == 1 and env["FINAL_FORK"] == old_src, (pp, old_src)
    if src and src != old_src:
        assert sha and len(sha) == 40, "--src needs the full 40-hex --sha"
        assert Path(host_path(src) + "/python/sglang").is_dir(), src
        pp[pp.index(old_src + "/python")] = src + "/python"
        for k, v in (("V4_SOURCE", src), ("FINAL_FORK", src), ("V4_SHA", sha)):
            changed[k] = [env[k], v]
            env[k] = v
    # the probe module first on the path (pdgate_kl, imported by the server to unpickle the processor)
    pp.insert(0, "/out/tools")
    changed["PYTHONPATH"] = [env["PYTHONPATH"], ":".join(pp)]
    env["PYTHONPATH"] = ":".join(pp)
    env["PDGATE_ROLE"] = role
    changed["PDGATE_ROLE"] = [None, role]
    for k, v in extra_env.items():
        changed[k] = [env.get(k), str(v)]
        env[k] = str(v)
    argv = list(rec["command"])
    assert argv[2] == "sglang.launch_server" and argv[argv.index("--tp") + 1] == "2", argv[:6]
    assert argv[argv.index("--disaggregation-mode") + 1] == {"P": "prefill", "D": "decode"}[role]
    arg_set(argv, "--port", str(PORTS[role]["http"]), changed)
    arg_set(argv, "--nccl-port", str(PORTS[role]["nccl"]), changed)
    arg_set(argv, "--disaggregation-bootstrap-port", str(PORTS["P"]["bootstrap"] if role == "P" else PORTS["D"]["bootstrap"]),
            changed)
    for flag in ("--enable-custom-logit-processor", "--disable-overlap-schedule"):
        # the probe needs both (teacher forcing reads req.output_ids in the sampler); P already has the second
        if flag not in argv:
            argv.append(flag)
            changed[flag] = [None, True]
    shallow = "--flashnext-pd-shallow-prefill" in argv
    mk = sorted({env[k] for k in CACHE + LOCAL if k in env} | {f"{local}/hf"})
    spec = dict(arm=arm, role=role, cls=cls, receipt=str(receipt), gpus=GPUS[role], ports=PORTS[role], shallow=shallow,
                source=env["V4_SOURCE"], source_sha=env["V4_SHA"], changed=changed, argv=argv, mkdirs=mk,
                cache_env={k: env[k] for k in CACHE if k in env})
    return spec, env, argv, mounts


HELPER_CONT = "/work/results/remeasure-v6-fivearm-tp1-20261004/inputs/helper-runtime"


def tree_digest(root):
    """sha256 over (relative path, file sha256) of every file under root, __pycache__ excluded."""
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(f"{p.relative_to(root)}\0{sha256(p)}\n".encode())
    return h.hexdigest()


def mounts_for(mounts, out, helper=None, helper_overlay=None):
    """The receipt's container mounts (the window's exact layout), campaign trees read-only, model as @MODEL@, + /out.
    helper: host directory that replaces the helper-runtime tree (P3 ships helper code); the package's file overlays onto
    that tree are kept only when they equal the new tree's file, or by an explicit --helper-overlay keep / drop."""
    res, notes = [], []
    for m in mounts:
        parts = m.split(":")
        host, cont, opts = parts[0], parts[1], parts[2] if len(parts) > 2 else ""
        if helper and cont == HELPER_CONT:
            # the receipt's layout: helpers/twinstar_sgl (+ helpers/twinstar); a bare infra/sglang checkout would not import
            assert Path(helper, "helpers/twinstar_sgl/__init__.py").is_file(), \
                f"{helper}: not a helper-runtime tree (build one with pdgate_helper.py)"
            hs = Path(helper, "helper-source.json")
            if hs.is_file():
                d = json.loads(hs.read_text()).get("diff_vs_frozen") or {}
                notes.append(f"helper {json.loads(hs.read_text()).get('sha')}: vs frozen changed "
                             f"{len(d.get('changed', []))} added {len(d.get('added', []))} removed {len(d.get('removed', []))}")
            notes.append(f"helper tree {host} -> {helper}")
            host = helper
        elif helper and cont.startswith(HELPER_CONT + "/"):
            mine = Path(helper) / cont[len(HELPER_CONT) + 1:]
            same = mine.is_file() and sha256(mine) == sha256(host)
            if not same and helper_overlay is None:
                raise SystemExit(f"package overlay {host} -> {cont} differs from the new helper tree's {mine}; "
                                 "pass --helper-overlay keep|drop")
            if not same and helper_overlay == "drop":
                notes.append(f"overlay dropped: {cont} (new helper tree's own file)")
                continue
            notes.append(f"overlay kept: {cont} ({'identical' if same else 'differs, kept by --helper-overlay keep'})")
        if cont == "/model":
            assert host == MODEL_HOST, host
            host = "@MODEL@"
        if re.match(rf"{W}/(results|archives)/[^/]+$", host) and opts != "ro":
            opts = "ro"  # the window's campaign / archive trees: the gate writes only to /out and node-local /tmp
        res.append(":".join([host, cont] + ([opts] if opts else [])))
    if helper:
        assert any(n.startswith("helper tree") for n in notes), "receipt has no helper-runtime mount"
    res.append(f"{out}:/out")
    return ",".join(res), notes


def expectations(arms, declared_env, extra, no_line, hit_pass=False):
    """Activation lines per (arm, role): every switch whose value differs from `before` must show its line on that role
    of the candidate and must not show it on `before`."""
    exp, notes = {}, []
    before = arms["before"]
    for arm, roles in arms.items():
        if arm == "before":
            continue
        for role, env in roles.items():
            declared = set(declared_env[arm][role]) | set(declared_env["before"][role])
            delta = sorted(k for k in set(env) | set(before[role])
                           if (k.startswith(("SGLANG_", "TWINSTAR_")) or k in declared) and env.get(k) != before[role].get(k))
            req = exp.setdefault(arm, {}).setdefault(role, [])
            for k in delta:
                on = env.get(k, "") not in FALSE
                if not on:
                    notes.append(f"{arm}/{role}: {k} switched off vs before ({before[role].get(k)!r} -> {env.get(k)!r}); "
                                 "no line to require")
                    continue
                req.append(dict(key=k, kind="env", regex=rf"PDGATE_ENV {re.escape(k)}={re.escape(env[k])}$"))
                if k in KNOWN:
                    req.append(dict(key=k, kind="line", regex=KNOWN[k][0], note=KNOWN[k][1], forbid_in_before=True))
                elif k in COVERED_BY and env.get(COVERED_BY[k][0], "") not in FALSE and COVERED_BY[k][0] in delta:
                    notes.append(f"{arm}/{role}: {k}: {COVERED_BY[k][1]}")
                elif k in NOT_EXERCISED and hit_pass:
                    req.append(dict(key=k, kind="line", regex=r"PREFIX_RESTORE_FUSED_ACTIVE", note="prefix-hit pass",
                                    forbid_in_before=True))
                elif k in NOT_EXERCISED:
                    if k not in no_line:
                        raise SystemExit(f"{arm}/{role}: {k} is not exercised by this gate ({NOT_EXERCISED[k]}); "
                                         f"pass --no-line {k}='<reason>' to gate it anyway")
                    notes.append(f"{arm}/{role}: {k} not exercised, accepted: {no_line[k]}")
                elif any(e[0] == arm and e[1] == role and e[2] == k for e in extra):
                    pass
                elif k in no_line:
                    notes.append(f"{arm}/{role}: {k} has no activation line, accepted: {no_line[k]}")
                else:
                    raise SystemExit(f"{arm}/{role}: no activation line known for {k}; pass --expect {arm}:{role}:<regex>"
                                     f" (with key {k}) or --no-line {k}='<reason>'")
    for arm, role, key, regex in extra:
        exp.setdefault(arm, {}).setdefault(role, []).append(dict(key=key, kind="line", regex=regex, note="--expect",
                                                                 forbid_in_before=True))
    return exp, notes


REUSE_FILES = ("rec-before-score.P.jsonl", "rec-before-score.D.jsonl", "calls-before.P.jsonl", "calls-before.D.jsonl",
               "score-meta-before.json", "smoke-before.json", "gsm-before.json", "srv-before-P.log", "srv-before-D.log",
               "phase-before.log", "rec-before-score-hit.P.jsonl", "rec-before-score-hit.D.jsonl",
               "calls-before-hit.P.jsonl", "calls-before-hit.D.jsonl", "score-meta-before-hit.json")


def reuse_before(src, out, gate_root, arm_plan, a):
    """`before` of an earlier gate stands in for this one only when its launch is the same launch: the four g-before
    files equal byte for byte once that gate's path names are replaced by this gate's, the same reference / windows /
    image, and that gate's before phase finished (smoke, score, GSM)."""
    old = json.loads((src / "plan.json").read_text())
    for k in ("image",):
        assert old[k] == getattr(a, k), (k, old[k])
    assert bool(old.get("hit_pass")) or not (a.hit_pass or a.hit_gate), "earlier gate had no prefix-hit pass"
    assert old["arms"]["before"].get("helper_digest") == arm_plan["helper_digest"], "helper tree differs"
    assert old["windows"]["sha256"] == sha256(a.windows) and old["ref"]["path"] == a.ref, "reference / windows differ"
    assert old["arms"]["before"]["receipt"] == arm_plan["receipt"], "receipt differs"
    for f in ("g-before-P.env", "g-before-P.argv", "g-before-D.env", "g-before-D.argv", "g-before.mounts"):
        mine = (out / f).read_text()
        theirs = (src / f).read_text().replace(old["gate_root"], gate_root).replace(f"{src}:/out", f"{out}:/out")
        assert mine == theirs, f"{f} differs from {src}"
    assert "PHASE-DONE before" in (src / "phase-before.log").read_text(), "earlier before phase did not finish"
    for f in REUSE_FILES:
        if (src / f).exists():
            shutil.copyfile(src / f, out / f)
    return dict(source=str(src), files=[f for f in REUSE_FILES if (out / f).exists()],
                job_log=(src / "job.log").read_text().splitlines()[:1] if (src / "job.log").exists() else None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", help="tree for every arm (container path /work/...); default: each receipt's capsule")
    ap.add_argument("--sha")
    ap.add_argument("--helper", help="host dir replacing the helper-runtime tree for every arm (P3 helper code)")
    ap.add_argument("--helper-overlay", choices=("keep", "drop"))
    ap.add_argument("--hit-pass", action="store_true", help="second KL pass on P's cached prefixes (restore path), reported")
    ap.add_argument("--hit-gate", action="store_true", help="the prefix-hit pass is part of the verdict (implies --hit-pass)")
    for arm in ("before", "cand1", "cand2"):
        ap.add_argument(f"--{arm}", choices=sorted(RECEIPTS), required=arm != "cand2")
        ap.add_argument(f"--{arm}-receipt")
        ap.add_argument(f"--{arm}-src")
        ap.add_argument(f"--{arm}-sha")
        ap.add_argument(f"--{arm}-helper")
        ap.add_argument(f"--{arm}-p-env", default="{}")
        ap.add_argument(f"--{arm}-d-env", default="{}")
    ap.add_argument("--expect", action="append", default=[], help="ARM:ROLE:KEY:REGEX")
    ap.add_argument("--no-line", action="append", default=[], help="KEY=reason")
    ap.add_argument("--before-from", help="a finished gate whose `before` was measured with the identical launch: reuse it "
                                           "(byte-identical g-before-* after the gate path is normalised) instead of a phase")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--windows", default=WINDOWS)
    ap.add_argument("--tools", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--image", default=f"{W}/containers/lmsysorg+sglang+nightly-dev-cu13-20260913-fa663e72-arm64-enroot.sqsh")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "plan.json").exists(), f"{out} already prepared"
    gate_root = f"/tmp/pdgate/{out.name}"
    extra = []
    for e in a.expect:
        arm, role, key, regex = e.split(":", 3)
        assert role in ("P", "D"), e
        extra.append((arm, role, key, regex))
    no_line = dict(x.split("=", 1) for x in a.no_line)
    phases = [x for x in ("before", "cand1", "cand2") if getattr(a, x)]
    for arm in phases[1:]:  # one class per gate: the paired difference is one switch set on one served route
        assert getattr(a, arm) == a.before, f"{arm} class {getattr(a, arm)} != before class {a.before}"
    arms_env, plan_arms = {}, {}
    for arm in phases:
        cls = getattr(a, arm)
        receipt = getattr(a, f"{arm}_receipt") or RECEIPTS[cls]
        src = getattr(a, f"{arm}_src") or a.src
        sha = getattr(a, f"{arm}_sha") or a.sha
        role_env = {"P": json.loads(getattr(a, f"{arm}_p_env")), "D": json.loads(getattr(a, f"{arm}_d_env"))}
        specs, mounts_seen = {}, set()
        arms_env[arm] = {}
        for role in ("P", "D"):
            spec, env, argv, mounts = build_role(arm, role, cls, receipt, src, sha, role_env[role], gate_root)
            mounts_seen.add(",".join(mounts))
            (out / f"g-{arm}-{role}.env").write_text("".join(f"export {k}={shlex.quote(v)}\n" for k, v in sorted(env.items())))
            (out / f"g-{arm}-{role}.argv").write_text(" ".join(shlex.quote(x) for x in argv))
            (out / f"g-{arm}-{role}.json").write_text(json.dumps(spec, indent=1))
            specs[role] = spec
            arms_env[arm][role] = env
        assert len(mounts_seen) == 1, f"{arm}: P and D receipts mount differently"
        assert specs["P"]["shallow"] == ("--flashnext-pd-shallow-prefill" in specs["D"]["argv"]), arm
        assert specs["P"]["source"] == specs["D"]["source"], arm
        helper = getattr(a, f"{arm}_helper") or a.helper
        mnt, mnotes = mounts_for(mounts_seen.pop().split(","), str(out), helper, a.helper_overlay)
        (out / f"g-{arm}.mounts").write_text(mnt)
        bundle = arms_env[arm]["P"]["PARITY_BUNDLE"]
        assert arms_env[arm]["D"]["PARITY_BUNDLE"] == bundle, arm
        plan_arms[arm] = dict(cls=cls, receipt=receipt, source=specs["P"]["source"], source_sha=specs["P"]["source_sha"],
                              p_env=role_env["P"], d_env=role_env["D"], step0_owner="D" if specs["P"]["shallow"] else "P",
                              bundle=bundle, prelude=f"{host_path(bundle)}/support/prelude/{PRELUDE[cls]}.json",
                              cache_env={r: specs[r]["cache_env"] for r in ("P", "D")}, helper=helper,
                              helper_digest=tree_digest(helper) if helper else None, mount_notes=mnotes)
        assert Path(plan_arms[arm]["prelude"]).is_file(), plan_arms[arm]["prelude"]
    a.hit_pass = a.hit_pass or a.hit_gate
    exp, notes = expectations(arms_env, {x: {"P": plan_arms[x]["p_env"], "D": plan_arms[x]["d_env"]} for x in phases},
                              extra, no_line, a.hit_pass)
    # reference + windows (hard links, like the agg gate's ref-stock)
    assert sha256(a.windows) == WINDOWS_SHA or a.windows != WINDOWS, "windows.json changed"
    spec = json.loads(Path(a.windows).read_text())
    n, cont = len(spec["windows"]), spec["cont"]
    shutil.copyfile(a.windows, out / "windows.json")
    (out / "ref-s32").mkdir(exist_ok=True)
    for f in ("val.f32", "idx.i32"):
        src_f = Path(a.ref) / f
        assert src_f.stat().st_size == n * cont * 8192 * 4, (src_f, src_f.stat().st_size)
        if not (out / "ref-s32" / f).exists():
            os.link(src_f, out / "ref-s32" / f)
    # tools: this directory (the uploaded, blob-checked copy) -> $OUT/tools, with hashes
    (out / "tools").mkdir(exist_ok=True)
    tools = {}
    for f in sorted(Path(a.tools).glob("pdgate_*")):
        if f.suffix in (".py", ".sh"):
            shutil.copyfile(f, out / "tools" / f.name)
            tools[f.name] = sha256(f)
    run_phases = list(phases)
    reused = None
    if a.before_from:
        reused = reuse_before(Path(a.before_from), out, gate_root, plan_arms["before"], a)
        run_phases.remove("before")
    plan = dict(hit_pass=a.hit_pass, hit_gate=a.hit_gate, phases=phases, run_phases=run_phases, before_reused=reused, arms=plan_arms, ports=PORTS, gpus=GPUS, gate_root=gate_root, image=a.image,
                model_host=MODEL_HOST, ref=dict(name="s32", path=a.ref, job="4538682 (docs/218 kl-fdt-2, agg S_fp32)"),
                windows=dict(path=a.windows, sha256=sha256(a.windows), n=n, cont=cont), expectations=exp,
                expectation_notes=notes, no_line=no_line, tools=tools, prep_argv=sys.argv if argv is None else argv,
                gate=dict(kl="KL(cand) - KL(before) <= 2 s.e. (paired over windows)",
                          gsm="GSM200 paired diff >= -2 s.e.", activation="every expectation met, none in before"))
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    for arm in phases:
        print(arm, plan_arms[arm]["cls"], "step0", plan_arms[arm]["step0_owner"], plan_arms[arm]["source_sha"][:11],
              "P+", sorted(plan_arms[arm]["p_env"]), "D+", sorted(plan_arms[arm]["d_env"]))
    for arm, roles in exp.items():
        for role, items in roles.items():
            for it in items:
                print("expect", arm, role, it["kind"], it["regex"])
    for s in notes:
        print("note", s)
    return plan


if __name__ == "__main__":
    main()
