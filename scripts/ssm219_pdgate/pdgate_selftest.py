"""docs/222 §15 PD gate CPU self-test (zero GPU).

  prep   (CMH login, python3)  pdgate_prep.py on the real window receipts: C+ with the P1 switches, PC+ with a declared
         shallow switch, S floor; checks the launch transformation (dropped / moved / added keys, ports, flags, tree swap,
         mounts), that before and candidate differ by exactly the declared switches, the activation expectations, and the
         refusals (switch without a line, CHOLQR_MIXED alone, prefix-hit-only switch, cross-class pair).
  units  (torch + sglang, e.g. the image on a CPU node; or locally with --stub-sglang)
         probe: role split for C+ (step 0 on P) and PC+ (P placeholder forced only, step 0 on D after the pop), KL numbers
         identical to acc_audit_kl.KLProbe, history / role errors, dill round trip by reference;
         dispatch: fake P / D servers (same room both halves, P blocks until D has its half -> proves the two halves run
         together), cold-prefix and completion checks, GSM answer taken from D;
         summary: PASS / FAIL (KL, GSM) / VOID (line missing, line in before) cases, and gate numbers identical to the agg
         tool gate_summary.py on the same per-window inputs.
    python3 pdgate_selftest.py prep  --root <scratch dir on the reference's file system>
    python  pdgate_selftest.py units --acc-audit <acc_audit_kl.py> --gate-summary <gate_summary.py> [--stub-sglang]
"""
import argparse
import importlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def raises(fn, exc, what):
    try:
        fn()
    except exc as e:
        check(True, f"{what} -> {type(e).__name__}: {str(e)[:120]}")
        return
    check(False, f"{what} (no {exc})")


# ------------------------------------------------------------------------------------------------------------------ prep
def run_prep(root, name, args):
    sys.path.insert(0, str(HERE))
    import pdgate_prep
    out = Path(root) / name
    if out.exists():
        shutil.rmtree(out)
    return pdgate_prep, out, pdgate_prep.main(["--out", str(out)] + args)


def load_env(path):
    import shlex
    env = {}
    for line in Path(path).read_text().splitlines():
        k, v = line[len("export "):].split("=", 1)
        env[k] = shlex.split(v)[0] if v else ""
    return env


def prep_tests(root):
    P1 = {"SGLANG_GDN_K31_FUSED": "1", "SGLANG_GDN_K31_CHOLQR_MIXED": "1"}
    pp, out, plan = run_prep(root, "cplus-p1", ["--before", "C+", "--cand1", "C+", "--cand1-p-env", json.dumps(P1)])
    check(plan["phases"] == ["before", "cand1"], "C+: phases before, cand1")
    check(plan["arms"]["before"]["step0_owner"] == "P" and plan["arms"]["cand1"]["step0_owner"] == "P", "C+: step 0 on P")
    for arm in ("before", "cand1"):
        for role in ("P", "D"):
            env = load_env(out / f"g-{arm}-{role}.env")
            argv = (out / f"g-{arm}-{role}.argv").read_text().split()
            spec = json.loads((out / f"g-{arm}-{role}.json").read_text())
            tag = f"C+ {arm}/{role}"
            check(not [k for k in env if k in pp.DROP or k.startswith(pp.DROP_PREFIX)], f"{tag}: dropped keys absent")
            check(env["PDGATE_ROLE"] == role and env["REMEASURE_NATIVE_LAUNCH"] == "0", f"{tag}: role / native launch off")
            check(env["PYTHONPATH"].split(":")[0] == "/out/tools", f"{tag}: probe module first on PYTHONPATH")
            check(env["SGLANG_GDN_PROMPT_ONLY_STATE_CACHE"] == "1", f"{tag}: launcher env prefix kept")
            check(all(env[k].startswith(f"{plan['gate_root']}/{arm}-{role}/") for k in pp.LOCAL if k in env),
                  f"{tag}: output paths node-local per arm/role")
            check(all(65000 < v < 65536 for r in pp.PORTS.values() for v in r.values()),
                  f"{tag}: ports above the ephemeral range (9000-65000)")
            check(argv[argv.index("--port") + 1] == str(pp.PORTS[role]["http"]) and
                  argv[argv.index("--nccl-port") + 1] == str(pp.PORTS[role]["nccl"]), f"{tag}: ports")
            check(argv.count("--enable-custom-logit-processor") == 1 and argv.count("--disable-overlap-schedule") == 1,
                  f"{tag}: probe flags once")
            check(spec["gpus"] == pp.GPUS[role] and spec["source_sha"].startswith("1fde0db52ee"), f"{tag}: gpus / tree")
    for role in ("P", "D"):
        b, c = load_env(out / f"g-before-{role}.env"), load_env(out / f"g-cand1-{role}.env")
        diff = {k for k in set(b) | set(c) if b.get(k) != c.get(k)} - set(pp.LOCAL)
        check(diff == (set(P1) if role == "P" else set()), f"C+ before vs cand1 {role}: env differs by exactly {sorted(diff)}")
        check((out / f"g-before-{role}.argv").read_text() == (out / f"g-cand1-{role}.argv").read_text(),
              f"C+ before vs cand1 {role}: argv identical")
    m = (out / "g-cand1.mounts").read_text().split(",")
    check(m[-1] == f"{out}:/out" and any(x.startswith("@MODEL@:/model") for x in m), "mounts: /out last, model placeholder")
    check(all(x.endswith(":ro") for x in m if "/results/ssm219-pd-20261016:" in x or "/archives/ssm219-pd-20261016:" in x),
          "mounts: campaign / archive trees read-only")
    exp = plan["expectations"]["cand1"]["P"]
    check(any(e["kind"] == "line" and e["regex"].startswith("K31_FUSED_ACTIVE") for e in exp) and
          sum(e["kind"] == "env" for e in exp) == 2 and plan["expectations"]["cand1"]["D"] == [],
          "C+ P1: K31_FUSED_ACTIVE + two env lines on P, nothing on D")
    check(any("SGLANG_GDN_K31_CHOLQR_MIXED" in n and "K31_FUSED_ACTIVE implies" in n for n in plan["expectation_notes"]),
          "C+ P1: CHOLQR_MIXED recorded as covered by K31_FUSED_ACTIVE")
    check(os.stat(out / "ref-s32" / "val.f32").st_nlink >= 2 and (out / "windows.json").exists(), "reference hard-linked")
    check({"pdgate_kl.py", "pdgate_phase.sh", "pdgate_hjob.sh", "pdgate_summary.py", "pdgate_seed.py"} <= set(plan["tools"]),
          "tools copied with hashes")
    # reuse of a finished `before` (identity-checked) by a later gate on the same launch
    (out / "phase-before.log").write_text("READY before-P\nPHASE-DONE before 01:00:00\n")
    (out / "rec-before-score.P.jsonl").write_text('{"window": 0}\n')
    pp2, out2, plan2 = run_prep(root, "cplus-reuse", ["--before", "C+", "--cand1", "C+", "--before-from", str(out),
                                                      "--cand1-p-env", '{"SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED":"1"}'])
    check(plan2["run_phases"] == ["cand1"] and (out2 / "rec-before-score.P.jsonl").exists() and
          plan2["before_reused"]["source"] == str(out), "reuse: identical before -> only cand1 runs, records copied")
    raises(lambda: run_prep(root, "cplus-reuse-bad", ["--before", "C+", "--cand1", "C+", "--before-from", str(out),
                                                       "--before-d-env", '{"SGLANG_GDN_FACTORED_TRUNC_CUDA":"1"}']),
           AssertionError, "reuse refused when the before launch differs")
    # a passed candidate as the base of a stacked change: the earlier gate's cand1 stands in for this gate's before
    (out / "phase-cand1.log").write_text("READY cand1-P\nPHASE-DONE cand1 02:00:00\n")
    (out / "rec-cand1-score.P.jsonl").write_text('{"window": 1}\n')
    (out / "srv-cand1-P.log").write_text("cand1 server log\n")
    pp3, out3, plan3 = run_prep(root, "cplus-stack", ["--before", "C+", "--cand1", "C+", "--before-from", str(out),
                                                      "--before-from-arm", "cand1", "--before-p-env", json.dumps(P1),
                                                      "--cand1-p-env", json.dumps(dict(P1, SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED="1"))])
    check(plan3["run_phases"] == ["cand1"] and (out3 / "rec-before-score.P.jsonl").read_text() == '{"window": 1}\n' and
          (out3 / "srv-before-P.log").read_text() == "cand1 server log\n" and plan3["before_reused"]["source_arm"] == "cand1",
          "reuse of cand1 as before: identical launch after the arm directory is mapped, artifacts copied under before names")
    raises(lambda: run_prep(root, "cplus-stack-bad", ["--before", "C+", "--cand1", "C+", "--before-from", str(out),
                                                       "--before-from-arm", "cand1"]),
           AssertionError, "reuse of cand1 refused when this before lacks cand1's P env")
    for d in (out2, out3, Path(root) / "cplus-stack-bad", Path(root) / "cplus-reuse-bad"):
        shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(out)
    # refusals
    for name, args, exc in (
            ("cholqr-alone", ["--before", "C+", "--cand1", "C+", "--cand1-p-env", '{"SGLANG_GDN_K31_CHOLQR_MIXED":"1"}'], SystemExit),
            ("unknown", ["--before", "C+", "--cand1", "C+", "--cand1-p-env", '{"SGLANG_GDN_PDGATE_SELFTEST_UNKNOWN":"1"}'], SystemExit),
            ("prefix-hit", ["--before", "C+", "--cand1", "C+", "--before-p-env", '{"SGLANG_GDN_PREFIX_RESTORE_FUSED":"0"}'],
             SystemExit),
            ("cross-class", ["--before", "C+", "--cand1", "PC+"], AssertionError)):
        raises(lambda: run_prep(root, "refuse-" + name, args), exc, f"prep refuses {name}")
        shutil.rmtree(Path(root) / ("refuse-" + name), ignore_errors=True)
    # a tree under /work/src (shadowed by the receipts' stable-src mount): must get its own mount (4560175 / 4560392)
    pp, out, plan = run_prep(root, "src-shadow", ["--src", "/work/src/ssm219-int-d053bd5e346",
                                                  "--sha", "d053bd5e346b41bd701adc14b8bd412a4af177e7", "--before", "C+", "--cand1", "C+"])
    env = load_env(out / "g-cand1-P.env")
    spec = json.loads((out / "g-cand1-P.json").read_text())
    m = (out / "g-cand1.mounts").read_text().split(",")
    check(env["V4_SOURCE"] == "/pdgate/src/ssm219-int-d053bd5e346" and env["PYTHONPATH"].split(":")[1] == env["V4_SOURCE"] + "/python"
          and any(x.endswith(":/pdgate/src/ssm219-int-d053bd5e346:ro") for x in m)
          and spec["sglang_expect"] == "/pdgate/src/ssm219-int-d053bd5e346/python/sglang/",
          "shadowed /work/src tree -> own mount /pdgate/src/<name>, PYTHONPATH and sglang_expect follow")
    check(pp.resolve(m, spec["sglang_expect"] + "__init__.py") is not None, "sglang_expect resolves through the gate mounts")
    shutil.rmtree(out)
    # PC+ with a declared line for a switch the fork does not have yet
    pp, out, plan = run_prep(root, "pcplus", ["--before", "PC+", "--cand1", "PC+", "--cand1-p-env",
                                              '{"SGLANG_GDN_PD_SHALLOW_FULLN":"1"}',
                                              "--expect", "cand1:P:SGLANG_GDN_PD_SHALLOW_FULLN:PD_SHALLOW_FULLN_ACTIVE"])
    check(plan["arms"]["cand1"]["step0_owner"] == "D", "PC+: step 0 on D (shallow P)")
    check(plan["arms"]["before"]["source_sha"].startswith("7a841e4189"), "PC+: stock capsule tree")
    check(any(e["regex"] == "PD_SHALLOW_FULLN_ACTIVE" for e in plan["expectations"]["cand1"]["P"]), "PC+: --expect line kept")
    shutil.rmtree(out)
    helper = "/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911/results/remeasure-v6-fivearm-tp1-20261004/inputs/helper-runtime"
    raises(lambda: run_prep(root, "helper-noflag", ["--before", "C+", "--cand1", "C+", "--helper", helper]), SystemExit,
           "helper override refuses a differing package overlay without --helper-overlay")
    shutil.rmtree(Path(root) / "helper-noflag", ignore_errors=True)
    raises(lambda: run_prep(root, "helper-layout", ["--before", "PC+", "--cand1", "PC+", "--helper",
                                                    "/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911/results/"
                                                    "r16-pd-20261014/overlay/helper-runtime/helpers"]),
           AssertionError, "helper override refuses a tree without helpers/twinstar_sgl (e.g. a bare infra/sglang)")
    shutil.rmtree(Path(root) / "helper-layout", ignore_errors=True)
    pp, out, plan = run_prep(root, "helper", ["--before", "C+", "--cand1", "C+", "--helper", helper, "--helper-overlay", "drop",
                                              "--hit-pass", "--before-p-env", '{"SGLANG_GDN_PREFIX_RESTORE_FUSED":"0"}'])
    m = (out / "g-cand1.mounts").read_text().split(",")
    check(f"{helper}:{pp.HELPER_CONT}:ro" in m and not any(x.split(":")[1].startswith(pp.HELPER_CONT + "/") for x in m)
          and plan["arms"]["cand1"]["helper_digest"], "helper override: tree replaced, overlay dropped, digest recorded")
    check(any(e["regex"] == "PREFIX_RESTORE_FUSED_ACTIVE" for e in plan["expectations"]["cand1"]["P"]) and plan["hit_pass"],
          "hit pass: prefix-restore switch becomes gateable (its line required)")
    shutil.rmtree(out)
    # extra window set: windows copied, reference root mounted into every arm, reference completeness recorded
    xr = Path(root) / "xw-root"
    xr.mkdir(exist_ok=True)
    (xr / "windows-long.json").write_text(json.dumps({"seed": 1, "prefix": 48000, "cont": 256, "starts": [0], "windows": [[1] * 48256]}))
    pp, out, plan = run_prep(root, "extra", ["--before", "PC+", "--cand1", "PC+", "--extra-windows",
                                             f"long:{xr}/windows-long.json:{xr}"])
    e = plan["extra_windows"][0]
    check(e["name"] == "long" and e["ref_cont"] == "/pdgate/xw-long/ref-s32" and not e["ref_ready"]
          and f"{xr}:/pdgate/xw-long:ro" in (out / "g-cand1.mounts").read_text().split(",")
          and (out / "g-cand1.mounts").read_text().split(",")[-1].endswith(":/out") and (out / "windows-long.json").exists(),
          "extra windows: copied, reference root mounted read-only in every arm, pending reference recorded")
    shutil.rmtree(out)
    shutil.rmtree(xr)
    pp, out, plan = run_prep(root, "sfloor", ["--before", "S", "--cand1", "S"])
    check(plan["arms"]["before"]["step0_owner"] == "P" and plan["expectations"]["cand1"] == {"P": [], "D": []},
          "S floor: no switch, no expectation")
    shutil.rmtree(out)


# ----------------------------------------------------------------------------------------------------------------- units
def stub_sglang(tmp):
    """Minimal sglang.srt.sampling.custom_logit_processor for a machine without sglang (same to_str / from_str)."""
    d = Path(tmp) / "stub" / "sglang" / "srt" / "sampling"
    d.mkdir(parents=True)
    for p in (d.parent.parent, d.parent, d):
        (p / "__init__.py").write_text("")
    (d / "custom_logit_processor.py").write_text(
        "import json, dill\n"
        "class CustomLogitProcessor:\n"
        "    @classmethod\n    def to_str(cls):\n        return json.dumps({'callable': dill.dumps(cls).hex()})\n"
        "    @classmethod\n    def from_str(cls, s):\n        return dill.loads(bytes.fromhex(json.loads(s)['callable']))()\n")
    sys.path.insert(0, str(d.parent.parent.parent))


def probe_tests(tmp, acc_audit):
    import numpy as np
    import torch
    sys.path.insert(0, str(HERE))
    kl = importlib.import_module("pdgate_kl")
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
    spec = importlib.util.spec_from_file_location("acc_audit_kl", acc_audit)
    acc = importlib.util.module_from_spec(spec)
    sys.modules["acc_audit_kl"] = acc
    spec.loader.exec_module(acc)
    V, K, NW, CONT = 64, 16, 3, 5
    g = torch.Generator().manual_seed(7)
    tmp = Path(tmp)
    ref = tmp / "ref"
    ref.mkdir()
    val = np.memmap(ref / "val.f32", dtype=np.float32, mode="w+", shape=(NW, CONT, K))
    idx = np.memmap(ref / "idx.i32", dtype=np.int32, mode="w+", shape=(NW, CONT, K))
    for w in range(NW):
        for s in range(CONT):
            lp = torch.log_softmax(torch.randn(V, generator=g), -1)
            v, i = torch.topk(lp, K)
            val[w, s], idx[w, s] = v.numpy(), i.numpy()
    val.flush(); idx.flush()
    targets = {w: torch.randint(0, V, (CONT,), generator=g).tolist() for w in range(NW)}
    logits = {(w, s): torch.randn(V, generator=g) * 3 for w in range(NW) for s in range(CONT)}

    def params(w, owner, tag):
        return {"targets": targets[w], "window": w, "n_windows": NW, "topk": K, "refs": {"s32": str(ref)},
                "record": str(tmp / f"rec-{tag}"), "calls": str(tmp / f"calls-{tag}"), "step0_owner": owner}

    def call(proc, role, w, s, owner, tag, history=None):
        os.environ["PDGATE_ROLE"] = role
        req = SimpleNamespace(output_ids=list(targets[w][:s]) if history is None else history, origin_input_ids=[1, 2, 3],
                              extend_range=None)
        x = logits[(w, s)].clone().unsqueeze(0)
        p = dict(params(w, owner, tag), __req__=req)
        y = proc(x, [p])
        return y[0]

    proc = CustomLogitProcessor.from_str(importlib.import_module("pdgate_kl").PDKLProbe.to_str())
    check(type(proc).__module__ == "pdgate_kl" and type(proc).__name__ == "PDKLProbe", "dill round trip by reference")
    # C+: P samples step 0, D the rest
    for w in range(NW):
        y = call(proc, "P", w, 0, "P", "cp")
        check(int(y.argmax()) == targets[w][0] and torch.isinf(y).sum() == V - 1, f"C+ w{w}: P forces step 0") if w == 0 else None
        for s in range(1, CONT):
            call(proc, "D", w, s, "P", "cp")
    rp = [json.loads(x) for x in (tmp / "rec-cp.P.jsonl").read_text().splitlines()]
    rd = [json.loads(x) for x in (tmp / "rec-cp.D.jsonl").read_text().splitlines()]
    check(len(rp) == NW and all(r["step"] == 0 for r in rp) and len(rd) == NW * (CONT - 1), "C+: P scores step 0, D steps >= 1")
    # PC+: P's placeholder call is forced only; D's boundary call (history popped -> step 0) is scored
    for w in range(NW):
        call(proc, "P", w, 0, "D", "pcp")
        for s in range(CONT):
            call(proc, "D", w, s, "D", "pcp")
    check(not (tmp / "rec-pcp.P.jsonl").exists(), "PC+: P writes no KL record")
    rd2 = [json.loads(x) for x in (tmp / "rec-pcp.D.jsonl").read_text().splitlines()]
    calls = [json.loads(x) for x in (tmp / "calls-pcp.P.jsonl").read_text().splitlines()]
    check(len(rd2) == NW * CONT and len(calls) == NW and not any(c["scored"] for c in calls), "PC+: D scores all, P forced only")
    # identical numbers to acc_audit_kl.KLProbe on the same logits
    ref_proc = acc.KLProbe()
    for w in range(NW):
        for s in range(CONT):
            req = SimpleNamespace(output_ids=list(targets[w][:s]), origin_input_ids=[1, 2, 3], extend_range=None)
            p = {"targets": targets[w], "window": w, "n_windows": NW, "topk": K, "mode": "score", "ref_dir": str(tmp / "nodump"),
                 "refs": {"s32": str(ref)}, "record": str(tmp / "rec-acc.jsonl"), "__req__": req}
            ref_proc(logits[(w, s)].clone().unsqueeze(0), [p])
    ra = {(r["window"], r["step"]): r for r in map(json.loads, (tmp / "rec-acc.jsonl").read_text().splitlines())}
    mine = {(r["window"], r["step"]): r for r in rp + rd}
    same = all(ra[k]["kl_s32"] == mine[k]["kl_s32"] and ra[k]["nll"] == mine[k]["nll"] and ra[k]["argmax"] == mine[k]["argmax"]
               for k in ra)
    check(len(ra) == len(mine) == NW * CONT and same, "KL / nll / argmax identical to acc_audit_kl.KLProbe (C+ layout)")
    mine2 = {(r["window"], r["step"]): r for r in rd2}
    check(all(ra[k]["kl_s32"] == mine2[k]["kl_s32"] for k in ra), "KL identical to acc_audit_kl.KLProbe (PC+ layout)")
    raises(lambda: call(proc, "P", 0, 2, "P", "err"), RuntimeError, "P sampling a decode step")
    raises(lambda: call(proc, "D", 0, 2, "P", "err", history=[targets[0][0], (targets[0][1] + 1) % V]), RuntimeError,
           "history mismatch")
    os.environ.pop("PDGATE_ROLE")
    raises(lambda: proc(logits[(0, 0)].clone().unsqueeze(0), [dict(params(0, "P", "err"),
           __req__=SimpleNamespace(output_ids=[], origin_input_ids=[1], extend_range=None))]), RuntimeError, "PDGATE_ROLE unset")


def dispatch_tests(tmp):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    sys.path.insert(0, str(HERE))
    kl = importlib.import_module("pdgate_kl")
    seen = {"P": [], "D": [], "flush": []}
    arrived = {}
    cond = threading.Condition()
    mode = {"cached": 0, "short": False}

    def handler(role):
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/flush_cache":
                    seen["flush"].append(role)
                    out = {}
                else:
                    seen[role].append(body)
                    room = body["bootstrap_room"]
                    with cond:
                        arrived.setdefault(room, set()).add(role)
                        cond.notify_all()
                        if role == "P":  # P's transfer completes only once D holds its half (real PD behaviour)
                            if not cond.wait_for(lambda: "D" in arrived[room], timeout=10):
                                self.send_response(500); self.end_headers(); return
                    if self.path == "/generate":
                        sp = body["sampling_params"]
                        cp = sp.get("custom_params")
                        if cp and role == "D":  # emulate the probe: D writes steps; step 0 on the owner
                            for s in range(len(cp["targets"])):
                                r = "D" if s > 0 or cp["step0_owner"] == "D" else "P"
                                kl._append(f"{cp['record']}.{r}.jsonl", {"window": cp["window"], "step": s, "role": r,
                                                                          "nll": 1.0, "kl_s32": 0.01})
                        n_out = sp["max_new_tokens"] - (1 if mode["short"] else 0)
                        meta = ({"cached_tokens": mode["cached"], "completion_tokens": 1} if role == "P"
                                else {"cached_tokens": 0, "completion_tokens": n_out})
                        out = {"text": "x", "meta_info": meta}
                    else:
                        txt = "P side" if role == "P" else "so \\boxed{18}"
                        out = {"choices": [{"message": {"content": txt}, "finish_reason": "stop"}],
                               "usage": {"completion_tokens": 9}}
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        return H

    srv = {r: ThreadingHTTPServer(("127.0.0.1", 0), handler(r)) for r in ("P", "D")}
    for s in srv.values():
        threading.Thread(target=s.serve_forever, daemon=True).start()
    url = {r: f"http://127.0.0.1:{s.server_address[1]}" for r, s in srv.items()}
    tmp = Path(tmp)
    win = {"prefix": 4, "cont": 3, "windows": [[i] * 7 for i in range(20)]}
    (tmp / "w.json").write_text(json.dumps(win))
    base = ["--p", url["P"], "--d", url["D"], "--bootstrap-host", "nodeX", "--bootstrap-port", "30303"]

    def run(*args):
        sys.argv = ["pdgate_kl"] + list(args)
        kl.main()

    t0 = time.time()
    run("score", *base, "--windows", str(tmp / "w.json"), "--arm", "cand1", "--refs", "s32=/x", "--step0", "P",
        "--out-root", str(tmp), "--conc", "16")
    check(time.time() - t0 < 30, "score: 20 windows at conc 16 finish (both halves concurrent, no deadlock)")
    rooms_p = [b["bootstrap_room"] for b in seen["P"]]
    check(sorted(rooms_p) == sorted(b["bootstrap_room"] for b in seen["D"]) and len(set(rooms_p)) == 20,
          "score: each window one fresh room, same on P and D")
    check(all(b["bootstrap_host"] == "nodeX" and b["bootstrap_port"] == 30303 and "custom_logit_processor" in b
              for b in seen["P"] + seen["D"]), "score: bootstrap fields + processor on both halves")
    check(sorted(seen["flush"]) == ["D", "P"], "score: flushes P and D first")
    meta = json.loads((tmp / "score-meta-cand1.json").read_text())
    check(meta["step0_by_role"] == {"P": 20, "D": 0} and meta["by_role"]["D"] == 40, "score: role split recorded")
    mode["cached"] = 64
    raises(lambda: run("score", *base, "--windows", str(tmp / "w.json"), "--arm", "c2", "--refs", "s32=/x", "--step0", "P",
                       "--out-root", str(tmp)), ValueError, "score refuses a cached prefix on P")
    mode.update(cached=0, short=True)
    raises(lambda: run("score", *base, "--windows", str(tmp / "w.json"), "--arm", "c3", "--refs", "s32=/x", "--step0", "P",
                       "--out-root", str(tmp)), ValueError, "score refuses an incomplete forced continuation")
    mode["short"] = False
    mode["cached"] = 3840
    run("score", *base, "--windows", str(tmp / "w.json"), "--arm", "cand1", "--refs", "s32=/x", "--step0", "P",
        "--out-root", str(tmp), "--tag=-hit", "--prefix-hit")
    hm = json.loads((tmp / "score-meta-cand1-hit.json").read_text())
    check(hm["prefix_hit"] and hm["flush"] is None and (tmp / "rec-cand1-score-hit.D.jsonl").exists(),
          "score --prefix-hit: no flush, hit records under the -hit tag")
    mode["cached"] = 0
    raises(lambda: run("score", *base, "--windows", str(tmp / "w.json"), "--arm", "c4", "--refs", "s32=/x", "--step0", "P",
                       "--out-root", str(tmp), "--tag=-hit", "--prefix-hit"), ValueError, "prefix-hit pass refuses a miss on P")
    data = tmp / "gsm.jsonl"
    data.write_text("".join(json.dumps({"question": f"q{i}", "answer": "#### 18"}) + "\n" for i in range(5)))
    run("gsm", *base, "--data", str(data), "--arm", "cand1", "--out", str(tmp / "gsm.json"), "--n", "5", "--conc", "5")
    g = json.loads((tmp / "gsm.json").read_text())
    check(g["correct"] == 5 and g["dispatch"] == "PD pair", "gsm: answers come from D")
    run("smoke", *base, "--out", str(tmp / "smoke.json"))
    check(json.loads((tmp / "smoke.json").read_text())["D_meta"]["completion_tokens"] == 32, "smoke: 32 tokens from D")
    for s in srv.values():
        s.shutdown()


def summary_tests(tmp, gate_summary):
    import random
    sys.path.insert(0, str(HERE))
    sm = importlib.import_module("pdgate_summary")
    NW, CONT = 48, 8
    rng = random.Random(3)

    def make(name, shift, gsm_c, lines_c, lines_b, step0="P", hit=None):
        out = Path(tmp) / name
        out.mkdir()
        plan = {"phases": ["before", "cand1"], "ref": {"name": "s32"}, "windows": {"n": NW, "cont": CONT},
                "arms": {a: {"step0_owner": step0} for a in ("before", "cand1")}, "expectation_notes": [],
                "hit_pass": hit is not None, "hit_gate": hit is not None,
                "expectations": {"cand1": {"P": [{"key": "SGLANG_GDN_K31_FUSED", "kind": "line", "regex": r"K31_FUSED_ACTIVE 36 \d+ 24",
                                                  "forbid_in_before": True},
                                                 {"key": "SGLANG_GDN_K31_FUSED", "kind": "env",
                                                  "regex": "PDGATE_ENV SGLANG_GDN_K31_FUSED=1$"}], "D": []}}}
        (out / "plan.json").write_text(json.dumps(plan))
        base = [rng.uniform(0.01, 0.03) for _ in range(NW)]
        per = {}
        tags = [("", 0.0)] + ([("-hit", hit)] if hit is not None else [])
        for (arm, sh), (tag, hsh) in [(x, y) for x in (("before", 0.0), ("cand1", shift)) for y in tags]:
            sh = sh + (hsh if arm == "cand1" else 0.0)
            per[arm] = []
            for role in ("P", "D"):
                with open(out / f"rec-{arm}-score{tag}.{role}.jsonl", "w") as f:
                    for w in range(NW):
                        for s in range(CONT):
                            if (s == 0 and role == step0) or (s > 0 and role == "D"):
                                f.write(json.dumps({"window": w, "step": s, "role": role,
                                                    "kl_s32": base[w] + sh + rng.gauss(0, 0.002)}) + "\n")
            (out / f"score-meta-{arm}.json").write_text(json.dumps({"by_role": {}, "step0_by_role": {}, "windows": {"0": {"P_cached": 0}}}))
            correct = {"before": 190, "cand1": gsm_c}[arm]
            (out / f"gsm-{arm}.json").write_text(json.dumps({"arm": arm, "correct": correct, "records": [
                {"correct": i < correct} for i in range(200)]}))
        (out / "srv-cand1-P.log").write_text("".join(x + "\n" for x in lines_c))
        (out / "srv-before-P.log").write_text("".join(x + "\n" for x in lines_b))
        (out / "srv-cand1-D.log").write_text("")
        (out / "srv-before-D.log").write_text("")
        sm.main(out)
        return out, json.loads((out / "gate-summary.json").read_text())

    good = ["[TP0] K31_FUSED_ACTIVE 36 1 24", "PDGATE_ENV SGLANG_GDN_K31_FUSED=1"]
    out, r = make("pass", 0.0, 190, good, [])
    check(r["verdict"]["cand1"] == "PASS", "summary: equal KL / GSM, lines present -> PASS")
    # same numbers as the agg tool on the same per-window inputs
    gs_dir = Path(tmp) / "agg"
    gs_dir.mkdir()
    for arm in ("before", "cand1"):
        rows = [json.loads(x) for role in ("P", "D") for x in (out / f"rec-{arm}-score.{role}.jsonl").read_text().splitlines()]
        (gs_dir / f"rec-{arm}-score.jsonl").write_text("".join(json.dumps({"window": x["window"], "kl_v7": x["kl_s32"]}) + "\n"
                                                            for x in rows))
        shutil.copy(out / f"gsm-{arm}.json", gs_dir / f"gsm-{arm}.json")
    import subprocess
    subprocess.run([sys.executable, gate_summary, str(gs_dir)], check=True, stdout=subprocess.DEVNULL)
    agg = json.loads((gs_dir / "gate-summary.json").read_text())["gate"]["cand1"]
    mine = r["gate"]["cand1"]
    check(all(abs(agg[k] - mine[k]) < 1e-15 for k in ("kl_minus_before", "se", "z", "gsm_diff_pp", "gsm_se_pp"))
          and agg["bistable"] == mine["bistable"] and agg["kl_pass"] == mine["kl_pass"], "summary: identical to gate_summary.py")
    check(make("klfail", 0.004, 190, good, [])[1]["verdict"]["cand1"] == "FAIL", "summary: KL shift -> FAIL")
    check(make("gsmfail", 0.0, 170, good, [])[1]["verdict"]["cand1"] == "FAIL", "summary: GSM -20 -> FAIL")
    check(make("noline", 0.0, 190, good[1:], [])[1]["verdict"]["cand1"].startswith("VOID"), "summary: line missing -> VOID")
    check(make("inbefore", 0.0, 190, good, good[:1])[1]["verdict"]["cand1"].startswith("VOID"),
          "summary: line also in before -> VOID")
    check(make("pcplus", 0.0, 190, good, [], step0="D")[1]["verdict"]["cand1"] == "PASS", "summary: PC+ layout (step 0 on D)")
    # an extra window set (the long set): gated separately, a shift there fails the gate even when the standard set passes
    out, r = make("extra", 0.0, 190, good, [])
    plan = json.loads((out / "plan.json").read_text())
    plan["extra_windows"] = [dict(name="long", n=16, cont=4, prefix=48000, seed=1, sha256="x")]
    (out / "plan.json").write_text(json.dumps(plan))
    for arm, sh in (("before", 0.0), ("cand1", 0.01)):
        for role in ("P", "D"):
            with open(out / f"rec-{arm}-score-long.{role}.jsonl", "w") as f:
                for w in range(16):
                    for s in range(4):
                        if (s == 0) == (role == "P"):
                            f.write(json.dumps({"window": w, "step": s, "role": role, "kl_s32": 0.03 + sh + rng.gauss(0, 0.002)}) + "\n")
    sm.main(out)
    r = json.loads((out / "gate-summary.json").read_text())
    check(r["gate"]["cand1"]["kl_pass"] and not r["gate_extra"]["long"]["cand1"]["kl_pass"] and r["verdict"]["cand1"] == "FAIL",
          "summary: standard set passes, long set shifted -> FAIL (both sets required)")
    (out / "rec-cand1-score-long.D.jsonl").unlink()
    sm.main(out)
    check(json.loads((out / "gate-summary.json").read_text())["verdict"]["cand1"] == "INCOMPLETE", "summary: long set missing -> INCOMPLETE")
    r = make("hitok", 0.0, 190, good, [], hit=0.0)[1]
    check(r["verdict"]["cand1"] == "PASS" and r["gate_hit"]["cand1"]["kl_pass"], "summary: hit gate, equal -> PASS")
    r = make("hitfail", 0.0, 190, good, [], hit=0.004)[1]
    check(r["verdict"]["cand1"] == "FAIL" and r["gate"]["cand1"]["kl_pass"], "summary: cold pass ok, hit pass shifted -> FAIL")


STUB_SRUN = r"""#!/bin/bash
# stub srun for the hjob dry run: host GPU query, fake servers, fake client phase, real summary
args="$*"
if [[ $args == *nvidia-smi* ]]; then exit 0; fi
if [[ $args == *pdgate_summary.py* ]]; then exec python3 $STUB_TOOLS/pdgate_summary.py $STUB_OUT; fi
if [[ $args == *pdgate_phase.sh* ]]; then
  arm=$(echo "$args" | grep -oE 'pdgate_phase.sh [a-z0-9]+' | cut -d' ' -f2)
  exec python3 $STUB_TOOLS/pdgate_selftest.py _fakephase $STUB_OUT $arm
fi
if [[ $args == *"eval exec"* ]]; then
  a=$(echo "$args" | grep -oE 'g-[a-z0-9]+-[PD][.]json' | head -1); a=${a%.json}; role=${a##*-}; arm=${a#g-}; arm=${arm%-*}
  echo "PDGATE_ENV PDGATE_ROLE=$role"
  if [[ $arm == cand1 && $role == P ]]; then echo "PDGATE_ENV SGLANG_GDN_K31_FUSED=1"; echo "[TP0] K31_FUSED_ACTIVE 36 1 24"; fi
  if [[ "${STUB_DIE:-}" == "$arm-$role" ]]; then sleep 3; exit 1; fi
  exec sleep 600
fi
echo "stub srun: unknown $args" >&2; exit 99
"""


def fake_phase(out, arm):
    """Client stand-in for the hjob dry run: the files a finished phase leaves (step 0 on P)."""
    import random
    out = Path(out)
    print(f"READY {arm}-P after 1 s\nREADY {arm}-D after 1 s", flush=True)
    if os.environ.get("STUB_SLOW") == arm:
        time.sleep(120)
    rng = random.Random(arm)
    for role in ("P", "D"):
        with open(out / f"rec-{arm}-score.{role}.jsonl", "w") as f:
            for w in range(48):
                for s in range(8):
                    if (s == 0) == (role == "P"):
                        f.write(json.dumps({"window": w, "step": s, "role": role, "kl_s32": 0.02 + rng.gauss(0, 0.002)}) + "\n")
    (out / f"score-meta-{arm}.json").write_text(json.dumps({"by_role": {}, "step0_by_role": {}, "windows": {"0": {"P_cached": 0}}}))
    (out / f"gsm-{arm}.json").write_text(json.dumps({"arm": arm, "correct": 190, "records": [{"correct": i < 190} for i in range(200)]}))
    print(f"PHASE-DONE {arm}", flush=True)


def hjob_tests(tmp):
    import subprocess
    tmp = Path(tmp)
    tools, stub = tmp / "tools", tmp / "stub"
    tools.mkdir()
    stub.mkdir()
    for f in ("pdgate_hjob.sh", "pdgate_seed.py", "pdgate_summary.py", "pdgate_selftest.py"):
        shutil.copy(HERE / f, tools / f)
    (stub / "srun").write_text(STUB_SRUN)
    (stub / "srun").chmod(0o755)
    # stub ss: STUB_SS_BUSY=N -> the first N queries per run report a TIME_WAIT socket, then free; "always" -> never free
    (stub / "ss").write_text("#!/bin/bash\nc=$STUB_OUT/.ss_calls; n=$(cat $c 2>/dev/null || echo 0); echo $((n+1)) > $c\n"
                             "case ${STUB_SS_BUSY:-0} in always) echo 'TIME-WAIT 0 0 127.0.0.1:65301 127.0.0.1:40000';; "
                             "*) [ $n -lt ${STUB_SS_BUSY:-0} ] && echo 'TIME-WAIT 0 0 127.0.0.1:65301 127.0.0.1:40000';; esac\nexit 0\n")
    (stub / "ss").chmod(0o755)
    (stub / "hlib.sh").write_text("W=/tmp; R=/tmp\nh_image() { echo IMG; }\nh_set() { echo $2; }\n")
    (stub / "Cplus.json").write_text(json.dumps({"seed": "x", "seed_MANIFEST_SHA256": "y", "cache_environment": {}}))
    (stub / "cache_seed_node.py").write_text(
        "import json\ndef populate(seed, expected, env, receipt):\n"
        "    r = dict(passed=True, mappings={})\n    open(receipt, 'w').write(json.dumps(r))\n    return r\n")

    def run(name, extra_env):
        out = tmp / name
        out.mkdir()
        (out / "tools").symlink_to(tools)
        exp = {"cand1": {"P": [{"key": "SGLANG_GDN_K31_FUSED", "kind": "line", "regex": r"K31_FUSED_ACTIVE 36 \d+ 24",
                                "forbid_in_before": True},
                               {"key": "SGLANG_GDN_K31_FUSED", "kind": "env", "regex": "PDGATE_ENV SGLANG_GDN_K31_FUSED=1$"}],
                         "D": []}}
        arms = {a: {"step0_owner": "P", "prelude": str(stub / "Cplus.json"), "cache_env": {"P": {}, "D": {}}}
                for a in ("before", "cand1")}
        (out / "plan.json").write_text(json.dumps({"phases": ["before", "cand1"], "run_phases": ["before", "cand1"],
                                                   "ports": {"P": {"http": 65301}, "D": {"http": 65302}},
                                                   "arms": arms, "ref": {"name": "s32"}, "windows": {"n": 48, "cont": 8},
                                                   "expectations": exp, "expectation_notes": []}))
        for a in ("before", "cand1"):
            (out / f"g-{a}.mounts").write_text("@MODEL@:/model:ro")
            for r, g in (("P", "0,1"), ("D", "2,3")):
                (out / f"g-{a}-{r}.json").write_text(json.dumps({"gpus": g}))
        env = dict(os.environ, OUT=str(out), HOLDER_JOB="1", HOLDER_CVD="0,1,2,3", PDGATE_HLIB=str(stub / "hlib.sh"),
                   PATH=f"{stub}:{os.environ['PATH']}", STUB_OUT=str(out), STUB_TOOLS=str(tools), **extra_env)
        t0 = time.time()
        p = subprocess.run(["bash", "-c", f"source {out}/tools/pdgate_hjob.sh"], env=env, capture_output=True, text=True,
                           timeout=400)
        log = (out / "job.log").read_text()
        summ = json.loads((out / "gate-summary.json").read_text()) if (out / "gate-summary.json").exists() else {}
        return p.returncode, log, summ, time.time() - t0

    rc, log, summ, dt = run("ok", {})
    check(rc == 0 and "PHASE before rc=0" in log and "PHASE cand1 rc=0" in log and "END rc=0" in log,
          f"hjob dry run: two phases, rc 0 ({dt:.0f} s)")
    check(summ.get("verdict", {}).get("cand1") == "PASS", "hjob dry run: summary verdict PASS from the stub lines")
    rc, log, summ, dt = run("ports-wait", {"STUB_SS_BUSY": "8"})
    check(rc == 0 and "ports clear before" in log and "PORTS-BUSY" not in log,
          f"hjob ports: TIME_WAIT on the pair's ports -> waits, then runs ({dt:.0f} s)")
    rc, log, summ, dt = run("ports-busy", {"STUB_SS_BUSY": "always", "PDGATE_PORT_WAIT_S": "12"})
    check(rc == 11 and "PORTS-BUSY before before after" in log, "hjob ports: never free -> PORTS-BUSY rc 11 after the wait")
    rc, log, summ, dt = run("die", {"STUB_DIE": "cand1-D", "STUB_SLOW": "cand1"})
    check(rc == 10 and "SERVER-EXIT cand1" in log and dt < 100, f"hjob watchdog: D exits mid-phase -> rc 10 in {dt:.0f} s")
    if dt >= 100:
        print(log)
    check(summ.get("verdict", {}).get("cand1") == "INCOMPLETE", "hjob watchdog: verdict INCOMPLETE")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_fakephase":
        return fake_phase(sys.argv[2], sys.argv[3])
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("prep", "units", "hjob"))
    ap.add_argument("--root")
    ap.add_argument("--acc-audit")
    ap.add_argument("--gate-summary")
    ap.add_argument("--stub-sglang", action="store_true")
    a = ap.parse_args()
    if a.what == "prep":
        prep_tests(a.root)
    elif a.what == "hjob":
        with tempfile.TemporaryDirectory() as tmp:
            hjob_tests(tmp)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            if a.stub_sglang:
                stub_sglang(tmp)
            for name, fn, args in (("probe", probe_tests, (a.acc_audit,)), ("dispatch", dispatch_tests, ()),
                                   ("summary", summary_tests, (a.gate_summary,))):
                d = Path(tmp) / name
                d.mkdir()
                fn(d, *args)
    print(f"SELFTEST {a.what}: {'PASS' if not FAILS else 'FAIL ' + str(len(FAILS))}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
