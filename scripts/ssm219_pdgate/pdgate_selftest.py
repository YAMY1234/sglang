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
    shutil.rmtree(out)
    # refusals
    for name, args, exc in (
            ("cholqr-alone", ["--before", "C+", "--cand1", "C+", "--cand1-p-env", '{"SGLANG_GDN_K31_CHOLQR_MIXED":"1"}'], SystemExit),
            ("unknown", ["--before", "C+", "--cand1", "C+", "--cand1-p-env", '{"SGLANG_GDN_PD_SHALLOW_FULLN":"1"}'], SystemExit),
            ("prefix-hit", ["--before", "C+", "--cand1", "C+", "--before-p-env", '{"SGLANG_GDN_PREFIX_RESTORE_FUSED":"0"}'],
             SystemExit),
            ("cross-class", ["--before", "C+", "--cand1", "PC+"], AssertionError)):
        raises(lambda: run_prep(root, "refuse-" + name, args), exc, f"prep refuses {name}")
        shutil.rmtree(Path(root) / ("refuse-" + name), ignore_errors=True)
    # PC+ with a declared line for a switch the fork does not have yet
    pp, out, plan = run_prep(root, "pcplus", ["--before", "PC+", "--cand1", "PC+", "--cand1-p-env",
                                              '{"SGLANG_GDN_PD_SHALLOW_FULLN":"1"}',
                                              "--expect", "cand1:P:SGLANG_GDN_PD_SHALLOW_FULLN:PD_SHALLOW_FULLN_ACTIVE"])
    check(plan["arms"]["cand1"]["step0_owner"] == "D", "PC+: step 0 on D (shallow P)")
    check(plan["arms"]["before"]["source_sha"].startswith("7a841e4189"), "PC+: stock capsule tree")
    check(any(e["regex"] == "PD_SHALLOW_FULLN_ACTIVE" for e in plan["expectations"]["cand1"]["P"]), "PC+: --expect line kept")
    shutil.rmtree(out)
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

    def make(name, shift, gsm_c, lines_c, lines_b, step0="P"):
        out = Path(tmp) / name
        out.mkdir()
        plan = {"phases": ["before", "cand1"], "ref": {"name": "s32"}, "windows": {"n": NW, "cont": CONT},
                "arms": {a: {"step0_owner": step0} for a in ("before", "cand1")}, "expectation_notes": [],
                "expectations": {"cand1": {"P": [{"key": "SGLANG_GDN_K31_FUSED", "kind": "line", "regex": r"K31_FUSED_ACTIVE 36 \d+ 24",
                                                  "forbid_in_before": True},
                                                 {"key": "SGLANG_GDN_K31_FUSED", "kind": "env",
                                                  "regex": "PDGATE_ENV SGLANG_GDN_K31_FUSED=1$"}], "D": []}}}
        (out / "plan.json").write_text(json.dumps(plan))
        base = [rng.uniform(0.01, 0.03) for _ in range(NW)]
        per = {}
        for arm, sh in (("before", 0.0), ("cand1", shift)):
            per[arm] = []
            for role in ("P", "D"):
                with open(out / f"rec-{arm}-score.{role}.jsonl", "w") as f:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("prep", "units"))
    ap.add_argument("--root")
    ap.add_argument("--acc-audit")
    ap.add_argument("--gate-summary")
    ap.add_argument("--stub-sglang", action="store_true")
    a = ap.parse_args()
    if a.what == "prep":
        prep_tests(a.root)
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
