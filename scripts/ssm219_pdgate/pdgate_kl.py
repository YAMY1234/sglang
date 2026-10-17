"""docs/222 §15 PD serving numerics gate: teacher-forced KL(48x256) + GSM200 through one 1P:1D pair (lead 10-17 18:44).

Same measurement as the agg gate (docs/219 §8 rule 5, docs/215 acc_audit_kl): the probe runs inside the servers as a custom
logit processor, scores the raw logits of every forced step against a reference top-K dump (top 8192 + one tail bucket),
then forces the target token.  What is new is the PD dispatch and the role split:

  dispatch   the client is its own one-pair PD load balancer (what sglang's mini_lb does): every request goes to P and to D
             at once with the same bootstrap_host / bootstrap_port / bootstrap_room; the answer is D's.
  roles      step 0 (the first output token) is sampled on P when P runs the full model (S, C+), and on D after the
             boundary forward when P is shallow (PC+: P sends a placeholder that D pops, twinstar_sgl/pd_shallow.py
             decode_boundary); steps >= 1 are always D.  Each server knows its role from PDGATE_ROLE; the client names the
             step-0 owner per request.  The non-owner still forces the target (so the history check holds) but writes no
             KL record; every call is logged to calls-*.jsonl so the reducer can prove the role split.
  records    one file per role (rec-<arm>-score.<role>.jsonl): two processes never append to the same file.

The KL arithmetic of PDKLProbe is copied from acc_audit_kl.KLProbe (acc-audit 481041ee0) line for line; the CPU self-test
checks that both give identical numbers on the same logits.

The module must be importable on the server side as `pdgate_kl` (PYTHONPATH /out/tools): the processor is sent as the class
of the imported module (dill pickles it by reference, docs/215 4500095).

    python -m pdgate_kl smoke --p URL --d URL --bootstrap-host H --bootstrap-port N --out smoke.json
    python -m pdgate_kl score --p URL --d URL --bootstrap-host H --bootstrap-port N --windows W --arm A --refs s32=/out/ref-s32 --step0 P|D
    python -m pdgate_kl gsm   --p URL --d URL --bootstrap-host H --bootstrap-port N --data gsm8k.jsonl --arm A --out gsm-A.json
"""
import argparse
import concurrent.futures as cf
import json
import random
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

_MM = {}
ROLES = ("P", "D")


def _memmap(path, shape, dtype, mode):
    import numpy as np
    key = (path, mode)
    if key not in _MM:
        _MM[key] = np.memmap(path, dtype=dtype, mode=mode, shape=tuple(shape))
    return _MM[key]


def _append(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


class PDKLProbe(CustomLogitProcessor):
    def __call__(self, logits, custom_param_list):
        import math
        import os

        import numpy as np
        import torch
        from pdgate_kl import _append, _memmap
        role = os.environ.get("PDGATE_ROLE")
        if role not in ("P", "D"):
            raise RuntimeError(f"PDGATE_ROLE must be P or D, got {role!r}")
        # TP > 1: every rank runs the processor on the same full-vocabulary logits; all ranks force the target, only rank 0
        # scores and writes (CMH 4503507: TP2 wrote each record twice)
        writer = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        for row, params in enumerate(custom_param_list):
            if not params or "targets" not in params:
                continue
            req = params["__req__"]
            ext = getattr(req, "extend_range", None)
            if ext is not None and ext.end < len(req.origin_input_ids):
                continue  # intermediate prefill chunk: no output token
            targets = params["targets"]
            step = len(req.output_ids)
            if not 0 <= step < len(targets) or list(req.output_ids) != targets[:step]:
                raise RuntimeError(f"teacher-forced history differs at step {step} on {role}")
            if step > 0 and role != "D":
                raise RuntimeError(f"PD gate: role {role} sampled decode step {step} (only D decodes)")
            target = targets[step]
            scored = role == "D" if step > 0 else role == params["step0_owner"]
            if writer:
                _append(f"{params['calls']}.{role}.jsonl", {"window": params["window"], "step": step, "role": role,
                                                            "scored": scored})
            if not writer or not scored:
                logits[row].fill_(-float("inf"))
                logits[row, target] = 0
                continue
            # ---- from here on: acc_audit_kl.KLProbe (481041ee0), unchanged except the role field and the record path
            raw = logits[row].float()
            if not torch.isfinite(raw).all().item():
                raise RuntimeError("non-finite raw logits")
            lp = torch.log_softmax(raw, -1)
            w, k = params["window"], params["topk"]
            shape = (params["n_windows"], len(targets), k)
            rec = {"window": w, "step": step, "token": target, "nll": -lp[target].item(), "argmax": int(raw.argmax().item()),
                   "role": role}
            for name, ref in (params.get("refs") or {}).items():
                val = torch.from_numpy(np.array(_memmap(ref + "/val.f32", shape, np.float32, "r")[w, step])).to(lp.device).double()
                idx = torch.from_numpy(np.array(_memmap(ref + "/idx.i32", shape, np.int32, "r")[w, step])).to(lp.device).long()
                lq = lp[idx].double()
                p = val.exp()
                kl = (p * (val - lq)).sum().item()
                pt, qt = 1.0 - p.sum().item(), 1.0 - lq.exp().sum().item()
                if pt > 1e-12:
                    kl += pt * math.log(pt / max(qt, 1e-300))
                rec["kl_" + name] = max(0.0, kl)
            _append(f"{params['record']}.{role}.jsonl", rec)
            logits[row].fill_(-float("inf"))
            logits[row, target] = 0
        return logits


# ------------------------------------------------------------------------------------------------------------- client
def _post(url, payload, timeout=7200):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _flush(endpoint, strict):
    req = urllib.request.Request(endpoint + "/flush_cache", data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            status, body = r.status, r.read()[:200]
    except urllib.error.HTTPError as e:
        status, body = e.code, e.read()[:200]
    if strict and status != 200:
        raise RuntimeError(f"flush_cache {endpoint}: HTTP {status} {body!r}")
    return status


class PD:
    """One P / D pair; every request goes to both with one bootstrap room (sglang mini_lb semantics)."""

    def __init__(self, p, d, host, port):
        self.p, self.d, self.host, self.port = p.rstrip("/"), d.rstrip("/"), host, int(port)
        self._rooms = set()
        self._rng = random.SystemRandom()
        self._lock = threading.Lock()

    def room(self):
        with self._lock:
            while True:
                r = self._rng.randrange(1 << 62)
                if r not in self._rooms:
                    self._rooms.add(r)
                    return r

    def call(self, path, payload, timeout=7200):
        """P in this thread, D in its own thread: the two halves of one request always run together (a shared pool
        could fill with P halves whose transfers wait for D halves still queued behind them)."""
        body = dict(payload, bootstrap_host=self.host, bootstrap_port=self.port, bootstrap_room=self.room())
        box = {}

        def decode():
            try:
                box["d"] = _post(self.d + path, body, timeout)
            except BaseException as e:  # re-raised in the caller
                box["e"] = e

        t = threading.Thread(target=decode, daemon=True)
        t.start()
        op = _post(self.p + path, body, timeout)
        t.join()
        if "e" in box:
            raise box["e"]
        return op, box["d"]

    def flush(self):
        # P serves the prefix from its radix cache; D runs --disable-radix-cache (recorded, not required)
        return {"P": _flush(self.p, True), "D": _flush(self.d, False)}


def _pd(a):
    return PD(a.p, a.d, a.bootstrap_host, a.bootstrap_port)


def smoke(a):
    """First PD request of a phase (canary): the pair must hand off and decode 32 greedy tokens."""
    pd = _pd(a)
    t = time.time()
    op, od = pd.call("/generate", {"text": "Hello, warm-up.", "sampling_params": {"max_new_tokens": 32, "temperature": 0,
                                                                                  "ignore_eos": True}}, timeout=a.timeout)
    res = {"seconds": round(time.time() - t, 1), "P_meta": op.get("meta_info"), "D_meta": od.get("meta_info"),
           "D_text": od.get("text")}
    Path(a.out).write_text(json.dumps(res, indent=1))
    if od["meta_info"]["completion_tokens"] != 32:
        raise SystemExit(f"SMOKE FAIL: D completion_tokens {od['meta_info']['completion_tokens']} != 32")
    print(f"SMOKE ok in {res['seconds']} s: {od.get('text', '')[:80]!r}", flush=True)


def score(a):
    spec = json.loads(Path(a.windows).read_text())
    rows, prefix, cont = spec["windows"], spec["prefix"], spec["cont"]
    out = Path(a.out_root)
    record, calls = out / f"rec-{a.arm}-score{a.tag}", out / f"calls-{a.arm}{a.tag}"
    for r in ROLES:
        for f in (Path(f"{record}.{r}.jsonl"), Path(f"{calls}.{r}.jsonl")):
            if f.exists():
                raise SystemExit(f"{f} exists")
    refs = dict(kv.split("=", 1) for kv in a.refs.split(",")) if a.refs else {}
    pd = _pd(a)
    if a.prefix_hit:  # docs/215 D9: keep P's radix cache, the windows' prefixes come back as hits (restore path)
        if not a.tag:
            raise SystemExit("--prefix-hit is a tagged score pass")
        flushed = None
    else:
        flushed = pd.flush()  # cold prefill: an earlier pass must not serve the prefix from P's radix cache
    import importlib
    proc = importlib.import_module("pdgate_kl").PDKLProbe.to_str()  # by reference, not __main__ by value
    metas = {}

    def one(i):
        params = {"temperature": 0, "max_new_tokens": cont, "ignore_eos": True,
                  "custom_params": {"targets": rows[i][prefix:prefix + cont], "window": i, "n_windows": len(rows),
                                    "topk": a.topk, "refs": refs, "record": str(record), "calls": str(calls),
                                    "step0_owner": a.step0}}
        t = time.time()
        op, od = pd.call("/generate", {"input_ids": rows[i][:prefix], "sampling_params": params,
                                       "custom_logit_processor": proc})
        mp, md = op["meta_info"], od["meta_info"]
        if md["completion_tokens"] != cont:
            raise ValueError(f"window {i}: D returned {md['completion_tokens']} of {cont} forced tokens: {md}")
        if (mp.get("cached_tokens", 0) == 0) == bool(a.prefix_hit):
            raise ValueError(f"window {i}: P cached_tokens {mp.get('cached_tokens')} on a "
                             f"{'prefix-hit' if a.prefix_hit else 'cold'} pass: {mp}")
        metas[i] = {"P_cached": mp.get("cached_tokens"), "D_cached": md.get("cached_tokens"),
                    "P_completion": mp.get("completion_tokens"), "seconds": round(time.time() - t, 2)}

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.conc) as ex:
        list(ex.map(one, range(len(rows))))
    recs = [json.loads(s) for r in ROLES if Path(f"{record}.{r}.jsonl").exists()
            for s in Path(f"{record}.{r}.jsonl").read_text().splitlines()]
    keys = [(r["window"], r["step"]) for r in recs]
    if len(recs) != len(rows) * cont or len(set(keys)) != len(recs):
        raise ValueError(f"{record}: {len(recs)} scored records ({len(set(keys))} unique), expected {len(rows) * cont}")
    by_role = {r: sum(x["role"] == r for x in recs) for r in ROLES}
    step0 = {r: sum(x["role"] == r and x["step"] == 0 for x in recs) for r in ROLES}
    if step0[a.step0] != len(rows):
        raise ValueError(f"step 0 scored by {step0}, expected all {len(rows)} on {a.step0}")
    Path(f"{out}/score-meta-{a.arm}{a.tag}.json").write_text(json.dumps(
        {"flush": flushed, "prefix_hit": bool(a.prefix_hit), "by_role": by_role, "step0_by_role": step0, "step0_owner": a.step0, "seconds": round(time.time() - t0, 1),
         "windows": metas}, indent=1))
    print(f"SCORE {a.arm}{a.tag}: {len(recs)} positions in {time.time() - t0:.0f} s (P {by_role['P']}, D {by_role['D']}); "
          f"mean nll {statistics.fmean(r['nll'] for r in recs):.5f}"
          + "".join(f"; KL[{n}] {statistics.fmean(r['kl_' + n] for r in recs):.6f}" for n in refs), flush=True)


# GSM200 exactly as acc_audit_kl.gsm (prompt, parsing, greedy 1024, thinking off), dispatched through the pair
def _boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i + 7:j - 1]


def _num(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").replace("\\!", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    v = float(m[-1])
    return int(v) if v == int(v) else v


def gsm(a):
    rows = [json.loads(s) for s in Path(a.data).read_text().splitlines()][:a.n]
    pd = _pd(a)

    def one(i):
        q = rows[i]["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        _, o = pd.call("/v1/chat/completions", {"model": "default", "messages": [{"role": "user", "content": q}],
                                                "temperature": 0, "max_tokens": a.max_new,
                                                "chat_template_kwargs": {"enable_thinking": False}})
        text = o["choices"][0]["message"].get("content") or ""
        pred = _num(_boxed(text)) if _boxed(text) is not None else _num(text)
        gold = _num(rows[i]["answer"].split("####")[-1])
        return {"index": i, "gold": gold, "pred": pred, "correct": pred is not None and pred == gold,
                "finish": o["choices"][0].get("finish_reason"), "completion_tokens": o["usage"]["completion_tokens"]}

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.conc) as ex:
        recs = list(ex.map(one, range(len(rows))))
    res = {"arm": a.arm, "n": len(recs), "correct": sum(r["correct"] for r in recs), "seconds": round(time.time() - t0, 1),
           "max_new_tokens": a.max_new, "temperature": 0, "thinking": False, "dispatch": "PD pair", "records": recs}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"GSM {a.arm}: {res['correct']}/{res['n']} in {res['seconds']} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("smoke", "score", "gsm"):
        p = sub.add_parser(name)
        p.add_argument("--p", required=True); p.add_argument("--d", required=True)
        p.add_argument("--bootstrap-host", required=True); p.add_argument("--bootstrap-port", required=True, type=int)
        if name == "smoke":
            p.add_argument("--out", required=True); p.add_argument("--timeout", type=int, default=600)
        elif name == "score":
            p.add_argument("--windows", required=True); p.add_argument("--arm", required=True)
            p.add_argument("--refs", required=True); p.add_argument("--step0", choices=ROLES, required=True)
            p.add_argument("--out-root", default="/out"); p.add_argument("--conc", type=int, default=16)
            p.add_argument("--topk", type=int, default=8192); p.add_argument("--tag", default="")
            p.add_argument("--prefix-hit", action="store_true", help="tagged pass on P's cached prefixes (no flush)")
        else:
            p.add_argument("--data", required=True); p.add_argument("--arm", required=True); p.add_argument("--out", required=True)
            p.add_argument("--n", type=int, default=200); p.add_argument("--conc", type=int, default=64)
            p.add_argument("--max-new", type=int, default=1024)
    a = ap.parse_args()
    {"smoke": smoke, "score": score, "gsm": gsm}[a.cmd](a)


if __name__ == "__main__":
    main()
