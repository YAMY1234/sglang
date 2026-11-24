"""ARIA's wrapper around InferenceX's AgentX client (infra/inferencex/bench.py render; mounted at /aria-agentx).

  python3 agentx.py hook SRC DST
      DST: the pin's srt_agentic.sh (SRC) unchanged but for one line after its benchmark_lib.sh, which sources agentx.sh
      (beside this file): InferenceX's write_agentic_result_json, the step that builds agentx_c<C>.json from aiperf's
      exports, runs as it is under a new name.
  python3 agentx.py defect RESULT_DIR OUTPUT_DIR RESULT_FILENAME RC
      agentx.sh, when that step exited RC: exit 0 with OUTPUT_DIR/agentx-defect.json when the step could not read some
      per-request records (profile_export.jsonl) while aiperf's own aggregate (profile_export_aiperf.json) is complete;
      the client then goes on to InferenceX's validation of that aggregate and its exit codes. Otherwise exit RC.
  python3 agentx.py aiperf SRC DST
      agentx.sh, before InferenceX's install_agentic_deps: prints the aiperf tree to install. SRC (the pin's utils/aiperf)
      already fixed: SRC. At the pin: DST, a copy of SRC (no .git) with the upstream files in aiperf-9adff5e4/ beside this file
      (AIPERF_FIXES). Anything else: exit 1, naming the file.
  python3 agentx.py aiperf-check   (with aiperf's venv Python, after the install)
      exit 1 unless the aiperf it imports carries AIPERF_FIXES.
  python3 agentx.py watch ROOT_PID AIPERF_CLI profile ARGS...
      agentx.sh, beside InferenceX's aiperf replay (ROOT_PID: the shell running it). Once aiperf's profile results are in
      and it waits only for its other result domains (server_metrics): first ARIA_RESULT_COPY (the node spool's copy of
      the per-request records and logs to shared storage), then SIGINT to aiperf's main process after ARIA_RESULT_WAIT_S
      (default 300) or at the job's end (a holder iteration's ARIA_ITERATION_END_TIME, else SLURM_JOB_END_TIME) less
      ARIA_RESULT_RESERVE_S (default 900), whichever is
      first. aiperf's own cancel path exports what it has (profile_export_aiperf.json, without server metrics); once
      aiperf has exited, ARIA_RESULT_COPY again, before InferenceX's result step. A cut is recorded in
      AGENTIC_OUTPUT_DIR/agentx-result-wait.json (result.py: stats.result_wait). After the SIGINT, no signal at all while
      aiperf runs (its records manager's last results, its loop holding them, its export) until one hard
      deadline, the job's end less HARD_S (300 s: the spool's early copy-back, then InferenceX's result step, about
      1.5 min, and the cleanup), never before 2 x FORCE_S (240 s) after the SIGINT; then SIGKILL to its process tree.
      That is the bounded failure policy: aiperf killed there exports nothing, the client exits as InferenceX says (its
      replay's status, 137), and result.py judges the point from the per-request records copied before the wait. No job
      end (no supported launch: native jobs have Slurm's SLURM_JOB_END_TIME, holder iterations launch.py's
      ARIA_ITERATION_END_TIME): as if the reserve had sent the SIGINT, so RESERVE_S - HARD_S (600 s) after it.
      aiperf takes a SIGINT on its main event loop, which is blocked while it receives its server metrics (in earlier runs:
      taken 43-127 s after it was sent, as they joined), so a second one, its force quit, ends it at best as a SIGKILL
      would (no aggregate) and none is sent. In one run (C720) the records manager's server metrics landed 6 s
      after a SIGINT taken at once, its loop then held them over 2 min, and the force quit then sent 120 s after the take
      killed it before its export (exit 137), 14 min before the job's end.

In one run, one record with a raw control character, written as a full home filesystem filled, stopped InferenceX's loader after the
hour; the client, and the job, exited 1 before the aggregate's validation ran. The point is not qualified either way
(result.py names the defect); what changes is that the job's exit code is InferenceX's verdict on the aggregate.
In one run (C504, 4 prefill workers' and the decode's metrics scraped): aiperf exported server metrics for 35 min after
the window until the job's limit killed it, before any aggregate or copy-back. The rule: at most
5 min for server metrics, then skip them; the records and logs on shared storage before that wait. Healthy full runs
took 5.5-6.3 min there (12 jobs at C300-C504), so most full runs are cut; a recipe that requires the export
(AIPERF_REQUIRED_SERVER_METRIC_PREFIX) then fails InferenceX's check of it, and the client exits as InferenceX says.
"""
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

ANCHOR = 'source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"'  # the full library, after --validation-only
DEFECT = "agentx-defect.json"
WAITED = "agentx-result-wait.json"
WAIT_S, RESERVE_S, FORCE_S, POLL_S, COPY_S = 300, 900, 120, 5, 300  # reserve 15 min before the job end
HARD_S = 300  # aiperf still running at the job's end less this is SIGKILLed (spool.sh ARIA_SPOOL_EARLY_S's 300 s, 19669)
DOMAINS = re.compile(r"Waiting for result domains: ([\w, ]+)")  # aiperf's system_controller, on each change
DONE = "All results received, initiating shutdown"
TOOK = re.compile(r"Signal \d+ received, initiating graceful shutdown")  # aiperf's SIGINT handler, on its event loop
EXPORTING = "Exporting all records"  # aiperf's controller begins its export (exporter_manager.py:69): logged, not a bound
# relpath: (sha256 at the pin, aiperf 754356e9; sha256 fixed, the file in aiperf-9adff5e4/ here). SemiAnalysisAI/aiperf 9adff5e4
# (#52): a SPAWN subagent stays on the aiperf worker of its first turn. At the pin its later turns go to the least-loaded
# worker once its parent's routing entry is gone, and each worker it lands on keeps a full copy of its conversation that
# nothing evicts: about 2 MB per request, the client node's host OOM at c11000 and c14000. Routing only:
# every request, its payload and its time are the same, and so is every metric.
AIPERF_FIXES = {"src/aiperf/credit/sticky_router.py": (
    "60648cf7f2a1836e2466cf53c1c5d9942e7a63133298ea01f196e3e8865494b6",
    "af14c5d48e7b161cb16cda547f06ed37890bc954b2f9f4441f3292c1572d811d")}


def hook(src, dst):
    lines = open(src).read().split("\n")
    at = [i for i, line in enumerate(lines) if line.strip() == ANCHOR]
    if len(at) != 1:
        sys.exit(f"agentx hook: {src} sources {ANCHOR!r} {len(at)} times; expected once")
    lines.insert(at[0] + 1, f"source {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agentx.sh')}  # ARIA")
    open(dst, "w").write("\n".join(lines))
    return 0


def _sha(path):
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
    except OSError as e:
        return f"unreadable ({e.strerror})"


def aiperf(src, dst):
    have = {rel: _sha(os.path.join(src, rel)) for rel in AIPERF_FIXES}
    todo = [rel for rel, (_, fixed) in AIPERF_FIXES.items() if have[rel] != fixed]
    bad = [f"{rel} sha256 {have[rel]}" for rel in todo if have[rel] != AIPERF_FIXES[rel][0]]
    if bad:
        sys.exit(f"agentx aiperf: {src} is neither aiperf 754356e9 nor fixed ({'; '.join(bad)}): re-derive AIPERF_FIXES")
    if todo:
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns(".git"))
        for rel in todo:
            shutil.copyfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "aiperf-9adff5e4", rel), os.path.join(dst, rel))
        print(f"ARIA agentx: aiperf from {dst}: {src} with {', '.join(todo)} of SemiAnalysisAI/aiperf 9adff5e4", file=sys.stderr)
    print(dst if todo else src)
    return 0


def aiperf_check():
    import importlib.util
    for rel, (_, fixed) in AIPERF_FIXES.items():
        path = importlib.util.find_spec(rel.removeprefix("src/").removesuffix(".py").replace("/", ".")).origin
        print(f"ARIA agentx: aiperf's {rel}: {path} sha256 {_sha(path)}", flush=True)
        if _sha(path) != fixed:
            sys.exit(f"agentx aiperf-check: {path} is not the fixed {rel} (sha256 {fixed})")
    return 0


def artifacts(result_dir):  # InferenceX's resolve_artifact_dir (infx/results/agentic/artifacts.py)
    base = os.path.join(result_dir, "aiperf_artifacts")
    if os.path.isfile(os.path.join(base, "profile_export.jsonl")):
        return base
    for child in sorted(os.listdir(base)) if os.path.isdir(base) else ():
        if os.path.isfile(os.path.join(base, child, "profile_export.jsonl")):
            return os.path.join(base, child)
    return base


def aggregate(path):
    """aiperf's aggregate as InferenceX's validator reads it: a JSON object with a numeric request_count.avg; else why not."""
    try:
        d = json.load(open(path))
    except (OSError, ValueError) as e:
        return None, f"{path}: {e}"
    count = (d.get("request_count") or {}).get("avg") if isinstance(d, dict) and isinstance(d.get("request_count"), dict) else None
    if not isinstance(count, (int, float)) or isinstance(count, bool):
        return None, f"{path}: no numeric request_count.avg"
    return {k: (d.get(k) or {}).get("avg") for k in ("request_count", "error_request_count", "completed_request_count")
            if isinstance(d.get(k), dict)}, ""


def scan(path):
    """(lines, [(line number, error)] of the lines InferenceX's loader cannot read: not UTF-8 or not JSON)."""
    n, bad = 0, []
    with open(path, "rb") as f:
        for i, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            n += 1
            try:
                json.loads(raw.decode("utf-8"))
            except ValueError as e:  # UnicodeDecodeError and JSONDecodeError
                bad.append((i, str(e)[:200]))
    return n, bad


def defect(result_dir, output_dir, result_filename, rc):
    rc = int(rc)
    art = artifacts(result_dir)
    agg_path, jsonl = os.path.join(art, "profile_export_aiperf.json"), os.path.join(art, "profile_export.jsonl")
    if os.path.exists(os.path.join(output_dir, f"{result_filename}.json")):
        return rc  # the aggregate was built: the step failed after it, not on the records
    agg, why = aggregate(agg_path)
    if agg is None:
        print(f"ARIA agentx: InferenceX's result step exited {rc} and aiperf's aggregate is not complete ({why})", flush=True)
        return rc
    if not os.path.isfile(jsonl):
        print(f"ARIA agentx: InferenceX's result step exited {rc}; {jsonl} is missing", flush=True)
        return rc
    lines, bad = scan(jsonl)
    if not bad:  # every record reads: the step failed for another reason, which stays the client's
        print(f"ARIA agentx: InferenceX's result step exited {rc} with all {lines} per-request records readable", flush=True)
        return rc
    record = {"defect": "per-request records InferenceX's loader cannot read", "loader_exit": rc,
              "records": os.path.relpath(jsonl, result_dir), "lines": lines, "unreadable": len(bad),
              "first": [{"line": i, "error": e} for i, e in bad[:5]],
              "aggregate": {"file": os.path.relpath(agg_path, result_dir), **agg}, "not_built": f"{result_filename}.json"}
    os.makedirs(output_dir, exist_ok=True)
    tmp = os.path.join(output_dir, DEFECT + ".tmp")
    open(tmp, "w").write(json.dumps(record, indent=1) + "\n")
    os.replace(tmp, os.path.join(output_dir, DEFECT))
    print(f"ARIA agentx: {result_filename}.json not built: {len(bad)} of {lines} lines of {record['records']} unreadable "
          f"(first: line {bad[0][0]}: {bad[0][1]}); aiperf's aggregate is complete (request_count {agg['request_count']:g}); "
          f"recorded in {os.path.join(output_dir, DEFECT)}; InferenceX's validation of the aggregate follows", flush=True)
    return 0


def _ppids():  # {pid: ppid} of every process (pyxis containers share the node's PID namespace)
    out = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    s = f.read()
                out[int(d)] = int(s[s.rindex(")") + 2:].split()[1])
            except (OSError, ValueError, IndexError):
                pass
    return out


def _tree(root):  # ROOT's descendants, breadth first
    kids = {}
    for p, pp in _ppids().items():
        kids.setdefault(pp, []).append(p)
    out, todo = [], [root]
    while todo:
        for c in sorted(kids.get(todo.pop(0), ())):
            out.append(c)
            todo.append(c)
    return out


def _aiperf(root, cli):
    """aiperf's main process under ROOT: the first, breadth first, that runs CLI profile, itself or as its interpreter's
    script (its forked services share its argv and ignore SIGINT; this watcher only names it among its arguments)."""
    for p in _tree(root):
        try:
            with open(f"/proc/{p}/cmdline", "rb") as f:
                a = f.read().decode(errors="replace").split("\0")
        except OSError:
            continue
        if p != os.getpid() and any(a[i:i + 2] == [cli, "profile"] for i in (0, 1)):
            return p
    return None


def _alive(pid):  # not a zombie its shell has yet to reap
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


class _Tail:  # the whole lines appended to a file since the last call
    def __init__(self, path):
        self.path, self.at, self.part = path, 0, b""

    def lines(self):
        try:
            with open(self.path, "rb") as f:
                size = os.fstat(f.fileno()).st_size
                if size < self.at:
                    self.at, self.part = 0, b""
                f.seek(self.at)
                data, self.at = self.part + f.read(size - self.at), size
        except OSError:
            return []
        *whole, self.part = data.split(b"\n")
        return [x.decode(errors="replace") for x in whole]


def _secs(name, default):
    v = os.environ.get(name, "")
    try:
        return float(v) if v and float(v) >= 0 else float(default)
    except ValueError:
        return float(default)


def _iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)) if t else None


def _copy(cmd, why):  # ARIA_RESULT_COPY, bounded
    print(f"ARIA agentx: {why}: {cmd}", flush=True)
    try:
        subprocess.run(["bash", "-c", cmd], timeout=COPY_S)
    except subprocess.TimeoutExpired:
        print(f"ARIA agentx: the copy did not end within {COPY_S} s; going on", flush=True)


def watch(root, argv, poll=POLL_S, force=FORCE_S, hard=HARD_S):
    """agentx.sh's watcher (see the module's doc); returns when aiperf has exited, or 1 after a SIGKILL."""
    root, cli = int(root), argv[0]
    wait, reserve = _secs("ARIA_RESULT_WAIT_S", WAIT_S), _secs("ARIA_RESULT_RESERVE_S", RESERVE_S)
    end = os.environ.get("ARIA_ITERATION_END_TIME") or os.environ.get("SLURM_JOB_END_TIME", "")
    end = int(end) if end.isdigit() and 0 < int(end) - time.time() < 30 * 86400 else None  # unlimited jobs: none
    at = argv.index("--output-artifact-dir") + 1 if "--output-artifact-dir" in argv[:-1] else 0
    out = argv[at].rstrip("/") if at else ""
    tails = [_Tail(p) for p in ((os.path.join(out, "logs", "aiperf.log"), os.path.join(os.path.dirname(out), "benchmark.log")) if out else ())]
    print(f"ARIA agentx: after its profile results aiperf may wait {wait:g} s for its other result domains"
          + (f", and until {_iso(end - reserve)} (the job's end {_iso(end)} less {reserve:g} s)" if end else
             " (no SLURM_JOB_END_TIME: no deadline)") + ("" if out else "; no --output-artifact-dir, so nothing to watch"), flush=True)
    main = armed = early = took = exporting = None
    pending, sent, joined, copy = [], [], False, os.environ.get("ARIA_RESULT_COPY", "")
    record = os.path.join(os.environ.get("AGENTIC_OUTPUT_DIR") or os.environ.get("INFMAX_CONTAINER_WORKSPACE") or ".", WAITED)
    while True:
        if main is None:
            main = _aiperf(root, cli)
            if main is None and not _alive(root):
                return 0
        elif not _alive(main):
            if armed is not None and copy:
                if early is not None:
                    try:
                        early.wait(timeout=COPY_S)
                    except subprocess.TimeoutExpired:
                        pass
                _copy(copy, "aiperf has exited: its export to shared storage before InferenceX's result step")
            return 0
        for t in tails:
            for line in t.lines():
                joined = joined or DONE in line  # every domain joined: aiperf exports as it always does
                if sent and took is None and TOOK.search(line):
                    took = time.time()
                    print(f"ARIA agentx: aiperf took the SIGINT {took - sent[0][0]:.0f} s after it was sent", flush=True)
                if sent and exporting is None and EXPORTING in line:
                    exporting = time.time()
                    print(f"ARIA agentx: aiperf began its export {exporting - sent[0][0]:.0f} s after the SIGINT", flush=True)
                m = DOMAINS.search(line)
                names = [d.strip() for d in m.group(1).split(",") if d.strip()] if m else []
                if names and "profile" not in names and armed is None:
                    armed, pending = time.time(), names
                    if copy:  # in the background: the wait's clock runs from here
                        print(f"ARIA agentx: aiperf's profile results are in: its records and logs to shared storage before "
                              f"it waits for {', '.join(names)}: {copy}", flush=True)
                        early = subprocess.Popen(["bash", "-c", copy])
        now = time.time()
        if main is not None and armed is not None and not joined:
            why = ""  # after the SIGINT, only the hard deadline: the job's end, or with none the reserve's, less HARD_S
            kill_at = max(sent[0][0] + 2 * force, (end or sent[0][0] + reserve) - hard) if sent else None
            if not sent and now - armed >= wait:
                why = f"aiperf waited {now - armed:.0f} s for {', '.join(pending)} after its profile results (ARIA_RESULT_WAIT_S={wait:g})"
            elif not sent and end and now >= end - reserve:
                why = f"the job ends at {_iso(end)}, {end - now:.0f} s from now (ARIA_RESULT_RESERVE_S={reserve:g})"
            elif sent and now >= kill_at:
                why = (f"aiperf still exports {now - exporting:.0f} s after it began" if exporting else
                       f"aiperf still runs {now - sent[-1][0]:.0f} s after {sent[-1][1]}") + (
                    f", at the job's end less {end - kill_at:g} s" if end else "")
            if why:
                sig = "SIGKILL" if sent else "SIGINT"
                for p in ([main] if sig == "SIGINT" else [main] + _tree(main)):
                    try:
                        os.kill(p, getattr(signal, sig))
                    except OSError:
                        pass
                sent.append((now, sig))
                print(f"ARIA agentx: {why}: {sig} to aiperf (pid {main})" + (
                    "; aiperf exports what it has without them, then InferenceX's result step and the copy-back run"
                    if len(sent) == 1 else ""), flush=True)
                rec = {"cut": pending, "why": why, "waited_s": round(now - armed), "signals": [s for _, s in sent],
                       "first_at": _iso(sent[0][0]), "job_end": _iso(end), "wait_s": wait, "reserve_s": reserve}
                try:
                    os.makedirs(os.path.dirname(record), exist_ok=True)
                    with open(record + ".tmp", "w") as f:
                        f.write(json.dumps(rec, indent=1) + "\n")
                    os.replace(record + ".tmp", record)
                except OSError as e:
                    print(f"ARIA agentx: cannot write {record}: {e}", flush=True)
                if sig == "SIGKILL":
                    return 1
        time.sleep(poll)


if __name__ == "__main__":
    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if cmd == "hook" and len(args) == 2:
        sys.exit(hook(*args))
    if cmd == "aiperf" and len(args) == 2:
        sys.exit(aiperf(*args))
    if cmd == "aiperf-check" and not args:
        sys.exit(aiperf_check())
    if cmd == "defect" and len(args) == 4:
        sys.exit(defect(*args))
    if cmd == "watch" and len(args) >= 3 and args[0].isdigit() and args[2] == "profile":
        sys.exit(watch(args[0], args[1:], min(POLL_S, max(0.2, _secs("ARIA_RESULT_WAIT_S", WAIT_S) / 10))))
    sys.exit(__doc__)
