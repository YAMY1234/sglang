"""One repetition of the exams against one served arm = one lm-eval process.

    python -m benchmark.duet_eval.lmeval.run --family kimi --arm duet --base-url http://127.0.0.1:31001 \\
        --tokenizer /models/Kimi-Linear-48B-A3B-Instruct --out OUT/duet --rep 0 \\
        --exams gpqa_se,aime24,aime25,math500,mmlu_se,ifeval --num-concurrent 48 [--limit 20] [--pair-seed 20260930]

Repetition k runs every listed exam whose repetition count exceeds k (tasks.EXAMS: gpqa_se / AIME / MATH-500 / LCB / IFEval 8,
mmlu_se / RULER 1, HumanEval+ 8, LongBench v2 4); LiveCodeBench, AA-LCR, HumanEval+, LongBench v2 and RULER run as their own
processes (--exams lcb / aa_lcr / humaneval_plus / longbench_v2 / ruler), gpqa_se / mmlu_se may (one doc shard per process:
--doc-shard i/n; MMLU's 14042 questions are usually split over nodes; LiveCodeBench, HumanEval+ and LongBench v2 take --doc-shard
too, through lm-eval --samples).  The sampling seed of each request is derived from
(pair_seed + rep, prompt, occurrence of the prompt in this process), identical for both arms when both run the same exams and
doc shards per process (report_lmeval.py checks it).  Output:
OUT/rep<k>/<gen|lcb|aa_lcr|humaneval_plus|longbench_v2|ruler|gpqa_se|mmlu_se>[-s<i>of<n>]/ with lm-eval's results_*.json and samples_*.jsonl, sidecar.jsonl (model.py) and run.json (the exact
lm-eval argv): one subdirectory per process kind, because the general exams and LiveCodeBench run as two processes on the same
repetition (runs before 2026-10-02 wrote rep<k>/ directly and shared one sidecar.jsonl and run.json)."""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

from . import model  # noqa: F401  (registers lm-eval model `sglang-duet`)
from . import tasks


def patch_ruler_sources(data_dir):
    """Serve lm-eval RULER's external data from a local directory (compute nodes have no internet): the essay haystack
    (HF dataset baber/paul_graham_essays -> essays.parquet) and the QA sources (dev-v2.0.json, hotpot_dev_distractor_v1.json),
    copies of the files lm-eval downloads (prepare_data.sh: sha256-pinned; HotpotQA from the Hub mirror namlh2004/hotpotqa),
    processed exactly as lm-eval does.  Must run before the task modules import."""
    import re
    import pyarrow.parquet as pq
    from lm_eval.tasks.ruler import prepare_niah, qa_utils
    data_dir = pathlib.Path(data_dir)
    original = prepare_niah.get_haystack

    def get_haystack(type_haystack):
        if type_haystack != "essay":
            return original(type_haystack)
        essay = " ".join(pq.read_table(data_dir / "essays.parquet").column("text").to_pylist())
        return re.sub(r"\s+", " ", essay).split(" ")

    def download_json(url):
        with open(data_dir / url.rsplit("/", 1)[1], encoding="utf-8") as fh:
            return json.load(fh)

    prepare_niah.get_haystack = get_haystack
    qa_utils.download_json = download_json
    # lm-eval's !function loader re-executes a task module unless its recorded mtime matches; pin it so the patch survives
    qa_utils.__mtime__ = pathlib.Path(qa_utils.__file__).stat().st_mtime_ns


# RULER subtask -> its lm-eval document builder (the yaml's custom_dataset: module in lm_eval.tasks.ruler, function)
RULER_BUILDERS = {**{t: ("niah_utils", t) for t in tasks.RULER_RETRIEVAL}, "vt": ("vt_utils", "get_vt_dataset"),
                  "cwe": ("cwe_utils", "get_cw_dataset"), "fwe": ("fwe_utils", "fwe_download"), "qa_squad": ("qa_utils", "get_squad"),
                  "qa_hotpot": ("qa_utils", "get_hotpotqa")}


def patch_ruler_builds(n, seed, cache_dir=None):
    """Long-context RULER (the protocol's 64K and 128K runs, and 256K-1M): lm-eval builds 500 documents per subtask and length,
    and at 1M tokens each one is a full
    tokenisation of its prompt; only the scored ones are needed.  Changes, all on the document side (prompts, answers and grading
    are lm-eval's own):
    * every lm-eval generator with a `num_samples` argument builds min(n, num_samples) documents;
    * each subtask is built one length at a time from a fresh `seed` state of python's and numpy's RNGs (lm-eval seeds them once
      and builds the subtasks in sequence), so a document depends only on (subtask, tokenizer, length, n) -- runs that split the
      subtasks over nodes, and a later larger n (whose first documents are these), get the same documents;
    * with cache_dir, a build is written there once (<subtask>-<tokenizer>-<length>-n<n>.jsonl) and read by every later run; a
      build of more documents serves a smaller n with its first n (the documents are drawn in sequence from the same state).
    Must run before the task modules import (after patch_ruler_sources)."""
    import fcntl
    import hashlib
    import importlib
    import inspect
    import random
    import datasets
    import numpy as np
    from lm_eval.tasks.ruler import common_utils, prepare_niah

    def capped(fn):
        signature = inspect.signature(fn)

        def call(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            bound.arguments["num_samples"] = min(n, bound.arguments["num_samples"])
            return fn(*bound.args, **bound.kwargs)
        return call

    def per_length(name, builder):
        def build(**kwargs):
            lengths = kwargs.pop("max_seq_lengths", common_utils.DEFAULT_SEQ_LENGTHS)
            tokenizer = common_utils.resolve_tokenizer_name(kwargs)
            tag = pathlib.Path(tokenizer).name + "-" + hashlib.sha1(str(tokenizer).encode()).hexdigest()[:8]
            rows = []
            for length in lengths:
                path = cache_dir / f"{name}-{tag}-{length}-n{n}.jsonl" if cache_dir else None
                with open(path.with_suffix(".lock"), "w") if path else open(os.devnull) as lock:
                    if path:
                        fcntl.flock(lock, fcntl.LOCK_EX)       # several processes may ask for the same build: one builds
                        larger = sorted(cache_dir.glob(f"{name}-{tag}-{length}-n*.jsonl"), key=lambda q: int(q.stem.rsplit("-n", 1)[1]))
                        larger = [q for q in larger if int(q.stem.rsplit("-n", 1)[1]) >= n]
                    if path and larger:
                        with larger[0].open(encoding="utf-8") as fh:
                            built = [json.loads(line) for _, line in zip(range(n), fh)]
                    else:
                        random.seed(seed)
                        np.random.seed(seed)
                        built = builder(**kwargs, max_seq_lengths=[length])["test"].to_list()
                        if path:
                            tmp = path.with_suffix(".tmp")
                            tmp.write_text("".join(json.dumps(r) + "\n" for r in built), encoding="utf-8")
                            tmp.rename(path)
                if len(built) != n:
                    raise SystemExit(f"RULER {name} at {length}: {len(built)} documents built, {n} expected "
                                     "(lm-eval skips a document that does not fit the length)")
                rows += built
            return {"test": datasets.Dataset.from_list(rows, split=datasets.Split.TEST)}
        return build

    if cache_dir:
        cache_dir = pathlib.Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    modules = {m: importlib.import_module(f"lm_eval.tasks.ruler.{m}") for m, _ in RULER_BUILDERS.values()}
    for module in (prepare_niah, *modules.values()):
        for name, fn in list(vars(module).items()):
            if inspect.isfunction(fn) and "num_samples" in inspect.signature(fn).parameters:
                setattr(module, name, capped(fn))
    for task, (m, fn) in RULER_BUILDERS.items():
        setattr(modules[m], fn, per_length(task, getattr(modules[m], fn)))
    for module in modules.values():
        module.__mtime__ = pathlib.Path(module.__file__).stat().st_mtime_ns     # as in patch_ruler_sources


def claim_rep_dir(rep_dir, exams, overwrite):
    """One process per repetition directory: two processes in one directory (e.g. two jobs sharing an OUT with different exam lists
    but the same repetition) would share sidecar.jsonl and overwrite each other's run.json.  The claim file stays after the run."""
    claim = rep_dir / "claim.json"
    record = json.dumps({"exams": exams, "pid": os.getpid(), "host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID"),
                         "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    if overwrite:
        claim.write_text(record)
        return
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise SystemExit(f"{rep_dir} is claimed by another run ({claim.read_text().strip()}); give each job its own OUT or "
                         "disjoint repetitions, rerun the job with RESUME=1 when a time limit interrupted this repetition, or use --overwrite") from None
    with os.fdopen(fd, "w") as fh:
        fh.write(record)


def git_head(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=sorted(tasks.FAMILIES))
    ap.add_argument("--arm", required=True, choices=("duet", "stock"))
    ap.add_argument("--base-url", required=True, help="server or balancer base URL (no path)")
    ap.add_argument("--tokenizer", required=True, help="base checkpoint directory (tokenizer + chat template)")
    ap.add_argument("--out", required=True, type=pathlib.Path, help="arm directory; this run writes <out>/rep<k>/")
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--exams", default="gpqa_se,aime24,aime25,math500,mmlu_se,ifeval")
    ap.add_argument("--num-concurrent", type=int, default=48)
    ap.add_argument("--max-length", type=int, default=262144, help="server context length (generation budget = this - prompt)")
    ap.add_argument("--timeout", type=int, default=86400, help="per-request client timeout, s")
    ap.add_argument("--pair-seed", type=int, default=20260930, help="base of the paired sampling seeds (--no-pair-seed disables seeding)")
    ap.add_argument("--no-pair-seed", action="store_true")
    ap.add_argument("--ruler-lengths", default=",".join(map(str, tasks.RULER_LENGTHS_P0)),
                    help="RULER context lengths (default tasks.RULER_LENGTHS_P0, 32768; the protocol's other runs pass 65536 / 131072)")
    ap.add_argument("--limit", type=int, default=None, help="questions per task (smoke tests only)")
    ap.add_argument("--tasks-dir", type=pathlib.Path, default=None, help="where the duet_*.yaml files go (default <out>/../tasks)")
    ap.add_argument("--lmeval-extra", default="", help="extra lm-eval arguments appended verbatim")
    ap.add_argument("--overwrite", action="store_true", help="discard an earlier run of this repetition in --out")
    ap.add_argument("--all-reps", action="store_true", help="run every listed exam at this repetition even beyond its planned count (extra reps)")
    ap.add_argument("--ruler-per-length", type=int, default=None,
                    help="RULER only: documents per subtask and length (lm-eval builds 500 per length; the protocol scores 100); "
                         "selects them with lm-eval --samples, so it cannot be combined with --limit")
    ap.add_argument("--server-context", type=int, default=None,
                    help="the servers' context length when it differs from --max-length (the per-request generation budget is "
                         "server context - prompt); the job passes it with --max-length = context + 1024 for RULER and context + 229376 for "
                         "AA-LCR and LongBench v2, so lm-eval keeps the whole prompt")
    ap.add_argument("--ruler-build-samples", type=int, default=None,
                    help="RULER only: build this many documents per subtask and length, each subtask and length from a fresh RNG state "
                         "(patch_ruler_builds; default: lm-eval's 500 in one sequence); the 64K and 128K runs set it to --ruler-per-length")
    ap.add_argument("--ruler-cache", default=None, help="with --ruler-build-samples: directory where each build is written once and reused")
    ap.add_argument("--ruler-tasks", default="all", help="RULER subtasks: 'all' (13), 'retrieval' (8 NIAH tasks) or a comma list")
    ap.add_argument("--doc-shard", default=None,
                    help="gpqa_se / mmlu_se / lcb / humaneval_plus / longbench_v2 only: 'i/n' runs docs i, i+n, ... (one shard per job; the "
                         "report pairs simple-evals docs on their se_id, the others on doc_id, which lm-eval --samples keeps from the full set)")
    ap.add_argument("--ruler-data", default=None, help="directory with essays.parquet, dev-v2.0.json, hotpot_dev_distractor_v1.json (offline RULER)")
    args = ap.parse_args()
    if args.ruler_data:
        patch_ruler_sources(args.ruler_data)
    if args.ruler_build_samples:
        patch_ruler_builds(args.ruler_build_samples, seed=1234, cache_dir=args.ruler_cache)       # 1234 = the --seed below
    elif args.ruler_cache:
        raise SystemExit("--ruler-cache needs --ruler-build-samples")

    exams = [e for e in args.exams.split(",") if e]
    tasks_dir = args.tasks_dir or args.out.parent / "tasks"
    built = tasks_dir / "manifest.json"
    if built.is_file() and json.loads(built.read_text())["family"] == args.family:
        manifest = json.loads(built.read_text())["exams"]          # written once by the job before the rep processes start
    else:
        manifest = tasks.build(args.family, tasks_dir, exams, ruler_tasks=tasks.parse_ruler_tasks(args.ruler_tasks))
    selected = [manifest[e]["run_as"] for e in exams if args.all_reps or manifest[e]["reps"] > args.rep]
    if not selected:
        print(f"rep {args.rep}: no exam has that many repetitions; nothing to do", flush=True)
        return
    ruler_only = exams == ["ruler"]
    se = [e for e in exams if e in tasks.SE_EXAMS]
    sharded = exams[0] if len(exams) == 1 and exams[0] in tasks.DOC_SHARD_EXAMS else None      # --doc-shard through lm-eval --samples
    if args.doc_shard and se != exams and not sharded:
        raise SystemExit(f"--doc-shard applies to {'/'.join(tasks.SE_EXAMS + tasks.DOC_SHARD_EXAMS)} runs only (one exam per process "
                         "except the simple-evals pair)")
    kind = exams[0] if len(exams) == 1 and exams[0] in (*tasks.OWN_PROCESS, *tasks.SE_EXAMS) else "ruler" if ruler_only else "gen"
    rep_dir = args.out / f"rep{args.rep}" / (kind + (f"-s{args.doc_shard.replace('/', 'of')}" if args.doc_shard else ""))
    rep_dir.mkdir(parents=True, exist_ok=True)
    claim_rep_dir(rep_dir, exams, args.overwrite)
    stale = list(rep_dir.glob("**/samples_*.jsonl"))
    if stale and not args.overwrite:
        raise SystemExit(f"{rep_dir} already holds {len(stale)} sample files (lm-eval appends); use a fresh --out or --overwrite")
    for p in stale + list(rep_dir.glob("**/results_*.json")) + [rep_dir / "sidecar.jsonl"]:
        p.unlink(missing_ok=True)
    fam = tasks.FAMILIES[args.family]
    if args.ruler_per_length and not ruler_only:
        raise SystemExit("--ruler-per-length applies to a RULER-only run (--exams ruler)")
    chat_kwargs = fam["ruler_chat_template_kwargs"] if ruler_only else fam["chat_template_kwargs"]   # RULER: thinking off
    think_required = 0 if ruler_only else fam["think_required"]
    pair = "none" if (args.no_pair_seed or ruler_only) else str(args.pair_seed + args.rep)     # RULER is greedy: no seed
    model_args = ",".join([
        f"base_url={args.base_url.rstrip('/')}/generate", f"model={args.tokenizer}", "tokenizer_backend=huggingface",
        "tokenized_requests=True", f"num_concurrent={args.num_concurrent}", f"max_length={args.max_length}",
        f"timeout={args.timeout}", "max_retries=5", "trust_remote_code=True", f"chat_template_kwargs={chat_kwargs}",
        f"think_end={fam['think_end']}", f"think_required={think_required}", f"pair_seed={pair}",
        f"sidecar={rep_dir / 'sidecar.jsonl'}"] + ([f"server_context={args.server_context}"] if args.server_context else []))
    lengths = [int(x) for x in args.ruler_lengths.split(",") if x]
    argv = ["lm_eval", "run", "--model", "sglang-duet", "--model_args", model_args, "--tasks", ",".join(selected),
            "--include_path", str(tasks_dir), "--apply_chat_template", "--log_samples", "--output_path", str(rep_dir),
            "--seed", "1234", "--trust_remote_code", "--confirm_run_unsafe_code",
            "--metadata", json.dumps({"max_seq_lengths": lengths, "tokenizer": str(args.tokenizer),
                                      **({"rep": args.rep, "shard": args.doc_shard} if se else {})})]
    if exams == ["lcb"]:
        from .lcb_utils import SYSTEM_MESSAGE                 # the official LiveCodeBench system message, both arms
        argv += ["--system_instruction", SYSTEM_MESSAGE]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.ruler_per_length:
        # lm-eval lays the RULER documents out length by length, 500 (or --ruler-build-samples) per length: the first n of each block
        per_block = args.ruler_build_samples or 500
        if args.ruler_per_length > per_block:
            raise SystemExit(f"--ruler-per-length {args.ruler_per_length} > the {per_block} documents built per length")
        ids = [i for block in range(len(lengths)) for i in range(block * per_block, block * per_block + args.ruler_per_length)]
        argv += ["--samples", json.dumps({name: ids for name in manifest["ruler"]["tasks"]})]
    if args.doc_shard and sharded:
        # documents i, i+n, ... of the full set (lm-eval --samples), each logged under its doc_id in the full set, which the report
        # pairs on.  A reply's sampling seed depends on the repetition, the prompt and its occurrence in this process, not on the
        # other documents of the process: every document keeps the seed of the unsharded run.
        i, n = (int(x) for x in args.doc_shard.split("/"))
        if not 0 <= i < n:
            raise SystemExit(f"--doc-shard {args.doc_shard}: need 0 <= i < n")
        argv += ["--samples", json.dumps({selected[0]: [d for d in range(tasks.EXAMS[sharded]["n_docs"]) if d % n == i]})]
    argv += args.lmeval_extra.split()
    record = {"family": args.family, "arm": args.arm, "rep": args.rep, "exams": exams, "tasks": selected, "argv": argv,
              "fork_commit": git_head(pathlib.Path(__file__).resolve().parents[3]) or os.environ.get("FORK_COMMIT"), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "env": {k: v for k, v in os.environ.items()
                      if k.startswith(("SGLANG_", "HF_", "NVIDIA_TF32")) and "TOKEN" not in k}}   # never a Hub token
    (rep_dir / "run.json").write_text(json.dumps(record, indent=1))
    print("LMEVAL", " ".join(argv), flush=True)
    if "ifeval" in exams:
        # IFEval's language checks call langdetect, which samples unless seeded: a fixed seed grades a reply the same on every run
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0
    sys.argv = argv
    from lm_eval.__main__ import cli_evaluate
    cli_evaluate()
    print(f"REP DONE {args.arm} rep{args.rep} ({time.strftime('%H:%M')})", flush=True)


if __name__ == "__main__":
    main()
