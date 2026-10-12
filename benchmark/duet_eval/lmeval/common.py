"""Readers of an exam's output directory shared by the report and the diagnostics: lm-eval sample-file names, the repetition
directories, the templated prompt and the reply of a sample line, and the sha1 that joins a sample line with its sidecar record
(model.py writes the sidecar's prompt_sha1 and answer_sha1 with this same function).  Standard library only, so the report runs
where lm-eval is not installed."""
import glob
import hashlib
import os
import re

SAMPLE_RE = re.compile(r"samples_(?P<task>.+)_(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)\.jsonl$")


def sha1(text):
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()


def ids_key(ids):
    """The key of a tokenized prompt: model.py derives the sampling seed and the request id from it, lb.py --resume matches on it."""
    return hashlib.blake2b(",".join(map(str, ids)).encode(), digest_size=12).hexdigest()


def first_string(x):
    """The first string of a nested structure: a sample line's `arguments` hold the templated prompt first."""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for v in x.values():
            s = first_string(v)
            if s is not None:
                return s
    if isinstance(x, (list, tuple)):
        for v in x:
            s = first_string(v)
            if s is not None:
                return s
    return None


def response_of(line):
    """The reply of a sample line as lm-eval stored it (model.py already removed the thinking); '' when there is none."""
    r = line.get("resps")
    while isinstance(r, list) and r:
        r = r[0]
    return r if isinstance(r, str) else ""


def rep_dirs(root):
    """[(k, <root>/rep<k>)] in repetition order."""
    dirs = glob.glob(os.path.join(root, "rep[0-9]*"))
    return sorted(((int(os.path.basename(d)[3:]), d) for d in dirs if os.path.basename(d)[3:].isdigit()), key=lambda x: x[0])


def sample_files(rep_dir):
    """[(lm-eval task name, path)] of every samples_*.jsonl below a repetition directory."""
    out = []
    for path in sorted(glob.glob(os.path.join(rep_dir, "**", "samples_*.jsonl"), recursive=True)):
        m = SAMPLE_RE.search(os.path.basename(path))
        if m:
            out.append((m.group("task"), path))
    return out
