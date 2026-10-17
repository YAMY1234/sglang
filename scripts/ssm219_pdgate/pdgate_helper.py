"""docs/222 §15.7 PD gate: build a helper-runtime tree from a twinstar-pd-models commit, in the layout the PD receipts
mount at /work/results/remeasure-v6-fivearm-tp1-20261004/inputs/helper-runtime (helpers/twinstar_sgl, helpers/twinstar,
tests/...), as a tar stream from `git archive` (no worktree copy).  The frozen helper of every PD window is twinstar
deb51f49 with exactly this mapping (helper-source.json: 1221 / 1221 files equal):
    helpers/twinstar_sgl/<f>  <- infra/sglang/twinstar_sgl/<f>      helpers/twinstar/<f> <- twinstar/<f>      tests/<f> <- tests/<f>
A new helper-source.json records the commit, every file hash and the file-level difference to the frozen helper, so the
gate (and the reader) sees everything the swap changes, not only the candidate's own edits.
    python3 pdgate_helper.py --repo <twinstar worktree> --sha <40> --frozen <frozen helper-source.json> \
        | ssh <cmh> "bash -c 'mkdir -p <dest> && tar -xf - -C <dest>'"        (summary on stderr)
"""
import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile

MAP = (("infra/sglang/twinstar_sgl/", "helpers/twinstar_sgl/"), ("twinstar/", "helpers/twinstar/"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--frozen", required=True, help="helper-source.json of the frozen helper (deb51f49)")
    a = ap.parse_args()
    assert len(a.sha) == 40, "full sha"
    frozen = json.load(open(a.frozen))
    tests = sorted(k for k in frozen["files"] if k.startswith("tests/"))
    raw = subprocess.run(["git", "-C", a.repo, "archive", "--format=tar", a.sha, "infra/sglang/twinstar_sgl", "twinstar",
                          *tests], check=True, capture_output=True).stdout
    src = tarfile.open(fileobj=io.BytesIO(raw))
    buf = io.BytesIO()
    dst = tarfile.open(fileobj=buf, mode="w")
    files = {}
    for m in src.getmembers():
        if not m.isfile():
            continue  # directories are recreated; symlinks are not part of the helper
        name = next((new + m.name[len(old):] for old, new in MAP if m.name.startswith(old)), m.name)
        if "__pycache__" in name:
            continue
        data = src.extractfile(m).read()
        files[name] = hashlib.sha256(data).hexdigest()
        info = tarfile.TarInfo(name)
        info.size, info.mode, info.mtime = len(data), m.mode, m.mtime
        dst.addfile(info, io.BytesIO(data))
    old = frozen["files"]
    diff = dict(changed=sorted(k for k in files if k in old and old[k] != files[k]),
                added=sorted(k for k in files if k not in old), removed=sorted(k for k in old if k not in files))
    meta = dict(repository=frozen.get("repository"), sha=a.sha, files=files, built_by="pdgate_helper.py (docs/222 §15.7)",
                frozen_base=frozen.get("sha"), diff_vs_frozen=diff)
    data = json.dumps(meta, indent=2).encode()
    info = tarfile.TarInfo("helper-source.json")
    info.size = len(data)
    dst.addfile(info, io.BytesIO(data))
    dst.close()
    sys.stdout.buffer.write(buf.getvalue())
    sgl = lambda xs: [x for x in xs if x.startswith("helpers/twinstar_sgl/")]
    print(f"helper {a.sha[:11]}: {len(files)} files; vs frozen {frozen.get('sha', '?')[:11]}: changed {len(diff['changed'])} "
          f"(twinstar_sgl {len(sgl(diff['changed']))}), added {len(diff['added'])}, removed {len(diff['removed'])} "
          f"(twinstar_sgl {len(sgl(diff['removed']))})", file=sys.stderr)
    for k in sgl(diff["changed"]) + sgl(diff["added"]) + sgl(diff["removed"]):
        print("  twinstar_sgl", "changed" if k in diff["changed"] else "added" if k in diff["added"] else "removed", k,
              file=sys.stderr)


if __name__ == "__main__":
    main()
