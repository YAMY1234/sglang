# copied unchanged from integration docs/scripts/ssm_kernel_219/pd/freeze_capsule.py (docs/219 PD stack v1)
"""docs/219 PD stack v1: freeze a git-archived source tree as a capsule the r16-pd / fp32-pd package can authorize.

Same catalog rule as the stock capsule source-42bbc39e-capsule-f28999a8 (CANDIDATE_FILE_SHA256.json: {relative path:
sha256 of the raw bytes}, excluding SHA, BASE_SHA, CANDIDATE_FILE_SHA256.json, FILE_SHA256.json, SOURCE_IDENTITY.json
and every __pycache__ file); symlinks are not catalogued (git archive keeps 3 of them; nothing imports them).
Writes SHA, the catalog and a FREEZE.json receipt, then removes write permission from the whole tree.
  python3 freeze_capsule.py <capsule dir> <full commit sha>   -> prints catalog sha256 and file count
"""
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

EXCLUDE = {'SHA', 'BASE_SHA', 'CANDIDATE_FILE_SHA256.json', 'FILE_SHA256.json', 'SOURCE_IDENTITY.json', 'FREEZE.json'}


def main(root, sha):
    root = Path(root)
    assert len(sha) == 40 and not (root / 'SHA').exists(), 'capsule already frozen or bad sha'
    files = {}
    links = []
    for p in sorted(root.rglob('*')):
        rel = str(p.relative_to(root))
        if '__pycache__' in p.parts or p.name in EXCLUDE and p.parent == root:
            continue
        if p.is_symlink():
            links.append(rel)
            continue
        if p.is_file():
            files[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    (root / 'SHA').write_text(sha + '\n')
    cat = json.dumps(files, indent=2, sort_keys=True) + '\n'
    (root / 'CANDIDATE_FILE_SHA256.json').write_text(cat)
    digest = hashlib.sha256(cat.encode()).hexdigest()
    (root / 'FREEZE.json').write_text(json.dumps(dict(sha=sha, catalog_sha256=digest, files=len(files),
                                                      symlinks_not_catalogued=links), indent=2) + '\n')
    for p in [root, *root.rglob('*')]:
        if not p.is_symlink():
            m = p.stat().st_mode
            p.chmod(m & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print(json.dumps(dict(sha=sha, catalog_sha256=digest, files=len(files), symlinks=links)))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
