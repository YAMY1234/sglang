"""docs/222 §20 LHR port: per-item manifest (every file's sha256 / size, symlink targets, empty dirs) of items.txt.
  python3 manifest.py <items.txt> <out dir> [--only copy,verify] [-j N]   -> <out>/<item-key>.tsv + <out>/summary.json
Item digest = sha256 over the sorted lines "<relpath>\t<f|l|d>\t<sha256 | link target | ->\t<bytes>" (same on both clusters).
"""
import hashlib, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
W = '/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911'


def h(p):
    d = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            d.update(b)
    return d.hexdigest()


def walk(root):
    if os.path.islink(root) or os.path.isfile(root):
        yield '', root
        return
    for dp, dn, fn in os.walk(root):
        rel = os.path.relpath(dp, root)
        for d in list(dn):
            p = os.path.join(dp, d)
            if os.path.islink(p):
                dn.remove(d); yield os.path.normpath(os.path.join(rel, d)), p
        if not dn and not fn:
            yield rel + '/', dp
        for f in fn:
            yield os.path.normpath(os.path.join(rel, f)), os.path.join(dp, f)


def entry(item):
    rel, p = item
    if os.path.islink(p):
        return f'{rel}\tl\t{os.readlink(p)}\t-'
    if rel.endswith('/') or os.path.isdir(p):
        return f'{rel}\td\t-\t-'
    return f'{rel}\tf\t{h(p)}\t{os.path.getsize(p)}'


def main():
    items, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    only = set((sys.argv[sys.argv.index('--only') + 1] if '--only' in sys.argv else 'copy,verify').split(','))
    jobs = int(sys.argv[sys.argv.index('-j') + 1]) if '-j' in sys.argv else 16
    summary = {}
    for line in open(items):
        if not line.strip() or line.startswith('#'): continue
        path, action = line.split()[:2]
        if action not in only: continue
        root = os.path.join(W, path)
        if not os.path.lexists(root):
            summary[path] = dict(action=action, present=False); continue
        with ThreadPoolExecutor(jobs) as ex:
            rows = sorted(ex.map(entry, walk(root)))
        key = path.replace('/', '__')
        (out / f'{key}.tsv').write_text(''.join(r + '\n' for r in rows))
        nb = sum(int(r.split('\t')[3]) for r in rows if r.split('\t')[1] == 'f')
        summary[path] = dict(action=action, present=True, entries=len(rows), files=sum(r.split('\t')[1] == 'f' for r in rows),
                             bytes=nb, digest=hashlib.sha256(''.join(r + '\n' for r in rows).encode()).hexdigest())
        print(path, summary[path]['files'], nb, summary[path]['digest'][:16], flush=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    main()
