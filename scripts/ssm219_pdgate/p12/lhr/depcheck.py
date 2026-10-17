"""docs/222 §20.3: submit-time dependency check on the cluster about to run (zero GPU, login node).
  python3 depcheck.py <items.txt> <package dir>   -> DEPCHECK PASS | DEPCHECK FAIL <missing ...>
Every copy / verify / mkdir item of items.txt must exist under $W; and every /work/results/<campaign> directory named inside
the package's prelude compile-cache seeds (support/prelude/*.json 'seed') must exist (the seeded inductor kernels read their
autotune best_config files there; a missing one makes autotune write under read-only /work and the scheduler dies, LHR 252802).
"""
import json, os, re, sys
from pathlib import Path
W = '/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911'


def main():
    items, pkg = Path(sys.argv[1]), Path(sys.argv[2])
    missing, checked = [], 0
    for line in items.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        p = line.split()[0]
        checked += 1
        if not os.path.lexists(os.path.join(W, p)):
            missing.append(p)
    seeds = {json.loads(f.read_text())['seed'] for f in (pkg / 'support/prelude').glob('*.json')}
    pat = re.compile(rb'/work/results/([A-Za-z0-9_.+-]+)')
    named = set()
    for s in seeds:
        for dp, _, fn in os.walk(s):
            for f in fn:
                try:
                    named |= {m.decode() for m in pat.findall(Path(dp, f).read_bytes())}
                except OSError:
                    pass
    for c in sorted(named):
        checked += 1
        if not os.path.isdir(os.path.join(W, 'results', c)):
            missing.append(f'results/{c} (named by the prelude seed)')
    print('DEPCHECK', 'PASS' if not missing else 'FAIL ' + ' '.join(missing), f'checked={checked} seed_campaigns={sorted(named)}')
    sys.exit(1 if missing else 0)


if __name__ == '__main__':
    main()
