"""docs/222 §20 LHR port: compare two manifest.py outputs (CMH vs LHR) item by item.
  python3 compare.py <cmh dir> <lhr dir>   -> one line per item: SAME | DIFF (counts + first differing paths) | MISSING
The model item compares the model files only: CMH's .cache/huggingface download leftovers and LHR-only stage metadata
(.MANIFEST .REVISION .STATUS .gitattributes .list .tree.json LICENSE) are listed, not compared.
"""
import json, sys
from pathlib import Path
MODEL = 'results/bf16-regress-20261013/staging/model'
LHR_STAGE_META = {'.MANIFEST', '.REVISION', '.STATUS', '.gitattributes', '.list', '.tree.json', 'LICENSE'}


def rows(d, key):
    p = d / f'{key}.tsv'
    return {r.split('\t')[0]: r for r in p.read_text().splitlines()} if p.exists() else None


def main():
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    sa, sb = json.loads((a / 'summary.json').read_text()), json.loads((b / 'summary.json').read_text())
    bad = 0
    for item, x in sa.items():
        key = item.replace('/', '__')
        ra, rb = rows(a, key), rows(b, key)
        if rb is None:
            print('MISSING', item); bad += 1; continue
        note = ''
        if item == MODEL:
            only_a = sorted(k for k in ra if k.startswith('.cache/'))
            only_b = sorted(k for k in rb if k in LHR_STAGE_META)
            ra = {k: v for k, v in ra.items() if not k.startswith('.cache/')}
            rb = {k: v for k, v in rb.items() if k not in LHR_STAGE_META}
            note = f' (not compared: CMH .cache/ {len(only_a)} entries, LHR stage metadata {only_b})'
        diff = sorted(k for k in set(ra) | set(rb) if ra.get(k) != rb.get(k))
        if diff:
            bad += 1
            print('DIFF', item, len(diff), diff[:5])
        else:
            print('SAME', item, len(ra), 'entries' + note)
    print('COMPARE', 'PASS' if not bad else f'FAIL {bad}')


if __name__ == '__main__':
    main()
