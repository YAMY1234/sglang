"""Fail closed on ptxas resource violations for every emitted step template."""
import re
from pathlib import Path
import sys
build=Path(sys.argv[1]).read_text()
usage=Path(sys.argv[2]).read_text()
regs=[int(n) for n in re.findall(r'REG:(\d+)',usage)]
assert len(regs)>=12, f'expected 3 bias x INVALIDATE x NORM: {len(regs)}'
assert max(regs)<=128, regs
assert not re.search(r'[1-9]\d* bytes spill (?:stores|loads)',build), 'ptxas spills'
assert not re.search(r'LOCAL:[1-9]\d*',usage), 'local memory'
print('RESOURCE_PASS',regs)
