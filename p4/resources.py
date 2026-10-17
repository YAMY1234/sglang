"""Fail closed on ptxas resource violations for every emitted step template."""
import re
from pathlib import Path
import sys

build = Path(sys.argv[1]).read_text()
usage = Path(sys.argv[2]).read_text()
steps = [part for part in usage.split(" Function ")[1:]
         if "factored_step_r32_kernel" in part.splitlines()[0]]
assert len(steps) == 12, f"expected 3 bias x INVALIDATE x NORM, got {len(steps)}"
regs = []
for part in steps:
    fields = {k: int(v) for k, v in re.findall(r"(REG|STACK|LOCAL):(\d+)", part)}
    assert fields["REG"] <= 128 and fields["STACK"] == fields["LOCAL"] == 0, fields
    regs.append(fields["REG"])
assert not re.search(r"[1-9]\d* bytes spill (?:stores|loads)", build), "ptxas spills"
print("RESOURCE_PASS", regs)
