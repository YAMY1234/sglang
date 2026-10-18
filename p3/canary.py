"""Fail closed on a P-role log missing any docs/229 publication canary."""
import argparse
import re
from pathlib import Path


def check(log):
    required = (
        r"GDN P48 AGG contract installed: role=prefill",
        r"PD full-N deferred publication enabled[^\n]*join_offload=True",
        r"K31_FUSED_ACTIVE 36 1 24(?:\s|$)",
        r"R3A_PREFILL_MIXED_ACTIVE",
    )
    missing = [pattern for pattern in required if not re.search(pattern, log)]
    if missing:
        raise ValueError("P5 missing canaries: " + "; ".join(missing))
    summaries = re.findall(r"GDN full-N prefill: role=prefill[^\n]*fallbacks=(\d+)", log)
    if not summaries or any(int(value) for value in summaries):
        raise ValueError("P5 requires a completed full-N summary with zero fallbacks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    check(args.log.read_text())
    print("P5_CANARY_PASS")
