"""Validate exported GLM fallback counts, without affecting a serving rank.

Run after measurement, with /server_info snapshots from every server/DP group.
Missing telemetry fails the receipt; it must never be substituted with zero.
"""

import argparse
import hashlib
import json
from pathlib import Path


def evaluate(snapshots):
    rows = []
    errors = []
    for source, info in snapshots:
        if not isinstance(info, dict):
            errors.append(f"{source}: server_info must be an object")
            continue
        states = info.get("internal_states")
        if not isinstance(states, list) or not states:
            errors.append(f"{source}: missing nonempty internal_states")
            continue
        for index, state in enumerate(states):
            diagnostic = (
                state.get("duet_state_policy") if isinstance(state, dict) else None
            )
            count = (
                diagnostic.get("slot_fallback_count")
                if isinstance(diagnostic, dict)
                else None
            )
            rows.append(
                {"source": source, "internal_state_index": index, "count": count}
            )
            if type(count) is not int or count < 0:
                errors.append(f"{source}[{index}]: missing or invalid fallback count")
            elif count != 0:
                errors.append(f"{source}[{index}]: fallback count {count} != 0")
    if not snapshots:
        errors.append("no server_info snapshots")
    return {
        "passed": not errors,
        "rows": rows,
        "errors": errors,
        "scope": "reported scheduler/DP counts; no GPU admission or math qualification",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-info", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate([(p, json.loads(Path(p).read_text())) for p in args.server_info])
    result["snapshots"] = [
        {"path": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
        for p in args.server_info
    ]
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(0 if result["passed"] else 1)
