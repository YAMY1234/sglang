#!/usr/bin/env python3
"""Reject same-line shell declarations whose later assignment uses an earlier one."""

from __future__ import annotations

import argparse
import json
import pathlib
import re


DECLARATION = re.compile(r"^\s*(?:local|declare|export|readonly)\b")
ASSIGNMENT = re.compile(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)=([^\s;]+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    violations = []
    lines_scanned = 0
    declaration_lines = 0
    for path in args.paths:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            lines_scanned += 1
            if not DECLARATION.match(line):
                continue
            declaration_lines += 1
            earlier = []
            for match in ASSIGNMENT.finditer(line):
                name, rhs = match.groups()
                referenced = [
                    prior
                    for prior in earlier
                    if re.search(rf"\$(?:{{{re.escape(prior)}}}|{re.escape(prior)}\b)", rhs)
                ]
                if referenced:
                    violations.append(
                        {
                            "file": str(path),
                            "line": number,
                            "assignment": name,
                            "references_earlier": referenced,
                            "text": line,
                        }
                    )
                earlier.append(name)
    record = {
        "files": [str(path) for path in args.paths],
        "lines_scanned": lines_scanned,
        "declaration_lines": declaration_lines,
        "violations": violations,
        "verdict": "PASS" if not violations else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        "STATIC_SWEEP",
        f"files={len(args.paths)}",
        f"lines={lines_scanned}",
        f"declarations={declaration_lines}",
        f"violations={len(violations)}",
        f"verdict={record['verdict']}",
    )
    if violations:
        for violation in violations:
            print(
                "STATIC_SWEEP_VIOLATION",
                f"{violation['file']}:{violation['line']}",
                f"assignment={violation['assignment']}",
                f"references={','.join(violation['references_earlier'])}",
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
