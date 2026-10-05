#!/usr/bin/env python3
"""Build the one-line R12 job summary and apply preregistered drift labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(root: Path, name: str) -> dict:
    return json.load((root / name / "point-summary.json").open())


def relative_midpoint(left: float, right: float) -> float:
    return abs(left - right) / ((left + right) / 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--pp-chunk", type=int, required=True)
    parser.add_argument("--decode-restart-fallback", type=int, choices=(0, 1), required=True)
    parser.add_argument("--a-repeat-skipped", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.decode_restart_fallback == 0

    primary_names = [f"{arm}-C{c}" for arm in ("B", "A") for c in (16, 32, 64)]
    arms = {name: load(args.root, name) for name in primary_names}
    b_repeat = load(args.root, "B-C16-repeat")
    if args.a_repeat_skipped:
        a_repeat = None
    else:
        a_repeat = load(args.root, "A-C16-repeat")

    def throughput(row: dict) -> float:
        return row["benchmark"]["median_input_throughput"]

    b_first = throughput(arms["B-C16"])
    b_later = throughput(b_repeat)
    delta_b = relative_midpoint(b_first, b_later)
    if a_repeat is None:
        rounds = [
            row["input_throughput"]
            for row in arms["A-C16"]["benchmark"]["rounds"]
        ]
        delta_a = max(rounds) / min(rounds) - 1
        delta_a_source = "A-C16 formal max/min upper bound (timeout cut)"
    else:
        delta_a = relative_midpoint(throughput(arms["A-C16"]), throughput(a_repeat))
        delta_a_source = "A-C16 first vs repeat relative midpoint"
    threshold = delta_a + delta_b

    decisions = {}
    for concurrency in (16, 32, 64):
        a = throughput(arms[f"A-C{concurrency}"])
        b = throughput(arms[f"B-C{concurrency}"])
        if a > b and (a - b) / b > threshold:
            label = "PP4_LEADS"
        elif b > a and (b - a) / a > threshold:
            label = "PP4_TRAILS"
        else:
            label = "TIE"
        decisions[f"C{concurrency}"] = {
            "a_tok_s": a,
            "b_tok_s": b,
            "pp4_over_tep8": a / b,
            "drift_sum_threshold": threshold,
            "label": label,
        }

    all_rows = dict(arms)
    all_rows["B-C16-repeat"] = b_repeat
    if a_repeat is not None:
        all_rows["A-C16-repeat"] = a_repeat
    jitter = {
        name: row["benchmark"]["max_over_min"]
        for name, row in all_rows.items()
        if row["benchmark"]["max_over_min"] > 1.15
    }
    trace_mode = "native_logs"
    trace_completeness = "N/A_NO_TRACE"
    out = {
        "job": args.job,
        "pp_chunk": args.pp_chunk,
        "source_sha": "cb0b3498fcc2f398229b0b8cb9df0a5825e1438a",
        "arms": all_rows,
        "drift": {
            "delta_a": delta_a,
            "delta_a_source": delta_a_source,
            "delta_b": delta_b,
            "delta_b_source": "B-C16 first vs repeat relative midpoint",
            "sum": threshold,
        },
        "decisions": decisions,
        "gates": {
            "FIRST_ARM_COLD": "WARN" if delta_b > 0.05 else "PASS",
            "FIRST_ARM_COLD_fraction": delta_b,
            "JITTER": jitter,
            "CROSS_JOB_INTERFERENCE": "N/A_DIFFERENT_RACK",
            "SERVICE_LIFECYCLE": "whole_group_restart",
            "A_REPEAT_SKIPPED_TIMEOUT": args.a_repeat_skipped,
            "TRACE_COMPLETENESS": trace_completeness,
        },
        "trace_mode": trace_mode,
        "decision_rule": (
            "PP4 leads iff A>B and (A-B)/B > delta_A+delta_B; PP4 trails iff "
            "B>A and (B-A)/A > delta_A+delta_B; otherwise tie."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("FIRST_ARM_COLD=" + out["gates"]["FIRST_ARM_COLD"], f"fraction={delta_b:.6f}")
    print("R12_SUMMARY", json.dumps(out, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
