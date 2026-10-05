#!/usr/bin/env python3
"""Reconstruct an R12 summary from durable wrapper and benchmark records.

This is the post-mortem path for a Slurm time-limit cut: ``R12_POINT`` lines
are the authoritative analyzed rows, while the original formal benchmark JSON
files independently prove the request gate and throughput values.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PRIMARY_NAMES = [f"{arm}-C{concurrency}" for arm in ("B", "A") for concurrency in (16, 32, 64)]
OPTIONAL_NAMES = ("B-C16-repeat", "A-C16-repeat")
KNOWN_NAMES = set(PRIMARY_NAMES + list(OPTIONAL_NAMES))
SOURCE_SHA = "cb0b3498fcc2f398229b0b8cb9df0a5825e1438a"


def relative_midpoint(left: float, right: float) -> float:
    return abs(left - right) / ((left + right) / 2)


def parse_points(wrapper_log: Path) -> dict[str, dict[str, Any]]:
    points: dict[str, dict[str, Any]] = {}
    with wrapper_log.open(errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            marker = line.find("R12_POINT ")
            if marker < 0:
                continue
            payload = line[marker + len("R12_POINT ") :].strip()
            try:
                row = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{wrapper_log}:{line_number}: invalid R12_POINT JSON: {exc}") from exc
            arm = row.get("arm")
            if arm not in KNOWN_NAMES:
                raise ValueError(f"{wrapper_log}:{line_number}: unexpected arm {arm!r}")
            if arm in points and row != points[arm]:
                raise ValueError(f"{wrapper_log}:{line_number}: conflicting duplicate arm {arm}")
            points[arm] = row
    if not points:
        raise ValueError(f"no R12_POINT records in {wrapper_log}")
    return points


def index_bench_json(bench_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in bench_dir.rglob("*.json"):
        if not path.name.startswith("bench-"):
            continue
        if path.name in index:
            raise ValueError(f"duplicate benchmark basename {path.name}: {index[path.name]} and {path}")
        index[path.name] = path
    return index


def close_enough(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-9)


def validate_point_benchmarks(
    arm: str, row: dict[str, Any], bench_index: dict[str, Path]
) -> list[str]:
    benchmark = row["benchmark"]
    rounds = benchmark["rounds"]
    if len(rounds) != 3:
        raise ValueError(f"{arm}: expected three formal rounds, got {len(rounds)}")

    verified: list[str] = []
    completed_total = 0
    incomplete_total = 0
    for expected_formal, round_row in enumerate(rounds, 1):
        if round_row["formal"] != expected_formal:
            raise ValueError(f"{arm}: formal order mismatch at round {expected_formal}")
        basename = Path(round_row["path"]).name
        if basename not in bench_index:
            raise ValueError(f"{arm}: missing original benchmark JSON {basename}")
        bench_path = bench_index[basename]
        bench = json.loads(bench_path.read_text())
        completed = int(bench["completed"])
        expected = int(round_row["expected"])
        incomplete = expected - completed
        if completed != int(round_row["completed"]):
            raise ValueError(f"{arm}: {basename}: completed differs from R12_POINT")
        if incomplete != int(round_row["incomplete"]):
            raise ValueError(f"{arm}: {basename}: incomplete differs from R12_POINT")
        for key in ("input_throughput", "median_ttft_ms", "p90_ttft_ms"):
            if not close_enough(bench[key], round_row[key]):
                raise ValueError(f"{arm}: {basename}: {key} differs from R12_POINT")
        completed_total += completed
        incomplete_total += incomplete
        verified.append(str(bench_path))

    request_trace = benchmark["request_trace"]
    if (
        completed_total != 480
        or incomplete_total != 0
        or int(request_trace["completed"]) != completed_total
        or int(request_trace["expected"]) != 480
        or int(request_trace["incomplete"]) != 0
        or request_trace["rating"] != "PASS"
    ):
        raise ValueError(
            f"{arm}: request gate failed: completed={completed_total}/480 "
            f"incomplete={incomplete_total} trace={request_trace}"
        )
    return verified


def throughput(row: dict[str, Any]) -> float:
    return float(row["benchmark"]["median_input_throughput"])


def fallback_delta(row: dict[str, Any]) -> float:
    return float(row["benchmark"]["max_over_min"]) - 1.0


def build_summary(
    points: dict[str, dict[str, Any]], job: str, pp_chunk: int, verified_files: dict[str, list[str]]
) -> dict[str, Any]:
    missing_primary = [name for name in PRIMARY_NAMES if name not in points]

    if "B-C16" in points:
        if "B-C16-repeat" in points:
            delta_b = relative_midpoint(throughput(points["B-C16"]), throughput(points["B-C16-repeat"]))
            delta_b_source = "B-C16 first vs repeat relative midpoint"
        else:
            delta_b = fallback_delta(points["B-C16"])
            delta_b_source = "B-C16 formal max/min upper bound (repeat unavailable)"
    else:
        delta_b = None
        delta_b_source = "unavailable: B-C16 missing"

    if "A-C16" in points:
        if "A-C16-repeat" in points:
            delta_a = relative_midpoint(throughput(points["A-C16"]), throughput(points["A-C16-repeat"]))
            delta_a_source = "A-C16 first vs repeat relative midpoint"
        else:
            delta_a = fallback_delta(points["A-C16"])
            delta_a_source = "A-C16 formal max/min upper bound (repeat unavailable)"
    else:
        delta_a = None
        delta_a_source = "unavailable: A-C16 missing"

    threshold = None if delta_a is None or delta_b is None else delta_a + delta_b
    decisions: dict[str, dict[str, Any]] = {}
    for concurrency in (16, 32, 64):
        a_name = f"A-C{concurrency}"
        b_name = f"B-C{concurrency}"
        if a_name not in points or b_name not in points or threshold is None:
            continue
        a = throughput(points[a_name])
        b = throughput(points[b_name])
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

    jitter = {
        name: row["benchmark"]["max_over_min"]
        for name, row in points.items()
        if row["benchmark"]["max_over_min"] > 1.15
    }
    complete = not missing_primary
    return {
        "job": job,
        "pp_chunk": pp_chunk,
        "source_sha": SOURCE_SHA,
        "verdict": "PASS" if complete else "PARTIAL",
        "reconstructed_offline": True,
        "reconstruction_sources": {
            "r12_point_arms": sorted(points),
            "bench_json_verified": verified_files,
        },
        "arms": {name: points[name] for name in sorted(points)},
        "missing_primary_arms": missing_primary,
        "drift": {
            "delta_a": delta_a,
            "delta_a_source": delta_a_source,
            "delta_b": delta_b,
            "delta_b_source": delta_b_source,
            "sum": threshold,
        },
        "decisions": decisions,
        "gates": {
            "REQUEST_COMPLETENESS": "PASS" if points else "FAIL",
            "FIRST_ARM_COLD": (
                "N/A" if delta_b is None else "WARN" if delta_b > 0.05 else "PASS"
            ),
            "FIRST_ARM_COLD_fraction": delta_b,
            "JITTER": jitter,
            "CROSS_JOB_INTERFERENCE": "N/A_DIFFERENT_RACK",
            "SERVICE_LIFECYCLE": "whole_group_restart",
            "TRACE_COMPLETENESS": "N/A_NO_TRACE",
        },
        "trace_mode": "native_logs",
        "decision_rule": (
            "PP4 leads iff A>B and (A-B)/B > delta_A+delta_B; PP4 trails iff "
            "B>A and (B-A)/A > delta_A+delta_B; otherwise tie."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrapper-log", type=Path, required=True)
    parser.add_argument("--bench-dir", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--pp-chunk", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="emit a PARTIAL record when one or more of the six main points is absent",
    )
    args = parser.parse_args()

    points = parse_points(args.wrapper_log)
    bench_index = index_bench_json(args.bench_dir)
    verified_files = {
        arm: validate_point_benchmarks(arm, row, bench_index) for arm, row in points.items()
    }
    summary = build_summary(points, args.job, args.pp_chunk, verified_files)
    if summary["verdict"] != "PASS" and not args.allow_partial:
        missing = ",".join(summary["missing_primary_arms"])
        raise SystemExit(f"cannot reconstruct complete summary; missing primary arms: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        "OFFLINE_BENCH_PROOF=PASS",
        f"arms={len(points)}",
        f"json_files={sum(map(len, verified_files.values()))}",
    )
    print("R12_SUMMARY", json.dumps(summary, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
