#!/usr/bin/env python3
"""Unit coverage for the R12 offline summary reconstruction path."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import reconstruct_r12_summary as reconstruct


def make_row(arm: str, throughput: float, jitter: float = 1.02) -> dict:
    rounds = []
    for formal, scale in enumerate((1.0, 1.01, 0.99), 1):
        rounds.append(
            {
                "formal": formal,
                "path": f"/remote/bench-{arm}-formal{formal}.json",
                "completed": 160,
                "expected": 160,
                "incomplete": 0,
                "input_throughput": throughput * scale,
                "median_ttft_ms": 10.0 + formal,
                "p90_ttft_ms": 20.0 + formal,
            }
        )
    return {
        "arm": arm,
        "topology": "pp" if arm.startswith("A-") else "tep",
        "benchmark": {
            "median_input_throughput": throughput,
            "max_over_min": jitter,
            "rounds": rounds,
            "request_trace": {
                "completed": 480,
                "expected": 480,
                "incomplete": 0,
                "rating": "PASS",
            },
        },
        "mechanism": {"trace_mode": "native_logs", "trace_completeness": "N/A_NO_TRACE"},
    }


class ReconstructionTest(unittest.TestCase):
    def test_complete_with_repeat_and_fallback(self) -> None:
        points = {
            "B-C16": make_row("B-C16", 100.0, 1.04),
            "B-C32": make_row("B-C32", 100.0),
            "B-C64": make_row("B-C64", 100.0),
            "A-C16": make_row("A-C16", 110.0, 1.03),
            "A-C32": make_row("A-C32", 120.0),
            "A-C64": make_row("A-C64", 130.0),
            "B-C16-repeat": make_row("B-C16-repeat", 98.0),
        }
        verified = {name: [f"{name}-{i}.json" for i in range(3)] for name in points}
        summary = reconstruct.build_summary(points, "job7b", 8192, verified)
        self.assertEqual(summary["verdict"], "PASS")
        self.assertEqual(
            summary["drift"]["delta_a_source"],
            "A-C16 formal max/min upper bound (repeat unavailable)",
        )
        self.assertEqual(
            summary["drift"]["delta_b_source"],
            "B-C16 first vs repeat relative midpoint",
        )
        self.assertEqual(summary["decisions"]["C16"]["label"], "PP4_LEADS")

    def test_log_and_bench_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = make_row("B-C16", 100.0)
            wrapper = root / "wrapper.out"
            wrapper.write_text("prefix\nR12_POINT " + json.dumps(row) + "\n")
            for round_row in row["benchmark"]["rounds"]:
                (root / Path(round_row["path"]).name).write_text(
                    json.dumps(
                        {
                            "completed": 160,
                            "input_throughput": round_row["input_throughput"],
                            "median_ttft_ms": round_row["median_ttft_ms"],
                            "p90_ttft_ms": round_row["p90_ttft_ms"],
                        }
                    )
                )
            parsed = reconstruct.parse_points(wrapper)
            verified = reconstruct.validate_point_benchmarks(
                "B-C16", parsed["B-C16"], reconstruct.index_bench_json(root)
            )
            self.assertEqual(len(verified), 3)


if __name__ == "__main__":
    unittest.main()
