"""Attribute diagnostic traces; never substitute profiled time for event timing."""

import argparse
import collections
import gzip
import json
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    result = {}
    for arm in ("original-off", "bridge", "factored"):
        result[arm] = {}
        for batch in (1, 8, 32):
            files = sorted(
                (args.root / "micro" / arm / "probe").glob(
                    f"B{batch}-r3-trace-step*.json.gz"
                )
            )
            rows = []
            totals = collections.Counter()
            for file in files:
                with gzip.open(file, "rt") as f:
                    trace = json.load(f)
                kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
                by_name = collections.Counter()
                for k in kernels:
                    by_name[k["name"]] += k.get("dur", 0) / 1000
                totals.update(by_name)
                # Names of the direct factor kernels are deliberately exact.
                state = {
                    k: v
                    for k, v in by_name.items()
                    if k in ("_step", "_truncate")
                    or "selective_scan_update" in k
                    or "selective_state_update" in k
                }
                rows.append(
                    {
                        "file": file.name,
                        "kernels": len(kernels),
                        "sum_kernel_ms": sum(by_name.values()),
                        "state_kernel_ms": state,
                    }
                )
            result[arm][batch] = {
                "steps": rows,
                "cycle16_mean_kernel_ms": (
                    statistics.mean(r["sum_kernel_ms"] for r in rows[:16])
                    if len(rows) >= 16
                    else None
                ),
                "cycle16_mean_state_ms": (
                    statistics.mean(
                        sum(r["state_kernel_ms"].values()) for r in rows[:16]
                    )
                    if len(rows) >= 16
                    else None
                ),
                "top_kernels_total_ms": totals.most_common(20),
                "mean_kernel_ms": statistics.mean(r["sum_kernel_ms"] for r in rows)
                if rows
                else None,
            }
    (args.root / "factored-trace-summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    for arm, batches in result.items():
        for batch, row in batches.items():
            print(
                arm,
                batch,
                "kernel_ms",
                row["mean_kernel_ms"],
                "top",
                row["top_kernels_total_ms"][:6],
            )


if __name__ == "__main__":
    main()
