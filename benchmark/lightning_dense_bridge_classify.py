"""Derive kernel-only launch counts from frozen Lightning Chrome traces.

Preserves the original timings, JSONL and traces. Completed micro arms only;
safe to run while the remaining qualification arms are still in progress.
"""

import argparse
import collections
import gzip
import hashlib
import json
import statistics
from pathlib import Path


def classify(root):
    execution = json.loads((root / "source-execution.json").read_text())
    inventory = {}
    for completion in sorted(root.glob("micro/*/*/micro-complete.json")):
        arm_root = completion.parent
        probe = arm_root / "probe"
        source = probe / "launches.jsonl"
        destination = probe / "launches-classified.jsonl"
        if destination.exists():
            classified = [
                json.loads(line) for line in destination.read_text().splitlines()
            ]
        else:
            classified = []
            for line in source.read_text().splitlines():
                row = json.loads(line)
                trace_path = probe / f"{row['tag']}-step{row['step']:03d}.json.gz"
                with gzip.open(trace_path, "rt") as trace_file:
                    trace = json.load(trace_file)
                categories = collections.Counter(
                    event.get("cat") for event in trace["traceEvents"]
                )
                kernels = [
                    event
                    for event in trace["traceEvents"]
                    if event.get("cat") == "kernel"
                ]
                row.update(
                    original_kernel_events=row["kernel_events"],
                    original_kernel_device_sum_ms=row["kernel_device_sum_ms"],
                    original_cuda_activity_events=row.get(
                        "cuda_activity_events", row["kernel_events"]
                    ),
                    kernel_events=len(kernels),
                    kernel_device_sum_ms=sum(event.get("dur", 0) for event in kernels)
                    / 1000,
                    memcpy_events=categories["gpu_memcpy"],
                    memset_events=categories["gpu_memset"],
                    gpu_annotation_events=categories["gpu_user_annotation"],
                    trace=str(trace_path.relative_to(root)),
                    classification="Chrome traceEvents cat == kernel",
                )
                classified.append(row)
            temp = destination.with_suffix(".tmp")
            temp.write_text("".join(json.dumps(row) + "\n" for row in classified))
            temp.replace(destination)
        grouped = {}
        for tag in sorted({row["tag"] for row in classified}):
            rows = [row for row in classified if row["tag"] == tag]
            grouped[tag] = {
                "steps": len(rows),
                "actual_B": sorted({row["batch_size"] for row in rows}),
                "kernel_counts": [row["kernel_events"] for row in rows],
                "kernel_mean": statistics.mean(row["kernel_events"] for row in rows),
                "original_cuda_activity_counts": [
                    row["original_cuda_activity_events"] for row in rows
                ],
                "memcpy_counts": [row["memcpy_events"] for row in rows],
                "gpu_annotation_counts": [row["gpu_annotation_events"] for row in rows],
            }
            if all("due_layer_slots" in row for row in rows):
                grouped[tag]["bridge_phase_kernel_means"] = {
                    phase: statistics.mean(
                        row["kernel_events"]
                        for row in rows
                        if bool(row["due_layer_slots"]) == due
                    )
                    for phase, due in (("ordinary", False), ("due", True))
                    if any(bool(row["due_layer_slots"]) == due for row in rows)
                }
        inventory[str(arm_root.relative_to(root))] = grouped
    report = {
        "actual_execution_source_sha": execution["source_sha"],
        "classifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "definition": "Chrome traceEvents cat == kernel; excludes memcpy/memset/GPU annotations",
        "timings_modified": False,
        "original_artifacts_preserved": True,
        "completed_micro_arms": len(inventory),
        "arms": inventory,
    }
    destination = root / "qualification-kernel-counts.json"
    temp = destination.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(destination)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", type=Path, required=True)
    args = parser.parse_args()
    report = classify(args.job_root)
    print(
        json.dumps(
            {
                "completed_micro_arms": report["completed_micro_arms"],
                "actual_execution_source_sha": report["actual_execution_source_sha"],
            }
        )
    )
