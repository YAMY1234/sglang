"""Reduce unprofiled rounds, first-token/greedy agreement, and profiler counts."""

import argparse
import json
import statistics
from pathlib import Path


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def reduce(root, matched_native=False):
    arms = {}
    for arm in ("original-off", "bridge", "factored"):
        directory = root / "micro" / arm
        rows = read_lines(directory / "probe/steps.jsonl")
        launches = read_lines(directory / "probe/launches.jsonl")
        arms[arm] = {}
        for batch in (1, 8, 32):
            rounds = []
            for repeat in (1, 2, 3):
                tag = f"B{batch}-r{repeat}"
                selected = [row for row in rows if row["tag"] == tag]
                assert len(selected) == 64, (arm, tag, len(selected))
                assert all(row["batch_size"] == batch for row in selected)
                rounds.append(
                    {
                        "round": repeat,
                        "mean_device_ms": statistics.mean(
                            row["device_span_ms"] for row in selected
                        ),
                        "mean_host_ms": statistics.mean(
                            row["host_submit_ms"] for row in selected
                        ),
                        "graph_replays": sum(row.get("can_run_graph", False) for row in selected),
                        "graph_witness_available": all("can_run_graph" in row for row in selected),
                        "per_step_ms": [
                            row["device_span_ms"]
                            for row in sorted(selected, key=lambda x: x["step"])
                        ],
                    }
                )
            traces = [row for row in launches if row["tag"] == f"B{batch}-r3-trace"]
            arms[arm][batch] = {
                "rounds": rounds,
                "median_ms": statistics.median(row["mean_device_ms"] for row in rounds),
                "profiler_kernel_counts": [row["kernel_events"] for row in traces],
            }
    verdicts = {}
    numerical = {}
    for batch, support, reverse in ((1, 50, 61), (8, 57, 79), (32, 57, 94)):
        value = arms["factored"][batch]["median_ms"]
        verdicts[batch] = (
            "support"
            if value <= support
            else ("reverse" if value > reverse else "partial")
        )
        comparisons = {}
        for arm in ("bridge", "factored"):
            first = all_tokens = total_first = total_tokens = 0
            for repeat in (1, 2, 3):
                filename = f"B{batch}-r{repeat}-responses.json"
                a = json.loads((root / "micro" / "original-off" / filename).read_text())
                b = json.loads((root / "micro" / arm / filename).read_text())
                if isinstance(a, dict):
                    a = [a]
                if isinstance(b, dict):
                    b = [b]
                assert len(a) == len(b) == batch
                for x, y in zip(a, b):
                    xx, yy = x["output_ids"], y["output_ids"]
                    assert len(xx) == len(yy) == 65
                    first += xx[0] == yy[0]
                    total_first += 1
                    all_tokens += sum(i == j for i, j in zip(xx, yy))
                    total_tokens += len(xx)
            comparisons[arm] = {
                "first_token_equal": first,
                "first_token_total": total_first,
                "greedy_token_equal": all_tokens,
                "greedy_token_total": total_tokens,
                "scope": "synthetic fixed prompt; not teacher-forced logits/NLL",
            }
        numerical[batch] = comparisons
        if matched_native:
            verdicts[batch] = "support" if value <= arms["original-off"][batch]["median_ms"] else "not_supported"
    return {
        "arms": arms,
        "verdicts": verdicts,
        "numerical": numerical,
        "score": "median of three full-64-step arithmetic means; includes W16 expiry",
        "criterion": "factor <= matched original for every B" if matched_native else "M2 absolute thresholds",
        "m3_eligible": all(v == "support" for v in verdicts.values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--matched-native", action="store_true")
    args = ap.parse_args()
    result = reduce(args.root, args.matched_native)
    (args.root / "factored-summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    for b in (1, 8, 32):
        print(
            b,
            *(
                f"{result['arms'][a][b]['median_ms']:.6f}"
                for a in ("original-off", "bridge", "factored")
            ),
            result["verdicts"][b],
        )
    print(json.dumps(result["numerical"]))


if __name__ == "__main__":
    main()
