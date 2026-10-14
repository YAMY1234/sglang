"""Reduce paired GSM200 or sealed C32 cells without relaxing docs/197 gates."""

import argparse
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def gsm(root):
    receipts = [read(root / arm / "complete.json") for arm in ("original", "factored")]
    assert all(x["complete"] for x in receipts)
    assert receipts[0]["source_manifest"] == receipts[1]["source_manifest"]
    assert receipts[0]["node"].split("-T")[0] == receipts[1]["node"].split("-T")[0]
    results = [read(root / arm / "gsm200.json") for arm in ("original", "factored")]
    assert all(x["complete"] for x in results)
    assert results[0]["sampling"] == results[1]["sampling"]
    assert results[0]["chat_kwargs"] == results[1]["chat_kwargs"]
    a, b = [x["results"] for x in results]
    assert len(a) == len(b) == 200
    for x, y in zip(a, b):
        assert all(x[k] == y[k] for k in ("id", "gold", "prompt_hash"))
    lost = sum(bool(x["correct"]) and not bool(y["correct"]) for x, y in zip(a, b))
    gained = sum(not bool(x["correct"]) and bool(y["correct"]) for x, y in zip(a, b))
    delta = gained - lost
    se = math.sqrt(gained + lost)
    return {
        "complete": True,
        "mode": "gsm200",
        "original_correct": sum(x["correct"] for x in a),
        "factored_correct": sum(x["correct"] for x in b),
        "b": lost,
        "c": gained,
        "delta_count": delta,
        "se_count": se,
        "gate": "PASS" if delta >= -2 * se else "FAIL",
        "rule": "c-b >= -2*sqrt(b+c)",
        "source_manifest": receipts[0]["source_manifest"],
    }


def perf(root):
    from twinstar.bench.flashnext_pd_metrics_reader import load_window_stream

    receipts = [read(root / f"arm-{i}/actual-build.json") for i in (0, 1)]
    complete = [read(root / f"arm-{i}/complete.json") for i in (0, 1)]
    assert all(
        x["complete"] and x["measurement_gate"] == "EVIDENCE_COMPLETE" for x in complete
    )
    assert receipts[0]["source_manifest"] == receipts[1]["source_manifest"]
    assert (
        receipts[0]["sealed_worker_ast_sha256"]
        == receipts[1]["sealed_worker_ast_sha256"]
    )
    assert receipts[0]["node"] != receipts[1]["node"]
    assert receipts[0]["node"].split("-T")[0] == receipts[1]["node"].split("-T")[0]
    cells = [read(root / f"arm-{i}/cells.json")[0] for i in (0, 1)]
    starts = []
    for i, (arm, cell) in enumerate(zip(("original", "seed"), cells)):
        assert cell["complete"] and cell["window_seconds"] == 900 and cell["C"] == 32
        assert (
            cell["warmup"]["complete"]
            and sum(cell["warmup"]["completed_by_lane"]) == 95
        )
        assert cell["drain"]["errors"] == 0
        path = (
            root
            / f"arm-{i}"
            / arm
            / "C32/rep1/aiperf_artifacts/server_metrics_export.json"
        )
        phase = load_window_stream(path, duration=900)["summary"]["phase_time_ranges"][
            "profiling"
        ]
        assert phase["end_ns"] >= phase["start_ns"] + 900 * 10**9
        starts.append(phase["start_ns"])
    overlap = max(0, 900 - abs(starts[1] - starts[0]) / 1e9)
    assert overlap > 0, "paired profiling windows did not overlap"
    a, b = cells
    tps = b["tps_gpu"] / a["tps_gpu"]
    ttft = b["ttft_p50_s"] / a["ttft_p50_s"]
    hit = b["hit"] - a["hit"]
    gates = {"TPS": tps >= 0.97, "TTFT50": ttft <= 1.05, "hit": hit >= -0.01}
    return {
        "complete": True,
        "mode": "C32",
        "original": {k: a[k] for k in ("tps_gpu", "ttft_p50_s", "hit")},
        "factored": {k: b[k] for k in ("tps_gpu", "ttft_p50_s", "hit")},
        "tps_ratio": tps,
        "ttft50_ratio": ttft,
        "hit_delta_pp": hit * 100,
        "gates": gates,
        "gate": "PASS" if all(gates.values()) else "FAIL",
        "phase_start_delta_seconds": (starts[1] - starts[0]) / 1e9,
        "profiling_overlap_seconds": overlap,
        "source_manifest": receipts[0]["source_manifest"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["gsm", "perf"])
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = gsm(args.root) if args.mode == "gsm" else perf(args.root)
    (args.root / "paired-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
