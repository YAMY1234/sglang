#!/usr/bin/env python3
"""Streaming R12 point analyzer for benchmark, PP-stage, and TEP-rank evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo


WINDOW_RE = re.compile(
    r"BENCH_(?P<edge>BEGIN|END) (?P<ts>\S+) .*label=(?P<label>formal[123])(?: |$)"
)
BATCH_RE = re.compile(
    r"^\[(?P<ts>[^ ]+ [^ ]+) "
    r"(?P<prefix>(?:PP(?P<pp>\d+) )?TP(?P<tp>\d+) EP\d+)\] "
    r"(?P<body>Prefill batch(?: \[\d+\])?,.*)$"
)
TRACE_RE = re.compile(r"R7_TRACE (?P<body>.*)$")


def field(body: str, name: str, kind: type[int] | type[float]):
    match = re.search(rf"(?:^|, ){re.escape(name)}: ([0-9.]+)(?:,|$)", body)
    return kind(match.group(1)) if match else None


def parse_time(value: str, naive_timezone: str = "UTC") -> dt.datetime:
    if "T" in value:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f").replace(
        tzinfo=ZoneInfo(naive_timezone)
    )


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
    }


def parse_windows(path: Path) -> dict[str, tuple[dt.datetime, dt.datetime]]:
    edges: dict[str, dict[str, dt.datetime]] = defaultdict(dict)
    with path.open(errors="replace") as handle:
        for line in handle:
            match = WINDOW_RE.search(line)
            if match:
                edges[match["label"]][match["edge"]] = parse_time(match["ts"])
    windows = {
        label: (item["BEGIN"], item["END"])
        for label, item in edges.items()
        if "BEGIN" in item and "END" in item
    }
    if sorted(windows) != ["formal1", "formal2", "formal3"]:
        raise RuntimeError(f"expected formal1/2/3 windows, got {sorted(windows)}")
    return windows


def formal_for(
    stamp: dt.datetime, windows: dict[str, tuple[dt.datetime, dt.datetime]]
) -> str | None:
    for label, (begin, end) in windows.items():
        if begin <= stamp <= end:
            return label
    return None


def analyze_bench(paths: list[Path]) -> dict:
    rows = []
    for index, path in enumerate(paths, 1):
        data = json.load(path.open())
        completed = int(data["completed"])
        expected = 160
        incomplete = expected - completed
        throughput = float(data["input_throughput"])
        rows.append(
            {
                "formal": index,
                "path": str(path),
                "completed": completed,
                "expected": expected,
                "incomplete": incomplete,
                "input_throughput": throughput,
                "per_prefill_gpu": throughput / 8,
                "median_ttft_ms": float(data["median_ttft_ms"]),
                "p90_ttft_ms": float(data["p90_ttft_ms"]),
            }
        )
    throughputs = [row["input_throughput"] for row in rows]
    median_total = statistics.median(throughputs)
    median_per_gpu = median_total / 8
    completed = sum(row["completed"] for row in rows)
    incomplete = sum(row["incomplete"] for row in rows)
    request_pass = all(
        row["completed"] == row["expected"] and row["incomplete"] == 0
        for row in rows
    )
    return {
        "rounds": rows,
        "median_input_throughput": median_total,
        "median_per_prefill_gpu": median_per_gpu,
        "median_of_round_median_ttft_ms": statistics.median(
            row["median_ttft_ms"] for row in rows
        ),
        "median_of_round_p90_ttft_ms": statistics.median(
            row["p90_ttft_ms"] for row in rows
        ),
        "max_over_min": max(throughputs) / min(throughputs),
        "jitter_rating": "JITTER" if max(throughputs) / min(throughputs) > 1.15 else "STABLE",
        "roofline_tok_s_per_gpu": 24000.0,
        "roofline_percent": 100 * median_per_gpu / 24000.0,
        "request_trace": {
            "completed": completed,
            "expected": 480,
            "incomplete": incomplete,
            "rating": "PASS" if request_pass else "FAIL",
        },
    }


def analyze_logs(
    topology: str,
    paths: list[Path],
    windows: dict[str, tuple[dt.datetime, dt.datetime]],
    server_log_timezone: str,
) -> dict:
    batches: dict[int, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    trace_rooms: dict[str, set[str]] = defaultdict(set)
    trace_event_counts: dict[str, int] = defaultdict(int)
    trace_formal_event_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for path in paths:
        with path.open(errors="replace") as handle:
            for line_no, line in enumerate(handle, 1):
                match = BATCH_RE.search(line)
                if match:
                    stamp = parse_time(match["ts"], server_log_timezone)
                    formal = formal_for(stamp, windows)
                    if formal:
                        body = match["body"]
                        tokens = field(body, "#new-token", int)
                        usage = field(body, "full token usage", float)
                        if usage is None:
                            usage = field(body, "token usage", float)
                        running = field(body, "#running-req", int)
                        queue = field(body, "#queue-req", int)
                        inflight = field(body, "#inflight-req", int)
                        rate = field(body, "input throughput (token/s)", float)
                        if None in (tokens, usage, running, queue, inflight, rate):
                            continue
                        key = int(match["pp"] if topology == "pp" else match["tp"])
                        batches[key][formal].append(
                            {
                                "stamp": stamp,
                                "tokens": tokens,
                                "usage": usage,
                                "mamba_usage": field(body, "mamba usage", float),
                                "running": running,
                                "queue": queue,
                                "inflight": inflight,
                                "logged_rate": rate,
                                "source": f"{path}:{line_no}",
                            }
                        )
                trace = TRACE_RE.search(line)
                if trace:
                    fields = dict(
                        token.split("=", 1)
                        for token in trace["body"].split()
                        if "=" in token
                    )
                    event = fields.get("event", "unknown")
                    trace_event_counts[event] += 1
                    wall = fields.get("wall_ns")
                    room = fields.get("room")
                    if wall and room:
                        stamp = dt.datetime.fromtimestamp(
                            int(wall) / 1e9, tz=dt.timezone.utc
                        )
                        formal = formal_for(stamp, windows)
                        if formal:
                            trace_rooms[formal].add(room)
                    if wall:
                        stamp = dt.datetime.fromtimestamp(
                            int(wall) / 1e9, tz=dt.timezone.utc
                        )
                        formal = formal_for(stamp, windows)
                        if formal:
                            trace_formal_event_counts[formal][event] += 1

    expected_keys = range(4) if topology == "pp" else range(8)
    entities = {}
    for key in expected_keys:
        per_formal = {}
        combined: list[dict] = []
        for formal in ("formal1", "formal2", "formal3"):
            rows = sorted(batches[key][formal], key=lambda row: row["stamp"])
            combined.extend(rows)
            intervals = [
                (right["stamp"] - left["stamp"]).total_seconds() * 1000
                for left, right in zip(rows, rows[1:])
            ]
            per_formal[formal] = {
                "batches": len(rows),
                "batch_tokens": stats([row["tokens"] for row in rows]),
                "batch_interval_ms": stats(intervals),
                "full_token_usage": stats([row["usage"] for row in rows]),
                "mamba_usage": stats(
                    [row["mamba_usage"] for row in rows if row["mamba_usage"] is not None]
                ),
                "running_req": stats([row["running"] for row in rows]),
                "queue_req": stats([row["queue"] for row in rows]),
                "inflight": stats([row["inflight"] for row in rows]),
                "logged_input_tok_s": stats([row["logged_rate"] for row in rows]),
                "first_source": rows[0]["source"] if rows else None,
                "last_source": rows[-1]["source"] if rows else None,
            }
        max_inflight = max((row["inflight"] for row in combined), default=0)
        # D6's backpressure signature was the conjunction of high token-pool
        # usage (~0.87) and inflight requests pinned at the observed ceiling.
        # Merely reaching the maximum observed inflight value is not saturation.
        pinned_samples = sum(
            row["usage"] >= 0.80
            and max_inflight > 0
            and row["inflight"] == max_inflight
            for row in combined
        )
        entities[str(key)] = {
            "observed": bool(combined),
            "samples": len(combined),
            "max_inflight": max_inflight,
            "pool_pinned_fraction_upper_bound": (
                pinned_samples / len(combined) if combined else None
            ),
            "per_formal": per_formal,
        }

    interval_p50 = []
    pinned = []
    for entity in entities.values():
        if entity["pool_pinned_fraction_upper_bound"] is not None:
            pinned.append(entity["pool_pinned_fraction_upper_bound"])
        for formal in entity["per_formal"].values():
            value = formal["batch_interval_ms"]["p50"]
            if value is not None:
                interval_p50.append(value)
    r7_counts = {label: len(trace_rooms[label]) for label in sorted(windows)}
    r7_total = sum(r7_counts.values())
    formal_engine_returns = {
        label: trace_formal_event_counts[label].get("engine_call_return", 0)
        for label in sorted(windows)
    }
    mechanism = {
        "topology": topology,
        "entity_kind": "PP stage" if topology == "pp" else "TEP rank",
        "mechanism_basis": "scheduler Prefill batch logs",
        "entities": entities,
        "max_entity_batch_interval_p50_ms": max(interval_p50, default=None),
        "max_pool_pinned_fraction_upper_bound": max(pinned, default=None),
        "d6_mechanism_disappeared": (
            topology == "pp"
            and bool(interval_p50)
            and max(interval_p50) < 500
            and bool(pinned)
            and max(pinned) < 0.8
        ),
        "r7_trace_rooms": r7_counts,
        "r7_trace_total_rooms": r7_total,
        "r7_trace_rating": "N/A_NO_TRACE",
        "r7_trace_event_counts": dict(sorted(trace_event_counts.items())),
        "r7_formal_engine_returns": formal_engine_returns,
        "trace_mode": "native_logs",
        "trace_completeness": "N/A_NO_TRACE",
        "kv_transfer": "TRACE_UNAVAILABLE",
        "note": (
            "PP uses PP0..PP3 scheduler cadence; TEP reports TP0..TP7 native "
            "scheduler lines. No per-request transfer event is synthesized."
        ),
    }
    return mechanism


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--topology", choices=("pp", "tep"), required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--prefill", type=Path, action="append", required=True)
    parser.add_argument("--bench-json", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--server-log-timezone",
        default="America/Los_Angeles",
        help="IANA zone for naive scheduler timestamps; wrapper windows are UTC",
    )
    args = parser.parse_args()
    if len(args.bench_json) != 3:
        raise SystemExit("exactly three --bench-json files are required")
    windows = parse_windows(args.windows)
    benchmark = analyze_bench(args.bench_json)
    mechanism = analyze_logs(
        args.topology, args.prefill, windows, args.server_log_timezone
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "arm": args.arm,
        "topology": args.topology,
        "benchmark": benchmark,
        "mechanism": mechanism,
        "windows": str(args.windows),
        "prefill_logs": [str(path) for path in args.prefill],
    }
    (args.out_dir / "point-summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print("R12_POINT", json.dumps(output, sort_keys=True, separators=(",", ":")))
    if benchmark["request_trace"]["rating"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
