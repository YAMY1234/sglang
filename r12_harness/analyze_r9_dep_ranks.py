#!/usr/bin/env python3
"""Summarize DEP4 R7_TRACE events by DP rank with raw-line provenance."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


TRACE_RE = re.compile(r"R7_TRACE (?P<body>.*)$")
PREFIX_DP_RE = re.compile(r"\bDP(?P<dp>[0-3])\b")
WINDOW_RE = re.compile(
    r"BENCH_(?P<edge>BEGIN|END) (?P<ts>\S+) .*label=(?P<label>formal[123])(?: |$)"
)
EVENTS = {
    "prefill_done",
    "sender_enqueue",
    "worker_dequeue",
    "transfer_start",
    "transfer_success",
}


def parse_ts_ns(value: str) -> int:
    stamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return int(stamp.timestamp() * 1e9)


def fields_from_body(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in body.split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


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
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def parse_windows(path: Path) -> dict[str, tuple[int, int]]:
    edges: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(errors="replace") as handle:
        for line in handle:
            match = WINDOW_RE.search(line)
            if match:
                edges[match["label"]][match["edge"]] = parse_ts_ns(match["ts"])
    windows = {
        label: (item["BEGIN"], item["END"])
        for label, item in edges.items()
        if "BEGIN" in item and "END" in item
    }
    if sorted(windows) != ["formal1", "formal2", "formal3"]:
        raise RuntimeError(f"expected three formal windows, got {sorted(windows)}")
    return windows


def formal_for(wall_ns: int, windows: dict[str, tuple[int, int]]) -> str | None:
    for label, (begin, end) in windows.items():
        if begin <= wall_ns <= end:
            return label
    return None


def first(events: dict[str, list[dict]], name: str) -> dict | None:
    values = events.get(name, [])
    return min(values, key=lambda item: item["mono_ns"]) if values else None


def last(events: dict[str, list[dict]], name: str) -> dict | None:
    values = events.get(name, [])
    return max(values, key=lambda item: item["mono_ns"]) if values else None


def analyze(
    windows_path: Path, prefill_path: Path, input_tokens: int
) -> tuple[list[dict], dict]:
    windows = parse_windows(windows_path)
    rooms: dict[tuple[int, int], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    event_counts = Counter()
    with prefill_path.open(errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            match = TRACE_RE.search(line)
            if not match:
                continue
            fields = fields_from_body(match["body"])
            event = fields.get("event")
            if event not in EVENTS or fields.get("is_last") == "False":
                continue
            prefix_dp = PREFIX_DP_RE.search(line)
            try:
                dp = int(fields.get("dp", prefix_dp["dp"] if prefix_dp else ""))
                room = int(fields["room"])
                fields.update(
                    dp=dp,
                    room=room,
                    wall_ns=int(fields["wall_ns"]),
                    mono_ns=int(fields["mono_ns"]),
                    source_line=line_no,
                )
            except (KeyError, TypeError, ValueError):
                continue
            event_counts[(dp, event)] += 1
            rooms[(dp, room)][event].append(fields)

    rows = []
    incomplete = Counter()
    for (dp, room), events in rooms.items():
        selected = {
            "P": first(events, "prefill_done"),
            "E": last(events, "sender_enqueue"),
            "D": first(events, "worker_dequeue"),
            "T": first(events, "transfer_start"),
            "C": last(events, "transfer_success"),
        }
        if selected["P"] is None:
            continue
        formal = formal_for(selected["P"]["wall_ns"], windows)
        if formal is None:
            continue
        missing = [name for name, value in selected.items() if value is None]
        if missing:
            for name in missing:
                incomplete[(dp, name)] += 1
            continue
        p, e, d, t, c = (selected[name]["mono_ns"] for name in "PEDTC")
        row = {
            "formal": formal,
            "dp": dp,
            "room": room,
            "rid": selected["P"].get("rid", ""),
            "b_done_to_enqueue_ms": (e - p) / 1e6,
            "c_queue_wait_ms": (d - e) / 1e6,
            "c_worker_to_start_ms": (t - d) / 1e6,
            "d_start_to_success_ms": (c - t) / 1e6,
            "prefill_done_wall_ns": selected["P"]["wall_ns"],
            "transfer_success_wall_ns": selected["C"]["wall_ns"],
        }
        for name in "PEDTC":
            row[f"{name}_line"] = selected[name]["source_line"]
        rows.append(row)

    rank_summary = {}
    global_prefill_done_spans_s = {}
    global_transfer_success_spans_s = {}
    for label in sorted(windows):
        formal_rows = [row for row in rows if row["formal"] == label]
        prefill_done_ns = sorted(row["prefill_done_wall_ns"] for row in formal_rows)
        transfer_success_ns = sorted(
            row["transfer_success_wall_ns"] for row in formal_rows
        )
        global_prefill_done_spans_s[label] = (
            (prefill_done_ns[-1] - prefill_done_ns[0]) / 1e9
            if len(prefill_done_ns) > 1
            else None
        )
        global_transfer_success_spans_s[label] = (
            (transfer_success_ns[-1] - transfer_success_ns[0]) / 1e9
            if len(transfer_success_ns) > 1
            else None
        )
    for dp in range(4):
        dp_rows = [row for row in rows if row["dp"] == dp]
        per_formal = {}
        for label, (begin, end) in sorted(windows.items()):
            selected = sorted(
                (row for row in dp_rows if row["formal"] == label),
                key=lambda row: row["prefill_done_wall_ns"],
            )
            prefill_done_ns = [row["prefill_done_wall_ns"] for row in selected]
            transfer_success_ns = sorted(
                row["transfer_success_wall_ns"] for row in selected
            )
            prefill_done_intervals_ms = [
                (right - left) / 1e6
                for left, right in zip(prefill_done_ns, prefill_done_ns[1:])
            ]
            transfer_success_intervals_ms = [
                (right - left) / 1e6
                for left, right in zip(
                    transfer_success_ns, transfer_success_ns[1:]
                )
            ]
            duration_s = (end - begin) / 1e9
            global_prefill_span_s = global_prefill_done_spans_s[label]
            global_success_span_s = global_transfer_success_spans_s[label]
            prefill_active_span_s = (
                (prefill_done_ns[-1] - prefill_done_ns[0]) / 1e9
                if len(prefill_done_ns) > 1
                else None
            )
            success_active_span_s = (
                (transfer_success_ns[-1] - transfer_success_ns[0]) / 1e9
                if len(transfer_success_ns) > 1
                else None
            )
            per_formal[label] = {
                "rooms": len(selected),
                "window_seconds": duration_s,
                "observed_prefill_done_input_tok_s": (
                    len(selected) * input_tokens / global_prefill_span_s
                    if global_prefill_span_s
                    else None
                ),
                "observed_transfer_success_input_tok_s": (
                    len(selected) * input_tokens / global_success_span_s
                    if global_success_span_s
                    else None
                ),
                "prefill_done_active_span_input_tok_s": (
                    (len(selected) - 1) * input_tokens / prefill_active_span_s
                    if prefill_active_span_s
                    else None
                ),
                "transfer_success_active_span_input_tok_s": (
                    (len(selected) - 1) * input_tokens / success_active_span_s
                    if success_active_span_s
                    else None
                ),
                "prefill_done_interval_ms": stats(prefill_done_intervals_ms),
                "transfer_success_interval_ms": stats(transfer_success_intervals_ms),
                "d_start_to_success_ms": stats(
                    [row["d_start_to_success_ms"] for row in selected]
                ),
                "first_P_line": selected[0]["P_line"] if selected else None,
                "last_C_line": selected[-1]["C_line"] if selected else None,
            }
        rank_summary[str(dp)] = {
            "rooms": len(dp_rows),
            "b_done_to_enqueue_ms": stats(
                [row["b_done_to_enqueue_ms"] for row in dp_rows]
            ),
            "c_queue_wait_ms": stats([row["c_queue_wait_ms"] for row in dp_rows]),
            "c_worker_to_start_ms": stats(
                [row["c_worker_to_start_ms"] for row in dp_rows]
            ),
            "d_start_to_success_ms": stats(
                [row["d_start_to_success_ms"] for row in dp_rows]
            ),
            "per_formal": per_formal,
        }

    summary = {
        "definition": (
            "Observed rank rates divide that rank's completed input tokens by the "
            "global first-to-last trace span for either prefill_done or final "
            "transfer_success; the four ranks sum to aggregate observed rate over the "
            "corresponding span. Private active-span rates and intervals expose "
            "within-rank skew. Transfer d may overlap across four workers."
        ),
        "windows": str(windows_path),
        "prefill": str(prefill_path),
        "input_tokens_per_room": input_tokens,
        "complete_rooms": len(rows),
        "formal_counts": dict(sorted(Counter(row["formal"] for row in rows).items())),
        "global_prefill_done_spans_s": global_prefill_done_spans_s,
        "global_transfer_success_spans_s": global_transfer_success_spans_s,
        "rank_counts": dict(sorted(Counter(row["dp"] for row in rows).items())),
        "event_counts": {
            f"dp{dp}_{event}": count
            for (dp, event), count in sorted(event_counts.items())
        },
        "incomplete": {
            f"dp{dp}_{name}": count
            for (dp, name), count in sorted(incomplete.items())
        },
        "ranks": rank_summary,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--prefill", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--input-tokens", type=int, default=8192)
    args = parser.parse_args()
    rows, summary = analyze(args.windows, args.prefill, args.input_tokens)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (args.out_dir / "dep4-per-room.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.out_dir / "dep4-rank-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
