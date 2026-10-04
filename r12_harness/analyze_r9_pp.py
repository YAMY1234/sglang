#!/usr/bin/env python3
"""Analyze R7 INFO-only PP stage-skew traces without external dependencies."""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


TRACE_RE = re.compile(r"R7_TRACE (?P<body>.*)$")
WINDOW_RE = re.compile(
    r"BENCH_(?P<edge>BEGIN|END) (?P<ts>\S+) .*label=(?P<label>formal[123])(?: |$)"
)
STAGES = range(4)
SEGMENTS = ("a_pipeline", "b_done_to_enqueue", "c_enqueue_to_start", "d_start_to_success")
DETAIL_SEGMENTS = ("c_queue_wait", "c_worker_to_start")
ALL_SEGMENTS = SEGMENTS + DETAIL_SEGMENTS
NATIVE_PHASES = ("mutex_wait", "cuda_submit", "completion_wait")


def parse_ts_ns(value: str) -> int:
    return int(dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1e9)


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
        "mean_ms": statistics.fmean(values) if values else None,
        "p05_ms": percentile(values, 0.05),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
    }


def ratio_stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "p50": percentile(values, 0.50),
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


def parse_trace(path: Path):
    rooms = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    event_counts = Counter()
    engine_returns = []
    all_enqueues = []
    all_dequeues = []
    with path.open(errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            match = TRACE_RE.search(line)
            if not match:
                continue
            fields = fields_from_body(match["body"])
            event = fields.get("event")
            if event not in {
                "prefill_done",
                "sender_enqueue",
                "worker_dequeue",
                "transfer_start",
                "transfer_success",
                "engine_call_enter",
                "engine_call_return",
            }:
                continue
            event_counts[event] += 1
            if event.startswith("engine_call"):
                if event == "engine_call_return":
                    stage_match = re.search(r"\bPP([0-3])\]", line)
                    try:
                        call = {
                                "stage": int(stage_match.group(1)),
                                "blocks": int(fields["blocks"]),
                                "bytes": int(fields["bytes"]),
                                "elapsed_ms": float(fields["elapsed_ms"]),
                                "thread": fields["thread"],
                                "wall_ns": int(fields["wall_ns"]),
                                "mono_ns": int(fields["mono_ns"]),
                                "source_line": line_no,
                            }
                        if "native_total_ms" in fields:
                            call.update(
                                {
                                    "native_rc": int(fields["native_rc"]),
                                    "native_total_ms": float(fields["native_total_ms"]),
                                    "mutex_wait_ms": float(fields["mutex_wait_ms"]),
                                    "cuda_submit_ms": float(fields["cuda_submit_ms"]),
                                    "completion_wait_ms": float(
                                        fields["completion_wait_ms"]
                                    ),
                                    "cuda_api_ms": float(fields["cuda_api_ms"]),
                                    "query_api_ms": float(fields["query_api_ms"]),
                                    "mutex_completion_ms": float(
                                        fields["mutex_completion_ms"]
                                    ),
                                    "submit_calls": int(fields["submit_calls"]),
                                    "query_calls": int(fields["query_calls"]),
                                    "native_closure_ms": float(
                                        fields["native_closure_ms"]
                                    ),
                                }
                            )
                        engine_returns.append(call)
                    except (AttributeError, KeyError, ValueError):
                        pass
                continue
            try:
                room = int(fields["room"])
                stage = int(fields["stage"])
                wall_ns = int(fields["wall_ns"])
                mono_ns = int(fields["mono_ns"])
            except (KeyError, ValueError):
                continue
            fields.update(
                room=room,
                stage=stage,
                source_line=line_no,
                source_log=str(path),
                wall_ns=wall_ns,
                mono_ns=mono_ns,
            )
            if event == "sender_enqueue":
                all_enqueues.append(fields.copy())
            elif event == "worker_dequeue":
                all_dequeues.append(fields.copy())
            if fields.get("is_last") == "False":
                continue
            rooms[room][stage][event].append(fields)
    return rooms, event_counts, engine_returns, all_enqueues, all_dequeues


def first(events: dict, name: str):
    values = events.get(name, [])
    return min(values, key=lambda item: item["mono_ns"]) if values else None


def last(events: dict, name: str):
    values = events.get(name, [])
    return max(values, key=lambda item: item["mono_ns"]) if values else None


def formal_for(wall_ns: int, windows: dict[str, tuple[int, int]]) -> str | None:
    for label, (begin, end) in windows.items():
        if begin <= wall_ns <= end:
            return label
    return None


def numeric_stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p05": percentile(values, 0.05),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def summarize_native_phases(calls: list[dict]) -> dict:
    traced = [call for call in calls if "native_total_ms" in call]
    valid = [call for call in traced if call["native_rc"] == 0]
    total_ms = sum(call["native_total_ms"] for call in valid)
    phase_sum_ms = {
        phase: sum(call[f"{phase}_ms"] for call in valid)
        for phase in NATIVE_PHASES
    }
    return {
        "definition": (
            "LD_PRELOAD boundaries inside batch_transfer_sync_write: mutex_wait is "
            "pthread_mutex_lock time before first cudaStreamQuery; cuda_submit is "
            "the remaining reset-to-first-query interval (including request prep, "
            "source-readiness synchronization, and CUDA submission); completion_wait "
            "is first-query-to-return. Shares use sums, not medians of ratios."
        ),
        "traced_calls": len(traced),
        "valid_calls": len(valid),
        "native_rc_counts": dict(
            sorted(Counter(call["native_rc"] for call in traced).items())
        ),
        "native_total": stats([call["native_total_ms"] for call in valid]),
        "native_total_sum_ms": total_ms,
        "phase": {
            phase: {
                **stats([call[f"{phase}_ms"] for call in valid]),
                "sum_ms": phase_sum_ms[phase],
                "sum_share": (
                    phase_sum_ms[phase] / total_ms if total_ms else None
                ),
            }
            for phase in NATIVE_PHASES
        },
        "lock_plus_submit_sum_share": (
            (phase_sum_ms["mutex_wait"] + phase_sum_ms["cuda_submit"])
            / total_ms
            if total_ms
            else None
        ),
        "cuda_api": stats([call["cuda_api_ms"] for call in valid]),
        "query_api": stats([call["query_api_ms"] for call in valid]),
        "mutex_completion": stats(
            [call["mutex_completion_ms"] for call in valid]
        ),
        "submit_calls": numeric_stats(
            [call["submit_calls"] for call in valid]
        ),
        "query_calls": numeric_stats([call["query_calls"] for call in valid]),
        "closure": stats([call["native_closure_ms"] for call in valid]),
        "closure_abs_max_ms": max(
            (abs(call["native_closure_ms"]) for call in valid), default=None
        ),
    }


def summarize_rows(rows: list[dict]) -> dict:
    return {
        "start_skew": stats([row["start_skew_ms"] for row in rows]),
        "completion_skew": stats([row["completion_skew_ms"] for row in rows]),
        "critical": stats([row["critical_ms"] for row in rows]),
        "edge_contributions": {
            name: stats([row[f"{name}_edge_contribution_ms"] for row in rows])
            for name in ("a", "b", "c")
        },
        "edge_contribution_shares": {
            "a": ratio_stats([row["a_pipeline_share"] for row in rows]),
            "b": ratio_stats([row["b_done_to_enqueue_share"] for row in rows]),
            "c": ratio_stats([row["c_enqueue_to_start_share"] for row in rows]),
        },
        "stage_segments": {
            str(stage): {
                segment: stats(
                    [row[f"stage{stage}_{segment}_ms"] for row in rows]
                )
                for segment in ALL_SEGMENTS
            }
            for stage in STAGES
        },
        "argmax_counts": {
            "prefill_done": dict(
                sorted(
                    Counter(
                        max(
                            STAGES,
                            key=lambda stage: row[f"stage{stage}_a_pipeline_ms"],
                        )
                        for row in rows
                    ).items()
                )
            ),
            **{
                segment: dict(
                    sorted(
                        Counter(
                            max(
                                STAGES,
                                key=lambda stage: row[f"stage{stage}_{segment}_ms"],
                            )
                            for row in rows
                        ).items()
                    )
                )
                for segment in ALL_SEGMENTS[1:]
            },
            "last_start": dict(
                sorted(Counter(row["late_start_stage"] for row in rows).items())
            ),
            "last_success": dict(
                sorted(Counter(row["slowest_success_stage"] for row in rows).items())
            ),
        },
        "closure_error_abs_max_ms": max(
            (abs(row["closure_error_ms"]) for row in rows), default=None
        ),
    }


def summarize_engine_calls(
    engine_returns: list[dict], windows: dict[str, tuple[int, int]]
) -> dict:
    formal_calls = []
    for call in engine_returns:
        formal = formal_for(call["wall_ns"], windows)
        if formal is not None:
            formal_calls.append({**call, "formal": formal})

    large_intervals = defaultdict(list)
    for call in formal_calls:
        if call["bytes"] < 4_000_000:
            continue
        end_ns = call["mono_ns"]
        start_ns = end_ns - round(call["elapsed_ms"] * 1e6)
        large_intervals[call["stage"]].append((start_ns, end_ns, call))
    starts = {
        stage: sorted(start for start, _, _ in large_intervals[stage])
        for stage in STAGES
    }
    ends = {
        stage: sorted(end for _, end, _ in large_intervals[stage])
        for stage in STAGES
    }

    def active_at(stage: int, timestamp_ns: int) -> int:
        return bisect.bisect_right(starts[stage], timestamp_ns) - bisect.bisect_right(
            ends[stage], timestamp_ns
        )

    concurrency = {}
    for stage in STAGES:
        samples = []
        elapsed_by_other_active = defaultdict(list)
        elapsed_by_other_active_bucket = defaultdict(list)
        calls_by_other_active_bucket = defaultdict(list)
        per_other_stage = defaultdict(list)
        for start_ns, end_ns, call in large_intervals[stage]:
            midpoint_ns = start_ns + (end_ns - start_ns) // 2
            other_counts = {
                other: active_at(other, midpoint_ns)
                for other in STAGES
                if other != stage
            }
            other_active = sum(other_counts.values())
            samples.append(other_active)
            elapsed_by_other_active[other_active].append(call["elapsed_ms"])
            if other_active <= 4:
                bucket = "00-04"
            elif other_active <= 8:
                bucket = "05-08"
            elif other_active <= 12:
                bucket = "09-12"
            elif other_active <= 16:
                bucket = "13-16"
            elif other_active <= 20:
                bucket = "17-20"
            else:
                bucket = "21-24"
            elapsed_by_other_active_bucket[bucket].append(call["elapsed_ms"])
            calls_by_other_active_bucket[bucket].append(call)
            for other, count in other_counts.items():
                per_other_stage[other].append(count)
        concurrency[str(stage)] = {
            "definition": "active >=4,000,000-byte calls from other PP stages at this call midpoint; start reconstructed as return_mono_ns-elapsed_ms",
            "other_stage_active_at_midpoint": numeric_stats(samples),
            "other_stage_any_fraction": (
                sum(value > 0 for value in samples) / len(samples) if samples else None
            ),
            "active_calls_by_other_stage": {
                str(other): numeric_stats(per_other_stage[other])
                for other in STAGES
                if other != stage
            },
            "elapsed_by_other_stage_active": {
                str(count): stats(values)
                for count, values in sorted(elapsed_by_other_active.items())
            },
            "elapsed_by_other_stage_active_bucket": {
                bucket: stats(values)
                for bucket, values in sorted(elapsed_by_other_active_bucket.items())
            },
            "native_phases_by_other_stage_active_bucket": {
                bucket: summarize_native_phases(values)
                for bucket, values in sorted(calls_by_other_active_bucket.items())
            },
        }

    def group(calls: list[dict]) -> dict:
        grouped = {}
        for stage in STAGES:
            stage_calls = [call for call in calls if call["stage"] == stage]
            large_calls = [call for call in stage_calls if call["bytes"] >= 4_000_000]
            pool_threads = sorted(
                {call["thread"] for call in stage_calls if call["thread"].startswith("ThreadPoolExecutor-")}
            )
            direct_threads = sorted(
                {call["thread"] for call in stage_calls if not call["thread"].startswith("ThreadPoolExecutor-")}
            )
            executor_prefixes = sorted(
                {call["thread"].rsplit("_", 1)[0] for call in stage_calls if call["thread"].startswith("ThreadPoolExecutor-")}
            )
            grouped[str(stage)] = {
                "calls": len(stage_calls),
                "blocks_total": sum(call["blocks"] for call in stage_calls),
                "bytes_total": sum(call["bytes"] for call in stage_calls),
                "bytes_gib": sum(call["bytes"] for call in stage_calls) / 2**30,
                "blocks_per_call": numeric_stats(
                    [call["blocks"] for call in stage_calls]
                ),
                "bytes_per_call": numeric_stats(
                    [call["bytes"] for call in stage_calls]
                ),
                "elapsed": stats([call["elapsed_ms"] for call in stage_calls]),
                "elapsed_sum_ms": sum(call["elapsed_ms"] for call in stage_calls),
                "large_call_definition": "bytes >= 4000000",
                "large_calls": len(large_calls),
                "large_bytes_total": sum(call["bytes"] for call in large_calls),
                "large_elapsed": stats(
                    [call["elapsed_ms"] for call in large_calls]
                ),
                "native_phases": summarize_native_phases(stage_calls),
                "large_native_phases": summarize_native_phases(large_calls),
                "pool_executor_prefixes": executor_prefixes,
                "pool_executor_call_counts": dict(
                    sorted(
                        Counter(
                            call["thread"].rsplit("_", 1)[0]
                            for call in stage_calls
                            if call["thread"].startswith("ThreadPoolExecutor-")
                        ).items()
                    )
                ),
                "pool_threads": pool_threads,
                "pool_thread_count": len(pool_threads),
                "pool_thread_call_counts": dict(
                    sorted(
                        Counter(
                            call["thread"]
                            for call in stage_calls
                            if call["thread"].startswith("ThreadPoolExecutor-")
                        ).items()
                    )
                ),
                "direct_threads": direct_threads,
                "direct_thread_call_counts": dict(
                    sorted(
                        Counter(
                            call["thread"]
                            for call in stage_calls
                            if not call["thread"].startswith("ThreadPoolExecutor-")
                        ).items()
                    )
                ),
            }
        return grouped

    return {
        "formal_return_events": len(formal_calls),
        "all_formal": group(formal_calls),
        "per_formal": {
            label: group([call for call in formal_calls if call["formal"] == label])
            for label in sorted(windows)
        },
        "large_call_cross_stage_concurrency": concurrency,
    }


_UINT64_MASK = (1 << 64) - 1


def mix_room_id(room: int) -> int:
    """Mirror the exact temporary SplitMix64 room-shard patch."""
    mixed = (room + 0x9E3779B97F4A7C15) & _UINT64_MASK
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (mixed ^ (mixed >> 31)) & _UINT64_MASK


def interarrival_summary(events: list[dict]) -> dict:
    ordered = sorted(events, key=lambda event: event["mono_ns"])
    gaps_ms = [
        (right["mono_ns"] - left["mono_ns"]) / 1e6
        for left, right in zip(ordered, ordered[1:])
    ]
    mean = statistics.fmean(gaps_ms) if gaps_ms else None
    stddev = statistics.pstdev(gaps_ms) if len(gaps_ms) > 1 else None
    burst_runs = []
    current_run = 1 if ordered else 0
    for gap in gaps_ms:
        if gap <= 10.0:
            current_run += 1
        else:
            burst_runs.append(current_run)
            current_run = 1
    if current_run:
        burst_runs.append(current_run)
    return {
        "events": len(ordered),
        "intervals": stats(gaps_ms),
        "coefficient_of_variation": (
            stddev / mean if mean is not None and not math.isclose(mean, 0.0) else None
        ),
        "gap_fraction_le_0_1ms": (
            sum(gap <= 0.1 for gap in gaps_ms) / len(gaps_ms) if gaps_ms else None
        ),
        "gap_fraction_le_1ms": (
            sum(gap <= 1.0 for gap in gaps_ms) / len(gaps_ms) if gaps_ms else None
        ),
        "gap_fraction_le_10ms": (
            sum(gap <= 10.0 for gap in gaps_ms) / len(gaps_ms) if gaps_ms else None
        ),
        "gap_fraction_le_100ms": (
            sum(gap <= 100.0 for gap in gaps_ms) / len(gaps_ms) if gaps_ms else None
        ),
        "burst_definition": "maximal run whose adjacent gaps are <=10ms",
        "burst_run_size": numeric_stats(burst_runs),
    }


def summarize_arrivals_and_workers(
    rows: list[dict], all_enqueues: list[dict], all_dequeues: list[dict]
) -> dict:
    room_formal = {row["room"]: row["formal"] for row in rows}
    selected_enqueues = [
        {**event, "formal": room_formal[event["room"]]}
        for event in all_enqueues
        if event["room"] in room_formal
    ]
    selected_dequeues = [
        {**event, "formal": room_formal[event["room"]]}
        for event in all_dequeues
        if event["room"] in room_formal
    ]

    def arrival_group(events: list[dict]) -> dict:
        result = {}
        for stage in STAGES:
            stage_events = [event for event in events if event["stage"] == stage]
            per_formal = {
                label: interarrival_summary(
                    [event for event in stage_events if event["formal"] == label]
                )
                for label in ("formal1", "formal2", "formal3")
            }
            # Pool only within-formal gaps; never bridge benchmark boundaries.
            pooled_gaps = []
            for label in ("formal1", "formal2", "formal3"):
                ordered = sorted(
                    [event for event in stage_events if event["formal"] == label],
                    key=lambda event: event["mono_ns"],
                )
                pooled_gaps.extend(
                    (right["mono_ns"] - left["mono_ns"]) / 1e6
                    for left, right in zip(ordered, ordered[1:])
                )
            pooled = interarrival_summary([])
            pooled["events"] = len(stage_events)
            pooled["intervals"] = stats(pooled_gaps)
            mean = statistics.fmean(pooled_gaps) if pooled_gaps else None
            stddev = statistics.pstdev(pooled_gaps) if len(pooled_gaps) > 1 else None
            pooled["coefficient_of_variation"] = (
                stddev / mean
                if mean is not None and not math.isclose(mean, 0.0)
                else None
            )
            for threshold, key in (
                (0.1, "gap_fraction_le_0_1ms"),
                (1.0, "gap_fraction_le_1ms"),
                (10.0, "gap_fraction_le_10ms"),
                (100.0, "gap_fraction_le_100ms"),
            ):
                pooled[key] = (
                    sum(gap <= threshold for gap in pooled_gaps) / len(pooled_gaps)
                    if pooled_gaps
                    else None
                )
            burst_runs = []
            for label in ("formal1", "formal2", "formal3"):
                ordered = sorted(
                    [event for event in stage_events if event["formal"] == label],
                    key=lambda event: event["mono_ns"],
                )
                if not ordered:
                    continue
                current_run = 1
                for left, right in zip(ordered, ordered[1:]):
                    gap = (right["mono_ns"] - left["mono_ns"]) / 1e6
                    if gap <= 10.0:
                        current_run += 1
                    else:
                        burst_runs.append(current_run)
                        current_run = 1
                burst_runs.append(current_run)
            pooled["burst_run_size"] = numeric_stats(burst_runs)
            result[str(stage)] = {"pooled": pooled, "per_formal": per_formal}
        return result

    worker_summary = {}
    for stage in STAGES:
        final_events = [
            event
            for event in selected_dequeues
            if event["stage"] == stage and event.get("is_last") == "True"
        ]
        all_stage_events = [
            event for event in selected_dequeues if event["stage"] == stage
        ]
        mismatches = [
            event
            for event in all_stage_events
            if int(event["worker"]) != mix_room_id(event["room"]) % 4
        ]
        worker_summary[str(stage)] = {
            "all_chunk_dequeues": len(all_stage_events),
            "all_chunk_worker_counts": dict(
                sorted(Counter(int(event["worker"]) for event in all_stage_events).items())
            ),
            "final_dequeues": len(final_events),
            "final_worker_counts": dict(
                sorted(Counter(int(event["worker"]) for event in final_events).items())
            ),
            "observed_workers": sorted(
                {int(event["worker"]) for event in all_stage_events}
            ),
            "splitmix_mismatch_count": len(mismatches),
            "splitmix_mismatch_examples": [
                {
                    "room": event["room"],
                    "worker": int(event["worker"]),
                    "expected": mix_room_id(event["room"]) % 4,
                    "source_log": event["source_log"],
                    "source_line": event["source_line"],
                }
                for event in mismatches[:10]
            ],
            "final_queue_wait_by_worker": {
                str(worker): stats(
                    [
                        row[f"stage{stage}_c_queue_wait_ms"]
                        for row in rows
                        if row[f"stage{stage}_worker"] == worker
                    ]
                )
                for worker in range(4)
            },
        }

    return {
        "sender_enqueue_selected_events": len(selected_enqueues),
        "sender_enqueue_final": arrival_group(
            [event for event in selected_enqueues if event.get("is_last") == "True"]
        ),
        "sender_enqueue_all_chunks": arrival_group(selected_enqueues),
        "worker_shards": worker_summary,
    }


def analyze(wrapper: Path, prefill: Path):
    windows = parse_windows(wrapper)
    rooms, event_counts, engine_returns, all_enqueues, all_dequeues = parse_trace(prefill)
    rows = []
    incomplete = Counter()
    incomplete_examples = defaultdict(list)

    for room, stage_map in rooms.items():
        p0 = first(stage_map[0], "prefill_done")
        if p0 is None:
            continue
        formal = formal_for(p0["wall_ns"], windows)
        if formal is None:
            continue
        selected = {}
        missing = []
        for stage in STAGES:
            events = stage_map[stage]
            selected[stage] = {
                "P": first(events, "prefill_done"),
                "E": last(events, "sender_enqueue"),
                "D": first(events, "worker_dequeue"),
                "T": first(events, "transfer_start"),
                "C": last(events, "transfer_success"),
            }
            for key, value in selected[stage].items():
                if value is None:
                    missing.append(f"stage{stage}_{key}")
        if missing:
            for key in missing:
                incomplete[key] += 1
                if len(incomplete_examples[key]) < 10:
                    incomplete_examples[key].append(room)
            continue

        p0_mono = selected[0]["P"]["mono_ns"]
        stage_values = {}
        for stage in STAGES:
            event = selected[stage]
            p, e, d, t, c = (
                event[key]["mono_ns"] for key in ("P", "E", "D", "T", "C")
            )
            stage_values[stage] = {
                "a_pipeline": (p - p0_mono) / 1e6,
                "b_done_to_enqueue": (e - p) / 1e6,
                "c_enqueue_to_start": (t - e) / 1e6,
                "d_start_to_success": (c - t) / 1e6,
                "c_queue_wait": (d - e) / 1e6,
                "c_worker_to_start": (t - d) / 1e6,
            }

        early = min(STAGES, key=lambda stage: selected[stage]["T"]["mono_ns"])
        late = max(STAGES, key=lambda stage: selected[stage]["T"]["mono_ns"])
        slowest_success = max(STAGES, key=lambda stage: selected[stage]["C"]["mono_ns"])
        start_skew = (
            selected[late]["T"]["mono_ns"] - selected[early]["T"]["mono_ns"]
        ) / 1e6
        contribution = {
            segment: stage_values[late][segment] - stage_values[early][segment]
            for segment in ("a_pipeline", "b_done_to_enqueue", "c_enqueue_to_start")
        }
        closure_error = sum(contribution.values()) - start_skew
        row = {
            "formal": formal,
            "room": room,
            "rid": selected[0]["P"].get("rid", ""),
            "source_log": str(prefill),
            "early_start_stage": early,
            "late_start_stage": late,
            "slowest_success_stage": slowest_success,
            "start_skew_ms": start_skew,
            "completion_skew_ms": (
                max(selected[s]["C"]["mono_ns"] for s in STAGES)
                - min(selected[s]["C"]["mono_ns"] for s in STAGES)
            )
            / 1e6,
            "critical_ms": (
                max(selected[s]["C"]["mono_ns"] for s in STAGES)
                - min(selected[s]["T"]["mono_ns"] for s in STAGES)
            )
            / 1e6,
            "a_edge_contribution_ms": contribution["a_pipeline"],
            "b_edge_contribution_ms": contribution["b_done_to_enqueue"],
            "c_edge_contribution_ms": contribution["c_enqueue_to_start"],
            "closure_error_ms": closure_error,
        }
        for segment in contribution:
            row[f"{segment}_share"] = (
                contribution[segment] / start_skew if start_skew else 0.0
            )
        for stage in STAGES:
            for segment in ALL_SEGMENTS:
                row[f"stage{stage}_{segment}_ms"] = stage_values[stage][segment]
            for event_name in ("P", "E", "D", "T", "C"):
                event = selected[stage][event_name]
                row[f"stage{stage}_{event_name}_line"] = event["source_line"]
                row[f"stage{stage}_{event_name}_log"] = event["source_log"]
            row[f"stage{stage}_worker"] = int(selected[stage]["D"]["worker"])
        rows.append(row)

    pooled_metrics = summarize_rows(rows)
    summary = {
        "wrapper": str(wrapper),
        "prefill": str(prefill),
        "windows_wall_ns": windows,
        "trace_event_counts": dict(sorted(event_counts.items())),
        "complete_rooms": len(rows),
        "formal_counts": dict(sorted(Counter(row["formal"] for row in rows).items())),
        "incomplete": dict(sorted(incomplete.items())),
        "incomplete_examples": dict(sorted(incomplete_examples.items())),
        **pooled_metrics,
        "per_formal": {
            label: summarize_rows(
                [row for row in rows if row["formal"] == label]
            )
            for label in sorted(windows)
        },
        "engine_calls": summarize_engine_calls(engine_returns, windows),
        "arrivals_and_workers": summarize_arrivals_and_workers(
            rows, all_enqueues, all_dequeues
        ),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--prefill", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = analyze(args.wrapper, args.prefill)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "pp4-per-room.csv"
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
