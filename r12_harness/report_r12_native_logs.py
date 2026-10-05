#!/usr/bin/env python3
"""Build line-addressable PP/TEP cadence evidence from native scheduler logs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from analyze_r12 import BATCH_RE, field, parse_time, parse_windows, percentile


def read_batches(paths: list[Path], timezone: str) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                match = BATCH_RE.search(line)
                if not match:
                    continue
                body = match["body"]
                rows.append(
                    {
                        "stamp": parse_time(match["ts"], timezone),
                        "pp": int(match["pp"]) if match["pp"] is not None else None,
                        "tp": int(match["tp"]),
                        "tokens": field(body, "#new-token", int),
                        "running": field(body, "#running-req", int),
                        "queue": field(body, "#queue-req", int),
                        "inflight": field(body, "#inflight-req", int),
                        "pending_tokens": field(body, "#pending-token", int),
                        "bootstrap": field(body, "#bootstrap-req", int),
                        "transferring": field(body, "#transferring-req", int),
                        "usage": field(body, "full token usage", float),
                        "mamba_usage": field(body, "mamba usage", float),
                        "rate": field(body, "input throughput (token/s)", float),
                        "source": f"{path}:{line_number}",
                    }
                )
    return rows


def in_window(rows: list[dict], begin, end) -> list[dict]:
    return sorted(
        (row for row in rows if begin <= row["stamp"] <= end),
        key=lambda row: row["stamp"],
    )


def summarize(rows: list[dict], duration_s: float) -> dict:
    intervals = [
        (right["stamp"] - left["stamp"]).total_seconds() * 1000
        for left, right in zip(rows, rows[1:])
    ]

    def distribution(name: str):
        values = [row[name] for row in rows if row[name] is not None]
        return {
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "max": max(values) if values else None,
        }

    return {
        "count": len(rows),
        "window_s": duration_s,
        "interval_p50_ms": percentile(intervals, 0.50),
        "interval_p90_ms": percentile(intervals, 0.90),
        "running_req": distribution("running"),
        "queue_req": distribution("queue"),
        "inflight_req": distribution("inflight"),
        "pending_token": distribution("pending_tokens"),
        "bootstrap_req": distribution("bootstrap"),
        "transferring_req": distribution("transferring"),
        "full_usage": distribution("usage"),
        "mamba_usage": distribution("mamba_usage"),
        "first_source": rows[0]["source"] if rows else None,
        "last_source": rows[-1]["source"] if rows else None,
    }


def point_cadence(
    log_dir: Path, service: str, topology: str, concurrency: int, timezone: str
) -> dict:
    windows = parse_windows(log_dir / f"{service}-C{concurrency}-windows.out")
    rows = read_batches(
        sorted(log_dir.glob(f"{service}-prefill-rank-*.out")), timezone
    )
    entity_key = "pp" if topology == "pp" else "tp"
    entities = range(4) if topology == "pp" else range(8)
    result = {}
    for formal, (begin, end) in windows.items():
        duration_s = (end - begin).total_seconds()
        result[formal] = {
            str(entity): summarize(
                in_window(
                    [row for row in rows if row[entity_key] == entity], begin, end
                ),
                duration_s,
            )
            for entity in entities
        }
    return result


def add_busy_proxy(pp: dict, tep: dict) -> None:
    for formal, stages in pp.items():
        tep_reference = tep[formal]["0"]["interval_p50_ms"]
        for stage in stages.values():
            interval = stage["interval_p50_ms"]
            if tep_reference is None or interval in (None, 0):
                busy = None
            else:
                busy = min(
                    1.0,
                    stage["count"] * tep_reference / 1000.0 / stage["window_s"],
                )
            stage["tep_chunk_reference_ms"] = tep_reference
            stage["busy_proxy"] = busy
            stage["idle_proxy"] = None if busy is None else 1.0 - busy


def matching_evidence(log_dir: Path, service: str) -> dict:
    needles = {
        "kv_wait_or_transfer": ("waiting for kv", "kv transfer"),
        "release_or_consensus": ("release", "consensus"),
        "abort_req": ("abortreq",),
        "bootstrap_timing": ("bootstrap", " ms"),
    }
    evidence = defaultdict(list)
    for role in ("prefill", "decode"):
        for path in sorted(log_dir.glob(f"{service}-{role}-rank-*.out")):
            with path.open(errors="replace") as stream:
                for line_number, line in enumerate(stream, 1):
                    lowered = line.lower()
                    if "server_args=" in lowered:
                        continue
                    for label, terms in needles.items():
                        if all(term in lowered for term in terms):
                            evidence[f"{role}:{label}"].append(
                                {"source": f"{path}:{line_number}", "text": line.strip()}
                            )
    return dict(evidence)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log-timezone", default="America/Los_Angeles")
    args = parser.parse_args()

    out = {"points": {}, "transfer_release_evidence": {}}
    for concurrency in (16, 32, 64):
        b = point_cadence(
            args.log_dir, "B-main", "tep", concurrency, args.server_log_timezone
        )
        a = point_cadence(
            args.log_dir, "A-main", "pp", concurrency, args.server_log_timezone
        )
        add_busy_proxy(a, b)
        out["points"][f"B-C{concurrency}"] = b
        out["points"][f"A-C{concurrency}"] = a
    out["transfer_release_evidence"]["B-main"] = matching_evidence(
        args.log_dir, "B-main"
    )
    out["transfer_release_evidence"]["A-main"] = matching_evidence(
        args.log_dir, "A-main"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("NATIVE_LOG_REPORT=PASS", f"output={args.output}")


if __name__ == "__main__":
    main()
