"""CPU-only reconstruction of isolated TTFT examples in C8 job 4484543.

Metrics have no request IDs. Associate only count-one scheduler time bins with
an isolated request and its unique token-count log; preserve this limitation.
No component timing or GPU kernel timing is inferred from the histogram.
"""
import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo


EXAMPLES = {
    "original-off": "8851402c-7e3d-44d8-84d5-0f9aa2ef15dc",
    "factored": "3144f5b7-c96b-4e07-84e5-0013d013c2c2",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    hashes, result = {}, {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return raw.decode()

    for arm, rid in EXAMPLES.items():
        root = args.root / arm
        artifacts = root / "C8/rep1/aiperf_artifacts"
        # Flattened copies are convenient for offline CPU-only audits.
        if not artifacts.exists():
            artifacts = root
        requests = [json.loads(line) for line in
                    read(artifacts / "profile_export.jsonl").splitlines()]
        exported = json.loads(read(artifacts / "server_metrics_export.json"))
        metrics = exported["metrics"]
        phase = exported["summary"]["phase_time_ranges"]["profiling"]
        interior_metrics = {}
        for name in ("sglang:queue_time_seconds", "sglang:per_stage_req_latency_seconds"):
            series, = [s for s in metrics[name]["series"]
                       if s["labels"].get("stage", "prefill_forward") == "prefill_forward"]
            bins = [t for t in series["timeslices"] if
                    phase["start_ns"] <= t["start_ns"] and t["end_ns"] <= phase["end_ns"]]
            count = sum(t["count"] for t in bins)
            singleton_ms = [t["sum"] * 1000 for t in bins if t["count"] == 1]
            interior_metrics[name] = dict(
                count=count, mean_ms=sum(t["sum"] for t in bins) * 1000 / count,
                singleton_count=len(singleton_ms),
                singleton_median_ms=statistics.median(singleton_ms),
                scope="Complete time bins inside profiling; boundary bins excluded. "
                "Queue and forward events have different cohorts; do not sum means or medians.")
        request, = [r for r in requests if r["metadata"]["x_request_id"] == rid]
        meta, values = request["metadata"], request["metrics"]
        assert meta["benchmark_phase"] == "profiling"
        assert values["input_sequence_length"]["value"] == 26006
        assert values["time_to_first_token"]["unit"] == "ms"
        start, ack = meta["request_start_ns"], meta["request_ack_ns"]

        def singleton_bin(name, event_ns, **labels):
            series, = [s for s in metrics[name]["series"]
                       if all(s["labels"].get(k) == v for k, v in labels.items())]
            row, = [t for t in series["timeslices"]
                    if t["start_ns"] <= event_ns < t["end_ns"]]
            assert row["count"] == 1
            # Guard against silently assigning a multi-request event to this ID.
            event_field = "request_start_ns" if "queue" in name else "request_ack_ns"
            overlaps = [r for r in requests if row["start_ns"] <=
                        r["metadata"][event_field] < row["end_ns"]]
            assert len(overlaps) == 1 and overlaps[0]["metadata"]["x_request_id"] == rid
            return {k: row[k] for k in ("start_ns", "end_ns", "count", "sum")}

        queue = singleton_bin("sglang:queue_time_seconds", start)
        forward = singleton_bin("sglang:per_stage_req_latency_seconds", ack,
                                stage="prefill_forward")
        logs = read(root / "server.log").splitlines()
        log_rows = []
        for lineno, line in enumerate(logs, 1):
            if "Prefill batch" not in line or "#new-token: 26006," not in line:
                continue
            stamp = datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=ZoneInfo("America/Los_Angeles")).timestamp() * 1e9
            if stamp <= ack < stamp + 1e9:
                assert "#new-seq: 1," in line and "#running-req: 0," in line
                log_rows.append(dict(line_number=lineno, text=line))
        assert len(log_rows) == 1
        execution = json.loads(read(root / "execution.json"))
        handoffs = []
        if arm == "factored":
            for lineno, line in enumerate(logs, 1):
                marker = "Lightning DUET batched extend: "
                if marker not in line:
                    continue
                handoff = ast.literal_eval(line.split(marker, 1)[1])
                assert handoff["prefill_commit_mode"] == "device"
                assert handoff["shallow_request_local"] is False
                assert handoff["commit_groups"] == 1
                handoffs.append(handoff)
            preceding = logs[log_rows[0]["line_number"] - 2]
            assert "'tokens': 26005," in preceding
            log_rows.insert(0, dict(line_number=log_rows[0]["line_number"] - 1,
                                   text=preceding))
            assert execution["env"]["SGLANG_LTN_PREFILL_COMMIT"] == "device"
            assert execution["env"]["SGLANG_LTN_NATIVE_BATCHED_SHALLOW"] == "1"
        ttft = values["time_to_first_token"]["value"]
        queue_ms, forward_ms = queue["sum"] * 1000, forward["sum"] * 1000
        residual = ttft - queue_ms - forward_ms
        assert residual >= 0
        result[arm] = dict(
            request_id=rid, trace=meta["source_trace_id"], outer=meta["source_outer_idx"],
            turn=meta["turn_index"], input_tokens=26006, request_start_ns=start,
            first_ack_ns=ack, ttft_ms=ttft, scheduler_queue_ms=queue_ms,
            prefill_forward_ms=forward_ms, outside_queue_and_forward_ms=residual,
            queue_bin=queue, forward_bin=forward, supporting_log_lines=log_rows,
            source_sha=execution["source_sha"],
            routing_handoffs_including_warmup=len(handoffs),
            interior_metrics=interior_metrics,
        )
    stock, method = result["original-off"], result["factored"]
    assert (stock["trace"], stock["outer"], stock["turn"]) == (
        method["trace"], method["outer"], method["turn"])
    delta = {k: method[k] - stock[k] for k in (
        "ttft_ms", "scheduler_queue_ms", "prefill_forward_ms",
        "outside_queue_and_forward_ms")}
    report = dict(arms=result, method_minus_stock_ms=delta,
                  source_sha256=hashes,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="Isolated example associated via count-one time bins, client times, "
                  "trace identity and unique B1 token-count logs; not request-ID server timers. "
                  "Boundary codec, one batched tail step, emitter and commit are INSIDE "
                  "prefill_forward; their individual times were not recorded. Residual includes "
                  "tokenization, pre-scheduler delay and first-token transport, not just sampling. "
                  "These examples do not decompose or sum population medians.")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(arms=result, method_minus_stock_ms=delta), indent=2))


if __name__ == "__main__":
    main()
