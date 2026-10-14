"""CPU-only audit of sealed M3 r2 windows, GSM forward events and MGS/N1."""

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    hashes = {}

    def read(relative, lines=False):
        raw = (args.root / relative).read_bytes()
        hashes[str(relative)] = hashlib.sha256(raw).hexdigest()
        return [json.loads(x) for x in raw.splitlines()] if lines else json.loads(raw)

    out = {"windows": {}, "gsm": {}, "n1": {}, "mgs": []}
    for c, job in ((8, 4476292), (32, 4476293)):
        arms = {}
        for arm in ("original-off", "factored"):
            row = read(f"m3-r2-C{c}-j{job}/{arm}/reduced-from-raw.json")
            assert row["complete_900s_cohort"]
            arms[arm] = {k: row[k] for k in (
                "tps_gpu", "ttft_p50_s", "hit", "user_p90", "formal_valid",
                "complete_900s_cohort", "reducer_sha256", "phase_ranges")}
            arms[arm]["completed"] = row["summary"]["completed_in_window"]
            arms[arm]["drain"] = row["latency"]["client_phase_including_drain"]
        a, b = arms["original-off"], arms["factored"]
        assert a["reducer_sha256"] == b["reducer_sha256"]
        ratios = {k: b[k] / a[k] for k in ("tps_gpu", "ttft_p50_s", "user_p90")}
        gates = dict(tps=ratios["tps_gpu"] >= .97,
                     ttft=ratios["ttft_p50_s"] <= 1.05, hit=b["hit"] >= a["hit"] - .01)
        out["windows"][c] = dict(arms=arms, ratios=ratios, numerical_gates=gates,
            preregistered_numerical_support=(ratios["user_p90"] >= .95 if c == 8
                                            else ratios["tps_gpu"] >= 1.),
            scope="reduced from raw; warmup protocol REVIEW is not waived")

    gsm_root = Path("m3-gsm-segments-j4476295")
    gsm_results = {}
    for arm in ("original", "iter3"):
        root = gsm_root / arm
        gsm = read(root / "gsm200.json")
        assert gsm["complete"] and len(gsm["results"]) == gsm["n_questions"] == 200
        gsm_results[arm] = gsm["results"]
        steps = read(root / "probe/steps.jsonl", lines=True)
        assert sorted(x["step"] for x in steps) == list(range(len(steps)))
        grouped = collections.defaultdict(list)
        for row in steps:
            grouped[(row["forward_kind"], row["can_run_graph"])].append(row)
        segments = {}
        for (kind, graph), rows in grouped.items():
            segments[f"{kind}/graph={graph}"] = dict(
                calls=len(rows), batch_members=sum(x["batch_size"] for x in rows),
                tokens=sum(x["input_tokens"] for x in rows),
                device_span_sum_ms=sum(x["device_span_ms"] for x in rows),
                device_span_mean_ms=statistics.mean(x["device_span_ms"] for x in rows),
                device_span_p50_ms=statistics.median(x["device_span_ms"] for x in rows),
                host_submit_sum_ms=sum(x["host_submit_ms"] for x in rows))
        pf = sorted(grouped[("prefill", False)], key=lambda x: x["step"])
        assert len(pf) == 25 and all(x["batch_size"] == 8 for x in pf)
        assert sum(x["batch_size"] for x in steps if x["forward_kind"] == "decode") == sum(
            x["gen_tokens"] - 1 for x in gsm["results"])
        out["gsm"][arm] = dict(seconds=gsm["seconds"], accuracy=gsm["acc"], segments=segments,
            first_prefill_ms=pf[0]["device_span_ms"],
            remaining_24_prefill_mean_ms=statistics.mean(x["device_span_ms"] for x in pf[1:]),
            scope="CUDA event spans, not kernel sums or per-request TTFT; host and device overlap")
    assert [(x["id"], x["gold"], x["prompt_hash"]) for x in gsm_results["original"]] == [
        (x["id"], x["gold"], x["prompt_hash"]) for x in gsm_results["iter3"]]

    base = read(gsm_root / "original/n1.json")
    assert base["complete"] and len(base["rows"]) == 9
    for arm in ("iter3", "iter2", "iter1"):
        candidate = read(gsm_root / arm / "n1.json")
        assert candidate["complete"] and len(candidate["rows"]) == 9
        first = tokens = requests = 0
        for a, b in zip(base["rows"], candidate["rows"]):
            assert (a["round"], a["B"]) == (b["round"], b["B"])
            assert len(a["responses"]) == len(b["responses"]) == a["B"]
            for x, y in zip(a["responses"], b["responses"]):
                xs, ys = x["output_ids"], y["output_ids"]
                assert len(xs) == len(ys) == 65
                requests += 1
                first += xs[0] == ys[0]
                tokens += sum(i == j for i, j in zip(xs, ys))
        assert requests == 123
        out["n1"][arm] = dict(requests=requests, first_matches=first,
                               greedy_matches=tokens, greedy_total=requests * 65)
    sweep = read("m3-r2-mgs-j4476294/mgs-sweep.json")
    assert {(r["iterations"], r["B"]) for r in sweep["rows"]} == {
        (i, b) for i in (1, 2, 3) for b in (1, 8, 32)}
    for row in sweep["rows"]:
        out["mgs"].append(dict(iterations=row["iterations"], B=row["B"], expiry_ms=row["expiry_ms"],
            numerical=[{k: v for k, v in n.items() if k not in ("rows", "cuts")}
                       for n in row["numerical"]]))
    out["input_sha256"] = hashes
    out["report_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
