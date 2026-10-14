"""CPU-only reducer for I4 E2; every requested prompt/tag must be present."""

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    hashes = {}

    def read(path, lines=False):
        path = args.root / path
        raw = path.read_bytes()
        hashes[str(path.relative_to(args.root))] = hashlib.sha256(raw).hexdigest()
        return [json.loads(x) for x in raw.splitlines()] if lines else json.loads(raw)

    out = dict(arms={}, n1={}, errors={})
    requests = {}
    for arm in ("original", "current", "batched"):
        try:
            raw = read(f"{arm}/prefill.json")
            assert raw["complete"] and len(raw["rows"]) == 450
            events = read(f"{arm}/probe/steps.jsonl", lines=True)
            grouped = collections.defaultdict(list)
            for event in events:
                grouped[event["tag"]].append(event)
            buckets = collections.defaultdict(list)
            for request in raw["rows"]:
                tag = request["tag"]
                rows = grouped.pop(tag)
                assert rows and all(x["forward_kind"] == "prefill" for x in rows)
                assert sum(x["batch_size"] for x in rows) == request["B"]
                if arm == "batched":
                    assert all(x["handoff"].get("batched_extend") for x in rows), tag
                buckets[(request["B"], request["round"])].append(dict(
                    tag=tag, calls=len(rows), B=request["B"],
                    completed_wall_ms=sum(x["prefill_completed_wall_ms"] for x in rows),
                    device_span_ms=sum(x["device_span_ms"] for x in rows),
                    endpoint_wall_ms=request["endpoint_wall_ms"]))
            assert not grouped
            summary = {}
            for (batch, repeat), samples in buckets.items():
                assert len(samples) * batch == 200
                summary[f"B{batch}-r{repeat}"] = dict(
                    requests=200, calls=sum(x["calls"] for x in samples),
                    completed_wall_ms_per_request=sum(x["completed_wall_ms"] for x in samples)/200,
                    device_span_ms_per_request=sum(x["device_span_ms"] for x in samples)/200,
                    batch_wall_p50_ms=statistics.median(x["completed_wall_ms"] for x in samples),
                    samples=samples)
            assert len(summary) == 4
            out["arms"][arm] = summary
        except Exception as error:
            out["errors"][arm+"/prefill"] = repr(error)
        for kind in ("n1", "variable-n1"):
            try:
                requests[(arm, kind)] = read(f"{arm}/{kind}.json")
                assert requests[(arm, kind)]["complete"]
            except Exception as error:
                out["errors"][arm+"/"+kind] = repr(error)
    for kind in ("n1", "variable-n1"):
        for ref in ("current", "original"):
            try:
                a, b = (requests[(arm, kind)]["rows"] for arm in (ref, "batched"))
                assert len(a) == len(b) == (9 if kind == "n1" else 3)
                first = tokens = total = count = exact = 0
                for x, y in zip(a, b):
                    assert len(x["responses"]) == len(y["responses"])
                    if kind == "variable-n1":
                        assert x["input_ids"] == y["input_ids"]
                    else:
                        assert (x["round"], x["B"]) == (y["round"], y["B"])
                    for u, v in zip(x["responses"], y["responses"]):
                        ui, vi = u["output_ids"], v["output_ids"]
                        assert len(ui) == len(vi) == 65
                        first += ui[0] == vi[0]
                        exact += ui == vi
                        tokens += sum(i == j for i, j in zip(ui, vi))
                        total += 65
                        count += 1
                assert count == (123 if kind == "n1" else 24)
                out["n1"][f"{kind}/batched-vs-{ref}"] = dict(requests=count, first_matches=first,
                    exact_sequences=exact, greedy_matches=tokens, greedy_total=total,
                    passed=first == count and tokens == total)
            except Exception as error:
                out["errors"][f"{kind}/vs-{ref}"] = repr(error)
    try:
        numerical = read("extend-numerical.json")
        out["extend_numerical"] = numerical
        assert numerical["complete"]
        m1 = read("m1/mgs-sweep.json")
        assert m1["complete"] and len(m1["rows"]) == 3
        out["m1"] = [{"B": r["B"], "numerical": [{k: v for k, v in n.items()
            if k not in ("rows", "cuts")} for n in r["numerical"]]} for r in m1["rows"]]
    except Exception as error:
        out["errors"]["numerical"] = repr(error)
    if "batched" in out["arms"]:
        out["prefill_support"] = {
            f"B{b}": statistics.mean(out["arms"]["batched"][f"B{b}-r{r}"]["completed_wall_ms_per_request"]
                                    for r in (0, 1)) <= 50 for b in (1, 8)}
    out["scope"] = "Completed prefill wall; includes tail and logits, excludes generation decode. No prefill graph. N1 reference=current. M1 failures retained."
    out["input_sha256"] = hashes
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.root / "reduced.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("arms", "extend_numerical", "m1", "input_sha256")}, indent=2))


if __name__ == "__main__":
    main()
