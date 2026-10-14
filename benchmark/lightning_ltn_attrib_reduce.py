"""CPU reduction for docs/209 job: per-arm B1 prefill wall and per-stage cost vs prompt length.

Plain wall = client /generate with max_new_tokens=1 (no probe trigger), median
of 3 per length. Stage numbers come from the i4 probe pass and are reported as
exclusive host time and exclusive CUDA-event span; the two overlap and are
never added. Fits are ordinary least squares over the six lengths.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

ARMS = ("original", "served", "transient", "devgap8")


def fit(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return dict(intercept_ms=my - slope * mx, us_per_token=slope * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    hashes, result = {}, dict(arms={})

    def read(path):
        raw = path.read_bytes()
        hashes[str(path.relative_to(args.root))] = hashlib.sha256(raw).hexdigest()
        return raw.decode()

    for arm in ARMS:
        root = args.root / arm
        timing = json.loads(read(root / "timing.json"))
        assert timing["complete"], arm
        plain = {}
        for row in timing["rows"]:
            if row["mode"] == "plain":
                plain.setdefault(row["length"], []).append(row["wall_ms"])
        lengths = sorted(plain)
        wall = {n: statistics.median(plain[n]) for n in lengths}
        stages = {}
        for line in read(root / "probe" / "i4-stages.jsonl").splitlines():
            row = json.loads(line)
            n = row["input_tokens"]
            for name, s in row["stages"].items():
                stages.setdefault(name, {}).setdefault(n, []).append(s)
            stages.setdefault("_completed_wall", {}).setdefault(n, []).append(
                dict(host_exclusive_ms=row["completed_wall_ms"], cuda_span_exclusive_ms=row["device_span_ms"]))
        stage_table = {}
        for name, by_len in stages.items():
            ns = sorted(by_len)
            host = {n: statistics.median(s["host_exclusive_ms"] for s in by_len[n]) for n in ns}
            cuda = {n: statistics.median(s["cuda_span_exclusive_ms"] for s in by_len[n]) for n in ns}
            stage_table[name] = dict(
                host_ms=host, cuda_ms=cuda,
                host_fit=fit(ns, [host[n] for n in ns]), cuda_fit=fit(ns, [cuda[n] for n in ns]))
        result["arms"][arm] = dict(
            plain_wall_ms=wall, plain_fit=fit(lengths, [wall[n] for n in lengths]),
            stages=stage_table,
            config_diff=json.loads(read(root / "config-diff.json")),
            load_seconds=json.loads(read(root / "load.json"))["seconds"])
    equality = {arm: json.loads(read(args.root / arm / "equality.json")) for arm in ARMS[1:]}
    result["greedy_equal_to_served"] = {
        arm: {n: equality[arm][n]["output_ids"] == equality["served"][n]["output_ids"]
                 and equality[arm][n]["text"] == equality["served"][n]["text"] for n in equality["served"]}
        for arm in ARMS[2:]}
    result["inputs_sha256"] = hashes
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for arm, r in result["arms"].items():
        f = r["plain_fit"]
        print(f"{arm:10s} plain wall = {f['intercept_ms']:8.1f} ms + {f['us_per_token']:6.2f} us/token  "
              + " ".join(f"L{n}:{v:.0f}" for n, v in r["plain_wall_ms"].items()))
        for name, s in sorted(r["stages"].items()):
            print(f"   {name:18s} host {s['host_fit']['intercept_ms']:8.2f} ms + {s['host_fit']['us_per_token']:6.2f} us/tok | "
                  f"cuda {s['cuda_fit']['intercept_ms']:8.2f} ms + {s['cuda_fit']['us_per_token']:6.2f} us/tok")
    print("greedy equal to served:", result["greedy_equal_to_served"])


if __name__ == "__main__":
    main()
