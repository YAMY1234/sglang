"""docs/222 §15 PD gate: long teacher-forced windows (WS-4 P6 (b), docs/233 §4.5: the S vs B difference shows only at >= 49k
context, the standard 48 x (3,840 + 256) windows put the boundary at ~3.8k).  Same token source and offset as acc_audit_kl
prep (tok-qwen35 eval stream, offset 12M), but windows are drawn from disjoint slots of prefix + cont tokens, so no two
windows overlap.  Plain python (no numpy), runs on the CMH login node.  Output has acc_audit_kl's windows.json schema.
    python3 pdgate_windows.py --out windows-long.json [--tokens T] [--windows 16] [--prefix 48000] [--cont 256] [--seed 4848]
"""
import argparse
import array
import json
import os
import random

TOKENS = "/lustre/fsw/portfolios/coreai/users/yangminl/twinstar-20260911/data/tok-qwen35/eval/tokens.bin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default=TOKENS)
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--prefix", type=int, default=48000)
    ap.add_argument("--cont", type=int, default=256)
    ap.add_argument("--seed", type=int, default=4848)
    ap.add_argument("--offset", type=int, default=12_000_000)
    a = ap.parse_args()
    n = os.path.getsize(a.tokens) // 4  # uint32 tokens
    length = a.prefix + a.cont
    slots = (n - a.offset) // length
    assert slots >= a.windows, (n, slots)
    picks = sorted(random.Random(a.seed).sample(range(slots), a.windows))
    starts = [a.offset + s * length for s in picks]
    rows = []
    with open(a.tokens, "rb") as f:
        for s in starts:
            f.seek(4 * s)
            buf = array.array("I")
            buf.frombytes(f.read(4 * length))
            rows.append(buf.tolist())
    assert all(len(r) == length for r in rows)
    assert all(b - a_ >= length for a_, b in zip(starts, starts[1:])), "overlap"
    with open(a.out, "w") as f:
        json.dump({"seed": a.seed, "prefix": a.prefix, "cont": a.cont, "starts": starts, "offset": a.offset,
                   "slot": length, "tokens_file": a.tokens, "tokens_in_file": n, "windows": rows}, f)
    print(f"WINDOWS {len(rows)} x ({a.prefix} + {a.cont}) seed {a.seed}, {slots} slots of {length} after offset "
          f"{a.offset} in {n} tokens; starts {starts}")


if __name__ == "__main__":
    main()
