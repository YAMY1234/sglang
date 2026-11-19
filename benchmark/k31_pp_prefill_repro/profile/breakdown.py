"""GPU-ms per 1k new tokens (summed over the 8 GPUs of an arm) by exclusive class, from classify_gpu.py outputs.

usage: breakdown.py --arm pp8 16384 RUN_PP8/prof/ana/gpu-P128-*.json --arm tep8 32768 RUN_TEP8/prof/ana/gpu-P128-*.json
Each --arm takes a name, the new tokens per step (the chunk size; every batch in a saturated window is a full chunk)
and the per-node JSONs (one per node, 4 GPUs each). Per GPU: class ms / steps (anchor kernel count); summed over GPUs;
divided by tokens per step / 1000.
"""
import json
import sys

CATS = [
    ("Compute, excluding MoE and fused collectives", ["compute"]),
    ("MoE (TEP8 includes EP dispatch/combine)", ["moe"]),
    ("Fused TP collectives (GEMM+RS; attn-residual+AG)", ["tp_fused_comm"]),
    ("NCCL-only (send/recv, all-gather, all-reduce, barriers)", ["nccl_p2p", "nccl_allgather", "nccl_allreduce", "barrier"]),
    ("GPU idle (incl. copy-only)", ["idle", "copy_only"]),
]


def parse(argv):
    arms, cur = {}, None
    i = 0
    while i < len(argv):
        if argv[i] == "--arm":
            cur = argv[i + 1]
            arms[cur] = {"tok": int(argv[i + 2]), "files": []}
            i += 3
        else:
            arms[cur]["files"].append(argv[i])
            i += 1
    return arms


def main():
    arms = parse(sys.argv[1:])
    table = {}
    for name, a in arms.items():
        devs = [d for f in a["files"] for d in json.load(open(f))["devices"].values()]
        if len(devs) != 8:
            print(f"warning: {name} has {len(devs)} GPUs in the given files (expected 8)", file=sys.stderr)
        row = {}
        for lab, members in CATS:
            per_step = sum(sum(d["exclusive_ms"].get(c, 0.0) for c in members) / d["steps"] for d in devs)
            row[lab] = per_step / (a["tok"] / 1000)
        row["Total"] = sum(row.values())
        steps = [d["steps"] for d in devs]
        print(f"{name}: {len(devs)} GPUs, steps per GPU {min(steps):.0f}-{max(steps):.0f}", file=sys.stderr)
        table[name] = row
    names = list(table)
    print(f"{'GPU-ms per 1k new tokens':58s}" + "".join(f"{n:>10s}" for n in names))
    for lab in [c[0] for c in CATS] + ["Total"]:
        print(f"{lab:58s}" + "".join(f"{table[n][lab]:10.1f}" for n in names))


if __name__ == "__main__":
    main()
