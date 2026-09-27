"""Check exact prefill factor equivalence before attempting layer-pipelined commits.

This is an isolated primitive experiment. It does not change the runtime,
truncation method, persistent pool, or accepted-token semantics.
"""
import argparse
import json
import os
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--device", choices=("cpu", "cuda"), required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
os.environ["REPLAY_TEST_DEVICE"] = a.device
import torch
import test_factored_prefill_graph as tx


def difference(left, right):
    bits_left = left.contiguous().view(torch.uint8)
    bits_right = right.contiguous().view(torch.uint8)
    return dict(bitwise=bool(torch.equal(bits_left, bits_right)),
                different_elements=int((left != right).sum().item()),
                max_abs=float((left.float() - right.float()).abs().max().item()))


def case(batch, heads, key, zero_vbar, tf32, seed):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.manual_seed(seed)
    layers = 36
    cfg = tx.module.FactoredGDNConfig(dtype=torch.float16, strict_chunk=1, factored_prefix=1)
    dense = torch.randn(layers, batch, heads, key, key)
    vbar = torch.zeros(layers, heads, key) if zero_vbar else torch.randn(layers, heads, key) * .01
    # factorize_layers reproduces the seed-0 probe separately for every layer.
    full = tx.module.factorize_layers(tuple(dense.unbind(0)), vbar, cfg)
    results = []
    for group in (12, 18):
        partitioned = []
        for begin in range(0, layers, group):
            end = min(begin + group, layers)
            partitioned.extend(tx.module.factorize_layers(tuple(dense[begin:end].unbind(0)), vbar[begin:end], cfg))
        fields = {name: difference(torch.stack([x[i] for x in full]), torch.stack([x[i] for x in partitioned]))
                  for i, name in enumerate(("a", "U", "W"))}
        results.append(dict(group_layers=group, fields=fields,
                            passed=all(v["bitwise"] for v in fields.values())))
    return dict(layers=layers, batch=batch, heads=heads, key=key, zero_vbar=zero_vbar,
                allow_tf32=tf32, seed=seed, groups=results, passed=all(x["passed"] for x in results))


start = time.monotonic()
rows = []
for tf32 in ((False, True) if tx.GPU else (False,)):
    for batch in (1, 2):
        for zero_vbar in (True, False):
            for seed in (298, 298832):
                row = case(batch, 24 if tx.GPU else 2, 128 if tx.GPU else 16, zero_vbar, tf32, seed)
                rows.append(row)
                print(json.dumps(row), flush=True)
result = dict(complete=True, passed=all(r["passed"] for r in rows), device=a.device,
              scope="Primitive equivalence only; no performance or model admission claim",
              seconds=time.monotonic() - start, rows=rows)
a.out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(dict(complete=True, passed=result["passed"], seconds=result["seconds"])), flush=True)
raise SystemExit(0 if result["passed"] else 1)
