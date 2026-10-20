"""Validate the q16 split-KV verify producer's causal mask across every request-length residue."""

import argparse
import json

import torch

from sglang.kernels.ops.attention.skv2_q16 import Q16SplitPlan

QLEN, HEADS, D, PAGE, TILE = 7, 16, 256, 64, 128


def mask_from(length, begin_tok, nblk):
    """Python mirror of the kernel's first masked block of a split (Q16SplitDecode.kernel)."""
    first = nblk - 1
    hidden = length - QLEN + 1 - begin_tok
    if hidden < (nblk - 1) * TILE:
        first = max(hidden, 0) // TILE
    return first


def brute_force(lengths, split_counts):
    """Every block that holds a token past the split end or one of the last QLEN - 1 tokens takes the masked body."""
    missed = []
    for length in lengths:
        for splits in split_counts:
            pps = -(-(-(-length // PAGE)) // splits)
            for split in range(splits):
                begin, end = split * pps * PAGE, min((split + 1) * pps * PAGE, length)
                if begin >= end:
                    continue
                nblk = -(-(end - begin) // TILE)
                first = mask_from(length, begin, nblk)
                for block in range(nblk):
                    last_tok = begin + (block + 1) * TILE - 1
                    if (
                        last_tok >= end or last_tok >= length - QLEN + 1
                    ) and block < first:
                        missed.append((length, splits, split, block))
    return missed


def reference(q, k, v, length, qk_scale):
    scores = q.double().transpose(0, 1) @ k.double().T * qk_scale
    ends = length - QLEN + torch.arange(QLEN, device=q.device)
    tokens = torch.arange(length, device=q.device)
    scores.masked_fill_(tokens[None, None, :] > ends[None, :, None], -float("inf"))
    return (scores.softmax(-1) @ v.double()).transpose(0, 1)


def run_case(length_sets, splits, partial_dtype, seed):
    """One plan, one launch per length set (one length per request). Each request's last QLEN - 1 tokens get keys
    aligned with its row-0 head-0 query, so a row that sees one of them is pulled far from the reference."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(length_sets[0])
    width = max(-(-n // PAGE) for lengths in length_sets for n in lengths)
    pool = width * batch * 2
    k0 = (torch.randn((pool, 1, PAGE, D), generator=gen, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    v = (torch.randn((pool, 1, PAGE, D), generator=gen, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    table = torch.randperm(pool, generator=gen, device="cuda")[: width * batch].view(
        batch, width
    )
    table = table.to(torch.int32).contiguous()
    q = (torch.randn((batch * QLEN, HEADS, D), generator=gen, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    lens = torch.tensor(length_sets[0], dtype=torch.int32, device="cuda")
    k = k0.clone()
    plan = Q16SplitPlan(
        q,
        (k, v),
        table,
        lens,
        splits=splits,
        merge_tile=128,
        merge_warps=4,
        partial_dtype=partial_dtype,
    )
    worst = (0.0, None)
    for lengths in length_sets:
        k.copy_(k0)
        lens.copy_(torch.tensor(lengths, dtype=torch.int32))
        for b, length in enumerate(lengths):
            key = (q[b * QLEN, 0].float() * 16).clamp(-448, 448).to(torch.float8_e4m3fn)
            for tok in range(length - QLEN + 1, length):
                k[table[b, tok // PAGE], 0, tok % PAGE] = key
        out = plan.run(
            q,
            (k, v),
            table,
            lens,
            qk_scale=1 / 16,
            v_scale=1.0,
            out=torch.empty(q.shape, dtype=torch.bfloat16, device="cuda"),
        )
        for b, length in enumerate(lengths):
            pages = table[b, : -(-length // PAGE)].long()
            kb = k[pages].reshape(-1, D)[:length]
            vb = v[pages].reshape(-1, D)[:length]
            ref = reference(q[b * QLEN : (b + 1) * QLEN], kb, vb, length, 1 / 16)
            err = float((out[b * QLEN : (b + 1) * QLEN].double() - ref).abs().max())
            worst = max(worst, (err, lengths))
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--base",
        type=int,
        default=2048,
        help="sweep lengths base .. base + 135 (every residue)",
    )
    # A row that sees an aligned hidden key moves by O(1) (1.4-1.7 before the fix); FP8 P rounding stays below 0.01.
    parser.add_argument("--tol", type=float, default=0.05)
    args = parser.parse_args()
    sweep = list(range(args.base, args.base + TILE + 8))
    missed = brute_force(
        list(range(QLEN, 4 * TILE + QLEN)) + sweep + [65537, 131071, 150021],
        (1, 2, 3, 49, 74, 148),
    )
    records = [dict(check="brute_force", missed=len(missed), first=missed[:5])]
    # Two splits per request, so each split spans many blocks and every length residue mod 128 shows up; then the
    # served split counts at lengths whose last block holds 5 tokens (B1 150021, B2 150021 + 131077).
    for name, partial_dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        cases = (
            ("sweep", [(n,) for n in sweep], 2),
            ("b1_150021", [(150021,)], 148),
            ("b2_150021_131077", [(150021, 131077)], 74),
        )
        for case, sets, splits in cases:
            err, at = run_case(sets, splits, partial_dtype, seed=20261020)
            records.append(
                dict(check=f"{case}_{name}", splits=splits, max_abs=err, at=at)
            )
    ok = not missed and all(r["max_abs"] <= args.tol for r in records[1:])
    records.append(dict(ok=ok, tol=args.tol, device=torch.cuda.get_device_name()))
    with open(args.output, "w") as f:
        json.dump(records, f, indent=1)
    print(json.dumps(records))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
