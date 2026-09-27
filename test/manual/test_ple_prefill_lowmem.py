"""#886 plan A GPU check: the per-request PLE short conv (SGLANG_PLE_PREFILL_LOWMEM) is bitwise identical to the padded
path it replaces, and its peak memory. Real Flash-Next PLE dims: channels 10240 (hc 4 x 2560), kernel 4, dilation 3
(ngram size), state length 9, bf16.
usage: python test_ple_prefill_lowmem.py [--out result.json]
"""
import argparse
import json

import torch
import torch.nn.functional as F

from sglang.srt.models.qwen4_exp import _ple_short_conv_per_request

C, K, DIL = 10240, 4, 3
S = (K - 1) * DIL


def padded_reference(x, state, weight, lengths, track_offsets):
    """The original prefill path of Qwen4ExpPLELayer._short_conv (scatter into [n, W, C], concat, conv, gather)."""
    dev = x.device
    n, width = len(lengths), max(max(lengths), 1)
    req = torch.repeat_interleave(torch.arange(n, device=dev), torch.tensor(lengths, device=dev))
    off = torch.cat([torch.arange(l, device=dev) for l in lengths]) if sum(lengths) else torch.empty(0, dtype=torch.long, device=dev)
    padded = x.new_zeros((n, width, C))
    padded[req, off] = x
    conv_input = torch.cat([state, padded.transpose(1, 2)], dim=-1)
    conv_output = F.conv1d(conv_input, weight, bias=None, dilation=DIL, groups=C).transpose(1, 2)
    cols = torch.arange(S, device=dev, dtype=torch.long)

    def gather_at(offsets):
        return conv_input.gather(2, (offsets.unsqueeze(1) + cols.unsqueeze(0)).unsqueeze(1).expand(-1, C, -1))

    next_state = gather_at(torch.tensor(lengths, device=dev))
    track = gather_at(torch.tensor(track_offsets, device=dev)) if track_offsets is not None else None
    return conv_output[req, off], next_state, track


def case(lengths, seed, measure):
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = sum(lengths)
    x = torch.randn(T, C, device="cuda", generator=g).to(torch.bfloat16)
    state = torch.randn(len(lengths), C, S, device="cuda", generator=g).to(torch.bfloat16)
    weight = (torch.randn(C, 1, K, device="cuda", generator=g) * 0.3).to(torch.bfloat16)
    track_offsets = [(l // 256) * 256 if l >= 256 else 0 for l in lengths]
    result = {}
    for name, fn in (("padded", lambda: padded_reference(x, state, weight, lengths, track_offsets)),
                     ("per_request", lambda: _ple_short_conv_per_request(x, state, weight, DIL, C, lengths, S, track_offsets))):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = fn()
        torch.cuda.synchronize()
        result[name] = (out, torch.cuda.max_memory_allocated() - base)
    (ra, na, ta), pa = result["padded"]
    (rb, nb, tb), pb = result["per_request"]
    return dict(lengths=lengths if len(lengths) <= 8 else f"{len(lengths)} x {lengths[0]}",
                rows_bitwise=bool(torch.equal(ra, rb)), next_state_bitwise=bool(torch.equal(na, nb)),
                track_bitwise=bool(torch.equal(ta, tb)),
                peak_padded_gib=round(pa / 2**30, 3), peak_per_request_gib=round(pb / 2**30, 3))


def a0_checks():
    g = torch.Generator(device="cuda").manual_seed(7)
    a = torch.randn(4096, C, device="cuda", generator=g).to(torch.bfloat16)
    b = torch.randn(4096, C, device="cuda", generator=g).to(torch.bfloat16)
    valid = torch.rand(4096, device="cuda", generator=g) > 0.2
    ref = torch.where(valid.unsqueeze(-1), a + b, torch.zeros_like(a))
    alt = a.clone().add_(b)
    alt.masked_fill_(~valid.unsqueeze(-1), 0)
    x = torch.randn(2048, C, device="cuda", generator=g).to(torch.bfloat16)
    return dict(add_masked_fill_bitwise=bool(torch.equal(ref, alt)),
                silu_inplace_bitwise=bool(torch.equal(F.silu(x), F.silu(x.clone(), inplace=True))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()
    cases = [[32768], [32768, 100, 257, 1024, 3], [16384, 0, 2048, 5000], [256] * 40, [1, 2, 3, 9, 10], [32536]]
    rows = [case(l, i, True) for i, l in enumerate(cases)]
    result = dict(cases=rows, a0=a0_checks())
    result["passed"] = all(r["rows_bitwise"] and r["next_state_bitwise"] and r["track_bitwise"] for r in rows) and all(result["a0"].values())
    print(json.dumps(result, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    assert result["passed"], "per-request short conv or A0 trims differ from the padded path"


if __name__ == "__main__":
    main()
