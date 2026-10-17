"""docs/223 §11 gate (b): the fixed full-N radix checkpoint against the legacy route and the pre-fix arm (1 GPU).

Per shape (prefix length, prompt length P, chunk end): the served Triton GDN chunk kernel (chunk_gated_delta_rule, as
TritonGDNKernel.extend calls it) runs
  full-N : the whole extend from an initial state -> S_N in place (fp32 pool) + h (bf16 chunk states);
  legacy : the N-1 route's extend (one token less on the final chunk) from the same initial state.
Which state each route donates comes from the real ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend and
MambaAttnBackendBase._init_track_ssm_indices (test_gdn_fulln_checkpoint helpers): last_recurrent_state for an aligned
true end, else h[index]. Gate (lead 21:43, restated):
  * P % 64 != 1: donated depths agree and the fixed checkpoint is torch.equal to the legacy one;
  * P % 64 == 1: legacy donates the fp32 final state, full-N the bf16 h state of the same depth (as the pre-fix arm
    did): every element |delta| <= bf16-ulp(|S|) / 2 of the legacy fp32 state S, no non-finite values;
  * in-flight S_N bitwise unchanged (tracking never enters the kernel: two runs compared);
  * negative control: donating S_N at the fixed depth differs from the legacy checkpoint;
  * reported, not gated alone: fixed vs pre-fix (SGLANG_GDN_FULLN_CKPT_N_MINUS_1=0) donation, equal except P % 64 == 0.
  PYTHONPATH=python:test/srt python3 test/srt/fulln_ckpt_bitwise.py --out ckpt-bitwise.json
--snapshot (A', docs/223 §11): the fixed full-N forward runs again with the fp32 track_state snapshot wired as
gdn_backend does (rows from MambaAttnBackendBase._init_fp32_track_rows, chunk index from _init_track_ssm_indices).
Gate (b'): o / S_N / h bitwise identical with and without the snapshot on every shape; every P % 64 == 1 checkpoint
taken from the snapshot is torch.equal to the legacy fp32 final state (so all residues are bitwise); negative control =
the bf16 h value at those rows differs from it.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_gdn_fulln_checkpoint as tc  # noqa: E402
from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule  # noqa: E402

H, K, V = 8, 128, 128
CHUNK = tc.CHUNK
# (prefix, prompt length, chunk end or None for the final chunk)
SHAPES = [(prefix, prefix + ext + (r - (prefix + ext) % CHUNK) % CHUNK, None)
          for prefix in (0, 65536) for ext in (130, 1500) for r in (0, 1, 33, 63)]
SHAPES += [(0, P, None) for P in (65, 128, 129, 4096, 4097)]  # the P3 line's length set
SHAPES += [(0, 4097, 128)]  # non-final 128-token chunk of a 4097-token prompt
UNTRACKED = [(0, 40, None), (65536, 65536 + 64, None)]  # extent < grid: no checkpoint either way


def inputs_for(gen, T):
    q = torch.randn(1, T, H, K, generator=gen, device="cuda").to(torch.bfloat16)
    k = torch.randn(1, T, H, K, generator=gen, device="cuda").to(torch.bfloat16)
    v = torch.randn(1, T, H, V, generator=gen, device="cuda").to(torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(1, T, H, generator=gen, device="cuda")).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(1, T, H, generator=gen, device="cuda")).to(torch.bfloat16)
    return q, k, v, g, beta


def extend(inputs, s0, n, track_chunk_idx=None, with_o=False):
    q, k, v, g, beta = (x[:, :n].contiguous() for x in inputs)
    pool = s0.clone()
    idx = torch.zeros(1, dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, n], dtype=torch.long, device="cuda")
    track = {}
    snap = None
    if track_chunk_idx is not None:
        snap = torch.full_like(s0, float("nan"))
        track = dict(track_state=snap, track_chunk_idx=track_chunk_idx)
    o, _, h = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=pool, initial_state_indices=idx,
                                     cu_seqlens=cu, head_first=False, use_qk_l2norm_in_kernel=True, **track)
    torch.cuda.synchronize()
    if with_o:
        return pool[0].clone(), h.squeeze(0).clone(), o.clone(), None if snap is None else snap[0].clone()
    return pool[0].clone(), h.squeeze(0).clone()


def snapshot_metadata(prompt_len, prefix, end, forward_len):
    """(track_chunk_idx on cuda, fp32 row mask or None) exactly as init_forward_metadata builds them."""
    from types import SimpleNamespace
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase

    os.environ["SGLANG_GDN_FULLN_CKPT_N_MINUS_1"] = "1"
    entry, _ = tc._track(list(range(prompt_len)), prefix, None, True, end=end)
    if not entry.track_mask:
        return None, None
    fb = SimpleNamespace(mamba_track_mask=torch.tensor([True]), extend_seq_lens=torch.tensor([forward_len]),
                         mamba_track_indices=torch.tensor([7]), mamba_track_seqlens=torch.tensor([entry.track_seqlen]),
                         extend_prefix_lens=torch.tensor([prefix]))
    me = SimpleNamespace(mamba_chunk_size=CHUNK, device="cuda")
    os.environ["SGLANG_GDN_FULLN_CKPT_FP32_SNAPSHOT"] = "1"
    rows, any_ = MambaAttnBackendBase._init_fp32_track_rows(me, fb)
    chunk_idx = MambaAttnBackendBase._init_track_ssm_indices(me, torch.tensor([3]), fb)[0]
    return chunk_idx, (rows if any_ else None)


def donated(fulln, prompt_len, prefix, end, final_state, h, forward_len, switch="1"):
    os.environ["SGLANG_GDN_FULLN_CKPT_N_MINUS_1"] = switch
    entry, depth = tc._track(list(range(prompt_len)), prefix, None, fulln, end=end)
    if not entry.track_mask:
        return None, None, None
    src, pos = tc._backend_source(entry, prefix, forward_len)
    assert pos == depth, (src, pos, depth)
    if src == "final":
        return depth, src, final_state
    return depth, src, h[(depth - prefix) // CHUNK]


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def bf16_half_ulp_ok(new, ref):
    """|new - ref| <= bf16-ulp(|ref|) / 2 elementwise (ref fp32), and both finite."""
    newf, ref = new.float(), ref.float()
    finite = bool(torch.isfinite(newf).all() and torch.isfinite(ref).all())
    a = ref.abs()
    tiny = torch.finfo(torch.bfloat16).tiny
    exp = torch.floor(torch.log2(torch.clamp(a, min=tiny)))
    ulp = torch.pow(2.0, exp - 7)  # bf16: 7 explicit fraction bits
    delta = (newf - ref).abs()
    ok = bool((delta <= ulp / 2).all())
    return finite, ok, float(delta.max()), bool(torch.equal(new.to(torch.bfloat16), ref.to(torch.bfloat16)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=223)
    ap.add_argument("--snapshot", action="store_true", help="gate (b'): the fp32 track_state snapshot (A')")
    a = ap.parse_args()
    tc._setup()
    rows, ok = [], True
    for i, (prefix, P, end) in enumerate(SHAPES + UNTRACKED):
        gen = torch.Generator(device="cuda").manual_seed(a.seed * 1000 + i)
        s0 = (torch.randn(1, H, V, K, generator=gen, device="cuda") * 0.1).float()
        stop = P if end is None else end
        ext = stop - prefix
        legacy_len = ext - (1 if end is None else 0)  # the N-1 route drops only the final chunk's last token
        inputs = inputs_for(gen, ext)
        sN, hN = extend(inputs, s0, ext)            # full-N forward
        sN2, _ = extend(inputs, s0, ext)            # in-flight S_N, second run
        sL, hL = extend(inputs, s0, legacy_len)     # legacy forward
        dN, srcN, cN = donated(True, P, prefix, end, sN, hN, ext)
        dO, srcO, cO = donated(True, P, prefix, end, sN, hN, ext, switch="0")
        dL, srcL, cL = donated(False, P, prefix, end, sL, hL, legacy_len)
        row = dict(prefix=prefix, P=P, end=end, P_mod_64=P % CHUNK, ext=ext, depth_fixed=dN, depth_prefix=dO,
                   depth_legacy=dL, src_fixed=srcN, src_prefix=srcO, src_legacy=srcL,
                   s_n_bitwise=torch.equal(sN, sN2))
        row["fixed_vs_prefix_equal"] = (dN == dO and (cN is None or same(cN, cO)))
        if dN is None or dL is None:
            row["tracked"] = False
            row["pass"] = dN is None and dL is None and row["s_n_bitwise"]
        else:
            row["tracked"] = True
            row["depth_equal"] = dN == dL
            # where the donation is the forward's true end (non-final chunk) S_N is the right state: no control
            row["negative_detected"] = (None if dN == stop else not same(sN.to(cL.dtype), cL))
            if end is None and P % CHUNK == 1:
                finite, bound, mx, rne = bf16_half_ulp_ok(cN, cL)
                row.update(rule="bf16_half_ulp", finite=finite, within_half_ulp=bound, max_abs=mx,
                           equals_bf16_of_legacy=rne)
                row["checkpoint_ok"] = row["depth_equal"] and finite and bound
            else:
                row.update(rule="torch.equal")
                row["checkpoint_ok"] = row["depth_equal"] and same(cN, cL)
            row["pass"] = row["checkpoint_ok"] and row["s_n_bitwise"] and row["negative_detected"] is not False
        if a.snapshot:
            plain = extend(inputs, s0, ext, with_o=True)
            chunk_idx, mask = snapshot_metadata(P, prefix, end, ext)
            if chunk_idx is None:
                row.update(snap_on_off_bitwise=True, snap_rule="untracked")
            else:
                snapped = extend(inputs, s0, ext, track_chunk_idx=chunk_idx, with_o=True)
                row["snap_on_off_bitwise"] = all(torch.equal(x, y) for x, y in zip(plain[:3], snapped[:3]))
                row["snap_rows"] = None if mask is None else mask.tolist()
                if mask is not None and bool(mask[0]):
                    snap = snapped[3]
                    row["snap_rule"] = "fp32 snapshot == legacy final"
                    row["snap_equal_legacy"] = same(snap, cL)
                    row["snap_negative_detected"] = not same(cN.float(), cL)  # the bf16 h (v9) is not it
                    row["snap_ok"] = (row["snap_on_off_bitwise"] and row["snap_equal_legacy"]
                                      and row["snap_negative_detected"])
                else:
                    row["snap_rule"] = "unchanged (v9 checkpoint)"
                    row["snap_ok"] = row["snap_on_off_bitwise"] and (end is not None or P % CHUNK != 1)
            row["snap_ok"] = row.get("snap_ok", row["snap_on_off_bitwise"])
            row["pass"] = row["pass"] and row["snap_ok"]
        ok &= row["pass"]
        rows.append(row)
        print(json.dumps(row), flush=True)
    res = dict(pass_=ok, n=len(rows), tracked=sum(r["tracked"] for r in rows),
               device=torch.cuda.get_device_name(), torch=torch.__version__, rows=rows)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"{'PASS' if ok else 'FAIL'}: {sum(r['pass'] for r in rows)}/{len(rows)} shapes ({res['tracked']} tracked); "
          f"fixed==pre-fix on {sum(r['fixed_vs_prefix_equal'] for r in rows)}/{len(rows)} "
          f"(expected unequal only where P % 64 == 0 on a final chunk)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
