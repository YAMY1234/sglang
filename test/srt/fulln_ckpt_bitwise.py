"""docs/223 §11 gate (b): the fixed full-N radix checkpoint is bitwise the legacy route's checkpoint (1 GPU).

Per shape (prefix length, prompt length P): the served Triton GDN chunk kernel (chunk_gated_delta_rule, as
TritonGDNKernel.extend calls it) runs
  full-N : the whole extend (P - prefix tokens) from an initial state -> S_N in place + h;
  legacy : the N-1 route's extend (P - prefix - 1 tokens) from the same initial state -> S_{N-1} in place + h.
Which state each route donates comes from the real ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend and
MambaAttnBackendBase._init_track_ssm_indices (test_gdn_fulln_checkpoint helpers): last_recurrent_state for an aligned
true end, else h[index]. Pass = for every shape the two donated depths agree and the states are torch.equal; the
in-flight S_N is bitwise unchanged by the fix (the kernel inputs do not depend on tracking: two runs compared);
negative control = donating S_N at the fixed depth must differ from the legacy checkpoint on every shape.
  PYTHONPATH=python:test/srt python3 test/srt/fulln_ckpt_bitwise.py --out ckpt-bitwise.json
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
# P % 64 in {0, 1, 33, 63} x extend within one chunk-ish / many chunks x prefix 0 / 64k  (16 tracked shapes)
SHAPES = [(prefix, prefix + ext + (r - (prefix + ext) % CHUNK) % CHUNK)
          for prefix in (0, 65536) for ext in (130, 1500) for r in (0, 1, 33, 63)]
UNTRACKED = [(0, 40), (65536, 65536 + 64)]  # extent < grid: no checkpoint either way


def run(prompt_len, prefix, s0, gen, ext_tokens):
    T = ext_tokens
    q = torch.randn(1, T, H, K, generator=gen, device="cuda").to(torch.bfloat16)
    k = torch.randn(1, T, H, K, generator=gen, device="cuda").to(torch.bfloat16)
    v = torch.randn(1, T, H, V, generator=gen, device="cuda").to(torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(1, T, H, generator=gen, device="cuda")).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(1, T, H, generator=gen, device="cuda")).to(torch.bfloat16)
    return q, k, v, g, beta


def extend(inputs, s0, n):
    q, k, v, g, beta = (x[:, :n].contiguous() for x in inputs)
    pool = s0.clone()
    idx = torch.zeros(1, dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, n], dtype=torch.long, device="cuda")
    _, _, h = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=pool, initial_state_indices=idx,
                                     cu_seqlens=cu, head_first=False, use_qk_l2norm_in_kernel=True)
    torch.cuda.synchronize()
    return pool[0].clone(), h.squeeze(0).clone()


def donated(fulln, prompt_len, prefix, final_state, h, forward_len):
    os.environ["SGLANG_GDN_FULLN_CKPT_N_MINUS_1"] = "1"
    entry, depth = tc._track(list(range(prompt_len)), prefix, None, fulln)
    if not entry.track_mask:
        return None, None, None
    src, pos = tc._backend_source(entry, prefix, forward_len)
    assert pos == depth, (src, pos, depth)
    if src == "final":
        return depth, src, final_state
    return depth, src, h[(depth - prefix) // CHUNK]


def bits(t):
    return t.contiguous().view(torch.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=223)
    a = ap.parse_args()
    tc._setup()
    rows, ok = [], True
    for i, (prefix, P) in enumerate(SHAPES + UNTRACKED):
        gen = torch.Generator(device="cuda").manual_seed(a.seed * 1000 + i)
        s0 = (torch.randn(1, H, V, K, generator=gen, device="cuda") * 0.1).float()
        ext = P - prefix
        inputs = run(P, prefix, s0, gen, ext)
        sN, hN = extend(inputs, s0, ext)            # full-N forward
        sN2, _ = extend(inputs, s0, ext)            # in-flight S_N, second run (tracking never enters the kernel)
        sL, hL = extend(inputs, s0, ext - 1)        # legacy N-1 forward
        dN, srcN, cN = donated(True, P, prefix, sN, hN, ext)
        dL, srcL, cL = donated(False, P, prefix, sL, hL, ext - 1)
        row = dict(prefix=prefix, P=P, P_mod_64=P % CHUNK, ext=ext, depth_fulln=dN, depth_legacy=dL,
                   src_fulln=srcN, src_legacy=srcL, s_n_bitwise=torch.equal(bits(sN), bits(sN2)))
        if dN is None or dL is None:
            row["tracked"] = False
            row["pass"] = dN is None and dL is None and row["s_n_bitwise"]
        else:
            row["tracked"] = True
            row["checkpoint_bitwise"] = dN == dL and torch.equal(bits(cN), bits(cL))
            row["max_abs"] = float((cN - cL).abs().max()) if dN == dL else None
            row["negative_detected"] = not torch.equal(bits(sN), bits(cL))
            row["pass"] = row["checkpoint_bitwise"] and row["s_n_bitwise"] and row["negative_detected"]
        ok &= row["pass"]
        rows.append(row)
        print(json.dumps(row), flush=True)
    res = dict(pass_=ok, n=len(rows), tracked=sum(r["tracked"] for r in rows),
               device=torch.cuda.get_device_name(), torch=torch.__version__, rows=rows)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"{'PASS' if ok else 'FAIL'}: {sum(r['pass'] for r in rows)}/{len(rows)} shapes ({res['tracked']} tracked)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
