# SPDX-License-Identifier: Apache-2.0
"""PDL scheduling for the native MTP ordered local routed finalize."""
import triton
import triton.language as tl


@triton.jit
def _q38_mk_mtp_finalize_pdl(ExpertRows, Weights, Out,
                           H: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    # Release the existing shared-gate static-weight prologue. Its own wait
    # still covers this whole kernel; current expert rows are read only below.
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.program_id(1)
    total = tl.full((BLOCK,), 0, tl.float32)
    for slot in tl.static_range(TOPK):
        value = tl.load(ExpertRows + (row * TOPK + slot) * H + col,
                        col < H, 0).to(tl.float32)
        weight = tl.load(Weights + row * TOPK + slot).to(tl.float32)
        total = total + value * weight
    tl.store(Out + row * H + col, total.to(tl.bfloat16), col < H)
