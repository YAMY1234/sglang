"""FN HC4 input partials; communication and its true completion remain explicit."""
import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _module():
    return load_jit("q38_mk_mtp_input",
                    cuda_files=["elementwise/q38_mk_mtp_input.cuh"],
                    cuda_wrappers=[("run", "Q38MkMtpInput::run")])


def input_partials(embedding, hidden, gamma_e, gamma_h, weight_e, weight_h,
                   eps_e, eps_h, rank=0, world=4):
    partials = torch.empty((5, 2560), device=embedding.device, dtype=torch.float32)
    _module().run(embedding, hidden, gamma_e, gamma_h, weight_e, weight_h,
                  partials, eps_e, eps_h, rank, world)
    return partials


@triton.jit
def _q38_mk_mtp_input_finish(Reduced, Out, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # FCs round independently, THEN embedding broadcasts into four HC streams.
    e = tl.load(Reduced + x, x < 2560, 0).to(tl.bfloat16).to(tl.float32)
    for b in tl.static_range(4):
        h = tl.load(Reduced + (b + 1) * 2560 + x, x < 2560, 0).to(tl.bfloat16).to(tl.float32)
        tl.store(Out + b * 2560 + x, (e + h).to(tl.bfloat16), x < 2560)


def finish_input(reduced):
    """Only call AFTER the original TP communicator completed FP32 reduction."""
    out = torch.empty((1, 10240), device=reduced.device, dtype=torch.bfloat16)
    _q38_mk_mtp_input_finish[(10,)](reduced, out, BLOCK=256, num_warps=4)
    return out
