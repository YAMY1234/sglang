import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _jit_module():
    return load_jit(
        "q38c_mtp_input_add_norm",
        cuda_files=["elementwise/q38c_mtp_input_add_norm.cuh"],
        cuda_wrappers=[("run", "Q38cMtpInputAddNorm::run")],
    )


def mtp_input_add_norm(embedding, hidden, weight, eps):
    hidden = hidden.view(embedding.shape[0], 10240)
    combined = torch.empty_like(hidden)
    normalized = torch.empty_like(hidden)
    _jit_module().run(embedding, hidden, weight, combined, normalized, eps)
    return combined, normalized
