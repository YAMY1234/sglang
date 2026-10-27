import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _jit_module():
    return load_jit(
        "q38c_mtp_gated_oproj",
        cuda_files=["elementwise/q38c_mtp_gated_oproj.cuh"],
        cuda_wrappers=[("run", "Q38cMtpGatedOproj::run")],
    )


def mtp_gated_oproj(x, gate, weight):
    # CUDA's fused QK norm/RoPE producer returns a flat gate. The native
    # producer returns a strided 3D view; both use the same kernel ABI.
    if gate.ndim == 2:
        gate = gate.view(1, 24, 256)
    output = torch.empty((1, 2560), dtype=x.dtype, device=x.device)
    _jit_module().run(x, gate, weight, output)
    return output
