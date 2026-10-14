from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _compact_module(dtype: torch.dtype, top_k: int, device_type: str) -> Module:
    if device_type != "cuda" or dtype != torch.float32:
        raise RuntimeError("Compact support requires CUDA float32 tensors")
    if not 1 <= top_k <= 32:
        raise RuntimeError("Compact support requires top_k in [1, 32]")
    # 4096 columns gives 427 scan CTAs for the actual 7x248320 target on 152
    # GB300 SMs. This is an initial geometry to qualify, not a measured optimum.
    args = make_cpp_args(top_k, 256, 4096)
    return load_jit(
        "compact_support",
        *args,
        cuda_files=["sampling/compact_support.cuh"],
        cuda_wrappers=[
            ("check_inputs", f"compact_check_inputs<{top_k}>"),
            ("run", f"compact_support<{args}>"),
        ],
        extra_cuda_cflags=["--fmad=false"],
    )


def compact_top_k_top_p_probs(
    probs: torch.Tensor,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Renormalize CUDA FP32 probabilities using bounded value groups.

    Contiguous [rows, vocab] nonnegative probabilities with positive mass
    are required. All cutoff ties are counted, including arbitrarily many
    equal tokens. The native FP32 softmax stays upstream so probability
    ties created by its rounding remain in the support. Reduction rounding
    changes still require numerical qualification. This consumes no RNG.
    """
    from flashinfer import top_k as select_top_k

    module = _compact_module(probs.dtype, top_k, probs.device.type)
    module.check_inputs(probs)
    values, indices = select_top_k(
        probs, top_k, sorted=True, deterministic=True, dsa_graph_safe=True
    )
    rows, vocab = probs.shape
    counts = torch.empty(
        (rows, (vocab + 4095) // 4096), dtype=torch.int32, device=probs.device
    )
    higher_probs = torch.empty_like(values)
    cutoff_params = torch.empty((rows, 2), dtype=torch.float32, device=probs.device)
    output = torch.empty_like(probs)
    module.run(
        output, probs, values, indices, counts, higher_probs, cutoff_params, top_p
    )
    return output
