"""DEP MoE exit in one launch (csrc/elementwise/qwen_dep_moe_exit.cuh): gated shared-expert add, residual add,
Gemma RMSNorm and static FP8 quant, equal to fused_gate_sigmoid_mul_add + gemma_fused_add_rmsnorm + static_quant_fp8."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args
from sglang.kernels.ops.elementwise.elementwise import _fused_gate_sigmoid_mul_add_kernel

if TYPE_CHECKING:
    from tvm_ffi.module import Module

HIDDEN = 4096
# kSquareFma: the operation order that reproduced the stock chain bit for bit in the T1 equivalence search
# (exit-r1: M 7 and 56); div.rn vs div.full for 1 / scale was not yet separated by that data.
DEFAULT_VARIANT = 4


@cache_once
def _jit_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "qwen_dep_moe_exit",
        *args,
        cuda_files=["elementwise/qwen_dep_moe_exit.cuh"],
        cuda_wrappers=[("run", f"sglang::MoeExit<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def moe_exit_gate(hidden_states: torch.Tensor, gate_weight: torch.Tensor) -> torch.Tensor:
    """fp32 [M] sigmoid gate of the shared expert, from the stock Triton kernel's GATE_ONLY path with the launch
    configuration fused_gate_sigmoid_mul_add uses (so its reduction order is the stock one)."""
    m, hidden_dim = hidden_states.shape
    gate = torch.empty((m,), dtype=torch.float32, device=hidden_states.device)
    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    _fused_gate_sigmoid_mul_add_kernel[(m,)](
        hidden_states,
        gate_weight,
        None,
        gate,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=HIDDEN,
        DO_ADD=False,
        USE_PDL=use_pdl,
        GATE_ONLY=True,
        num_warps=16,
        **pdl_kwargs,
    )
    return gate


def moe_exit(
    routed: torch.Tensor,
    shared: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    *,
    out: Optional[torch.Tensor] = None,
    variant: int = DEFAULT_VARIANT,
    dbg_final: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Writes the new residual into `residual`; returns (normed bf16 [M, 4096], e4m3 [M, 4096])."""
    if out is None:
        out = routed
    out_fp8 = torch.empty(routed.shape, dtype=torch.float8_e4m3fn, device=routed.device)
    if dbg_final is None:
        dbg_final = routed.new_empty((0, HIDDEN))
    _jit_module().run(
        routed, shared, gate, residual, out, out_fp8.view(torch.uint8), norm_weight, scale, dbg_final, eps, variant
    )
    return out, out_fp8
