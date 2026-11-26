"""D-side KV record writer for the in-place NVFP4 "nvfp4.fp8rope" layout (352 B per token per layer; FlashInfer 0.7.6 PrimTS sparse-MLA decode reads it in place).

write_fp8rope_records(dst, loc, k_nope, k_rope): one launch, in place, no return, no host sync, CUDA-graph capturable.
  dst     uint8 [rows, 352] contiguous: one layer's record pool
  loc     int32 / int64 [T]: record index of each token in dst; loc == skip_index (default none) or < 0 skips the token
  k_nope  bf16 [T, 512] or [T, 1, 512] (row stride in elements * 2 must be 16-byte aligned)
  k_rope  bf16 [T, 64] or [T, 1, 64], post-RoPE
split=True (default, mode 4): two lanes per NoPE block, ~0.25 us per call faster than fast=True; bit-identical on every input set tested. fast=True (mode 3): the rcp + hardware-cvt path, bit-identical to scale_true_div=True (verified on random, heavy-tailed, edge and exact-midpoint inputs) and faster. scale_true_div=True (mode 2, e9: the reference ran on CUDA tensors): IEEE amax / 6. False: amax * fl(1/6), same bytes on every input set tested.
trt_numerics=False: bit-exact with the torch reference fp8rope_records() (d576_micro_v4.py). True: TensorRT-LLM style rcp.approx scale and hardware cvt (ties to even).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

RECORD_BYTES = 352


@cache_once
def _module() -> Module:
    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("The NVFP4 fp8rope record writer requires SM10x")
    return load_jit(
        "nvfp4_fp8rope_d_sm10x",
        cuda_files=["dsa/nvfp4_fp8rope_d_writer.cuh"],
        cuda_wrappers=[
            ("write", "nvfp4_fp8rope_d::WriterKernel<0>::run"),
            ("write_trt", "nvfp4_fp8rope_d::WriterKernel<1>::run"),
            ("write_div", "nvfp4_fp8rope_d::WriterKernel<2>::run"),
            ("write_fast", "nvfp4_fp8rope_d::WriterKernel<3>::run"),
            ("write_split", "nvfp4_fp8rope_d::WriterKernel<4>::run"),
            ("write_split_recip", "nvfp4_fp8rope_d::WriterKernel<5>::run"),
        ],
        extra_cuda_cflags=["-O3", "-DNDEBUG"],
    )


def write_fp8rope_records(
    dst: torch.Tensor,
    loc: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    skip_index: int = -1,
    trt_numerics: bool = False,
    scale_true_div: bool = True,
    fast: bool = True,
    split: bool = True,
    recip_scale: bool = False,
) -> None:
    n = loc.shape[0]
    mod = _module()
    fn = mod.write_trt if trt_numerics else ((mod.write_split_recip if recip_scale else mod.write_split) if (split or recip_scale) else (mod.write_fast if fast else (mod.write_div if scale_true_div else mod.write)))
    fn(dst, loc, k_nope.view(n, -1), k_rope.view(n, -1), skip_index)
