"""MIDQ: FlashInfer MoE all-to-all built from a patched copy of its kernel source in which
the combine splits each token row over several CTAs. Every output element is computed by the
same instruction sequence as the stock kernel, so the combine output is bit-identical."""

from __future__ import annotations

import logging
import pathlib
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_KERNEL_SOURCE = (
    pathlib.Path(__file__).resolve().parent / "midq_csrc" / "moeAlltoAllKernels_midq.cu"
)
_installed_split: Optional[int] = None
_stock_module = None


def _split_combine_spec(split: int):
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import gen_jit_spec

    # Same sources and flags as FlashInfer 0.7.1 gen_moe_alltoall_module("legacy"),
    # with moeAlltoAllKernels.cu replaced by the patched copy.
    csrc = jit_env.FLASHINFER_CSRC_DIR
    common = csrc / "nv_internal" / "cpp" / "common"
    return gen_jit_spec(
        f"mnnvl_moe_alltoall_midq_s{split}",
        [
            csrc / "trtllm_moe_alltoall.cu",
            _KERNEL_SOURCE,
            common / "envUtils.cpp",
            common / "tllmException.cpp",
            common / "logger.cpp",
            common / "stringUtils.cpp",
        ],
        extra_include_paths=[
            str(csrc / "nv_internal"),
            str(csrc / "nv_internal" / "include"),
        ],
        extra_cuda_cflags=["-DENABLE_BF16", f"-DMIDQ_COMBINE_SPLIT={split}"],
    )


def install_split_combine(split: int) -> None:
    """Make FlashInfer's default (trtllm) MoE a2a module the split-combine build.

    Must run before the process loads the stock module; dispatch, combine and the
    workspace layout then all come from one module.
    """
    global _installed_split
    if _installed_split == split:
        return
    import flashinfer.comm.trtllm_moe_alltoall as fi_a2a

    if fi_a2a._get_moe_alltoall_module_for_target.cache_info().currsize:
        raise RuntimeError(
            "MIDQ split combine must be installed before FlashInfer loads its "
            "MoE all-to-all module"
        )
    stock_gen = fi_a2a.gen_moe_alltoall_module

    def gen(target="legacy"):
        return _split_combine_spec(split) if target == "legacy" else stock_gen(target)

    fi_a2a.gen_moe_alltoall_module = gen
    _installed_split = split
    logger.info(
        "MIDQ FlashInfer a2a combine: split-token build mnnvl_moe_alltoall_midq_s%d installed",
        split,
    )


def stock_combine_module():
    """The unpatched FlashInfer 0.7.1 module, loaded raw (no custom-op registration)."""
    global _stock_module
    if _stock_module is None:
        from flashinfer.jit.comm import gen_moe_alltoall_module

        _stock_module = gen_moe_alltoall_module("legacy").build_and_load()
    return _stock_module


class CombineProof:
    """Eager-only check that the installed combine equals the stock kernel bit for bit.

    One instance per process and per (target, draft); every dispatcher of that kind shares it.
    """

    def __init__(self, speculative: bool):
        self.speculative = speculative
        self.calls = 0
        self.elements = 0

    def check(self, ref: torch.Tensor, out: torch.Tensor) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MIDQ combine proof runs eager only (--disable-cuda-graph)")
        mismatched = int(
            torch.count_nonzero(ref.view(torch.int16) != out.view(torch.int16)).item()
        )
        self.calls += 1
        self.elements += ref.numel()
        if mismatched:
            raise RuntimeError(
                f"MIDQ combine proof: {mismatched} of {ref.numel()} elements differ "
                f"(speculative={self.speculative}, call {self.calls})"
            )
        if self.calls % 200 == 0:
            logger.info(
                "MIDQ combine proof: speculative=%s calls=%d elements=%d mismatched=0",
                self.speculative,
                self.calls,
                self.elements,
            )


_proofs: dict[bool, CombineProof] = {}


def get_combine_proof(speculative: bool) -> CombineProof:
    if speculative not in _proofs:
        _proofs[speculative] = CombineProof(speculative)
    return _proofs[speculative]
