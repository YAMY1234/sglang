"""Fused Qwen3.5 full-attention prologue / epilogue for bs1 MTP verify (kernels in mka_fused.cu)."""

import threading
from pathlib import Path

import torch

_EXT = None
_LOCK = threading.Lock()


def _ext():
    global _EXT
    if _EXT is None:
        with _LOCK:
            if _EXT is None:
                from torch.utils.cpp_extension import load

                major, minor = torch.cuda.get_device_capability()
                _EXT = load(
                    name="sglang_q35mf_mka_fused",
                    sources=[str(Path(__file__).with_suffix(".cu"))],
                    extra_cuda_cflags=[
                        "-O3",
                        "--std=c++17",
                        f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}",
                    ],
                    verbose=False,
                )
    return _EXT


def ensure_built() -> None:
    """Compile (or load) the extension now, before any collective can time out on a JIT build."""
    if torch.cuda.is_available():
        _ext()


def _pack_fragments(w_nk: torch.Tensor, units: int, steps: int) -> torch.Tensor:
    # [units*256, K] e4m3 -> [unit][rank 8][warp 8][step][m16 block 2][lane 32][16 B] mma.m16n8k32 A
    # fragments; rank r covers K [r*steps*32, (r+1)*steps*32), warp w rows [32w, 32w + 32) of the unit.
    w = w_nk.contiguous().view(torch.uint8)
    assert w.shape[0] == units * 256 and w.shape[1] == 8 * steps * 32, w.shape
    v = w.view(units, 8, 2, 2, 8, 8, steps, 2, 4, 4)
    return v.permute(0, 5, 1, 6, 2, 4, 8, 7, 3, 9).contiguous().view(-1)


def pack_qkv(w_nk: torch.Tensor) -> torch.Tensor:
    """qkv(+gate) weight [N, 4096] e4m3 -> prologue layout (K chunk 512 per cluster rank)."""
    return _pack_fragments(w_nk, w_nk.shape[0] // 256, 16)


def pack_o(w_nk: torch.Tensor) -> torch.Tensor:
    """o_proj weight [4096, nq*256] e4m3 -> epilogue layout (K chunk nq*32 per cluster rank)."""
    return _pack_fragments(w_nk, 16, w_nk.shape[1] // 256)


def prologue(
    x, w_packed, sx, sw, q_norm_w, k_norm_w, cos_sin, positions, axis_map, cache_loc,
    k_cache, v_cache, k_scale, v_scale, q_out, gate_out, nq: int, eps: float,
):
    _ext().prologue(
        x, w_packed, sx, sw, q_norm_w, k_norm_w, cos_sin, positions, axis_map, cache_loc,
        k_cache, v_cache, k_scale, v_scale, q_out, gate_out, nq, eps, True,
    )


def epilogue(attn, gate, w_packed, sx, sw, out, nq: int):
    _ext().epilogue(attn, gate, w_packed, sx, sw, out, nq, True)
