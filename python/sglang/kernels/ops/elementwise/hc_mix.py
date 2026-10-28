from __future__ import annotations

import os

import torch

_MAX_ROWS = 32
_KPAD_ALIGN = 128
_M4_TACTIC = int(os.environ.get("SGLANG_Q38C_HC_M4_MIX_TACTIC", "0"))
if _M4_TACTIC not in (0, 1, 2, 3, 4):
    raise ValueError("SGLANG_Q38C_HC_M4_MIX_TACTIC must be 0..4")


def pad_lowrank(lowrank: int) -> int:
    return (lowrank + _KPAD_ALIGN - 1) // _KPAD_ALIGN * _KPAD_ALIGN


def permute_pad_up_weight(w_up: torch.Tensor, hc_count: int) -> torch.Tensor:
    k, lowrank = w_up.shape
    hidden_size = k // hc_count
    padded = torch.zeros(
        (k, pad_lowrank(lowrank)), dtype=w_up.dtype, device=w_up.device
    )
    padded[:, :lowrank] = w_up.detach()
    src = torch.arange(k, device=w_up.device)
    perm = torch.empty(k, dtype=torch.long, device=w_up.device)
    perm[(src % hidden_size) * hc_count + (src // hidden_size)] = src
    return padded[perm].contiguous()


_scratch_cache = {}
_tactic_cache = {}


def _get_scratch(lowrank: int, dtype: torch.dtype, device: torch.device):
    key = (device, lowrank, dtype)
    buf = _scratch_cache.get(key)
    if buf is None:
        buf = torch.zeros((_MAX_ROWS, pad_lowrank(lowrank)), dtype=dtype, device=device)
        _scratch_cache[key] = buf
    return buf


def _get_tactic(m: int, n: int, k: int):
    key = (m, n, k)
    tactic = _tactic_cache.get(key)
    if tactic is None:
        from sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue import (
            SplitKTactic,
            default_tactic,
            validate_tactic,
        )

        if m == 4 and (n, k) == (320, 10240) and _M4_TACTIC in (1, 4):
            tactic = (
                SplitKTactic(64, 8, 8, 6)
                if _M4_TACTIC == 1
                else SplitKTactic(64, 8, 16, 4)
            )
            validate_tactic(tactic, m, n, k)
        elif m == 4 and (n, k) == (10240, 384) and _M4_TACTIC in (2, 3):
            tactic = (
                SplitKTactic(64, 8, 1, 6)
                if _M4_TACTIC == 2
                else SplitKTactic(128, 8, 1, 3)
            )
            validate_tactic(tactic, m, n, k)
        elif n <= 512 and k % (128 * 16) == 0:
            tactic = SplitKTactic(mma_m=64, mma_n=8, split_k=16, ab_stages=6)
            try:
                validate_tactic(tactic, m, n, k)
            except ValueError:
                tactic = default_tactic(m, n, k)
        else:
            tactic = default_tactic(m, n, k)
        _tactic_cache[key] = tactic
    return tactic


def hc_mix(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up_permuted_padded: torch.Tensor,
    hc_count: int,
    hidden_size: int,
) -> torch.Tensor:
    from sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue import (
        run_splitk_dense_gate,
        run_splitk_dense_silu,
    )

    rows, k = hyper_input_normed.shape
    lowrank = w_down.shape[0]
    kpad = w_up_permuted_padded.shape[1]
    out = torch.empty(
        (rows, hidden_size),
        dtype=hyper_input_normed.dtype,
        device=hyper_input_normed.device,
    )
    if rows == 0:
        return out
    inv_hc = 1.0 / hc_count
    t_pad = _get_scratch(lowrank, hyper_input_normed.dtype, hyper_input_normed.device)
    for row_start in range(0, rows, _MAX_ROWS):
        row_end = min(row_start + _MAX_ROWS, rows)
        m = row_end - row_start
        chunk = hyper_input_normed[row_start:row_end]
        run_splitk_dense_silu(
            chunk,
            w_down.T,
            t_pad[:m, :lowrank],
            True,
            _get_tactic(m, lowrank, k),
            inv_hc,
        )
        run_splitk_dense_gate(
            t_pad[:m],
            w_up_permuted_padded.T,
            chunk,
            out[row_start:row_end],
            True,
            _get_tactic(m, k, kpad),
            inv_hc,
            hc_count,
        )
    return out
