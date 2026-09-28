"""Opt-in real-weight FP8 LM head with BF16 nonfinite-row repair.

Restricted to the qualified Qwen3.5 BF16 head. Unsupported inputs return None
so the normal LM-head path retains its existing behavior. Weight conversion
happens once in eager warmup, never inside a captured graph.
"""
import logging
import os

import torch
import triton
import triton.language as tl
import sgl_kernel


@triton.jit
def quant_with_nonfinite_flags(X, Q, S, BAD, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + j, j < K, other=0).to(tl.float32)
    nonfinite = (x != x) | (tl.abs(x) == float('inf'))
    bad = tl.sum(nonfinite.to(tl.int32), 0) != 0
    clean = tl.where(nonfinite, 0., x)
    scale = tl.maximum(tl.max(tl.abs(clean), 0) / 448., tl.full((), 1.17549435e-38, tl.float32))
    q = tl.minimum(tl.maximum(clean / scale, -448.), 448.)
    tl.store(Q + row * K + j, q, j < K)
    tl.store(S + row, scale)
    tl.store(BAD + row, bad)


@triton.jit
def repair_bad_rows(X, W, BAD, OUT, M: tl.constexpr, V: tl.constexpr,
                    K: tl.constexpr, ROWS: tl.constexpr, BN: tl.constexpr,
                    BK: tl.constexpr):
    # Grid size depends only on vocabulary. Do not launch M * V tiles just to
    # discover that every row is finite at inference time.
    rows = tl.arange(0, ROWS)
    flags = tl.load(BAD + rows, rows < M, other=0)
    if tl.sum(flags.to(tl.int32), 0) != 0:
        n = tl.program_id(0) * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        mx = tl.arange(0, 16)
        for row in range(M):
            bad = tl.load(BAD + row)
            if bad:
                acc = tl.zeros((16, BN), tl.float32)
                for kb in range(tl.cdiv(K, BK)):
                    k = kb * BK + ks
                    x = tl.load(X + row * K + k, k < K, other=0)
                    a = tl.where(mx[:, None] == 0, x[None, :], 0).to(tl.bfloat16)
                    w = tl.load(W + n[None, :] * K + k[:, None],
                                (k[:, None] < K) & (n[None, :] < V), other=0)
                    acc = tl.dot(a, w, acc)
                output = tl.sum(tl.where(mx[:, None] == 0, acc, 0.), 0)
                tl.store(OUT + row * V + n, output.to(tl.bfloat16).to(tl.float32), n < V)


def run(x, w_bf16, w_fp8, w_scale, x_fp8, x_scale, bad_rows, output):
    m, k = x.shape
    quant_with_nonfinite_flags[(m,)](x, x_fp8, x_scale, bad_rows, k,
                                    triton.next_power_of_2(k), num_warps=4)
    output.copy_(sgl_kernel.fp8_scaled_mm(x_fp8, w_fp8.T, x_scale,
                                         w_scale.T, torch.bfloat16))
    repair_bad_rows[(triton.cdiv(len(w_bf16), 128),)](
        x, w_bf16, bad_rows, output, m, len(w_bf16), k,
        triton.next_power_of_2(m), 128, 32, num_warps=4)
    return output


logger = logging.getLogger(__name__)
ENABLED = os.environ.get("SGLANG_Q35_FP8_LM_HEAD", "0") == "1"


def try_fp8_lm_head(hidden_states, lm_head):
    if not ENABLED:
        return None
    w = lm_head.weight
    if (not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16
            or w.dtype != torch.bfloat16 or tuple(w.shape) != (248320, 4096)
            or hidden_states.ndim != 2 or hidden_states.shape[1] != 4096
            or not hidden_states.is_contiguous() or not w.is_contiguous()
            or hidden_states.shape[0] == 0 or hidden_states.shape[0] > 512):
        return None
    # Target and draft share this exact weight Parameter. Attaching to it avoids
    # retaining a second full-vocabulary FP8 copy for the draft worker.
    state = getattr(w, "_q35_fp8_lm_head_cache", None)
    if state is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("FP8 LM head requires eager warmup before graph capture")
        if not bool(torch.isfinite(w).all()):
            raise RuntimeError("FP8 LM head requires finite frozen BF16 weights")
        wq = torch.empty_like(w, dtype=torch.float8_e4m3fn)
        ws = torch.empty((w.shape[0], 1), device=w.device, dtype=torch.float32)
        flags = torch.empty((w.shape[0],), device=w.device, dtype=torch.int32)
        quant_with_nonfinite_flags[(len(w),)](w, wq, ws, flags, w.shape[1],
                                            triton.next_power_of_2(w.shape[1]), num_warps=4)
        state = (w._version, wq, ws)
        w._q35_fp8_lm_head_cache = state
        logger.info("Q35_FP8_LM_HEAD_READY shape=%s finite_weights=True", tuple(w.shape))
    if w._version != state[0]:
        raise RuntimeError("FP8 LM head weight changed after qualification/capture")
    m, k = hidden_states.shape
    xq = torch.empty_like(hidden_states, dtype=torch.float8_e4m3fn)
    xs = torch.empty((m, 1), device=w.device, dtype=torch.float32)
    bad = torch.empty((m,), device=w.device, dtype=torch.int32)
    out = torch.empty((m, len(w)), device=w.device, dtype=torch.bfloat16)
    return run(hidden_states, w, state[1], state[2], xq, xs, bad, out)
