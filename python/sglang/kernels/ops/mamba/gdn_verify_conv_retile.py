"""The causal conv update of a GDN target verify on a [rows, block_n] tile.

``causal_conv1d_update`` with continuous batching, pad slots, saved windows, no tree
and width 4 (more verify tokens than state columns): the same values and the same
arithmetic per element as ``_causal_conv1d_update_kernel``, so bit for bit its output,
new conv state and saved windows. It loads every x row up front and, under PDL, can
load the conv weights and the prior conv state before waiting for the producer.
"""

import logging
from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.mamba.causal_conv1d_triton import PAD_SLOT_ID
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class RetileConfig(NamedTuple):
    block_n: int
    num_warps: int
    # batch rows per program (a power of 2)
    rows: int = 1
    # 0: every load after the PDL wait; 1: the conv weights before it; 2: also the
    # conv-state indices and the prior conv state (written only by earlier forwards,
    # never by a kernel of this verify before the conv)
    prefetch: int = 0


# Measured by the source on VR200 at verify batches 4-104; re-tune when the kernel
# changes.
RETILE_DEFAULT = RetileConfig(block_n=256, num_warps=4, rows=1, prefetch=2)


def parse_retile_config(*, enabled: bool, raw: Optional[str]) -> Optional[RetileConfig]:
    """SGLANG_GDN_VERIFY_CONV_RETILE_CONFIG (A's name):
    "block_n,num_warps[,rows[,prefetch]]"."""
    if not enabled:
        return None
    if raw is None or not raw.strip():
        return RETILE_DEFAULT
    vals = [int(v) for v in raw.split(",")]
    if not 2 <= len(vals) <= 4:
        raise ValueError(
            f"SGLANG_GDN_VERIFY_CONV_RETILE_CONFIG takes 2-4 integers, not {raw!r}"
        )
    cfg = RetileConfig(*vals)

    def pow2(v, lo, hi):
        return lo <= v <= hi and v & (v - 1) == 0

    if not (
        pow2(cfg.block_n, 16, 4096)
        and pow2(cfg.num_warps, 1, 16)
        and pow2(cfg.rows, 1, 16)
        and cfg.prefetch in (0, 1, 2)
    ):
        raise ValueError(
            "SGLANG_GDN_VERIFY_CONV_RETILE_CONFIG: block_n a power of 2 in 16-4096, "
            "num_warps in 1-16, rows in 1-16 (powers of 2), prefetch 0-2; "
            f"got {raw!r}"
        )
    return cfg


# SGLANG_OPT_GDN_VERIFY_CONV_RETILE; read once.
RETILE_CONFIG = parse_retile_config(
    enabled=envs.SGLANG_OPT_GDN_VERIFY_CONV_RETILE.get(),
    raw=envs.SGLANG_GDN_VERIFY_CONV_RETILE_CONFIG.get(),
)
_retile_logged: set = set()


def retile_unsupported_reason(
    *, x, conv_state, weight, intermediate_conv_window
) -> Optional[str]:
    """None when the retiled kernel reproduces ``_causal_conv1d_update_kernel``."""
    seqlen, width = x.shape[2], weight.shape[1]
    if width != 4 or conv_state.shape[-1] != width - 1:
        return f"conv width {width}, state length {conv_state.shape[-1]} (needs 4, 3)"
    if not width - 1 < seqlen <= 8:
        return f"{seqlen} verify tokens (needs 4-8)"
    if not (x.dtype == conv_state.dtype == intermediate_conv_window.dtype):
        return (
            f"dtypes x {x.dtype}, state {conv_state.dtype}, "
            f"window {intermediate_conv_window.dtype}"
        )
    if intermediate_conv_window.dim() != 4:
        return (
            f"window of {intermediate_conv_window.dim()} dims "
            "(needs [slot, step, dim, win])"
        )
    return None


def _log_retile_once(*, reason: Optional[str], batch: int, dim: int, seqlen: int):
    key = reason is None
    if key in _retile_logged:
        return
    _retile_logged.add(key)
    if reason is None:
        logger.info(
            "GDN verify conv retile engaged (SGLANG_OPT_GDN_VERIFY_CONV_RETILE): %s, "
            "dim %d, %d tokens, first batch %d",
            RETILE_CONFIG,
            dim,
            seqlen,
            batch,
        )
    else:
        logger.warning("GDN verify conv retile not taken (stock conv): %s", reason)


@triton.jit
def _gdn_verify_conv_retiled_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    inter_ptr,
    inter_indices_ptr,
    o_ptr,
    batch,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_cs_seq: tl.constexpr,
    stride_cs_dim: tl.constexpr,
    stride_cs_tok: tl.constexpr,
    stride_csi: tl.constexpr,
    stride_inter_seq: tl.constexpr,
    stride_inter_step: tl.constexpr,
    stride_inter_dim: tl.constexpr,
    stride_inter_win: tl.constexpr,
    stride_ii: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
    PREFETCH: tl.constexpr,
    WINDOW_TILE: tl.constexpr,
    NP2_WIN: tl.constexpr,
    USE_GDC: tl.constexpr,
):
    feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    fmask = feats < dim
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    if USE_GDC and PREFETCH == 0:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    w_base = w_ptr + feats * stride_w_dim
    w0 = tl.load(w_base + 0 * stride_w_width, fmask, other=0.0)
    w1 = tl.load(w_base + 1 * stride_w_width, fmask, other=0.0)
    w2 = tl.load(w_base + 2 * stride_w_width, fmask, other=0.0)
    w3 = tl.load(w_base + 3 * stride_w_width, fmask, other=0.0)
    if USE_GDC and PREFETCH == 1:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    in_batch = rows < batch
    coord = tl.load(
        conv_state_indices_ptr + rows * stride_csi, mask=in_batch, other=pad_slot_id
    ).to(tl.int64)
    # padded rows (pad slot) write nothing, as the stock kernel's early return
    m = (in_batch & (coord != pad_slot_id))[:, None] & fmask[None, :]
    cs = (
        conv_state_ptr
        + coord[:, None] * stride_cs_seq
        + (feats * stride_cs_dim)[None, :]
    )
    col0 = tl.load(cs, m, 0.0)
    col1 = tl.load(cs + 1 * stride_cs_tok, m, 0.0)
    col2 = tl.load(cs + 2 * stride_cs_tok, m, 0.0)
    if USE_GDC and PREFETCH == 2:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    # after the wait: under PP spec decoding the launch just before the conv writes it
    inter_coord = tl.load(
        inter_indices_ptr + rows * stride_ii, mask=in_batch, other=0
    ).to(tl.int64)
    prior1 = col1
    prior2 = col2
    # every x row before the first store (a store may alias a later load for the compiler)
    xb = x_ptr + rows[:, None] * stride_x_seq + (feats * stride_x_dim)[None, :]
    x0 = tl.load(xb + 0 * stride_x_token, mask=m)
    x1 = tl.load(xb + 1 * stride_x_token, mask=m)
    x2 = tl.load(xb + 2 * stride_x_token, mask=m)
    x3 = tl.load(xb + 3 * stride_x_token, mask=m)
    if seqlen > 4:
        x4 = tl.load(xb + 4 * stride_x_token, mask=m)
    if seqlen > 5:
        x5 = tl.load(xb + 5 * stride_x_token, mask=m)
    if seqlen > 6:
        x6 = tl.load(xb + 6 * stride_x_token, mask=m)
    if seqlen > 7:
        x7 = tl.load(xb + 7 * stride_x_token, mask=m)
    # the new conv state: the last 3 x rows (stock STEP 2 with seqlen > state_len)
    n0 = tl.load(xb + (seqlen - 3) * stride_x_token, mask=m, other=0.0)
    n1 = tl.load(xb + (seqlen - 2) * stride_x_token, mask=m, other=0.0)
    n2 = tl.load(xb + (seqlen - 1) * stride_x_token, mask=m, other=0.0)
    if WINDOW_TILE:
        # step t's window is columns t..t+2 of [state[1], state[2], x_0, ..., x_{seqlen-1}]
        c = tl.arange(0, NP2_WIN)
        c_ok = c < seqlen + 2
        xw = tl.load(
            xb[:, :, None] + ((c - 2) * stride_x_token)[None, None, :],
            mask=m[:, :, None] & ((c >= 2) & c_ok)[None, None, :],
            other=0.0,
        )
        win = tl.where(
            (c == 0)[None, None, :],
            prior1[:, :, None],
            tl.where((c == 1)[None, None, :], prior2[:, :, None], xw),
        )
    tl.store(cs, n0, m)
    tl.store(cs + 1 * stride_cs_tok, n1, m)
    tl.store(cs + 2 * stride_cs_tok, n2, m)
    inter_base = (
        inter_ptr
        + inter_coord[:, None] * stride_inter_seq
        + (feats * stride_inter_dim)[None, :]
    )
    if WINDOW_TILE:
        tl.store(
            inter_base[:, :, None] + (c * stride_inter_win)[None, None, :],
            win,
            mask=m[:, :, None] & c_ok[None, None, :],
        )
    if HAS_BIAS:
        b = tl.load(bias_ptr + feats, mask=fmask, other=0.0).to(tl.float32)
        acc_preload = tl.broadcast_to(b[None, :], (ROWS, BLOCK_N))
    else:
        acc_preload = tl.zeros((ROWS, BLOCK_N), dtype=tl.float32)
    ob = o_ptr + rows[:, None] * stride_o_seq + (feats * stride_o_dim)[None, :]
    for t in tl.static_range(seqlen):
        if t == 0:
            xt = x0
        elif t == 1:
            xt = x1
        elif t == 2:
            xt = x2
        elif t == 3:
            xt = x3
        elif t == 4:
            xt = x4
        elif t == 5:
            xt = x5
        elif t == 6:
            xt = x6
        else:
            xt = x7
        # the stock kernel's order: state columns oldest first, then the token
        acc = acc_preload
        acc += col0 * w0[None, :]
        acc += col1 * w1[None, :]
        acc += col2 * w2[None, :]
        acc += xt * w3[None, :]
        col0 = col1
        col1 = col2
        col2 = xt
        if not WINDOW_TILE:
            step = inter_base + t * stride_inter_step
            tl.store(step + 0 * stride_inter_win, col0, mask=m)
            tl.store(step + 1 * stride_inter_win, col1, mask=m)
            tl.store(step + 2 * stride_inter_win, col2, mask=m)
        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        tl.store(ob + t * stride_o_token, acc, mask=m)


def gdn_verify_conv_retiled(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation,
    conv_state_indices: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
) -> Optional[torch.Tensor]:
    """``causal_conv1d_update`` of a linear-chain verify x (batch, dim, seqlen) with
    those arguments on RETILE_CONFIG's tile, or None (nothing written) when the call
    is not one the retiled kernel reproduces."""
    batch, dim, seqlen = x.shape
    reason = retile_unsupported_reason(
        x=x,
        conv_state=conv_state,
        weight=weight,
        intermediate_conv_window=intermediate_conv_window,
    )
    _log_retile_once(reason=reason, batch=batch, dim=dim, seqlen=seqlen)
    if reason is not None:
        return None
    if isinstance(activation, bool):
        activation = "silu" if activation else None
    cfg = RETILE_CONFIG
    use_gdc = is_arch_support_pdl()
    out = torch.empty_like(x)
    win_strides = intermediate_conv_window.stride()
    _gdn_verify_conv_retiled_kernel[
        (triton.cdiv(batch, cfg.rows), triton.cdiv(dim, cfg.block_n))
    ](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        out,
        batch,
        dim,
        seqlen,
        *x.stride(),
        *weight.stride(),
        *conv_state.stride(),
        conv_state_indices.stride(0),
        *win_strides,
        intermediate_state_indices.stride(0),
        *out.stride(),
        PAD_SLOT_ID,
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        BLOCK_N=cfg.block_n,
        ROWS=cfg.rows,
        PREFETCH=cfg.prefetch,
        WINDOW_TILE=win_strides[1] == win_strides[3],
        NP2_WIN=triton.next_power_of_2(seqlen + 2),
        USE_GDC=use_gdc,
        num_warps=cfg.num_warps,
        **({"launch_pdl": True} if use_gdc else {}),
    )
    return out
