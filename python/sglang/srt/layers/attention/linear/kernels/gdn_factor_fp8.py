"""FP8 factor IO around the unchanged bf16 K3-A compute kernels.

Only the active batch has bf16 scratch; persistent slots are byte factors plus
per-column fp32 scales. Conversion launches must be included in kernel timing.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _factor_fp8_io_impl(A, U, W, COUNT, US, WS, BA, BU, BW, BC, IDX, STALE, LOCAL, BSTALE,
                   S: tl.constexpr, B: tl.constexpr, HV: tl.constexpr, R: tl.constexpr,
                   STORE: tl.constexpr, MAKE_LOCAL: tl.constexpr):
    bh, col, layer = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row, h = bh // HV, bh % HV
    slot = tl.load(IDX + row).to(tl.int64)
    if MAKE_LOCAL and not STORE:
        if layer == 0 and col == 0 and h == 0:
            tl.store(LOCAL + row, tl.where(slot >= 0, row, -1))
            tl.store(BSTALE + row, 0)
    d = tl.arange(0, 128)
    local = ((layer * B + row) * HV + h).to(tl.int64)
    glob = (layer * S + tl.maximum(slot, 0)) * HV + h
    if STORE:
        if slot >= 0:
            u = tl.load(BU + (local * R + col) * 128 + d).to(tl.float32)
            w = tl.load(BW + (local * R + col) * 128 + d).to(tl.float32)
            su = tl.maximum(tl.max(tl.abs(u), 0) / 448., 1.e-30)
            sw = tl.maximum(tl.max(tl.abs(w), 0) / 448., 1.e-30)
            tl.store(U + (glob * R + col) * 128 + d, (u / su).to(tl.float8e4nv).to(tl.uint8, bitcast=True))
            tl.store(W + (glob * R + col) * 128 + d, (w / sw).to(tl.float8e4nv).to(tl.uint8, bitcast=True))
            tl.store(US + glob * R + col, su)
            tl.store(WS + glob * R + col, sw)
            if col == 0:
                tl.store(A + glob * 128 + d, tl.load(BA + local * 128 + d))
                tl.store(COUNT + glob, tl.load(BC + local))
                tl.store(STALE + slot, 1)
    else:
        u = tl.load(U + (glob * R + col) * 128 + d).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        w = tl.load(W + (glob * R + col) * 128 + d).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        su, sw = tl.load(US + glob * R + col), tl.load(WS + glob * R + col)
        tl.store(BU + (local * R + col) * 128 + d, tl.where(slot >= 0, u * su, 0.))
        tl.store(BW + (local * R + col) * 128 + d, tl.where(slot >= 0, w * sw, 0.))
        if col == 0:
            tl.store(BA + local * 128 + d, tl.where(slot >= 0, tl.load(A + glob * 128 + d), 0.))
            tl.store(BC + local, tl.where(slot >= 0, tl.load(COUNT + glob), 0))


@triton.jit
def _factor_fp8_io(A, U, W, COUNT, US, WS, BA, BU, BW, BC, IDX, STALE, LOCAL, BSTALE,
                   S: tl.constexpr, B: tl.constexpr, HV: tl.constexpr, R: tl.constexpr,
                   STORE: tl.constexpr, MAKE_LOCAL: tl.constexpr):
    _factor_fp8_io_impl(A, U, W, COUNT, US, WS, BA, BU, BW, BC, IDX, STALE, LOCAL, BSTALE,
                       S, B, HV, R, STORE, MAKE_LOCAL)


@triton.jit
def _factor_fp8_decode_io(A, U, W, COUNT, US, WS, BA, BU, BW, BC, IDX, STALE, LOCAL, BSTALE,
                          S: tl.constexpr, B: tl.constexpr, HV: tl.constexpr, R: tl.constexpr,
                          STORE: tl.constexpr, MAKE_LOCAL: tl.constexpr):
    _factor_fp8_io_impl(A, U, W, COUNT, US, WS, BA, BU, BW, BC, IDX, STALE, LOCAL, BSTALE,
                       S, B, HV, R, STORE, MAKE_LOCAL)


def transfer(a, u, w, count, us, ws, batch, slots, stale, store=False, local=None, batch_stale=None, decode=False):
    ba, bu, bw, bc = batch
    # Shapes include L even for a single layer.
    kernel = _factor_fp8_decode_io if decode else _factor_fp8_io
    kernel[(slots.numel() * a.shape[2], u.shape[3], a.shape[0])](
        a, u, w, count, us, ws, ba, bu, bw, bc, slots, stale, local, batch_stale,
        a.shape[1], slots.numel(), a.shape[2], u.shape[3], store, local is not None, num_warps=4,
    )


def allocate_batch(layers, b, hv, r, device):
    return (torch.empty(layers, b, hv, 128, device=device, dtype=torch.float32),
            torch.empty(layers, b, hv, r, 128, device=device, dtype=torch.bfloat16),
            torch.empty(layers, b, hv, r, 128, device=device, dtype=torch.bfloat16),
            torch.empty(layers, b, hv, device=device, dtype=torch.int32))
