"""Fused singleton prefix restore: factored GDN state -> dense initial states for ALL layers in one launch.

Replaces the PrefillRestoreGraph replay (36 x [gather, mask, einsum, mul, add] + stack, then clone) with one Triton
kernel that writes a fresh (L, 1, HV, V, K) fp32 stage:

    S[l, 0, h, v, k] = sum_{r < count[l, slot, h]} W[l, slot, h, r, v] * U[l, slot, h, r, k]
                       + vbar[l, h, v] * a[l, slot, h, k]

Arithmetic is ordered to reproduce the reference `densify()` bit for bit when its fp32 einsum runs as a cuBLAS
non-TF32 SIMT GEMM (sequential FMA over the reduction dim from a zero accumulator):
  * acc starts at +0 and takes acc = fma(W[r, v], U[r, k], acc) for r = 0, 1, ... in order (explicit tl.fma, no
    tl.dot). Rows r >= count are skipped: in the reference they are multiplied by a zero mask and contribute
    fma(+-0, +-0, acc) == acc exactly (up to the sign of an exact zero), so skipping them is value-identical.
  * t = vbar * a is a separately rounded multiply and out = acc + t a separately rounded add, as in torch's eager
    mul then add. Both are emitted as inline PTX `mul.rn.f32` / `add.rn.f32`: PTX never contracts instructions with
    an explicit rounding modifier into an fma, independent of Triton's enable_fp_fusion default.
The slot index is read on device from a 1-element tensor (clamped at 0 like the eager path); no host sync.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _mul_rn(x, y):
    # Explicit-rounding PTX multiply: never contracted with the following add.
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=r,r,r",
        [x.to(tl.int32, bitcast=True), y.to(tl.int32, bitcast=True)],
        dtype=tl.int32, is_pure=True, pack=1).to(tl.float32, bitcast=True)


@triton.jit
def _add_rn(x, y):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", "=r,r,r",
        [x.to(tl.int32, bitcast=True), y.to(tl.int32, bitcast=True)],
        dtype=tl.int32, is_pure=True, pack=1).to(tl.float32, bitcast=True)


@triton.jit
def _prefix_restore_kernel(
    slot_ptr, a_ptr, u_ptr, w_ptr, cnt_ptr, vbar_ptr, out_ptr,
    sa_l, sa_s, sa_h,
    su_l, su_s, su_h, su_r,
    sw_l, sw_s, sw_h, sw_r,
    sc_l, sc_s, sc_h,
    svb_l, svb_h,
    HV: tl.constexpr, R: tl.constexpr, V: tl.constexpr, K: tl.constexpr,
    BV: tl.constexpr, BK: tl.constexpr,
):
    pid_lh = tl.program_id(0)
    pid_t = tl.program_id(1)
    l = (pid_lh // HV).to(tl.int64)
    h = (pid_lh % HV).to(tl.int64)
    tv = pid_t // (K // BK)
    tk = pid_t % (K // BK)
    slot = tl.maximum(tl.load(slot_ptr).to(tl.int64), 0)

    cnt = tl.load(cnt_ptr + l * sc_l + slot * sc_s + h * sc_h).to(tl.int32)
    cnt = tl.minimum(cnt, R)  # the reference mask is r < count over r < RMAX

    offs_v = tv * BV + tl.arange(0, BV)
    offs_k = tk * BK + tl.arange(0, BK)
    u_row = u_ptr + l * su_l + slot * su_s + h * su_h + offs_k
    w_row = w_ptr + l * sw_l + slot * sw_s + h * sw_h + offs_v

    acc = tl.zeros((BV, BK), dtype=tl.float32)
    for r in range(0, cnt):
        w = tl.load(w_row + r * sw_r).to(tl.float32)
        u = tl.load(u_row + r * su_r).to(tl.float32)
        acc = tl.fma(tl.broadcast_to(w[:, None], (BV, BK)), tl.broadcast_to(u[None, :], (BV, BK)), acc)

    a = tl.load(a_ptr + l * sa_l + slot * sa_s + h * sa_h + offs_k).to(tl.float32)
    vb = tl.load(vbar_ptr + l * svb_l + h * svb_h + offs_v).to(tl.float32)
    t = _mul_rn(tl.broadcast_to(vb[:, None], (BV, BK)), tl.broadcast_to(a[None, :], (BV, BK)))
    out = _add_rn(acc, t)

    out_base = out_ptr + (l * HV + h) * (V * K)
    tl.store(out_base + offs_v[:, None] * K + offs_k[None, :], out)


def fused_restore(pool, slot_tensor: torch.Tensor, *, block_v: int = 64, block_k: int = 64,
                  num_warps: int = 4) -> torch.Tensor:
    """pool: FactoredGDNPool; slot_tensor: 1-element int device tensor (the plan's slot).

    Returns a NEW (L, 1, HV, V, K) fp32 tensor equal to
    torch.stack([densify(a[l][s], U[l][s], W[l][s], count[l][s], vbar[l]) for l in layers]), s = slot.clamp(min=0).
    """
    a, U, W, count, vbar = pool.a, pool.U, pool.W, pool.count, pool.vbar
    L, _, hv, R, k = U.shape
    v = W.shape[-1]
    if L != len(pool.layer_ids) or slot_tensor.numel() != 1:
        raise ValueError("fused_restore needs every pool layer and exactly one slot")
    if a.stride(-1) != 1 or U.stride(-1) != 1 or W.stride(-1) != 1 or vbar.stride(-1) != 1:
        raise ValueError("fused_restore needs unit inner strides")
    bv, bk = min(block_v, v), min(block_k, k)
    if v % bv or k % bk:
        raise ValueError(f"V={v}/K={k} not divisible by block {bv}x{bk}")
    out = torch.empty(L, 1, hv, v, k, dtype=torch.float32, device=U.device)
    grid = (L * hv, (v // bv) * (k // bk))
    _prefix_restore_kernel[grid](
        slot_tensor, a, U, W, count, vbar, out,
        a.stride(0), a.stride(1), a.stride(2),
        U.stride(0), U.stride(1), U.stride(2), U.stride(3),
        W.stride(0), W.stride(1), W.stride(2), W.stride(3),
        count.stride(0), count.stride(1), count.stride(2),
        vbar.stride(0), vbar.stride(1),
        HV=hv, R=R, V=v, K=k, BV=bv, BK=bk,
        num_warps=num_warps,
    )
    return out


def restore_bytes(pool, counts=None) -> dict:
    """Minimal HBM traffic of one fused restore: full-R factor read, count-limited factor read, and the stage write."""
    L, _, hv, R, k = pool.U.shape
    v = pool.W.shape[-1]
    esz = pool.U.element_size()
    small = L * hv * (k * 4 + v * 4 + 4)  # a, vbar, count
    write = L * hv * v * k * 4
    full = L * hv * R * (k + v) * esz
    res = dict(write=write, read_full_r=full + small, total_full_r=full + small + write)
    if counts is not None:
        used = int(counts.clamp(max=R).sum()) * (k + v) * esz
        res.update(read_counted=used + small, total_counted=used + small + write)
    return res
