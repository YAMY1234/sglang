"""Batched factored-state stores without per-layer host synchronization."""
import triton
import triton.language as tl


@triton.jit
def _store_factored_kernel(
    A, U, W, FA, FU, FW, COUNT, STALE, DENSE_OF, SLOTS,
    DENSE, RING, RING_DST,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, RMAX: tl.constexpr,
    A0: tl.constexpr, A1: tl.constexpr, A2: tl.constexpr, SLOT_STRIDE: tl.constexpr,
    R: tl.constexpr, STALE_VALUE: tl.constexpr, WRITE_RING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    tile = tl.program_id(2)
    slot = tl.load(SLOTS + row * SLOT_STRIDE).to(tl.int64)
    if slot < 0:
        return
    src = row.to(tl.int64) * H + head
    dst = slot * H + head
    x = tile * BLOCK + tl.arange(0, BLOCK)
    av = tl.load(A + row * A0 + head * A1 + x * A2, x < K, other=0)
    uv = tl.load(U + src * RMAX * K + x, x < RMAX * K, other=0)
    wv = tl.load(W + src * RMAX * V + x, x < RMAX * V, other=0)
    tl.store(FA + dst * K + x, av, x < K)
    tl.store(FU + dst * RMAX * K + x, uv, x < RMAX * K)
    tl.store(FW + dst * RMAX * V + x, wv, x < RMAX * V)
    if tile == 0:
        tl.store(COUNT + dst, R)
        if head == 0:
            tl.store(STALE + slot, STALE_VALUE)
            if STALE_VALUE == 1:
                tl.store(DENSE_OF + slot, -1)
    if WRITE_RING:
        ring_slot = tl.load(RING_DST + row).to(tl.int64)
        if ring_slot >= 0:
            dense = tl.load(DENSE + src * V * K + x, x < V * K, other=0)
            tl.store(RING + (ring_slot * H + head) * V * K + x, dense, x < V * K)


def store_factored(a, u, w, fa, fu, fw, count, stale, dense_of, slots, r,
                   *, stale_value, dense=None, ring=None, ring_dst=None):
    """Scatter factors; negative slots leave every pool untouched.

    Caller guarantees unique nonnegative slots and ring destinations. This is
    the same contract as the pool's previous indexed-assignment path.
    """
    if slots.numel() == 0:
        return
    assert u.is_contiguous() and w.is_contiguous()
    b, h, rmax, k = u.shape
    v = w.shape[-1]
    write_ring = dense is not None
    if write_ring:
        assert dense.is_contiguous() and ring_dst.is_contiguous()
    extent = max(k, rmax*k, rmax*v, v*k if write_ring else 0)
    _store_factored_kernel[(b, h, triton.cdiv(extent, 1024))](
        a, u, w, fa, fu, fw, count, stale, dense_of, slots,
        dense if write_ring else a, ring if write_ring else a,
        ring_dst if write_ring else slots, h, k, v, rmax, *a.stride(), slots.stride(0), r,
        stale_value, write_ring, 1024, num_warps=4,
    )
