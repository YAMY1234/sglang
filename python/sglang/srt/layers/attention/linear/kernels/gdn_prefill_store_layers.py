"""Scatter an unchanged packed factorization across independent GDN layers."""
import triton
import triton.language as tl


@triton.jit
def _store(A, U, W, FA, FU, FW, COUNT, STALE, DENSE_OF, SLOTS,
           DENSE, RING, RING_DST,
           L: tl.constexpr, B: tl.constexpr, H: tl.constexpr,
           K: tl.constexpr, V: tl.constexpr, RMAX: tl.constexpr,
           N: tl.constexpr, NR: tl.constexpr,
           A0: tl.constexpr, A1: tl.constexpr, A2: tl.constexpr,
           SLOT_STRIDE: tl.constexpr, R: tl.constexpr,
           STALE_VALUE: tl.constexpr, WRITE_RING: tl.constexpr,
           BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lh = tl.program_id(1)
    layer, head = lh // H, lh % H
    tile = tl.program_id(2)
    slot = tl.load(SLOTS + row * SLOT_STRIDE).to(tl.int64)
    if slot < 0:
        return
    src = row.to(tl.int64) * L * H + lh
    dst = (layer.to(tl.int64) * N + slot) * H + head
    x = tile * BLOCK + tl.arange(0, BLOCK)
    av = tl.load(A + row * A0 + lh * A1 + x * A2, x < K, other=0)
    uv = tl.load(U + src * RMAX * K + x, x < RMAX * K, other=0)
    wv = tl.load(W + src * RMAX * V + x, x < RMAX * V, other=0)
    tl.store(FA + dst * K + x, av, x < K)
    tl.store(FU + dst * RMAX * K + x, uv, x < RMAX * K)
    tl.store(FW + dst * RMAX * V + x, wv, x < RMAX * V)
    if tile == 0:
        tl.store(COUNT + dst, R)
        # Shared slot metadata has one writer. The caller publishes validity
        # only after this kernel and any tracked-state scatter have completed.
        if lh == 0:
            tl.store(STALE + slot, STALE_VALUE)
            if STALE_VALUE == 1:
                tl.store(DENSE_OF + slot, -1)
    if WRITE_RING:
        ring_slot = tl.load(RING_DST + row).to(tl.int64)
        if ring_slot >= 0:
            dense_src = ((layer.to(tl.int64) * B + row) * H + head) * V * K
            dense_dst = ((layer.to(tl.int64) * NR + ring_slot) * H + head) * V * K
            dv = tl.load(DENSE + dense_src + x, x < V * K, other=0)
            tl.store(RING + dense_dst + x, dv, x < V * K)


def store_layers(factors, pool, slots, *, stale_value, dense=None, ring_dst=None):
    """Same unique-slot contract as store_factored; negative rows are inert."""
    a, u, w = factors
    layers, n, h, k = pool.a.shape
    b, heads, rmax, uk = u.shape
    v = w.shape[-1]
    assert heads == layers*h and uk == k
    assert a.shape == (b, heads, k) and w.shape == (b, heads, rmax, v)
    assert slots.numel() == b
    assert u.is_contiguous() and w.is_contiguous()
    assert all(t.is_contiguous() for t in (pool.a, pool.U, pool.W, pool.count, pool.dense_ring))
    assert pool.U.shape == (layers, n, h, rmax, k)
    assert pool.W.shape == (layers, n, h, rmax, v)
    assert stale_value in (0, 1)
    write_ring = dense is not None
    if write_ring:
        assert dense.shape == (layers, b, h, v, k) and dense.is_contiguous()
        assert ring_dst.shape == (b,) and ring_dst.is_contiguous()
    if b == 0:
        return
    extent = max(k, rmax*k, rmax*v, v*k if write_ring else 0)
    _store[(b, heads, triton.cdiv(extent, 1024))](
        a, u, w, pool.a, pool.U, pool.W, pool.count, pool.stale, pool.dense_of, slots,
        dense if write_ring else a, pool.dense_ring,
        ring_dst if write_ring else slots,
        layers, b, h, k, v, rmax, n, pool.dense_ring.shape[1],
        *a.stride(), slots.stride(0), pool.cfg.r, stale_value, write_ring,
        1024, num_warps=4)
