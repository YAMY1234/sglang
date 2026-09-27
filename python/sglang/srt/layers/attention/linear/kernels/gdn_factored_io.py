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


@triton.jit
def _store_factored_layers_kernel(
    A, U, W, FA, FU, FW, COUNT, STALE, DENSE_OF, SLOTS,
    DENSE, RING, RING_DST,
    B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    RMAX: tl.constexpr, CAPACITY: tl.constexpr, RINGS: tl.constexpr,
    A0: tl.constexpr, A1: tl.constexpr, A2: tl.constexpr,
    U0: tl.constexpr, W0: tl.constexpr, SLOT_STRIDE: tl.constexpr,
    R: tl.constexpr, STALE_VALUE: tl.constexpr, WRITE_RING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    layer_row = tl.program_id(0)
    layer, row = layer_row // B, layer_row % B
    head, tile = tl.program_id(1), tl.program_id(2)
    slot = tl.load(SLOTS + row * SLOT_STRIDE).to(tl.int64)
    if slot < 0:
        return
    dst = (layer.to(tl.int64) * CAPACITY + slot) * H + head
    x = tile * BLOCK + tl.arange(0, BLOCK)
    av = tl.load(A + row*A0 + (layer*H+head)*A1 + x*A2, x < K, other=0)
    uv = tl.load(U + row*U0 + (layer*H+head)*RMAX*K + x, x < RMAX*K, other=0)
    wv = tl.load(W + row*W0 + (layer*H+head)*RMAX*V + x, x < RMAX*V, other=0)
    tl.store(FA + dst*K + x, av, x < K)
    tl.store(FU + dst*RMAX*K + x, uv, x < RMAX*K)
    tl.store(FW + dst*RMAX*V + x, wv, x < RMAX*V)
    if tile == 0:
        tl.store(COUNT + dst, R)
        if head == 0 and layer == 0:
            tl.store(STALE + slot, STALE_VALUE)
            if STALE_VALUE == 1:
                tl.store(DENSE_OF + slot, -1)
    if WRITE_RING:
        ring_slot = tl.load(RING_DST + row).to(tl.int64)
        if ring_slot >= 0:
            src = (layer.to(tl.int64)*B+row)*H+head
            state = tl.load(DENSE + src*V*K + x, x < V*K, other=0)
            dest = (layer.to(tl.int64)*RINGS+ring_slot)*H+head
            tl.store(RING + dest*V*K + x, state, x < V*K)


def store_factored_layers(factors, pool, slots, r, *, stale_value, dense=None, ring_dst=None):
    """Publish factorize_layers' shared backing in one launch, or leave the old path to the caller.

    No packing allocation: independently allocated per-layer outputs are rejected before any write.
    Negative slots and final/tracked publication retain store_factored's contract.
    """
    if not factors or slots.numel() == 0:
        return False
    a, u, w = factors[0]
    layers, batch = len(factors), slots.numel()
    b, h, rank, k = u.shape
    v = w.shape[-1]
    if b != batch or layers != pool.a.shape[0]:
        return False
    for column, first, layer_stride in ((0,a,h*a.stride(1)), (1,u,h*rank*k), (2,w,h*rank*v)):
        backing = first.untyped_storage().data_ptr()
        for li, tensors in enumerate(factors):
            t = tensors[column]
            if (t.shape != first.shape or t.stride() != first.stride()
                    or t.untyped_storage().data_ptr() != backing
                    or t.data_ptr()-first.data_ptr() != li*layer_stride*t.element_size()):
                return False
    if not all(t.is_contiguous() for t in (u,w,pool.a,pool.U,pool.W,pool.count)):
        return False
    if dense is not None and (not dense.is_contiguous() or dense.shape != (layers,b,h,v,k)
                              or ring_dst is None or not ring_dst.is_contiguous()):
        return False
    extent = max(k,rank*k,rank*v,v*k if dense is not None else 0)
    _store_factored_layers_kernel[(layers*b,h,triton.cdiv(extent,1024))](
        a,u,w,pool.a,pool.U,pool.W,pool.count,pool.stale,pool.dense_of,slots,
        dense if dense is not None else a,pool.dense_ring if dense is not None else a,
        ring_dst if dense is not None else slots,b,h,k,v,rank,pool.a.shape[1],
        pool.dense_ring.shape[1] if dense is not None else 0,*a.stride(),u.stride(0),w.stride(0),
        slots.stride(0),r,stale_value,dense is not None,1024,num_warps=4)
    return True
