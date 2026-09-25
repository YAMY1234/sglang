"""One indexed launch for all layers' r8 verify entry snapshots (flag gated)."""
import triton
import triton.language as tl


@triton.jit
def _snapshot_factors_kernel(A, U, W, C, DA, DU, DW, DC, SLOTS,
                             H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                             R: tl.constexpr, S: tl.constexpr, CAP: tl.constexpr,
                             SLOT_STRIDE: tl.constexpr,
                             BLOCK: tl.constexpr):
    row = tl.program_id(0)
    layer = tl.program_id(1)
    tile = tl.program_id(2)
    slot = tl.load(SLOTS + row*SLOT_STRIDE).to(tl.int64)
    # Commit-graph warmup/capture uses padding only: no pool or working writes.
    if slot < 0:
        return
    src = (layer.to(tl.int64) * S + slot) * H
    dst = (layer.to(tl.int64) * CAP + row) * H
    x = tile * BLOCK + tl.arange(0, BLOCK)
    av = tl.load(A + src*K + x, x < H*K, 0)
    uv = tl.load(U + src*R*K + x, x < H*R*K, 0)
    wv = tl.load(W + src*R*V + x, x < H*R*V, 0)
    cv = tl.load(C + src + x, x < H, 0)
    tl.store(DA + dst*K + x, av, x < H*K)
    tl.store(DU + dst*R*K + x, uv, x < H*R*K)
    tl.store(DW + dst*R*V + x, wv, x < H*R*V)
    tl.store(DC + dst + x, cv, x < H)


def snapshot_factors(pool, working, slots):
    layers, size, heads, rank, key = pool.U.shape
    value = pool.W.shape[-1]
    capacity = working['U'].shape[1]
    extent = heads * max(key, rank*key, rank*value)
    _snapshot_factors_kernel[(slots.numel(), layers, triton.cdiv(extent, 1024))](
        pool.a, pool.U, pool.W, pool.count,
        working['a'], working['U'], working['W'], working['count'], slots,
        heads, key, value, rank, size, capacity, slots.stride(0), 1024, num_warps=4)


@triton.jit
def _publish_factors_kernel(A, U, W, C, DA, DU, DW, DC, SLOTS, VALID,
                            H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                            R: tl.constexpr, S: tl.constexpr, CAP: tl.constexpr,
                            SLOT_STRIDE: tl.constexpr, VALID_STRIDE: tl.constexpr,
                            BLOCK: tl.constexpr,
                            STALE=None, DENSE_OF=None, DENSE_REQUIRED=None, PREFIX_VALID=None,
                            STEPS_AND_METADATA: tl.constexpr = False):
    row = tl.program_id(0)
    layer = tl.program_id(1).to(tl.int64)
    tile = tl.program_id(2)
    slot = tl.load(SLOTS + row * SLOT_STRIDE).to(tl.int64)
    valid = tl.load(VALID + row * VALID_STRIDE)
    if STEPS_AND_METADATA:
        valid = valid >= 0
    if slot < 0 or not valid:
        return
    src = (layer * CAP + row) * H
    dst = (layer * S + slot) * H
    x = tile * BLOCK + tl.arange(0, BLOCK)
    av = tl.load(A + src*K + x, x < H*K, 0)
    uv = tl.load(U + src*R*K + x, x < H*R*K, 0)
    wv = tl.load(W + src*R*V + x, x < H*R*V, 0)
    cv = tl.load(C + src + x, x < H, 0)
    tl.store(DA + dst*K + x, av, x < H*K)
    tl.store(DU + dst*R*K + x, uv, x < H*R*K)
    tl.store(DW + dst*R*V + x, wv, x < H*R*V)
    tl.store(DC + dst + x, cv, x < H)
    if STEPS_AND_METADATA:
        if layer == 0 and tile == 0:
            if STALE is not None:
                tl.store(STALE + slot, 1)
            if DENSE_OF is not None:
                tl.store(DENSE_OF + slot, -1)
            if DENSE_REQUIRED is not None:
                tl.store(DENSE_REQUIRED + slot, 0)
            if PREFIX_VALID is not None:
                tl.store(PREFIX_VALID + slot, 0)


def publish_factors(pool, working, slots, valid, *, steps_and_metadata=False):
    """Publish all four factor arrays and all layers in one indexed launch."""
    layers, size, heads, rank, key = pool.U.shape
    value = pool.W.shape[-1]
    capacity = working['U'].shape[1]
    extent = heads * max(key, rank*key, rank*value)
    _publish_factors_kernel[(slots.numel(), layers, triton.cdiv(extent, 1024))](
        working['a'], working['U'], working['W'], working['count'],
        pool.a, pool.U, pool.W, pool.count, slots, valid,
        heads, key, value, rank, size, capacity, slots.stride(0), valid.stride(0),
        1024, pool.stale if steps_and_metadata else None,
        pool.dense_of if steps_and_metadata else None,
        pool.dense_required if steps_and_metadata else None,
        pool.prefix_valid if steps_and_metadata else None,
        steps_and_metadata, num_warps=4)
