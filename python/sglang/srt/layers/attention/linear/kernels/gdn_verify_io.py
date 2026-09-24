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
