"""One launch for a single slot's per-layer prefix snapshot; no deferred publication."""
import triton
import triton.language as tl


@triton.jit
def _copy(A, U, W, C, STALE, DENSE, VALID, SRC, DST,
          AS: tl.constexpr, AH: tl.constexpr, AK: tl.constexpr,
          US: tl.constexpr, UH: tl.constexpr, UR: tl.constexpr, UK: tl.constexpr,
          WS: tl.constexpr, WH: tl.constexpr, WR: tl.constexpr, WV: tl.constexpr,
          CS: tl.constexpr, CH: tl.constexpr, SS: tl.constexpr, DS: tl.constexpr,
          VS: tl.constexpr, S: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
          R: tl.constexpr, HAS_VALID: tl.constexpr, FIRST: tl.constexpr,
          LAST: tl.constexpr, B: tl.constexpr):
    h, block = tl.program_id(0), tl.program_id(1)
    raw_s, raw_d = tl.load(SRC).to(tl.int64), tl.load(DST).to(tl.int64)
    s, d = raw_s, raw_d
    s, d = tl.where(s < 0, s + S, s), tl.where(d < 0, d + S, d)
    x = block * B + tl.arange(0, B)
    av = tl.load(A + s*AS + h*AH + x*AK, x < K, other=0)
    uv = tl.load(U + s*US + h*UH + (x//K)*UR + (x%K)*UK, x < R*K, other=0)
    wv = tl.load(W + s*WS + h*WH + (x//V)*WR + (x%V)*WV, x < R*V, other=0)
    tl.store(A + d*AS + h*AH + x*AK, av, x < K)
    tl.store(U + d*US + h*UH + (x//K)*UR + (x%K)*UK, uv, x < R*K)
    tl.store(W + d*WS + h*WH + (x//V)*WR + (x%V)*WV, wv, x < R*V)
    if block == 0:
        count = tl.load(C + s*CS + h*CH)
        tl.store(C + d*CS + h*CH, count)
        if h == 0:
            tl.store(STALE + d*SS, 1)
            tl.store(DENSE + d*DS, -1)
            if HAS_VALID:
                if LAST:
                    valid = tl.load(VALID + s*VS)
                    if FIRST:
                        valid = tl.where((s == d) & (raw_s != raw_d), 0, valid)
                    tl.store(VALID + d*VS, valid)
                elif FIRST:
                    valid = tl.load(VALID + d*VS)
                    tl.store(VALID + d*VS, tl.where(raw_s == raw_d, valid, 0))


def copy_layer(pool, layer_id, src, dst):
    li = pool.layer_map[layer_id]
    if li >= pool.prefix_layer_count():
        return
    if src.numel() != 1 or dst.numel() != 1 or pool.prefix_dense is not None:
        raise ValueError('single slot factored prefix required')
    a, u, w, count = (getattr(pool, key)[li] for key in ('a', 'U', 'W', 'count'))
    valid = pool.prefix_valid
    block = 256
    _copy[(a.shape[1], triton.cdiv(max(a.shape[-1], u.shape[-2]*u.shape[-1],
                                     w.shape[-2]*w.shape[-1]), block))](
        a, u, w, count, pool.stale, pool.dense_of, valid if valid is not None else pool.stale,
        src, dst, *a.stride(), *u.stride(), *w.stride(), *count.stride(),
        pool.stale.stride(0), pool.dense_of.stride(0), valid.stride(0) if valid is not None else 1,
        a.shape[0], a.shape[-1], w.shape[-1], u.shape[-2], valid is not None,
        li == 0, li == pool.prefix_layer_count()-1, block, num_warps=4)
