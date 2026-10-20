"""Target-verify sampling on vocab shards for chain (topk=1) drafts under TP.

The verify logits stay on each rank's vocab shard instead of being all-gathered.
Every rank reduces its shard per row (max, sum-exp at the request temperature,
argmax with torch.argmax ties/NaN, and the logit of the row's next draft token),
one small all-reduce of slot-packed rows replaces the logits all-gather, and all
ranks then run the same chain acceptance with rank 0's coins. The final draw
picks a shard and a chunk by vocab-order mass; that shard's owner scans the
chunk and a second small all-reduce returns the token. Greedy batches take only
the argmax. The argmax equals torch.argmax over the gathered logits; sampled
tokens follow the same distribution as tree_speculative_sampling_target_only.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel, get_spec

logger = logging.getLogger(__name__)

ENABLED = envs.SGLANG_Q35MF_PDDEC_SHARDED_VERIFY_SAMPLING.get()

_CHUNK = 4096
# Per-row payload slots: max, sum-exp, next-draft-token logit, argmax key (hi, lo), argmax id.
_P = 6
_KEY_NAN = 1 << 32
_ID_NONE = 1 << 30


@triton.jit
def _order_key(x):
    # Monotonic integer key of a float32 (int64); -0.0 == +0.0, NaN above every value.
    u = x.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    key = tl.where(u >= 0x80000000, 0xFFFFFFFF - u, u + 0x80000000)
    key = tl.where(x == 0.0, 0x80000000, key)
    return tl.where(x != x, 0x100000000, key)


@triton.jit
def _shard_stats_kernel(
    logits_ptr,
    stride_row,
    temps_ptr,
    local_vocab,
    nchunk,
    out_m_ptr,
    out_z_ptr,
    out_key_ptr,
    out_idx_ptr,
    D: tl.constexpr,
    CHUNK: tl.constexpr,
):
    r = tl.program_id(0)
    c = tl.program_id(1)
    t = tl.load(temps_ptr + r // D)
    inv_t = tl.where(t > 0, 1.0 / t, 1.0)
    offs = c * CHUNK + tl.arange(0, CHUNK)
    mask = offs < local_vocab
    x = tl.load(logits_ptr + r * stride_row + offs, mask=mask, other=float("-inf")).to(
        tl.float32
    )
    key = tl.where(mask, _order_key(x), -1)
    kmax = tl.max(key, axis=0)
    idx = tl.min(tl.where(key == kmax, offs, 1 << 30), axis=0)
    xs = tl.where(x != x, float("-inf"), x)
    m = tl.max(xs, axis=0)
    finite = xs > float("-inf")
    z = tl.sum(
        tl.where(
            finite, tl.exp((xs - tl.where(m > float("-inf"), m, 0.0)) * inv_t), 0.0
        ),
        axis=0,
    )
    o = r * nchunk + c
    tl.store(out_m_ptr + o, m)
    tl.store(out_z_ptr + o, z)
    tl.store(out_key_ptr + o, kmax)
    tl.store(out_idx_ptr + o, idx)


@triton.jit
def _pack_kernel(
    m_ptr,
    z_ptr,
    key_ptr,
    idx_ptr,
    nchunk,
    logits_ptr,
    stride_row,
    cand_ptr,
    temps_ptr,
    coins_ptr,
    coins_final_ptr,
    vocab_start,
    local_vocab,
    rank,
    buf_ptr,
    slot_stride,
    coin_off,
    D: tl.constexpr,
    DP2: tl.constexpr,
    NCHUNK_P2: tl.constexpr,
    P: tl.constexpr,
):
    r = tl.program_id(0)
    b = r // D
    j = r % D
    t = tl.load(temps_ptr + b)
    inv_t = tl.where(t > 0, 1.0 / t, 1.0)
    cs = tl.arange(0, NCHUNK_P2)
    cm = cs < nchunk
    mc = tl.load(m_ptr + r * nchunk + cs, mask=cm, other=float("-inf"))
    zc = tl.load(z_ptr + r * nchunk + cs, mask=cm, other=0.0)
    kc = tl.load(key_ptr + r * nchunk + cs, mask=cm, other=-1)
    ic = tl.load(idx_ptr + r * nchunk + cs, mask=cm, other=1 << 30)
    m = tl.max(mc, axis=0)
    m_safe = tl.where(m > float("-inf"), m, 0.0)
    s = tl.sum(
        tl.where(mc > float("-inf"), zc * tl.exp((mc - m_safe) * inv_t), 0.0), axis=0
    )
    kmax = tl.max(kc, axis=0)
    idx = tl.min(tl.where(kc == kmax, ic, 1 << 30), axis=0) + vocab_start
    # Logit of the draft token this row verifies (row j checks candidates[b, j + 1]).
    has_next = j < D - 1
    d = tl.load(cand_ptr + b * D + j + 1, mask=has_next, other=-1)
    local = d - vocab_start
    owned = has_next & (local >= 0) & (local < local_vocab)
    xd = tl.load(
        logits_ptr + r * stride_row + local, mask=owned, other=float("-inf")
    ).to(tl.float32)
    xd = tl.where(xd != xd, float("-inf"), xd)
    base = buf_ptr + rank * slot_stride + r * P
    tl.store(base + 0, m)
    tl.store(base + 1, s)
    tl.store(base + 2, xd)
    tl.store(base + 3, (kmax >> 16).to(tl.float32))
    tl.store(base + 4, (kmax & 0xFFFF).to(tl.float32))
    tl.store(base + 5, idx.to(tl.float32))
    # Rank 0 also ships its coins, so every rank decides with the same ones.
    if (rank == 0) & (j == 0):
        ks = tl.arange(0, DP2)
        coins = tl.load(coins_ptr + b * D + ks, mask=ks < D, other=0.0)
        tl.store(buf_ptr + coin_off + b * (D + 1) + ks, coins, mask=ks < D)
        tl.store(buf_ptr + coin_off + b * (D + 1) + D, tl.load(coins_final_ptr + b))


@triton.jit
def _decide_kernel(
    buf_ptr,
    slot_stride,
    coin_off,
    temps_ptr,
    cand_ptr,
    ret_index_ptr,
    predict_ptr,
    accept_index_ptr,
    num_correct_ptr,
    argmax_ptr,
    sel_ptr,
    local_vocab,
    W,
    D: tl.constexpr,
    WORLD: tl.constexpr,
    P: tl.constexpr,
    SAMPLE: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.load(temps_ptr + b)
    inv_t = tl.where(t > 0, 1.0 / t, 1.0)
    # Global argmax per row: higher key wins, ties go to the lower rank (lower ids).
    for j in tl.static_range(D):
        r = b * D + j
        best_key = tl.full((), -1, tl.int64)
        best_id = tl.full((), 0, tl.int64)
        for w in tl.static_range(WORLD):
            base = buf_ptr + w * slot_stride + r * P
            key = tl.load(base + 3).to(tl.int64) * 65536 + tl.load(base + 4).to(
                tl.int64
            )
            take = key > best_key
            best_key = tl.where(take, key, best_key)
            best_id = tl.where(take, tl.load(base + 5).to(tl.int64), best_id)
        tl.store(argmax_ptr + r, best_id)
    if SAMPLE:
        coin = tl.load(buf_ptr + coin_off + b * (D + 1))
        last_ret = tl.load(ret_index_ptr + b * D).to(tl.int32)
        tl.store(accept_index_ptr + b * W, last_ret)
        accepted = tl.full((), 0, tl.int32)
        cur_row = tl.full((), 0, tl.int32)
        alive = tl.full((), 1, tl.int1)
        excl = tl.full((), -1, tl.int64)
        q_excl = tl.full((), 0.0, tl.float32)
        for j in tl.static_range(1, D):
            if j < W:
                # Row j - 1 holds the probability of draft j while the chain is alive.
                r = b * D + j - 1
                m = tl.full((), float("-inf"), tl.float32)
                xd = tl.full((), float("-inf"), tl.float32)
                for w in tl.static_range(WORLD):
                    base = buf_ptr + w * slot_stride + r * P
                    m = tl.maximum(m, tl.load(base + 0))
                    xd = tl.maximum(xd, tl.load(base + 2))
                m_safe = tl.where(m > float("-inf"), m, 0.0)
                s = tl.full((), 0.0, tl.float32)
                for w in tl.static_range(WORLD):
                    base = buf_ptr + w * slot_stride + r * P
                    mw = tl.load(base + 0)
                    s += tl.where(
                        mw > float("-inf"),
                        tl.load(base + 1) * tl.exp((mw - m_safe) * inv_t),
                        0.0,
                    )
                q = tl.where(xd > float("-inf"), tl.exp((xd - m_safe) * inv_t) / s, 0.0)
                d = tl.load(cand_ptr + b * D + j)
                acc = alive & (q > 0) & ((coin < q) | (q >= 1.0))
                rej = alive & (~acc)
                tl.store(predict_ptr + last_ret, d.to(tl.int32), mask=acc)
                ret_j = tl.load(ret_index_ptr + b * D + j).to(tl.int32)
                accepted = tl.where(acc, accepted + 1, accepted)
                tl.store(accept_index_ptr + b * W + accepted, ret_j, mask=acc)
                last_ret = tl.where(acc, ret_j, last_ret)
                cur_row = tl.where(acc, j, cur_row)
                coin = tl.where(
                    acc, tl.load(buf_ptr + coin_off + b * (D + 1) + j), coin
                )
                excl = tl.where(rej, d.to(tl.int64), excl)
                q_excl = tl.where(rej, q, q_excl)
                alive = alive & acc
        tl.store(num_correct_ptr + b, accepted)
        # Final draw from the last accepted row, minus the rejected draft's mass.
        r = b * D + cur_row
        m = tl.full((), float("-inf"), tl.float32)
        for w in tl.static_range(WORLD):
            m = tl.maximum(m, tl.load(buf_ptr + w * slot_stride + r * P))
        m_safe = tl.where(m > float("-inf"), m, 0.0)
        s = tl.full((), 0.0, tl.float32)
        for w in tl.static_range(WORLD):
            base = buf_ptr + w * slot_stride + r * P
            mw = tl.load(base + 0)
            s += tl.where(
                mw > float("-inf"),
                tl.load(base + 1) * tl.exp((mw - m_safe) * inv_t),
                0.0,
            )
        excl_owner = tl.where(excl >= 0, excl // local_vocab, -1)
        total = tl.full((), 0.0, tl.float32)
        for w in tl.static_range(WORLD):
            base = buf_ptr + w * slot_stride + r * P
            mw = tl.load(base + 0)
            mass = tl.where(
                mw > float("-inf"),
                tl.load(base + 1) * tl.exp((mw - m_safe) * inv_t) / s,
                0.0,
            )
            mass = tl.maximum(tl.where(excl_owner == w, mass - q_excl, mass), 0.0)
            total += mass
        u = tl.load(buf_ptr + coin_off + b * (D + 1) + D) * total
        cum = tl.full((), 0.0, tl.float32)
        sel = tl.full((), -1, tl.int32)
        u_sel = tl.full((), 0.0, tl.float32)
        last_pos = tl.full((), 0, tl.int32)
        last_mass = tl.full((), 0.0, tl.float32)
        for w in tl.static_range(WORLD):
            base = buf_ptr + w * slot_stride + r * P
            mw = tl.load(base + 0)
            mass = tl.where(
                mw > float("-inf"),
                tl.load(base + 1) * tl.exp((mw - m_safe) * inv_t) / s,
                0.0,
            )
            mass = tl.maximum(tl.where(excl_owner == w, mass - q_excl, mass), 0.0)
            take = (sel < 0) & (cum + mass > u)
            sel = tl.where(take, w, sel)
            u_sel = tl.where(take, u - cum, u_sel)
            cum += mass
            last_pos = tl.where(mass > 0, w, last_pos)
            last_mass = tl.where(mass > 0, mass, last_mass)
        u_sel = tl.where(sel < 0, last_mass, u_sel)
        sel = tl.where(sel < 0, last_pos, sel)
        o = sel_ptr + b * 8
        tl.store(o + 0, sel.to(tl.float32))
        tl.store(o + 1, r.to(tl.float32))
        tl.store(o + 2, u_sel)
        tl.store(o + 3, excl.to(tl.float32))
        tl.store(o + 4, m_safe)
        tl.store(o + 5, s)
        tl.store(o + 6, last_ret.to(tl.float32))


@triton.jit
def _final_draw_kernel(
    sel_ptr,
    logits_ptr,
    stride_row,
    m_ptr,
    z_ptr,
    nchunk,
    temps_ptr,
    vocab_start,
    local_vocab,
    rank,
    tok_ptr,
    D: tl.constexpr,
    CHUNK: tl.constexpr,
    NCHUNK_P2: tl.constexpr,
):
    b = tl.program_id(0)
    o = sel_ptr + b * 8
    owner = tl.load(o + 0).to(tl.int32) == rank
    r = tl.load(o + 1).to(tl.int32)
    u = tl.load(o + 2)
    excl = tl.load(o + 3).to(tl.int64)
    m = tl.load(o + 4)
    s = tl.load(o + 5)
    t = tl.load(temps_ptr + b)
    inv_t = tl.where(t > 0, 1.0 / t, 1.0)
    excl_local = excl - vocab_start
    excl_here = (excl >= 0) & (excl_local >= 0) & (excl_local < local_vocab)
    x_excl = tl.load(
        logits_ptr + r * stride_row + excl_local,
        mask=owner & excl_here,
        other=float("-inf"),
    ).to(tl.float32)
    q_excl = tl.where(
        owner & excl_here & (x_excl == x_excl), tl.exp((x_excl - m) * inv_t) / s, 0.0
    )
    cs = tl.arange(0, NCHUNK_P2)
    cm = owner & (cs < nchunk)
    mc = tl.load(m_ptr + r * nchunk + cs, mask=cm, other=float("-inf"))
    zc = tl.load(z_ptr + r * nchunk + cs, mask=cm, other=0.0)
    mass = tl.where(mc > float("-inf"), zc * tl.exp((mc - m) * inv_t) / s, 0.0)
    mass = tl.where(cs == excl_local // CHUNK, tl.maximum(mass - q_excl, 0.0), mass)
    cum = tl.cumsum(mass, axis=0)
    hit = (cum > u) & cm
    c_sel = tl.min(tl.where(hit, cs, NCHUNK_P2), axis=0)
    c_last = tl.max(tl.where(cm & (mass > 0), cs, -1), axis=0)
    found_c = c_sel < NCHUNK_P2
    c_sel = tl.where(found_c, c_sel, tl.maximum(c_last, 0))
    before = tl.sum(tl.where(cs < c_sel, mass, 0.0), axis=0)
    u2 = tl.where(found_c, u - before, tl.sum(tl.where(cs == c_sel, mass, 0.0), axis=0))
    offs = c_sel * CHUNK + tl.arange(0, CHUNK)
    vm = owner & (offs < local_vocab)
    x = tl.load(logits_ptr + r * stride_row + offs, mask=vm, other=float("-inf")).to(
        tl.float32
    )
    q = tl.where(vm & (x == x) & (x > float("-inf")), tl.exp((x - m) * inv_t) / s, 0.0)
    q = tl.where(excl_here & (offs == excl_local), 0.0, q)
    cq = tl.cumsum(q, axis=0)
    first = tl.min(tl.where(vm & (cq > u2), offs, 1 << 30), axis=0)
    last_valid = tl.max(tl.where(q > 0, offs, -1), axis=0)
    pick = tl.where(
        first < (1 << 30), first, tl.where(last_valid >= 0, last_valid, local_vocab - 1)
    )
    tok = tl.where(owner, (pick + vocab_start).to(tl.float32), 0.0)
    tl.store(tok_ptr + b, tok)


@triton.jit
def _commit_kernel(sel_ptr, tok_ptr, predict_ptr):
    b = tl.program_id(0)
    last_ret = tl.load(sel_ptr + b * 8 + 6).to(tl.int32)
    tl.store(predict_ptr + last_ret, tl.load(tok_ptr + b).to(tl.int32))


# ---------------------------------------------------------------- compact top-k support on shards
# Candidates travel as packed int64 keys: the high half orders the float32 value (larger first, -0.0 equal
# to +0.0, NaN above everything, as torch.argmax ranks it) and the low half the inverted global id, so a
# descending sort puts the lower id first among equal values.
_DB1_CHUNK = 1024
# Candidate buffer per row; rows with more elements at or above the threshold take the exact chunked path.
_DB1_CAP = 256
DB1_MAX_TOP_K = 32
_I64_MIN = tl.constexpr(-9223372036854775807 - 1)


@triton.jit
def _db1_key(x):
    u = x.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    u = tl.where(u == 0x80000000, 0, u)
    key = tl.where(u >= 0x80000000, 0xFFFFFFFF - u, u + 0x80000000)
    return tl.where(x != x, 0xFFFFFFFF, key)


@triton.jit
def _db1_value(key):
    bits = tl.where(key >= 0x80000000, key - 0x80000000, 0xFFFFFFFF - key)
    bits = tl.where(key == 0xFFFFFFFF, 0x7FC00000, bits)
    return bits.to(tl.uint32).to(tl.float32, bitcast=True)


@triton.jit
def _db1_chunk_max_kernel(
    logits_ptr, stride_row, local_vocab, cmax_ptr, nchunk, cnt_ptr, CH: tl.constexpr
):
    r = tl.program_id(0)
    c = tl.program_id(1)
    offs = c * CH + tl.arange(0, CH)
    m = offs < local_vocab
    x = tl.load(logits_ptr + r * stride_row + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(cmax_ptr + r * nchunk + c, tl.max(tl.where(m, _db1_key(x), -1), axis=0))
    if c == 0:
        tl.store(cnt_ptr + r, 0)


@triton.jit
def _db1_compact_kernel(
    logits_ptr,
    stride_row,
    local_vocab,
    vocab_start,
    cmax_ptr,
    nchunk,
    cnt_ptr,
    cand_ptr,
    K: tl.constexpr,
    CAP: tl.constexpr,
    CH: tl.constexpr,
    BN: tl.constexpr,
):
    # Threshold = the k-th largest chunk maximum: at least k elements of the row reach it, so the row's
    # top-k (with every element equal to the k-th value) lies at or above it. Fewer than k chunks: none.
    r = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.arange(0, BN)
    v = tl.load(cmax_ptr + r * nchunk + i, mask=i < nchunk, other=-1)
    thr = tl.max(tl.where(i == K - 1, tl.sort(v, 0, descending=True), -1), axis=0)
    offs = c * CH + tl.arange(0, CH)
    m = offs < local_vocab
    x = tl.load(logits_ptr + r * stride_row + offs, mask=m, other=0.0).to(tl.float32)
    key = _db1_key(x)
    sel = m & (key >= thr)
    n = tl.sum(sel.to(tl.int32), axis=0)
    if n > 0:
        base = tl.atomic_add(cnt_ptr + r, n)
        pos = base + tl.cumsum(sel.to(tl.int32), axis=0) - 1
        packed = ((key - 2147483648) << 32) | (0x7FFFFFFF - (offs + vocab_start).to(tl.int64))
        tl.store(cand_ptr + r * CAP + pos, packed, mask=sel & (pos < CAP))


@triton.jit
def _db1_select_kernel(
    logits_ptr,
    stride_row,
    local_vocab,
    vocab_start,
    nchunk,
    cand_ptr,
    cnt_ptr,
    buf_ptr,
    rank,
    rows,
    WORLD: tl.constexpr,
    K: tl.constexpr,
    KP2: tl.constexpr,
    CAP: tl.constexpr,
    CH: tl.constexpr,
):
    # This rank's top-k of the row into its slot of buf and zeros into the other ranks' slots, so the
    # all-reduce gathers. Rows whose candidates overflow CAP take an exact 8-bit radix select of the k-th
    # key, then the keys above it and the lowest ids equal to it.
    r = tl.program_id(0)
    jj = tl.arange(0, 2 * KP2)
    for w in tl.static_range(WORLD):
        tl.store(
            buf_ptr + (w * rows + r) * K * 2 + jj, 0.0, mask=(jj < 2 * K) & (w != rank)
        )
    base = buf_ptr + (rank * rows + r) * K * 2
    crow = cand_ptr + r * CAP
    n = tl.load(cnt_ptr + r)
    if n <= CAP:
        i = tl.arange(0, CAP)
        x = tl.load(crow + i, mask=i < n, other=_I64_MIN)
        srt = tl.sort(x, 0, descending=True)
        key = (srt >> 32) + 2147483648
        gid = 0x7FFFFFFF - (srt & 0xFFFFFFFF)
        tl.store(base + 2 * i, _db1_value(key), mask=i < K)
        tl.store(base + 2 * i + 1, gid.to(tl.float32), mask=i < K)
    else:
        row = logits_ptr + r * stride_row
        o = tl.arange(0, CH)
        b = tl.arange(0, 256)
        prefix = (r * 0).to(tl.int64)
        need = r * 0 + K
        for s in tl.static_range(4):
            shift = 24 - 8 * s
            hist = tl.zeros([256], dtype=tl.int32)
            miss = r * 0
            for c in range(0, nchunk):
                offs = c * CH + o
                m = offs < local_vocab
                key = _db1_key(tl.load(row + offs, mask=m, other=0.0).to(tl.float32))
                match = m & ((key >> (shift + 8)) == (prefix >> (shift + 8)))
                dig = ((key >> shift) & 255).to(tl.int32)
                hist += tl.histogram(tl.where(match, dig, 0), 256)
                miss += tl.sum((match == 0).to(tl.int32), axis=0)
            hist = tl.where(b == 0, hist - miss, hist)
            cum_ge = tl.sum(hist, axis=0) - (tl.cumsum(hist, axis=0) - hist)
            d = tl.max(tl.where(cum_ge >= need, b, -1), axis=0)
            need = need - tl.sum(tl.where(b > d, hist, 0), axis=0)
            prefix = prefix | (d.to(tl.int64) << shift)
        ngt = K - need
        cg = r * 0
        ce = r * 0
        for c in range(0, nchunk):
            offs = c * CH + o
            m = offs < local_vocab
            key = _db1_key(tl.load(row + offs, mask=m, other=0.0).to(tl.float32))
            packed = ((key - 2147483648) << 32) | (0x7FFFFFFF - (offs + vocab_start).to(tl.int64))
            gt = m & (key > prefix)
            eq = m & (key == prefix)
            pg = cg + tl.cumsum(gt.to(tl.int32), axis=0) - 1
            pe = ce + tl.cumsum(eq.to(tl.int32), axis=0) - 1
            tl.store(crow + pg, packed, mask=gt)
            tl.store(crow + ngt + pe, packed, mask=eq & (pe < need))
            cg += tl.sum(gt.to(tl.int32), axis=0)
            ce += tl.sum(eq.to(tl.int32), axis=0)
        tl.debug_barrier()
        i = tl.arange(0, KP2)
        x = tl.load(crow + i, mask=i < K, other=_I64_MIN)
        srt = tl.sort(x, 0, descending=True)
        key = (srt >> 32) + 2147483648
        gid = 0x7FFFFFFF - (srt & 0xFFFFFFFF)
        tl.store(base + 2 * i, _db1_value(key), mask=i < K)
        tl.store(base + 2 * i + 1, gid.to(tl.float32), mask=i < K)


@triton.jit
def _db1_global_merge_kernel(
    buf_ptr,
    rows,
    out_val_ptr,
    out_id_ptr,
    out_arg_ptr,
    WORLD: tl.constexpr,
    K: tl.constexpr,
    BW: tl.constexpr,
):
    r = tl.program_id(0)
    i = tl.arange(0, BW)
    valid = i < WORLD * K
    off = (((i // K) * rows + r) * K + i % K) * 2
    v = tl.load(buf_ptr + off, mask=valid, other=0.0)
    g = tl.load(buf_ptr + off + 1, mask=valid, other=0.0).to(tl.int64)
    packed = ((_db1_key(v) - 2147483648) << 32) | (0x7FFFFFFF - g)
    packed = tl.where(valid, packed, _I64_MIN)
    srt = tl.sort(packed, 0, descending=True)
    key = (srt >> 32) + 2147483648
    gid = 0x7FFFFFFF - (srt & 0xFFFFFFFF)
    m = i < K
    tl.store(out_val_ptr + r * K + i, _db1_value(key), mask=m)
    tl.store(out_id_ptr + r * K + i, gid, mask=m)
    tl.store(out_arg_ptr + r, tl.max(tl.where(i == 0, gid, -1), axis=0))


def _next_pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def sample_chain_full_vocab(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    coins: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """The sharded chain sampler on one rank that holds the whole vocabulary (no collectives).

    Writes the sampled chain into predict / accept_index / num_correct_drafts and returns the
    argmax per row [R] (int64), equal to torch.argmax(logits, dim=-1).
    """
    rows, vocab = logits.shape
    bs, d = candidates.shape
    dev = logits.device
    nchunk = triton.cdiv(vocab, _CHUNK)
    nchunk_p2 = _next_pow2(nchunk)
    temps = temperatures.reshape(-1).to(torch.float32)
    m = torch.empty((rows, nchunk), dtype=torch.float32, device=dev)
    z = torch.empty_like(m)
    key = torch.empty((rows, nchunk), dtype=torch.int64, device=dev)
    idx = torch.empty((rows, nchunk), dtype=torch.int32, device=dev)
    _shard_stats_kernel[(rows, nchunk)](
        logits,
        logits.stride(0),
        temps,
        vocab,
        nchunk,
        m,
        z,
        key,
        idx,
        D=d,
        CHUNK=_CHUNK,
        num_warps=8,
    )
    # One slot of per-row stats, then the coins; every read entry is written by _pack_kernel.
    slot = rows * _P
    total = slot + bs * (d + 1)
    total += (-total) % 4
    buf = torch.empty((total,), dtype=torch.float32, device=dev)
    _pack_kernel[(rows,)](
        m,
        z,
        key,
        idx,
        nchunk,
        logits,
        logits.stride(0),
        candidates,
        temps,
        coins[0],
        coins[1],
        0,
        vocab,
        0,
        buf,
        slot,
        slot,
        D=d,
        DP2=_next_pow2(d),
        NCHUNK_P2=nchunk_p2,
        P=_P,
    )
    argmax = torch.empty((rows,), dtype=torch.int64, device=dev)
    sel = torch.empty((bs, 8), dtype=torch.float32, device=dev)
    _decide_kernel[(bs,)](
        buf,
        slot,
        slot,
        temps,
        candidates,
        retrieve_index,
        predict,
        accept_index,
        num_correct_drafts,
        argmax,
        sel,
        vocab,
        accept_index.shape[1],
        D=d,
        WORLD=1,
        P=_P,
        SAMPLE=True,
    )
    tok = torch.empty((bs + (-bs) % 4,), dtype=torch.float32, device=dev)
    _final_draw_kernel[(bs,)](
        sel,
        logits,
        logits.stride(0),
        m,
        z,
        nchunk,
        temps,
        0,
        vocab,
        0,
        tok,
        D=d,
        CHUNK=_CHUNK,
        NCHUNK_P2=nchunk_p2,
        num_warps=8,
    )
    _commit_kernel[(bs,)](sel, tok, predict)
    return argmax


class ShardedVerifySampler:
    def __init__(self) -> None:
        parallel = get_parallel()
        self.group = parallel.tp_group
        self.world = parallel.tp_size
        self.rank = parallel.tp_rank
        self.logged = False
        self.rounds = {}
        self._log_every = 2000
        self._next_log = self._log_every

    def eligible_reason(
        self,
        *,
        logits,
        sampling_info,
        verify_input,
        batch,
        grammar_mask,
        watermark_state,
        use_rejection_sampling,
        uno_target_max_top_k,
    ) -> Optional[str]:
        from sglang.srt.speculative.post_verify_graph import PostVerifySamplingView

        # The post-verify graph captures this sampler on purpose; other captures gather.
        if torch.cuda.is_current_stream_capturing() and not isinstance(
            sampling_info, PostVerifySamplingView
        ):
            return "inside a graph capture"
        if logits.shape[-1] * self.world != sampling_info.vocab_size:
            return "shards do not tile the vocabulary"
        if verify_input.tree_topk != 1 or use_rejection_sampling:
            return "not a topk-1 chain"
        spec = get_spec()
        if (
            spec.speculative_accept_threshold_single != 1.0
            or spec.speculative_accept_threshold_acc != 1.0
        ):
            return "acceptance thresholds below 1"
        if grammar_mask is not None or (
            watermark_state is not None and sampling_info.has_watermark_candidates
        ):
            return "grammar or watermark"
        if (
            sampling_info.acc_additive_penalties is not None
            or sampling_info.acc_scaling_penalties is not None
            or sampling_info.logit_bias is not None
            or sampling_info.has_custom_logit_processor
        ):
            return "penalties, bias or custom processor"
        if batch.return_logprob:
            return "logprobs"
        if sampling_info.is_all_greedy:
            return None
        if (
            sampling_info.need_top_k_sampling
            or sampling_info.need_top_p_sampling
            or sampling_info.need_min_p_sampling
            or uno_target_max_top_k is not None
        ):
            return "top-k / top-p / min-p"
        return None

    def run(
        self,
        *,
        logits,
        sampling_info,
        verify_input,
        candidates,
        predict,
        accept_index,
        num_correct_drafts,
        coins: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Return the global argmax per row [R] (int64); with coins, also write the sampled
        chain into predict / accept_index / num_correct_drafts."""
        rows, local_vocab = logits.shape
        bs, d = candidates.shape
        dev = logits.device
        nchunk = triton.cdiv(local_vocab, _CHUNK)
        nchunk_p2 = _next_pow2(nchunk)
        temps = sampling_info.temperatures.reshape(-1).to(torch.float32)
        m = torch.empty((rows, nchunk), dtype=torch.float32, device=dev)
        z = torch.empty_like(m)
        key = torch.empty((rows, nchunk), dtype=torch.int64, device=dev)
        idx = torch.empty((rows, nchunk), dtype=torch.int32, device=dev)
        _shard_stats_kernel[(rows, nchunk)](
            logits,
            logits.stride(0),
            temps,
            local_vocab,
            nchunk,
            m,
            z,
            key,
            idx,
            D=d,
            CHUNK=_CHUNK,
            num_warps=8,
        )
        slot = rows * _P
        coin_off = slot * self.world
        total = coin_off + bs * (d + 1)
        total += (-total) % 4
        buf = torch.zeros((total,), dtype=torch.float32, device=dev)
        sample = coins is not None
        if coins is None:
            coins = (
                torch.zeros((bs, d), dtype=torch.float32, device=dev),
                torch.zeros((bs,), dtype=torch.float32, device=dev),
            )
        _pack_kernel[(rows,)](
            m,
            z,
            key,
            idx,
            nchunk,
            logits,
            logits.stride(0),
            candidates,
            temps,
            coins[0],
            coins[1],
            self.rank * local_vocab,
            local_vocab,
            self.rank,
            buf,
            slot,
            coin_off,
            D=d,
            DP2=_next_pow2(d),
            NCHUNK_P2=nchunk_p2,
            P=_P,
        )
        buf = tensor_model_parallel_all_reduce(buf)
        argmax = torch.empty((rows,), dtype=torch.int64, device=dev)
        sel = torch.empty((bs, 8), dtype=torch.float32, device=dev)
        _decide_kernel[(bs,)](
            buf,
            slot,
            coin_off,
            temps,
            candidates,
            verify_input.retrieve_index,
            predict,
            accept_index,
            num_correct_drafts,
            argmax,
            sel,
            local_vocab,
            accept_index.shape[1],
            D=d,
            WORLD=self.world,
            P=_P,
            SAMPLE=sample,
        )
        if sample:
            tok = torch.zeros((bs + (-bs) % 4,), dtype=torch.float32, device=dev)
            _final_draw_kernel[(bs,)](
                sel,
                logits,
                logits.stride(0),
                m,
                z,
                nchunk,
                temps,
                self.rank * local_vocab,
                local_vocab,
                self.rank,
                tok,
                D=d,
                CHUNK=_CHUNK,
                NCHUNK_P2=nchunk_p2,
                num_warps=8,
            )
            tok = tensor_model_parallel_all_reduce(tok)
            _commit_kernel[(bs,)](sel, tok, predict)
        return argmax

    def db1_reason(
        self,
        *,
        logits,
        sampling_info,
        verify_input,
        batch,
        grammar_mask,
        watermark_state,
        use_rejection_sampling,
        max_top_k,
    ) -> Optional[str]:
        """Why a sampled verify cannot build its compact top-k support on the shards (None: it can)."""
        if max_top_k is None or max_top_k > DB1_MAX_TOP_K:
            return "no compact top-k support"
        if sampling_info.is_all_greedy:
            return "greedy"
        if sampling_info.need_min_p_sampling:
            return "min-p"
        reason = self.eligible_reason(
            logits=logits,
            sampling_info=sampling_info,
            verify_input=verify_input,
            batch=batch,
            grammar_mask=grammar_mask,
            watermark_state=watermark_state,
            use_rejection_sampling=use_rejection_sampling,
            uno_target_max_top_k=None,
        )
        return None if reason == "top-k / top-p / min-p" else reason

    def sparse_support(
        self, *, logits: torch.Tensor, max_top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Global top-k of every row from shard-local logits [R, V / world], identical on every rank:
        (ids [R, k] int64, float32 values [R, k], argmax [R] int64), larger value first and the lower id
        among equals; the values equal the gathered float32 logits, the argmax equals torch.argmax."""
        rows, local_vocab = logits.shape
        dev = logits.device
        k = int(max_top_k)
        nchunk = triton.cdiv(local_vocab, _DB1_CHUNK)
        vocab_start = self.rank * local_vocab
        cmax = torch.empty((rows, nchunk), dtype=torch.int64, device=dev)
        cnt = torch.empty((rows,), dtype=torch.int32, device=dev)
        cand = torch.empty((rows, _DB1_CAP), dtype=torch.int64, device=dev)
        buf = torch.empty((self.world * rows * k * 2,), dtype=torch.float32, device=dev)
        _db1_chunk_max_kernel[(rows, nchunk)](
            logits, logits.stride(0), local_vocab, cmax, nchunk, cnt, CH=_DB1_CHUNK, num_warps=4
        )
        _db1_compact_kernel[(rows, nchunk)](
            logits,
            logits.stride(0),
            local_vocab,
            vocab_start,
            cmax,
            nchunk,
            cnt,
            cand,
            K=k,
            CAP=_DB1_CAP,
            CH=_DB1_CHUNK,
            BN=_next_pow2(nchunk),
            num_warps=4,
        )
        _db1_select_kernel[(rows,)](
            logits,
            logits.stride(0),
            local_vocab,
            vocab_start,
            nchunk,
            cand,
            cnt,
            buf,
            self.rank,
            rows,
            WORLD=self.world,
            K=k,
            KP2=_next_pow2(k),
            CAP=_DB1_CAP,
            CH=_DB1_CHUNK,
            num_warps=4,
        )
        if self.world > 1:
            buf = tensor_model_parallel_all_reduce(buf)
        vals = torch.empty((rows, k), dtype=torch.float32, device=dev)
        ids = torch.empty((rows, k), dtype=torch.int64, device=dev)
        argmax = torch.empty((rows,), dtype=torch.int64, device=dev)
        _db1_global_merge_kernel[(rows,)](
            buf,
            rows,
            vals,
            ids,
            argmax,
            WORLD=self.world,
            K=k,
            BW=_next_pow2(self.world * k),
            num_warps=4,
        )
        return ids, vals, argmax

    def count(self, kind: str, bs: int) -> None:
        if not self.logged and kind != "gathered":
            self.logged = True
            logger.info(
                "Sharded verify sampling active (tp=%d, rank=%d).",
                self.world,
                self.rank,
            )
        key = f"{kind}@bs{bs}"
        self.rounds[key] = self.rounds.get(key, 0) + 1
        n = sum(self.rounds.values())
        if n >= self._next_log:
            self._next_log += self._log_every
            logger.info(
                "Sharded verify sampling rounds: %s", dict(sorted(self.rounds.items()))
            )


_SAMPLER: Optional[ShardedVerifySampler] = None


def get_sampler() -> ShardedVerifySampler:
    global _SAMPLER
    if _SAMPLER is None:
        _SAMPLER = ShardedVerifySampler()
    return _SAMPLER
