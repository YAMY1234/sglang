"""Experimental single-launch online QSA read for residual/bitmap pages.

No service dispatch uses this candidate until mixed-page and model guards pass.
The encoder, selected tokens and exact indexer are unchanged. Each program keeps
one shared softmax and decodes the final weighted value code exactly once.
"""

import triton
import triton.language as tl


def fused_page_attention(
    q,
    slots,
    use_code,
    pool,
    layer,
    workspace=None,
    *,
    scale=None,
    key_block=32,
    value_block=128,
    split_bf16=False,
    token_to_batch=None,
    request_ids=None,
    sequence_lengths=None,
    request_table=None,
    prefix_lengths=None,
):
    """Experimental oracle/benchmark entry; no intermediate read workspace.

    With request metadata, ``slots`` contains logical top-k indices. Otherwise
    it contains virtual token slots with explicit representation roles.
    """
    import torch

    layout = pool.layout
    if not (layout.stored_residuals and layout.value_bitmap):
        raise ValueError("Fused candidate requires residual/bitmap code pages")
    if key_block not in (16, 32, 64) or value_block not in (64, 128, 256):
        raise ValueError("Unsupported fused tile")
    if not q.is_contiguous() or not slots.is_contiguous():
        raise ValueError("Fused inputs must be contiguous")
    rows, hq, hd = q.shape
    if hd != 256 or slots.ndim != 2 or slots.shape[0] != rows:
        raise ValueError("Unexpected QSA query/selection shape")
    hk = pool.head_count
    if hq % hk:
        raise ValueError("Query heads must be divisible by KV heads")
    indexed = request_table is not None
    metadata = (
        token_to_batch,
        request_ids,
        sequence_lengths,
        request_table,
        prefix_lengths,
    )
    if indexed and any(t is None for t in metadata):
        raise ValueError("Indexed reads require all live request metadata")
    if not indexed:
        if (
            use_code is None
            or use_code.shape != slots.shape
            or not use_code.is_contiguous()
        ):
            raise ValueError("Direct reads require contiguous representation roles")
        metadata = (slots,) * 5  # Dead constexpr branch, never dereferenced.
    code = pool.codes[layer]
    kw, vw = pool.weights[layer]
    ek, ev = pool.exact[layer]
    out = torch.empty_like(q)
    _qsa_fused_online[(rows, hk, triton.cdiv(hd, value_block))](
        q,
        slots,
        slots if indexed else use_code,
        pool.code_page,
        pool.exact_page,
        code.rotary,
        code.key.latent,
        code.key.indices,
        code.key.originals,
        ek,
        code.value.latent,
        code.value.indices.view(torch.int32),
        code.value.originals,
        ev,
        kw.decoder,
        kw.mean,
        vw.decoder,
        vw.mean,
        out,
        *metadata,
        pool.code_valid_tokens,
        pool.publish_age,
        INDEXED=indexed,
        TABLE_WIDTH=request_table.shape[1] if indexed else 0,
        TOP=slots.shape[1],
        HQ=hq,
        HK=hk,
        HD=hd,
        ROT=64,
        PS=pool.page_size,
        RK=layout.key_rank,
        RV=layout.value_rank,
        MK=layout.key_sparse,
        MV=layout.value_sparse,
        GROUP=hq // hk,
        BH=triton.next_power_of_2(hq // hk),
        BN=key_block,
        BKR=triton.next_power_of_2(layout.key_rank),
        BVR=triton.next_power_of_2(layout.value_rank),
        BD=value_block,
        SCALE=hd**-0.5 if scale is None else scale,
        SPLIT_BF16=split_bf16,
        num_warps=8,
        num_stages=1,
    )
    return out


@triton.jit
def _popcount(word):
    return tl.inline_asm_elementwise(
        "popc.b32 $0, $1;", "=r,r", [word], dtype=tl.uint32, is_pure=True, pack=1
    )


@triton.jit
def _mixed_dot(a, b, accumulator, SPLIT: tl.constexpr):
    # Three BF16 components retain FP32 coefficients. The stored BF16 operand
    # is already exact in this format; do not compute products of zero tails.
    # This is an experimental arithmetic path, gated by the same dense oracle.
    if SPLIT and a.dtype == tl.float32 and b.dtype == tl.bfloat16:
        hi = a.to(tl.bfloat16)
        rest = a - hi.to(tl.float32)
        mid = rest.to(tl.bfloat16)
        lo = (rest - mid.to(tl.float32)).to(tl.bfloat16)
        accumulator = tl.dot(lo, b, accumulator)
        accumulator = tl.dot(mid, b, accumulator)
        accumulator = tl.dot(hi, b, accumulator)
    elif SPLIT and a.dtype == tl.bfloat16 and b.dtype == tl.float32:
        hi = b.to(tl.bfloat16)
        rest = b - hi.to(tl.float32)
        mid = rest.to(tl.bfloat16)
        lo = (rest - mid.to(tl.float32)).to(tl.bfloat16)
        accumulator = tl.dot(a, lo, accumulator)
        accumulator = tl.dot(a, mid, accumulator)
        accumulator = tl.dot(a, hi, accumulator)
    else:
        if a.dtype != b.dtype:
            a, b = a.to(tl.float32), b.to(tl.float32)
        accumulator = tl.dot(a, b, accumulator, input_precision="tf32x3")
    return accumulator


@triton.jit
def _qsa_fused_online(
    Q,
    Slots,
    UseCode,
    CPage,
    EPage,
    KRot,
    KZ,
    KIndex,
    KCorr,
    ExactK,
    VZ,
    VBits,
    VCorr,
    ExactV,
    KD,
    KMean,
    VD,
    VMean,
    Out,
    TokenBatch,
    RequestIds,
    SeqLengths,
    RequestTable,
    PrefixLengths,
    CodeValid,
    Age,
    INDEXED: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    TOP: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    HD: tl.constexpr,
    ROT: tl.constexpr,
    PS: tl.constexpr,
    RK: tl.constexpr,
    RV: tl.constexpr,
    MK: tl.constexpr,
    MV: tl.constexpr,
    GROUP: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    BKR: tl.constexpr,
    BVR: tl.constexpr,
    BD: tl.constexpr,
    SCALE: tl.constexpr,
    SPLIT_BF16: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    hh = head * GROUP + tl.arange(0, BH)
    hm = hh < (head + 1) * GROUP
    rr = tl.arange(0, BKR)
    d64 = tl.arange(0, 64)
    # Disjoint output-coordinate tiles reduce register pressure. Each value
    # coordinate is decoded once, after its full weighted latent reduction.
    dd = tl.program_id(2) * BD + tl.arange(0, BD)
    vr = tl.arange(0, BVR)
    nn = tl.arange(0, BN)
    qcode = tl.zeros((BH, BKR), tl.float32)
    qmean = tl.zeros((BH,), tl.float32)
    for base in range(0, HD - ROT, 64):
        d = base + d64
        qn = tl.load(
            Q + (row * HQ + hh[:, None]) * HD + ROT + d[None, :],
            hm[:, None] & (d[None, :] < HD - ROT),
            0,
        )
        decoder = tl.load(
            KD + (head * (HD - ROT) + d[:, None]) * RK + rr[None, :],
            (d[:, None] < HD - ROT) & (rr[None, :] < RK),
            0,
        )
        qcode = _mixed_dot(qn, decoder, qcode, SPLIT_BF16)
        mean = tl.load(KMean + head * (HD - ROT) + d, d < HD - ROT, 0)
        qmean += tl.sum(qn.to(tl.float32) * mean[None, :], 1)
    qrot = tl.load(
        Q + (row * HQ + hh[:, None]) * HD + d64[None, :],
        hm[:, None] & (d64[None, :] < ROT),
        0,
    )
    if INDEXED:
        batch = tl.load(TokenBatch + row)
        req = tl.load(RequestIds + batch)
        seq_len = tl.load(SeqLengths + batch)
        prefix = tl.load(PrefixLengths + req)
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    normalizer = tl.zeros((BH,), tl.float32)
    latent = tl.zeros((BH, BVR), tl.float32)
    output = tl.zeros((BH, BD), tl.float32)
    mass = tl.zeros((BH,), tl.float32)
    for base in range(0, TOP, BN):
        token = base + nn
        selected = tl.load(Slots + row * TOP + token, token < TOP, -1)
        if INDEXED:
            logical_valid = (token < TOP) & (selected >= 0) & (selected < seq_len)
            logical_valid &= selected < TABLE_WIDTH
            slot = tl.load(
                RequestTable + req * TABLE_WIDTH + selected, logical_valid, 0
            )
        else:
            slot = selected
        page = tl.maximum(slot, 0) // PS
        cp, ep = tl.load(CPage + page), tl.load(EPage + page)
        if INDEXED:
            age, length = tl.load(Age + page), tl.load(CodeValid + page)
            want = (selected < prefix) | ((ep == 0) & ((age < 0) | (age >= 256)))
            want &= (slot % PS) < length
        else:
            want = tl.load(UseCode + row * TOP + token, token < TOP, 0)
        coded = want & (cp > 0)
        valid = (token < TOP) & (slot > 0) & (coded | (ep > 0))
        cm, em = valid & coded, valid & ~coded
        co, eo = cp * PS + slot % PS, ep * PS + slot % PS
        score = tl.zeros((BH, BN), tl.float32)
        if tl.sum(cm.to(tl.int32), 0) > 0:
            kr = tl.load(
                KRot + (co[None, :] * HK + head) * ROT + d64[:, None],
                cm[None, :] & (d64[:, None] < ROT),
                0,
            ).to(Q.dtype.element_ty)
            score = tl.dot(qrot, kr, score, input_precision="tf32x3")
            zk = tl.load(
                KZ + (co[None, :] * HK + head) * RK + rr[:, None],
                cm[None, :] & (rr[:, None] < RK),
                0,
            )
            score = _mixed_dot(qcode, zk, score, SPLIT_BF16)
            # Two-dimensional indexed gathers; no [heads,tokens,spikes] tile.
            for spike in range(MK):
                ix = tl.load(KIndex + (co * HK + head) * MK + spike, cm, 0).to(tl.int32)
                c = tl.load(KCorr + (co * HK + head) * MK + spike, cm, 0).to(tl.float32)
                qs = tl.load(
                    Q + (row * HQ + hh[:, None]) * HD + ROT + ix[None, :],
                    hm[:, None] & cm[None, :],
                    0,
                ).to(tl.float32)
                score += qs * c[None, :]
            score += qmean[:, None]
        exact_score = tl.zeros((BH, BN), tl.float32)
        if tl.sum(em.to(tl.int32), 0) > 0:
            for base_d in range(0, HD, 64):
                d = base_d + d64
                q = tl.load(
                    Q + (row * HQ + hh[:, None]) * HD + d[None, :],
                    hm[:, None] & (d[None, :] < HD),
                    0,
                )
                k = tl.load(
                    ExactK + (eo[None, :] * HK + head) * HD + d[:, None],
                    em[None, :] & (d[:, None] < HD),
                    0,
                ).to(Q.dtype.element_ty)
                exact_score = tl.dot(q, k, exact_score, input_precision="tf32x3")
        score = tl.where(coded[None, :], score, exact_score) * SCALE
        score = tl.where(valid[None, :], score, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(score, 1))
        safe_max = tl.where(new_max == -float("inf"), 0, new_max)
        alpha = tl.exp(maximum - safe_max)
        probability = tl.exp(score - safe_max[:, None])
        normalizer = normalizer * alpha + tl.sum(probability, 1)
        latent *= alpha[:, None]
        output *= alpha[:, None]
        mass *= alpha
        pc = tl.where(cm[None, :], probability, 0)
        zv = tl.load(
            VZ + (co[:, None] * HK + head) * RV + vr[None, :],
            cm[:, None] & (vr[None, :] < RV),
            0,
        )
        latent = _mixed_dot(pc, zv, latent, SPLIT_BF16)
        mass += tl.sum(pc, 1)
        if tl.sum(em.to(tl.int32), 0) > 0:
            v = tl.load(
                ExactV + (eo[:, None] * HK + head) * HD + dd[None, :],
                em[:, None] & (dd[None, :] < HD),
                0,
            )
            output = _mixed_dot(probability, v, output, SPLIT_BF16)
        if tl.sum(cm.to(tl.int32), 0) > 0:
            word = tl.load(
                VBits + (co[:, None] * HK + head) * 8 + dd[None, :] // 32,
                cm[:, None] & (dd[None, :] < HD),
                0,
            ).to(tl.uint32)
            before = tl.zeros((BN, BD), tl.uint32)
            for word_id in tl.static_range(7):
                previous = tl.load(VBits + (co * HK + head) * 8 + word_id, cm, 0).to(
                    tl.uint32
                )
                before += tl.where(
                    dd[None, :] // 32 > word_id, _popcount(previous)[:, None], 0
                )
            bit = dd[None, :] % 32
            lower = (tl.full((BN, BD), 1, tl.uint32) << bit) - 1
            address = before + _popcount(word & lower)
            present = ((word >> bit) & 1) != 0
            correction = tl.load(
                VCorr + (co[:, None] * HK + head) * MV + address,
                cm[:, None] & present & (address < MV) & (dd[None, :] < HD),
                0,
            )
            output = _mixed_dot(pc, correction, output, SPLIT_BF16)
        maximum = new_max
    denominator = tl.where(normalizer > 0, normalizer, 1)
    latent /= denominator[:, None]
    output /= denominator[:, None]
    mass /= denominator
    decoder = tl.load(
        VD + (head * HD + dd[None, :]) * RV + vr[:, None],
        (dd[None, :] < HD) & (vr[:, None] < RV),
        0,
    )
    output = tl.dot(latent, decoder, output, input_precision="tf32x3")
    mean = tl.load(VMean + head * HD + dd, dd < HD, 0)
    output += mass[:, None] * mean[None, :]
    tl.store(
        Out + (row * HQ + hh[:, None]) * HD + dd[None, :],
        output,
        hm[:, None] & (dd[None, :] < HD),
    )
