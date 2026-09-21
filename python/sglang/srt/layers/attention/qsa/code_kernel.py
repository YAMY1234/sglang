"""Absorbed mixed-page QSA read: project Q, decode spikes, reduce codes.

No full key/value reconstruction is allocated. Exact current-turn rows share
the same softmax with code rows. Scratch is shared across layers and sized by
the caller before graph capture.
"""

import torch
import triton
import triton.language as tl


class QSAReadWorkspace:
    def __init__(
        self,
        queries,
        topk,
        query_heads,
        kv_heads,
        layout,
        device,
        *,
        score_block=16,
        value_block=16,
        value_warps=8,
        serial_merge=False,
    ):
        if (
            score_block not in (16, 32, 64)
            or value_block not in (16, 32, 64)
            or value_warps not in (4, 8)
        ):
            raise ValueError("Unsupported QSA tuning launch shape")
        self.score_block, self.value_block = score_block, value_block
        self.value_warps, self.serial_merge = value_warps, serial_merge
        self.queries, self.topk = queries, topk
        self.value_splits = min(16, triton.cdiv(topk, 128))
        self.value_partials = torch.empty(
            (queries, query_heads, self.value_splits, layout.head_dim),
            dtype=torch.float32,
            device=device,
        )
        self.latent_partials = torch.empty(
            (queries, query_heads, self.value_splits, layout.value_rank),
            dtype=torch.float32,
            device=device,
        )
        self.mass_partials = torch.empty(
            (queries, query_heads, self.value_splits),
            dtype=torch.float32,
            device=device,
        )
        self.qcode = torch.empty(
            (queries, query_heads, layout.key_rank), dtype=torch.float32, device=device
        )
        self.qmean = torch.empty(
            (queries, query_heads), dtype=torch.float32, device=device
        )
        correction_queries = 0 if layout.stored_residuals else queries
        self.kcorrection = torch.empty(
            (correction_queries, topk, kv_heads, layout.key_sparse),
            dtype=torch.float32,
            device=device,
        )
        self.vcorrection = torch.empty(
            (correction_queries, topk, kv_heads, layout.value_sparse),
            dtype=torch.float32,
            device=device,
        )
        self.scores = torch.empty(
            (queries, query_heads, topk), dtype=torch.float32, device=device
        )
        self.probabilities = torch.empty_like(self.scores)

    @property
    def nbytes(self):
        return sum(
            t.nbytes
            for t in (
                self.qcode,
                self.qmean,
                self.kcorrection,
                self.vcorrection,
                self.scores,
                self.probabilities,
                self.value_partials,
                self.latent_partials,
                self.mass_partials,
            )
        )


@triton.jit
def _project_q(
    Q,
    D,
    Mean,
    Out,
    MeanOut,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    HD: tl.constexpr,
    ROT: tl.constexpr,
    RK: tl.constexpr,
    GROUP: tl.constexpr,
    BH: tl.constexpr,
    BR: tl.constexpr,
):
    batch, head = tl.program_id(0), tl.program_id(1)
    h = head * GROUP + tl.arange(0, BH)
    rr = tl.arange(0, BR)
    dd = tl.arange(0, 64)
    acc = tl.zeros((BH, BR), tl.float32)
    mean_dot = tl.zeros((BH,), tl.float32)
    for start in range(0, HD - ROT, 64):
        d = start + dd
        q = tl.load(
            Q + (batch * HQ + h[:, None]) * HD + ROT + d[None, :],
            (h[:, None] < (head + 1) * GROUP) & (d[None, :] < HD - ROT),
            0,
        ).to(tl.float32)
        w = tl.load(
            D + (head * (HD - ROT) + d[:, None]) * RK + rr[None, :],
            (d[:, None] < HD - ROT) & (rr[None, :] < RK),
            0,
        )
        acc = tl.dot(q, w, acc, input_precision="tf32x3")
        mean = tl.load(Mean + head * (HD - ROT) + d, d < HD - ROT, 0)
        mean_dot += tl.sum(q * mean[None, :], 1)
    tl.store(
        Out + (batch * HQ + h[:, None]) * RK + rr[None, :],
        acc,
        (h[:, None] < (head + 1) * GROUP) & (rr[None, :] < RK),
    )

    tl.store(MeanOut + batch * HQ + h, mean_dot, h < (head + 1) * GROUP)


@triton.jit
def _spike_correction(
    Slots,
    UseCode,
    CodePage,
    Z,
    Indices,
    Originals,
    Decoder,
    Mean,
    Out,
    NT: tl.constexpr,
    HK: tl.constexpr,
    PS: tl.constexpr,
    R: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BR: tl.constexpr,
    BN: tl.constexpr,
):
    tokens = tl.program_id(0) * BN + tl.arange(0, BN)
    head = tl.program_id(1)
    slots = tl.load(Slots + tokens, tokens < NT, 0)
    want = tl.load(UseCode + tokens, tokens < NT, 0)
    page = tl.load(CodePage + tl.maximum(slots, 0) // PS)
    valid = (tokens < NT) & (slots > 0) & want & (page > 0)
    physical = page * PS + slots % PS
    mm, rr = tl.arange(0, BM), tl.arange(0, BR)
    ix = tl.load(
        Indices + (physical[:, None] * HK + head) * M + mm[None, :],
        valid[:, None] & (mm[None, :] < M),
        0,
    ).to(tl.int32)
    original = tl.load(
        Originals + (physical[:, None] * HK + head) * M + mm[None, :],
        valid[:, None] & (mm[None, :] < M),
        0,
    ).to(tl.float32)
    z = tl.load(
        Z + (physical[:, None] * HK + head) * R + rr[None, :],
        valid[:, None] & (rr[None, :] < R),
        0,
    ).to(tl.float32)
    w = tl.load(
        Decoder + (head * D + ix[:, :, None]) * R + rr[None, None, :],
        valid[:, None, None] & (mm[None, :, None] < M) & (rr[None, None, :] < R),
        0,
    )
    mean = tl.load(Mean + head * D + ix, valid[:, None] & (mm[None, :] < M), 0)
    residual = original - mean - tl.sum(w * z[:, None, :], 2)
    tl.store(
        Out + (tokens[:, None] * HK + head) * M + mm[None, :],
        tl.where(valid[:, None], residual, 0),
        (tokens[:, None] < NT) & (mm[None, :] < M),
    )


@triton.jit
def _scores(
    Q,
    QCode,
    Slots,
    UseCode,
    CPage,
    EPage,
    KRot,
    KZ,
    KIndex,
    KCorr,
    QMean,
    ExactK,
    Out,
    RESIDUALS: tl.constexpr,
    SCALE: tl.constexpr,
    TOP: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    HD: tl.constexpr,
    PS: tl.constexpr,
    ROT: tl.constexpr,
    RK: tl.constexpr,
    MK: tl.constexpr,
    GROUP: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
    BM: tl.constexpr,
):
    batch, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    tt = split * BN + tl.arange(0, BN)
    hh = head * GROUP + tl.arange(0, BH)
    slot = tl.load(Slots + batch * TOP + tt, tt < TOP, 0)
    cp = tl.load(CPage + tl.maximum(slot, 0) // PS)
    ep = tl.load(EPage + tl.maximum(slot, 0) // PS)
    want = tl.load(UseCode + batch * TOP + tt, tt < TOP, 0)
    coded = want & (cp > 0)
    valid = (tt < TOP) & (slot > 0) & (coded | (ep > 0))
    co = cp * PS + slot % PS
    eo = ep * PS + slot % PS
    dd = tl.arange(0, 64)
    score = tl.zeros((BH, BN), tl.float32)
    if tl.sum((valid & coded).to(tl.int32), 0) > 0:
        qrot = tl.load(
            Q + (batch * HQ + hh[:, None]) * HD + dd[None, :],
            (hh[:, None] < (head + 1) * GROUP) & (dd[None, :] < ROT),
            0,
        )
        krot = tl.load(
            KRot + (co[None, :] * HK + head) * ROT + dd[:, None],
            valid[None, :] & coded[None, :] & (dd[:, None] < ROT),
            0,
        ).to(Q.dtype.element_ty)
        score = tl.dot(qrot, krot, score, input_precision="tf32x3")
        rr = tl.arange(0, BR)
        qcode = tl.load(
            QCode + (batch * HQ + hh[:, None]) * RK + rr[None, :],
            (hh[:, None] < (head + 1) * GROUP) & (rr[None, :] < RK),
            0,
        )
        z = tl.load(
            KZ + (co[None, :] * HK + head) * RK + rr[:, None],
            valid[None, :] & coded[None, :] & (rr[:, None] < RK),
            0,
        ).to(tl.float32)
        score = tl.dot(qcode, z, score, input_precision="tf32x3")
        mm = tl.arange(0, BM)
        ix = tl.load(
            KIndex + (co[:, None] * HK + head) * MK + mm[None, :],
            valid[:, None] & coded[:, None] & (mm[None, :] < MK),
            0,
        ).to(tl.int32)
        correction_row = co if RESIDUALS else batch * TOP + tt
        correction = tl.load(
            KCorr + (correction_row[:, None] * HK + head) * MK + mm[None, :],
            valid[:, None] & coded[:, None] & (mm[None, :] < MK),
            0,
        ).to(tl.float32)
        qsparse = tl.load(
            Q + (batch * HQ + hh[:, None, None]) * HD + ROT + ix[None, :, :],
            (hh[:, None, None] < (head + 1) * GROUP)
            & valid[None, :, None]
            & coded[None, :, None]
            & (mm[None, None, :] < MK),
            0,
        ).to(tl.float32)
        mean_dot = tl.load(QMean + batch * HQ + hh, hh < (head + 1) * GROUP, 0)
        score += mean_dot[:, None] + tl.sum(qsparse * correction[None, :, :], 2)
    exact_score = tl.zeros((BH, BN), tl.float32)
    if tl.sum((valid & ~coded).to(tl.int32), 0) > 0:
        for start in range(0, HD, 64):
            d = start + dd
            query = tl.load(
                Q + (batch * HQ + hh[:, None]) * HD + d[None, :],
                (hh[:, None] < (head + 1) * GROUP) & (d[None, :] < HD),
                0,
            )
            ek = tl.load(
                ExactK + (eo[None, :] * HK + head) * HD + d[:, None],
                valid[None, :] & ~coded[None, :] & (d[:, None] < HD),
                0,
            ).to(Q.dtype.element_ty)
            exact_score = tl.dot(query, ek, exact_score, input_precision="tf32x3")
    score = tl.where(coded[None, :], score, exact_score) * SCALE
    tl.store(
        Out + (batch * HQ + hh[:, None]) * TOP + tt[None, :],
        tl.where(valid[None, :], score, -float("inf")),
        (hh[:, None] < (head + 1) * GROUP) & (tt[None, :] < TOP),
    )


@triton.jit
def _popcount(word):
    return tl.inline_asm_elementwise(
        "popc.b32 $0, $1;",
        constraints="=r,r",
        args=[word],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _values(
    P,
    Slots,
    UseCode,
    CPage,
    EPage,
    Z,
    Indices,
    Corrections,
    ExactV,
    Out,
    LatentOut,
    MassOut,
    RESIDUALS: tl.constexpr,
    BITMAP: tl.constexpr,
    TOP: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    HD: tl.constexpr,
    PS: tl.constexpr,
    RV: tl.constexpr,
    MV: tl.constexpr,
    GROUP: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    BR: tl.constexpr,
    BM: tl.constexpr,
    NS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    batch, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hh = head * GROUP + tl.arange(0, BH)
    dd = tl.arange(0, BD)
    nn, rr, mm = tl.arange(0, BN), tl.arange(0, BR), tl.arange(0, BM)
    latent_sum = tl.zeros((BH, BR), tl.float32)
    output = tl.zeros((BH, BD), tl.float32)
    mass = tl.zeros((BH,), tl.float32)
    for start in range(0, CHUNK, BN):
        tt = split * CHUNK + start + nn
        slot = tl.load(Slots + batch * TOP + tt, tt < TOP, 0)
        cp = tl.load(CPage + tl.maximum(slot, 0) // PS)
        ep = tl.load(EPage + tl.maximum(slot, 0) // PS)
        want = tl.load(UseCode + batch * TOP + tt, tt < TOP, 0)
        coded = want & (cp > 0)
        valid = (tt < TOP) & (slot > 0)
        co, eo = cp * PS + slot % PS, ep * PS + slot % PS
        p = tl.load(
            P + (batch * HQ + hh[:, None]) * TOP + tt[None, :],
            (hh[:, None] < (head + 1) * GROUP) & (tt[None, :] < TOP),
            0,
        )
        pc = tl.where(coded[None, :] & valid[None, :], p, 0)
        z = tl.load(
            Z + (co[:, None] * HK + head) * RV + rr[None, :],
            valid[:, None] & coded[:, None] & (rr[None, :] < RV),
            0,
        ).to(tl.float32)
        latent_sum = tl.dot(pc, z, latent_sum, input_precision="tf32x3")
        mass += tl.sum(pc, 1)
        if tl.sum((valid & ~coded & (ep > 0)).to(tl.int32), 0) > 0:
            exact = tl.load(
                ExactV + (eo[:, None] * HK + head) * HD + dd[None, :],
                valid[:, None]
                & ~coded[:, None]
                & (ep[:, None] > 0)
                & (dd[None, :] < HD),
                0,
            ).to(tl.float32)
            output = tl.dot(p, exact, output, input_precision="tf32x3")
        if BITMAP:
            # Direct word loads avoid a cross-warp 8->256 gather, which
            # this deployment's Triton compiler cannot lower. Repeated word
            # addresses are cache-coalesced; only eight bitmap words exist.
            word = tl.load(
                Indices + (co[:, None] * HK + head) * 8 + dd[None, :] // 32,
                valid[:, None] & coded[:, None],
                0,
            ).to(tl.uint32)
            before_word = tl.zeros((BN, BD), tl.uint32)
            for word_id in tl.static_range(7):
                previous = tl.load(
                    Indices + (co * HK + head) * 8 + word_id,
                    valid & coded,
                    0,
                ).to(tl.uint32)
                before_word += tl.where(
                    dd[None, :] // 32 > word_id, _popcount(previous)[:, None], 0
                )
            bit = dd[None, :] % 32
            lower_mask = (tl.full((BN, BD), 1, tl.uint32) << bit) - 1
            address = before_word + _popcount(word & lower_mask)
            present = ((word >> bit) & 1) != 0
            sparse_tile = tl.load(
                Corrections + (co[:, None] * HK + head) * MV + address,
                valid[:, None] & coded[:, None] & present & (address < MV),
                0,
            ).to(tl.float32)
        else:
            ix = tl.load(
                Indices + (co[:, None] * HK + head) * MV + mm[None, :],
                valid[:, None] & coded[:, None] & (mm[None, :] < MV),
                0,
            ).to(tl.int32)
            correction_row = co if RESIDUALS else batch * TOP + tt
            correction = tl.load(
                Corrections + (correction_row[:, None] * HK + head) * MV + mm[None, :],
                valid[:, None] & coded[:, None] & (mm[None, :] < MV),
                0,
            ).to(tl.float32)
            # Only sparse residuals become an on-chip tile. V itself is never decoded.
            # Topk indices are unique. Invert their sorted address map with a
            # binary search, avoiding a [tokens, spikes, dimensions] broadcast.
            # Packed positions carry the corresponding residual through the sort.
            packed = tl.sort((ix << 5) + mm[None, :], dim=1, descending=False)
            sorted_ix = packed >> 5
            sorted_c = tl.gather(correction, packed & 31, axis=1)
            lo = tl.full((BN, BD), 0, tl.int32)
            hi = tl.full((BN, BD), MV, tl.int32)
            for _ in tl.static_range(6):
                mid = (lo + hi) // 2
                value = tl.gather(sorted_ix, tl.minimum(mid, MV - 1), axis=1)
                left = (mid < MV) & (value < dd[None, :])
                lo = tl.where(left, mid + 1, lo)
                hi = tl.where(left, hi, mid)
            found = tl.gather(sorted_ix, tl.minimum(lo, MV - 1), axis=1)
            sparse_tile = tl.where(
                (lo < MV) & (found == dd[None, :]) & valid[:, None] & coded[:, None],
                tl.gather(sorted_c, tl.minimum(lo, MV - 1), axis=1),
                0,
            )
        output = tl.dot(pc, sparse_tile, output, input_precision="tf32x3")
    tl.store(
        LatentOut + ((batch * HQ + hh[:, None]) * NS + split) * RV + rr[None, :],
        latent_sum,
        (hh[:, None] < (head + 1) * GROUP) & (rr[None, :] < RV),
    )
    tl.store(
        MassOut + (batch * HQ + hh) * NS + split,
        mass,
        hh < (head + 1) * GROUP,
    )
    tl.store(
        Out + ((batch * HQ + hh[:, None]) * NS + split) * HD + dd[None, :],
        output,
        (hh[:, None] < (head + 1) * GROUP) & (dd[None, :] < HD),
    )


@triton.jit
def _merge_values(
    Parts,
    Latents,
    Masses,
    Decoder,
    Mean,
    Out,
    HQ: tl.constexpr,
    GROUP: tl.constexpr,
    NS: tl.constexpr,
    HD: tl.constexpr,
    RV: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
    BR: tl.constexpr,
    BH: tl.constexpr,
    SERIAL: tl.constexpr,
):
    batch, head = tl.program_id(0), tl.program_id(1)
    hh = head * GROUP + tl.arange(0, BH)
    ss, dd, rr = tl.arange(0, BS), tl.arange(0, BD), tl.arange(0, BR)
    if SERIAL:
        latent_sum = tl.zeros((BH, BR), tl.float32)
        partial_sum = tl.zeros((BH, BD), tl.float32)
        mass_sum = tl.zeros((BH,), tl.float32)
        for split in range(NS):
            latent_sum += tl.load(
                Latents + ((batch * HQ + hh[:, None]) * NS + split) * RV + rr[None, :],
                (hh[:, None] < (head + 1) * GROUP) & (rr[None, :] < RV),
                0,
            )
            partial_sum += tl.load(
                Parts + ((batch * HQ + hh[:, None]) * NS + split) * HD + dd[None, :],
                (hh[:, None] < (head + 1) * GROUP) & (dd[None, :] < HD),
                0,
            )
            mass_sum += tl.load(
                Masses + (batch * HQ + hh) * NS + split, hh < (head + 1) * GROUP, 0
            )
    else:
        latent = tl.load(
            Latents
            + ((batch * HQ + hh[:, None, None]) * NS + ss[None, :, None]) * RV
            + rr[None, None, :],
            (hh[:, None, None] < (head + 1) * GROUP)
            & (ss[None, :, None] < NS)
            & (rr[None, None, :] < RV),
            0,
        )
        partial = tl.load(
            Parts
            + ((batch * HQ + hh[:, None, None]) * NS + ss[None, :, None]) * HD
            + dd[None, None, :],
            (hh[:, None, None] < (head + 1) * GROUP)
            & (ss[None, :, None] < NS)
            & (dd[None, None, :] < HD),
            0,
        )
        mass = tl.load(
            Masses + (batch * HQ + hh[:, None]) * NS + ss[None, :],
            (hh[:, None] < (head + 1) * GROUP) & (ss[None, :] < NS),
            0,
        )
        latent_sum, partial_sum, mass_sum = (
            tl.sum(latent, 1),
            tl.sum(partial, 1),
            tl.sum(mass, 1),
        )
    decoder = tl.load(
        Decoder + (head * HD + dd[None, :]) * RV + rr[:, None],
        (rr[:, None] < RV) & (dd[None, :] < HD),
        0,
    )
    # Decode the weighted code once, after every selected-token split reduces.
    output = tl.dot(latent_sum, decoder, partial_sum, input_precision="tf32x3")
    mean = tl.load(Mean + head * HD + dd, dd < HD, 0)
    output += mass_sum[:, None] * mean[None, :]
    tl.store(
        Out + (batch * HQ + hh[:, None]) * HD + dd[None, :],
        output,
        (hh[:, None] < (head + 1) * GROUP) & (dd[None, :] < HD),
    )


def absorbed_page_attention(q, slots, use_code, pool, layer, workspace, *, scale):
    """Read selected virtual slots; use_code carries the request's prefix role.

    A partial page without a published code falls back to its exact view. Caller
    supplies valid allocated virtual slots (or <=0 padding), contiguous tensors,
    and a workspace shared across layers. Ownership is managed outside the graph.
    """
    batch, hq, hd = q.shape
    topk = slots.shape[1]
    if not q.is_cuda or slots.shape != use_code.shape or slots.shape[0] != batch:
        raise ValueError(
            "CUDA queries and matching [batch, topk] slot/role tables required"
        )
    if not all(t.is_contiguous() for t in (q, slots, use_code)):
        raise ValueError("QSA read inputs must be contiguous")
    if topk != workspace.topk or batch > workspace.queries:
        raise ValueError("QSA read workspace is too small or has a different topk")
    layout, hk, ps = pool.layout, pool.head_count, pool.page_size
    if hd != 256 or layout.rotary_dim != 64 or layout.value_sparse != 32 or hq % hk:
        raise ValueError(
            "QSA absorbed kernel requires head_dim256, rotary64 and grouped heads"
        )
    kw, vw = pool.weights[layer]
    code = pool.codes[layer]
    exact_k, exact_v = pool.exact[layer]
    qcode = workspace.qcode[:batch]
    scores, probabilities = workspace.scores[:batch], workspace.probabilities[:batch]
    group = hq // hk
    bh = max(16, triton.next_power_of_2(group))
    _project_q[(batch, hk)](
        q,
        kw.decoder,
        kw.mean,
        qcode,
        workspace.qmean,
        hq,
        hk,
        hd,
        64,
        layout.key_rank,
        group,
        bh,
        triton.next_power_of_2(layout.key_rank),
        num_warps=4,
    )
    if not layout.stored_residuals:
        for sparse_code, weights, correction, rank, sparse, dim in (
            (
                code.key,
                kw,
                workspace.kcorrection,
                layout.key_rank,
                layout.key_sparse,
                192,
            ),
            (
                code.value,
                vw,
                workspace.vcorrection,
                layout.value_rank,
                layout.value_sparse,
                256,
            ),
        ):
            _spike_correction[(triton.cdiv(batch * topk, 4), hk)](
                slots,
                use_code,
                pool.code_page,
                sparse_code.latent,
                sparse_code.indices,
                sparse_code.originals,
                weights.decoder,
                weights.mean,
                correction,
                batch * topk,
                hk,
                ps,
                rank,
                sparse,
                dim,
                triton.next_power_of_2(sparse),
                triton.next_power_of_2(rank),
                4,
                num_warps=4,
            )
    _scores[(batch, hk, triton.cdiv(topk, workspace.score_block))](
        q,
        qcode,
        slots,
        use_code,
        pool.code_page,
        pool.exact_page,
        code.rotary,
        code.key.latent,
        code.key.indices,
        code.key.originals if layout.stored_residuals else workspace.kcorrection,
        workspace.qmean,
        exact_k,
        scores,
        layout.stored_residuals,
        scale,
        topk,
        hq,
        hk,
        hd,
        ps,
        64,
        layout.key_rank,
        layout.key_sparse,
        group,
        bh,
        workspace.score_block,
        triton.next_power_of_2(layout.key_rank),
        triton.next_power_of_2(layout.key_sparse),
        num_warps=4,
    )
    torch.softmax(scores, dim=-1, out=probabilities)
    torch.nan_to_num(probabilities, nan=0.0, out=probabilities)
    output = torch.empty_like(q)
    _values[(batch, hk, workspace.value_splits)](
        probabilities,
        slots,
        use_code,
        pool.code_page,
        pool.exact_page,
        code.value.latent,
        code.value.indices.view(torch.int32)
        if layout.value_bitmap
        else code.value.indices,
        code.value.originals if layout.stored_residuals else workspace.vcorrection,
        exact_v,
        workspace.value_partials,
        workspace.latent_partials,
        workspace.mass_partials,
        layout.stored_residuals,
        layout.value_bitmap,
        topk,
        hq,
        hk,
        hd,
        ps,
        layout.value_rank,
        layout.value_sparse,
        group,
        bh,
        workspace.value_block,
        hd,
        triton.next_power_of_2(layout.value_rank),
        triton.next_power_of_2(layout.value_sparse),
        workspace.value_splits,
        triton.cdiv(topk, workspace.value_splits * workspace.value_block)
        * workspace.value_block,
        num_warps=workspace.value_warps,
    )
    _merge_values[(batch, hk)](
        workspace.value_partials,
        workspace.latent_partials,
        workspace.mass_partials,
        vw.decoder,
        vw.mean,
        output,
        hq,
        group,
        workspace.value_splits,
        hd,
        layout.value_rank,
        triton.next_power_of_2(workspace.value_splits),
        triton.next_power_of_2(hd),
        triton.next_power_of_2(layout.value_rank),
        bh,
        workspace.serial_merge,
        num_warps=8,
    )
    return output


@triton.jit
def _write_exact(
    K,
    V,
    Loc,
    Page,
    OutK,
    OutV,
    ROW: tl.constexpr,
    PS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    dd = tl.arange(0, BLOCK)
    slot = tl.load(Loc + token)
    physical_page = tl.load(Page + tl.maximum(slot, 0) // PS)
    physical = physical_page * PS + slot % PS
    k = tl.load(K + token * ROW + dd, dd < ROW, 0)
    v = tl.load(V + token * ROW + dd, dd < ROW, 0)
    writable = (slot > 0) & (physical_page > 0)
    tl.store(OutK + physical * ROW + dd, k, writable & (dd < ROW))
    tl.store(OutV + physical * ROW + dd, v, writable & (dd < ROW))


def write_exact_tokens(k, v, locations, pool, layer):
    """Graph-safe write to preallocated exact pages; padding never touches data.

    The scheduler reserves exact pages before capture/replay. Code pages are
    immutable and never destinations. Draft tokens use this same exact writer.
    """
    row = pool.head_count * pool.layout.head_dim
    if k.numel() != locations.numel() * row or v.shape != k.shape:
        raise ValueError("QSA exact write shapes disagree")
    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("x256 exact token writes require bf16")
    if locations.numel():
        _write_exact[(locations.numel(),)](
            k.contiguous(),
            v.contiguous(),
            locations.contiguous(),
            pool.exact_page,
            *pool.exact[layer],
            row,
            pool.page_size,
            triton.next_power_of_2(row),
        )
