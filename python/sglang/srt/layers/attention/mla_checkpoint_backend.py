"""Two-pass native/latent MLA attention with one global softmax.

No prompt c or dense h is materialized. Reconstructed RoPE keys use one shared
64-coordinate workspace. Sparse weighted residuals use bitmap rank/select tiles and tensor-core products.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _sparse_idx(Idx, loc, offsets, SP: tl.constexpr):
    pair = offsets // 2
    a = tl.load(Idx + loc[:, None]*(SP*3//2) + pair[None, :]*3)
    b = tl.load(Idx + loc[:, None]*(SP*3//2) + pair[None, :]*3+1)
    c = tl.load(Idx + loc[:, None]*(SP*3//2) + pair[None, :]*3+2)
    return tl.where(offsets[None, :] % 2 == 0,
                    a.to(tl.int32) | ((b.to(tl.int32) & 15) << 8),
                    (b.to(tl.int32) >> 4) | (c.to(tl.int32) << 4))


@tr.jit
def _dot_split_bf16(a, b, acc, A_EXACT: tl.constexpr = False, B_EXACT: tl.constexpr = False):
    # Two bf16 components per fp32 operand; fp32 accumulation, including low*low.
    ah = a.to(tl.bfloat16)
    bh = b.to(tl.bfloat16)
    acc = tl.dot(ah, bh, acc)
    if not A_EXACT:
        al = (a-ah.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(al, bh, acc)
    if not B_EXACT:
        bl = (b-bh.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(ah, bl, acc)
    if not A_EXACT and not B_EXACT:
        acc = tl.dot(al, bl, acc)
    return acc


@tr.jit
def _residual_tile(Bitmap, Prefix, Val, Sc, loc, valid, d, SP: tl.constexpr):
    bits = tl.load(Bitmap+loc[:, None]*64+d[None, :]//32, valid[:, None], 0).to(tl.uint32)
    prefix = tl.load(Prefix+loc[:, None]*64+d[None, :]//32, valid[:, None], 0).to(tl.int32)
    bit = tl.full((d.shape[0],), 1, tl.uint32) << (d % 32)
    below = bits & (bit[None, :]-1)
    count = tl.inline_asm_elementwise("popc.b32 $0, $1;", constraints="=r,r",
                                     args=[below], dtype=tl.int32, is_pure=True, pack=1)
    hit = valid[:, None] & ((bits & bit[None, :]) != 0)
    residual = tl.load(Val+loc[:, None]*SP+prefix+count, hit, 0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    return residual*tl.load(Sc+loc)[:, None]


@tr.jit
def _rope_sparse(Idx, Val, Sc, A, Native, Req, Rows, Lens, Scratch,
                 ROW: tl.constexpr, SP: tl.constexpr, BS: tl.constexpr):
    b, split = tl.program_id(0).to(tl.int64), tl.program_id(1)
    length = tl.load(Lens+b)
    row = tl.load(Rows+b).to(tl.int64)
    t = tl.arange(0, 4)
    ss = tl.arange(0, BS)
    d = tl.arange(0, 64)
    for start in range(split*4, length, 64*4):
        valid = start+t < length
        loc = tl.load(Req+row*ROW+start+t, valid, 0).to(tl.int64)
        valid = valid & (tl.load(Native+loc) < 0)
        if tl.sum(valid.to(tl.int32), 0) > 0:
            ix = _sparse_idx(Idx, loc, tl.minimum(ss, SP-1), SP)
            val = tl.load(Val+loc[:, None]*SP+ss[None, :], valid[:, None] & (ss[None, :] < SP), 0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
            a = tl.load(A+ix[:, :, None]*64+d[None, None, :], valid[:, None, None] & (ss[None, :, None] < SP), 0)
            result = tl.sum(a*val[:, :, None], 1)*tl.load(Sc+loc)[:, None]
            tl.store(Scratch+(b*ROW+start+t[:, None])*64+d[None, :], result, valid[:, None])


@tr.jit
def _sparse_lse(Qh, Idx, Val, Sc, Norm, Native, Req, Rows, Lens, Score, Part,
                ROW: tl.constexpr, HEADS: tl.constexpr, SP: tl.constexpr, BS: tl.constexpr,
                NS: tl.constexpr, L, SPLITS: tl.constexpr, BH: tl.constexpr, SCALE: tl.constexpr):
    b, split = tl.program_id(0).to(tl.int64), tl.program_id(1)
    length = tl.load(Lens+b)
    row = tl.load(Rows+b).to(tl.int64)
    h = tl.arange(0, BH)
    t = tl.arange(0, 8)
    ss = tl.arange(0, BS)
    m = tl.full((BH,), -float("inf"), tl.float32)
    total = tl.full((BH,), 0., tl.float32)
    for start in range(split*8, length, SPLITS*8):
        valid = start+t < length
        loc = tl.load(Req+row*ROW+start+t, valid, 0).to(tl.int64)
        valid = valid & (tl.load(Native+loc) < 0)
        if tl.sum(valid.to(tl.int32), 0) > 0:
            ix = _sparse_idx(Idx, loc, tl.minimum(ss, SP-1), SP)
            val = tl.load(Val+loc[:, None]*SP+ss[None, :], valid[:, None] & (ss[None, :] < SP), 0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
            qh = tl.load(Qh+(b*2048+ix[None, :, :])*HEADS+h[:, None, None],
                         (h[:, None, None] < HEADS) & valid[None, :, None] & (ss[None, None, :] < SP), 0)
            correction = tl.sum(qh*val[None, :, :], 2)*tl.load(Sc+loc)[None, :]
            norm = tl.load(Norm+loc*NS)*tl.load(Norm+loc*NS+L)
            score = tl.load(Score+(b*HEADS+h[:, None])*ROW+start+t[None, :], (h[:, None] < HEADS) & valid[None, :], 0)
            score += correction/norm[None, :]*SCALE
            score = tl.where(valid[None, :], score, -float("inf"))
            next_m = tl.maximum(m, tl.max(score, 1))
            total = total*tl.exp(m-next_m)+tl.sum(tl.exp(score-next_m[:, None]), 1)
            m = next_m
            tl.store(Score+(b*HEADS+h[:, None])*ROW+start+t[None, :], score, (h[:, None] < HEADS) & valid[None, :])
    tl.store(Part+(b*HEADS+h)*SPLITS+split, tl.where(total > 0, m+tl.log(total), -float("inf")), h < HEADS)


@tr.jit
def _rope(Z, Idx, Bitmap, Prefix, Val, Sc, SparseRope, Norm, Pos, Native, Re, Rm, A, CosSin, Rope,
          Req, Rows, Lens, ROW: tl.constexpr, R: tl.constexpr, SP: tl.constexpr,
          NS: tl.constexpr, SPLITS: tl.constexpr, BR: tl.constexpr):
    b, split = tl.program_id(0).to(tl.int64), tl.program_id(1)
    length = tl.load(Lens+b)
    row = tl.load(Rows+b).to(tl.int64)
    t = tl.arange(0, 32)
    rk = tl.arange(0, 64)
    d = tl.arange(0, 64)
    # Disjoint subsequences; only compact slots need reconstructed RoPE.
    for start in range(split*32, length, SPLITS*32):
        loc = tl.load(Req + row*ROW + start+t, start+t < length, 0).to(tl.int64)
        native = tl.load(Native+loc)
        valid = (start+t < length) & (native < 0)
        if tl.sum(valid.to(tl.int32), 0) > 0:
            pe = tl.full((32, 64), 0., tl.float32)
            for kstart in range(0, R, 64):
                r = kstart+rk
                z = tl.load(Z+loc[:, None]*R+r[None, :], valid[:, None] & (r[None, :] < R), 0).to(tl.float32)
                re = tl.load(Re+d[None, :]*R+r[:, None], r[:, None] < R, 0)
                pe = _dot_split_bf16(z, re, pe, True, False)
            pe += tl.load(Rm+d)[None, :]
            if SP > 0:
                pe += tl.load(SparseRope+(b*ROW+start+t[:, None])*64+d[None, :], valid[:, None], 0)
            pe /= tl.load(Norm+loc*NS)[:, None]
            pos = tl.load(Pos+loc).to(tl.int64)
            cos = tl.load(CosSin+pos[:, None]*64+(d[None, :]//2))
            sin = tl.load(CosSin+pos[:, None]*64+32+(d[None, :]//2))
            paired = tl.gather(pe, tl.broadcast_to((d ^ 1)[None, :], (32, 64)), 1)
            pe = pe*cos + tl.where(d[None, :] % 2 == 0, -paired, paired)*sin
            tl.store(Rope+loc[:, None]*64+d[None, :], pe, valid[:, None])


@tr.jit
def _native(Q, Native, KV, Req, Rows, Lens, Part, Uc,
            ROW: tl.constexpr, HEADS: tl.constexpr, SPLITS: tl.constexpr,
            SCALE: tl.constexpr, BH: tl.constexpr):
    b, split = tl.program_id(0).to(tl.int64), tl.program_id(1)
    length = tl.load(Lens+b)
    row = tl.load(Rows+b).to(tl.int64)
    h = tl.arange(0, BH)
    t = tl.arange(0, 32)
    k = tl.arange(0, 64)
    d = tl.arange(0, 512)
    qp = tl.load(Q+(b*HEADS+h[:, None])*576+512+k[None, :], h[:, None] < HEADS, 0)
    m = tl.full((BH,), -float("inf"), tl.float32)
    total = tl.full((BH,), 0., tl.float32)
    acc = tl.full((BH, 512), 0., tl.float32)
    for start in range(split*32, length, SPLITS*32):
        valid = start+t < length
        loc = tl.load(Req+row*ROW+start+t, valid, 0).to(tl.int64)
        native = tl.load(Native+loc)
        valid = valid & (native >= 0)
        if tl.sum(valid.to(tl.int32), 0) > 0:
            nr = tl.maximum(native, 0)
            score = tl.full((BH, 32), 0., tl.float32)
            for ks in range(0, 512, 64):
                qc = tl.load(Q+(b*HEADS+h[:, None])*576+ks+k[None, :], h[:, None] < HEADS, 0)
                ck = tl.load(KV+nr[:, None]*576+ks+k[None, :], valid[:, None], 0)
                score = tl.dot(qc, tl.trans(ck), score)
            pe = tl.load(KV+nr[:, None]*576+512+k[None, :], valid[:, None], 0)
            score = tl.dot(qp, tl.trans(pe), score)*SCALE
            score = tl.where(valid[None, :], score, -float("inf"))
            new_m = tl.maximum(m, tl.max(score, 1))
            alpha = tl.exp(m-new_m)
            weights = tl.exp(score-new_m[:, None])
            c = tl.load(KV+nr[:, None]*576+d[None, :], valid[:, None], 0)
            acc = _dot_split_bf16(weights, c, acc*alpha[:, None], False, True)
            total = total*alpha+tl.sum(weights, 1)
            m = new_m
    tl.store(Part+(b*HEADS+h)*SPLITS+split, tl.where(total > 0, m+tl.log(total), -float("inf")), h < HEADS)
    tl.store(Uc+((b*HEADS+h[:, None])*SPLITS+split)*512+d[None, :],
             tl.where(total[:, None] > 0, acc/total[:, None], 0.), h[:, None] < HEADS)


@tr.jit
def _compact_lse(Q, Qz, Qh, Bias, Z, Idx, Bitmap, Prefix, Val, Sc, Norm, Native, Rope,
                 Req, Rows, Lens, Score, Part, ROW: tl.constexpr, HEADS: tl.constexpr,
                 R: tl.constexpr, SP: tl.constexpr, NS: tl.constexpr, L,
                 SPLITS: tl.constexpr, BH: tl.constexpr, SCALE: tl.constexpr):
    b, split = tl.program_id(0).to(tl.int64), tl.program_id(1)
    length = tl.load(Lens+b)
    row = tl.load(Rows+b).to(tl.int64)
    h = tl.arange(0, BH)
    t = tl.arange(0, 64)
    rk = tl.arange(0, 64)
    d = tl.arange(0, 64)
    bias = tl.load(Bias+b*HEADS+h, h < HEADS, 0)
    qp = tl.load(Q+(b*HEADS+h[:, None])*576+512+d[None, :], h[:, None] < HEADS, 0).to(tl.float32)
    m = tl.full((BH,), -float("inf"), tl.float32)
    total = tl.full((BH,), 0., tl.float32)
    for start in range(split*64, length, SPLITS*64):
        valid = start+t < length
        loc = tl.load(Req+row*ROW+start+t, valid, 0).to(tl.int64)
        valid = valid & (tl.load(Native+loc) < 0)
        if tl.sum(valid.to(tl.int32), 0) > 0:
            score = tl.full((BH, 64), 0., tl.float32)
            for kstart in range(0, R, 64):
                k = kstart+rk
                qq = tl.load(Qz+(b*HEADS+h[:, None])*R+k[None, :], (h[:, None] < HEADS) & (k[None, :] < R), 0)
                zz = tl.load(Z+loc[:, None]*R+k[None, :], valid[:, None] & (k[None, :] < R), 0).to(tl.float32)
                score = _dot_split_bf16(qq, tl.trans(zz), score, False, True)
            score += bias[:, None]
            if SP > 0:
                for ds in range(0, 2048, 64):
                    hidden = ds+rk
                    residual = _residual_tile(Bitmap, Prefix, Val, Sc, loc, valid, hidden, SP)
                    qh = tl.load(Qh+(b*2048+hidden[None, :])*HEADS+h[:, None], h[:, None] < HEADS, 0)
                    score = _dot_split_bf16(qh, tl.trans(residual), score)
            norm = tl.load(Norm+loc*NS)*tl.load(Norm+loc*NS+L)
            score /= norm[None, :]
            rp = tl.load(Rope+loc[:, None]*64+d[None, :], valid[:, None], 0)
            score = _dot_split_bf16(qp, tl.trans(rp), score, True, False)
            score = tl.where(valid[None, :], score*SCALE, -float("inf"))
            next_m = tl.maximum(m, tl.max(score, 1))
            total = total*tl.exp(m-next_m)+tl.sum(tl.exp(score-next_m[:, None]), 1)
            m = next_m
            tl.store(Score+(b*HEADS+h[:, None])*ROW+start+t[None, :], score,
                     (h[:, None] < HEADS) & valid[None, :])
    tl.store(Part+(b*HEADS+h)*SPLITS+split,
             tl.where(total > 0, m+tl.log(total), -float("inf")), h < HEADS)


@tr.jit
def _prob(Norm, Native, Req, Rows, Lens, Part, NativePart, Score, Mass,
          ROW: tl.constexpr, HEADS: tl.constexpr, NS: tl.constexpr, L,
          SPLITS: tl.constexpr, BH: tl.constexpr):
    b, split = tl.program_id(0).to(tl.int64), tl.program_id(1)
    length = tl.load(Lens+b)
    row = tl.load(Rows+b).to(tl.int64)
    h = tl.arange(0, BH)
    s = tl.arange(0, SPLITS)
    part = tl.load(Part+(b*HEADS+h[:, None])*SPLITS+s[None, :], h[:, None] < HEADS, -float("inf"))
    native_part = tl.load(NativePart+(b*HEADS+h[:, None])*SPLITS+s[None, :], h[:, None] < HEADS, -float("inf"))
    m = tl.maximum(tl.max(part, 1), tl.max(native_part, 1))
    logsum = m+tl.log(tl.sum(tl.exp(part-m[:, None]), 1)+tl.sum(tl.exp(native_part-m[:, None]), 1))
    logsum = tl.where(length > 0, logsum, 0.)
    t = tl.arange(0, 128)
    mass = tl.full((BH,), 0., tl.float32)
    for start in range(split*128, length, SPLITS*128):
        valid = start+t < length
        loc = tl.load(Req+row*ROW+start+t, valid, 0).to(tl.int64)
        valid = valid & (tl.load(Native+loc) < 0)
        score = tl.load(Score+(b*HEADS+h[:, None])*ROW+start+t[None, :],
                        (h[:, None] < HEADS) & valid[None, :], 0)
        norm = tl.load(Norm+loc*NS)*tl.load(Norm+loc*NS+L)
        ps = tl.where(valid[None, :], tl.exp(score-logsum[:, None])/norm[None, :], 0.)
        mass += tl.sum(ps, 1)
        tl.store(Score+(b*HEADS+h[:, None])*ROW+start+t[None, :], ps,
                 (h[:, None] < HEADS) & valid[None, :])
    tl.store(Mass+(b*HEADS+h)*SPLITS+split, mass, h < HEADS)


@tr.jit
def _latent_value(Z, Native, Req, Rows, Lens, Prob, Uz,
                  ROW: tl.constexpr, HEADS: tl.constexpr, R: tl.constexpr,
                  SPLITS: tl.constexpr, BH: tl.constexpr, BD: tl.constexpr):
    b, dim, split = tl.program_id(0).to(tl.int64), tl.program_id(1), tl.program_id(2)
    length = tl.load(Lens+b)
    row = tl.load(Rows+b).to(tl.int64)
    t = tl.arange(0, 32)
    h = tl.arange(0, BH)
    d = dim*BD+tl.arange(0, BD)
    acc = tl.full((BH, BD), 0., tl.float32)
    for start in range(split*32, length, SPLITS*32):
        valid = start+t < length
        loc = tl.load(Req+row*ROW+start+t, valid, 0).to(tl.int64)
        valid = valid & (tl.load(Native+loc) < 0)
        if tl.sum(valid.to(tl.int32), 0) > 0:
            z = tl.load(Z+loc[:, None]*R+d[None, :], valid[:, None] & (d[None, :] < R), 0).to(tl.float32)
            weights = tl.load(Prob+(b*HEADS+h[:, None])*ROW+start+t[None, :],
                              (h[:, None] < HEADS) & valid[None, :], 0)
            acc = _dot_split_bf16(weights, z, acc, False, True)
    tl.store(Uz+((b*HEADS+h[:, None])*SPLITS+split)*R+d[None, :], acc,
             (h[:, None] < HEADS) & (d[None, :] < R))


@tr.jit
def _residual(Bitmap, Prefix, Val, Sc, Native, Req, Rows, Lens, Prob, Uh,
              ROW: tl.constexpr, HEADS: tl.constexpr, SP: tl.constexpr,
              SPLITS: tl.constexpr, BH: tl.constexpr, BD: tl.constexpr):
    # Rank/select metadata reconstructs a small residual tile. No per-token
    # dense h/c is retained, and heads share the tensor-core weighted product.
    b, dim, split = tl.program_id(0).to(tl.int64), tl.program_id(1), tl.program_id(2)
    length = tl.load(Lens+b)
    row = tl.load(Rows+b).to(tl.int64)
    t = tl.arange(0, 32)
    h = tl.arange(0, BH)
    d = dim*BD+tl.arange(0, BD)
    bit = tl.full((BD,), 1, tl.uint32) << (d % 32)
    acc = tl.full((BH, BD), 0., tl.float32)
    for start in range(split*32, length, SPLITS*32):
        valid = start+t < length
        loc = tl.load(Req+row*ROW+start+t, valid, 0).to(tl.int64)
        valid = valid & (tl.load(Native+loc) < 0)
        if tl.sum(valid.to(tl.int32), 0) > 0:
            bits = tl.load(Bitmap+loc[:, None]*64+d[None, :]//32, valid[:, None], 0).to(tl.uint32)
            prefix = tl.load(Prefix+loc[:, None]*64+d[None, :]//32, valid[:, None], 0).to(tl.int32)
            below = bits & (bit[None, :]-1)
            count = tl.inline_asm_elementwise("popc.b32 $0, $1;", constraints="=r,r",
                                             args=[below], dtype=tl.int32, is_pure=True, pack=1)
            offset = prefix+count
            hit = valid[:, None] & ((bits & bit[None, :]) != 0)
            residual = tl.load(Val+loc[:, None]*SP+offset, hit, 0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
            residual *= tl.load(Sc+loc)[:, None]
            weights = tl.load(Prob+(b*HEADS+h[:, None])*ROW+start+t[None, :],
                              (h[:, None] < HEADS) & valid[None, :], 0)
            acc = _dot_split_bf16(weights.to(tl.bfloat16), residual, acc, True, False)
    tl.store(Uh+((b*HEADS+h[:, None])*SPLITS+split)*2048+d[None, :], acc, h[:, None] < HEADS)


def checkpoint_attention(pool, q, layer_id, req_to_token, req_indices, seq_lens, scale):
    """Stock absorbed bf16 query, compact/native ONE softmax, tiled latent output."""
    cfg = pool.checkpoint_config
    q = q.reshape(q.shape[0], -1, 576).contiguous()
    b, heads = q.shape[:2]
    if b == 0:
        return q.new_empty((0, heads, 512))
    w = pool.folded[layer_id]
    qc = q[..., :512].float()
    qz = (qc @ w['pe']).contiguous()
    qh = (qc @ w['p']).transpose(1, 2).to(torch.bfloat16).contiguous() if cfg.sparse else qc.new_empty(0)
    bias = (qc @ w['pm']).contiguous()
    splits = 8
    row = req_to_token.stride(0)
    bh = max(16, tr.next_power_of_2(heads))
    part = torch.empty((b, heads, splits), dtype=torch.float32, device=q.device)
    native_part = torch.empty_like(part)
    uz = torch.empty((b, heads, splits, cfg.rank), dtype=torch.float32, device=q.device)
    uc = torch.empty((b, heads, splits, 512), dtype=torch.float32, device=q.device)
    prob = torch.empty((b, heads, row), dtype=torch.float32, device=q.device)
    uh_parts = torch.empty((b, heads, splits, 2048 if cfg.sparse else 0), dtype=torch.float32, device=q.device)
    mass = torch.empty_like(part)
    ns = pool.norms.shape[1]
    lid = layer_id-cfg.first_layer+1
    sparse_rope = torch.empty((b, row, 64 if cfg.sparse else 0), dtype=torch.float32, device=q.device)
    if cfg.sparse:
        _rope_sparse[(b,64)](pool.indices,pool.values,pool.residual_scale,w['ar_t_bf16'],pool.native_of,
            req_to_token,req_indices,seq_lens,sparse_rope,row,cfg.sparse,tr.next_power_of_2(cfg.sparse),num_warps=8)
    _rope[(b, splits)](pool.z,pool.indices,pool.bitmap,pool.bitmap_prefix,pool.values,pool.residual_scale,sparse_rope,pool.norms,pool.positions,
        pool.native_of,w['re'],w['rm'],w['ar_t'],pool.cos_sin_cache,pool.rope_scratch,
        req_to_token,req_indices,seq_lens,row,cfg.rank,cfg.sparse,ns,splits,
        tr.next_power_of_2(cfg.rank),num_warps=8)
    _native[(b,splits)](q,pool.native_of,pool.kv_buffer[layer_id],req_to_token,
        req_indices,seq_lens,native_part,uc,row,heads,splits,scale,bh,num_warps=8)
    _compact_lse[(b,splits)](q,qz,qh,bias,pool.z,pool.indices,pool.bitmap,pool.bitmap_prefix,pool.values,pool.residual_scale,
        pool.norms,pool.native_of,pool.rope_scratch,req_to_token,req_indices,seq_lens,
        prob,part,row,heads,cfg.rank,0,ns,lid,splits,bh,scale,num_warps=8)
    if cfg.sparse:
        _sparse_lse[(b,splits)](qh,pool.indices,pool.values,pool.residual_scale,pool.norms,pool.native_of,
            req_to_token,req_indices,seq_lens,prob,part,row,heads,cfg.sparse,tr.next_power_of_2(cfg.sparse),
            ns,lid,splits,bh,scale,num_warps=8)
    _prob[(b,splits)](pool.norms,pool.native_of,req_to_token,req_indices,seq_lens,
        part,native_part,prob,mass,row,heads,ns,lid,splits,bh,num_warps=4)
    _latent_value[(b,tr.cdiv(cfg.rank,128),splits)](pool.z,pool.native_of,req_to_token,
        req_indices,seq_lens,prob,uz,row,heads,cfg.rank,splits,bh,128,num_warps=8)
    c = uz.sum(2) @ w['pe'].T
    if cfg.sparse:
        _residual[(b,16,splits)](pool.bitmap,pool.bitmap_prefix,pool.values,pool.residual_scale,
            pool.native_of,req_to_token,req_indices,seq_lens,prob,uh_parts,row,heads,
            cfg.sparse,splits,bh,128,num_warps=8)
        c = c + uh_parts.sum(2) @ w['p'].T
    weights = torch.softmax(torch.cat((part,native_part),-1),-1)[...,splits:]
    weights = torch.nan_to_num(weights, nan=0.)
    c = c + mass.sum(2)[..., None]*w['pm'] + (uc*weights[...,None]).sum(2)
    return c.to(q.dtype)
