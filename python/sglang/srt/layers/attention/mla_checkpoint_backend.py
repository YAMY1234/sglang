"""Two-pass native/latent MLA attention with one global softmax.

No prompt c or dense h is materialized. Reconstructed RoPE keys use one shared
64-coordinate workspace. Sparse weighted residuals use sorted-coordinate tiles and tensor-core products.
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
def _rope(Z, Idx, Val, Sc, Norm, Pos, Native, Re, Rm, A, CosSin, Rope,
          Req, Rows, Lens, ROW: tl.constexpr, R: tl.constexpr, SP: tl.constexpr,
          NS: tl.constexpr, SPLITS: tl.constexpr, BR: tl.constexpr):
    b, split = tl.program_id(0), tl.program_id(1)
    length = tl.load(Lens+b)
    row = tl.load(Rows+b).to(tl.int64)
    t = tl.arange(0, 16)
    rk = tl.arange(0, 128)
    d = tl.arange(0, 64)
    # Disjoint subsequences; only compact slots need reconstructed RoPE.
    for start in range(split*16, length, SPLITS*16):
        loc = tl.load(Req + row*ROW + start+t, start+t < length, 0).to(tl.int64)
        native = tl.load(Native+loc)
        valid = (start+t < length) & (native < 0)
        if tl.sum(valid.to(tl.int32), 0) > 0:
            pe = tl.full((16, 64), 0., tl.float32)
            for kstart in range(0, R, 128):
                r = kstart+rk
                z = tl.load(Z+loc[:, None]*R+r[None, :], valid[:, None] & (r[None, :] < R), 0).to(tl.float32)
                re = tl.load(Re+d[None, :]*R+r[:, None], r[:, None] < R, 0)
                pe += tl.dot(z, re, input_precision="tf32x3")
            pe += tl.load(Rm+d)[None, :]
            if SP > 0:
                for s in range(0, SP, 8):
                    ss = s+tl.arange(0, 8)
                    ix = _sparse_idx(Idx, loc, ss, SP)
                    val = tl.load(Val+loc[:, None]*SP+ss[None, :]).to(tl.float8e4nv, bitcast=True).to(tl.float32)
                    w = tl.load(A+(512+d[None, None, :])*2048+ix[:, :, None])
                    pe += tl.sum(w*val[:, :, None], 1)*tl.load(Sc+loc)[:, None]
            pe /= tl.load(Norm+loc*NS)[:, None]
            pos = tl.load(Pos+loc).to(tl.int64)
            cos = tl.load(CosSin+pos[:, None]*64+(d[None, :]//2))
            sin = tl.load(CosSin+pos[:, None]*64+32+(d[None, :]//2))
            paired = tl.gather(pe, tl.broadcast_to((d ^ 1)[None, :], (16, 64)), 1)
            pe = pe*cos + tl.where(d[None, :] % 2 == 0, -paired, paired)*sin
            tl.store(Rope+loc[:, None]*64+d[None, :], pe, valid[:, None])


@tr.jit
def _scores(Q, Qz, Qh, Bias, Z, Idx, Val, Sc, Norm, Rope,
            loc, valid, bh, R: tl.constexpr, SP: tl.constexpr, NS: tl.constexpr,
            L, BR: tl.constexpr, BS: tl.constexpr, SCALE: tl.constexpr):
    r = tl.arange(0, BR)
    z = tl.load(Z+loc[:, None]*R+r[None, :], valid[:, None] & (r[None, :] < R), 0).to(tl.float32)
    qz = tl.load(Qz+bh*R+r, r < R, 0)
    score = tl.sum(z*qz[None, :], 1)+tl.load(Bias+bh)
    if SP > 0:
        ss = tl.arange(0, BS)
        ix = _sparse_idx(Idx, loc, tl.minimum(ss, SP-1), SP)
        val = tl.load(Val+loc[:, None]*SP+ss[None, :], ss[None, :] < SP, 0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        qh = tl.load(Qh+bh*2048+ix)
        score += tl.sum(qh*val, 1)*tl.load(Sc+loc)
    norm = tl.load(Norm+loc*NS)*tl.load(Norm+loc*NS+L)
    score /= norm
    d = tl.arange(0, 64)
    rp = tl.load(Rope+loc[:, None]*64+d[None, :], valid[:, None], 0)
    qp = tl.load(Q+bh*576+512+d).to(tl.float32)
    score += tl.sum(rp*qp[None, :], 1)
    return tl.where(valid, score*SCALE, -float("inf")), z, norm


@tr.jit
def _native(Q, Native, KV, Req, Rows, Lens, Part, Uc,
            ROW: tl.constexpr, HEADS: tl.constexpr, SPLITS: tl.constexpr,
            SCALE: tl.constexpr):
    """Online native-only attention: no latent-sized tensors in this kernel."""
    bh, split = tl.program_id(0), tl.program_id(1)
    b = bh//HEADS
    length = tl.load(Lens+b)
    row = tl.load(Rows+b).to(tl.int64)
    t = tl.arange(0, 32)
    d = tl.arange(0, 512)
    r = tl.arange(0, 64)
    qc = tl.load(Q+bh*576+d).to(tl.float32)
    qp = tl.load(Q+bh*576+512+r).to(tl.float32)
    m = -float("inf")
    total = 0.
    acc = tl.full((512,), 0., tl.float32)
    for start in range(split*32, length, SPLITS*32):
        valid = start+t < length
        loc = tl.load(Req+row*ROW+start+t, valid, 0).to(tl.int64)
        native = tl.load(Native+loc)
        valid = valid & (native >= 0)
        if tl.sum(valid.to(tl.int32), 0) > 0:
            nr = tl.maximum(native, 0)
            c = tl.load(KV+nr[:, None]*576+d[None, :], valid[:, None], 0).to(tl.float32)
            pe = tl.load(KV+nr[:, None]*576+512+r[None, :], valid[:, None], 0).to(tl.float32)
            score = (tl.sum(c*qc[None, :], 1)+tl.sum(pe*qp[None, :], 1))*SCALE
            score = tl.where(valid, score, -float("inf"))
            new_m = tl.maximum(m, tl.max(score, 0))
            alpha = tl.exp(m-new_m)
            weights = tl.exp(score-new_m)
            acc = acc*alpha+tl.sum(weights[:, None]*c, 0)
            total = total*alpha+tl.sum(weights, 0)
            m = new_m
    tl.store(Part+bh*SPLITS+split, tl.where(total > 0, m+tl.log(total), -float("inf")))
    tl.store(Uc+(bh*SPLITS+split)*512+d, tl.where(total > 0, acc/total, 0.))


@tr.jit
def _lse(Q, Qz, Qh, Bias, Z, Idx, Val, Sc, Norm, Native, KV, Rope,
         Req, Rows, Lens, Part, ROW: tl.constexpr, HEADS: tl.constexpr,
         R: tl.constexpr, SP: tl.constexpr, NS: tl.constexpr, L, BT: tl.constexpr,
         BR: tl.constexpr, BS: tl.constexpr, SPLITS: tl.constexpr, SCALE: tl.constexpr):
    bh, split = tl.program_id(0), tl.program_id(1)
    b = bh//HEADS
    length = tl.load(Lens+b)
    row = tl.load(Rows+b).to(tl.int64)
    t = tl.arange(0, BT)
    m = -float("inf")
    total = 0.
    for start in range(split*BT, length, SPLITS*BT):
        valid = start+t < length
        loc = tl.load(Req+row*ROW+start+t, valid, 0).to(tl.int64)
        valid = valid & (tl.load(Native+loc) < 0)
        if tl.sum(valid.to(tl.int32), 0) > 0:
            scores, _, _ = _scores(Q,Qz,Qh,Bias,Z,Idx,Val,Sc,Norm,Rope,
                loc,valid,bh,R,SP,NS,L,BR,BS,SCALE)
            new_m = tl.maximum(m, tl.max(scores, 0))
            total = total*tl.exp(m-new_m)+tl.sum(tl.exp(scores-new_m), 0)
            m = new_m
    tl.store(Part+bh*SPLITS+split, tl.where(total > 0, m+tl.log(total), -float("inf")))


@tr.jit
def _output(Q, Qz, Qh, Bias, Z, Idx, Val, Sc, Norm, Native, KV, Rope,
            Req, Rows, Lens, Part, NativePart, Uz, Prob, Mass, ROW: tl.constexpr, HEADS: tl.constexpr,
            R: tl.constexpr, SP: tl.constexpr, NS: tl.constexpr, L, BT: tl.constexpr,
            BR: tl.constexpr, BS: tl.constexpr, SPLITS: tl.constexpr, SCALE: tl.constexpr):
    bh, split = tl.program_id(0), tl.program_id(1)
    b = bh//HEADS
    length = tl.load(Lens+b)
    row = tl.load(Rows+b).to(tl.int64)
    partial = tl.load(Part+bh*SPLITS+tl.arange(0, SPLITS))
    native_partial = tl.load(NativePart+bh*SPLITS+tl.arange(0, SPLITS))
    max_lse = tl.maximum(tl.max(partial, 0), tl.max(native_partial, 0))
    logsum = max_lse + tl.log(tl.sum(tl.exp(partial-max_lse), 0)
                             + tl.sum(tl.exp(native_partial-max_lse), 0))
    logsum = tl.where(length > 0, logsum, 0.)
    t = tl.arange(0, BT)
    r = tl.arange(0, BR)
    uz = tl.full((BR,), 0., tl.float32)
    mass = 0.
    for start in range(split*BT, length, SPLITS*BT):
        valid = start+t < length
        loc = tl.load(Req+row*ROW+start+t, valid, 0).to(tl.int64)
        compact = valid & (tl.load(Native+loc) < 0)
        if tl.sum(compact.to(tl.int32), 0) > 0:
            scores, z, norm = _scores(Q,Qz,Qh,Bias,Z,Idx,Val,Sc,Norm,Rope,
                loc,compact,bh,R,SP,NS,L,BR,BS,SCALE)
            ps = tl.where(compact, tl.exp(scores-logsum)/norm, 0.)
            uz += tl.sum(ps[:, None]*z, 0)
            mass += tl.sum(ps, 0)
            if SP > 0:
                tl.store(Prob+bh*ROW+start+t, ps, compact)
    tl.store(Uz+(bh*SPLITS+split)*R+r, uz, r < R)
    tl.store(Mass+bh*SPLITS+split, mass)


@tr.jit
def _residual(Idx, Val, Sc, Native, Req, Rows, Lens, Prob, Uh,
              ROW: tl.constexpr, HEADS: tl.constexpr, SP: tl.constexpr,
              SPLITS: tl.constexpr, BH: tl.constexpr, BD: tl.constexpr,
              SEARCH: tl.constexpr):
    # Sparse indices are sorted at encode time. Binary search reconstructs only
    # one 16-by-BD residual tile in registers; no dense per-token h/c is stored.
    b, dim, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(Lens+b)
    row = tl.load(Rows+b).to(tl.int64)
    t = tl.arange(0, 16)
    h = tl.arange(0, BH)
    d = dim*BD+tl.arange(0, BD)
    acc = tl.full((BH, BD), 0., tl.float32)
    for start in range(split*16, length, SPLITS*16):
        valid = start+t < length
        loc = tl.load(Req+row*ROW+start+t, valid, 0).to(tl.int64)
        valid = valid & (tl.load(Native+loc) < 0)
        if tl.sum(valid.to(tl.int32), 0) > 0:
            lo = tl.full((16, BD), 0, tl.int32)
            hi = tl.full((16, BD), SP, tl.int32)
            for step in range(SEARCH):
                mid = (lo+hi)//2
                pair = tl.minimum(mid, SP-1)//2
                base = Idx+loc[:, None]*(SP*3//2)+pair*3
                a = tl.load(base, valid[:, None], 0).to(tl.int32)
                bb = tl.load(base+1, valid[:, None], 0).to(tl.int32)
                c = tl.load(base+2, valid[:, None], 0).to(tl.int32)
                ix = tl.where(mid % 2 == 0, a | ((bb & 15)<<8), (bb>>4) | (c<<4))
                advance = (mid < SP) & (ix < d[None, :])
                lo = tl.where(advance, mid+1, lo)
                hi = tl.where(advance, hi, mid)
            offset = tl.minimum(lo, SP-1)
            base = Idx+loc[:, None]*(SP*3//2)+(offset//2)*3
            a = tl.load(base, valid[:, None], 0).to(tl.int32)
            bb = tl.load(base+1, valid[:, None], 0).to(tl.int32)
            c = tl.load(base+2, valid[:, None], 0).to(tl.int32)
            ix = tl.where(offset % 2 == 0, a | ((bb & 15)<<8), (bb>>4) | (c<<4))
            hit = valid[:, None] & (lo < SP) & (ix == d[None, :])
            residual = tl.load(Val+loc[:, None]*SP+offset, hit, 0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
            residual *= tl.load(Sc+loc)[:, None]
            weights = tl.load(Prob+(b*HEADS+h[:, None])*ROW+start+t[None, :],
                              (h[:, None] < HEADS) & valid[None, :], 0)
            acc = tl.dot(weights, residual, acc, input_precision="tf32x3")
    tl.store(Uh+((b*HEADS+h[:, None])*SPLITS+split)*2048+d[None, :], acc, h[:, None] < HEADS)


def checkpoint_attention(pool, q, layer_id, req_to_token, req_indices, seq_lens, scale):
    """q contains stock absorbed bf16 [q_c512 | rotated q_pe64]."""
    cfg = pool.checkpoint_config
    q = q.reshape(q.shape[0], -1, 576).contiguous()
    b, heads = q.shape[:2]
    if b == 0:
        return q.new_empty((0, heads, 512))
    w = pool.folded[layer_id]
    qc = q[..., :512].float()
    qz = (qc @ w['pe']).contiguous()
    qh = (qc @ w['p']).contiguous()
    bias = (qc @ w['pm']).contiguous()
    splits = 8
    part = torch.empty((b, heads, splits), dtype=torch.float32, device=q.device)
    native_part = torch.empty_like(part)
    uz = torch.empty((b, heads, splits, cfg.rank), dtype=torch.float32, device=q.device)
    uc = torch.empty((b, heads, splits, 512), dtype=torch.float32, device=q.device)
    prob = torch.empty((b, heads, req_to_token.stride(0) if cfg.sparse else 0), dtype=torch.float32, device=q.device)
    uh_parts = torch.empty((b, heads, splits, 2048 if cfg.sparse else 0), dtype=torch.float32, device=q.device)
    mass = torch.empty_like(part)
    # Raw bytes avoid masked-load integer-to-fp8 casts; kernels bitcast explicitly.
    values = pool.values
    ns = pool.norms.shape[1]
    _rope[(b, splits)](pool.z,pool.indices,values,pool.residual_scale,pool.norms,pool.positions,
        pool.native_of,w['re'],w['rm'],w['a'],pool.cos_sin_cache,pool.rope_scratch,
        req_to_token,req_indices,seq_lens,req_to_token.stride(0),cfg.rank,cfg.sparse,ns,splits,
        tr.next_power_of_2(cfg.rank),num_warps=4)
    common = (q,qz,qh,bias,pool.z,pool.indices,values,pool.residual_scale,pool.norms,
              pool.native_of,pool.kv_buffer[layer_id],pool.rope_scratch,req_to_token,req_indices,seq_lens)
    constants = dict(ROW=req_to_token.stride(0),HEADS=heads,R=cfg.rank,SP=cfg.sparse,NS=ns,
        L=layer_id-cfg.first_layer+1,BT=2 if cfg.rank>1024 else 16,BR=tr.next_power_of_2(cfg.rank),
        BS=tr.next_power_of_2(max(1,cfg.sparse)),SPLITS=splits,SCALE=scale,num_warps=4)
    _native[(b*heads,splits)](q,pool.native_of,pool.kv_buffer[layer_id],req_to_token,
        req_indices,seq_lens,native_part,uc,req_to_token.stride(0),heads,splits,scale,num_warps=4)
    _lse[(b*heads,splits)](*common,part,**constants)
    _output[(b*heads,splits)](*common,part,native_part,uz,prob,mass,**constants)
    c = uz.sum(2) @ w['pe'].T
    if cfg.sparse:
        _residual[(b, 16, splits)](pool.indices,values,pool.residual_scale,pool.native_of,
            req_to_token,req_indices,seq_lens,prob,uh_parts,req_to_token.stride(0),heads,
            cfg.sparse,splits,max(16,tr.next_power_of_2(heads)),128,
            tr.next_power_of_2(cfg.sparse).bit_length(),num_warps=4)
        c = c + uh_parts.sum(2) @ w['p'].T
    # Each native split is normalized locally, then receives its fraction of
    # the ONE softmax partition function shared with every compact split.
    weights = torch.softmax(torch.cat((part,native_part),-1),-1)[...,splits:]
    weights = torch.nan_to_num(weights, nan=0.)  # graph padding has no keys
    c = c + mass.sum(2)[..., None]*w['pm'] + (uc*weights[...,None]).sum(2)
    return c.to(q.dtype)
