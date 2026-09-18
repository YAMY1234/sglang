"""Two-pass native/latent MLA attention with one global softmax.

No prompt c or dense h is materialized. Reconstructed RoPE keys use one shared
64-coordinate workspace. Sparse weighted residuals use FP32 atomic addition.
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
def _scores(Q, Qz, Qh, Bias, Z, Idx, Val, Sc, Norm, Native, KV, Rope,
            loc, valid, bh, R: tl.constexpr, SP: tl.constexpr, NS: tl.constexpr,
            L, BR: tl.constexpr, BS: tl.constexpr, SCALE: tl.constexpr):
    native = tl.load(Native+loc)
    compact = valid & (native < 0)
    nr = tl.maximum(native, 0)
    r = tl.arange(0, BR)
    z = tl.full((16, BR), 0., tl.float32)
    score = tl.full((16,), 0., tl.float32)
    norm = tl.full((16,), 1., tl.float32)
    if tl.sum(compact.to(tl.int32), 0) > 0:
        z = tl.load(Z+loc[:, None]*R+r[None, :], compact[:, None] & (r[None, :] < R), 0).to(tl.float32)
        qz = tl.load(Qz+bh*R+r, r < R, 0)
        score = tl.sum(z*qz[None, :], 1)+tl.load(Bias+bh)
        if SP > 0:
            ss = tl.arange(0, BS)
            # SP supported in multiples of powers of two for this kernel.
            ix = _sparse_idx(Idx, loc, tl.minimum(ss, SP-1), SP)
            val = tl.load(Val+loc[:, None]*SP+ss[None, :], ss[None, :] < SP, 0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
            qh = tl.load(Qh+bh*2048+ix)
            score += tl.sum(qh*val, 1)*tl.load(Sc+loc)
        norm = tl.load(Norm+loc*NS)*tl.load(Norm+loc*NS+L)
        score /= norm
    d = tl.arange(0, 512)
    c = tl.load(KV+nr[:, None]*576+d[None, :], valid[:, None] & ~compact[:, None], 0).to(tl.float32)
    qc = tl.load(Q+bh*576+d).to(tl.float32)
    native_score = tl.sum(c*qc[None, :], 1)
    p = tl.arange(0, 64)
    rp = tl.load(Rope+loc[:, None]*64+p[None, :], compact[:, None], 0)
    kp = tl.load(KV+nr[:, None]*576+512+p[None, :], valid[:, None] & ~compact[:, None], 0).to(tl.float32)
    qp = tl.load(Q+bh*576+512+p).to(tl.float32)
    rope_score = tl.sum(tl.where(compact[:, None], rp, kp)*qp[None, :], 1)
    return tl.where(valid, (tl.where(compact, score, native_score)+rope_score)*SCALE, -float("inf")), z, c, compact, norm


@tr.jit
def _lse(Q, Qz, Qh, Bias, Z, Idx, Val, Sc, Norm, Native, KV, Rope,
         Req, Rows, Lens, Part, ROW: tl.constexpr, HEADS: tl.constexpr,
         R: tl.constexpr, SP: tl.constexpr, NS: tl.constexpr, L,
         BR: tl.constexpr, BS: tl.constexpr, SPLITS: tl.constexpr, SCALE: tl.constexpr):
    bh, split = tl.program_id(0), tl.program_id(1)
    b = bh//HEADS
    length = tl.load(Lens+b)
    row = tl.load(Rows+b).to(tl.int64)
    t = tl.arange(0, 16)
    m = -float("inf")
    total = 0.
    for start in range(split*16, length, SPLITS*16):
        valid = start+t < length
        loc = tl.load(Req+row*ROW+start+t, valid, 0).to(tl.int64)
        scores, _, _, _, _ = _scores(Q,Qz,Qh,Bias,Z,Idx,Val,Sc,Norm,Native,KV,Rope,
            loc,valid,bh,R,SP,NS,L,BR,BS,SCALE)
        new_m = tl.maximum(m, tl.max(scores, 0))
        total = total*tl.exp(m-new_m)+tl.sum(tl.exp(scores-new_m), 0)
        m = new_m
    tl.store(Part+bh*SPLITS+split, tl.where(total > 0, m+tl.log(total), -float("inf")))


@tr.jit
def _output(Q, Qz, Qh, Bias, Z, Idx, Val, Sc, Norm, Native, KV, Rope,
            Req, Rows, Lens, Part, Uz, Uc, Uh, Mass, ROW: tl.constexpr, HEADS: tl.constexpr,
            R: tl.constexpr, SP: tl.constexpr, NS: tl.constexpr, L,
            BR: tl.constexpr, BS: tl.constexpr, SPLITS: tl.constexpr, SCALE: tl.constexpr):
    bh, split = tl.program_id(0), tl.program_id(1)
    b = bh//HEADS
    length = tl.load(Lens+b)
    row = tl.load(Rows+b).to(tl.int64)
    partial = tl.load(Part+bh*SPLITS+tl.arange(0, SPLITS))
    max_lse = tl.max(partial, 0)
    logsum = max_lse + tl.log(tl.sum(tl.exp(partial-max_lse), 0))
    # Padded requests have a valid dummy slot and do not contribute downstream.
    logsum = tl.where(length > 0, logsum, 0.)
    t = tl.arange(0, 16)
    r = tl.arange(0, BR)
    d = tl.arange(0, 512)
    uz = tl.full((BR,), 0., tl.float32)
    uc = tl.full((512,), 0., tl.float32)
    mass = 0.
    for start in range(split*16, length, SPLITS*16):
        valid = start+t < length
        loc = tl.load(Req+row*ROW+start+t, valid, 0).to(tl.int64)
        scores, z, c, compact, norm = _scores(Q,Qz,Qh,Bias,Z,Idx,Val,Sc,Norm,Native,KV,Rope,
            loc,valid,bh,R,SP,NS,L,BR,BS,SCALE)
        p = tl.exp(scores-logsum)
        ps = tl.where(compact, p/norm, 0.)
        uc += tl.sum(p[:, None]*c, 0)
        if tl.sum(compact.to(tl.int32), 0) > 0:
            uz += tl.sum(ps[:, None]*z, 0)
            mass += tl.sum(ps, 0)
            if SP > 0:
                ss = tl.arange(0, BS)
                ix = _sparse_idx(Idx, loc, tl.minimum(ss, SP-1), SP)
                val = tl.load(Val+loc[:, None]*SP+ss[None, :], ss[None, :] < SP, 0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
                value = (ps*tl.load(Sc+loc))[:, None]*val
                tl.atomic_add(Uh+bh*2048+ix, value, compact[:, None] & (ss[None, :] < SP), sem="relaxed")
    tl.store(Uz+(bh*SPLITS+split)*R+r, uz, r < R)
    tl.store(Uc+(bh*SPLITS+split)*512+d, uc)
    tl.store(Mass+bh*SPLITS+split, mass)


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
    uz = torch.empty((b, heads, splits, cfg.rank), dtype=torch.float32, device=q.device)
    uc = torch.empty((b, heads, splits, 512), dtype=torch.float32, device=q.device)
    uh = torch.zeros((b, heads, 2048), dtype=torch.float32, device=q.device)
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
        L=layer_id-cfg.first_layer+1,BR=tr.next_power_of_2(cfg.rank),
        BS=tr.next_power_of_2(max(1,cfg.sparse)),SPLITS=splits,SCALE=scale,num_warps=4)
    _lse[(b*heads,splits)](*common,part,**constants)
    _output[(b*heads,splits)](*common,part,uz,uc,uh,mass,**constants)
    c = uz.sum(2) @ w['pe'].T + uh @ w['p'].T
    c = c + mass.sum(2)[..., None]*w['pm'] + uc.sum(2)
    return c.to(q.dtype)
