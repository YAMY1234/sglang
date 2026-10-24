# SPDX-License-Identifier: Apache-2.0
"""Target sampling of an EAGLE verify on the vocabulary-sharded LM head output
(SGLANG_Q35MF_SHARDED_VERIFY): every rank reduces its [rows, V / tp] shard to one int64
key per row, an all-gather of the keys and a max pick the token. Greedy rows match
torch.argmax over the gathered FP32 logits bit for bit; temperature-only rows draw a
Gumbel-max sample, the same distribution as softmax(logits / T)."""

from __future__ import annotations

from typing import Any, Optional

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.eagle import _chain_finish, _chain_walk
from sglang.srt.utils.common import print_info_once
from sglang.srt.environ import envs

_MC_GATHER = envs.SGLANG_PERFLL_VERIFY_MC_GATHER.get()
_SPLIT = envs.SGLANG_PERFLL_VERIFY_KEY_SPLIT.get()
_LL_KEYS = envs.SGLANG_PERFLL_VERIFY_LL_KEYS.get()


class VocabShard(msgspec.Struct, frozen=True):
    """Columns [0, valid) of this rank's logits are global token ids start + j."""

    start: int
    valid: int
    vocab_size: int


# key of (-inf, token id 0xFFFFFFFF): below every real column's key
_KEY_FLOOR = -2139095041 << 32


@triton.jit
def _plain_key(x, gid, mask, KEY_FLOOR: tl.constexpr):
    # the argmax key of FP32 values x, as _shard_key_kernel builds it without temperatures
    x = tl.where(x == x, x, float("-inf")) + 0.0
    bits = x.to(tl.int32, bitcast=True)
    ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)
    key = (ordered << 32) | (4294967295 - gid.to(tl.int64))
    return tl.where(mask, key, KEY_FLOOR)


@triton.jit
def _shard_key_kernel(
    logits_ptr,
    stride_row,
    temp_ptr,
    seed_ptr,
    keys_ptr,
    start,
    valid,
    vocab_size,
    span,
    KEY_FLOOR: tl.constexpr,
    HAS_TEMP: tl.constexpr,
    BLOCK: tl.constexpr,
    plain_keys_ptr=None,
    PLAIN: tl.constexpr = False,
    TEMP_GROUP: tl.constexpr = 1,
):
    # program (row, part): columns [part * span, (part + 1) * span) of the row's shard
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    lo = part * span
    hi = tl.minimum(lo + span, valid)
    best = tl.full([BLOCK], KEY_FLOOR, tl.int64)
    # PLAIN: the argmax keys too, from the same loads (before the Gumbel noise)
    if PLAIN:
        best_plain = tl.full([BLOCK], KEY_FLOOR, tl.int64)
    if HAS_TEMP:
        # TEMP_GROUP > 1: one temperature per TEMP_GROUP consecutive rows (a request's verify rows)
        if TEMP_GROUP > 1:
            temp = tl.load(temp_ptr + row // TEMP_GROUP)
        else:
            temp = tl.load(temp_ptr + row)
        inv_t = 1.0 / tl.maximum(temp, 1e-20)
        seed = tl.load(seed_ptr + row)
    for off in range(lo, hi, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        gid = start + cols
        mask = (cols < hi) & (gid < vocab_size)
        x = tl.load(
            logits_ptr + row * stride_row + cols, mask=mask, other=float("-inf")
        )
        x = x.to(tl.float32)
        if PLAIN:
            best_plain = tl.maximum(best_plain, _plain_key(x, gid, mask, KEY_FLOOR))
        if HAS_TEMP:
            if temp > 0.0:
                u = tl.rand(seed, gid)
                x = x * inv_t - tl.log(-tl.log(u))
        # NaN never wins, as after sanitize_nan_logits; -0.0 ties +0.0 as in torch.argmax
        x = tl.where(x == x, x, float("-inf")) + 0.0
        bits = x.to(tl.int32, bitcast=True)
        # monotone in x as a signed int32; the low word prefers the smaller token id
        ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)
        key = (ordered << 32) | (4294967295 - gid.to(tl.int64))
        key = tl.where(mask, key, KEY_FLOOR)
        best = tl.maximum(best, key)
    tl.store(keys_ptr + row * tl.num_programs(1) + part, tl.max(best, axis=0))
    if PLAIN:
        tl.store(
            plain_keys_ptr + row * tl.num_programs(1) + part, tl.max(best_plain, axis=0)
        )


_KEY_BLOCK = 2048


def key_splits(valid: int) -> int:
    """SGLANG_PERFLL_VERIFY_KEY_SPLIT: programs a row, one per _KEY_BLOCK columns of the
    shard, even (a row of keys is whole 16-byte multimem chunks)."""
    n = -(-valid // _KEY_BLOCK)
    return n + (n & 1)


def shard_keys(
    logits: torch.Tensor,
    shard: VocabShard,
    temperatures: Optional[torch.Tensor] = None,
    seeds: Optional[torch.Tensor] = None,
    splits: int = 1,
) -> torch.Tensor:
    """One int64 key per row of this rank's shard: (order-preserving FP32 bits of the
    row's best value) << 32 | (0xFFFFFFFF - token id). temperatures: [rows] float32;
    rows at T <= 0 take the plain logit, rows at T > 0 logit / T + Gumbel noise drawn
    from seeds [rows] int64 and the global token id. splits > 1: [rows, splits], the
    best key of each of `splits` column ranges (the row's key is their max)."""
    assert logits.stride(-1) == 1
    rows = logits.shape[0]
    keys = torch.empty(
        (rows, splits) if splits > 1 else (rows,),
        dtype=torch.int64,
        device=logits.device,
    )
    if rows == 0:
        return keys
    has_temp = temperatures is not None
    _shard_key_kernel[(rows, splits)](
        logits,
        logits.stride(0),
        temperatures if has_temp else keys,
        seeds if has_temp else keys,
        keys,
        shard.start,
        shard.valid,
        shard.vocab_size,
        -(-shard.valid // splits),
        KEY_FLOOR=_KEY_FLOOR,
        HAS_TEMP=has_temp,
        BLOCK=_KEY_BLOCK,
        num_warps=8,
    )
    return keys


@triton.jit
def _keys_to_tokens_kernel(
    keys_ptr,
    tokens_ptr,
    stride_rank,
    stride_row,
    world,
    parts,
    KEY_FLOOR: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, BLOCK_R)[:, None]
    p = tl.arange(0, BLOCK_P)[None, :]
    k = tl.load(
        keys_ptr + row * stride_row + r * stride_rank + p,
        mask=(r < world) & (p < parts),
        other=KEY_FLOOR,
    )
    best = tl.max(tl.max(k, axis=1), axis=0)
    tl.store(tokens_ptr + row, 4294967295 - (best & 4294967295))


def _reduce_keys(keys, rows, world, parts, stride_rank, stride_row) -> torch.Tensor:
    """[rows] token ids from the keys at keys[row * stride_row + rank * stride_rank + part]
    (int64 elements), the max over every rank and part of a row, in one launch."""
    tokens = torch.empty(rows, dtype=torch.int64, device=keys.device)
    if rows:
        _keys_to_tokens_kernel[(rows,)](
            keys,
            tokens,
            stride_rank,
            stride_row,
            world,
            parts,
            KEY_FLOOR=_KEY_FLOOR,
            BLOCK_R=triton.next_power_of_2(world),
            BLOCK_P=triton.next_power_of_2(parts),
            num_warps=1,
        )
    return tokens


def keys_to_tokens(gathered: torch.Tensor, tp_size: int, splits: int = 1) -> torch.Tensor:
    """[tp * rows * splits] keys of all ranks (rank-major) -> [rows] int64 token ids."""
    if splits == 1:
        best = gathered.view(tp_size, -1).amax(dim=0)
        return 0xFFFFFFFF - (best & 0xFFFFFFFF)
    rows = gathered.numel() // (tp_size * splits)
    return _reduce_keys(gathered, rows, tp_size, splits, rows * splits, splits)


# SGLANG_PERFLL_VERIFY_MC_GATHER: the key exchange's multimem state, built by the first
# (eager, every rank in lockstep) call; a row of a rank is `width` int64 (4 bf16 each)
_MC_MAX_ROWS = 1024
_mc_state = None
_mc_width = 0


def _mc_build(tp_group, hidden_size: int, device):
    """The multimem state, or False (NCCL keeps the exchange) when the buffer has no
    multicast mapping, e.g. a TP group across nodes on torch symmetric memory."""
    from sglang.srt.distributed.device_communicators import triton_symm_mem_ag

    try:
        state = triton_symm_mem_ag.create_state(
            tp_group.device_group,
            tp_group.rank_in_group,
            max_tokens=_MC_MAX_ROWS,
            hidden_size=hidden_size,
            device=device,
        )
    except Exception as e:  # rendezvous refused: every rank sees the same failure
        print_info_once(f"SGLANG_PERFLL_VERIFY_MC_GATHER: multimem state failed ({e}); NCCL gather")
        return False
    if state.symm_mem_hdl.multicast_ptr == 0:
        print_info_once("SGLANG_PERFLL_VERIFY_MC_GATHER: no multicast mapping; NCCL gather")
        return False
    return state


def _mc_gather_keys(keys: torch.Tensor, tp_group) -> Optional[torch.Tensor]:
    """[rows] or [rows, parts] int64 keys of every rank -> [rows] token ids, by one
    multimem all-gather (rows of at least 2 int64: a single key goes twice); None when
    there is no multicast state."""
    global _mc_state, _mc_width
    from sglang.srt.distributed.device_communicators import triton_symm_mem_ag

    world, rows = tp_group.world_size, keys.shape[0]
    if keys.dim() == 1:
        keys = torch.stack((keys, keys), dim=1)
    width = keys.shape[1]
    if _mc_state is None:
        _mc_state = _mc_build(tp_group, 4 * width * world, keys.device)
        _mc_width = width
    if _mc_state is False:
        return None
    assert width == _mc_width, (width, _mc_width)
    out = triton_symm_mem_ag.all_gather_inner(
        _mc_state, keys.view(torch.bfloat16), 4 * width * world, safe=False
    )
    if not _SPLIT:
        return 0xFFFFFFFF - (
            out.contiguous().view(torch.int64).view(rows, world, width)[:, :, 0].amax(dim=1)
            & 0xFFFFFFFF
        )
    # rows of the symmetric buffer; rank r's keys at int64 column width * r
    out = out.view(torch.int64)
    return _reduce_keys(out, rows, world, width, width, out.stride(0))


# SGLANG_PERFLL_VERIFY_LL_KEYS: epoch-tagged exchange of one key per row (low / high 32 bits in two
# 64-bit words whose high halves carry the epoch), three slots rotating by epoch: a slot is rewritten
# three calls later, after two more exchanges that each needed every rank's words.
_LL_SLOTS = 3
_LL_MAX_ROWS = 64
_LL_SPIN_LIMIT = 1 << 26
_ll_state = None


@triton.jit
def _ll_st_b64(addr, val, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $3, 1;
            @!%p0 bra end;
            multimem.st.relaxed.sys.global.b64 [$1], $2;
            end:
        }
        """,
        "=r,l,l,r",
        args=[addr, val, mask.to(tl.int32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _ll_ld_b64(addr, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $2, 1;
            mov.u64 $0, 0;
            @!%p0 bra end;
            ld.relaxed.sys.global.u64 $0, [$1];
            end:
        }
        """,
        "=l,l,r",
        args=[addr, mask.to(tl.int32)],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _ll_trap():
    tl.inline_asm_elementwise("trap;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _ll_keys_kernel(
    keys_ptr,
    parts,
    rows,
    mc_addr,
    uc_addr,
    epoch_ptr,
    tokens_ptr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    MAXR: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_P: tl.constexpr,
    KEY_FLOOR: tl.constexpr,
    SPIN_LIMIT: tl.constexpr,
):
    tag = (tl.load(epoch_ptr) + 1).to(tl.uint64)
    slot = (tag % 3).to(tl.int64)
    r = tl.arange(0, BLOCK_R)
    p = tl.arange(0, BLOCK_P)
    rmask = r < rows
    k = tl.load(
        keys_ptr + r[:, None] * parts + p[None, :],
        mask=rmask[:, None] & (p[None, :] < parts),
        other=KEY_FLOOR,
    )
    key = tl.max(k, axis=1).to(tl.uint64, bitcast=True)
    tagw = tag << 32
    mc = mc_addr.to(tl.pointer_type(tl.uint64))
    word = ((slot * WORLD + RANK) * MAXR + r) * 2
    _ll_st_b64(mc + word, (key & 0xFFFFFFFF) | tagw, rmask)
    _ll_st_b64(mc + word + 1, (key >> 32) | tagw, rmask)
    s = tl.arange(0, WORLD)
    m2 = rmask[None, :] & (s[:, None] < WORLD)
    uc = uc_addr.to(tl.pointer_type(tl.uint64))
    words = ((slot * WORLD + s[:, None]) * MAXR + r[None, :]) * 2
    # a CTA barrier before each block-wide count of the poll: without them some warps left it
    # on a stale count and took words a slot held from an earlier launch
    tl.debug_barrier()
    lo = _ll_ld_b64(uc + words, m2)
    hi = _ll_ld_b64(uc + words + 1, m2)
    ok = (((lo >> 32) == tag) & ((hi >> 32) == tag)) | (~m2)
    pending = tl.sum(1 - ok.to(tl.int32))
    spins = 0
    while pending > 0:
        tl.debug_barrier()
        lo = _ll_ld_b64(uc + words, m2)
        hi = _ll_ld_b64(uc + words + 1, m2)
        ok = (((lo >> 32) == tag) & ((hi >> 32) == tag)) | (~m2)
        pending = tl.sum(1 - ok.to(tl.int32))
        spins += 1
        if spins > SPIN_LIMIT:
            _ll_trap()
    got = (((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)).to(tl.int64, bitcast=True)
    best = tl.max(tl.where(m2, got, KEY_FLOOR), axis=0)
    tl.store(tokens_ptr + r, 4294967295 - (best & 4294967295), mask=rmask)
    tl.store(epoch_ptr, tag.to(tl.int32))


def _ll_build(
    tp_group, device, max_rows: int = _LL_MAX_ROWS, name: str = "SGLANG_PERFLL_VERIFY_LL_KEYS"
):
    """The LL exchange's symmetric buffer (zeroed before any peer can write it) and epoch, or
    False (the other exchanges keep the call) when it has no multicast mapping."""
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    world = tp_group.world_size
    try:
        with torch.inference_mode(False), torch.no_grad():
            buf = symm_mem.empty(
                (_LL_SLOTS * world * max_rows * 2,), dtype=torch.int64, device=device
            )
        hdl = symm_mem.rendezvous(buf, group=tp_group.device_group)
    except Exception as e:
        print_info_once(f"{name}: symmetric buffer failed ({e}); not used")
        return False
    if hdl.multicast_ptr == 0:
        print_info_once(f"{name}: no multicast mapping; not used")
        return False
    buf.zero_()
    torch.cuda.synchronize(device)
    dist.barrier(group=tp_group.cpu_group if hasattr(tp_group, "cpu_group") else tp_group.device_group)
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    return (buf, hdl, epoch)


def _ll_keys(keys: torch.Tensor, tp_group) -> Optional[torch.Tensor]:
    """[rows] or [rows, parts] int64 keys of every rank -> [rows] token ids in one launch, or None
    when the LL exchange is unavailable."""
    global _ll_state
    if _ll_state is None:
        _ll_state = _ll_build(tp_group, keys.device)
    if _ll_state is False:
        return None
    buf, hdl, epoch = _ll_state
    rows = keys.shape[0]
    parts = keys.shape[1] if keys.dim() == 2 else 1
    tokens = torch.empty(rows, dtype=torch.int64, device=keys.device)
    _ll_keys_kernel[(1,)](
        keys,
        parts,
        rows,
        hdl.multicast_ptr,
        hdl.buffer_ptrs[hdl.rank],
        epoch,
        tokens,
        RANK=hdl.rank,
        WORLD=tp_group.world_size,
        MAXR=_LL_MAX_ROWS,
        BLOCK_R=max(16, triton.next_power_of_2(rows)),
        BLOCK_P=max(2, triton.next_power_of_2(parts)),
        KEY_FLOOR=_KEY_FLOOR,
        SPIN_LIMIT=_LL_SPIN_LIMIT,
        num_warps=4,
    )
    return tokens


# SGLANG_PERFLL_VERIFY_FUSED_TAIL: the LL exchange of one or two keys per row fused with the chain
# accept, on a channel (buffer, epoch) of its own: the layout is fixed per channel (key k of row r at
# virtual row _TAIL_ROWS * k + r), as a slow rank may still poll words of the previous call.
_TAIL_ROWS = 64
_tail_state = None


class TailChannel(msgspec.Struct, frozen=True):
    """The fused tail's exchange state: `buffer` (int64 words) at `uc_addr`, written through
    `mc_addr` (multicast; a plain store when `multicast` is False, a one-rank channel)."""

    buffer: Any
    epoch: Any
    mc_addr: int
    uc_addr: int
    rank: int
    multicast: bool


@triton.jit
def _tail_st_b64(addr, val, mask, MULTICAST: tl.constexpr):
    if MULTICAST:
        return _ll_st_b64(addr, val, mask)
    else:
        return tl.inline_asm_elementwise(
            """
            {
                .reg .pred %p0;
                setp.eq.s32 %p0, $3, 1;
                @!%p0 bra end;
                st.relaxed.sys.global.b64 [$1], $2;
                end:
            }
            """,
            "=r,l,l,r",
            args=[addr, val, mask.to(tl.int32)],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )


@triton.jit
def _tail_push(
    mc,
    key,
    vrow,
    slot,
    tagw,
    mask,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    MAXR: tl.constexpr,
    MULTICAST: tl.constexpr,
):
    # this rank's words of virtual rows vrow, the _ll_keys_kernel words
    key = key.to(tl.uint64, bitcast=True)
    word = ((slot * WORLD + RANK) * MAXR + vrow) * 2
    _tail_st_b64(mc + word, (key & 0xFFFFFFFF) | tagw, mask, MULTICAST)
    _tail_st_b64(mc + word + 1, (key >> 32) | tagw, mask, MULTICAST)


@triton.jit
def _tail_pending(uc, words, tag, mask):
    lo = _ll_ld_b64(uc + words, mask)
    hi = _ll_ld_b64(uc + words + 1, mask)
    ok = (((lo >> 32) == tag) & ((hi >> 32) == tag)) | (~mask)
    return lo, hi, tl.sum(1 - ok.to(tl.int32))


@triton.jit
def _tail_tokens(lo, hi, mask, KEY_FLOOR: tl.constexpr):
    # the max over the ranks (axis 0) of the exchanged keys -> token ids
    got = (((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)).to(tl.int64, bitcast=True)
    best = tl.max(tl.where(mask, got, KEY_FLOOR), axis=0)
    return 4294967295 - (best & 4294967295)


@triton.jit
def _ll_accept_kernel(
    keys_ptr,
    plain_keys_ptr,
    parts,
    mc_addr,
    uc_addr,
    epoch_ptr,
    tokens_ptr,
    bonus_tokens_ptr,
    predict,
    accept_index,
    num_correct_drafts,
    accept_lens,
    sim_accept_index,
    candidates,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    simulated_acc_len,
    bs,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    MAXR: tl.constexpr,
    PLAIN_ROW: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_P: tl.constexpr,
    KEY_FLOOR: tl.constexpr,
    SPIN_LIMIT: tl.constexpr,
    PLAIN: tl.constexpr,
    SIMULATE: tl.constexpr,
    REAL_DRAFT_TOKENS: tl.constexpr,
    MULTICAST: tl.constexpr,
):
    # One CTA: _ll_keys_kernel's exchange of every row's key (and plain key) into the token
    # buffers, then _chain_accept_kernel(VERIFY, FINISH) for every request on the tokens read
    # back; request b's verify rows are [b * WIDTH, (b + 1) * WIDTH) of every row input.
    rb = tl.arange(0, BLOCK_B)
    b = rb[:, None].to(tl.int64)
    w = tl.arange(0, BLOCK_W)[None, :]
    m2 = (b < bs) & (w < WIDTH)
    row = b * WIDTH + w
    # the request trees (staged ahead of this graph), loaded ahead of the exchange
    nxt = tl.load(retrieve_next_token + row, mask=m2, other=-1).to(tl.int64)
    sib = tl.load(retrieve_next_sibling + row, mask=m2, other=-1).to(tl.int64)
    idx = tl.load(retrieve_index + row, mask=m2, other=0).to(tl.int64)
    drf = tl.load(candidates + row + 1, mask=(b < bs) & (w < WIDTH - 1), other=0)

    tag = (tl.load(epoch_ptr) + 1).to(tl.uint64)
    slot = (tag % 3).to(tl.int64)
    # every warp has read the epoch before any warp can advance it below: a warp whose
    # poll needs only its own words may finish first, and a later reader would take a
    # slot of stale words
    tl.debug_barrier()
    r = tl.arange(0, BLOCK_R)
    rmask = r < bs * WIDTH
    p = tl.arange(0, BLOCK_P)
    kaddr = r[:, None] * parts + p[None, :]
    kmask = rmask[:, None] & (p[None, :] < parts)
    key = tl.max(tl.load(keys_ptr + kaddr, mask=kmask, other=KEY_FLOOR), axis=1)
    tagw = tag << 32
    mc = mc_addr.to(tl.pointer_type(tl.uint64))
    uc = uc_addr.to(tl.pointer_type(tl.uint64))
    _tail_push(mc, key, r, slot, tagw, rmask, RANK, WORLD, MAXR, MULTICAST)
    if PLAIN:
        plain = tl.max(tl.load(plain_keys_ptr + kaddr, mask=kmask, other=KEY_FLOOR), axis=1)
        _tail_push(mc, plain, r + PLAIN_ROW, slot, tagw, rmask, RANK, WORLD, MAXR, MULTICAST)
    s = tl.arange(0, WORLD)
    mw = rmask[None, :] & (s[:, None] < WORLD)
    words = ((slot * WORLD + s[:, None]) * MAXR + r[None, :]) * 2
    lo, hi, pending = _tail_pending(uc, words, tag, mw)
    if PLAIN:
        lo_p, hi_p, pending_p = _tail_pending(uc, words + 2 * PLAIN_ROW, tag, mw)
        pending += pending_p
    spins = 0
    while pending > 0:
        lo, hi, pending = _tail_pending(uc, words, tag, mw)
        if PLAIN:
            lo_p, hi_p, pending_p = _tail_pending(uc, words + 2 * PLAIN_ROW, tag, mw)
            pending += pending_p
        spins += 1
        if spins > SPIN_LIMIT:
            _ll_trap()
    tl.store(tokens_ptr + r, _tail_tokens(lo, hi, mw, KEY_FLOOR), mask=rmask)
    if PLAIN:
        tl.store(bonus_tokens_ptr + r, _tail_tokens(lo_p, hi_p, mw, KEY_FLOOR), mask=rmask)
    # every warp is done polling, and the accept below reads back every warp's tokens
    tl.debug_barrier()
    tl.store(epoch_ptr, tag.to(tl.int32))

    # _chain_is_canonical per request: node i is row base + i, its only child node i + 1
    want = tl.where(w < WIDTH - 1, w + 1, -1)
    bad = m2 & ((nxt != want) | (sib != -1) | (idx != row))
    canon = tl.sum(bad.to(tl.int32), axis=1) == 0
    if tl.sum(((~canon) & (rb < bs)).to(tl.int32)) > 0:
        # a tree that is not a canonical chain: _chain_accept_kernel's walk and writes for
        # every request
        cols = tl.arange(0, BLOCK_W)
        for i in range(0, bs):
            b_i = i.to(tl.int64)
            first_i, acc_i, pred_i, num_i = _chain_walk(
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
                candidates,
                tokens_ptr,
                b_i * WIDTH,
                cols,
                WIDTH,
                BLOCK_W,
            )
            _chain_finish(
                predict,
                accept_index,
                num_correct_drafts,
                accept_lens,
                sim_accept_index,
                candidates,
                bonus_tokens_ptr,
                simulated_acc_len,
                b_i,
                b_i * WIDTH,
                cols,
                first_i,
                acc_i,
                pred_i,
                num_i,
                WIDTH,
                BLOCK_W,
                True,
                True,
                SIMULATE,
                REAL_DRAFT_TOKENS,
            )
    else:
        _chain_linear_finish(
            predict,
            accept_index,
            num_correct_drafts,
            accept_lens,
            sim_accept_index,
            candidates,
            simulated_acc_len,
            tl.load(tokens_ptr + row, mask=m2, other=0),
            tl.load(bonus_tokens_ptr + row, mask=m2, other=0),
            drf,
            rb,
            w,
            row,
            m2,
            bs,
            WIDTH,
            BLOCK_B,
            BLOCK_W,
            SIMULATE,
            REAL_DRAFT_TOKENS,
        )


@triton.jit
def _chain_linear_finish(
    predict,
    accept_index,
    num_correct_drafts,
    accept_lens,
    sim_accept_index,
    candidates,
    simulated_acc_len,
    tgt,
    bonus,
    drf,
    rb,
    w,
    row,
    m2,
    bs,
    WIDTH: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_W: tl.constexpr,
    SIMULATE: tl.constexpr,
    REAL_DRAFT_TOKENS: tl.constexpr,
):
    # _chain_linear and _chain_finish(VERIFY, FINISH) for [BLOCK_B] canonical chains at once
    hit = (w < WIDTH - 1) & (drf.to(tl.int64) == tgt)
    num = tl.min(tl.where(hit, WIDTH - 1, w), axis=1).to(tl.int64)
    acc = tl.where(w <= num[:, None], row, -1)
    pred = tl.where(w <= num[:, None], tgt, 0)
    tl.store(accept_index + row, acc.to(tl.int32), mask=m2)
    if SIMULATE:
        length = tl.load(simulated_acc_len).to(tl.int64)
        first = rb[:, None].to(tl.int64) * WIDTH
        sim = tl.where(w < length, first + w, -1)
        tl.store(sim_accept_index + row, sim.to(tl.int32), mask=m2)
        num = tl.zeros([BLOCK_B], dtype=tl.int64) + (length - 1)
        if REAL_DRAFT_TOKENS:
            nxt = tl.minimum(w + 1, WIDTH - 1)
            drafted = tl.load(candidates + row - w + nxt, mask=m2, other=0)
            pred = tl.where(
                w < length - 1,
                drafted.to(tl.int32).to(tl.int64),
                tl.where(w == length - 1, bonus.to(tl.int32).to(tl.int64), pred),
            )
        else:
            pred = tl.full([BLOCK_B, BLOCK_W], 100, dtype=tl.int64)
    tl.store(predict + row, pred.to(tl.int32), mask=m2)
    tl.store(num_correct_drafts + rb, num.to(tl.int32), mask=rb < bs)
    tl.store(accept_lens + rb, (num + 1).to(tl.int32), mask=rb < bs)


def _tail_channel(tp_group, device) -> Optional[TailChannel]:
    global _tail_state
    if _tail_state is None:
        built = _ll_build(
            tp_group, device, 2 * _TAIL_ROWS, name="SGLANG_PERFLL_VERIFY_FUSED_TAIL"
        )
        _tail_state = built and TailChannel(
            buffer=built[0],
            epoch=built[2],
            mc_addr=built[1].multicast_ptr,
            uc_addr=built[1].buffer_ptrs[built[1].rank],
            rank=built[1].rank,
            multicast=True,
        )
    return _tail_state or None


def fused_verify_tail(
    *,
    logits: torch.Tensor,
    shard: VocabShard,
    tp_group,
    temperatures: Optional[torch.Tensor],
    plain_bonus: bool,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    simulated_acc_len: Optional[torch.Tensor],
    real_draft_tokens: bool,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """SGLANG_PERFLL_VERIFY_FUSED_TAIL: sharded_target_tokens then chain_accept(verify, finish), in
    two launches, with every output chain_accept writes; returns (accept_lens, accept_index) as it
    does. temperatures: [bs] float32 for a Gumbel-max target (None: greedy); plain_bonus: the
    injected acceptance's bonus is the argmax, exchanged with the Gumbel keys. None when the tail's
    channel is unavailable or the batch too large: the caller runs the stock tail."""
    bs, width = accept_index.shape
    rows = logits.shape[0]
    if not 0 < rows <= _TAIL_ROWS or tp_group.world_size > 16:
        return None
    channel = _tail_channel(tp_group, logits.device)
    if channel is None:
        return None
    print_info_once(
        "SGLANG_PERFLL_VERIFY_FUSED_TAIL: target "
        + ("Gumbel-max" if temperatures is not None else "argmax")
        + (" with the argmax bonus" if plain_bonus else "")
        + " and the chain accept in two launches"
    )
    device = logits.device
    has_temp = temperatures is not None
    # the same draw as sharded_target_tokens makes, so the Gumbel noise is unchanged
    seeds = (
        torch.randint(0, 2**31 - 1, (rows,), dtype=torch.int64, device=device)
        if has_temp
        else None
    )
    splits = key_splits(shard.valid) if _SPLIT else 1
    keys = torch.empty((rows, splits), dtype=torch.int64, device=device)
    plain = torch.empty_like(keys) if plain_bonus else keys
    _shard_key_kernel[(rows, splits)](
        logits,
        logits.stride(0),
        temperatures if has_temp else keys,
        seeds if has_temp else keys,
        keys,
        shard.start,
        shard.valid,
        shard.vocab_size,
        -(-shard.valid // splits),
        KEY_FLOOR=_KEY_FLOOR,
        HAS_TEMP=has_temp,
        BLOCK=_KEY_BLOCK,
        plain_keys_ptr=plain,
        PLAIN=plain_bonus,
        TEMP_GROUP=width,
        num_warps=8,
    )
    tokens = torch.empty(rows, dtype=torch.int64, device=device)
    bonus_tokens = torch.empty_like(tokens) if plain_bonus else tokens
    accept_lens = torch.empty((bs,), dtype=torch.int32, device=device)
    simulate = simulated_acc_len is not None
    sim_accept_index = torch.empty_like(accept_index) if simulate else None
    _ll_accept_kernel[(1,)](
        keys,
        plain,
        splits,
        channel.mc_addr,
        channel.uc_addr,
        channel.epoch,
        tokens,
        bonus_tokens,
        predict,
        accept_index,
        num_correct_drafts,
        accept_lens,
        sim_accept_index if simulate else accept_index,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        simulated_acc_len if simulate else channel.epoch,
        bs,
        RANK=channel.rank,
        WORLD=tp_group.world_size,
        MAXR=2 * _TAIL_ROWS,
        PLAIN_ROW=_TAIL_ROWS,
        WIDTH=width,
        BLOCK_R=max(16, triton.next_power_of_2(rows)),
        BLOCK_B=max(2, triton.next_power_of_2(bs)),
        BLOCK_W=max(2, triton.next_power_of_2(width)),
        BLOCK_P=max(2, triton.next_power_of_2(splits)),
        KEY_FLOOR=_KEY_FLOOR,
        SPIN_LIMIT=_LL_SPIN_LIMIT,
        PLAIN=plain_bonus,
        SIMULATE=simulate,
        REAL_DRAFT_TOKENS=real_draft_tokens,
        MULTICAST=channel.multicast,
        num_warps=4,
    )
    return accept_lens, (accept_index if sim_accept_index is None else sim_accept_index)


def sharded_target_tokens(
    logits: torch.Tensor,
    shard: VocabShard,
    tp_group,
    temperatures: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[rows] token ids over the whole vocabulary from every rank's logits shard."""
    print_info_once(
        "SGLANG_Q35MF_SHARDED_VERIFY: target "
        + ("Gumbel-max" if temperatures is not None else "argmax")
        + " on the vocabulary shards"
        + (" (keys split, fused reduce)" if _SPLIT else "")
        + (" (multimem key gather)" if _MC_GATHER else "")
        + (" (LL key exchange <= 64 rows)" if _LL_KEYS else "")
    )
    seeds = None
    if temperatures is not None:
        seeds = torch.randint(
            0, 2**31 - 1, (logits.shape[0],), dtype=torch.int64, device=logits.device
        )
    splits = key_splits(shard.valid) if _SPLIT else 1
    keys = shard_keys(logits, shard, temperatures, seeds, splits)
    if _LL_KEYS and 0 < keys.shape[0] <= _LL_MAX_ROWS and tp_group.world_size <= 16:
        tokens = _ll_keys(keys, tp_group)
        if tokens is not None:
            return tokens
    if _MC_GATHER and 0 < keys.shape[0] <= _MC_MAX_ROWS:
        tokens = _mc_gather_keys(keys, tp_group)
        if tokens is not None:
            return tokens
    gathered = torch.empty(
        tp_group.world_size * keys.numel(), dtype=torch.int64, device=keys.device
    )
    tp_group.all_gather_into_tensor(gathered, keys.view(-1))
    return keys_to_tokens(gathered, tp_group.world_size, splits)


def gather_full_logits(
    logits: torch.Tensor, shard: VocabShard, tp_group
) -> torch.Tensor:
    """The FP32 [rows, vocab_size] logits the stock verify builds: all-gather, slice
    to the vocabulary, cast."""
    print_info_once(
        "SGLANG_Q35MF_SHARDED_VERIFY: a batch needs the full logits, gathered"
    )
    tp, rows = tp_group.world_size, logits.shape[0]
    # all_gather_into_tensor, not all_gather: the latter allocates in symmetric
    # memory, which the post-verify graph capture cannot record
    out = torch.empty(
        (tp,) + tuple(logits.shape), dtype=logits.dtype, device=logits.device
    )
    tp_group.all_gather_into_tensor(out.view(-1), logits.contiguous().view(-1))
    full = out.movedim(0, 1).reshape(rows, -1)
    return full[:, : shard.vocab_size].float()
